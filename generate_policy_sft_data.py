"""Generate aligned RuleBot or local-model SFT trajectories.

Both teachers use the same prompt builder and system prompt as deployment.
Only legal, positive winning trajectories are retained.  The game engine's
exchange transition is invoked exactly once through select_exchange_tiles().
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from experiment_trace import ensure_deterministic_hashing, write_json
from game import (
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    bot_decide_turn_action,
    parse_console_tile,
)
from llm_backends import build_llm_callable
from prompt_builder import ACTION_SYSTEM_PROMPT, build_base_decision_prompt, get_legal_actions


DecisionFn = Callable[[str], str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=["rule", "local_qwen"], required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026071401)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--model-path", default="models/Qwen-Mahjong-V1-Mixed-SFT-Merged")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-net", type=float, default=0.0)
    parser.add_argument("--require-hu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-illegal-trajectories", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def extract_action(raw: str, legal_actions: List[str]) -> Tuple[str, bool]:
    text = (raw or "").strip()
    if text.startswith("{"):
        try:
            text = str(json.loads(text).get("action", "")).strip()
        except json.JSONDecodeError:
            pass
    if text in legal_actions:
        return text, True
    match = re.search(r"(?:^|[^a-z])(d\s*[1-9][万筒条]|[hgpn])(?:$|[^a-z])", text)
    if match:
        candidate = re.sub(r"\s+", " ", match.group(1)).strip()
        if candidate.startswith("d") and not candidate.startswith("d "):
            candidate = "d " + candidate[1:].strip()
        if candidate in legal_actions:
            return candidate, True
    fallback = bot_fallback(legal_actions)
    return fallback, False


def bot_fallback(legal_actions: List[str]) -> str:
    for action in ("h", "g", "p", "n"):
        if action in legal_actions:
            return action
    return legal_actions[0]


def make_row(game, pid: int, prompt: str, action: str, legal: List[str], meta: Dict) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": action},
        ],
        "meta": {
            **meta,
            "game_id": game.game_id,
            "player_id": pid,
            "legal_actions": legal,
            "prompt_schema": "shared_base_action_v1",
            "shanten_engine": "exact_dfs_v2",
        },
    }


def choose_action(
    game,
    pid: int,
    legal: List[str],
    teacher: str,
    llm: Optional[DecisionFn],
    response_actions: Optional[List[str]] = None,
) -> Tuple[str, bool, str]:
    prompt = build_base_decision_prompt(game, pid, valid_actions=legal)
    if response_actions is not None:
        tile = game.last_discarded_tile
        source = game.last_discard_player_id
        prompt += f"\n\n【响应事件】\nP{source} 打出 {tile}，请从合法响应中选择。"
    if teacher == "rule":
        raw = (
            bot_decide_response(game.players[pid], response_actions or [])
            if response_actions is not None
            else bot_decide_turn_action(game.players[pid], game)
        )
    else:
        assert llm is not None
        raw = llm(prompt)
    action, valid = extract_action(raw, legal)
    return action, valid, prompt


def play_game(game_index: int, seed: int, args: argparse.Namespace, llm: Optional[DecisionFn]):
    random.seed(seed)
    game = MahjongGame(
        f"SFT_{args.teacher}_{seed}_{game_index}",
        [f"T{i}" for i in range(4)],
        bots=[args.teacher == "rule"] * 4,
    )
    start_balances = [p.balance for p in game.players]
    game.start_game()
    for player in game.players:
        if not game.select_exchange_tiles(player.player_id, bot_decide_exchange(player)):
            raise RuntimeError(f"exchange failed for P{player.player_id}")
    for player in game.players:
        if not game.set_missing_suit(player.player_id, bot_decide_missing_suit(player)):
            raise RuntimeError(f"missing-suit selection failed for P{player.player_id}")

    buffers: Dict[int, List[Dict]] = {pid: [] for pid in range(4)}
    clean = {pid: True for pid in range(4)}
    skip_draw = True
    steps = 0

    while not game.is_game_over and steps < args.max_steps:
        steps += 1
        if game.deck.remaining_count() == 0 or sum(p.is_hu for p in game.players) >= 3:
            game.check_game_over()
            break
        pid = game.current_player_id
        player = game.players[pid]
        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        drawn = None
        if not skip_draw:
            drawn = game.draw_tile(pid)
            if drawn is None:
                game.check_game_over()
                break
        else:
            skip_draw = False

        legal = get_legal_actions(game, pid)
        action, valid, prompt = choose_action(game, pid, legal, args.teacher, llm)
        clean[pid] = clean[pid] and valid
        buffers[pid].append(make_row(
            game, pid, prompt, action, legal,
            {"teacher": args.teacher, "seed": seed, "step": steps, "phase": "turn"},
        ))

        if action == "h" and player.can_hu():
            win_tile = drawn if drawn is not None else player.hand_tiles[-1]
            game.hu(pid, win_tile, True)
            game.check_game_over()
            if not game.is_game_over:
                game.next_player()
                skip_draw = False
            continue
        if action == "g":
            gang_info = game.can_self_gang(pid)
            if gang_info.get("can_gang"):
                game.gang(pid, gang_info["gang_tiles"][0])
                skip_draw = True
                continue

        tile = parse_console_tile(action[2:]) if action.startswith("d ") else None
        if tile is None or not game.discard_tile(pid, tile):
            clean[pid] = False
            game.next_player()
            skip_draw = False
            continue

        responses = game.check_responses(tile, pid)
        responded = False
        for rid, available in responses.items():
            if responded:
                break
            response_legal = get_legal_actions(game, rid, response_actions=available)
            response, response_valid, response_prompt = choose_action(
                game, rid, response_legal, args.teacher, llm, response_actions=available,
            )
            clean[rid] = clean[rid] and response_valid
            buffers[rid].append(make_row(
                game, rid, response_prompt, response, response_legal,
                {"teacher": args.teacher, "seed": seed, "step": steps, "phase": "response"},
            ))
            if response == "h" and "hu" in available:
                game.hu(rid, tile, False, pid)
                game.check_game_over()
                responded = True
            elif response == "g" and "gang" in available:
                game.gang(rid, tile, pid)
                game.current_player_id = rid
                responded = True
            elif response == "p" and "peng" in available:
                game.peng(rid, tile, pid)
                game.current_player_id = rid
                responded = True

        if game.is_game_over:
            break
        if responded:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    kept: List[Dict] = []
    player_stats = []
    for player in game.players:
        net = player.balance - start_balances[player.player_id]
        eligible = net > args.min_net and (player.is_hu or not args.require_hu)
        if args.drop_illegal_trajectories:
            eligible = eligible and clean[player.player_id]
        if eligible:
            for row in buffers[player.player_id]:
                row["meta"].update({"game_net": net, "game_hu": bool(player.is_hu)})
                kept.append(row)
        player_stats.append({"pid": player.player_id, "net": net, "hu": bool(player.is_hu), "clean": clean[player.player_id]})
    return kept, player_stats, steps


def main() -> None:
    ensure_deterministic_hashing()
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = args.summary_file or args.output_file.with_suffix(".summary.json")

    llm = None
    if args.teacher == "local_qwen":
        llm = build_llm_callable(
            "local_qwen",
            repo_dir,
            model_path=str((repo_dir / args.model_path).resolve()) if not Path(args.model_path).is_absolute() else args.model_path,
            adapter_path=args.adapter_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    total_rows = 0
    eligible_players = 0
    clean_players = 0
    total_steps = 0
    with args.output_file.open("w", encoding="utf-8") as handle:
        for game_index in tqdm(range(args.games), desc=f"{args.teacher} SFT games"):
            seed = args.seed + game_index
            rows, player_stats, steps = play_game(game_index, seed, args, llm)
            total_steps += steps
            eligible_players += sum(1 for row in player_stats if row["hu"] and row["net"] > args.min_net)
            clean_players += sum(1 for row in player_stats if row["clean"])
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    summary = {
        "teacher": args.teacher,
        "games": args.games,
        "seed": args.seed,
        "model_path": args.model_path if args.teacher == "local_qwen" else None,
        "output_file": str(args.output_file),
        "rows": total_rows,
        "eligible_players": eligible_players,
        "clean_player_rate": clean_players / (args.games * 4) if args.games else None,
        "avg_steps": total_steps / args.games if args.games else None,
        "prompt_schema": "shared_base_action_v1",
        "shanten_engine": "exact_dfs_v2",
    }
    write_json(summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
E0: LLM-native environment and baseline smoke test.

This script does not train a PPO model.  The project direction is now
LLM-native MASK, so E0 checks that the self-owned 4-player Sichuan mahjong
engine, Chinese prompt path, public z_j tracker, and LLM B_phi skeleton can run
end-to-end.  It supports:
  - random: legal random P0 baseline
  - greedy: existing rule bot P0 baseline
  - mask_stub: MASKLLMAgent with heuristic fallback, replaceable by an LLM
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from game import (
    STYLE_BOTS,
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    bot_decide_turn_action,
    parse_console_tile,
)
from mask_llm import MASKLLMAgent
from prompt_builder import get_legal_actions


RESULTS_DIR = Path("E0_results")


def choose_random_action(game: MahjongGame, player_id: int) -> str:
    return random.choice(get_legal_actions(game, player_id))


def choose_greedy_action(game: MahjongGame, player_id: int) -> str:
    return bot_decide_turn_action(game.players[player_id], game)


def init_game(seed: int, opponent_style: str = "greedy") -> Tuple[MahjongGame, Dict[int, Tuple]]:
    random.seed(seed)
    game = MahjongGame(f"E0_{seed}", ["Agent", "B1", "B2", "B3"], bots=[False, True, True, True])
    opp_funcs = {pid: STYLE_BOTS.get(opponent_style, STYLE_BOTS["greedy"]) for pid in (1, 2, 3)}
    game.start_game()

    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))

    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))

    return game, opp_funcs


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None) -> Optional[object]:
    player = game.players[pid]
    action = (action or "").strip()
    discarded_tile = None

    if action == "h" and player.can_hu():
        win_tile = drawn_tile if drawn_tile else (player.hand_tiles[-1] if player.hand_tiles else None)
        if win_tile is not None:
            game.hu(pid, win_tile, True)
            game.check_game_over()
        return None

    if action == "g":
        gang_info = game.can_self_gang(pid)
        if gang_info.get("can_gang"):
            game.gang(pid, gang_info["gang_tiles"][0])
            return None

    if action.startswith("d "):
        tile = parse_console_tile(action[2:])
        if tile and game.discard_tile(pid, tile):
            discarded_tile = tile

    if discarded_tile is None:
        legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
        if legal_discards:
            tile = parse_console_tile(legal_discards[0][2:])
            if tile and game.discard_tile(pid, tile):
                discarded_tile = tile

    return discarded_tile


def resolve_responses(game: MahjongGame, discarded_tile, acting_pid: int, agent: Optional[MASKLLMAgent]) -> Tuple[bool, bool]:
    responses = game.check_responses(discarded_tile, acting_pid)
    responded = False
    agent_won_by_discard = False

    for rid, acts in responses.items():
        if responded:
            break
        player = game.players[rid]
        if rid == 0 and agent is not None:
            valid = get_legal_actions(game, rid, response_actions=acts)
            # Force hu when legal for stable E0 accounting; later E2 can let LLM choose.
            response_action = "h" if "h" in valid else agent.decide(game, valid)
        else:
            response_action = bot_decide_response(player, acts)

        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            responded = True
            agent_won_by_discard = rid == 0
        elif response_action == "g" and "gang" in acts:
            game.gang(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True
        elif response_action == "p" and "peng" in acts:
            game.peng(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True

    return responded, agent_won_by_discard


def play_one_game(method: str, seed: int, opponent_style: str = "greedy", max_steps: int = 300) -> Dict:
    game, opp_funcs = init_game(seed, opponent_style)
    start_balances = [p.balance for p in game.players]
    agent = MASKLLMAgent(player_id=0) if method == "mask_stub" else None

    skip_draw = True
    steps = 0
    decision_times: List[float] = []
    mode_counts: Dict[str, int] = {"safe": 0, "exploit": 0, "deceive": 0}

    while not game.is_game_over and steps < max_steps:
        steps += 1
        if game.deck.remaining_count() == 0 or sum(1 for p in game.players if p.is_hu) >= 3:
            game.check_game_over()
            break

        pid = game.current_player_id
        player = game.players[pid]
        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        drawn_tile = None
        if not skip_draw:
            drawn_tile = game.draw_tile(pid)
            if not drawn_tile:
                game.check_game_over()
                break
        else:
            skip_draw = False

        if pid == 0:
            t0 = time.perf_counter()
            if method == "random":
                action = choose_random_action(game, 0)
            elif method == "greedy":
                action = choose_greedy_action(game, 0)
            elif method == "mask_stub":
                action = agent.decide(game) if agent else choose_greedy_action(game, 0)
                if agent:
                    mode = agent.last_decision.get("mode", "exploit")
                    if mode in mode_counts:
                        mode_counts[mode] += 1
            else:
                raise ValueError(f"Unknown method: {method}")
            decision_times.append((time.perf_counter() - t0) * 1000)
        else:
            action = opp_funcs[pid][0](player, game)

        discarded_tile = execute_action(game, pid, action, drawn_tile)
        if game.is_game_over:
            break

        if discarded_tile is None:
            # Gang already drew a supplement tile; keep same player without drawing.
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        responded, _ = resolve_responses(game, discarded_tile, pid, agent)
        if game.is_game_over:
            break
        if responded:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    end_balances = [p.balance for p in game.players]
    net = [end_balances[i] - start_balances[i] for i in range(4)]
    agent_player = game.players[0]
    agent_dealin = any(p.is_hu and not p.hu_is_self_drawn and p.hu_discard_player_id == 0 for p in game.players)

    return {
        "seed": seed,
        "method": method,
        "opponent_style": opponent_style,
        "steps": steps,
        "agent_hu": bool(agent_player.is_hu),
        "agent_net": net[0],
        "agent_dealin": bool(agent_dealin),
        "winners": list(game.winners),
        "decision_ms": decision_times,
        "mode_counts": mode_counts,
        "history_len": len(game.history),
    }


def summarize(results: List[Dict]) -> Dict:
    n = max(1, len(results))
    all_times = [t for r in results for t in r["decision_ms"]]
    sorted_times = sorted(all_times)

    def percentile(p: float) -> float:
        if not sorted_times:
            return 0.0
        idx = min(len(sorted_times) - 1, int(round((len(sorted_times) - 1) * p)))
        return sorted_times[idx]

    mode_totals = {"safe": 0, "exploit": 0, "deceive": 0}
    for r in results:
        for k, v in r["mode_counts"].items():
            mode_totals[k] = mode_totals.get(k, 0) + v

    return {
        "games": len(results),
        "hu_rate": sum(1 for r in results if r["agent_hu"]) / n,
        "money_rate": sum(1 for r in results if r["agent_net"] >= 0) / n,
        "avg_net": sum(r["agent_net"] for r in results) / n,
        "dealin_rate": sum(1 for r in results if r["agent_dealin"]) / n,
        "avg_steps": sum(r["steps"] for r in results) / n,
        "decision_latency_ms": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        },
        "mode_counts": mode_totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["random", "greedy", "mask_stub"])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--opponent-style", default="greedy", choices=sorted(STYLE_BOTS.keys()))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    full_report = {}

    for method in args.methods:
        results = [
            play_one_game(method, args.seed + i, opponent_style=args.opponent_style)
            for i in range(args.games)
        ]
        summary = summarize(results)
        full_report[method] = {"summary": summary, "games": results}
        print(f"\n[E0] {method}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = RESULTS_DIR / "e0_llm_baseline_report.json"
    output.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved E0 report: {output}")


if __name__ == "__main__":
    main()

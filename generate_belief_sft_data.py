"""
Generate belief-SFT data for B_phi.

Each record is a chat sample:
  system: B_phi role and JSON-only instruction
  user: public-only belief prompt from prompt_builder.build_belief_prompt
  assistant: oracle JSON label derived from the game engine's hidden state

The prompt does not expose P0's hand.  The label is allowed to use engine
information because this is supervised training data construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from belief_oracle import opponent_view_posterior, within_shanten
from experiment_trace import ensure_deterministic_hashing
from game import (
    STYLE_BOTS,
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    parse_console_tile,
)
from prompt_builder import build_belief_prompt, get_legal_actions
from rule_engine import ShantenCalculator


OPPONENT_STYLES = tuple(sorted(STYLE_BOTS.keys())) + ("mixed",)
DEFAULT_OUTPUT = Path("belief_sft_data.jsonl")


def init_game(seed: int, opponent_style: str, game_id: str) -> Tuple[MahjongGame, Dict[int, Tuple]]:
    random.seed(seed)
    game = MahjongGame(game_id, ["Agent", "B1", "B2", "B3"], bots=[False, True, True, True])
    if opponent_style == "mixed":
        styles = ["aggressive", "conservative", "random"]
        opponent_funcs = {pid: STYLE_BOTS[styles[pid - 1]] for pid in (1, 2, 3)}
    else:
        opponent_funcs = {pid: STYLE_BOTS.get(opponent_style, STYLE_BOTS["greedy"]) for pid in (1, 2, 3)}

    game.start_game()
    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))
    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))
    return game, opponent_funcs


def choose_min_shanten_action(game: MahjongGame, player_id: int) -> str:
    valid_actions = get_legal_actions(game, player_id)
    player = game.players[player_id]
    if "h" in valid_actions:
        return "h"
    if "g" in valid_actions:
        return "g"

    best_action = valid_actions[0] if valid_actions else "n"
    best_shanten = 99
    for action in valid_actions:
        if not action.startswith("d "):
            continue
        tile_text = action[2:]
        for tile in player.hand_tiles:
            if str(tile) != tile_text:
                continue
            temp = player.hand_tiles.copy()
            temp.remove(tile)
            shanten = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
            if shanten < best_shanten:
                best_shanten = shanten
                best_action = action
            break
    return best_action


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None) -> Optional[object]:
    player = game.players[pid]
    action = (action or "").strip()

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
            return tile

    legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
    if legal_discards:
        tile = parse_console_tile(legal_discards[0][2:])
        if tile and game.discard_tile(pid, tile):
            return tile
    return None


def resolve_responses(game: MahjongGame, discarded_tile, acting_pid: int) -> bool:
    responses = game.check_responses(discarded_tile, acting_pid)
    for rid, acts in responses.items():
        player = game.players[rid]
        response_action = bot_decide_response(player, acts)
        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            return True
        if response_action == "g" and "gang" in acts:
            game.gang(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            return True
        if response_action == "p" and "peng" in acts:
            game.peng(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            return True
    return False


def soft_tenpai_probability(shanten: int, is_ready: bool) -> float:
    if is_ready:
        return 0.95
    if shanten <= 1:
        return 0.55
    if shanten == 2:
        return 0.25
    return 0.05


def confidence_label(probability: float) -> str:
    if probability >= 0.65:
        return "yes"
    if probability <= 0.35:
        return "no"
    return "uncertain"


def infer_pattern(game: MahjongGame, player_id: int) -> str:
    player = game.players[player_id]
    visible_like_tiles = list(player.hand_tiles)
    for meld in player.open_melds:
        visible_like_tiles.extend(meld)

    suits = {tile.suit for tile in visible_like_tiles if player.missing_suit is None or tile.suit != player.missing_suit}
    if len(suits) == 1 and len(visible_like_tiles) >= 10:
        return "qingyise"

    counts = Counter((tile.suit.value, tile.number) for tile in player.hand_tiles)
    pair_count = sum(1 for count in counts.values() if count >= 2)
    if pair_count >= 5:
        return "qidui"

    triplet_like = len(player.open_melds)
    triplet_like += sum(1 for count in counts.values() if count >= 3)
    if triplet_like >= 3:
        return "pengpenghu"

    return "normal"


def public_pattern(game: MahjongGame, player_id: int) -> str:
    """Pattern guess from PUBLIC melds only (no concealed-hand peek)."""
    player = game.players[player_id]
    melds = player.open_melds
    if len(melds) >= 3:
        return "pengpenghu"
    meld_suits = {tile.suit for meld in melds for tile in meld}
    if melds and len(meld_suits) == 1:
        return "qingyise"
    return "unknown"


def oracle_belief_json(
    game: MahjongGame,
    player_id: int,
    target_opponent_id: int,
    label_source: str = "opponent_posterior",
    oracle_samples: int = 30,
    beta: float = 2.0,
    rng: Optional[random.Random] = None,
    precomputed_probability: Optional[float] = None,
    danger_threshold: int = 0,
) -> Dict[str, Any]:
    player = game.players[player_id]
    shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
    ready = within_shanten(player.hand_tiles, player.missing_suit, danger_threshold)

    if precomputed_probability is not None:
        # Reuse the single public-info posterior across all observers j, so MC
        # sampling noise cannot masquerade as a per-j difference.
        probability = float(precomputed_probability)
    elif label_source == "opponent_posterior":
        post = opponent_view_posterior(
            game, player_id, target_opponent_id,
            num_samples=oracle_samples, rng=rng, beta=beta, max_shanten=danger_threshold,
        )
        probability = float(post["tenpai_prob"])
    else:  # "true_tenpai": peeking soft proxy from real shanten (same across j)
        probability = soft_tenpai_probability(shanten, ready)

    own_discards = len(player.discarded_tiles)
    melds = len(player.open_melds)
    tiles_left = game.deck.remaining_count()

    # suspected_waits / danger left empty on purpose: concrete hidden waits are
    # not inferable from public info, so peeking them would teach hallucination.
    # suspected_pattern is derived from public melds only.
    return {
        "target_opponent": f"P{target_opponent_id}",
        "think_i_am_tenpai": confidence_label(probability),
        "tenpai_confidence": round(probability, 3),
        "suspected_waits": [],
        "suspected_pattern": public_pattern(game, player_id),
        "danger_tiles_for_me": [],
        "reason": f"P{target_opponent_id} 可见：我方已弃{own_discards}张、副露{melds}组、牌局剩余{tiles_left}张。",
    }


def make_record(
    game: MahjongGame,
    seed: int,
    step: int,
    player_id: int,
    target_id: int,
    label_source: str = "opponent_posterior",
    oracle_samples: int = 30,
    beta: float = 2.0,
    rng: Optional[random.Random] = None,
    precomputed_probability: Optional[float] = None,
    danger_threshold: int = 0,
) -> Dict[str, Any]:
    prompt = build_belief_prompt(game, player_id, target_id)
    label = oracle_belief_json(
        game, player_id, target_id,
        label_source=label_source, oracle_samples=oracle_samples, beta=beta, rng=rng,
        precomputed_probability=precomputed_probability, danger_threshold=danger_threshold,
    )
    player = game.players[player_id]
    _, waits = player.is_ready_with_missing_suit()
    shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
    ready = within_shanten(player.hand_tiles, player.missing_suit, danger_threshold)

    return {
        "messages": [
            {
                "role": "system",
                "content": "你是四川麻将 MASK 的 B_phi 信念估计器。只根据公开信息输出严格 JSON。",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
        "meta": {
            "game_id": game.game_id,
            "seed": seed,
            "step": step,
            "history_len": len(game.history),
            "player_id": player_id,
            "target_opponent_id": target_id,
            "true_tenpai": bool(ready),
            "true_waits": [str(tile) for tile in waits],
            "true_shanten": int(shanten),
            "label_source": label_source,
            "oracle_tenpai_prob": label["tenpai_confidence"],
            "oracle_scope": (
                "play-aware per-observer opponent-view posterior (no peek)"
                if label_source == "opponent_posterior"
                else "soft shanten-derived posterior proxy (peeking)"
            ),
        },
    }


def generate_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for game_idx in range(args.games):
        seed = args.seed + game_idx
        game, opponent_funcs = init_game(seed, args.opponent_style, f"BPHI_{seed}")
        oracle_rng = random.Random(seed * 1000 + 7)
        skip_draw = True
        steps = 0

        while not game.is_game_over and steps < args.max_steps:
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

            sample_player_ids = range(4) if args.all_players else (args.player_id,)
            if steps == 1 or steps % args.sample_every == 0:
                for source_pid in sample_player_ids:
                    if game.players[source_pid].is_hu:
                        continue
                    # Public-info posterior is observer-independent: compute it
                    # ONCE per source and reuse for j=1/2/3 (one MC call, no noise).
                    shared_prob: Optional[float] = None
                    if args.label_source == "opponent_posterior":
                        rep_obs = next(t for t in range(4) if t != source_pid)
                        post = opponent_view_posterior(
                            game, source_pid, rep_obs,
                            num_samples=args.oracle_samples, rng=oracle_rng, beta=args.oracle_beta,
                            max_shanten=args.danger_threshold,
                        )
                        shared_prob = float(post["tenpai_prob"])
                    for target_id in range(4):
                        if target_id == source_pid:
                            continue
                        record = make_record(
                            game, seed, steps, source_pid, target_id,
                            label_source=args.label_source,
                            oracle_samples=args.oracle_samples,
                            beta=args.oracle_beta,
                            rng=oracle_rng,
                            precomputed_probability=shared_prob,
                            danger_threshold=args.danger_threshold,
                        )
                        records.append(record)

            if len(records) >= args.max_records:
                return records

            if pid == args.player_id:
                action = choose_min_shanten_action(game, pid)
            else:
                action = opponent_funcs[pid][0](player, game)

            discarded_tile = execute_action(game, pid, action, drawn_tile)
            if game.is_game_over:
                break
            if discarded_tile is None:
                if action == "g":
                    skip_draw = True
                    continue
                game.next_player()
                skip_draw = False
                continue

            responded = resolve_responses(game, discarded_tile, pid)
            if game.is_game_over:
                break
            if responded:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False

    return records


def balance_records(records: List[Dict[str, Any]], max_per_label: int) -> List[Dict[str, Any]]:
    if max_per_label <= 0:
        return records

    grouped: Dict[str, List[Dict[str, Any]]] = {"yes": [], "uncertain": [], "no": []}
    for record in records:
        label = json.loads(record["messages"][-1]["content"]).get("think_i_am_tenpai", "uncertain")
        grouped.setdefault(label, []).append(record)

    rng = random.Random(3407)
    balanced: List[Dict[str, Any]] = []
    for label in ("yes", "uncertain", "no"):
        rows = grouped.get(label, [])
        rng.shuffle(rows)
        balanced.extend(rows[:max_per_label])
    rng.shuffle(balanced)
    return balanced


def target_balance_records(
    records: List[Dict[str, Any]],
    target_per_label: int,
    max_oversample_ratio: float = 3.0,
) -> List[Dict[str, Any]]:
    """Balance classes toward target_per_label, but cap duplication.

    Oversampling beyond a few times the natural count makes the model memorise a
    handful of rare-class prompts (low train loss, poor generalisation).
    ``max_oversample_ratio`` limits each class to at most ratio x its natural
    count, so the per-class size is min(target_per_label, ceil(ratio*natural)).
    """
    if target_per_label <= 0:
        return records

    grouped: Dict[str, List[Dict[str, Any]]] = {"yes": [], "uncertain": [], "no": []}
    for record in records:
        label = json.loads(record["messages"][-1]["content"]).get("think_i_am_tenpai", "uncertain")
        grouped.setdefault(label, []).append(record)

    rng = random.Random(3407)
    balanced: List[Dict[str, Any]] = []
    for label in ("yes", "uncertain", "no"):
        rows = grouped.get(label, [])
        if not rows:
            continue
        rng.shuffle(rows)
        cap = target_per_label
        if max_oversample_ratio > 0:
            cap = min(cap, int(math.ceil(max_oversample_ratio * len(rows))))
        if len(rows) >= cap:
            balanced.extend(rows[:cap])
        else:
            balanced.extend(rows)
            balanced.extend(rng.choice(rows) for _ in range(cap - len(rows)))
    rng.shuffle(balanced)
    return balanced


def label_of(record: Dict[str, Any]) -> str:
    return json.loads(record["messages"][-1]["content"]).get("think_i_am_tenpai", "uncertain")


def count_labels(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Counter = Counter()
    for record in records:
        counts[label_of(record)] += 1
    return dict(counts)


def write_splits(
    records: List[Dict[str, Any]],
    output: Path,
    train_ratio: float,
    max_per_label: int = 0,
    target_per_label: int = 0,
    max_oversample_ratio: float = 3.0,
    label_source: str = "opponent_posterior",
    danger_threshold: int = 0,
) -> Dict[str, Any]:
    """Split *before* any class balancing, then oversample the TRAIN split only.

    Oversampling duplicates rare-class records, so doing it before the split (the
    previous behaviour) let identical rows land in both train and eval, leaking
    eval into train.  Here the eval split stays natural (no duplication), and
    balancing is applied only to train.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    train_path = output.with_name(output.stem + "_train.jsonl")
    eval_path = output.with_name(output.stem + "_eval.jsonl")
    meta_path = output.with_name(output.stem + "_meta.json")

    random.Random(3407).shuffle(records)
    cutoff = max(1, int(len(records) * train_ratio))
    train = records[:cutoff]
    eval_records = records[cutoff:]

    # Balance the training split only; eval reflects the natural distribution so
    # metrics are honest (the evaluators can balance at eval time if needed).
    train = balance_records(train, max_per_label)
    train = target_balance_records(train, target_per_label, max_oversample_ratio=max_oversample_ratio)
    random.Random(11).shuffle(train)

    combined = train + eval_records
    for path, rows in ((output, combined), (train_path, train), (eval_path, eval_records)):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    shanten_counts = Counter()
    for row in eval_records:
        shanten_counts[str(row["meta"]["true_shanten"])] += 1

    meta = {
        "records": len(combined),
        "train_records": len(train),
        "eval_records": len(eval_records),
        "output": str(output),
        "train_output": str(train_path),
        "eval_output": str(eval_path),
        "train_label_counts": count_labels(train),
        "eval_label_counts": count_labels(eval_records),
        "eval_shanten_counts": dict(shanten_counts),
        "label_source": label_source,
        "danger_threshold": danger_threshold,
        "positive_class": f"shanten<={danger_threshold} ({'precise tenpai' if danger_threshold == 0 else 'danger/near-tenpai'})",
        "split_policy": "split-before-balance; train oversampled, eval left natural (leakage-safe)",
        "format": "chat messages; assistant content is strict JSON for B_phi",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--opponent-style", default="mixed", choices=list(OPPONENT_STYLES))
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--all-players", action="store_true")
    parser.add_argument("--max-steps", type=int, default=260)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--target-per-label", type=int, default=0)
    parser.add_argument("--max-oversample-ratio", type=float, default=3.0,
                        help="Cap per-class size to ratio x its natural count (limits duplication).")
    parser.add_argument("--label-source", default="opponent_posterior",
                        choices=["opponent_posterior", "true_tenpai"],
                        help="opponent_posterior = per-j play-aware MC posterior (no peek); true_tenpai = peeking soft proxy.")
    parser.add_argument("--oracle-samples", type=int, default=30,
                        help="MC samples per record for opponent_posterior labels.")
    parser.add_argument("--oracle-beta", type=float, default=2.0,
                        help="Play-aware importance weight exp(-beta*shanten).")
    parser.add_argument("--danger-threshold", type=int, default=0,
                        help="Positive class = shanten<=this. 0=precise tenpai; 1='danger' (public-readable, recommended).")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records = generate_records(args)
    meta = write_splits(
        records,
        args.output,
        args.train_ratio,
        max_per_label=args.max_per_label,
        target_per_label=args.target_per_label,
        max_oversample_ratio=args.max_oversample_ratio,
        label_source=args.label_source,
        danger_threshold=args.danger_threshold,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Self-play data generator for the ResponsiveDefender 'learned' danger model.

Records one row per P0 discard: the same public features discard_tell_threat()
reads (P0's own discard, meld count, tiles left, others' safe-discard set),
labeled with the oracle within_shanten(P0's hand) at that moment. Games mix
MASKLLMAgent(deceive_style="threat", forced_deceive="eligible") P0s (guarantees
real disguised-discard examples) with plain min-shanten P0s (honest baseline),
so the model has to discriminate deceptive-looking-safe-but-still-dangerous
hands from honestly-safe ones, not just "middle tiles -> danger".

P1-3 are always ResponsiveDefender (opponent_style="responsive"), reusing
init_game()/execute_action()/resolve_responses() from run_gate1_experiments.py.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

from belief_oracle import opponent_view_posterior
from experiment_trace import ensure_deterministic_hashing, write_json, write_jsonl
from opponent_classifier import extract_action_features
from prompt_builder import get_legal_actions
from rule_engine import ShantenCalculator

from mask_llm import MASKLLMAgent
from run_gate1_experiments import (
    choose_min_shanten_action,
    discard_tell_threat,
    execute_action,
    init_game,
    resolve_responses,
)


def play_one_data_game(
    game_id: str,
    seed: int,
    use_mask: bool,
    max_steps: int,
    danger_threshold: int,
    tell_window: int,
    mc_samples: int,
    sample_every: int,
    defender_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    game, opponent_funcs, defenders = init_game(seed, "responsive", game_id, defender_cfg)
    mask_agent = (
        MASKLLMAgent(
            player_id=0,
            mc_seed=seed * 13 + 1,
            mc_danger_threshold=danger_threshold,
            forced_deceive="eligible",
            deceive_style="threat",
        )
        if use_mask else None
    )
    mc_rng = random.Random(seed * 777 + 1)

    skip_draw = True
    steps = 0
    p0_discard_idx = 0
    step_rows: List[Dict[str, Any]] = []

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

        decision_state: Dict[str, Any] = {"mode": "plain", "deceive_style": "none"}
        others_before = None
        if pid == 0:
            legal_actions = get_legal_actions(game, 0)
            others_before = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
            if use_mask:
                assert mask_agent is not None
                action = mask_agent.decide(game, legal_actions)
                decision_state = dict(mask_agent.last_decision)
            else:
                action = choose_min_shanten_action(game, 0, legal_actions)
        elif pid in defenders:
            action = defenders[pid].turn(player, game, step=steps)
        else:
            action = opponent_funcs[pid][0](player, game)

        discarded_tile = execute_action(game, pid, action, drawn_tile)
        if game.is_game_over:
            break

        if pid == 0 and discarded_tile is not None:
            p0_discard_idx += 1
            if p0_discard_idx % sample_every == 0:
                is_safe = (discarded_tile.suit, discarded_tile.number) in others_before
                meld_count = len(player.open_melds)
                feats = extract_action_features(
                    discarded_tile, "discard", p0_discard_idx, meld_count, is_safe, meld_count,
                )
                true_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
                label = int(true_shanten <= danger_threshold)
                tell_score = discard_tell_threat(game, 0, tell_window)
                mc_score = None
                if mc_samples > 0:
                    post = opponent_view_posterior(
                        game, target_pid=0, observer_pid=1, num_samples=mc_samples,
                        rng=mc_rng, beta=2.0, max_shanten=danger_threshold,
                    )
                    mc_score = float(post["tenpai_prob"])
                step_rows.append({
                    "step_idx": p0_discard_idx,
                    "features": feats.tolist(),
                    "label": label,
                    "true_shanten": int(true_shanten),
                    "mode": decision_state.get("mode", "plain"),
                    "is_deceive_threat": bool(
                        decision_state.get("mode") == "deceive"
                        and decision_state.get("deceive_style") == "threat"
                    ),
                    "tell_score": round(tell_score, 4),
                    "mc_score": (round(mc_score, 4) if mc_score is not None else None),
                    "tiles_left": game.deck.remaining_count(),
                    "meld_count": meld_count,
                })

        if discarded_tile is None:
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        response_info = resolve_responses(game, discarded_tile, pid, mask_agent, defenders)
        if game.is_game_over:
            break
        if response_info["responded"]:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    return {
        "game_id": game.game_id,
        "seed": seed,
        "p0_mode": "mask" if use_mask else "plain",
        "steps": step_rows,
    }


def write_splits(games: List[Dict[str, Any]], output: Path, train_ratio: float) -> Dict[str, Any]:
    """Leak-safe split: shuffle whole games, then split by game (never by row)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    train_path = output.with_name(output.stem + "_train.jsonl")
    eval_path = output.with_name(output.stem + "_eval.jsonl")
    meta_path = output.with_name(output.stem + "_meta.json")

    random.Random(3407).shuffle(games)
    cutoff = max(1, int(len(games) * train_ratio))
    train_games = games[:cutoff]
    eval_games = games[cutoff:]

    write_jsonl(output, games)
    write_jsonl(train_path, train_games)
    write_jsonl(eval_path, eval_games)

    def _counts(gs: List[Dict[str, Any]]) -> Dict[str, Any]:
        rows = [s for g in gs for s in g["steps"]]
        pos = sum(1 for r in rows if r["label"] == 1)
        deceive_rows = sum(1 for r in rows if r["is_deceive_threat"])
        return {"games": len(gs), "rows": len(rows), "pos_rate": (pos / len(rows)) if rows else 0.0,
                "deceive_threat_rows": deceive_rows}

    meta = {
        "train_ratio": train_ratio,
        "train": _counts(train_games),
        "eval": _counts(eval_games),
    }
    write_json(meta_path, meta)
    return meta


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--tell-window", type=int, default=6)
    parser.add_argument("--mc-samples", type=int, default=20,
                         help="Reference-only MC posterior samples per row (0 disables it).")
    parser.add_argument("--sample-every", type=int, default=1,
                         help="Record every Nth P0 discard (1 = record all).")
    parser.add_argument("--deceive-fraction", type=float, default=0.5,
                         help="Share of games where P0 is a MASK threat-style deceiver.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--defender-threat-model", choices=["mc", "discard_tell", "blend"], default="blend")
    parser.add_argument("--output-file", type=Path, default=Path("Defender_danger_data/danger_data.jsonl"))
    args = parser.parse_args()

    defender_cfg = {
        "threat_threshold": 0.4,
        "oracle_samples": 30,
        "beta": 2.0,
        "danger_threshold": args.danger_threshold,
        "ffr_hand_shanten": 1,
        "threat_model": args.defender_threat_model,
        "tell_weight": 0.3,
        "tell_window": args.tell_window,
    }

    rng = random.Random(args.seed)
    games: List[Dict[str, Any]] = []
    for i in range(args.games):
        seed = args.seed + i
        use_mask = rng.random() < args.deceive_fraction
        game_id = f"DangerData_{'mask' if use_mask else 'plain'}_{seed}"
        row = play_one_data_game(
            game_id, seed, use_mask, args.max_steps, args.danger_threshold,
            args.tell_window, args.mc_samples, args.sample_every, defender_cfg,
        )
        games.append(row)

    meta = write_splits(games, args.output_file, args.train_ratio)
    print(f"Wrote {len(games)} games -> {args.output_file}")
    print(meta)


if __name__ == "__main__":
    main()

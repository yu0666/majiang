"""Collect imitation data for a neural opponent from responsive+learned defender."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_trace import ensure_deterministic_hashing, write_json
from game import bot_decide_response
from mask_llm import MASKLLMAgent
from neural_opponent_policy import ACTION_SPACE, FEATURE_VERSION, extract_policy_features
from prompt_builder import get_legal_actions
from run_gate1_experiments import (
    choose_min_shanten_action,
    choose_public_safe_no_regret,
    execute_action,
    init_game,
)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_p0_action(game, rng: random.Random, mask_agent: Optional[MASKLLMAgent], mask_rate: float) -> str:
    legal = get_legal_actions(game, 0)
    base = choose_min_shanten_action(game, 0, legal)
    if mask_agent is not None and rng.random() < mask_rate:
        try:
            return mask_agent.decide(game, legal)
        except Exception:
            return base
    if rng.random() < 0.25:
        return choose_public_safe_no_regret(game, 0, legal, base)
    return base


def record_example(
    rows: List[Dict[str, Any]],
    game,
    seed: int,
    game_index: int,
    step: int,
    pid: int,
    phase: str,
    legal_actions: List[str],
    action: str,
    response_actions: Optional[List[str]] = None,
) -> None:
    if action not in legal_actions:
        return
    rows.append(
        {
            "feature_version": FEATURE_VERSION,
            "seed": seed,
            "game_index": game_index,
            "game_id": game.game_id,
            "step": step,
            "pid": pid,
            "phase": phase,
            "legal_actions": legal_actions,
            "response_actions": response_actions or [],
            "action": action,
            "features": extract_policy_features(
                game,
                pid,
                legal_actions=legal_actions,
                response_actions=response_actions,
            ),
        }
    )


def collect(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_deterministic_hashing()
    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    defender_cfg = {
        "threat_threshold": args.threat_fold_threshold,
        "oracle_samples": args.oracle_samples,
        "beta": args.oracle_beta,
        "danger_threshold": args.danger_threshold,
        "ffr_hand_shanten": args.ffr_hand_shanten,
        "threat_model": "learned",
        "tell_weight": args.defender_tell_weight,
        "tell_window": args.defender_tell_window,
        "learned_model_path": args.defender_learned_model_path,
    }

    for game_index in range(args.games):
        game_seed = args.seed + game_index * 1009
        game, opponent_funcs, defenders = init_game(
            game_seed,
            "responsive",
            f"NeuralOpponentTeacher_{game_index}",
            defender_cfg,
        )
        mask_agent = (
            MASKLLMAgent(
                player_id=0,
                decision_llm=None,
                belief_llm=None,
                mc_seed=game_seed * 13 + 1,
                mc_oracle_samples=args.mask_oracle_samples,
                mc_beta=args.mask_oracle_beta,
                mc_danger_threshold=args.mask_danger_threshold,
                forced_deceive=args.mask_forced_deceive,
                deceive_style="threat",
                threat_response_model="blend",
                threat_response_tell_weight=args.defender_tell_weight,
                threat_target_signal="mc",
                threat_target_prob_threshold=0.78,
                threat_require_real_target=True,
                threat_target_max_shanten=1,
            )
            if args.p0_mask_rate > 0.0
            else None
        )

        skip_draw = True
        for step in range(1, args.max_steps + 1):
            if game.is_game_over or game.deck.remaining_count() == 0:
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
                action = choose_p0_action(game, rng, mask_agent, args.p0_mask_rate)
            elif pid in defenders:
                legal = get_legal_actions(game, pid)
                action = defenders[pid].turn(player, game, step=step)
                record_example(rows, game, game_seed, game_index, step, pid, "turn", legal, action)
                phase_counts["turn"] += 1
                action_counts[action] += 1
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

            responses = game.check_responses(discarded_tile, pid)
            responded = False
            for rid, acts in responses.items():
                if responded:
                    break
                responder = game.players[rid]
                if rid in defenders:
                    legal = get_legal_actions(game, rid, response_actions=acts)
                    response_action = defenders[rid].response(responder, acts, game)
                    record_example(
                        rows,
                        game,
                        game_seed,
                        game_index,
                        step,
                        rid,
                        "response",
                        legal,
                        response_action,
                        response_actions=list(acts),
                    )
                    phase_counts["response"] += 1
                    action_counts[response_action] += 1
                else:
                    response_action = "h" if "hu" in acts else bot_decide_response(responder, acts)

                if response_action == "h" and "hu" in acts:
                    game.hu(rid, discarded_tile, False, pid)
                    game.check_game_over()
                    responded = True
                elif response_action == "g" and "gang" in acts:
                    game.gang(rid, discarded_tile, pid)
                    game.current_player_id = rid
                    responded = True
                elif response_action == "p" and "peng" in acts:
                    game.peng(rid, discarded_tile, pid)
                    game.current_player_id = rid
                    responded = True

            if game.is_game_over:
                break
            if responded:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False

        if (game_index + 1) % max(1, args.log_every) == 0:
            print(
                f"[collect] games={game_index + 1}/{args.games} rows={len(rows)} "
                f"turn={phase_counts['turn']} response={phase_counts['response']}",
                flush=True,
            )

    write_jsonl(args.output_file, rows)
    summary = {
        "output_file": str(args.output_file),
        "games": args.games,
        "rows": len(rows),
        "feature_version": FEATURE_VERSION,
        "feature_dim": len(rows[0]["features"]) if rows else None,
        "actions": ACTION_SPACE,
        "phase_counts": dict(phase_counts),
        "action_counts": dict(action_counts),
        "teacher": "responsive_defender_with_learned_danger_model",
        "defender_cfg": defender_cfg,
        "p0_mask_rate": args.p0_mask_rate,
    }
    write_json(args.output_file.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026071601)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-file", type=Path, default=Path("Neural_opponent_data/responsive_learned_teacher.jsonl"))
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--oracle-samples", type=int, default=30)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--ffr-hand-shanten", type=int, default=1)
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth")
    parser.add_argument("--p0-mask-rate", type=float, default=0.35)
    parser.add_argument("--mask-oracle-samples", type=int, default=30)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=1)
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="eligible")
    args = parser.parse_args()
    summary = collect(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

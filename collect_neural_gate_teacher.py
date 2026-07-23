"""Collect online-feature neural gate distillation data from a rule teacher.

The teacher chooses the actual action/mode, usually continuous_v2.  The collected
row uses the same feature extractor and mode availability constraints as the
deployed neural gate, so train/serve feature and action-set drift is visible and
measurable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiment_trace import ensure_deterministic_hashing, write_json
from llm_backends import build_llm_callable
from mask_llm import MASKLLMAgent
from ppo_features import extract_state_features
from prompt_builder import get_legal_actions
from run_gate1_experiments import execute_action, init_game, resolve_responses


MODES = ("exploit", "safe", "deceive")


def defender_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "threat_threshold": args.threat_fold_threshold,
        "oracle_samples": args.oracle_samples,
        "beta": args.oracle_beta,
        "danger_threshold": args.danger_threshold,
        "ffr_hand_shanten": args.ffr_hand_shanten,
        "threat_model": args.defender_threat_model,
        "tell_weight": args.defender_tell_weight,
        "tell_window": args.defender_tell_window,
        "learned_model_path": args.defender_learned_model_path,
        "neural_model_path": args.neural_opponent_model_path,
        "neural_device": args.neural_opponent_device,
    }


def make_agent(seed: int, llm, args: argparse.Namespace, gate_policy: str) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=llm,
        belief_llm=llm,
        mc_seed=seed * 13 + 1,
        mc_oracle_samples=args.mask_oracle_samples,
        mc_beta=args.mask_oracle_beta,
        mc_danger_threshold=args.mask_danger_threshold,
        dir_ready_threshold=args.mask_dir_ready_threshold,
        deceive_threat_ceiling=args.mask_deceive_threat_ceiling,
        forced_deceive=args.mask_forced_deceive,
        deceive_style=args.mask_deceive_style,
        threat_allow_break_ready=args.mask_threat_allow_break_ready,
        threat_max_result_shanten=args.mask_threat_max_result_shanten,
        threat_max_shanten_regret=args.mask_threat_max_shanten_regret,
        threat_min_ukeire_ratio=args.mask_threat_min_ukeire_ratio,
        threat_gate_threshold=args.mask_threat_gate_threshold,
        threat_gate_margin=args.mask_threat_gate_margin,
        threat_min_delta=args.mask_threat_min_delta,
        threat_gate_mode=args.mask_threat_gate_mode,
        threat_response_model=args.mask_threat_response_model,
        threat_response_tell_weight=args.mask_threat_response_tell_weight,
        threat_tell_window=args.mask_threat_tell_window,
        threat_max_start_shanten=args.mask_threat_max_start_shanten,
        threat_require_non_exploit=not args.mask_threat_allow_exploit_overlap,
        threat_require_real_target=args.mask_threat_require_real_target,
        threat_target_max_shanten=args.mask_threat_target_max_shanten,
        threat_target_signal=args.mask_threat_target_signal,
        threat_target_prob_threshold=args.mask_threat_target_prob_threshold,
        log_counterfactual=False,
        gate_policy=gate_policy,
    )


def neural_available_modes(
    agent: MASKLLMAgent,
    game,
    legal_actions: List[str],
    beliefs: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    exploit_action, exploit_reason = agent._exploit_action(game, legal_actions)
    safe_action = agent._safe_discard(game, legal_actions, fallback_action=exploit_action)
    actions: Dict[str, str] = {"exploit": exploit_action}
    available = ["exploit"]
    if any(action.startswith("d ") for action in legal_actions):
        available.append("safe")
        actions["safe"] = safe_action

    rng_state = agent.rng.getstate()
    try:
        deceive_action = agent._deceptive_discard(
            game,
            legal_actions,
            beliefs,
            relaxed_gate=True,
        )
    finally:
        agent.rng.setstate(rng_state)

    can_deceive = bool(deceive_action and deceive_action.startswith("d ") and deceive_action in legal_actions)
    block_reason: Optional[str] = None
    if not deceive_action:
        block_reason = (agent._last_deceive_signal or {}).get("blocked_reason") or "candidate_unavailable"
    elif not deceive_action.startswith("d "):
        block_reason = "candidate_not_discard"
    if can_deceive and agent.deceive_style == "threat" and agent.threat_require_non_exploit:
        if deceive_action == exploit_action:
            can_deceive = False
            block_reason = "same_as_exploit"
        elif not exploit_action.startswith("d "):
            can_deceive = False
            block_reason = "counterfactual_exploit_not_discard"
    if can_deceive:
        available.append("deceive")
        actions["deceive"] = deceive_action
        block_reason = None

    return available, {
        "mode_actions": actions,
        "exploit_reason": exploit_reason,
        "deceive_action": deceive_action,
        "can_deceive": can_deceive,
        "deceive_block_reason": block_reason,
        "deceive_signal": agent._last_deceive_signal,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_deterministic_hashing()
    random.seed(args.seed)
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "teacher_gate_states.jsonl"
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    attempts = 0
    rows = 0
    quotas = {
        "exploit": args.target_exploit,
        "safe": args.target_safe,
        "deceive": args.target_deceive,
    }
    quota_mode = any(value > 0 for value in quotas.values())

    def collection_done() -> bool:
        if quota_mode:
            return all(counts[mode] >= target for mode, target in quotas.items())
        return rows >= args.target_states

    with rows_path.open("w", encoding="utf-8") as handle:
        for game_index in range(args.max_games):
            if collection_done():
                break
            game_seed = args.seed + game_index
            game, opponent_funcs, defenders = init_game(
                game_seed,
                args.opponent_style,
                f"NeuralGateTeacher_{game_seed}",
                defender_config(args),
            )
            teacher = make_agent(game_seed, llm, args, args.teacher_gate_policy)
            candidate_agent = make_agent(game_seed, llm, args, "rule")
            skip_draw = True
            steps = 0
            last_p0_state: Dict[str, Any] = {}
            deceive_active_until = -1

            while not game.is_game_over and steps < args.max_steps and not collection_done():
                steps += 1
                if game.deck.remaining_count() == 0 or sum(player.is_hu for player in game.players) >= 3:
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
                    if drawn_tile is None:
                        game.check_game_over()
                        break
                else:
                    skip_draw = False

                if pid == 0:
                    legal_actions = get_legal_actions(game, 0)
                    action = teacher.decide(game, legal_actions)
                    decision_state = dict(teacher.last_decision)
                    target_mode = str(decision_state.get("mode", "exploit"))
                    if (
                        "h" not in legal_actions
                        and any(item.startswith("d ") for item in legal_actions)
                        and steps % args.sample_stride == 0
                    ):
                        attempts += 1
                        available, diagnostics = neural_available_modes(
                            candidate_agent,
                            game,
                            legal_actions,
                            decision_state.get("beliefs", {}),
                        )
                        if target_mode not in available:
                            skipped[f"target_unavailable_{target_mode}"] += 1
                        elif quota_mode and counts[target_mode] >= quotas[target_mode]:
                            skipped[f"quota_full_{target_mode}"] += 1
                        else:
                            features = extract_state_features(
                                game,
                                0,
                                decision_state.get("z_state", {}),
                                decision_state.get("beliefs", {}),
                                decision_state.get("gate", {}),
                            )
                            mode_rewards = {
                                mode: (args.teacher_reward_margin if mode == target_mode else 0.0)
                                for mode in available
                            }
                            row = {
                                "state_id": f"g{game_index:04d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "game_index": game_index,
                                "step": steps,
                                "teacher_gate_policy": args.teacher_gate_policy,
                                "target_mode": target_mode,
                                "rule_mode": target_mode,
                                "available_modes": available,
                                "features": features,
                                "mode_rewards": mode_rewards,
                                "teacher_action": action,
                                "teacher_reason": decision_state.get("reason", ""),
                                "teacher_alpha": decision_state.get("alpha"),
                                "teacher_value_gate": decision_state.get("value_gate"),
                                "teacher_q_base": decision_state.get("q_base"),
                                "teacher_delta_shape": decision_state.get("delta_shape"),
                                "legal_actions": legal_actions,
                                **diagnostics,
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            rows += 1
                            counts[target_mode] += 1
                            if rows == 1 or rows % 25 == 0:
                                print(
                                    f"[NeuralGateTeacher] rows={rows}/{args.target_states} "
                                    f"attempts={attempts} counts={dict(counts)} skipped={dict(skipped)}",
                                    flush=True,
                                )

                    actual_mode = target_mode
                    last_p0_state = {
                        "step": steps,
                        "mode": actual_mode,
                        "action": action,
                        "reason": decision_state.get("reason", ""),
                    }
                    if actual_mode == "deceive":
                        deceive_active_until = steps + 4
                elif pid in defenders:
                    action = defenders[pid].turn(
                        player,
                        game,
                        step=steps,
                        last_p0_state=last_p0_state,
                        in_deceive_window=deceive_active_until >= steps,
                    )
                else:
                    action = opponent_funcs[pid][0](player, game)

                discarded_tile = execute_action(game, pid, action, drawn_tile)
                if game.is_game_over:
                    break
                if discarded_tile is None:
                    if action == "g":
                        skip_draw = True
                    else:
                        game.next_player()
                        skip_draw = False
                    continue
                resolve_responses(game, discarded_tile, pid, teacher, defenders)
                if game.is_game_over:
                    break

    summary = {
        "rows": rows,
        "attempts": attempts,
        "target_counts": dict(counts),
        "skipped": dict(skipped),
        "rows_path": str(rows_path),
        "config": {
            "teacher_gate_policy": args.teacher_gate_policy,
            "opponent_style": args.opponent_style,
            "backend": args.backend,
            "sample_stride": args.sample_stride,
            "target_states": args.target_states,
            "max_games": args.max_games,
            "teacher_reward_margin": args.teacher_reward_margin,
            "quotas": quotas,
            "quota_mode": quota_mode,
        },
    }
    write_json(args.output_dir / "teacher_gate_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=1000)
    parser.add_argument("--target-exploit", type=int, default=0)
    parser.add_argument("--target-safe", type=int, default=0)
    parser.add_argument("--target-deceive", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026072301)
    parser.add_argument("--backend", default="heuristic_fallback")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--teacher-gate-policy", default="continuous_v2")
    parser.add_argument("--teacher-reward-margin", type=float, default=10.0)
    parser.add_argument("--opponent-style", choices=["greedy", "aggressive", "conservative", "random", "mixed", "responsive", "neural"], default="neural")
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth")
    parser.add_argument("--neural-opponent-device", default="cpu")
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--oracle-samples", type=int, default=30)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--ffr-hand-shanten", type=int, default=1)
    parser.add_argument("--defender-threat-model", choices=["mc", "discard_tell", "blend", "learned"], default="blend")
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth")
    parser.add_argument("--mask-oracle-samples", type=int, default=30)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=1)
    parser.add_argument("--mask-dir-ready-threshold", type=int, default=0)
    parser.add_argument("--mask-deceive-threat-ceiling", type=float, default=0.5)
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="off")
    parser.add_argument("--mask-deceive-style", choices=["safe", "threat"], default="threat")
    parser.add_argument("--mask-threat-allow-break-ready", action="store_true")
    parser.add_argument("--mask-threat-max-result-shanten", type=int, default=0)
    parser.add_argument("--mask-threat-max-shanten-regret", type=int, default=0)
    parser.add_argument("--mask-threat-min-ukeire-ratio", type=float, default=1.0)
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.4)
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12)
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03)
    parser.add_argument("--mask-threat-gate-mode", choices=["cross", "delta_only"], default="cross")
    parser.add_argument("--mask-threat-response-model", choices=["tell", "blend"], default="tell")
    parser.add_argument("--mask-threat-response-tell-weight", type=float, default=1.0)
    parser.add_argument("--mask-threat-tell-window", type=int, default=6)
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=3)
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true")
    parser.add_argument("--mask-threat-require-real-target", action="store_true")
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=0)
    parser.add_argument("--mask-threat-target-signal", choices=["oracle", "mc"], default="oracle")
    parser.add_argument("--mask-threat-target-prob-threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

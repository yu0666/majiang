"""Measure the upper bound of rule-constrained MASK action candidates.

For each sampled P0 decision, every candidate is evaluated from an isolated
copy of the same in-progress game. Candidate rollouts share random seeds, so
the action is the main source of paired return differences. The live game
continues with the deployed MASK action; Oracle choices never alter collection.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from environment_rollout import (
    capture_rollout_snapshot,
    clone_mask_agent,
    heuristic_continuation_policy,
    mask_continuation_policy,
    rule_mask_continuation_policy,
    rollout_candidate,
)
from experiment_trace import ensure_deterministic_hashing, write_json, write_jsonl
from llm_backends import build_llm_callable
from mask_candidates import build_mode_candidates
from mask_llm import MASKLLMAgent
from prompt_builder import build_mask_decision_prompt, get_legal_actions
from run_gate1_experiments import init_game, execute_action, resolve_responses


DEFAULT_OUTPUT = Path("Candidate_oracle_results")


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def median(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.median(values) if values else None


def make_mask_agent(seed: int, llm, args: argparse.Namespace) -> MASKLLMAgent:
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
        log_counterfactual=True,
        use_candidate_reranker=args.mask_candidate_reranker,
        use_candidate_scoring=args.mask_candidate_scoring,
        reranker_max_candidates=args.mask_reranker_max_candidates,
    )


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
    }


def quotas(args: argparse.Namespace) -> Dict[str, int]:
    return {
        "exploit": args.target_exploit,
        "safe": args.target_safe,
        "deceive": args.target_deceive,
    }


def quotas_met(counts: Counter[str], targets: Dict[str, int]) -> bool:
    return all(counts[mode] >= target for mode, target in targets.items())


def collection_modes(
    actual_mode: str,
    counts: Counter[str],
    targets: Dict[str, int],
    augment_modes: bool,
) -> List[str]:
    modes = [actual_mode]
    if augment_modes:
        modes.extend(
            mode
            for mode in ("exploit", "safe", "deceive")
            if mode != actual_mode and counts[mode] < targets.get(mode, 0)
        )
    return modes


def select_candidates(actions: List[str], baseline: str, limit: int) -> List[str]:
    unique = []
    for action in actions + [baseline]:
        if action and action not in unique:
            unique.append(action)
    if len(unique) <= limit:
        return unique
    selected = unique[:limit]
    if baseline not in selected:
        selected[-1] = baseline
    return selected


def evaluate_state(
    snapshot,
    candidates: List[str],
    baseline_action: str,
    rollout_seeds: List[int],
    continuation: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    policies = {
        "heuristic": heuristic_continuation_policy,
        "rule_mask": rule_mask_continuation_policy,
        "mask": mask_continuation_policy,
    }
    policy = policies[continuation]
    selection_indices = list(range(0, len(rollout_seeds), 2))
    evaluation_indices = list(range(1, len(rollout_seeds), 2)) or selection_indices
    action_rows = []
    for action in candidates:
        results = [
            rollout_candidate(
                snapshot,
                action,
                rollout_seed,
                continuation_policy=policy,
                use_agent_for_responses=continuation == "mask",
            )
            for rollout_seed in rollout_seeds
        ]
        returns = [result.continuation_return for result in results]
        selection_returns = [returns[index] for index in selection_indices]
        evaluation_returns = [returns[index] for index in evaluation_indices]
        action_rows.append(
            {
                "action": action,
                "mean_return": mean(returns),
                "median_return": median(returns),
                "return_std": statistics.stdev(returns) if len(returns) > 1 else 0.0,
                "selection_mean_return": mean(selection_returns),
                "evaluation_mean_return": mean(evaluation_returns),
                "hu_rate": mean(float(result.agent_hu) for result in results),
                "mean_hu_fan": mean(float(result.agent_hu_fan) for result in results),
                "dealin_rate": mean(float(result.agent_dealin) for result in results),
                "settled_rate": mean(float(result.settled) for result in results),
                "rollouts": [asdict(result) for result in results],
            }
        )

    oracle = max(action_rows, key=lambda row: float(row["selection_mean_return"]))
    baseline = next(row for row in action_rows if row["action"] == baseline_action)
    comparison = {
        "baseline_action": baseline_action,
        "baseline_mean_return": baseline["evaluation_mean_return"],
        "oracle_action": oracle["action"],
        "oracle_mean_return": oracle["evaluation_mean_return"],
        "oracle_advantage": (
            float(oracle["evaluation_mean_return"])
            - float(baseline["evaluation_mean_return"])
        ),
        "selection_oracle_advantage": (
            float(oracle["selection_mean_return"])
            - float(baseline["selection_mean_return"])
        ),
        "all_rollout_oracle_advantage": (
            float(oracle["mean_return"]) - float(baseline["mean_return"])
        ),
        "oracle_changed_action": oracle["action"] != baseline_action,
        "estimator": "even-index selection / odd-index held-out evaluation",
    }
    return action_rows, comparison


def summarize(rows: List[Dict[str, Any]], considered: Counter[str], skipped: Counter[str]) -> Dict[str, Any]:
    def summarize_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        gaps = [float(row["comparison"]["oracle_advantage"]) for row in group]
        gap_std = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
        ci_95 = None
        t_p_value = None
        if len(gaps) > 1:
            try:
                from scipy import stats as scipy_stats

                sem = gap_std / (len(gaps) ** 0.5)
                critical = float(scipy_stats.t.ppf(0.975, len(gaps) - 1))
                center = float(statistics.mean(gaps))
                ci_95 = [center - critical * sem, center + critical * sem]
                raw_p_value = float(scipy_stats.ttest_1samp(gaps, 0.0).pvalue)
                t_p_value = raw_p_value if math.isfinite(raw_p_value) else None
            except Exception:
                pass
        return {
            "states": len(group),
            "baseline_mean_return": mean(row["comparison"]["baseline_mean_return"] for row in group),
            "oracle_mean_return": mean(row["comparison"]["oracle_mean_return"] for row in group),
            "mean_oracle_advantage": mean(gaps),
            "median_oracle_advantage": median(gaps),
            "oracle_advantage_std": gap_std,
            "oracle_advantage_t_ci_95": ci_95,
            "oracle_advantage_t_p_value": t_p_value,
            "positive_advantage_rate": mean(float(gap > 1e-9) for gap in gaps),
            "negative_advantage_rate": mean(float(gap < -1e-9) for gap in gaps),
            "zero_advantage_rate": mean(float(abs(gap) <= 1e-9) for gap in gaps),
            "oracle_action_change_rate": mean(
                float(row["comparison"]["oracle_changed_action"]) for row in group
            ),
            "mean_candidate_count": mean(len(row["candidates"]) for row in group),
        }

    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    llm_called = [row for row in rows if row["decision_state"].get("decision_llm_raw") is not None]
    llm_parsed = [row for row in llm_called if row["decision_state"].get("decision_llm_parsed")]
    return {
        "evaluated": summarize_group(rows),
        "by_mode": {mode: summarize_group(group) for mode, group in sorted(by_mode.items())},
        "considered_states": dict(considered),
        "skipped_states": dict(skipped),
        "model_output_diagnostics": {
            "llm_called_states": len(llm_called),
            "llm_parsed_states": len(llm_parsed),
            "parse_success_rate_when_called": (
                len(llm_parsed) / len(llm_called) if llm_called else None
            ),
        },
        "interpretation": (
            "The action is selected on even-index rollouts and evaluated on held-out odd-index "
            "rollouts. The reported advantage estimates a finite-sample rollout selector, not an "
            "optimistic in-sample maximum. A positive gap means that selector generalized beyond "
            "the deployed action under the configured continuation policy."
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    targets = quotas(args)
    collected: Counter[str] = Counter()
    considered: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    rows: List[Dict[str, Any]] = []
    state_index = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states_path = args.output_dir / "candidate_oracle_states.jsonl"

    for game_index in range(args.max_games):
        if quotas_met(collected, targets):
            break
        game_seed = args.seed + game_index
        game, opponent_funcs, defenders = init_game(
            game_seed,
            args.opponent_style,
            f"CandidateOracle_{game_seed}",
            defender_config(args),
        )
        start_balance = float(game.players[0].balance)
        agent = make_mask_agent(game_seed, llm, args)
        skip_draw = True
        steps = 0
        last_p0_state: Dict[str, Any] = {}
        deceive_active_until = -1

        while not game.is_game_over and steps < args.max_steps:
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

            decision_state: Dict[str, Any] = {}
            if pid == 0:
                legal_actions = get_legal_actions(game, 0)
                action = agent.decide(game, legal_actions)
                decision_state = dict(agent.last_decision)
                actual_mode = str(decision_state.get("mode", "exploit"))

                for mode in collection_modes(
                    actual_mode,
                    collected,
                    targets,
                    args.augment_modes,
                ):
                    considered[mode] += 1
                    if collected[mode] >= targets.get(mode, 0):
                        continue
                    candidate_agent = clone_mask_agent(agent)
                    candidate_set = build_mode_candidates(
                        candidate_agent,
                        game,
                        legal_actions,
                        mode,
                        mode_already_selected=True,
                        beliefs=decision_state.get("beliefs", {}),
                    )
                    mode_baseline = (
                        action
                        if mode == actual_mode and action in candidate_set.actions
                        else (candidate_set.actions[0] if candidate_set.actions else action)
                    )
                    candidates = select_candidates(
                        candidate_set.actions,
                        mode_baseline,
                        args.max_candidates,
                    )
                    if len(candidates) >= 2:
                        snapshot = capture_rollout_snapshot(
                            game=game,
                            opponent_funcs=opponent_funcs,
                            defenders=defenders,
                            mask_agent=agent,
                            steps=steps,
                            max_steps=args.max_steps,
                            episode_start_balance=start_balance,
                            drawn_tile=drawn_tile,
                            last_p0_state=last_p0_state,
                            deceive_active_until=deceive_active_until,
                        )
                        rollout_seeds = [
                            args.rollout_seed + state_index * 1009 + offset
                            for offset in range(args.rollouts_per_action)
                        ]
                        action_rows, comparison = evaluate_state(
                            snapshot,
                            candidates,
                            mode_baseline,
                            rollout_seeds,
                            args.continuation,
                        )
                        beliefs = decision_state.get("beliefs", {})
                        gate = decision_state.get("gate", {})
                        rows.append(
                            {
                                "state_id": f"g{game_index:04d}_s{steps:03d}_{mode}",
                                "game_seed": game_seed,
                                "game_index": game_index,
                                "step": steps,
                                "mode": mode,
                                "actual_mode": actual_mode,
                                "mode_augmented": mode != actual_mode,
                                "backend": args.backend,
                                "continuation": args.continuation,
                                "legal_actions": legal_actions,
                                "candidates": candidates,
                                "candidate_metadata": candidate_set.metadata,
                                "decision_state": decision_state,
                                "prompt": build_mask_decision_prompt(
                                    game, 0, beliefs, gate, valid_actions=legal_actions,
                                    output_format="action",
                                ),
                                "rollout_seeds": rollout_seeds,
                                "action_evaluations": action_rows,
                                "comparison": comparison,
                            }
                        )
                        write_jsonl(states_path, rows)
                        collected[mode] += 1
                        state_index += 1
                        gap = comparison["oracle_advantage"]
                        print(
                            f"[Oracle] {collected}/{targets} state={state_index} "
                            f"mode={mode} candidates={len(candidates)} gap={gap:.3f}",
                            flush=True,
                        )
                    else:
                        skipped[f"{mode}:fewer_than_two_candidates"] += 1

                if actual_mode == "deceive":
                    deceive_active_until = steps + 4
                last_p0_state = {
                    "step": steps,
                    "mode": actual_mode,
                    "action": action,
                    "reason": decision_state.get("reason", ""),
                }
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

            response_info = resolve_responses(game, discarded_tile, pid, agent, defenders)
            if game.is_game_over:
                break
            if response_info["responded"]:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False

    summary = {
        "config": {
            "backend": args.backend,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "temperature": args.temperature,
            "opponent_style": args.opponent_style,
            "defender": defender_config(args),
            "quotas": targets,
            "rollouts_per_action": args.rollouts_per_action,
            "max_candidates": args.max_candidates,
            "continuation": args.continuation,
            "augment_modes": args.augment_modes,
            "oracle_estimator": "even-index selection / odd-index held-out evaluation",
            "seed": args.seed,
            "rollout_seed": args.rollout_seed,
        },
        "completed_quotas": dict(collected),
        "quota_met": quotas_met(collected, targets),
        "metrics": summarize(rows, considered, skipped),
    }
    write_jsonl(states_path, rows)
    write_json(args.output_dir / "candidate_oracle_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--rollout-seed", type=int, default=731000)
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--target-exploit", type=int, default=12)
    parser.add_argument("--target-safe", type=int, default=4)
    parser.add_argument("--target-deceive", type=int, default=12)
    parser.add_argument("--rollouts-per-action", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument(
        "--augment-modes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate unmet fixed-mode quotas at the same public state without changing live play.",
    )
    parser.add_argument(
        "--continuation",
        choices=["heuristic", "rule_mask", "mask"],
        default="rule_mask",
    )
    parser.add_argument("--opponent-style", default="responsive", choices=["responsive"])

    parser.add_argument("--threat-fold-threshold", type=float, default=0.4)
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
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="eligible")
    parser.add_argument("--mask-deceive-style", choices=["safe", "threat"], default="threat")
    parser.add_argument("--mask-threat-allow-break-ready", action="store_true")
    parser.add_argument("--mask-threat-max-result-shanten", type=int, default=0)
    parser.add_argument("--mask-threat-max-shanten-regret", type=int, default=0)
    parser.add_argument("--mask-threat-min-ukeire-ratio", type=float, default=1.0)
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12)
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03)
    parser.add_argument("--mask-threat-gate-mode", choices=["cross", "delta_only"], default="cross")
    parser.add_argument("--mask-threat-response-model", choices=["tell", "blend"], default="blend")
    parser.add_argument("--mask-threat-response-tell-weight", type=float, default=0.3)
    parser.add_argument("--mask-threat-tell-window", type=int, default=6)
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=2)
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true")
    parser.add_argument("--mask-threat-require-real-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=1)
    parser.add_argument("--mask-threat-target-signal", choices=["oracle", "mc"], default="mc")
    parser.add_argument("--mask-threat-target-prob-threshold", type=float, default=0.78)
    parser.add_argument("--mask-candidate-reranker", action="store_true")
    parser.add_argument("--mask-candidate-scoring", action="store_true")
    parser.add_argument("--mask-reranker-max-candidates", type=int, default=6)

    parser.add_argument("--backend", choices=["heuristic_fallback", "local_qwen"], default="heuristic_fallback")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main() -> None:
    ensure_deterministic_hashing()
    args = build_parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved candidate Oracle outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()

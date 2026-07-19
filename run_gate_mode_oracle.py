"""Collect paired counterfactual rewards for learned MASK mode selection."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from environment_rollout import (
    capture_rollout_snapshot,
    rollout_candidate,
    rule_mask_continuation_policy,
)
from experiment_trace import ensure_deterministic_hashing, write_json, write_jsonl
from llm_backends import build_llm_callable
from mask_llm import MASKLLMAgent
from prompt_builder import build_gate_decision_prompt, get_legal_actions
from risk_aware_reward import add_reward_arguments, reward_config, score_rollouts
from run_candidate_oracle import defender_config, make_mask_agent
from run_gate1_experiments import init_game, execute_action, resolve_responses


def mode_actions(
    agent: MASKLLMAgent,
    game,
    legal_actions: List[str],
    decision_state: Dict[str, Any],
) -> Dict[str, str]:
    exploit_action, _ = agent._exploit_action(game, legal_actions)
    actions = {
        "exploit": exploit_action,
        "safe": agent._safe_discard(game, legal_actions, fallback_action=exploit_action),
    }
    if decision_state.get("deceive_ready"):
        rng_state = agent.rng.getstate()
        try:
            deceive_action = agent._deceptive_discard(
                game,
                legal_actions,
                decision_state.get("beliefs", {}),
            )
        finally:
            agent.rng.setstate(rng_state)
        if (
            deceive_action in legal_actions
            and deceive_action.startswith("d ")
            and (not agent.threat_require_non_exploit or deceive_action != exploit_action)
        ):
            actions["deceive"] = deceive_action
    return actions


def evaluate_modes(snapshot, actions: Dict[str, str], rollout_seeds: List[int], args):
    by_action: Dict[str, List[Dict[str, Any]]] = {}
    for action in sorted(set(actions.values())):
        by_action[action] = [
            asdict(
                rollout_candidate(
                    snapshot,
                    action,
                    rollout_seed,
                    continuation_policy=rule_mask_continuation_policy,
                )
            )
            for rollout_seed in rollout_seeds
        ]
    mode_scores = {}
    mode_diagnostics = {}
    for mode, action in actions.items():
        scored = score_rollouts(by_action[action], args)
        mode_scores[mode] = float(scored["risk_adjusted_score"])
        mode_diagnostics[mode] = {
            "action": action,
            **{
                key: scored[key]
                for key in (
                    "mean_raw_return",
                    "mean_fan",
                    "lower_tail_cvar",
                    "catastrophic_loss_rate",
                    "risk_adjusted_score",
                )
            },
            "rollouts": by_action[action],
        }
    return mode_scores, mode_diagnostics


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    quotas = {
        "exploit": args.target_exploit,
        "safe": args.target_safe,
        "deceive": args.target_deceive,
    }
    quota_mode = any(v > 0 for v in quotas.values())
    counts: Counter = Counter()
    attempts = 0

    def quotas_satisfied() -> bool:
        if quota_mode:
            return all(counts[mode] >= quota for mode, quota in quotas.items())
        return len(rows) >= args.target_states

    rows = []
    for game_index in range(args.max_games):
        if quotas_satisfied():
            break
        game_seed = args.seed + game_index
        game, opponent_funcs, defenders = init_game(
            game_seed,
            args.opponent_style,
            f"GateModeOracle_{game_seed}",
            defender_config(args),
        )
        start_balance = float(game.players[0].balance)
        agent = make_mask_agent(game_seed, llm, args)
        skip_draw = True
        steps = 0
        last_p0_state: Dict[str, Any] = {}
        deceive_active_until = -1
        while not game.is_game_over and steps < args.max_steps and not quotas_satisfied():
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
                if (
                    "h" not in legal_actions
                    and any(item.startswith("d ") for item in legal_actions)
                    and steps % args.sample_stride == 0
                ):
                    actions = mode_actions(agent, game, legal_actions, decision_state)
                    snapshot = capture_rollout_snapshot(
                        game,
                        opponent_funcs,
                        defenders,
                        agent,
                        steps,
                        args.max_steps,
                        start_balance,
                        drawn_tile,
                        last_p0_state,
                        deceive_active_until,
                    )
                    rollout_seeds = [
                        args.rollout_seed + len(rows) * 1009 + offset
                        for offset in range(args.rollouts_per_mode)
                    ]
                    mode_scores, diagnostics = evaluate_modes(
                        snapshot,
                        actions,
                        rollout_seeds,
                        args,
                    )
                    attempts += 1
                    best_score = max(mode_scores.values())
                    best_modes = [mode for mode, score in mode_scores.items() if score == best_score]
                    rule_mode = str(decision_state.get("mode", "exploit"))
                    target_mode = rule_mode if rule_mode in best_modes else sorted(best_modes)[0]
                    quota_full = quota_mode and counts[target_mode] >= quotas[target_mode]
                    if not quota_full:
                        rows.append(
                            {
                                "state_id": f"g{game_index:04d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "step": steps,
                                "prompt": build_gate_decision_prompt(
                                    game,
                                    0,
                                    decision_state.get("z_state", {}),
                                    decision_state.get("beliefs", {}),
                                    decision_state.get("gate", {}),
                                    list(actions),
                                ),
                                "available_modes": list(actions),
                                "mode_actions": actions,
                                "mode_rewards": mode_scores,
                                "mode_diagnostics": diagnostics,
                                "target_mode": target_mode,
                                "rule_mode": rule_mode,
                                "rollout_seeds": rollout_seeds,
                            }
                        )
                        counts[target_mode] += 1
                        progress = (
                            f"{dict(counts)}/{quotas}" if quota_mode
                            else f"{len(rows)}/{args.target_states}"
                        )
                        print(
                            f"[Gate oracle] states={progress} attempts={attempts} "
                            f"target={target_mode} modes={list(actions)}",
                            flush=True,
                        )
                    elif attempts % 10 == 0:
                        # Quota already full for this attempt's mode; still surface
                        # liveness so a long deceive-only search doesn't look stuck.
                        print(
                            f"[Gate oracle] states={dict(counts)}/{quotas} attempts={attempts} "
                            f"(skipped, {target_mode} quota full)",
                            flush=True,
                        )
                actual_mode = str(decision_state.get("mode", "exploit"))
                if actual_mode == "deceive":
                    deceive_active_until = steps + 4
                last_p0_state = {"step": steps, "mode": actual_mode, "action": action}
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
            response = resolve_responses(game, discarded_tile, pid, agent, defenders)
            if game.is_game_over:
                break
            if response["responded"]:
                skip_draw = True
            else:
                game.next_player()
                skip_draw = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    states_path = args.output_dir / "gate_mode_oracle.jsonl"
    write_jsonl(states_path, rows)
    summary = {
        "states": len(rows),
        "target_states": args.target_states,
        "quota_mode": quota_mode,
        "quotas": quotas,
        "target_counts": {
            mode: sum(row["target_mode"] == mode for row in rows)
            for mode in ("exploit", "safe", "deceive")
        },
        "reward": reward_config(args),
        "config": {
            "seed": args.seed,
            "rollout_seed": args.rollout_seed,
            "rollouts_per_mode": args.rollouts_per_mode,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
        },
        "output": str(states_path),
    }
    write_json(args.output_dir / "gate_mode_oracle_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=400,
                        help="Legacy flat cap; ignored once --target-exploit/--target-safe/"
                             "--target-deceive are all set above 0 (quota mode).")
    parser.add_argument("--target-exploit", type=int, default=0,
                        help="Per-mode quota. 0 disables quota mode (falls back to --target-states).")
    parser.add_argument("--target-safe", type=int, default=0)
    parser.add_argument("--target-deceive", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026071601)
    parser.add_argument("--rollout-seed", type=int, default=816000)
    parser.add_argument("--rollouts-per-mode", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--opponent-style", default="responsive", choices=["responsive"])
    parser.add_argument("--backend", choices=["heuristic_fallback", "local_qwen"], default="local_qwen")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--oracle-samples", type=int, default=30)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--ffr-hand-shanten", type=int, default=1)
    parser.add_argument("--defender-threat-model", default="blend")
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth")
    parser.add_argument("--mask-oracle-samples", type=int, default=30)
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=1)
    parser.add_argument("--mask-dir-ready-threshold", type=int, default=0)
    parser.add_argument("--mask-deceive-threat-ceiling", type=float, default=0.5)
    parser.add_argument("--mask-forced-deceive", default="off")
    parser.add_argument("--mask-deceive-style", default="threat")
    parser.add_argument("--mask-threat-allow-break-ready", action="store_true")
    parser.add_argument("--mask-threat-max-result-shanten", type=int, default=0)
    parser.add_argument("--mask-threat-max-shanten-regret", type=int, default=0)
    parser.add_argument("--mask-threat-min-ukeire-ratio", type=float, default=1.0)
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12)
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03)
    parser.add_argument("--mask-threat-gate-mode", default="cross")
    parser.add_argument("--mask-threat-response-model", default="blend")
    parser.add_argument("--mask-threat-response-tell-weight", type=float, default=0.3)
    parser.add_argument("--mask-threat-tell-window", type=int, default=6)
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=2)
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true")
    parser.add_argument("--mask-threat-require-real-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=1)
    parser.add_argument("--mask-threat-target-signal", default="mc")
    parser.add_argument("--mask-threat-target-prob-threshold", type=float, default=0.78)
    parser.add_argument("--mask-candidate-reranker", action="store_true")
    parser.add_argument("--mask-candidate-scoring", action="store_true")
    parser.add_argument("--mask-reranker-max-candidates", type=int, default=6)
    add_reward_arguments(parser)
    return parser


if __name__ == "__main__":
    ensure_deterministic_hashing()
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))

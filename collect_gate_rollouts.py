"""Collect paired counterfactual rewards for exploit/safe/deceive gate modes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from environment_rollout import (
    capture_rollout_snapshot,
    rollout_candidate,
    rule_mask_continuation_policy,
)
from experiment_trace import ensure_deterministic_hashing, write_json
from llm_backends import build_llm_callable
from mask_llm import MASKLLMAgent
from prompt_builder import build_gate_decision_prompt, get_legal_actions
from run_gate1_experiments import init_game, execute_action, resolve_responses


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


def build_agent(seed: int, llm) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=llm,
        belief_llm=llm,
        mc_seed=seed * 13 + 1,
        mc_oracle_samples=30,
        mc_beta=2.0,
        mc_danger_threshold=1,
        dir_ready_threshold=0,
        forced_deceive="off",
        deceive_style="threat",
        threat_max_result_shanten=0,
        threat_max_shanten_regret=0,
        threat_min_ukeire_ratio=1.0,
        threat_gate_threshold=0.7,
        threat_gate_margin=0.12,
        threat_min_delta=0.03,
        threat_gate_mode="cross",
        threat_response_model="blend",
        threat_response_tell_weight=0.3,
        threat_tell_window=6,
        threat_max_start_shanten=2,
        threat_require_non_exploit=True,
        threat_require_real_target=True,
        threat_target_max_shanten=1,
        threat_target_signal="mc",
        threat_target_prob_threshold=0.78,
    )


def mode_actions(
    agent: MASKLLMAgent,
    game,
    legal_actions: list[str],
    beliefs: Dict[str, Any],
) -> Dict[str, str]:
    exploit, _ = agent._exploit_action(game, legal_actions)
    actions = {"exploit": exploit}
    safe = agent._safe_discard(game, legal_actions, fallback_action=exploit)
    if safe in legal_actions:
        actions["safe"] = safe
    deceive = agent._deceptive_discard(game, legal_actions, beliefs, relaxed_gate=True)
    if deceive in legal_actions and deceive.startswith("d ") and deceive != exploit:
        actions["deceive"] = deceive
    return actions


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--seed", type=int, default=2026071601)
    parser.add_argument("--rollout-seed", type=int, default=916000)
    parser.add_argument("--max-states", type=int, default=400)
    parser.add_argument("--max-games", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--rollouts-per-mode", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--oracle-samples", type=int, default=30)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--ffr-hand-shanten", type=int, default=1)
    parser.add_argument(
        "--defender-threat-model",
        choices=["mc", "discard_tell", "blend", "learned"],
        default="blend",
    )
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument(
        "--defender-learned-model-path",
        default="Defender_danger_model/danger_model.pth",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend="local_qwen",
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "gate_rollout_states.jsonl"
    counts: Counter[str] = Counter()
    collected = 0
    with output.open("w", encoding="utf-8") as handle:
        for game_index in range(args.max_games):
            if collected >= args.max_states:
                break
            game_seed = args.seed + game_index
            game, opponent_funcs, defenders = init_game(
                game_seed,
                "responsive",
                f"GateOracle_{game_seed}",
                defender_config(args),
            )
            start_balance = float(game.players[0].balance)
            agent = build_agent(game_seed, llm)
            skip_draw = True
            steps = 0
            last_p0_state: Dict[str, Any] = {}
            deceive_active_until = -1
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
                        collected < args.max_states
                        and steps % args.sample_every == 0
                        and "h" not in legal_actions
                        and any(item.startswith("d ") for item in legal_actions)
                    ):
                        actions = mode_actions(
                            agent,
                            game,
                            legal_actions,
                            decision_state.get("beliefs", {}),
                        )
                        if len(set(actions.values())) >= 2:
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
                                args.rollout_seed + collected * 1009 + offset
                                for offset in range(args.rollouts_per_mode)
                            ]
                            evaluations = []
                            for mode, mode_action in actions.items():
                                results = [
                                    rollout_candidate(
                                        snapshot,
                                        mode_action,
                                        rollout_seed,
                                        continuation_policy=rule_mask_continuation_policy,
                                        initial_mode=mode,
                                    )
                                    for rollout_seed in rollout_seeds
                                ]
                                evaluations.append(
                                    {
                                        "mode": mode,
                                        "action": mode_action,
                                        "rollouts": [asdict(result) for result in results],
                                    }
                                )
                                counts[mode] += 1
                            row = {
                                "state_id": f"g{game_index:04d}_s{steps:03d}",
                                "game_seed": game_seed,
                                "game_index": game_index,
                                "step": steps,
                                "rule_mode": decision_state.get("mode", "exploit"),
                                "available_modes": list(actions),
                                "mode_actions": actions,
                                "prompt": build_gate_decision_prompt(
                                    game,
                                    0,
                                    decision_state.get("z_state", {}),
                                    decision_state.get("beliefs", {}),
                                    decision_state.get("gate", {}),
                                    list(actions),
                                ),
                                "rollout_seeds": rollout_seeds,
                                "mode_evaluations": evaluations,
                            }
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            handle.flush()
                            collected += 1
                            if collected == 1 or collected % 10 == 0:
                                print(f"[Gate oracle] state={collected}/{args.max_states}", flush=True)

                    actual_mode = str(decision_state.get("mode", "exploit"))
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

                discarded = execute_action(game, pid, action, drawn_tile)
                if game.is_game_over:
                    break
                if discarded is None:
                    if action == "g":
                        skip_draw = True
                    else:
                        game.next_player()
                        skip_draw = False
                    continue
                response = resolve_responses(game, discarded, pid, agent, defenders)
                if game.is_game_over:
                    break
                if response["responded"]:
                    skip_draw = True
                else:
                    game.next_player()
                    skip_draw = False

    summary = {
        "states": collected,
        "mode_evaluations": dict(counts),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "rollouts_per_mode": args.rollouts_per_mode,
        "defender_threat_model": args.defender_threat_model,
        "defender_learned_model_path": args.defender_learned_model_path,
        "output": str(output),
    }
    write_json(args.output_dir / "gate_rollout_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
Generate MASK-SFT data from the successful heuristic L2 teacher.

The teacher is:
  L2 = min-shanten exploit + public z_j(t) + MASK belief/risk/deceive gate

It plays as P0 against the same responsive/blend opponents used in Gate1.  Each
P0 turn is saved as chat-template compatible SFT data:

  system: role instruction
  user:   build_mask_decision_prompt(...)
  assistant: {"mode": "...", "action": "...", "reason": "..."}

Extra metadata is kept small in the SFT file and detailed traces are written
separately for filtering/audit.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from experiment_trace import ensure_deterministic_hashing, summarize_games, write_json
from mask_candidates import build_mode_candidates
from mask_llm import MASKLLMAgent
from prompt_builder import build_mask_decision_prompt, get_legal_actions
from run_gate1_experiments import (
    execute_action,
    init_game,
    make_step_trace,
    resolve_responses,
)


DEFAULT_SEEDS = [20260627, 20261627, 20262627]
SYSTEM_PROMPT = (
    "你是四川麻将 MASK 决策智能体。你需要结合公开信息、对手信念估计 "
    "B_phi、风险门控和合法动作空间，在 exploit/safe/deceive 三种模式中"
    "选择可执行动作。必须严格遵守合法动作列表，并只输出 JSON。"
)
RERANKER_SYSTEM_PROMPT = (
    "你是四川麻将候选动作重排器。外层规则已经固定 exploit/safe/deceive 模式，"
    "你不能改变模式，只能从规则约束候选动作中选择预期收益最高的一项。"
)


def dump_jsonl_row(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def build_defender_cfg(args: argparse.Namespace) -> Dict[str, Any]:
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


def build_mask_agent(seed: int, args: argparse.Namespace) -> MASKLLMAgent:
    return MASKLLMAgent(
        player_id=0,
        decision_llm=None,
        belief_llm=None,
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
        log_counterfactual=args.mask_log_counterfactual,
    )


def assistant_content(action: str, decision_state: Dict[str, Any], output_format: str) -> str:
    if output_format == "action":
        return action
    payload = {
        "mode": decision_state.get("mode", "exploit"),
        "action": action,
        "reason": decision_state.get("reason", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def make_sft_example(
    mask_agent: MASKLLMAgent,
    game,
    seed: int,
    step: int,
    action: str,
    legal_actions: List[str],
    decision_state: Dict[str, Any],
    output_format: str,
    metadata_level: str,
    reranker_max_candidates: int,
    reranker_completion_format: str,
) -> Optional[Dict[str, Any]]:
    prompt_actions = list(legal_actions)
    candidate_metadata: Dict[str, Any] = {}
    if output_format == "reranker":
        mode = str(decision_state.get("mode", "exploit"))
        rng_state = mask_agent.rng.getstate()
        try:
            candidate_set = build_mode_candidates(
                mask_agent,
                game,
                legal_actions,
                mode,
                mode_already_selected=True,
                beliefs=decision_state.get("beliefs", {}),
            )
        finally:
            mask_agent.rng.setstate(rng_state)
        prompt_actions = []
        for candidate in candidate_set.actions:
            if candidate in legal_actions and candidate not in prompt_actions:
                prompt_actions.append(candidate)
        if len(prompt_actions) > reranker_max_candidates:
            prompt_actions = prompt_actions[:reranker_max_candidates]
        if len(prompt_actions) < 2 or action not in prompt_actions:
            return None
        candidate_metadata = candidate_set.metadata

    prompt = build_mask_decision_prompt(
        game,
        player_id=0,
        belief_state=decision_state.get("beliefs", {}),
        gate_state=decision_state.get("gate", {}),
        valid_actions=prompt_actions,
        output_format=(
            reranker_completion_format if output_format == "reranker" else "json"
        ),
    )
    if output_format == "reranker":
        prompt = f"""
{prompt}

【规则约束候选重排】
外层规则已经固定当前模式为: {decision_state.get('mode', 'exploit')}
候选动作: {", ".join(prompt_actions)}

不要改变模式，只在候选动作中选择预期收益最高的一项。
{('只输出一个候选动作本身，例如: d 3万。不要输出解释或 JSON。' if reranker_completion_format == 'action' else '严格输出 JSON: {"action": "候选动作", "reason": "一句话依据"}')}
""".strip()
        assistant = (
            action
            if reranker_completion_format == "action"
            else json.dumps(
                {"action": action, "reason": "L2 rule-constrained teacher"},
                ensure_ascii=False,
            )
        )
    else:
        assistant = assistant_content(action, decision_state, output_format)
    row: Dict[str, Any] = {
        "messages": [
            {
                "role": "system",
                "content": RERANKER_SYSTEM_PROMPT if output_format == "reranker" else SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ]
    }
    if metadata_level != "none":
        meta = {
            "game_id": game.game_id,
            "seed": seed,
            "step": step,
            "teacher": "heuristic_l2_mask",
            "method": "llm_mask",
            "backend": "heuristic_fallback",
            "action": action,
            "legal_actions": prompt_actions,
            "all_legal_actions": legal_actions,
            "mode": decision_state.get("mode", "exploit"),
            "reason": decision_state.get("reason", ""),
            "own_shanten": decision_state.get("own_shanten"),
            "deceive_ready": decision_state.get("deceive_ready"),
            "ffr_ready": decision_state.get("ffr_ready"),
            "counterfactual_exploit_action": decision_state.get("counterfactual_exploit_action"),
            "disguise_equals_exploit": decision_state.get("disguise_equals_exploit"),
            "risk_budget": decision_state.get("gate", {}).get("risk_budget"),
            "candidate_source": candidate_metadata.get("candidate_source"),
        }
        if metadata_level == "full":
            meta.update(
                {
                    "z_state": decision_state.get("z_state", {}),
                    "beliefs": decision_state.get("beliefs", {}),
                    "gate": decision_state.get("gate", {}),
                    "deceive_signal": decision_state.get("deceive_signal", {}),
                }
            )
        row["meta"] = meta
    return row


def update_example_outcome(example: Dict[str, Any], game_row: Dict[str, Any]) -> None:
    meta = example.get("meta")
    if not isinstance(meta, dict):
        return
    meta["agent_net"] = game_row["agent_net"]
    meta["agent_hu"] = game_row["agent_hu"]
    meta["agent_dealin"] = game_row["agent_dealin"]
    meta["deceive_windows"] = game_row["deceive_windows"]
    meta["induced_dealin"] = game_row["induced_dealin"]
    meta["false_folds"] = game_row["false_folds"]
    meta["false_fold_opportunities"] = game_row["false_fold_opportunities"]


def keep_example(example: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, str]:
    meta = example.get("meta", {})
    if args.keep_modes:
        mode = str(meta.get("mode", ""))
        if mode not in set(args.keep_modes):
            return False, "mode_not_selected"
    if float(meta.get("agent_net", 0.0)) < args.min_game_net:
        return False, "low_game_net"
    if args.drop_dealin_games and meta.get("agent_dealin"):
        return False, "dealin_game"
    if args.drop_deceive_overlap and meta.get("mode") == "deceive" and meta.get("disguise_equals_exploit") is True:
        return False, "deceive_overlap"
    if args.max_own_shanten is not None:
        own_shanten = meta.get("own_shanten")
        if own_shanten is not None and int(own_shanten) > args.max_own_shanten:
            return False, "own_shanten_too_high"
    return True, "kept"


def reranker_training_specs(
    mask_agent: MASKLLMAgent,
    game,
    legal_actions: List[str],
    action: str,
    decision_state: Dict[str, Any],
    augment_modes: Optional[List[str]],
) -> List[Tuple[str, Dict[str, Any]]]:
    if not augment_modes:
        return [(action, decision_state)]

    specs: List[Tuple[str, Dict[str, Any]]] = []
    for mode in augment_modes:
        rng_state = mask_agent.rng.getstate()
        try:
            candidate_set = build_mode_candidates(
                mask_agent,
                game,
                legal_actions,
                mode,
                mode_already_selected=True,
                beliefs=decision_state.get("beliefs", {}),
            )
        finally:
            mask_agent.rng.setstate(rng_state)
        if not candidate_set.actions:
            continue
        target_action = candidate_set.actions[0]
        if target_action not in legal_actions:
            continue
        synthetic_state = dict(decision_state)
        synthetic_state["mode"] = mode
        synthetic_state["reason"] = f"counterfactual {mode} candidate teacher"
        specs.append((target_action, synthetic_state))
    return specs


def play_teacher_game(
    seed: int,
    game_index: int,
    args: argparse.Namespace,
    defender_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    game, opponent_funcs, defenders = init_game(
        seed=seed,
        opponent_style=args.opponent_style,
        game_id=f"MASKSFT_L2_{seed}_{game_index}",
        defender_cfg=defender_cfg,
    )
    start_balances = [p.balance for p in game.players]
    mask_agent = build_mask_agent(seed, args)

    samples: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    decision_times: List[float] = []
    mode_counts: Counter[str] = Counter()
    skip_draw = True
    steps = 0
    deceive_active_until = -1
    deceive_windows = 0
    induced_dealin = 0
    response_false_fold_opportunities = 0
    response_false_folds = 0
    last_p0_state: Dict[str, Any] = {}

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

        decision_state: Dict[str, Any] = {}
        legal_actions: List[str] = []
        decision_ms = 0.0
        current_samples: List[Dict[str, Any]] = []

        if pid == 0:
            legal_actions = get_legal_actions(game, 0)
            snap_seeds = [seed * 100003 + steps * 31 + k for k in range(max(1, args.snapshot_crn_seeds))]
            threat_before = {
                f"P{rid}": defenders[rid].threat_crn(game, snap_seeds, args.snapshot_oracle_samples)
                for rid in defenders
            }

            t0 = time.perf_counter()
            action = mask_agent.decide(game, legal_actions)
            decision_ms = (time.perf_counter() - t0) * 1000.0
            decision_state = dict(mask_agent.last_decision)
            decision_times.append(decision_ms)
            mode_counts[str(decision_state.get("mode", "exploit"))] += 1

            specs = reranker_training_specs(
                mask_agent,
                game,
                legal_actions,
                action,
                decision_state,
                args.reranker_augment_modes if args.assistant_format == "reranker" else None,
            )
            for sample_action, sample_state in specs:
                sample = make_sft_example(
                    mask_agent=mask_agent,
                    game=game,
                    seed=seed,
                    step=steps,
                    action=sample_action,
                    legal_actions=legal_actions,
                    decision_state=sample_state,
                    output_format=args.assistant_format,
                    metadata_level=args.metadata,
                    reranker_max_candidates=args.reranker_max_candidates,
                    reranker_completion_format=args.reranker_completion_format,
                )
                if sample is not None:
                    samples.append(sample)
                    current_samples.append(sample)
            step_rows.append(
                make_step_trace(
                    game=game,
                    method="llm_mask",
                    seed=seed,
                    step=steps,
                    pid=pid,
                    action=action,
                    legal_actions=legal_actions,
                    decision_ms=decision_ms,
                    decision_state=decision_state,
                    backend="heuristic_fallback",
                )
            )
            if decision_state.get("mode") == "deceive":
                deceive_active_until = steps + args.deceive_window_steps
                deceive_windows += 1
        else:
            if pid in defenders:
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

        if pid == 0:
            threat_after = {
                f"P{rid}": defenders[rid].threat_crn(game, snap_seeds, args.snapshot_oracle_samples)
                for rid in defenders
            }
            threat_delta = {
                key: round(threat_after.get(key, 0.0) - threat_before.get(key, 0.0), 4)
                for key in set(threat_before) | set(threat_after)
            }
            last_p0_state = {
                "step": steps,
                "mode": decision_state.get("mode", "none"),
                "action": action,
                "reason": decision_state.get("reason", ""),
                "threat_before": {k: round(v, 4) for k, v in threat_before.items()},
                "threat_after": {k: round(v, 4) for k, v in threat_after.items()},
                "threat_delta": threat_delta,
            }
            if step_rows and step_rows[-1]["actor"] == 0:
                step_rows[-1]["threat_before"] = last_p0_state["threat_before"]
                step_rows[-1]["threat_after"] = last_p0_state["threat_after"]
                step_rows[-1]["threat_delta"] = last_p0_state["threat_delta"]
            for current_sample in current_samples:
                if "meta" in current_sample:
                    current_sample["meta"]["threat_delta"] = threat_delta

        if discarded_tile is None:
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        response_info = resolve_responses(game, discarded_tile, pid, mask_agent, defenders)
        response_false_fold_opportunities += response_info["response_false_fold_opportunities"]
        response_false_folds += response_info["response_false_folds"]

        if response_info["agent_dealt_in"]:
            for row in reversed(step_rows):
                if row["actor"] == 0:
                    row["dealin"] = True
                    break

        if response_info["agent_won_by_discard"] and deceive_active_until >= steps:
            induced_dealin += 1
            for row in reversed(step_rows):
                if row.get("mode") == "deceive":
                    row["induced_dealin"] = True
                    break

        if game.is_game_over:
            break
        if response_info["responded"]:
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

    turn_ff_opportunities = sum(d.ff_opportunities for d in defenders.values())
    turn_ff_false = sum(d.ff_false for d in defenders.values())
    response_declines = sum(d.response_declines for d in defenders.values())
    ffr_events = [event for defender in defenders.values() for event in defender.ffr_events]
    false_fold_events = [event for defender in defenders.values() for event in defender.false_fold_events]
    for event in ffr_events:
        event["game_id"] = game.game_id
        event["seed"] = seed
        event["method"] = "llm_mask"
        event["backend"] = "heuristic_fallback"
    false_folds_after_deceive = sum(1 for event in false_fold_events if event.get("last_p0_mode") == "deceive")
    false_folds_after_exploit = sum(1 for event in false_fold_events if event.get("last_p0_mode") == "exploit")
    false_folds_in_deceive_window = sum(1 for event in false_fold_events if event.get("in_deceive_window"))

    game_row = {
        "seed": seed,
        "method": "llm_mask",
        "backend": "heuristic_fallback",
        "opponent_style": args.opponent_style,
        "steps": steps,
        "agent_hu": bool(agent_player.is_hu),
        "agent_net": net[0],
        "agent_dealin": bool(agent_dealin),
        "winners": list(game.winners),
        "decision_ms": decision_times,
        "mode_counts": dict(mode_counts),
        "deceive_windows": deceive_windows,
        "induced_dealin": induced_dealin,
        "false_fold_opportunities": turn_ff_opportunities,
        "false_folds": turn_ff_false,
        "false_folds_after_deceive": false_folds_after_deceive,
        "false_folds_after_exploit": false_folds_after_exploit,
        "false_folds_in_deceive_window": false_folds_in_deceive_window,
        "ffr_events": ffr_events,
        "false_fold_events": false_fold_events,
        "opponent_response_declines": response_declines,
        "response_false_fold_opportunities": response_false_fold_opportunities,
        "response_false_folds": response_false_folds,
        "history_len": len(game.history),
        "raw_sft_examples": len(samples),
    }
    for sample in samples:
        update_example_outcome(sample, game_row)
    return game_row, samples, step_rows


def parse_seeds(values: Optional[List[int]]) -> List[int]:
    return list(values) if values else list(DEFAULT_SEEDS)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_file
    if not output_file.is_absolute():
        output_file = output_dir / output_file
    games_path = output_dir / "mask_sft_teacher_games.jsonl"
    steps_path = output_dir / "mask_sft_teacher_steps.jsonl"
    summary_path = output_dir / "mask_sft_teacher_summary.json"

    seeds = parse_seeds(args.seeds)
    defender_cfg = build_defender_cfg(args)
    all_games: List[Dict[str, Any]] = []
    kept = 0
    filtered = 0
    mode_counts: Counter[str] = Counter()
    kept_mode_counts: Counter[str] = Counter()
    filter_reasons: Counter[str] = Counter()

    with output_file.open("w", encoding="utf-8") as sft_f, \
            games_path.open("w", encoding="utf-8") as games_f, \
            steps_path.open("w", encoding="utf-8") as steps_f:
        total_games = len(seeds) * args.games_per_seed
        pbar = tqdm(total=total_games, desc="MASK-SFT teacher games")
        for seed_base in seeds:
            for game_index in range(args.games_per_seed):
                seed = seed_base + game_index
                game_row, examples, step_rows = play_teacher_game(seed, game_index, args, defender_cfg)
                all_games.append(game_row)
                dump_jsonl_row(games_f, game_row)
                for step_row in step_rows:
                    dump_jsonl_row(steps_f, step_row)

                for example in examples:
                    meta = example.get("meta", {})
                    mode = str(meta.get("mode", "unknown"))
                    mode_counts[mode] += 1
                    ok, reason = keep_example(example, args)
                    filter_reasons[reason] += 1
                    if ok:
                        kept += 1
                        kept_mode_counts[mode] += 1
                        dump_jsonl_row(sft_f, example)
                    else:
                        filtered += 1
                pbar.update(1)
        pbar.close()

    summary = {
        "task": "MASK-SFT data generation",
        "teacher": "heuristic_l2_mask",
        "backend": "heuristic_fallback",
        "seeds": seeds,
        "games_per_seed": args.games_per_seed,
        "total_games": len(all_games),
        "output_file": str(output_file),
        "games_trace": str(games_path),
        "steps_trace": str(steps_path),
        "assistant_format": args.assistant_format,
        "reranker_completion_format": args.reranker_completion_format,
        "reranker_augment_modes": args.reranker_augment_modes,
        "metadata": args.metadata,
        "raw_examples": kept + filtered,
        "kept_examples": kept,
        "filtered_examples": filtered,
        "raw_mode_counts": dict(mode_counts),
        "kept_mode_counts": dict(kept_mode_counts),
        "filter_reasons": dict(filter_reasons),
        "game_metrics": summarize_games(all_games),
        "defender_cfg": defender_cfg,
        "mask_teacher_cfg": {
            "mask_oracle_samples": args.mask_oracle_samples,
            "mask_oracle_beta": args.mask_oracle_beta,
            "mask_danger_threshold": args.mask_danger_threshold,
            "mask_dir_ready_threshold": args.mask_dir_ready_threshold,
            "mask_forced_deceive": args.mask_forced_deceive,
            "mask_deceive_style": args.mask_deceive_style,
            "mask_threat_gate_mode": args.mask_threat_gate_mode,
            "mask_threat_max_shanten_regret": args.mask_threat_max_shanten_regret,
            "mask_threat_min_ukeire_ratio": args.mask_threat_min_ukeire_ratio,
            "mask_threat_response_model": args.mask_threat_response_model,
            "mask_threat_response_tell_weight": args.mask_threat_response_tell_weight,
            "mask_threat_min_delta": args.mask_threat_min_delta,
            "mask_threat_require_real_target": args.mask_threat_require_real_target,
            "mask_threat_target_signal": args.mask_threat_target_signal,
            "mask_threat_target_prob_threshold": args.mask_threat_target_prob_threshold,
            "mask_log_counterfactual": args.mask_log_counterfactual,
        },
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("MASK_SFT_teacher_l2_mc"))
    parser.add_argument("--output-file", type=Path, default=Path("mask_sft_l2_mc.jsonl"))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--games-per-seed", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--opponent-style", default="responsive", choices=["greedy", "aggressive", "conservative", "random", "mixed", "responsive"])

    # Same responsive/blend opponent setting used in the successful Gate1 H2 run.
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--oracle-samples", type=int, default=30)
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1)
    parser.add_argument("--ffr-hand-shanten", type=int, default=1)
    parser.add_argument("--defender-threat-model", choices=["mc", "discard_tell", "blend", "learned"], default="blend")
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth")

    # Teacher L2-MC defaults: the deployable public-info target filter.
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
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12)
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03)
    parser.add_argument("--mask-threat-gate-mode", choices=["cross", "delta_only"], default="cross")
    parser.add_argument("--mask-threat-response-model", choices=["mc", "tell", "blend"], default="blend")
    parser.add_argument("--mask-threat-response-tell-weight", type=float, default=0.3)
    parser.add_argument("--mask-threat-tell-window", type=int, default=6)
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=2)
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true")
    parser.add_argument("--mask-threat-require-real-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=1)
    parser.add_argument("--mask-threat-target-signal", choices=["oracle", "mc"], default="mc")
    parser.add_argument("--mask-threat-target-prob-threshold", type=float, default=0.78)
    parser.add_argument("--mask-log-counterfactual", action=argparse.BooleanOptionalAction, default=True)

    # Measurement-only labels for later filtering/analysis.
    parser.add_argument("--snapshot-oracle-samples", type=int, default=60)
    parser.add_argument("--snapshot-crn-seeds", type=int, default=1)
    parser.add_argument("--deceive-window-steps", type=int, default=4)

    # SFT row filtering.  Defaults keep broad coverage and only remove no-op deceive.
    parser.add_argument("--assistant-format", choices=["json", "action", "reranker"], default="json")
    parser.add_argument("--reranker-max-candidates", type=int, default=6)
    parser.add_argument(
        "--reranker-completion-format",
        choices=["json", "action"],
        default="action",
        help="Use action-only targets to match conditional-logprob reranking at evaluation time.",
    )
    parser.add_argument(
        "--reranker-augment-modes",
        nargs="+",
        choices=["exploit", "safe", "deceive"],
        default=None,
        help="Generate fixed-mode counterfactual reranker examples without changing live teacher play.",
    )
    parser.add_argument("--metadata", choices=["none", "minimal", "full"], default="minimal")
    parser.add_argument("--keep-modes", nargs="+", choices=["exploit", "safe", "deceive"], default=None)
    parser.add_argument("--min-game-net", type=float, default=-1_000_000.0)
    parser.add_argument("--drop-dealin-games", action="store_true")
    parser.add_argument("--drop-deceive-overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-own-shanten", type=int, default=None)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\nSaved MASK-SFT data: {summary['output_file']}")


if __name__ == "__main__":
    main()

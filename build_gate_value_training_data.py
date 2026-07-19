"""Build value-based learned-gate data from counterfactual mode rollouts.

This route treats exploit as the default baseline.  The target is safe/deceive
only when its risk-adjusted rollout value beats exploit by a margin; otherwise
the target stays exploit.  This implements a practical two-stage gate:

  1. intervene?  -> any non-exploit mode has positive value over exploit
  2. mode choice -> choose the best value-positive safe/deceive mode

The runtime model still outputs one of exploit/safe/deceive, so it remains
compatible with MASKLLMAgent._learned_gate_mode().
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from experiment_trace import write_json, write_jsonl
from risk_aware_reward import add_reward_arguments, reward_config, score_rollouts


SYSTEM = (
    "你是四川麻将 MASK 两阶段风险门控器。先判断是否值得偏离 exploit；"
    "只有 safe 或 deceive 的短窗口结算价值明显高于 exploit 时才介入。"
    "最终只能输出 exploit、safe 或 deceive 中的一个英文标签。"
)


def row_reward_map(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, float]:
    if "mode_rewards" in row:
        return {str(mode): float(value) for mode, value in row["mode_rewards"].items()}

    scores: Dict[str, float] = {}
    for evaluation in row.get("mode_evaluations", []):
        mode = str(evaluation["mode"])
        scored = score_rollouts(evaluation["rollouts"], args)
        scores[mode] = float(scored["risk_adjusted_score"])
    return scores


def choose_target(reward_map: Dict[str, float], args: argparse.Namespace) -> tuple[str, Dict[str, Any]]:
    if "exploit" not in reward_map:
        best = max(reward_map, key=reward_map.get)
        return best, {"reason": "no_exploit_baseline", "deltas": {}}

    exploit_value = reward_map["exploit"]
    deltas = {mode: value - exploit_value for mode, value in reward_map.items()}
    candidates = {
        mode: delta
        for mode, delta in deltas.items()
        if mode in {"safe", "deceive"} and delta >= args.intervention_margin
    }

    if args.require_deceive_margin and "deceive" in candidates:
        if candidates["deceive"] < args.deceive_margin:
            candidates.pop("deceive", None)

    if not candidates:
        return "exploit", {
            "reason": "no_intervention_beats_exploit",
            "exploit_value": exploit_value,
            "deltas": deltas,
        }

    # Prefer deceive over safe only when it is close enough to safe; otherwise
    # choose the stronger intervention.
    best_mode = max(candidates, key=candidates.get)
    if (
        "deceive" in candidates
        and best_mode == "safe"
        and candidates["safe"] - candidates["deceive"] <= args.deceive_tie_margin
    ):
        best_mode = "deceive"
    return best_mode, {
        "reason": "positive_intervention_delta",
        "exploit_value": exploit_value,
        "deltas": deltas,
        "intervention_candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--intervention-margin", type=float, default=2.0)
    parser.add_argument("--deceive-margin", type=float, default=1.0)
    parser.add_argument("--deceive-tie-margin", type=float, default=3.0)
    parser.add_argument("--require-deceive-margin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deceive-oversample", type=int, default=4)
    parser.add_argument("--safe-oversample", type=int, default=2)
    parser.add_argument("--min-grpo-range", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3427)
    add_reward_arguments(parser)
    args = parser.parse_args()

    sft_rows = []
    grpo_rows = []
    labels: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    source_rows = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_rows += 1
            row = json.loads(line)
            reward_map = row_reward_map(row, args)
            if len(reward_map) < 2:
                skipped["fewer_than_two_modes"] += 1
                continue
            target_mode, diagnostics = choose_target(reward_map, args)
            if target_mode not in reward_map:
                skipped["target_not_available"] += 1
                continue

            labels[target_mode] += 1
            prompt = str(row["prompt"])
            sft_row = {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_mode},
                ],
                "state_id": row.get("state_id"),
                "target_mode": target_mode,
                "rule_mode": row.get("rule_mode", "exploit"),
                "available_modes": list(reward_map),
                "mode_actions": row.get("mode_actions", {}),
                "mode_rewards": reward_map,
                "value_gate": diagnostics,
            }
            multiplier = 1
            if target_mode == "deceive":
                multiplier = max(1, args.deceive_oversample)
            elif target_mode == "safe":
                multiplier = max(1, args.safe_oversample)
            for _ in range(multiplier):
                sft_rows.append(dict(sft_row))

            deltas = diagnostics.get("deltas") or {
                mode: value - reward_map.get("exploit", 0.0)
                for mode, value in reward_map.items()
            }
            shaped = {
                mode: round(float(deltas.get(mode, value - reward_map.get("exploit", 0.0))) / args.reward_scale, 6)
                for mode, value in reward_map.items()
            }
            # Small target bonus improves gradient density but keeps rollout
            # deltas dominant.
            shaped[target_mode] = round(shaped[target_mode] + 0.15, 6)
            if max(shaped.values()) - min(shaped.values()) >= args.min_grpo_range / args.reward_scale:
                grpo_row = {
                    "prompt": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "mode_rewards_json": json.dumps(shaped, ensure_ascii=False, sort_keys=True),
                    "legal_modes": list(reward_map),
                    "reference_mode": row.get("rule_mode", "exploit"),
                    "target_mode": target_mode,
                    "state_id": row.get("state_id"),
                    "raw_reward_map": reward_map,
                    "value_gate": diagnostics,
                }
                for _ in range(multiplier):
                    grpo_rows.append(dict(grpo_row))
            else:
                skipped["low_grpo_range"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(sft_rows)
    rng.shuffle(grpo_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.output_dir / "gate_sft.jsonl"
    grpo_path = args.output_dir / "gate_grpo.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(grpo_path, grpo_rows)
    summary = {
        "input": str(args.input),
        "source_rows": source_rows,
        "sft_rows_after_oversampling": len(sft_rows),
        "grpo_rows_after_oversampling": len(grpo_rows),
        "raw_labels_before_oversampling": dict(labels),
        "skipped": dict(skipped),
        "value_gate_config": {
            "intervention_margin": args.intervention_margin,
            "deceive_margin": args.deceive_margin,
            "deceive_tie_margin": args.deceive_tie_margin,
            "deceive_oversample": args.deceive_oversample,
            "safe_oversample": args.safe_oversample,
            "reward_scale": args.reward_scale,
            "min_grpo_range": args.min_grpo_range,
        },
        "reward": reward_config(args),
        "sft_output": str(sft_path),
        "grpo_output": str(grpo_path),
    }
    write_json(args.output_dir / "gate_value_training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

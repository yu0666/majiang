"""Build learned-gate SFT and GRPO data from paired mode rollouts."""

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
    "你是四川麻将 MASK 风险门控器。只能从给出的 exploit、safe、deceive 模式中选择一个。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--min-reward-range", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3407)
    add_reward_arguments(parser)
    args = parser.parse_args()

    sft_rows = []
    grpo_rows = []
    labels: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scores: Dict[str, Dict[str, Any]] = {}
            for evaluation in row["mode_evaluations"]:
                scores[str(evaluation["mode"])] = score_rollouts(
                    evaluation["rollouts"], args
                )
            if len(scores) < 2:
                skipped["fewer_than_two_modes"] += 1
                continue
            reward_map = {
                mode: float(scored["risk_adjusted_score"])
                for mode, scored in scores.items()
            }
            reward_range = max(reward_map.values()) - min(reward_map.values())
            if reward_range < args.min_reward_range:
                skipped["low_reward_range"] += 1
                continue
            best_value = max(reward_map.values())
            best_modes = sorted(
                mode for mode, value in reward_map.items() if value == best_value
            )
            rule_mode = str(row.get("rule_mode", "exploit"))
            target_mode = rule_mode if rule_mode in best_modes else best_modes[0]
            labels[target_mode] += 1
            prompt = str(row["prompt"])
            sft_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target_mode},
                    ],
                    "state_id": row.get("state_id"),
                    "target_mode": target_mode,
                    "rule_mode": rule_mode,
                    "available_modes": list(scores),
                    "reward_range": reward_range,
                    "reward_diagnostics": scores,
                }
            )
            grpo_rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "mode_rewards_json": json.dumps(
                        {
                            mode: round(value / args.reward_scale, 6)
                            for mode, value in reward_map.items()
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "legal_modes": list(scores),
                    "reference_mode": rule_mode,
                    "best_modes": best_modes,
                    "state_id": row.get("state_id"),
                    "raw_reward_range": reward_range,
                    "reward_diagnostics": scores,
                }
            )

    order = list(range(len(sft_rows)))
    random.Random(args.seed).shuffle(order)
    sft_rows = [sft_rows[index] for index in order]
    grpo_rows = [grpo_rows[index] for index in order]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.output_dir / "gate_sft.jsonl"
    grpo_path = args.output_dir / "gate_grpo.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(grpo_path, grpo_rows)
    summary = {
        "input": str(args.input),
        "rows": len(sft_rows),
        "labels": dict(labels),
        "skipped": dict(skipped),
        "reward": {**reward_config(args), "reward_scale": args.reward_scale},
        "sft_output": str(sft_path),
        "grpo_output": str(grpo_path),
    }
    write_json(args.output_dir / "gate_training_data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

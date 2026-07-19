"""Build gate SFT and cached-reward GRPO data from mode-oracle states."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiment_trace import write_json, write_jsonl


SYSTEM = "你是四川麻将 MASK 风险门控器，只能从给定模式中选择一个。"


def run(args):
    sft_rows = []
    grpo_rows = []
    counts = Counter()
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            modes = list(row["available_modes"])
            target = str(row["target_mode"])
            rewards = {mode: float(row["mode_rewards"][mode]) for mode in modes}
            prompt = str(row["prompt"])
            sft_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": target},
                    ],
                    "target_mode": target,
                    "available_modes": modes,
                    "state_id": row["state_id"],
                }
            )
            counts[target] += 1
            if max(rewards.values()) - min(rewards.values()) >= args.min_reward_range:
                scaled = {mode: round(value / args.reward_scale, 6) for mode, value in rewards.items()}
                grpo_rows.append(
                    {
                        "prompt": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                        "mode_rewards_json": json.dumps(scaled, ensure_ascii=False, sort_keys=True),
                        "legal_modes": modes,
                        "reference_mode": row["rule_mode"],
                        "target_mode": target,
                        "state_id": row["state_id"],
                    }
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = args.output_dir / "gate_sft.jsonl"
    grpo_path = args.output_dir / "gate_grpo.jsonl"
    write_jsonl(sft_path, sft_rows)
    write_jsonl(grpo_path, grpo_rows)
    summary = {
        "sft_rows": len(sft_rows),
        "grpo_rows": len(grpo_rows),
        "target_counts": dict(counts),
        "reward_scale": args.reward_scale,
        "min_reward_range": args.min_reward_range,
        "sft_output": str(sft_path),
        "grpo_output": str(grpo_path),
    }
    write_json(args.output_dir / "gate_training_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--min-reward-range", type=float, default=2.0)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))

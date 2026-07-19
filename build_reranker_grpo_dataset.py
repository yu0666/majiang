"""Convert candidate rollout traces into environment-reward GRPO prompts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from experiment_trace import write_json, write_jsonl
from risk_aware_reward import add_reward_arguments, reward_config, score_rollouts


def candidate_orders(candidates: List[str], count: int) -> List[List[str]]:
    """Rotate candidates so no action has a fixed positional shortcut."""
    if not candidates:
        return []
    canonical = sorted(candidates)
    return [
        canonical[offset:] + canonical[:offset]
        for offset in range(min(max(1, count), len(canonical)))
    ]


def build_rows(row: Dict[str, Any], args: argparse.Namespace) -> tuple[List[Dict[str, Any]], str]:
    candidates = list(row.get("candidates") or [])
    if len(candidates) < 2:
        return [], "too_few_candidates"
    reward_map = {}
    reward_diagnostics = {}
    for action_row in row["action_evaluations"]:
        action = str(action_row["action"])
        scored = score_rollouts(action_row["rollouts"], args)
        reward_map[action] = scored["risk_adjusted_score"]
        reward_diagnostics[action] = {
            key: scored[key]
            for key in (
                "mean_raw_return",
                "mean_fan",
                "lower_tail_cvar",
                "catastrophic_loss_rate",
                "risk_adjusted_score",
            )
        }
    if any(action not in reward_map for action in candidates):
        return [], "missing_action_reward"
    reward_range = max(reward_map.values()) - min(reward_map.values())
    if reward_range < args.min_reward_range:
        return [], "low_reward_range"

    baseline = str(row["comparison"]["baseline_action"])
    best_reward = max(reward_map.values())
    best_actions = sorted(
        action for action, reward in reward_map.items() if reward == best_reward
    )
    scaled_rewards = {
        action: round(value / args.reward_scale, 6)
        for action, value in reward_map.items()
    }
    output_rows = []
    for order_index, ordered_candidates in enumerate(
        candidate_orders(candidates, args.permutations_per_state)
    ):
        reference_line = f"规则基线动作: {baseline}\n" if args.include_reference else ""
        if args.completion_format == "action_only":
            output_instruction = "只输出一个候选动作本身，例如: d 3万。不要输出解释或 JSON。"
        else:
            output_instruction = '严格输出 JSON: {"action": "候选动作", "reason": "一句话依据"}'
        user_prompt = f"""
{row['prompt']}

【规则约束候选重排】
外层规则已经固定当前模式为: {row['mode']}
{reference_line}候选动作: {", ".join(ordered_candidates)}

不要改变模式，只在候选动作中选择预期收益最高的一项。
{output_instruction}
""".strip()
        output_rows.append(
            {
                "prompt": [
                    {
                        "role": "system",
                        "content": "你是四川麻将候选动作重排器。模式由外层规则固定，只能从候选动作中选择。",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "action_rewards_json": json.dumps(
                    scaled_rewards,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "legal_actions": ordered_candidates,
                "reference_action": baseline,
                "best_actions": best_actions,
                "reference_is_best": baseline in best_actions,
                "mode": row["mode"],
                "state_id": row.get("state_id"),
                "candidate_order_index": order_index,
                "completion_format": args.completion_format,
                "reference_in_prompt": args.include_reference,
                "raw_reward_range": reward_range,
                "reward_diagnostics": reward_diagnostics,
                "rollouts_per_action": len(row["action_evaluations"][0]["rollouts"]),
            }
        )
    return output_rows, "kept"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    output_rows: List[Dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    best_positions: Counter[str] = Counter()
    reference_best_rows = 0
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows, outcome = build_rows(json.loads(line), args)
            outcomes[outcome] += 1
            for row in rows:
                output_rows.append(row)
                modes[str(row["mode"])] += 1
                reference_best_rows += int(bool(row["reference_is_best"]))
                positions = [
                    row["legal_actions"].index(action)
                    for action in row["best_actions"]
                ]
                best_positions[str(min(positions) + 1)] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "outcomes": dict(outcomes),
        "modes": dict(modes),
        "permutations_per_state": args.permutations_per_state,
        "completion_format": args.completion_format,
        "reference_in_prompt": args.include_reference,
        "reference_is_best_rate": (
            reference_best_rows / len(output_rows) if output_rows else None
        ),
        "best_action_position": dict(sorted(best_positions.items())),
        "utility": {
            **reward_config(args),
            "reward_scale": args.reward_scale,
            "min_reward_range": args.min_reward_range,
        },
    }
    write_json(args.output.with_name(f"{args.output.stem}_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("Reranker_grpo_oracle_ckpt100/candidate_oracle_states.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("Reranker_grpo_data/reranker_grpo_env.jsonl"))
    add_reward_arguments(parser)
    parser.add_argument("--reward-scale", type=float, default=20.0)
    parser.add_argument("--min-reward-range", type=float, default=5.0)
    parser.add_argument("--permutations-per-state", type=int, default=1)
    parser.add_argument(
        "--completion-format",
        choices=["json", "action_only"],
        default="json",
    )
    parser.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose the rule baseline in the prompt; disable to avoid copying bias.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

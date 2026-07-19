"""Build high-confidence candidate-reranker SFT examples from rollout traces."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiment_trace import write_json, write_jsonl
from risk_aware_reward import add_reward_arguments, reward_config, score_rollouts


DEFAULT_INPUT = Path("Candidate_oracle_v1_go_nogo/candidate_oracle_states.jsonl")
DEFAULT_OUTPUT = Path("Reranker_sft_data/reranker_sft_high_confidence.jsonl")


def paired_interval(differences: List[float], confidence: float) -> Tuple[float, float, float]:
    center = statistics.mean(differences)
    if len(differences) < 2:
        return center, center, center
    sem = statistics.stdev(differences) / math.sqrt(len(differences))
    try:
        from scipy import stats

        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(differences) - 1))
    except Exception:
        critical = 1.96
    return center, center - critical * sem, center + critical * sem


def sanitize_public_prompt(prompt: str) -> str:
    prompt = re.sub(
        r"(?m)^对手状态:.*$",
        "对手状态: 仅依据公开副露、弃牌、定缺进度与牌局阶段判断",
        prompt,
    )
    hidden_markers = ("可能已听牌(", "真实已听牌", "真实未听牌")
    if any(marker in prompt for marker in hidden_markers):
        raise ValueError("Hidden-hand risk text remains after prompt sanitization")
    return prompt


def action_scores(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    expected_seeds = list(row.get("rollout_seeds", []))
    for action_row in row["action_evaluations"]:
        results = action_row["rollouts"]
        seeds = [result["rollout_seed"] for result in results]
        if expected_seeds and seeds != expected_seeds:
            raise ValueError(f"Rollout seed mismatch in {row.get('state_id')}: {action_row['action']}")
        scored = score_rollouts(results, args)
        scores[action_row["action"]] = {
            "values": scored["adjusted_values"],
            **{key: value for key, value in scored.items() if key not in {"values", "adjusted_values"}},
        }
    return scores


def choose_label(
    row: Dict[str, Any],
    scores: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], str]:
    baseline = str(row["comparison"]["baseline_action"])
    if baseline not in scores:
        return None, "missing_baseline"
    rollout_count = len(scores[baseline]["values"])
    if rollout_count < args.min_rollouts:
        return None, "too_few_rollouts"

    if args.accept_best:
        target_action = max(
            scores,
            key=lambda action: float(scores[action]["risk_adjusted_score"]),
        )
        advantage = (
            float(scores[target_action]["risk_adjusted_score"])
            - float(scores[baseline]["risk_adjusted_score"])
        )
        return {
            "label_type": "risk_adjusted_best",
            "target_action": target_action,
            "baseline_action": baseline,
            "best_comparison": {
                "action": target_action,
                "mean_advantage": advantage,
                "ci_lower": advantage,
                "ci_upper": advantage,
            },
            "comparisons": [],
            "rollout_count": rollout_count,
        }, "accepted_risk_adjusted_best"

    comparisons = []
    baseline_values = scores[baseline]["values"]
    for action, action_score in scores.items():
        if action == baseline:
            continue
        differences = [
            candidate - base
            for candidate, base in zip(action_score["values"], baseline_values)
        ]
        mean_diff, ci_lower, ci_upper = paired_interval(differences, args.confidence)
        comparisons.append(
            {
                "action": action,
                "mean_advantage": mean_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    if not comparisons:
        return None, "no_alternative"

    best = max(comparisons, key=lambda item: item["mean_advantage"])
    if (
        best["mean_advantage"] >= args.min_mean_advantage
        and best["ci_lower"] >= args.min_ci_lower
    ):
        return {
            "label_type": "improvement",
            "target_action": best["action"],
            "baseline_action": baseline,
            "best_comparison": best,
            "comparisons": comparisons,
            "rollout_count": rollout_count,
        }, "accepted_improvement"

    if all(item["ci_upper"] <= -args.min_anchor_advantage for item in comparisons):
        strongest_challenger = max(comparisons, key=lambda item: item["ci_upper"])
        return {
            "label_type": "anchor",
            "target_action": baseline,
            "baseline_action": baseline,
            "best_comparison": strongest_challenger,
            "comparisons": comparisons,
            "rollout_count": rollout_count,
        }, "accepted_anchor"
    return None, "ambiguous_interval"


def build_messages(
    row: Dict[str, Any],
    label: Dict[str, Any],
    completion_format: str,
) -> List[Dict[str, str]]:
    candidates = [action_row["action"] for action_row in row["action_evaluations"]]
    prompt = sanitize_public_prompt(str(row["prompt"]))
    output_instruction = (
        "只输出一个候选动作本身，例如: d 3万。不要输出解释或 JSON。"
        if completion_format == "action_only"
        else '严格输出 JSON: {"action": "候选动作", "reason": "一句话依据"}'
    )
    user = f"""
{prompt}

【规则约束候选重排】
固定模式: {row['mode']}
规则基线动作: {label['baseline_action']}
候选动作: {", ".join(candidates)}

不要改变模式，只在候选动作中选择预期收益最高的一项。
{output_instruction}
""".strip()
    assistant = (
        label["target_action"]
        if completion_format == "action_only"
        else json.dumps(
            {"action": label["target_action"], "reason": "paired rollout high-confidence label"},
            ensure_ascii=False,
        )
    )
    return [
        {
            "role": "system",
            "content": "你是四川麻将候选动作重排器。模式由外层规则固定，只能从候选动作中选择。",
        },
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def load_rows(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield path, json.loads(line)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    examples = []
    audit_rows = []
    outcomes: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    label_types: Counter[str] = Counter()
    seen = set()

    for source_path, row in load_rows(args.inputs):
        identity = (str(source_path.resolve()), str(row.get("state_id")))
        if identity in seen:
            outcomes["duplicate"] += 1
            continue
        seen.add(identity)
        scores = action_scores(row, args)
        label, outcome = choose_label(row, scores, args)
        outcomes[outcome] += 1
        audit = {
            "source": str(source_path),
            "state_id": row.get("state_id"),
            "mode": row.get("mode"),
            "outcome": outcome,
            "label": label,
            "action_summary": {
                action: {
                    "mean_utility": values["mean_utility"],
                    "mean_raw_return": values["mean_raw_return"],
                    "mean_fan": values["mean_fan"],
                    "lower_tail_cvar": values["lower_tail_cvar"],
                    "catastrophic_loss_rate": values["catastrophic_loss_rate"],
                    "risk_adjusted_score": values["risk_adjusted_score"],
                }
                for action, values in scores.items()
            },
        }
        audit_rows.append(audit)
        if label is None:
            continue

        label_types[label["label_type"]] += 1
        modes[str(row.get("mode"))] += 1
        examples.append(
            {
                "messages": build_messages(row, label, args.completion_format),
                "state_id": row.get("state_id"),
                "mode": row.get("mode"),
                "target_action": label["target_action"],
                "baseline_action": label["baseline_action"],
                "label_type": label["label_type"],
                "mean_advantage": label["best_comparison"]["mean_advantage"],
                "ci_lower": label["best_comparison"]["ci_lower"],
                "ci_upper": label["best_comparison"]["ci_upper"],
                "rollout_count": label["rollout_count"],
                "source": str(source_path),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, examples)
    audit_path = args.output.with_name(f"{args.output.stem}_audit.jsonl")
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    write_jsonl(audit_path, audit_rows)
    summary = {
        "inputs": [str(path) for path in args.inputs],
        "states_seen": len(seen),
        "examples": len(examples),
        "acceptance_rate": len(examples) / len(seen) if seen else 0.0,
        "outcomes": dict(outcomes),
        "label_types": dict(label_types),
        "modes": dict(modes),
        "utility": reward_config(args),
        "filter": {
            "confidence": args.confidence,
            "min_rollouts": args.min_rollouts,
            "min_mean_advantage": args.min_mean_advantage,
            "min_ci_lower": args.min_ci_lower,
            "min_anchor_advantage": args.min_anchor_advantage,
        },
        "output": str(args.output),
        "audit": str(audit_path),
        "completion_format": args.completion_format,
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[DEFAULT_INPUT])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    add_reward_arguments(parser)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-rollouts", type=int, default=32)
    parser.add_argument("--min-mean-advantage", type=float, default=5.0)
    parser.add_argument("--min-ci-lower", type=float, default=0.0)
    parser.add_argument("--min-anchor-advantage", type=float, default=2.0)
    parser.add_argument(
        "--accept-best",
        action="store_true",
        help="Use the highest risk-adjusted rollout action for every state.",
    )
    parser.add_argument(
        "--completion-format",
        choices=["json", "action_only"],
        default="json",
    )
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

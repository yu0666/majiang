"""Shared settlement, fan, and downside-risk utilities for cached rollouts."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List


def clipped_return(value: float, limit: float) -> float:
    if limit <= 0:
        return value
    return max(-limit, min(limit, value))


def rollout_utility(
    result: Dict[str, Any],
    *,
    return_clip: float,
    hu_bonus: float,
    fan_bonus: float,
    dealin_penalty: float,
) -> float:
    settlement = clipped_return(float(result["continuation_return"]), return_clip)
    hu = float(bool(result.get("agent_hu")))
    fan = float(result.get("agent_hu_fan", 0) or 0)
    dealin = float(bool(result.get("agent_dealin")))
    return settlement + hu_bonus * hu + fan_bonus * fan - dealin_penalty * dealin


def lower_tail_cvar(values: Iterable[float], alpha: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    count = max(1, math.ceil(len(ordered) * max(0.0, min(1.0, alpha))))
    return statistics.mean(ordered[:count])


def score_rollouts(results: List[Dict[str, Any]], args: Any) -> Dict[str, Any]:
    values = [
        rollout_utility(
            result,
            return_clip=float(args.return_clip),
            hu_bonus=float(args.hu_bonus),
            fan_bonus=float(args.fan_bonus),
            dealin_penalty=float(args.dealin_penalty),
        )
        for result in results
    ]
    cvar = lower_tail_cvar(values, float(args.tail_alpha))
    downside = max(0.0, -cvar)
    threshold = abs(float(args.catastrophic_loss_threshold))
    catastrophic_rate = statistics.mean(
        float(float(result["continuation_return"]) <= -threshold)
        for result in results
    )
    tail_penalty = float(args.tail_risk_weight) * downside
    catastrophic_penalty = float(args.catastrophic_loss_penalty) * catastrophic_rate
    risk_adjusted_score = statistics.mean(values) - tail_penalty - catastrophic_penalty
    # Adding action-level risk penalties to each sample keeps paired confidence
    # intervals consistent with the scalar objective used by GRPO.
    adjusted_values = [value - tail_penalty - catastrophic_penalty for value in values]
    return {
        "values": values,
        "adjusted_values": adjusted_values,
        "mean_utility": statistics.mean(values),
        "mean_raw_return": statistics.mean(
            float(result["continuation_return"]) for result in results
        ),
        "mean_fan": statistics.mean(
            float(result.get("agent_hu_fan", 0) or 0) for result in results
        ),
        "lower_tail_cvar": cvar,
        "catastrophic_loss_rate": catastrophic_rate,
        "tail_penalty": tail_penalty,
        "catastrophic_penalty": catastrophic_penalty,
        "risk_adjusted_score": risk_adjusted_score,
    }


def add_reward_arguments(parser) -> None:
    parser.add_argument(
        "--return-clip",
        type=float,
        default=0.0,
        help="Absolute settlement clipping; 0 keeps the actual settlement return.",
    )
    parser.add_argument("--hu-bonus", type=float, default=5.0)
    parser.add_argument("--fan-bonus", type=float, default=3.0)
    parser.add_argument("--dealin-penalty", type=float, default=20.0)
    parser.add_argument("--tail-alpha", type=float, default=0.2)
    parser.add_argument("--tail-risk-weight", type=float, default=0.5)
    parser.add_argument("--catastrophic-loss-threshold", type=float, default=200.0)
    parser.add_argument("--catastrophic-loss-penalty", type=float, default=40.0)


def reward_config(args: Any) -> Dict[str, float]:
    return {
        key: float(getattr(args, key))
        for key in (
            "return_clip",
            "hu_bonus",
            "fan_bonus",
            "dealin_penalty",
            "tail_alpha",
            "tail_risk_weight",
            "catastrophic_loss_threshold",
            "catastrophic_loss_penalty",
        )
    }

"""
Trace and metric helpers for MASK Gate1 experiments.

The trace schema is intentionally JSONL-first so E1/E2/E6 can share the same
raw records.  Current experiments may use a heuristic fallback backend; the
backend field must be kept in reports to avoid treating smoke-test numbers as
trained LLM results.
"""

from __future__ import annotations

import bisect
import json
import os
import random
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def ensure_deterministic_hashing() -> None:
    """Re-exec the process once with PYTHONHASHSEED=0 for reproducibility.

    The mahjong engine iterates ``set(...)`` during play, so without a fixed
    hash seed two identical runs diverge (different gameplay -> different sample
    counts and metrics).  Call this as the first line of an experiment ``main()``
    so the gate decision is reproducible.
    """
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def binary_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    positives = [(s, i) for i, (y, s) in enumerate(zip(labels, scores)) if y == 1]
    negatives = [(s, i) for i, (y, s) in enumerate(zip(labels, scores)) if y == 0]
    if not positives or not negatives:
        return None

    wins = 0.0
    total = len(positives) * len(negatives)
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def calibration_error(labels: List[int], scores: List[float], bins: int = 10) -> Optional[float]:
    if not labels:
        return None
    buckets: Dict[int, List[tuple[int, float]]] = defaultdict(list)
    for label, score in zip(labels, scores):
        idx = min(bins - 1, max(0, int(score * bins)))
        buckets[idx].append((label, score))

    ece = 0.0
    n = len(labels)
    for values in buckets.values():
        conf = sum(score for _, score in values) / len(values)
        acc = sum(label for label, _ in values) / len(values)
        ece += (len(values) / n) * abs(conf - acc)
    return ece


def brier_score(labels: List[int], scores: List[float]) -> Optional[float]:
    """Mean squared error against binary outcomes: a strictly proper scoring rule.

    Unlike absolute error to a soft target (BSE), Brier rewards calibration *and*
    discrimination, so a constant base-rate predictor cannot game it on a
    class-balanced evaluation set.
    """
    if not labels:
        return None
    return sum((s - y) ** 2 for y, s in zip(labels, scores)) / len(labels)


def fit_isotonic(pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Pool-Adjacent-Violators isotonic regression.

    ``pairs`` are (score, target) points; the returned step function is
    non-decreasing in score, so applying it never inverts the ranking (AUC is
    preserved up to ties).  Targets are usually binary outcomes, giving a
    calibrated P(positive | score).
    """
    pts = sorted(pairs, key=lambda p: p[0])
    if not pts:
        return [(1.0, 0.5)]
    # blocks: [sum_target, count, score_right]
    blocks: List[List[float]] = []
    for score, target in pts:
        blocks.append([float(target), 1.0, float(score)])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            s2, c2, x2 = blocks.pop()
            s1, c1, _ = blocks.pop()
            blocks.append([s1 + s2, c1 + c2, x2])
    return [(x_right, s / c) for s, c, x_right in blocks]


def apply_isotonic(steps: List[Tuple[float, float]], score: float) -> float:
    if not steps:
        return max(0.0, min(1.0, score))
    thresholds = [x for x, _ in steps]
    idx = bisect.bisect_left(thresholds, score)
    if idx >= len(steps):
        idx = len(steps) - 1
    return max(0.0, min(1.0, steps[idx][1]))


def balance_by_binary(
    rows: List[Dict[str, Any]],
    label_fn: Callable[[Dict[str, Any]], int],
    seed: int = 12345,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Subsample a list to an equal number of positives and negatives.

    No oversampling/duplication, so the balanced subset stays leakage-safe.
    Returns ``(subset, True)`` when both classes exist, else ``(rows, False)``.
    """
    pos = [r for r in rows if label_fn(r) == 1]
    neg = [r for r in rows if label_fn(r) == 0]
    if not pos or not neg:
        return list(rows), False
    k = min(len(pos), len(neg))
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = pos[:k] + neg[:k]
    rng.shuffle(out)
    return out, True


def summarize_belief_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "samples": 0,
            "note": "No belief samples were collected.",
        }

    labels = [int(s["true_tenpai"]) for s in samples]
    scores = [float(s["tenpai_confidence"]) for s in samples]
    preds = [1 if s.get("think_i_am_tenpai") == "yes" or float(s["tenpai_confidence"]) >= 0.5 else 0 for s in samples]

    return {
        "samples": len(samples),
        "positive_rate": sum(labels) / len(labels),
        "accuracy": sum(1 for y, p in zip(labels, preds) if y == p) / len(labels),
        "auc": binary_auc(labels, scores),
        "brier": sum((score - y) ** 2 for y, score in zip(labels, scores)) / len(labels),
        "bse_abs": sum(abs(score - y) for y, score in zip(labels, scores)) / len(labels),
        "ece": calibration_error(labels, scores),
        "backend_counts": dict(Counter(s.get("backend", "unknown") for s in samples)),
        "label_scope": "current minimal oracle: true P0 tenpai, not a full opponent-posterior oracle yet",
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round((len(values) - 1) * p)))
    return values[idx]


def summarize_games(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not games:
        return {"games": 0}

    n = len(games)
    times = [t for g in games for t in g.get("decision_ms", [])]
    mode_counts = Counter()
    for g in games:
        mode_counts.update(g.get("mode_counts", {}))

    deceive_windows = sum(int(g.get("deceive_windows", 0)) for g in games)
    induced_dealins = sum(int(g.get("induced_dealin", 0)) for g in games)
    false_fold_opps = sum(int(g.get("false_fold_opportunities", 0)) for g in games)
    false_folds = sum(int(g.get("false_folds", 0)) for g in games)

    return {
        "games": n,
        "avg_net": sum(float(g["agent_net"]) for g in games) / n,
        "hu_rate": sum(1 for g in games if g.get("agent_hu")) / n,
        "dealin_rate": sum(1 for g in games if g.get("agent_dealin")) / n,
        "avg_steps": sum(float(g["steps"]) for g in games) / n,
        "DIR": induced_dealins / deceive_windows if deceive_windows else 0.0,
        "DIR_counts": {
            "induced_dealin": induced_dealins,
            "deceive_windows": deceive_windows,
        },
        "FFR": false_folds / false_fold_opps if false_fold_opps else 0.0,
        "FFR_counts": {
            "false_folds": false_folds,
            "false_fold_opportunities": false_fold_opps,
            "note": "Rule bots almost always hu when legal, so this stays near zero until LLM opponents are added.",
        },
        "decision_latency_ms": {
            "p50": percentile(times, 0.50),
            "p95": percentile(times, 0.95),
            "p99": percentile(times, 0.99),
        },
        "mode_counts": dict(mode_counts),
    }


def summarize_by_method(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for game in games:
        grouped[game["method"]].append(game)
    return {method: summarize_games(rows) for method, rows in sorted(grouped.items())}


def sign_test_p_value(deltas: List[float]) -> Optional[float]:
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return None
    positives = sum(1 for d in nonzero if d > 0)
    k = min(positives, n - positives)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_method_comparison(
    games: List[Dict[str, Any]],
    base_method: str = "llm_reactive_z",
    target_method: str = "llm_mask",
) -> Dict[str, Any]:
    by_key: Dict[tuple[int, str], Dict[str, Any]] = {}
    for game in games:
        by_key[(int(game["seed"]), game["method"])] = game

    seeds = sorted(
        seed
        for seed, method in by_key
        if method == base_method and (seed, target_method) in by_key
    )
    if not seeds:
        return {
            "base": base_method,
            "target": target_method,
            "paired_seeds": 0,
            "note": "No paired seeds available.",
        }

    net_deltas = []
    dealin_deltas = []
    hu_deltas = []
    step_deltas = []
    for seed in seeds:
        base = by_key[(seed, base_method)]
        target = by_key[(seed, target_method)]
        net_deltas.append(float(target["agent_net"]) - float(base["agent_net"]))
        dealin_deltas.append(float(target["agent_dealin"]) - float(base["agent_dealin"]))
        hu_deltas.append(float(target["agent_hu"]) - float(base["agent_hu"]))
        step_deltas.append(float(target["steps"]) - float(base["steps"]))

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "base": base_method,
        "target": target_method,
        "paired_seeds": len(seeds),
        "avg_delta_net": avg(net_deltas),
        "avg_delta_dealin_rate": avg(dealin_deltas),
        "avg_delta_hu_rate": avg(hu_deltas),
        "avg_delta_steps": avg(step_deltas),
        "net_positive_rate": sum(1 for d in net_deltas if d > 0) / len(net_deltas),
        "net_sign_test_p": sign_test_p_value(net_deltas),
        "gate1_direction": {
            "score_improved": avg(net_deltas) > 0,
            "dealin_not_worse": avg(dealin_deltas) <= 0,
            "interpretation": "Use only as smoke-test direction unless backend is a trained/local LLM and games are large enough.",
        },
    }

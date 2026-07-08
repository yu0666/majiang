"""
Post-hoc calibration for B_phi predictions.

This script reads evaluate_belief_sft.py prediction JSONL and tests whether a
simple calibration layer can reduce BSE without retraining the LLM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from experiment_trace import (
    apply_isotonic,
    balance_by_binary,
    binary_auc,
    brier_score,
    calibration_error,
    fit_isotonic,
    write_json,
    write_jsonl,
)


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_soft_label(row: Dict[str, Any]) -> float:
    return float(row["label"]["tenpai_confidence"])


def get_binary_label(row: Dict[str, Any]) -> int:
    return 1 if row["label"]["think_i_am_tenpai"] == "yes" else 0


def get_score(row: Dict[str, Any]) -> float:
    return float(row["prediction"]["tenpai_confidence"])


def split_rows(rows: List[Dict[str, Any]], calib_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cutoff = max(1, int(len(shuffled) * calib_ratio))
    return shuffled[:cutoff], shuffled[cutoff:]


def fit_linear(calib: List[Dict[str, Any]]) -> Dict[str, float]:
    xs = [get_score(row) for row in calib]
    ys = [get_soft_label(row) for row in calib]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 1e-12:
        return {"a": 0.0, "b": y_mean}
    a = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    # Keep the calibration monotone so AUC is not inverted.
    a = max(0.0, a)
    b = y_mean - a * x_mean
    return {"a": a, "b": b}


def apply_linear(score: float, params: Dict[str, float]) -> float:
    return clip01(params["a"] * score + params["b"])


def fit_histogram(calib: List[Dict[str, Any]], bins: int, alpha: float) -> List[float]:
    global_prior = sum(get_soft_label(row) for row in calib) / len(calib)
    bucket_values: List[List[float]] = [[] for _ in range(bins)]
    for row in calib:
        score = clip01(get_score(row))
        idx = min(bins - 1, int(score * bins))
        bucket_values[idx].append(get_soft_label(row))

    mapping = []
    for values in bucket_values:
        if values:
            calibrated = (sum(values) + alpha * global_prior) / (len(values) + alpha)
        else:
            calibrated = global_prior
        mapping.append(clip01(calibrated))
    return mapping


def apply_histogram(score: float, mapping: List[float]) -> float:
    bins = len(mapping)
    idx = min(bins - 1, int(clip01(score) * bins))
    return mapping[idx]


def fit_isotonic_binary(calib: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Isotonic (PAV) calibration to the binary tenpai outcome.

    Monotonic, so it fixes calibration without inverting the ranking the way the
    independent-bin histogram can (which previously collapsed AUC).
    """
    return fit_isotonic([(clip01(get_score(row)), float(get_binary_label(row))) for row in calib])


def summarize(rows: List[Dict[str, Any]], scores: List[float]) -> Dict[str, Any]:
    labels_binary = [get_binary_label(row) for row in rows]
    labels_soft = [get_soft_label(row) for row in rows]
    preds = [1 if score >= 0.5 else 0 for score in scores]
    n = len(rows)
    return {
        "samples": n,
        "positive_rate": sum(labels_binary) / n if n else 0.0,
        "soft_oracle_mean": sum(labels_soft) / n if n else 0.0,
        "accuracy": sum(1 for y, p in zip(labels_binary, preds) if y == p) / n if n else 0.0,
        "auc": binary_auc(labels_binary, scores) if n else None,
        "brier_soft": sum((score - y) ** 2 for score, y in zip(scores, labels_soft)) / n if n else 0.0,
        "brier_binary": brier_score(labels_binary, scores) if n else None,
        "BSE_abs": sum(abs(score - y) for score, y in zip(scores, labels_soft)) / n if n else 0.0,
        "ECE": calibration_error(labels_binary, scores) if n else None,
    }


def summarize_balanced(rows: List[Dict[str, Any]], scores: List[float]) -> Dict[str, Any]:
    """Brier/AUC on a class-balanced subset of the test rows (same row order)."""
    indexed = [{"_row": row, "_score": score} for row, score in zip(rows, scores)]
    balanced, ok = balance_by_binary(indexed, lambda it: get_binary_label(it["_row"]))
    if not ok:
        return {"samples": 0, "note": "single-class test set; cannot balance"}
    return summarize([it["_row"] for it in balanced], [it["_score"] for it in balanced])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("H1_belief_calibration"))
    parser.add_argument("--calib-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    calib_rows, test_rows = split_rows(rows, args.calib_ratio, args.seed)

    linear_params = fit_linear(calib_rows)
    hist_map = fit_histogram(calib_rows, args.bins, args.alpha)
    isotonic_steps = fit_isotonic_binary(calib_rows)

    methods = {
        "identity": [get_score(row) for row in test_rows],
        "linear": [apply_linear(get_score(row), linear_params) for row in test_rows],
        "histogram": [apply_histogram(get_score(row), hist_map) for row in test_rows],
        "isotonic": [apply_isotonic(isotonic_steps, get_score(row)) for row in test_rows],
    }

    summary = {
        "predictions": str(args.predictions),
        "calib_samples": len(calib_rows),
        "test_samples": len(test_rows),
        "linear_params": linear_params,
        "histogram_bins": hist_map,
        "isotonic_steps": isotonic_steps,
        "methods": {name: summarize(test_rows, scores) for name, scores in methods.items()},
        "methods_balanced": {name: summarize_balanced(test_rows, scores) for name, scores in methods.items()},
        "note": "Calibration is fit on a held-out calibration split. isotonic is monotonic (AUC-preserving); prefer methods_balanced + brier_binary over BSE_abs for fair comparison under class imbalance.",
    }

    calibrated_rows = []

    def selection_score(name: str) -> float:
        balanced = summary["methods_balanced"][name]
        brier = balanced.get("brier_binary") if isinstance(balanced, dict) else None
        if brier is None:  # single-class test set: fall back to natural Brier
            brier = summary["methods"][name].get("brier_binary")
        return brier if brier is not None else summary["methods"][name]["BSE_abs"]

    # Pick the calibrator that minimizes a proper score on the balanced subset,
    # not absolute error on the imbalanced set (which favours near-zero output).
    best_method = min(summary["methods"], key=selection_score)
    best_scores = methods[best_method]
    for row, score in zip(test_rows, best_scores):
        item = dict(row)
        item["calibrated_tenpai_confidence"] = score
        item["calibration_method"] = best_method
        calibrated_rows.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "calibration_summary.json", summary)
    write_jsonl(args.output_dir / "calibrated_predictions.jsonl", calibrated_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved calibration outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()

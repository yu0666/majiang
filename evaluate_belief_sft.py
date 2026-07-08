"""
Evaluate a trained LLM-B_phi adapter on belief-SFT eval data.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_trace import balance_by_binary, binary_auc, brier_score, calibration_error, write_json, write_jsonl
from llm_backends import build_llm_callable


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def iter_rows(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def get_prompt(row: Dict[str, Any]) -> str:
    messages = row["messages"]
    return messages[1]["content"]


def get_label(row: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(row["messages"][-1]["content"])


def normalize_prediction(parsed: Optional[Dict[str, Any]], fallback_target: str) -> Dict[str, Any]:
    if parsed is None:
        parsed = {}
    label = parsed.get("think_i_am_tenpai", "uncertain")
    if label not in {"yes", "no", "uncertain"}:
        label = "uncertain"
    try:
        conf = float(parsed.get("tenpai_confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return {
        "target_opponent": parsed.get("target_opponent", fallback_target),
        "think_i_am_tenpai": label,
        "tenpai_confidence": conf,
        "suspected_waits": parsed.get("suspected_waits", []) or [],
        "suspected_pattern": parsed.get("suspected_pattern", "unknown"),
        "danger_tiles_for_me": parsed.get("danger_tiles_for_me", []) or [],
        "reason": parsed.get("reason", ""),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels_binary = [1 if row["label"]["think_i_am_tenpai"] == "yes" else 0 for row in rows]
    labels_soft = [float(row["label"]["tenpai_confidence"]) for row in rows]
    scores = [float(row["prediction"]["tenpai_confidence"]) for row in rows]
    preds = [1 if score >= 0.5 else 0 for score in scores]
    n = len(rows)
    if n == 0:
        return {"samples": 0}
    return {
        "samples": n,
        "positive_rate": sum(labels_binary) / n,
        "soft_oracle_mean": sum(labels_soft) / n,
        "accuracy": sum(1 for y, p in zip(labels_binary, preds) if y == p) / n,
        "auc": binary_auc(labels_binary, scores),
        "brier_soft": sum((score - y) ** 2 for score, y in zip(scores, labels_soft)) / n,
        # Proper scoring rule vs the binary outcome; comparable across base rates.
        "brier_binary": brier_score(labels_binary, scores),
        "BSE_abs": sum(abs(score - y) for score, y in zip(scores, labels_soft)) / n,
        "ECE": calibration_error(labels_binary, scores),
        "json_parse_rate": sum(1 for row in rows if row["json_ok"]) / n,
        "latency_ms_avg": sum(float(row["latency_ms"]) for row in rows) / n,
    }


def summarize_with_balanced(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report metrics on the natural eval set and on a class-balanced subset.

    On a 3-5% positive eval set, BSE/Brier are dominated by the base rate; the
    balanced view shows whether B_phi actually discriminates tenpai.
    """
    natural = summarize(rows)
    balanced_rows, ok = balance_by_binary(
        rows, lambda r: 1 if r["label"]["think_i_am_tenpai"] == "yes" else 0
    )
    return {
        "natural_eval": natural,
        "balanced_eval": summarize(balanced_rows) if ok else {"samples": 0, "note": "single-class eval; cannot balance"},
        "balanced_eval_available": ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=Path, default=Path("belief_sft_data_eval.jsonl"))
    parser.add_argument("--model-path", default="models/Qwen-Mahjong-V3-Merged")
    parser.add_argument("--adapter-path", default="qwen-bphi-sft")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output-dir", type=Path, default=Path("H1_belief_sft_eval"))
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend="local_qwen",
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )

    out_rows = []
    for idx, row in enumerate(iter_rows(args.eval_file, args.limit)):
        prompt = get_prompt(row)
        label = get_label(row)
        target = label.get("target_opponent", f"P{row.get('meta', {}).get('target_opponent_id', '?')}")
        t0 = time.perf_counter()
        raw = llm(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        parsed = safe_json_loads(raw)
        pred = normalize_prediction(parsed, target)
        out_rows.append(
            {
                "idx": idx,
                "meta": row.get("meta", {}),
                "label": label,
                "prediction": pred,
                "raw": raw,
                "json_ok": parsed is not None,
                "latency_ms": latency_ms,
            }
        )

    summary = {
        "eval_file": str(args.eval_file),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "summary": summarize(out_rows),
        "summary_by_scope": summarize_with_balanced(out_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "belief_sft_eval_summary.json", summary)
    write_jsonl(args.output_dir / "belief_sft_eval_predictions.jsonl", out_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved belief-SFT eval under: {args.output_dir}")


if __name__ == "__main__":
    main()

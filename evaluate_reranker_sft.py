"""Evaluate reranker adapters on held-out public-info candidate prompts."""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_trace import write_json, write_jsonl
from llm_backends import LocalQwenCallable
from mask_llm import legalize_action


DEFAULT_DATA = Path("Reranker_validation_public_1seed_20games/reranker_validation.jsonl")
DEFAULT_ADAPTERS = [
    "base",
    "qwen-v1-candidate-reranker-sft/checkpoint-50",
    "qwen-v1-candidate-reranker-sft/checkpoint-100",
]


def load_rows(path: Path, per_mode: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mode = str((row.get("meta") or {}).get("mode", row.get("mode", "unknown")))
            if len(grouped[mode]) < per_mode:
                grouped[mode].append(row)
    return [row for mode in sorted(grouped) for row in grouped[mode]]


def parse_candidates(prompt: str) -> List[str]:
    matches = re.findall(r"候选动作:\s*(.*?)(?:\n|$)", prompt)
    if not matches:
        return []
    return [item.strip() for item in matches[-1].split(",") if item.strip()]


def parse_raw_action(raw: str) -> tuple[Optional[str], bool]:
    stripped = raw.strip()
    if re.fullmatch(r"d\s+[1-9][万条筒]|[hgpn]", stripped):
        return stripped, True
    try:
        parsed = json.loads(stripped)
        return str(parsed.get("action", "")).strip() or None, True
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return str(parsed.get("action", "")).strip() or None, True
            except json.JSONDecodeError:
                pass
    match = re.search(r"d\s+[1-9][万条筒]|[hgpn]", raw)
    return (match.group(0) if match else None), False


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(group)
        return {
            "examples": n,
            "json_parse_rate": sum(row["json_parsed"] for row in group) / n if n else 0.0,
            "raw_legal_rate": sum(row["raw_legal"] for row in group) / n if n else 0.0,
            "raw_accuracy": sum(row["raw_action"] == row["target_action"] for row in group) / n if n else 0.0,
            "deployed_accuracy": sum(row["deployed_action"] == row["target_action"] for row in group) / n if n else 0.0,
            "mean_latency_ms": statistics.mean(row["latency_ms"] for row in group) if group else 0.0,
        }

    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    return {
        "overall": group_summary(rows),
        "by_mode": {mode: group_summary(group) for mode, group in sorted(by_mode.items())},
        "deployed_action_counts": dict(Counter(row["deployed_action"] for row in rows)),
    }


def evaluate_adapter(
    model_path: str,
    adapter_label: str,
    rows: List[Dict[str, Any]],
    max_new_tokens: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    adapter_path = None if adapter_label == "base" else adapter_label
    llm = LocalQwenCallable(
        model_path=model_path,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    outputs = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        prompt = str(messages[-2]["content"])
        target = str((row.get("meta") or {}).get("action") or row.get("target_action"))
        mode = str((row.get("meta") or {}).get("mode") or row.get("mode"))
        candidates = parse_candidates(prompt)
        started = time.perf_counter()
        raw = llm(prompt)
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_action, parsed = parse_raw_action(raw)
        deployed = legalize_action(raw, candidates) if candidates else raw_action
        outputs.append(
            {
                "adapter": adapter_label,
                "index": index,
                "mode": mode,
                "target_action": target,
                "candidates": candidates,
                "raw": raw,
                "raw_action": raw_action,
                "json_parsed": parsed,
                "raw_legal": raw_action in candidates,
                "deployed_action": deployed,
                "latency_ms": latency_ms,
            }
        )
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"[Reranker eval] adapter={adapter_label} {index + 1}/{len(rows)}", flush=True)

    summary = summarize(outputs)
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/Qwen-Mahjong-V1-Mixed-SFT-Merged")
    parser.add_argument("--adapters", nargs="+", default=DEFAULT_ADAPTERS)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--per-mode", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-dir", type=Path, default=Path("Reranker_sft_eval"))
    args = parser.parse_args()

    rows = load_rows(args.data_file, args.per_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_outputs = []
    summaries = {}
    for adapter in args.adapters:
        outputs, summary = evaluate_adapter(args.model_path, adapter, rows, args.max_new_tokens)
        all_outputs.extend(outputs)
        summaries[adapter] = summary
    result = {
        "model_path": args.model_path,
        "data_file": str(args.data_file),
        "per_mode_limit": args.per_mode,
        "examples": len(rows),
        "adapters": summaries,
    }
    write_jsonl(args.output_dir / "reranker_eval_outputs.jsonl", all_outputs)
    write_json(args.output_dir / "reranker_eval_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

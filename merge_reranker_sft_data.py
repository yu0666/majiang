"""Merge public-info reranker teacher data with rollout improvement labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from experiment_trace import write_json, write_jsonl


DEFAULT_TEACHER_INPUTS = [
    Path("Reranker_teacher_l2_public_3seeds_100games/reranker_teacher_l2_public.jsonl"),
    Path("Reranker_teacher_safe_deceive_public_1seed_100games/reranker_teacher_safe_deceive.jsonl"),
]
DEFAULT_IMPROVEMENTS = Path("Reranker_sft_data/reranker_sft_high_confidence.jsonl")
DEFAULT_OUTPUT = Path("Reranker_sft_data/reranker_sft_mixed_public.jsonl")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def mode_of(row: Dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(row.get("mode") or meta.get("mode") or "unknown")


def label_type_of(row: Dict[str, Any]) -> str:
    return str(row.get("label_type") or "teacher_anchor")


def public_audit(row: Dict[str, Any]) -> None:
    messages = row.get("messages") or []
    text = "\n".join(str(message.get("content", "")) for message in messages)
    for marker in ("可能已听牌(", "真实已听牌", "真实未听牌"):
        if marker in text:
            raise ValueError(f"Hidden-hand marker found: {marker}")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    dedupe = set()

    for path in args.teacher_inputs:
        for row in read_jsonl(path):
            public_audit(row)
            key = json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True)
            if key in dedupe:
                continue
            dedupe.add(key)
            row["reranker_source"] = "l2_rule_teacher"
            rows.append(row)
            source_counts["l2_rule_teacher"] += 1

    improvement_rows = list(read_jsonl(args.improvement_input))
    for row in improvement_rows:
        public_audit(row)
        for repeat_index in range(args.improvement_repeat):
            copied = dict(row)
            copied["reranker_source"] = "rollout_high_confidence"
            copied["repeat_index"] = repeat_index
            rows.append(copied)
            source_counts["rollout_high_confidence"] += 1

    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    modes = Counter(mode_of(row) for row in rows)
    label_types = Counter(label_type_of(row) for row in rows)
    summary = {
        "teacher_inputs": [str(path) for path in args.teacher_inputs],
        "improvement_input": str(args.improvement_input),
        "improvement_repeat": args.improvement_repeat,
        "rows": len(rows),
        "unique_teacher_rows": len(dedupe),
        "sources": dict(source_counts),
        "modes": dict(modes),
        "label_types": dict(label_types),
        "seed": args.seed,
        "output": str(args.output),
    }
    write_json(args.output.with_name(f"{args.output.stem}_summary.json"), summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-inputs", nargs="+", type=Path, default=DEFAULT_TEACHER_INPUTS)
    parser.add_argument("--improvement-input", type=Path, default=DEFAULT_IMPROVEMENTS)
    parser.add_argument("--improvement-repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

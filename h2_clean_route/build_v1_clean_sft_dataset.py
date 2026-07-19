"""Build V1-clean SFT data from rule and self-play sources only."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_DIR = Path(__file__).resolve().parents[1]


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def valid_messages(row: Dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    roles = [message.get("role") for message in messages[:3] if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"]:
        return False
    return bool(str(messages[-1].get("content", "")).strip())


def assistant_action(row: Dict[str, Any]) -> str:
    content = str(row.get("messages", [{}])[-1].get("content", "")).strip()
    if content.startswith("{"):
        try:
            return str(json.loads(content).get("action", "")).strip()
        except json.JSONDecodeError:
            return content
    return content


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_DIR / path


def load_source(
    path: Path,
    source: str,
    cap: Optional[int],
    repeat: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    invalid = 0
    blocked_keywords = ("mask_l2_teacher", "mode_hint", "B_phi", "deceive")
    keyword_hits = Counter()
    for _, row in iter_jsonl(path):
        raw = json.dumps(row, ensure_ascii=False)
        for keyword in blocked_keywords:
            if keyword in raw:
                keyword_hits[keyword] += 1
        if not valid_messages(row):
            invalid += 1
            continue
        rows.append(row)
    available = len(rows)
    if cap is not None and available > cap:
        rows = rng.sample(rows, cap)

    emitted: List[Dict[str, Any]] = []
    for repeat_idx in range(max(1, repeat)):
        for row in rows:
            item = copy.deepcopy(row)
            meta = item.setdefault("meta", {})
            meta["v1_clean_source"] = source
            meta["v1_clean_repeat"] = repeat_idx
            emitted.append(item)
    stats = {
        "source": source,
        "path": str(path),
        "available_valid": available,
        "invalid": invalid,
        "cap": cap,
        "repeat": repeat,
        "emitted": len(emitted),
        "blocked_keyword_hits": dict(keyword_hits),
        "top_actions": dict(Counter(assistant_action(row) for row in emitted).most_common(30)),
    }
    return emitted, stats


def dedupe(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    output = []
    dropped = 0
    for row in rows:
        messages = row["messages"]
        key = (str(messages[1].get("content", "")), str(messages[-1].get("content", "")))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        output.append(row)
    return output, dropped


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-file", default="v1_clean_sft.jsonl")
    parser.add_argument("--summary-file", default="v1_clean_sft_summary.json")
    parser.add_argument("--rule-file", default="sft_data_elite.jsonl")
    parser.add_argument("--rule-cap", type=int, default=30000)
    parser.add_argument("--rule-repeat", type=int, default=1)
    parser.add_argument("--selfplay-file", default="sft_data_selfplay_v3.jsonl")
    parser.add_argument("--selfplay-cap", type=int, default=-1)
    parser.add_argument("--selfplay-repeat", type=int, default=1)
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def cap(value: int) -> Optional[int]:
        return None if value < 0 else value

    all_rows: List[Dict[str, Any]] = []
    source_stats = []
    for source, file_name, source_cap, repeat in (
        ("rule", args.rule_file, cap(args.rule_cap), args.rule_repeat),
        ("selfplay", args.selfplay_file, cap(args.selfplay_cap), args.selfplay_repeat),
    ):
        rows, stats = load_source(resolve(file_name), source, source_cap, repeat, rng)
        all_rows.extend(rows)
        source_stats.append(stats)
        print(f"[V1-clean data] source={source} emitted={stats['emitted']}", flush=True)

    rng.shuffle(all_rows)
    dropped_duplicates = 0
    if args.dedupe:
        all_rows, dropped_duplicates = dedupe(all_rows)
        rng.shuffle(all_rows)

    eval_count = int(round(len(all_rows) * max(0.0, min(0.5, args.eval_fraction))))
    eval_rows = all_rows[:eval_count]
    train_rows = all_rows[eval_count:]

    output_dir = resolve(args.output_dir)
    output_file = output_dir / args.output_file
    eval_file = output_dir / "v1_clean_sft_eval.jsonl"
    train_count = write_jsonl(output_file, train_rows)
    heldout_count = write_jsonl(eval_file, eval_rows) if eval_rows else 0
    summary = {
        "created_for": "V1-clean shared H2 backbone",
        "excludes": ["MASK teacher", "B_phi", "z_j(t)", "gate labels", "deceive teacher"],
        "seed": args.seed,
        "train_rows": train_count,
        "eval_rows": heldout_count,
        "total_rows_after_dedupe": len(all_rows),
        "dropped_duplicates": dropped_duplicates,
        "output_file": str(output_file),
        "eval_file": str(eval_file) if eval_rows else None,
        "source_stats": source_stats,
        "final_source_counts": dict(Counter(row.get("meta", {}).get("v1_clean_source", "unknown") for row in train_rows)),
        "final_top_actions": dict(Counter(assistant_action(row) for row in train_rows).most_common(30)),
    }
    write_json(output_dir / args.summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


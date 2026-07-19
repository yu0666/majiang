"""Build the V1 mixed SFT dataset from raw Qwen2.5 base-model training sources.

V1 is intended to learn three capabilities at once:
  1. basic Sichuan Mahjong legality and tile logic from rule/bot data;
  2. stronger attack distribution from self-play data;
  3. MASK belief/risk/deceive formatting and behavior from L2 teacher data.

The output remains the same JSONL chat format used by existing SFT scripts:
  {"messages": [...], "meta": {...}}
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from experiment_trace import write_json


DEFAULT_MASK_FILE = "MASK_SFT_teacher_l2_mc_3seeds_200games/mask_sft_l2_mc_3seeds_200games.jsonl"


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def valid_messages(row: Dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return False
    roles = [m.get("role") for m in messages[:3] if isinstance(m, dict)]
    if roles != ["system", "user", "assistant"]:
        return False
    return bool(str(messages[-1].get("content", "")).strip())


def action_from_row(row: Dict[str, Any]) -> str:
    content = str(row.get("messages", [{}])[-1].get("content", "")).strip()
    if content.startswith("{"):
        try:
            parsed = json.loads(content)
            return str(parsed.get("action", "")).strip()
        except json.JSONDecodeError:
            return content
    return content


def load_source(
    path: Path,
    source_name: str,
    cap: Optional[int],
    repeat: int,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    invalid = 0
    for _, row in iter_jsonl(path):
        if not valid_messages(row):
            invalid += 1
            continue
        rows.append(row)

    available = len(rows)
    if cap is not None and cap >= 0 and available > cap:
        rows = rng.sample(rows, cap)

    out: List[Dict[str, Any]] = []
    for repeat_idx in range(max(1, repeat)):
        for row in rows:
            item = copy.deepcopy(row)
            meta = item.setdefault("meta", {})
            meta["v1_source"] = source_name
            meta["v1_source_repeat"] = repeat_idx
            out.append(item)

    stats = {
        "source": source_name,
        "path": str(path),
        "available_valid": available,
        "invalid": invalid,
        "cap": cap,
        "repeat": repeat,
        "emitted": len(out),
        "actions": dict(Counter(action_from_row(row) for row in out).most_common(20)),
        "modes": dict(Counter(str(row.get("meta", {}).get("mode", "none")) for row in out)),
    }
    return out, stats


def dedupe_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    out = []
    dropped = 0
    for row in rows:
        messages = row["messages"]
        key = (
            str(messages[1].get("content", "")),
            str(messages[-1].get("content", "")),
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(row)
    return out, dropped


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("V1_mixed_sft_data"))
    parser.add_argument("--output-file", type=Path, default=Path("v1_mixed_sft_qwenbase.jsonl"))
    parser.add_argument("--summary-file", type=Path, default=Path("v1_mixed_sft_qwenbase_summary.json"))
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--rule-file", default="sft_data_elite.jsonl")
    parser.add_argument("--rule-cap", type=int, default=30000)
    parser.add_argument("--rule-repeat", type=int, default=1)

    parser.add_argument("--selfplay-file", default="sft_data_selfplay_v3.jsonl")
    parser.add_argument("--selfplay-cap", type=int, default=-1, help="-1 keeps all valid rows.")
    parser.add_argument("--selfplay-repeat", type=int, default=1)

    parser.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    parser.add_argument("--mask-cap", type=int, default=-1, help="-1 keeps all valid rows.")
    parser.add_argument("--mask-repeat", type=int, default=2)

    parser.add_argument("--extra-file", action="append", default=[],
                        help="Optional extra JSONL sources, format name:path:cap:repeat.")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    return parser.parse_args()


def source_specs(args: argparse.Namespace) -> List[Tuple[str, Path, Optional[int], int]]:
    def cap(value: int) -> Optional[int]:
        return None if value is None or value < 0 else value

    specs = [
        ("rule", Path(args.rule_file), cap(args.rule_cap), args.rule_repeat),
        ("selfplay", Path(args.selfplay_file), cap(args.selfplay_cap), args.selfplay_repeat),
        ("mask_l2_teacher", Path(args.mask_file), cap(args.mask_cap), args.mask_repeat),
    ]
    for raw in args.extra_file:
        parts = raw.split(":")
        if len(parts) != 4:
            raise ValueError("--extra-file must be name:path:cap:repeat")
        name, path, cap_text, repeat_text = parts
        specs.append((name, Path(path), cap(int(cap_text)), int(repeat_text)))
    return specs


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    source_stats: List[Dict[str, Any]] = []
    for name, path, cap, repeat in source_specs(args):
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {path}")
        rows, stats = load_source(path, name, cap, repeat, rng)
        all_rows.extend(rows)
        source_stats.append(stats)
        print(f"[V1 data] {name}: emitted={stats['emitted']} path={path}", flush=True)

    rng.shuffle(all_rows)
    dropped_duplicates = 0
    if args.dedupe:
        all_rows, dropped_duplicates = dedupe_rows(all_rows)
        rng.shuffle(all_rows)

    eval_rows: List[Dict[str, Any]] = []
    train_rows = all_rows
    if args.eval_fraction > 0:
        n_eval = int(round(len(all_rows) * args.eval_fraction))
        eval_rows = all_rows[:n_eval]
        train_rows = all_rows[n_eval:]

    output_file = args.output_file if args.output_file.is_absolute() else args.output_dir / args.output_file
    summary_file = args.summary_file if args.summary_file.is_absolute() else args.output_dir / args.summary_file
    train_count = write_jsonl(output_file, train_rows)
    eval_file = None
    eval_count = 0
    if eval_rows:
        eval_file = output_file.with_name(output_file.stem + "_eval.jsonl")
        eval_count = write_jsonl(eval_file, eval_rows)

    summary = {
        "created_for": "V1 mixed SFT from Qwen2.5 base",
        "seed": args.seed,
        "output_file": str(output_file),
        "eval_file": str(eval_file) if eval_file else None,
        "train_rows": train_count,
        "eval_rows": eval_count,
        "total_rows_before_split": len(all_rows),
        "dropped_duplicates": dropped_duplicates,
        "source_stats": source_stats,
        "final_source_counts": dict(Counter(row.get("meta", {}).get("v1_source", "unknown") for row in train_rows)),
        "final_mode_counts": dict(Counter(str(row.get("meta", {}).get("mode", "none")) for row in train_rows)),
        "final_top_actions": dict(Counter(action_from_row(row) for row in train_rows).most_common(30)),
    }
    write_json(summary_file, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

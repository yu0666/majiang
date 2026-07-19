"""Resume-safe wrapper for H2 learned-gate rollout collection.

The underlying collector writes one output file from scratch.  This wrapper
runs it in small chunks and appends successful chunks into the final training
file, so interrupted H2 runs can continue without losing collected states.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[1]


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def append_jsonl(source: Path, target: Path, limit: int) -> int:
    appended = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, target.open("a", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            json.loads(line)
            dst.write(line)
            appended += 1
            if appended >= limit:
                break
    return appended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-states", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-games-per-chunk", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2026071701)
    parser.add_argument("--rollout-seed", type=int, default=817100)
    parser.add_argument("--rollouts-per-mode", type=int, default=4)
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument(
        "--defender-threat-model",
        choices=["mc", "discard_tell", "blend", "learned"],
        default="blend",
    )
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
    parser.add_argument(
        "--defender-learned-model-path",
        default="Defender_danger_model/danger_model.pth",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "gate_rollout_states.jsonl"
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    collected = count_lines(final_path)
    chunk_index = max(
        len(list(chunks_dir.glob("chunk_*"))),
        (collected + args.chunk_size - 1) // args.chunk_size,
    )
    print(f"[Gate chunked] resume collected={collected}/{args.target_states}", flush=True)

    while collected < args.target_states:
        remaining = args.target_states - collected
        chunk_target = min(args.chunk_size, remaining)
        chunk_dir = chunks_dir / f"chunk_{chunk_index:04d}"
        chunk_output = chunk_dir / "gate_rollout_states.jsonl"
        chunk_summary = chunk_dir / "gate_rollout_summary.json"

        if count_lines(chunk_output) < chunk_target or not chunk_summary.is_file():
            command = [
                sys.executable,
                "collect_gate_rollouts.py",
                "--output-dir",
                str(chunk_dir),
                "--max-states",
                str(chunk_target),
                "--max-games",
                str(args.max_games_per_chunk),
                "--seed",
                str(args.seed + chunk_index * 10000),
                "--rollout-seed",
                str(args.rollout_seed + chunk_index * 100000),
                "--rollouts-per-mode",
                str(args.rollouts_per_mode),
                "--threat-fold-threshold",
                str(args.threat_fold_threshold),
                "--defender-threat-model",
                args.defender_threat_model,
                "--defender-tell-weight",
                str(args.defender_tell_weight),
                "--defender-tell-window",
                str(args.defender_tell_window),
                "--defender-learned-model-path",
                args.defender_learned_model_path,
                "--model-path",
                args.model_path,
                "--adapter-path",
                args.adapter_path,
                "--max-new-tokens",
                str(args.max_new_tokens),
            ]
            print(
                f"[Gate chunked] start chunk={chunk_index} "
                f"target={chunk_target} collected={collected}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_DIR, check=True)

        appended = append_jsonl(chunk_output, final_path, chunk_target)
        if appended <= 0:
            raise RuntimeError(f"Chunk produced no rows: {chunk_output}")
        collected += appended
        print(
            f"[Gate chunked] done chunk={chunk_index} appended={appended} "
            f"collected={collected}/{args.target_states}",
            flush=True,
        )
        chunk_index += 1

    summary = {
        "states": collected,
        "target_states": args.target_states,
        "chunk_size": args.chunk_size,
        "rollouts_per_mode": args.rollouts_per_mode,
        "defender_threat_model": args.defender_threat_model,
        "defender_learned_model_path": args.defender_learned_model_path,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "output": str(final_path),
    }
    with (output_dir / "gate_rollout_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""L0 / L1 / L2_continuous comparison against the distilled neural opponent,
using the real local_qwen LLM backend (not heuristic_fallback).

Mirrors h2_full_learned_route/evaluate_h2_neural_fast.py's subprocess +
ThreadPoolExecutor pattern (one run_gate1_experiments.py subprocess per
(variant, seed), safe to run concurrently since each is a separate process
with its own model load). L2_continuous uses --mask-gate-policy continuous
(the Chapter-5/6 alpha=f(u,rho) gate) instead of the discrete rule/learned
gate chain used by L2_retrained_gate/L2_value_gate in that script.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


METHOD = {
    "L0": "llm_base", "L1": "llm_reactive_z",
    "L2_continuous": "llm_mask", "L2_continuous_v2": "llm_mask", "L2_continuous_v3": "llm_mask",
    "L2_continuous_v4": "llm_mask", "L2_continuous_v5": "llm_mask", "L2_continuous_v6": "llm_mask",
    "L2_continuous_v7": "llm_mask",
}
DEFAULT_VARIANTS = ["L0", "L1", "L2_continuous"]


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def optional_extend(command: list[str], option: str, value: Optional[str]) -> None:
    if value:
        command.extend([option, value])


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    mode_counts: Dict[str, int] = {"exploit": 0, "safe": 0, "deceive": 0}
    false_folds = 0
    false_fold_opportunities = 0
    induced_dealin = 0
    deceive_windows = 0
    for row in rows:
        for mode in mode_counts:
            mode_counts[mode] += int((row.get("mode_counts") or {}).get(mode, 0))
        counts = row.get("FFR_counts") or {}
        false_folds += int(counts.get("false_folds", 0))
        false_fold_opportunities += int(counts.get("false_fold_opportunities", 0))
        dir_counts = row.get("DIR_counts") or {}
        induced_dealin += int(dir_counts.get("induced_dealin", 0))
        deceive_windows += int(dir_counts.get("deceive_windows", 0))
    return {
        "seeds": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "avg_net": mean(float(row["avg_net"]) for row in rows),
        "hu_rate": mean(float(row["hu_rate"]) for row in rows),
        "dealin_rate": mean(float(row["dealin_rate"]) for row in rows),
        "FFR": false_folds / false_fold_opportunities if false_fold_opportunities else 0.0,
        "FFR_counts": {
            "false_folds": false_folds,
            "false_fold_opportunities": false_fold_opportunities,
            "definition": "neural_proxy when opponent-style=neural",
        },
        "DIR_totals": {"induced_dealin": induced_dealin, "deceive_windows": deceive_windows},
        "mode_counts": mode_counts,
        "latency_ms": {
            key: mean(float(row["decision_latency_ms"][key]) for row in rows)
            for key in ("p50", "p95", "p99")
        },
    }


def compact_row(seed: int, row: Dict[str, Any]) -> Dict[str, Any]:
    counts = row.get("FFR_counts") or {}
    modes = row.get("mode_counts") or {}
    dir_counts = row.get("DIR_counts") or {}
    false_folds = int(counts.get("false_folds", 0))
    false_fold_opportunities = int(counts.get("false_fold_opportunities", 0))
    return {
        "seed": seed,
        "games": int(row["games"]),
        "avg_net": float(row["avg_net"]),
        "hu_rate": float(row["hu_rate"]),
        "dealin_rate": float(row["dealin_rate"]),
        "FFR": false_folds / false_fold_opportunities if false_fold_opportunities else 0.0,
        "false_folds": false_folds,
        "false_fold_opportunities": false_fold_opportunities,
        "exploit": int(modes.get("exploit", 0)),
        "safe": int(modes.get("safe", 0)),
        "deceive": int(modes.get("deceive", 0)),
        "induced_dealin": int(dir_counts.get("induced_dealin", 0)),
        "deceive_windows": int(dir_counts.get("deceive_windows", 0)),
    }


def common_command(args: argparse.Namespace, variant: str, seed: int, output: Path) -> list[str]:
    command = [
        str(args.python),
        "run_gate1_experiments.py",
        "--methods", METHOD[variant],
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", "neural",
        "--neural-opponent-model-path", args.neural_opponent_model_path,
        "--neural-opponent-device", args.neural_opponent_device,
        "--danger-threshold", "1",
        "--ffr-hand-shanten", "1",
        "--sample-every", "0",
        "--mask-log-counterfactual",
        "--snapshot-oracle-samples", str(args.snapshot_oracle_samples),
        "--snapshot-crn-seeds", "1",
        "--backend", "local_qwen",
        "--model-path", args.model_path,
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", "0.0",
        "--output-dir", str(output),
    ]
    optional_extend(command, "--adapter-path", args.adapter_path)
    if variant == "L2_continuous":
        command.extend(["--mask-gate-policy", "continuous"])
    elif variant == "L2_continuous_v2":
        command.extend(["--mask-gate-policy", "continuous_v2"])
    elif variant == "L2_continuous_v3":
        command.extend(["--mask-gate-policy", "continuous_v3"])
    elif variant == "L2_continuous_v4":
        command.extend(["--mask-gate-policy", "continuous_v4"])
    elif variant == "L2_continuous_v5":
        command.extend(["--mask-gate-policy", "continuous_v5"])
    elif variant == "L2_continuous_v6":
        command.extend(["--mask-gate-policy", "continuous_v6"])
    elif variant == "L2_continuous_v7":
        command.extend(["--mask-gate-policy", "continuous_v7"])
    return command


def load_variant_row(output: Path, method: str) -> Dict[str, Any]:
    summary = json.loads((output / "gate1_summary.json").read_text(encoding="utf-8"))
    return summary["E2_ladder"][method]


def run_one(args: argparse.Namespace, output_dir: Path, variant: str, seed: int) -> tuple[str, int, Dict[str, Any]]:
    output = output_dir / variant / f"seed_{seed}"
    summary_path = output / "gate1_summary.json"
    if args.force or not summary_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        command = common_command(args, variant, seed, output)
        print(f"[Neural LLM eval] start variant={variant} seed={seed}", flush=True)
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    row = load_variant_row(output, METHOD[variant])
    print(f"[Neural LLM eval] done variant={variant} seed={seed} net={row['avg_net']:.3f}", flush=True)
    return variant, seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--neural-opponent-model-path", default="Neural_opponent_model/neural_opponent_policy.pth")
    parser.add_argument("--neural-opponent-device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260627, 20261627, 20262627])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--snapshot-oracle-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--variants", nargs="+", choices=list(METHOD), default=DEFAULT_VARIANTS)
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_variant: Dict[str, list[Dict[str, Any]]] = {variant: [] for variant in args.variants}
    jobs = [(variant, seed) for variant in args.variants for seed in args.seeds]
    if args.parallel_workers <= 1:
        completed = [run_one(args, output_dir, variant, seed) for variant, seed in jobs]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
            futures = [pool.submit(run_one, args, output_dir, variant, seed) for variant, seed in jobs]
            for future in as_completed(futures):
                completed.append(future.result())

    rows_by_key = {(variant, seed): row for variant, seed, row in completed}
    for variant in args.variants:
        for seed in args.seeds:
            per_variant[variant].append(rows_by_key[(variant, seed)])

    result = {
        "config": {
            "seeds": args.seeds,
            "variants": args.variants,
            "games_per_seed_per_variant": args.games,
            "opponent_style": "neural",
            "neural_opponent_model_path": args.neural_opponent_model_path,
            "neural_ffr": "proxy: own_shanten<=1, P0_shanten>1, neural action more conservative than min-shanten push",
            "backend": "local_qwen",
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "gpu": args.gpu,
            "parallel_workers": args.parallel_workers,
        },
        "aggregate": {variant: aggregate(rows) for variant, rows in per_variant.items()},
        "per_seed": {
            variant: [compact_row(seed, row) for seed, row in zip(args.seeds, rows)]
            for variant, rows in per_variant.items()
        },
    }
    pairs = [("L1", "L0"), ("L2_continuous", "L1"), ("L2_continuous", "L0")]
    result["paired_net"] = {
        f"{left}_vs_{right}": paired_stats(
            [float(row["avg_net"]) for row in per_variant[left]],
            [float(row["avg_net"]) for row in per_variant[right]],
        )
        for left, right in pairs
        if left in per_variant and right in per_variant
    }
    summary_path = output_dir / "neural_llm_summary.json"
    write_json(summary_path, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

"""Serial paired evaluation of learned/rule gates and fair reranker variants."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


METHOD = {
    "L0": "llm_base",
    "L1": "llm_reactive_z",
    "L2_rule_off": "llm_mask",
    "L2_learned_off": "llm_mask",
    "L2_learned_direct_v2": "llm_mask",
    "L2_learned_trained": "llm_mask",
}


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    reranker = [row.get("candidate_reranker", {}) for row in rows]
    used = sum(int(item.get("used_states", 0)) for item in reranker)
    changed = sum(int(item.get("changed_actions", 0)) for item in reranker)
    return {
        "seeds": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "avg_net": mean(float(row["avg_net"]) for row in rows),
        "seed_std": statistics.stdev(float(row["avg_net"]) for row in rows) if len(rows) > 1 else 0.0,
        "trimmed_mean_10pct": mean(float(row["net_distribution"]["trimmed_mean_10pct"]) for row in rows),
        "hu_rate": mean(float(row["hu_rate"]) for row in rows),
        "avg_hu_fan": mean(float(row.get("avg_hu_fan", 0.0)) for row in rows),
        "fan_per_game": mean(float(row.get("fan_per_game", 0.0)) for row in rows),
        "dealin_rate": mean(float(row["dealin_rate"]) for row in rows),
        "DIR": mean(float(row["DIR"]) for row in rows),
        "FFR": mean(float(row["FFR"]) for row in rows),
        "reranker": {
            "used_states": used,
            "changed_actions": changed,
            "change_rate": changed / used if used else 0.0,
        },
        "latency_ms": {
            key: mean(float(row["decision_latency_ms"][key]) for row in rows)
            for key in ("p50", "p95", "p99")
        },
    }


def common_command(args: argparse.Namespace, variant: str, seed: int, output: Path) -> list[str]:
    command = [
        str(args.python),
        "run_gate1_experiments.py",
        "--methods", METHOD[variant],
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", "responsive",
        "--threat-fold-threshold", "0.7",
        "--oracle-samples", "30",
        "--oracle-beta", "2.0",
        "--danger-threshold", "1",
        "--ffr-hand-shanten", "1",
        "--defender-threat-model", "blend",
        "--defender-tell-weight", "0.3",
        "--defender-tell-window", "6",
        "--mask-oracle-samples", "30",
        "--mask-oracle-beta", "2.0",
        "--mask-danger-threshold", "1",
        "--mask-dir-ready-threshold", "0",
        "--mask-forced-deceive", "off",
        "--mask-deceive-style", "threat",
        "--mask-threat-max-result-shanten", "0",
        "--mask-threat-max-shanten-regret", "0",
        "--mask-threat-min-ukeire-ratio", "1.0",
        "--mask-threat-gate-threshold", "0.7",
        "--mask-threat-gate-margin", "0.12",
        "--mask-threat-min-delta", "0.03",
        "--mask-threat-gate-mode", "cross",
        "--mask-threat-response-model", "blend",
        "--mask-threat-response-tell-weight", "0.3",
        "--mask-threat-tell-window", "6",
        "--mask-threat-max-start-shanten", "2",
        "--mask-threat-require-real-target",
        "--mask-threat-target-max-shanten", "1",
        "--mask-threat-target-signal", "mc",
        "--mask-threat-target-prob-threshold", "0.78",
        "--mask-log-counterfactual",
        "--snapshot-oracle-samples", "120",
        "--snapshot-crn-seeds", "1",
        "--backend", "local_qwen",
        "--model-path", args.model_path,
        "--adapter-path", args.adapter_path,
        "--max-new-tokens", "64",
        "--output-dir", str(output),
    ]
    if variant.startswith("L2_learned"):
        command.extend([
            "--mask-gate-policy", "learned",
            "--gate-model-path", args.gate_model_path,
            "--gate-adapter-path", args.gate_adapter_path,
            "--gate-max-new-tokens", "8",
        ])
    if variant in {"L2_learned_direct_v2", "L2_learned_trained"}:
        command.extend([
            "--mask-candidate-reranker",
            "--mask-candidate-scoring",
            "--mask-reranker-max-candidates", "6",
        ])
    if variant == "L2_learned_trained":
        command.extend([
            "--reranker-model-path", args.reranker_model_path,
            "--reranker-adapter-path", args.reranker_adapter_path,
            "--reranker-max-new-tokens", "16",
        ])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--gate-model-path", required=True)
    parser.add_argument("--gate-adapter-path", required=True)
    parser.add_argument("--reranker-model-path", required=True)
    parser.add_argument("--reranker-adapter-path", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026071601, 2026072601, 2026073601])
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_variant: Dict[str, list[Dict[str, Any]]] = {variant: [] for variant in METHOD}
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    for variant in METHOD:
        for seed in args.seeds:
            output = args.output_dir / variant / f"seed_{seed}"
            summary_path = output / "gate1_summary.json"
            if args.force or not summary_path.is_file():
                output.mkdir(parents=True, exist_ok=True)
                log_path = output / "run.log"
                command = common_command(args, variant, seed, output)
                print(f"[Full comparison] start {variant} seed={seed}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row = summary["E2_ladder"][METHOD[variant]]
            per_variant[variant].append(row)
            print(f"[Full comparison] done {variant} seed={seed} net={row['avg_net']:.3f}", flush=True)

    pairs = [
        ("L2_rule_off", "L1"),
        ("L2_learned_off", "L2_rule_off"),
        ("L2_learned_direct_v2", "L2_learned_off"),
        ("L2_learned_trained", "L2_learned_off"),
        ("L2_learned_trained", "L2_learned_direct_v2"),
    ]
    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_variant": args.games,
            "serial": True,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "gate_model_path": args.gate_model_path,
            "gate_adapter_path": args.gate_adapter_path,
            "reranker_model_path": args.reranker_model_path,
            "reranker_adapter_path": args.reranker_adapter_path,
        },
        "aggregate": {variant: aggregate(rows) for variant, rows in per_variant.items()},
        "paired_net": {
            f"{left}_vs_{right}": paired_stats(
                [float(row["avg_net"]) for row in per_variant[left]],
                [float(row["avg_net"]) for row in per_variant[right]],
            )
            for left, right in pairs
        },
        "per_seed": {
            variant: [dict(seed=seed, **row) for seed, row in zip(args.seeds, rows)]
            for variant, rows in per_variant.items()
        },
    }
    output = args.output_dir / "full_comparison_summary.json"
    write_json(output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

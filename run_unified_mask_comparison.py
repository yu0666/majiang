"""Run the fair gate and reranker comparison in isolated serial subprocesses."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


VARIANTS = {
    "L0": {"method": "llm_base", "gate": "rule", "reranker": "off"},
    "L1": {"method": "llm_reactive_z", "gate": "rule", "reranker": "off"},
    "L2_rule_off": {"method": "llm_mask", "gate": "rule", "reranker": "off"},
    "L2_learned_off": {"method": "llm_mask", "gate": "learned", "reranker": "off"},
    "L2_learned_direct": {"method": "llm_mask", "gate": "learned", "reranker": "direct"},
    "L2_learned_trained": {"method": "llm_mask", "gate": "learned", "reranker": "trained"},
}


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def command_for(args, variant: Dict[str, str], seed: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "run_gate1_experiments.py",
        "--methods", variant["method"],
        "--games", str(args.games),
        "--seed", str(seed),
        "--opponent-style", "responsive",
        "--sample-every", "0",
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
    if variant["gate"] == "learned":
        command.extend([
            "--mask-gate-policy", "learned",
            "--gate-model-path", args.gate_model_path,
            "--gate-adapter-path", args.gate_adapter_path,
            "--gate-max-new-tokens", "8",
        ])
    if variant["reranker"] != "off":
        command.extend([
            "--mask-candidate-reranker",
            "--mask-candidate-scoring",
            "--mask-reranker-max-candidates", "6",
        ])
    if variant["reranker"] == "trained":
        command.extend([
            "--reranker-model-path", args.reranker_model_path,
            "--reranker-adapter-path", args.reranker_adapter_path,
            "--reranker-max-new-tokens", "16",
        ])
    return command


def load_method_summary(path: Path, method: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["E2_ladder"][method]


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    rerank = [row.get("candidate_reranker", {}) for row in rows]
    used = sum(int(item.get("used_states", 0)) for item in rerank)
    changed = sum(int(item.get("changed_actions", 0)) for item in rerank)
    gate_used = sum(int(row.get("learned_gate", {}).get("used_states", 0)) for row in rows)
    gate_parsed = sum(int(row.get("learned_gate", {}).get("parsed_outputs", 0)) for row in rows)
    return {
        "seeds": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "avg_net": mean(float(row["avg_net"]) for row in rows),
        "avg_net_seed_std": statistics.stdev(float(row["avg_net"]) for row in rows),
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
            "action_change_rate": changed / used if used else 0.0,
        },
        "learned_gate": {
            "used_states": gate_used,
            "parse_rate": gate_parsed / gate_used if gate_used else 0.0,
            "mode_counts": {
                mode: sum(
                    int(row.get("learned_gate", {}).get("mode_counts", {}).get(mode, 0))
                    for row in rows
                )
                for mode in ("exploit", "safe", "deceive")
            },
        },
        "latency_ms": {
            key: mean(float(row["decision_latency_ms"][key]) for row in rows)
            for key in ("p50", "p95", "p99")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026082101, 2026083101, 2026084101])
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--gate-model-path", required=True)
    parser.add_argument("--gate-adapter-path", required=True)
    parser.add_argument("--reranker-model-path", required=True)
    parser.add_argument("--reranker-adapter-path", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_variant: Dict[str, list[Dict[str, Any]]] = {name: [] for name in VARIANTS}
    for name, variant in VARIANTS.items():
        for seed in args.seeds:
            output = args.output_dir / name / f"seed_{seed}"
            summary_path = output / "gate1_summary.json"
            command = command_for(args, variant, seed, output)
            if args.dry_run:
                print(" ".join(command))
                continue
            if args.force or not summary_path.is_file():
                output.mkdir(parents=True, exist_ok=True)
                with (output / "run.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=Path(__file__).resolve().parent,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            row = load_method_summary(summary_path, variant["method"])
            per_variant[name].append(row)
            print(f"[{name}] seed={seed} net={row['avg_net']:.3f}", flush=True)
    if args.dry_run:
        return

    pairs = [
        ("L2_rule_off", "L1"),
        ("L2_learned_off", "L2_rule_off"),
        ("L2_learned_direct", "L2_learned_off"),
        ("L2_learned_trained", "L2_learned_off"),
        ("L2_learned_trained", "L2_learned_direct"),
    ]
    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_variant": args.games,
            "serial_subprocesses": True,
            "same_exploit_baseline": True,
        },
        "aggregate": {name: aggregate(rows) for name, rows in per_variant.items()},
        "paired_net": {
            f"{left}_vs_{right}": paired_stats(
                [float(row["avg_net"]) for row in per_variant[left]],
                [float(row["avg_net"]) for row in per_variant[right]],
            )
            for left, right in pairs
        },
        "per_seed": {
            name: [dict(seed=seed, **row) for seed, row in zip(args.seeds, rows)]
            for name, rows in per_variant.items()
        },
    }
    output = args.output_dir / "unified_mask_comparison.json"
    write_json(output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

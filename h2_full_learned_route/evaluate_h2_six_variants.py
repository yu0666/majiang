"""Serial H2 evaluation against the learned defender.

Variants:
  L0 = base policy, no opponent modeling
  L1 = base policy + reactive z
  L2_rule_gate = MASK with rule gate
  L2_learned_gate = MASK with the previous learned gate
  L2_retrained_gate = MASK with the gate retrained on this route
  L2_retrained_gate_reranker = retrained gate + dedicated candidate reranker
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

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiment_trace import write_json
from rerun_v2_e2_ladder_3seeds import paired_stats


METHOD = {
    "L0": "llm_base",
    "L1": "llm_reactive_z",
    "L2_rule_gate": "llm_mask",
    "L2_learned_gate": "llm_mask",
    "L2_retrained_gate": "llm_mask",
    "L2_retrained_gate_reranker": "llm_mask",
}


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def optional_extend(command: list[str], option: str, value: Optional[str]) -> None:
    if value:
        command.extend([option, value])


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    reranker_rows = [row.get("candidate_reranker", {}) for row in rows]
    gate_rows = [row.get("learned_gate", {}) for row in rows]
    reranker_used = sum(int(item.get("used_states", 0)) for item in reranker_rows)
    reranker_changed = sum(int(item.get("changed_actions", 0)) for item in reranker_rows)
    reranker_parsed = sum(int(item.get("parsed_outputs", 0)) for item in reranker_rows)
    gate_used = sum(int(item.get("used_states", 0)) for item in gate_rows)
    gate_parsed = sum(int(item.get("parsed_outputs", 0)) for item in gate_rows)
    gate_modes: Dict[str, int] = {}
    mode_counts: Dict[str, int] = {}
    for row in rows:
        for mode, count in (row.get("mode_counts") or {}).items():
            mode_counts[mode] = mode_counts.get(mode, 0) + int(count)
        for mode, count in (row.get("learned_gate", {}).get("mode_counts") or {}).items():
            gate_modes[mode] = gate_modes.get(mode, 0) + int(count)
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
        "mode_counts": mode_counts,
        "learned_gate": {
            "used_states": gate_used,
            "parsed_outputs": gate_parsed,
            "parse_rate": gate_parsed / gate_used if gate_used else 0.0,
            "mode_counts": gate_modes,
        },
        "candidate_reranker": {
            "used_states": reranker_used,
            "changed_actions": reranker_changed,
            "parsed_outputs": reranker_parsed,
            "change_rate_when_used": reranker_changed / reranker_used if reranker_used else 0.0,
            "parse_rate_when_used": reranker_parsed / reranker_used if reranker_used else 0.0,
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
        "--defender-threat-model", "learned",
        "--defender-learned-model-path", args.defender_learned_model_path,
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
        "--max-new-tokens", "64",
        "--temperature", "0.0",
        "--output-dir", str(output),
    ]
    optional_extend(command, "--adapter-path", args.adapter_path)
    if variant == "L2_learned_gate":
        command.extend([
            "--mask-gate-policy", "learned",
            "--gate-model-path", args.old_gate_model_path,
            "--gate-adapter-path", args.old_gate_adapter_path,
            "--gate-max-new-tokens", "8",
        ])
    if variant in {"L2_retrained_gate", "L2_retrained_gate_reranker"}:
        command.extend([
            "--mask-gate-policy", "learned",
            "--gate-model-path", args.gate_model_path,
            "--gate-adapter-path", args.gate_adapter_path,
            "--gate-max-new-tokens", "8",
        ])
    if variant == "L2_retrained_gate_reranker":
        command.extend([
            "--mask-candidate-reranker",
            "--mask-candidate-scoring",
            "--mask-reranker-max-candidates", "6",
            "--reranker-model-path", args.reranker_model_path,
            "--reranker-adapter-path", args.reranker_adapter_path,
            "--reranker-max-new-tokens", "16",
        ])
    return command


def load_variant_row(output: Path, method: str) -> Dict[str, Any]:
    summary_path = output / "gate1_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary["E2_ladder"][method]


def run_one(args: argparse.Namespace, output_dir: Path, variant: str, seed: int) -> tuple[str, int, Dict[str, Any]]:
    output = output_dir / variant / f"seed_{seed}"
    summary_path = output / "gate1_summary.json"
    if args.force or not summary_path.is_file():
        output.mkdir(parents=True, exist_ok=True)
        command = common_command(args, variant, seed, output)
        print(f"[H2 learned eval] start variant={variant} seed={seed}", flush=True)
        env = dict(os.environ)
        env.setdefault("PYTHONHASHSEED", "0")
        with (output / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=REPO_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    row = load_variant_row(output, METHOD[variant])
    print(f"[H2 learned eval] done variant={variant} seed={seed} net={row['avg_net']:.3f}", flush=True)
    return variant, seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("py10/bin/python3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--old-gate-model-path", required=True)
    parser.add_argument("--old-gate-adapter-path", required=True)
    parser.add_argument("--gate-model-path", required=True)
    parser.add_argument("--gate-adapter-path", required=True)
    parser.add_argument("--reranker-model-path", required=True)
    parser.add_argument("--reranker-adapter-path", required=True)
    parser.add_argument("--defender-learned-model-path", default="Defender_danger_model/danger_model.pth")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026071601, 2026072601, 2026073601])
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of variant/seed jobs to run concurrently. Keep small because each job loads the model.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_variant: Dict[str, list[Dict[str, Any]]] = {variant: [] for variant in METHOD}
    jobs = [(variant, seed) for variant in METHOD for seed in args.seeds]
    if args.parallel_workers <= 1:
        completed = [run_one(args, output_dir, variant, seed) for variant, seed in jobs]
    else:
        completed = []
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
            futures = [pool.submit(run_one, args, output_dir, variant, seed) for variant, seed in jobs]
            for future in as_completed(futures):
                completed.append(future.result())
    rows_by_key = {(variant, seed): row for variant, seed, row in completed}
    for variant in METHOD:
        for seed in args.seeds:
            per_variant[variant].append(rows_by_key[(variant, seed)])

    pairs = [
        ("L1", "L0"),
        ("L2_rule_gate", "L1"),
        ("L2_learned_gate", "L2_rule_gate"),
        ("L2_retrained_gate", "L2_learned_gate"),
        ("L2_retrained_gate_reranker", "L2_retrained_gate"),
        ("L2_retrained_gate", "L0"),
        ("L2_retrained_gate_reranker", "L0"),
    ]
    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_variant": args.games,
            "opponent_style": "responsive",
            "defender_threat_model": "learned",
            "defender_learned_model_path": args.defender_learned_model_path,
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "old_gate_model_path": args.old_gate_model_path,
            "old_gate_adapter_path": args.old_gate_adapter_path,
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
    summary_path = output_dir / "h2_six_variant_learned_defender_summary.json"
    write_json(summary_path, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()

"""Evaluate L0/L1/L2-rule/L2-learned for the H2 gate route.

This script is intentionally standalone and only calls existing project entry
points in subprocesses. It does not modify existing code.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from experiment_trace import sign_test_p_value, write_json  # noqa: E402


VARIANTS = {
    "L0": {"method": "llm_base", "gate": "rule"},
    "L1": {"method": "llm_reactive_z", "gate": "rule"},
    "L2_rule_gate": {"method": "llm_mask", "gate": "rule"},
    "L2_learned_gate": {"method": "llm_mask", "gate": "learned"},
}


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def two_tailed_t_pvalue(t_value: float, df: int) -> Optional[float]:
    try:
        from scipy import stats as scipy_stats

        return float(2.0 * scipy_stats.t.sf(abs(t_value), df))
    except Exception:
        return None


def paired_stats(xs: list[float], ys: list[float]) -> Dict[str, Any]:
    diffs = [x - y for x, y in zip(xs, ys)]
    n = len(diffs)
    mean_diff = statistics.mean(diffs) if diffs else None
    sd_diff = statistics.stdev(diffs) if n > 1 else None
    t_value = None
    p_value = None
    if n > 1 and sd_diff and sd_diff > 0:
        t_value = mean_diff / (sd_diff / math.sqrt(n))
        p_value = two_tailed_t_pvalue(t_value, n - 1)
    return {
        "n": n,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "t": t_value,
        "df": n - 1 if n else None,
        "t_p_value": p_value,
        "sign_test_p": sign_test_p_value(diffs),
    }


def command_for(args: argparse.Namespace, variant: Dict[str, str], seed: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        "run_gate1_experiments.py",
        "--methods",
        variant["method"],
        "--games",
        str(args.games),
        "--seed",
        str(seed),
        "--opponent-style",
        "responsive",
        "--sample-every",
        "0",
        "--threat-fold-threshold",
        str(args.threat_fold_threshold),
        "--oracle-samples",
        "30",
        "--oracle-beta",
        "2.0",
        "--danger-threshold",
        "1",
        "--ffr-hand-shanten",
        "1",
        "--defender-threat-model",
        "blend",
        "--defender-tell-weight",
        str(args.defender_tell_weight),
        "--defender-tell-window",
        str(args.defender_tell_window),
        "--mask-oracle-samples",
        "30",
        "--mask-oracle-beta",
        "2.0",
        "--mask-danger-threshold",
        "1",
        "--mask-dir-ready-threshold",
        "0",
        "--mask-forced-deceive",
        "off",
        "--mask-deceive-style",
        "threat",
        "--mask-threat-max-result-shanten",
        "0",
        "--mask-threat-max-shanten-regret",
        "0",
        "--mask-threat-min-ukeire-ratio",
        "1.0",
        "--mask-threat-gate-threshold",
        str(args.threat_fold_threshold),
        "--mask-threat-gate-margin",
        "0.12",
        "--mask-threat-min-delta",
        "0.03",
        "--mask-threat-gate-mode",
        "cross",
        "--mask-threat-response-model",
        "blend",
        "--mask-threat-response-tell-weight",
        str(args.defender_tell_weight),
        "--mask-threat-tell-window",
        str(args.defender_tell_window),
        "--mask-threat-max-start-shanten",
        "2",
        "--mask-threat-require-real-target",
        "--mask-threat-target-max-shanten",
        "1",
        "--mask-threat-target-signal",
        "mc",
        "--mask-threat-target-prob-threshold",
        "0.78",
        "--mask-log-counterfactual",
        "--snapshot-oracle-samples",
        "120",
        "--snapshot-crn-seeds",
        "1",
        "--backend",
        "local_qwen",
        "--model-path",
        args.policy_model,
        "--adapter-path",
        args.policy_adapter,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-dir",
        str(output),
    ]
    if variant["gate"] == "learned":
        command.extend(
            [
                "--mask-gate-policy",
                "learned",
                "--gate-model-path",
                args.gate_model,
                "--gate-adapter-path",
                args.gate_adapter,
                "--gate-max-new-tokens",
                str(args.gate_max_new_tokens),
            ]
        )
    return command


def load_summary(path: Path, method: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["E2_ladder"][method]


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    gate_used = sum(int(row.get("learned_gate", {}).get("used_states", 0)) for row in rows)
    gate_parsed = sum(int(row.get("learned_gate", {}).get("parsed_outputs", 0)) for row in rows)
    nets = [float(row["avg_net"]) for row in rows]
    return {
        "seeds": len(rows),
        "games": sum(int(row["games"]) for row in rows),
        "avg_net": mean(nets),
        "avg_net_seed_std": statistics.stdev(nets) if len(nets) > 1 else 0.0,
        "trimmed_mean_10pct": mean(float(row["net_distribution"]["trimmed_mean_10pct"]) for row in rows),
        "hu_rate": mean(float(row["hu_rate"]) for row in rows),
        "dealin_rate": mean(float(row["dealin_rate"]) for row in rows),
        "DIR": mean(float(row["DIR"]) for row in rows),
        "FFR": mean(float(row["FFR"]) for row in rows),
        "mode_counts": {
            mode: sum(int((row.get("mode_counts") or {}).get(mode, 0)) for row in rows)
            for mode in ("exploit", "safe", "deceive")
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026071701, 2026072701, 2026073701])
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--policy-adapter", required=True)
    parser.add_argument("--gate-model", required=True)
    parser.add_argument("--gate-adapter", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gate-max-new-tokens", type=int, default=8)
    parser.add_argument("--threat-fold-threshold", type=float, default=0.7)
    parser.add_argument("--defender-tell-weight", type=float, default=0.3)
    parser.add_argument("--defender-tell-window", type=int, default=6)
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
                print(f"[H2 gate eval] start variant={name} seed={seed}", flush=True)
                with (output / "run.log").open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=REPO_DIR,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            row = load_summary(summary_path, variant["method"])
            per_variant[name].append(row)
            print(f"[H2 gate eval] done variant={name} seed={seed} net={row['avg_net']:.3f}", flush=True)
    if args.dry_run:
        return

    pairs = [
        ("L1", "L0"),
        ("L2_rule_gate", "L1"),
        ("L2_learned_gate", "L2_rule_gate"),
        ("L2_learned_gate", "L1"),
    ]
    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_variant": args.games,
            "opponent": "responsive_blend",
            "policy_model": args.policy_model,
            "policy_adapter": args.policy_adapter,
            "gate_model": args.gate_model,
            "gate_adapter": args.gate_adapter,
            "reranker": "off",
            "isolation": "same policy/action backend; only gate differs between L2 variants",
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
    output = args.output_dir / "h2_gate_ladder_summary.json"
    write_json(output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()


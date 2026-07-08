"""
Multi-seed aggregation for the H1 gate.

The balanced-eval AUC sits near the 0.75 floor and wobbles seed-to-seed, so a
single run is not enough to claim a pass.  This runs the H1 experiment across
N seeds and reports mean +/- 95% CI for AUC and the Brier reductions, plus a
"robust pass" verdict (the AUC CI lower bound clears the threshold and every
seed clears Brier + significance).  Brier/p are the strong, stable signals; AUC
is the marginal one this exists to pin down.

Example (CPU, MC B_phi, danger target):
  python aggregate_h1_seeds.py --num-seeds 5 --games 200 \
      --b-phi-source mc --backend heuristic_fallback --danger-threshold 1
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_trace import ensure_deterministic_hashing, write_json
from run_h1_belief_experiment import run


# Two-sided 95% t critical values by degrees of freedom (n-1); >15 -> ~normal.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
        14: 2.145, 15: 2.131}


def t_critical(n: int) -> float:
    df = max(1, n - 1)
    return _T95.get(df, 1.96)


def mean_ci(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = statistics.mean(values) if n else 0.0
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci95_half": 0.0, "lo": mean, "hi": mean, "n": n,
                "min": mean, "max": mean}
    std = statistics.stdev(values)
    half = t_critical(n) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_half": half, "lo": mean - half, "hi": mean + half,
            "n": n, "min": min(values), "max": max(values)}


def seed_args(base: argparse.Namespace, seed: int, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        games=base.games, seed=seed, opponent_style=base.opponent_style,
        max_steps=base.max_steps, sample_every=base.sample_every,
        train_ratio=base.train_ratio, oracle_samples=base.oracle_samples,
        oracle_beta=base.oracle_beta, danger_threshold=base.danger_threshold,
        b_phi_source=base.b_phi_source, auc_threshold=base.auc_threshold,
        brier_reduction=base.brier_reduction, p_threshold=base.p_threshold,
        backend=base.backend, model_path=base.model_path, adapter_path=base.adapter_path,
        max_new_tokens=base.max_new_tokens, output_dir=out_dir,
    )


def main() -> None:
    ensure_deterministic_hashing()
    p = argparse.ArgumentParser()
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=20260627)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--opponent-style", default="mixed")
    p.add_argument("--max-steps", type=int, default=260)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--train-ratio", type=float, default=0.4)
    p.add_argument("--oracle-samples", type=int, default=60)
    p.add_argument("--oracle-beta", type=float, default=2.0)
    p.add_argument("--danger-threshold", type=int, default=1)
    p.add_argument("--b-phi-source", default="mc", choices=["llm", "mc"])
    p.add_argument("--backend", default="heuristic_fallback", choices=["heuristic_fallback", "local_qwen"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--adapter-path", default=None)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--auc-threshold", type=float, default=0.75)
    p.add_argument("--brier-reduction", type=float, default=0.20)
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=Path("H1_seed_aggregate"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []
    for k in range(args.num_seeds):
        seed = args.base_seed + k * 101  # spread seeds apart
        sa = seed_args(args, seed, args.output_dir / f"seed_{seed}")
        summary = run(sa)
        g = summary["H1_gate"]
        row = {
            "seed": seed,
            "balanced_n": summary["samples_balanced_eval"],
            "underpowered": g.get("underpowered"),
            "auc": g.get("B2_auc"),
            "brier_red_b0": g.get("B2_vs_B0_relative_brier_reduction"),
            "brier_red_b1": g.get("B2_vs_B1_relative_brier_reduction"),
            "p_b0": g.get("paired_test_vs_B0", {}).get("sign_test_p"),
            "p_b1": g.get("paired_test_vs_B1", {}).get("sign_test_p"),
            "pass": g.get("pass"),
        }
        per_seed.append(row)
        print(f"seed {seed}: n={row['balanced_n']} AUC={row['auc']:.3f} "
              f"brier_red={row['brier_red_b0']:.2f}/{row['brier_red_b1']:.2f} "
              f"p={row['p_b0']:.1e}/{row['p_b1']:.1e} pass={row['pass']}")

    auc = mean_ci([r["auc"] for r in per_seed])
    red0 = mean_ci([r["brier_red_b0"] for r in per_seed])
    red1 = mean_ci([r["brier_red_b1"] for r in per_seed])
    n_pass = sum(1 for r in per_seed if r["pass"])
    all_sig = all((r["p_b0"] is not None and r["p_b0"] < args.p_threshold
                   and r["p_b1"] is not None and r["p_b1"] < args.p_threshold) for r in per_seed)
    robust = {
        "auc_ci_lo_clears_threshold": auc["lo"] >= args.auc_threshold,
        "brier_red_b0_ci_lo_clears": red0["lo"] >= args.brier_reduction,
        "brier_red_b1_ci_lo_clears": red1["lo"] >= args.brier_reduction,
        "all_seeds_significant": all_sig,
        "seeds_passing": f"{n_pass}/{len(per_seed)}",
    }
    robust["robust_pass"] = (robust["auc_ci_lo_clears_threshold"]
                             and robust["brier_red_b0_ci_lo_clears"]
                             and robust["brier_red_b1_ci_lo_clears"]
                             and all_sig)

    out = {
        "config": {k: getattr(args, k) for k in (
            "num_seeds", "base_seed", "games", "sample_every", "train_ratio",
            "oracle_samples", "oracle_beta", "danger_threshold", "b_phi_source",
            "backend", "auc_threshold", "brier_reduction", "p_threshold")},
        "per_seed": per_seed,
        "aggregate": {"AUC": auc, "brier_reduction_vs_B0": red0, "brier_reduction_vs_B1": red1},
        "robustness": robust,
        "paper_row": (
            f"AUC {auc['mean']:.3f} +/- {auc['ci95_half']:.3f} (95% CI [{auc['lo']:.3f}, {auc['hi']:.3f}]); "
            f"Brier red vs B0 {red0['mean']:.2f} +/- {red0['ci95_half']:.2f}; "
            f"vs B1 {red1['mean']:.2f} +/- {red1['ci95_half']:.2f}; "
            f"all seeds p<{args.p_threshold}: {all_sig}; seeds passing {n_pass}/{len(per_seed)}"
        ),
    }
    write_json(args.output_dir / "h1_seed_aggregate.json", out)
    print("\n=== H1 multi-seed aggregate ===")
    print(out["paper_row"])
    print("robust_pass:", robust["robust_pass"], "|", json.dumps(robust, ensure_ascii=False))
    print(f"\nSaved: {args.output_dir / 'h1_seed_aggregate.json'}")


if __name__ == "__main__":
    main()

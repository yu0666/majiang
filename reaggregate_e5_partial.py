"""Reaggregate the E5 ladder over only the seeds that fully completed before the
13-seed run was stopped early. Reads each seed's already-saved gate1_summary.json
from E5_seed_aggregate/seed_<seed>_{baseline,learned}/ instead of re-running run()
(avoids re-executing ~7h of already-completed work). Mirrors aggregate_e5_seeds.py's
main() computation exactly, just sourced from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from aggregate_h1_seeds import mean_ci
from experiment_trace import sign_test_p_value, write_json

COMPLETED_SEEDS = [
    20260627, 20260728, 20260829, 20260930,
    20261031, 20261132, 20261233, 20261334,
]
OUT_DIR = Path("E5_seed_aggregate")
P_THRESHOLD = 0.05


def pairwise_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    pw = summary["Gate1_pairwise"]
    return {
        "paired_seeds": pw.get("paired_seeds", 0),
        "avg_delta_net": pw.get("avg_delta_net"),
        "avg_delta_dealin_rate": pw.get("avg_delta_dealin_rate"),
        "net_positive_rate": pw.get("net_positive_rate"),
        "net_sign_test_p": pw.get("net_sign_test_p"),
    }


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    per_seed: List[Dict[str, Any]] = []
    for seed in COMPLETED_SEEDS:
        baseline_summary = load_summary(OUT_DIR / f"seed_{seed}_baseline" / "gate1_summary.json")
        learned_summary = load_summary(OUT_DIR / f"seed_{seed}_learned" / "gate1_summary.json")

        baseline_pw = pairwise_row(baseline_summary)
        learned_pw = pairwise_row(learned_summary)
        row = {
            "seed": seed,
            "baseline": baseline_pw,
            "learned": learned_pw,
            "learned_minus_baseline_delta_net": (
                (learned_pw["avg_delta_net"] - baseline_pw["avg_delta_net"])
                if learned_pw["avg_delta_net"] is not None and baseline_pw["avg_delta_net"] is not None
                else None
            ),
        }
        per_seed.append(row)
        print(f"seed {seed}: baseline avg_delta_net={baseline_pw['avg_delta_net']} "
              f"(p={baseline_pw['net_sign_test_p']}) | "
              f"learned avg_delta_net={learned_pw['avg_delta_net']} "
              f"(p={learned_pw['net_sign_test_p']})")

    baseline_deltas = [r["baseline"]["avg_delta_net"] for r in per_seed if r["baseline"]["avg_delta_net"] is not None]
    learned_deltas = [r["learned"]["avg_delta_net"] for r in per_seed if r["learned"]["avg_delta_net"] is not None]
    paired_diffs = [r["learned_minus_baseline_delta_net"] for r in per_seed
                    if r["learned_minus_baseline_delta_net"] is not None]

    baseline_agg = mean_ci(baseline_deltas)
    learned_agg = mean_ci(learned_deltas)
    paired_agg = mean_ci(paired_diffs)
    paired_sign_p = sign_test_p_value(paired_diffs)

    n_learned_positive = sum(1 for d in learned_deltas if d > 0)
    robustness = {
        "learned_net_benefit_ci_excludes_zero_below": learned_agg["lo"] > 0,
        "seeds_with_positive_learned_net_benefit": f"{n_learned_positive}/{len(learned_deltas)}",
        "paired_diff_ci_excludes_zero": (paired_agg["lo"] > 0 or paired_agg["hi"] < 0),
        "paired_sign_test_p": paired_sign_p,
        "paired_significant": (paired_sign_p is not None and paired_sign_p < P_THRESHOLD),
    }

    out = {
        "config": {
            "num_seeds": len(COMPLETED_SEEDS), "seeds": COMPLETED_SEEDS, "games_per_seed": 30,
            "backend": "local_qwen", "model_path": "models/Qwen-Mahjong-V4-GRPO-Merged",
            "baseline_defender_threat_model": "discard_tell (tell_weight=1.0)",
            "learned_defender_threat_model": "learned",
            "note": (
                "PARTIAL ladder: original run targeted 13 seeds but was stopped early "
                "after user request to speed things up; this aggregate covers only the "
                "8 seeds that fully completed both baseline and learned arms. A 9th seed "
                "(20261435) has a baseline-only partial result and is excluded (no paired "
                "learned arm)."
            ),
        },
        "per_seed": per_seed,
        "aggregate": {
            "baseline_avg_delta_net": baseline_agg,
            "learned_avg_delta_net": learned_agg,
            "paired_diff_learned_minus_baseline": paired_agg,
        },
        "robustness": robustness,
        "paper_row": (
            f"Baseline (discard_tell) MASK net-benefit {baseline_agg['mean']:.1f} +/- {baseline_agg['ci95_half']:.1f}; "
            f"learned-opponent net-benefit {learned_agg['mean']:.1f} +/- {learned_agg['ci95_half']:.1f} "
            f"(95% CI [{learned_agg['lo']:.1f}, {learned_agg['hi']:.1f}]); "
            f"paired diff {paired_agg['mean']:.1f} +/- {paired_agg['ci95_half']:.1f}, "
            f"sign-test p={paired_sign_p}; seeds with positive learned net-benefit "
            f"{robustness['seeds_with_positive_learned_net_benefit']} (n=8/13 seeds, run stopped early)"
        ),
    }
    write_json(OUT_DIR / "e5_seed_aggregate.json", out)
    print("\n=== E5 multi-seed ladder aggregate (PARTIAL: 8/13 seeds) ===")
    print(out["paper_row"])
    print("robustness:", json.dumps(robustness, ensure_ascii=False))
    print(f"\nSaved: {OUT_DIR / 'e5_seed_aggregate.json'}")


if __name__ == "__main__":
    main()

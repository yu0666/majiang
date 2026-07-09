"""Rerun the July 7 Gate1 scale experiments.

This reproduces the 13-seed, 200-game-per-arm heuristic-fallback setup used for:

1. L2 MASK baseline vs oracle real-target filter vs deployable MC target filter.
2. L0/L1/L2(mc) ladder against the same responsive blend defender.

Outputs are written under July7_gate1_scale_rerun/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

from experiment_trace import ensure_deterministic_hashing, sign_test_p_value, write_json
from run_gate1_experiments import run


SEEDS = [20260627 + i * 1000 for i in range(13)]


def two_tailed_t_pvalue(t_value: float, df: int) -> Optional[float]:
    try:
        from scipy import stats as scipy_stats

        return float(2.0 * scipy_stats.t.sf(abs(t_value), df))
    except Exception:
        return None


def paired_stats(xs: List[float], ys: List[float]) -> Dict[str, Any]:
    diffs = [x - y for x, y in zip(xs, ys)]
    n = len(diffs)
    mean_diff = stats.mean(diffs) if diffs else None
    sd_diff = stats.stdev(diffs) if n > 1 else None
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


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return stats.mean(values) if values else None


def base_args(methods: List[str], seed: int, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        methods=methods,
        games=200,
        seed=seed,
        opponent_style="responsive",
        max_steps=300,
        sample_every=4,
        threat_fold_threshold=0.4,
        oracle_samples=30,
        oracle_beta=2.0,
        danger_threshold=1,
        ffr_hand_shanten=1,
        mask_oracle_samples=30,
        mask_oracle_beta=2.0,
        mask_danger_threshold=1,
        mask_dir_ready_threshold=0,
        mask_deceive_threat_ceiling=0.5,
        mask_forced_deceive="eligible",
        mask_deceive_style="threat",
        mask_threat_allow_break_ready=False,
        mask_threat_max_result_shanten=0,
        mask_threat_gate_threshold=0.4,
        mask_threat_gate_margin=0.12,
        mask_threat_min_delta=0.03,
        mask_threat_gate_mode="delta_only",
        mask_threat_tell_window=6,
        mask_threat_require_real_target=False,
        mask_threat_target_max_shanten=0,
        mask_threat_target_signal="oracle",
        mask_threat_target_prob_threshold=0.5,
        mask_threat_max_start_shanten=3,
        mask_threat_allow_exploit_overlap=False,
        mask_log_counterfactual=True,
        snapshot_oracle_samples=120,
        snapshot_crn_seeds=1,
        defender_threat_model="blend",
        defender_tell_weight=0.3,
        defender_tell_window=6,
        defender_learned_model_path="Defender_danger_model/danger_model.pth",
        backend="heuristic_fallback",
        model_path=None,
        adapter_path=None,
        belief_adapter_path=None,
        max_new_tokens=128,
        output_dir=output_dir,
    )


def l2_args(kind: str, seed: int, output_dir: Path) -> SimpleNamespace:
    args = base_args(["llm_mask"], seed, output_dir)
    if kind == "oracle":
        args.mask_threat_require_real_target = True
        args.mask_threat_target_max_shanten = 1
        args.mask_threat_target_signal = "oracle"
    elif kind == "mc":
        args.mask_threat_require_real_target = True
        args.mask_threat_target_max_shanten = 1
        args.mask_threat_target_signal = "mc"
        args.mask_threat_target_prob_threshold = 0.78
    elif kind != "baseline":
        raise ValueError(f"unknown L2 kind: {kind}")
    return args


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_or_load(args: SimpleNamespace, force: bool) -> Dict[str, Any]:
    summary_path = Path(args.output_dir) / "gate1_summary.json"
    if summary_path.exists() and not force:
        return load_summary(summary_path)
    return run(args)


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("July7_gate1_scale_rerun"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []

    for seed in SEEDS:
        row: Dict[str, Any] = {"seed": seed}
        for kind in ("baseline", "oracle", "mc"):
            out_dir = args.output_dir / f"{kind}_seed{seed}"
            summary = run_or_load(l2_args(kind, seed, out_dir), args.force)
            row[kind] = summary["E2_ladder"]["llm_mask"]
            print(
                f"seed={seed} {kind}: "
                f"avg_net={row[kind]['avg_net']:.3f} "
                f"hu={row[kind]['hu_rate']:.3f} "
                f"dealin={row[kind]['dealin_rate']:.3f}",
                flush=True,
            )

        l0l1_dir = args.output_dir / f"l0l1_seed{seed}"
        l0l1_summary = run_or_load(base_args(["llm_base", "llm_reactive_z"], seed, l0l1_dir), args.force)
        row["L0"] = l0l1_summary["E2_ladder"]["llm_base"]
        row["L1"] = l0l1_summary["E2_ladder"]["llm_reactive_z"]
        row["L2_mc"] = row["mc"]
        print(
            f"seed={seed} ladder: "
            f"L0={row['L0']['avg_net']:.3f} "
            f"L1={row['L1']['avg_net']:.3f} "
            f"L2_mc={row['L2_mc']['avg_net']:.3f}",
            flush=True,
        )
        per_seed.append(row)

    baseline = [r["baseline"]["avg_net"] for r in per_seed]
    oracle = [r["oracle"]["avg_net"] for r in per_seed]
    mc = [r["mc"]["avg_net"] for r in per_seed]
    l0 = [r["L0"]["avg_net"] for r in per_seed]
    l1 = [r["L1"]["avg_net"] for r in per_seed]

    aggregate = {
        "l2_filter_ablation": {
            "mean_net": {
                "baseline": mean(baseline),
                "oracle": mean(oracle),
                "mc": mean(mc),
            },
            "oracle_vs_baseline": paired_stats(oracle, baseline),
            "mc_vs_baseline": paired_stats(mc, baseline),
            "mc_vs_oracle": paired_stats(mc, oracle),
            "mean_hu": {
                "baseline": mean(r["baseline"]["hu_rate"] for r in per_seed),
                "oracle": mean(r["oracle"]["hu_rate"] for r in per_seed),
                "mc": mean(r["mc"]["hu_rate"] for r in per_seed),
            },
            "mean_dealin": {
                "baseline": mean(r["baseline"]["dealin_rate"] for r in per_seed),
                "oracle": mean(r["oracle"]["dealin_rate"] for r in per_seed),
                "mc": mean(r["mc"]["dealin_rate"] for r in per_seed),
            },
        },
        "l0_l1_l2_ladder": {
            "mean_net": {
                "L0": mean(l0),
                "L1": mean(l1),
                "L2_mc": mean(mc),
            },
            "L1_vs_L0": paired_stats(l1, l0),
            "L2_mc_vs_L0": paired_stats(mc, l0),
            "L2_mc_vs_L1": paired_stats(mc, l1),
            "mean_hu": {
                "L0": mean(r["L0"]["hu_rate"] for r in per_seed),
                "L1": mean(r["L1"]["hu_rate"] for r in per_seed),
                "L2_mc": mean(r["L2_mc"]["hu_rate"] for r in per_seed),
            },
            "mean_dealin": {
                "L0": mean(r["L0"]["dealin_rate"] for r in per_seed),
                "L1": mean(r["L1"]["dealin_rate"] for r in per_seed),
                "L2_mc": mean(r["L2_mc"]["dealin_rate"] for r in per_seed),
            },
        },
    }

    out = {
        "config": {
            "seeds": SEEDS,
            "games_per_seed_per_arm": 200,
            "backend": "heuristic_fallback",
            "opponent_style": "responsive",
            "defender_threat_model": "blend",
            "defender_tell_weight": 0.3,
            "mask_deceive_style": "threat",
            "mask_forced_deceive": "eligible",
            "mask_threat_gate_mode": "delta_only",
            "mc_target_prob_threshold": 0.78,
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    write_json(args.output_dir / "july7_gate1_scale_summary.json", out)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_dir / 'july7_gate1_scale_summary.json'}")


if __name__ == "__main__":
    main()

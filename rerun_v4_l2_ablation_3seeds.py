"""Run V4 L2 ablations to diagnose why MASK hurts the local-Qwen policy.

The baseline V4 E2 ladder showed:

    L1 > L2_MC, even though L2_MC increased FFR.

This runner keeps the same seeds, games, model, and responsive/blend opponent,
then isolates the likely failure causes:

    L1                  loaded from the previous V4 E2 ladder if available
    L2_current_mc        loaded from the previous V4 E2 ladder if available
    L2_no_forced         same MC target filter, but forced_deceive=off
    L2_bphi_risk_only    B_phi/risk prompt only; deceive disabled
    L2_strict_mc         forced_deceive=eligible, but much stricter MC target/gate
    L2_oracle_target     oracle target filter upper bound, same forced gate
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


SEEDS = [20260627, 20261627, 20262627]
MODEL_PATH = "models/Qwen-Mahjong-V4-GRPO-Merged"
PREVIOUS_E2_DIR = Path("V4_E2_ladder_3seeds_200games")


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return stats.mean(values) if values else None


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


def base_args(seed: int, output_dir: Path, model_path: str, games: int) -> SimpleNamespace:
    return SimpleNamespace(
        methods=["llm_mask"],
        games=games,
        seed=seed,
        opponent_style="responsive",
        max_steps=300,
        sample_every=0,
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
        mask_threat_require_real_target=True,
        mask_threat_target_max_shanten=1,
        mask_threat_target_signal="mc",
        mask_threat_target_prob_threshold=0.78,
        mask_threat_max_start_shanten=3,
        mask_threat_allow_exploit_overlap=False,
        mask_log_counterfactual=True,
        snapshot_oracle_samples=120,
        snapshot_crn_seeds=1,
        defender_threat_model="blend",
        defender_tell_weight=0.3,
        defender_tell_window=6,
        defender_learned_model_path="Defender_danger_model/danger_model.pth",
        backend="local_qwen",
        model_path=model_path,
        adapter_path=None,
        belief_adapter_path=None,
        max_new_tokens=128,
        output_dir=output_dir,
    )


def args_for_variant(variant: str, seed: int, output_dir: Path, model_path: str, games: int) -> SimpleNamespace:
    args = base_args(seed, output_dir, model_path, games)
    if variant == "no_forced":
        args.mask_forced_deceive = "off"
    elif variant == "bphi_risk_only":
        args.mask_forced_deceive = "off"
        args.mask_deceive_style = "safe"
        args.mask_dir_ready_threshold = -99
        args.mask_deceive_threat_ceiling = -1.0
        args.mask_threat_require_real_target = False
    elif variant == "strict_mc":
        args.mask_forced_deceive = "eligible"
        args.mask_threat_gate_mode = "cross"
        args.mask_threat_min_delta = 0.08
        args.mask_threat_max_start_shanten = 2
        args.mask_threat_target_prob_threshold = 0.90
    elif variant == "oracle_target":
        args.mask_forced_deceive = "eligible"
        args.mask_threat_target_signal = "oracle"
        args.mask_threat_target_prob_threshold = 0.5
    else:
        raise ValueError(f"unknown variant: {variant}")
    return args


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_or_load(args: SimpleNamespace, force: bool) -> Dict[str, Any]:
    summary_path = Path(args.output_dir) / "gate1_summary.json"
    if summary_path.exists() and not force:
        return load_summary(summary_path)
    return run(args)


def previous_seed_summary(previous_dir: Path, seed: int) -> Optional[Dict[str, Any]]:
    path = previous_dir / f"seed_{seed}" / "gate1_summary.json"
    if path.exists():
        return load_summary(path)
    return None


def metric(summary: Dict[str, Any], method: str) -> Dict[str, Any]:
    return summary["E2_ladder"][method]


def aggregate(per_seed: List[Dict[str, Any]], arms: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"mean": {}, "pairwise_vs_L1": {}}
    for arm in arms:
        rows = [row[arm] for row in per_seed if arm in row]
        if not rows:
            continue
        out["mean"][arm] = {
            "avg_net": mean(r["avg_net"] for r in rows),
            "hu_rate": mean(r["hu_rate"] for r in rows),
            "dealin_rate": mean(r["dealin_rate"] for r in rows),
            "DIR": mean(r["DIR"] for r in rows),
            "FFR": mean(r["FFR"] for r in rows),
            "avg_steps": mean(r["avg_steps"] for r in rows),
            "deceive_windows": sum(r["DIR_counts"]["deceive_windows"] for r in rows),
            "induced_dealin": sum(r["DIR_counts"]["induced_dealin"] for r in rows),
            "false_folds": sum(r["FFR_counts"]["false_folds"] for r in rows),
            "false_fold_opportunities": sum(r["FFR_counts"]["false_fold_opportunities"] for r in rows),
            "latency_ms": {
                "p50": mean(r["decision_latency_ms"]["p50"] for r in rows),
                "p95": mean(r["decision_latency_ms"]["p95"] for r in rows),
                "p99": mean(r["decision_latency_ms"]["p99"] for r in rows),
            },
            "mode_counts": {
                mode: sum((r.get("mode_counts") or {}).get(mode, 0) for r in rows)
                for mode in ("exploit", "safe", "deceive")
            },
        }
    if "L1" in arms:
        l1 = [row["L1"]["avg_net"] for row in per_seed if "L1" in row]
        for arm in arms:
            if arm == "L1" or arm not in out["mean"]:
                continue
            xs = [row[arm]["avg_net"] for row in per_seed if arm in row and "L1" in row]
            ys = l1[: len(xs)]
            out["pairwise_vs_L1"][f"{arm}_vs_L1_avg_net"] = paired_stats(xs, ys)
    return out


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("V4_L2_ablation_3seeds_200games"))
    parser.add_argument("--previous-e2-dir", type=Path, default=PREVIOUS_E2_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = ["no_forced", "bphi_risk_only", "strict_mc", "oracle_target"]
    arms = ["L1", "L2_current_mc", "L2_no_forced", "L2_bphi_risk_only", "L2_strict_mc", "L2_oracle_target"]

    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        row: Dict[str, Any] = {"seed": seed}
        previous = previous_seed_summary(args.previous_e2_dir, seed)
        if previous:
            row["L1"] = metric(previous, "llm_reactive_z")
            row["L2_current_mc"] = metric(previous, "llm_mask")
        else:
            print(f"warning: previous E2 summary missing for seed={seed}; L1/current not available", flush=True)

        for variant in variants:
            out_dir = args.output_dir / f"seed_{seed}_{variant}"
            summary = run_or_load(args_for_variant(variant, seed, out_dir, args.model_path, args.games), args.force)
            arm = "L2_" + variant
            row[arm] = metric(summary, "llm_mask")
            print(
                f"seed={seed} {arm}: "
                f"avg_net={row[arm]['avg_net']:.3f} "
                f"hu={row[arm]['hu_rate']:.3f} "
                f"dealin={row[arm]['dealin_rate']:.3f} "
                f"deceive={row[arm]['DIR_counts']['deceive_windows']}",
                flush=True,
            )
        per_seed.append(row)

    result = {
        "config": {
            "seeds": SEEDS,
            "games_per_seed_per_variant": args.games,
            "backend": "local_qwen",
            "model_path": args.model_path,
            "previous_e2_dir": str(args.previous_e2_dir),
            "opponent_style": "responsive",
            "defender_threat_model": "blend",
            "defender_tell_weight": 0.3,
            "variants": {
                "L2_current_mc": "previous failed run: forced_deceive=eligible, gate=delta_only, mc target threshold=0.78",
                "L2_no_forced": "same MC target filter but forced_deceive=off",
                "L2_bphi_risk_only": "deceive disabled; keep B_phi/risk prompt and safe/exploit behavior",
                "L2_strict_mc": "forced_deceive=eligible but cross gate, min_delta=0.08, max_start_shanten=2, mc target threshold=0.90",
                "L2_oracle_target": "oracle target upper bound with forced_deceive=eligible",
            },
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed, arms),
    }
    write_json(args.output_dir / "v4_l2_ablation_3seeds_summary.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_dir / 'v4_l2_ablation_3seeds_summary.json'}")


if __name__ == "__main__":
    main()

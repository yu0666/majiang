"""Run the V4 local-Qwen E2 ladder for the paper main figure.

This is the LLM version of the Gate1 ladder:

    L0 = llm_base
    L1 = llm_reactive_z
    L2 = llm_mask with deployable MC target filter

Each seed runs 200 games per method against the same responsive blend defender.
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


def args_for_seed(seed: int, output_dir: Path, model_path: str, games: int) -> SimpleNamespace:
    return SimpleNamespace(
        methods=["llm_base", "llm_reactive_z", "llm_mask"],
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


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_or_load(args: SimpleNamespace, force: bool) -> Dict[str, Any]:
    summary_path = Path(args.output_dir) / "gate1_summary.json"
    if summary_path.exists() and not force:
        return load_summary(summary_path)
    return run(args)


def aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods = {
        "L0": "llm_base",
        "L1": "llm_reactive_z",
        "L2_MC": "llm_mask",
    }
    out: Dict[str, Any] = {"mean": {}, "pairwise": {}}
    for label in methods:
        rows = [row[label] for row in per_seed]
        out["mean"][label] = {
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
    for lhs, rhs in (("L2_MC", "L1"), ("L2_MC", "L0"), ("L1", "L0")):
        out["pairwise"][f"{lhs}_vs_{rhs}_avg_net"] = paired_stats(
            [row[lhs]["avg_net"] for row in per_seed],
            [row[rhs]["avg_net"] for row in per_seed],
        )
    return out


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("V4_E2_ladder_3seeds_200games"))
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        out_dir = args.output_dir / f"seed_{seed}"
        summary = run_or_load(args_for_seed(seed, out_dir, args.model_path, args.games), args.force)
        row: Dict[str, Any] = {"seed": seed}
        row["L0"] = summary["E2_ladder"]["llm_base"]
        row["L1"] = summary["E2_ladder"]["llm_reactive_z"]
        row["L2_MC"] = summary["E2_ladder"]["llm_mask"]
        per_seed.append(row)
        print(
            f"seed={seed} "
            f"L0={row['L0']['avg_net']:.3f} "
            f"L1={row['L1']['avg_net']:.3f} "
            f"L2_MC={row['L2_MC']['avg_net']:.3f}",
            flush=True,
        )

    result = {
        "config": {
            "seeds": SEEDS,
            "games_per_seed_per_method": args.games,
            "backend": "local_qwen",
            "model_path": args.model_path,
            "opponent_style": "responsive",
            "defender_threat_model": "blend",
            "defender_tell_weight": 0.3,
            "l2_target_filter": "mc",
            "l2_target_prob_threshold": 0.78,
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    write_json(args.output_dir / "v4_e2_ladder_3seeds_summary.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_dir / 'v4_e2_ladder_3seeds_summary.json'}")


if __name__ == "__main__":
    main()

"""Evaluate MASK-SFT V5 as the L2 policy on the Gate1 main setup.

This reuses the V4 main-figure setup:
  * seeds: 20260627, 20261627, 20262627
  * 200 games per seed by default
  * responsive defender with blend threat model and tell_weight=0.3
  * L2 MASK with public-info MC target filter

The only difference is the decision model:
  base model: models/Qwen-Mahjong-V4-GRPO-Merged
  LoRA adapter: qwen-mask-sft-l2-mc-v5
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
ADAPTER_PATH = "qwen-mask-sft-l2-mc-v5"
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


def args_for_seed(
    seed: int,
    output_dir: Path,
    model_path: str,
    adapter_path: str,
    games: int,
    max_new_tokens: int,
) -> SimpleNamespace:
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
        adapter_path=adapter_path,
        belief_adapter_path=None,
        max_new_tokens=max_new_tokens,
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


def metric(summary: Dict[str, Any], method: str) -> Dict[str, Any]:
    return summary["E2_ladder"][method]


def previous_seed_summary(previous_dir: Path, seed: int) -> Optional[Dict[str, Any]]:
    path = previous_dir / f"seed_{seed}" / "gate1_summary.json"
    if path.exists():
        return load_summary(path)
    return None


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted({key for row in rows for key in row.keys() if key != "seed"})
    out: Dict[str, Any] = {"mean": {}, "pairwise": {}}
    for label in labels:
        metrics = [row[label] for row in rows if label in row]
        out["mean"][label] = {
            "avg_net": mean(r["avg_net"] for r in metrics),
            "hu_rate": mean(r["hu_rate"] for r in metrics),
            "dealin_rate": mean(r["dealin_rate"] for r in metrics),
            "DIR": mean(r["DIR"] for r in metrics),
            "FFR": mean(r["FFR"] for r in metrics),
            "avg_steps": mean(r["avg_steps"] for r in metrics),
            "deceive_windows": sum(r["DIR_counts"]["deceive_windows"] for r in metrics),
            "induced_dealin": sum(r["DIR_counts"]["induced_dealin"] for r in metrics),
            "false_folds": sum(r["FFR_counts"]["false_folds"] for r in metrics),
            "false_fold_opportunities": sum(r["FFR_counts"]["false_fold_opportunities"] for r in metrics),
            "latency_ms": {
                "p50": mean(r["decision_latency_ms"]["p50"] for r in metrics),
                "p95": mean(r["decision_latency_ms"]["p95"] for r in metrics),
                "p99": mean(r["decision_latency_ms"]["p99"] for r in metrics),
            },
            "mode_counts": {
                mode: sum((r.get("mode_counts") or {}).get(mode, 0) for r in metrics)
                for mode in ("exploit", "safe", "deceive")
            },
        }

    if "V5_MASK_SFT_L2" in labels:
        for baseline in ("V4_L1", "V4_L2_MC"):
            paired = [
                (row["V5_MASK_SFT_L2"]["avg_net"], row[baseline]["avg_net"])
                for row in rows
                if "V5_MASK_SFT_L2" in row and baseline in row
            ]
            if paired:
                xs, ys = zip(*paired)
                out["pairwise"][f"V5_MASK_SFT_L2_vs_{baseline}_avg_net"] = paired_stats(list(xs), list(ys))
    return out


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("MASK_SFT_v5_L2_3seeds_200games"))
    parser.add_argument("--previous-e2-dir", type=Path, default=PREVIOUS_E2_DIR)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []
    for seed in SEEDS:
        row: Dict[str, Any] = {"seed": seed}
        previous = previous_seed_summary(args.previous_e2_dir, seed)
        if previous:
            row["V4_L1"] = metric(previous, "llm_reactive_z")
            row["V4_L2_MC"] = metric(previous, "llm_mask")
        else:
            print(f"warning: previous V4 summary missing for seed={seed}", flush=True)

        out_dir = args.output_dir / f"seed_{seed}"
        summary = run_or_load(
            args_for_seed(seed, out_dir, args.model_path, args.adapter_path, args.games, args.max_new_tokens),
            args.force,
        )
        row["V5_MASK_SFT_L2"] = metric(summary, "llm_mask")
        per_seed.append(row)
        print(
            f"seed={seed} "
            f"V5_L2={row['V5_MASK_SFT_L2']['avg_net']:.3f} "
            f"hu={row['V5_MASK_SFT_L2']['hu_rate']:.3f} "
            f"dealin={row['V5_MASK_SFT_L2']['dealin_rate']:.3f} "
            f"deceive={row['V5_MASK_SFT_L2']['DIR_counts']['deceive_windows']}",
            flush=True,
        )

    result = {
        "config": {
            "seeds": SEEDS,
            "games_per_seed": args.games,
            "backend": "local_qwen",
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "previous_e2_dir": str(args.previous_e2_dir),
            "opponent_style": "responsive",
            "defender_threat_model": "blend",
            "defender_tell_weight": 0.3,
            "l2_target_filter": "mc",
            "l2_target_prob_threshold": 0.78,
            "max_new_tokens": args.max_new_tokens,
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    write_json(args.output_dir / "mask_sft_v5_l2_3seeds_summary.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_dir / 'mask_sft_v5_l2_3seeds_summary.json'}")


if __name__ == "__main__":
    main()

"""Run a configurable-seed V2 L0/L1/L2 ladder for the Qwen-base paper route.

Default model stack:
  model_path   = models/Qwen-Mahjong-V1-Mixed-SFT-Merged
  adapter_path = qwen-v2-grpo-diverse-v1-l2/best_grpo_adapter

All three methods use the same model:
  L0 = llm_base
  L1 = llm_reactive_z
  L2 = llm_mask with public-info MC target filter
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


SEEDS = [20260627, 20261627, 20262627,20262727, 2026062728, 2026062729, 2026062730, 2026062731, 2026062732, 2026062733]
MODEL_PATH = "models/Qwen-Mahjong-V1-Mixed-SFT-Merged"
ADAPTER_PATH = "qwen-v2-grpo-diverse-v1-l2/best_grpo_adapter"


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
    adapter_path: Optional[str],
    games: int,
    max_new_tokens: int,
    forced_deceive: str = "off",
    max_shanten_regret: int = 0,
    candidate_reranker: bool = False,
    candidate_scoring: bool = False,
    reranker_max_candidates: int = 6,
    reranker_model_path: Optional[str] = None,
    reranker_adapter_path: Optional[str] = None,
    gate_policy: str = "rule",
    gate_model_path: Optional[str] = None,
    gate_adapter_path: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        mask_candidate_reranker=candidate_reranker,
        mask_candidate_scoring=candidate_scoring,
        mask_reranker_max_candidates=reranker_max_candidates,
        reranker_model_path=reranker_model_path,
        reranker_adapter_path=reranker_adapter_path,
        reranker_max_new_tokens=16,
        mask_gate_policy=gate_policy,
        gate_model_path=gate_model_path,
        gate_adapter_path=gate_adapter_path,
        gate_max_new_tokens=8,
        methods=["llm_base", "llm_reactive_z", "llm_mask"],
        games=games,
        seed=seed,
        opponent_style="responsive",
        max_steps=300,
        sample_every=0,
        threat_fold_threshold=0.7,
        oracle_samples=30,
        oracle_beta=2.0,
        danger_threshold=1,
        ffr_hand_shanten=1,
        mask_oracle_samples=30,
        mask_oracle_beta=2.0,
        mask_danger_threshold=1,
        mask_dir_ready_threshold=0,
        mask_deceive_threat_ceiling=0.5,
        mask_forced_deceive=forced_deceive,
        mask_deceive_style="threat",
        mask_threat_allow_break_ready=False,
        mask_threat_max_result_shanten=0,
        mask_threat_max_shanten_regret=max_shanten_regret,
        mask_threat_min_ukeire_ratio=1.0,
        mask_threat_gate_threshold=0.7,
        mask_threat_gate_margin=0.12,
        mask_threat_min_delta=0.03,
        mask_threat_gate_mode="cross",
        mask_threat_response_model="blend",
        mask_threat_response_tell_weight=0.3,
        mask_threat_tell_window=6,
        mask_threat_require_real_target=True,
        mask_threat_target_max_shanten=1,
        mask_threat_target_signal="mc",
        mask_threat_target_prob_threshold=0.78,
        mask_threat_max_start_shanten=2,
        mask_threat_allow_exploit_overlap=False,
        mask_log_counterfactual=True,
        snapshot_oracle_samples=120,
        snapshot_crn_seeds=1,
        defender_threat_model="blend",
        defender_tell_weight=0.3,
        defender_tell_window=6,
        defender_learned_model_path="Defender_danger_model/danger_model.pth",
        neural_opponent_model_path="Neural_opponent_model/neural_opponent_policy.pth",
        neural_opponent_device="cpu",
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


def require_local_adapter(adapter_path: str) -> str:
    path = Path(adapter_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"V2 adapter is not ready: {config_path}\n"
            "Train it first with:\n"
            "  CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 ./py10/bin/python3 train_v2_grpo.py\n"
            "To evaluate the previous balanced adapter instead, pass:\n"
            "  --adapter-path qwen-v2-grpo-balanced-v1-l2/best_grpo_adapter"
        )
    return str(path)


def run_or_load(args: SimpleNamespace, force: bool) -> Dict[str, Any]:
    summary_path = Path(args.output_dir) / "gate1_summary.json"
    if summary_path.exists() and not force:
        return load_summary(summary_path)
    return run(args)


def aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = ["V2_L0", "V2_L1", "V2_L2_MC"]
    out: Dict[str, Any] = {"mean": {}, "pairwise": {}}
    for label in labels:
        rows = [row[label] for row in per_seed]
        out["mean"][label] = {
            "avg_net": mean(r["avg_net"] for r in rows),
            "net_distribution": {
                key: mean(
                    r.get("net_distribution", {}).get(key)
                    for r in rows
                    if r.get("net_distribution", {}).get(key) is not None
                )
                for key in ("median", "std", "trimmed_mean_10pct", "p10", "p90", "min", "max")
            },
            "hu_rate": mean(r["hu_rate"] for r in rows),
            "dealin_rate": mean(r["dealin_rate"] for r in rows),
            "DIR": mean(r["DIR"] for r in rows),
            "FFR": mean(r["FFR"] for r in rows),
            "avg_steps": mean(r["avg_steps"] for r in rows),
            "end_state": {
                key: mean(
                    r.get("end_state", {}).get(key)
                    for r in rows
                    if r.get("end_state", {}).get(key) is not None
                )
                for key in (
                    "wall_exhaustion_rate",
                    "final_ready_rate_non_hu",
                    "hua_zhu_rate_non_hu",
                    "avg_final_shanten_non_hu",
                )
            },
            "conditional_net": {
                key: mean(
                    r.get("conditional_net", {}).get(key)
                    for r in rows
                    if r.get("conditional_net", {}).get(key) is not None
                )
                for key in ("hu", "dealin", "neither_hu_nor_dealin")
            },
            "action_efficiency": {
                key: mean(r.get("action_efficiency", {}).get(key, 0.0) for r in rows)
                for key in ("avg_shanten_regret", "positive_shanten_regret_rate")
            },
            "safe_overrides": sum(
                int(r.get("action_efficiency", {}).get("safe_overrides", 0)) for r in rows
            ),
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
    for lhs, rhs in (("V2_L2_MC", "V2_L1"), ("V2_L2_MC", "V2_L0"), ("V2_L1", "V2_L0")):
        out["pairwise"][f"{lhs}_vs_{rhs}_avg_net"] = paired_stats(
            [row[lhs]["avg_net"] for row in per_seed],
            [row[rhs]["avg_net"] for row in per_seed],
        )
    return out


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("V2_diverse_E2_ladder"))
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="off")
    parser.add_argument("--mask-threat-max-shanten-regret", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mask-candidate-reranker", action="store_true")
    parser.add_argument("--mask-candidate-scoring", action="store_true")
    parser.add_argument("--mask-reranker-max-candidates", type=int, default=6)
    parser.add_argument("--reranker-model-path", default=None)
    parser.add_argument("--reranker-adapter-path", default=None)
    parser.add_argument("--gate-policy", choices=["rule", "learned"], default="rule")
    parser.add_argument("--gate-model-path", default=None)
    parser.add_argument("--gate-adapter-path", default=None)
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else require_local_adapter(args.adapter_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []
    for seed in args.seeds:
        out_dir = args.output_dir / f"seed_{seed}"
        summary = run_or_load(
            args_for_seed(
                seed, out_dir, args.model_path, adapter_path, args.games, args.max_new_tokens,
                forced_deceive=args.mask_forced_deceive,
                max_shanten_regret=args.mask_threat_max_shanten_regret,
                candidate_reranker=args.mask_candidate_reranker,
                candidate_scoring=args.mask_candidate_scoring,
                reranker_max_candidates=args.mask_reranker_max_candidates,
                reranker_model_path=args.reranker_model_path,
                reranker_adapter_path=args.reranker_adapter_path,
                gate_policy=args.gate_policy,
                gate_model_path=args.gate_model_path,
                gate_adapter_path=args.gate_adapter_path,
            ),
            args.force,
        )
        row = {
            "seed": seed,
            "V2_L0": summary["E2_ladder"]["llm_base"],
            "V2_L1": summary["E2_ladder"]["llm_reactive_z"],
            "V2_L2_MC": summary["E2_ladder"]["llm_mask"],
        }
        per_seed.append(row)
        print(
            f"seed={seed} "
            f"L0={row['V2_L0']['avg_net']:.3f} "
            f"L1={row['V2_L1']['avg_net']:.3f} "
            f"L2={row['V2_L2_MC']['avg_net']:.3f}",
            flush=True,
        )

    result = {
        "config": {
            "seeds": args.seeds,
            "games_per_seed_per_method": args.games,
            "backend": "local_qwen",
            "model_path": args.model_path,
            "adapter_path": adapter_path,
            "reranker_model_path": args.reranker_model_path,
            "reranker_adapter_path": args.reranker_adapter_path,
            "opponent_style": "responsive",
            "defender_threat_model": "blend",
            "defender_threat_threshold": 0.7,
            "defender_tell_weight": 0.3,
            "l2_target_filter": "mc",
            "l2_target_prob_threshold": 0.78,
            "l2_forced_deceive": args.mask_forced_deceive,
            "l2_max_shanten_regret": args.mask_threat_max_shanten_regret,
            "l2_min_ukeire_ratio": 1.0,
            "l2_threat_gate_threshold": 0.7,
            "l2_threat_response_model": "blend",
            "l2_threat_response_tell_weight": 0.3,
            "l2_max_start_shanten": 2,
            "max_new_tokens": args.max_new_tokens,
            "gate_policy": args.gate_policy,
            "gate_model_path": args.gate_model_path,
            "gate_adapter_path": args.gate_adapter_path,
        },
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed),
    }
    write_json(args.output_dir / "v2_e2_ladder_3seeds_summary.json", result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output_dir / 'v2_e2_ladder_3seeds_summary.json'}")


if __name__ == "__main__":
    main()

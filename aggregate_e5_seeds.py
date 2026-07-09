"""Multi-seed E5 ladder: does MASK's net-benefit survive a harder ResponsiveDefender?

Runs run_gate1_experiments.run() across N seeds for two defender configs -- the
existing discard_tell (tell_weight=1.0) baseline vs the trained "learned" danger
model (see train_defender_danger_model.py) -- and compares MASK's net-benefit
(Gate1_pairwise: llm_reactive_z vs llm_mask, paired by seed within each run) across
the two configs. Mirrors aggregate_h1_seeds.py's mean_ci/seed-spread pattern, reusing
its mean_ci/t_critical rather than re-deriving them.

Both configs share every other knob (mask_cfg, opponent_style, backend, model) so the
only thing that changes between arms is the defender's threat perception -- exactly
the plan's "same 13-seed harness, learned vs the current blend baseline" comparison.

Example (GPU1 pinned to dodge a contended GPU0; see gpu probe in this session):
  CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 ./py10/bin/python3 aggregate_e5_seeds.py \
      --num-seeds 13 --games 30 --backend local_qwen \
      --model-path models/Qwen-Mahjong-V4-GRPO-Merged
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from aggregate_h1_seeds import mean_ci
from experiment_trace import ensure_deterministic_hashing, sign_test_p_value, write_json
from run_gate1_experiments import run


def config_args(base: argparse.Namespace, seed: int, threat_model: str, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        methods=["llm_reactive_z", "llm_mask"],
        games=base.games, seed=seed, opponent_style="responsive",
        max_steps=base.max_steps, sample_every=base.sample_every,
        threat_fold_threshold=0.4, oracle_samples=30, oracle_beta=2.0,
        danger_threshold=1, ffr_hand_shanten=1,
        mask_oracle_samples=30, mask_oracle_beta=2.0, mask_danger_threshold=1,
        mask_dir_ready_threshold=0, mask_deceive_threat_ceiling=0.5,
        mask_forced_deceive="eligible", mask_deceive_style="threat",
        mask_threat_allow_break_ready=False, mask_threat_max_result_shanten=1,
        mask_threat_gate_threshold=0.4, mask_threat_gate_margin=0.12,
        mask_threat_min_delta=0.03, mask_threat_gate_mode="cross",
        mask_threat_tell_window=6, mask_threat_require_real_target=False,
        mask_threat_target_max_shanten=0, mask_threat_target_signal="oracle",
        mask_threat_target_prob_threshold=0.5, mask_threat_max_start_shanten=3,
        mask_threat_allow_exploit_overlap=False, mask_log_counterfactual=True,
        snapshot_oracle_samples=120, snapshot_crn_seeds=2,
        defender_threat_model=threat_model, defender_tell_weight=1.0,
        defender_tell_window=6, defender_learned_model_path=base.learned_model_path,
        backend=base.backend, model_path=base.model_path, adapter_path=None,
        belief_adapter_path=None, max_new_tokens=128, output_dir=out_dir,
    )


def pairwise_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    pw = summary["Gate1_pairwise"]
    return {
        "paired_seeds": pw.get("paired_seeds", 0),
        "avg_delta_net": pw.get("avg_delta_net"),
        "avg_delta_dealin_rate": pw.get("avg_delta_dealin_rate"),
        "net_positive_rate": pw.get("net_positive_rate"),
        "net_sign_test_p": pw.get("net_sign_test_p"),
    }


def main() -> None:
    ensure_deterministic_hashing()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-seeds", type=int, default=13)
    p.add_argument("--base-seed", type=int, default=20260627)
    p.add_argument("--games", type=int, default=30, help="Games per method per seed per config.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--backend", default="local_qwen", choices=["heuristic_fallback", "local_qwen"])
    p.add_argument("--model-path", default="models/Qwen-Mahjong-V4-GRPO-Merged")
    p.add_argument("--learned-model-path", default="Defender_danger_model/danger_model.pth")
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--output-dir", type=Path, default=Path("E5_seed_aggregate"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed: List[Dict[str, Any]] = []
    for k in range(args.num_seeds):
        seed = args.base_seed + k * 101
        baseline_sa = config_args(args, seed, "discard_tell", args.output_dir / f"seed_{seed}_baseline")
        learned_sa = config_args(args, seed, "learned", args.output_dir / f"seed_{seed}_learned")

        baseline_summary = run(baseline_sa)
        learned_summary = run(learned_sa)

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
    paired_agg = mean_ci(paired_diffs)  # CI on (learned - baseline) per seed == paired t-test
    paired_sign_p = sign_test_p_value(paired_diffs)

    n_learned_positive = sum(1 for d in learned_deltas if d > 0)
    robustness = {
        "learned_net_benefit_ci_excludes_zero_below": learned_agg["lo"] > 0,
        "seeds_with_positive_learned_net_benefit": f"{n_learned_positive}/{len(learned_deltas)}",
        "paired_diff_ci_excludes_zero": (paired_agg["lo"] > 0 or paired_agg["hi"] < 0),
        "paired_sign_test_p": paired_sign_p,
        "paired_significant": (paired_sign_p is not None and paired_sign_p < args.p_threshold),
    }

    out = {
        "config": {
            "num_seeds": args.num_seeds, "base_seed": args.base_seed, "games_per_seed": args.games,
            "backend": args.backend, "model_path": args.model_path,
            "baseline_defender_threat_model": "discard_tell (tell_weight=1.0)",
            "learned_defender_threat_model": "learned",
            "learned_model_path": args.learned_model_path,
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
            f"{robustness['seeds_with_positive_learned_net_benefit']}"
        ),
    }
    write_json(args.output_dir / "e5_seed_aggregate.json", out)
    print("\n=== E5 multi-seed ladder aggregate ===")
    print(out["paper_row"])
    print("robustness:", json.dumps(robustness, ensure_ascii=False))
    print(f"\nSaved: {args.output_dir / 'e5_seed_aggregate.json'}")


if __name__ == "__main__":
    main()

"""Reproduce/extend the remembered 4-arm shanten-filter ladder: L0 / L1 / L2-no-filter /
L2-shanten<=1-filter vs a blend(discard_tell=0.3, mc=0.7) ResponsiveDefender.

User-recalled table (13 seeds x 200 games/seed, public-info-only defender):
  L0  llm_base                         avg_net= 5.18  hu=0.431 dealin=0.153
  L1  llm_reactive_z                   avg_net=-26.02 hu=0.351 dealin=0.176
  L2  MASK, no high-risk target filter avg_net=26.72  hu=0.289 dealin=0.067
  L2  MASK, shanten<=1 target filter   avg_net=29.76  hu=0.292 dealin=0.063
Exhaustive search of this project found no gate1_summary.json matching these numbers or this
tell_weight=0.3 blend config for a Gate1 ladder (0.3 only appears as generate_defender_danger_
data.py's self-play data-gen default, a different script) -- so this run is a fresh reproduction
attempt with 3 new seeds, not a re-read of an existing file.

Both L2 arms share forced_deceive="eligible" (the tell-threshold deceive gate); they differ only
in whether an ADDITIONAL opponent-shanten filter is required before deceiving:
  L2-no-filter: mask_threat_require_real_target=False (default eligible gate only)
  L2-filter:    mask_threat_require_real_target=True, target_max_shanten=1, target_signal="mc"
                (the deployable public-info substitute -- NOT target_signal="oracle", which the
                code's own help text flags as a ground-truth-peek upper-bound ablation and would
                violate the "public info only, no god view" requirement this ladder is for)

As in run_e2_4arm_ladder.py, mask_cfg is one setting per run() call, so the two L2 variants need
two run() calls per seed (both use the same --seed/--games, so all 4 arms are matched 1:1 by
seed):
  call A: methods=[llm_base, llm_reactive_z, llm_mask], require_real_target=False
  call B: methods=[llm_mask],                            require_real_target=True

Usage (GPU1 pinned to dodge the contended GPU0):
  CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 ./py10/bin/python3 run_e2_shanten_filter_ladder.py \
      --games 200 --repeat-seeds 20270707 20270714 20270721
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from aggregate_h1_seeds import mean_ci
from experiment_trace import ensure_deterministic_hashing, sign_test_p_value, write_json
from run_gate1_experiments import run

ARMS = ["L0", "L1", "L2_nofilter", "L2_filter"]


def config_args(
    seed: int, games: int, methods: List[str], require_real_target: bool,
    out_dir: Path, base: argparse.Namespace,
) -> argparse.Namespace:
    return argparse.Namespace(
        methods=methods,
        games=games, seed=seed, opponent_style="responsive",
        max_steps=base.max_steps, sample_every=base.sample_every,
        threat_fold_threshold=0.4, oracle_samples=30, oracle_beta=2.0,
        danger_threshold=1, ffr_hand_shanten=1,
        mask_oracle_samples=30, mask_oracle_beta=2.0, mask_danger_threshold=1,
        mask_dir_ready_threshold=0, mask_deceive_threat_ceiling=0.5,
        mask_forced_deceive="eligible", mask_deceive_style="threat",
        mask_threat_allow_break_ready=False, mask_threat_max_result_shanten=1,
        mask_threat_gate_threshold=0.4, mask_threat_gate_margin=0.12,
        mask_threat_min_delta=0.03, mask_threat_gate_mode="cross",
        mask_threat_tell_window=6, mask_threat_require_real_target=require_real_target,
        mask_threat_target_max_shanten=1, mask_threat_target_signal="mc",
        mask_threat_target_prob_threshold=0.5, mask_threat_max_start_shanten=3,
        mask_threat_allow_exploit_overlap=False, mask_log_counterfactual=True,
        snapshot_oracle_samples=120, snapshot_crn_seeds=2,
        defender_threat_model="blend", defender_tell_weight=0.3,
        defender_tell_window=6, defender_learned_model_path=base.learned_model_path,
        backend=base.backend, model_path=base.model_path, adapter_path=None,
        belief_adapter_path=None, max_new_tokens=128, output_dir=out_dir,
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def per_seed_net(games: List[Dict[str, Any]], method: str) -> Dict[int, float]:
    return {g["seed"]: g["agent_net"] for g in games if g["method"] == method}


def per_seed_rate(games: List[Dict[str, Any]], method: str, key: str) -> Dict[int, float]:
    return {g["seed"]: (1.0 if g[key] else 0.0) for g in games if g["method"] == method}


def run_one_repeat(seed: int, games: int, base: argparse.Namespace, out_root: Path) -> Dict[str, Any]:
    a_dir = out_root / f"seed_{seed}_A_nofilter"
    b_dir = out_root / f"seed_{seed}_B_filter"

    a_args = config_args(seed, games, ["llm_base", "llm_reactive_z", "llm_mask"], False, a_dir, base)
    b_args = config_args(seed, games, ["llm_mask"], True, b_dir, base)

    print(f"\n=== repeat seed={seed}: L0/L1/L2-nofilter (require_real_target=False) ===")
    run(a_args)
    print(f"=== repeat seed={seed}: L2-filter (require_real_target=True, shanten<=1, mc signal) ===")
    run(b_args)

    a_games = load_jsonl(a_dir / "gate1_games.jsonl")
    b_games = load_jsonl(b_dir / "gate1_games.jsonl")

    net_by_arm = {
        "L0": per_seed_net(a_games, "llm_base"),
        "L1": per_seed_net(a_games, "llm_reactive_z"),
        "L2_nofilter": per_seed_net(a_games, "llm_mask"),
        "L2_filter": per_seed_net(b_games, "llm_mask"),
    }
    hu_by_arm = {
        "L0": per_seed_rate(a_games, "llm_base", "agent_hu"),
        "L1": per_seed_rate(a_games, "llm_reactive_z", "agent_hu"),
        "L2_nofilter": per_seed_rate(a_games, "llm_mask", "agent_hu"),
        "L2_filter": per_seed_rate(b_games, "llm_mask", "agent_hu"),
    }
    dealin_by_arm = {
        "L0": per_seed_rate(a_games, "llm_base", "agent_dealin"),
        "L1": per_seed_rate(a_games, "llm_reactive_z", "agent_dealin"),
        "L2_nofilter": per_seed_rate(a_games, "llm_mask", "agent_dealin"),
        "L2_filter": per_seed_rate(b_games, "llm_mask", "agent_dealin"),
    }

    seeds = sorted(set.intersection(*[set(net_by_arm[a].keys()) for a in ARMS]))
    per_seed_rows = [
        {"seed": s,
         **{f"{a}_net": net_by_arm[a][s] for a in ARMS},
         **{f"{a}_hu": hu_by_arm[a][s] for a in ARMS},
         **{f"{a}_dealin": dealin_by_arm[a][s] for a in ARMS}}
        for s in seeds
    ]

    arm_agg = {
        a: {
            "avg_net": mean_ci([r[f"{a}_net"] for r in per_seed_rows]),
            "hu_rate": sum(r[f"{a}_hu"] for r in per_seed_rows) / len(per_seed_rows),
            "dealin_rate": sum(r[f"{a}_dealin"] for r in per_seed_rows) / len(per_seed_rows),
        }
        for a in ARMS
    }

    pairs = [("L1", "L2_nofilter"), ("L1", "L2_filter"), ("L2_nofilter", "L2_filter"), ("L0", "L1")]
    pairwise = {}
    for base_arm, target_arm in pairs:
        deltas = [r[f"{target_arm}_net"] - r[f"{base_arm}_net"] for r in per_seed_rows]
        diff_agg = mean_ci(deltas)
        pairwise[f"{target_arm}_minus_{base_arm}"] = {
            "n": len(deltas),
            "mean_delta": diff_agg["mean"],
            "ci95_half": diff_agg["ci95_half"],
            "sign_test_p": sign_test_p_value(deltas),
        }

    return {
        "seed": seed,
        "n_matched_games": len(per_seed_rows),
        "per_seed": per_seed_rows,
        "arm_summary": arm_agg,
        "pairwise": pairwise,
    }


def main() -> None:
    ensure_deterministic_hashing()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat-seeds", type=int, nargs="+", default=[20270707, 20270714, 20270721])
    p.add_argument("--games", type=int, default=200, help="Games per arm per seed.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--backend", default="local_qwen", choices=["heuristic_fallback", "local_qwen"])
    p.add_argument("--model-path", default="models/Qwen-Mahjong-V4-GRPO-Merged")
    p.add_argument("--learned-model-path", default="Defender_danger_model/danger_model.pth")
    p.add_argument("--output-dir", type=Path, default=Path("E2_shanten_filter_ladder"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_ns = argparse.Namespace(
        max_steps=args.max_steps, sample_every=args.sample_every, backend=args.backend,
        model_path=args.model_path, learned_model_path=args.learned_model_path,
    )

    seed_results: List[Dict[str, Any]] = []
    for seed in args.repeat_seeds:
        r = run_one_repeat(seed, args.games, base_ns, args.output_dir)
        seed_results.append(r)
        print(f"\n--- seed={seed} done: n={r['n_matched_games']} ---")
        for arm in ARMS:
            s = r["arm_summary"][arm]
            print(f"  {arm}: avg_net={s['avg_net']['mean']:.2f} hu_rate={s['hu_rate']:.3f} dealin_rate={s['dealin_rate']:.3f}")
        for key, pw in r["pairwise"].items():
            print(f"  {key}: delta={pw['mean_delta']:.2f} +/- {pw['ci95_half']:.2f}, p={pw['sign_test_p']}")

    across_seed_arm_avg_net = {
        a: mean_ci([r["arm_summary"][a]["avg_net"]["mean"] for r in seed_results]) for a in ARMS
    }
    across_seed_hu = {a: sum(r["arm_summary"][a]["hu_rate"] for r in seed_results) / len(seed_results) for a in ARMS}
    across_seed_dealin = {a: sum(r["arm_summary"][a]["dealin_rate"] for r in seed_results) / len(seed_results) for a in ARMS}

    out = {
        "config": {
            "arms": {
                "L0": {"method": "llm_base"},
                "L1": {"method": "llm_reactive_z"},
                "L2_nofilter": {"method": "llm_mask", "forced_deceive": "eligible", "require_real_target": False},
                "L2_filter": {"method": "llm_mask", "forced_deceive": "eligible", "require_real_target": True,
                              "target_max_shanten": 1, "target_signal": "mc", "target_prob_threshold": 0.5},
            },
            "repeat_seeds": args.repeat_seeds,
            "games_per_arm_per_seed": args.games,
            "backend": args.backend, "model_path": args.model_path,
            "defender_threat_model": "blend (discard_tell=0.3, mc=0.7)",
            "opponent_style": "responsive",
            "mask_deceive_style": "threat",
            "note": "3-seed reproduction attempt of a remembered 13-seed x 200-game table; "
                    "no matching gate1_summary.json for this config was found anywhere in the project.",
        },
        "per_seed": seed_results,
        "across_seeds": {
            "avg_net_mean_of_seed_means": across_seed_arm_avg_net,
            "hu_rate_mean_of_seed_means": across_seed_hu,
            "dealin_rate_mean_of_seed_means": across_seed_dealin,
        },
    }
    write_json(args.output_dir / "e2_shanten_filter_ladder.json", out)
    print(f"\nSaved: {args.output_dir / 'e2_shanten_filter_ladder.json'}")


if __name__ == "__main__":
    main()

"""Genuine same-seed, 4-arm E2 ladder: L0 / L1 / L2-eligible / L2-always vs ResponsiveDefender.

Requested comparison:
  L0  = llm_base          (mask_cfg irrelevant)
  L1  = llm_reactive_z     (mask_cfg irrelevant)
  L2E = llm_mask, --mask-forced-deceive eligible  (opponent-shanten-gated deceive)
  L2A = llm_mask, --mask-forced-deceive always    (deceive unconditionally, never run/saved
        anywhere in this project before -- the CLI choice already existed but was unused)

run_gate1_experiments.run() dispatches P0's method by a literal string
("llm_base"/"llm_reactive_z"/"llm_mask") and mask_cfg["forced_deceive"] is one global setting
per run() call -- so L2E and L2A cannot share a single run() call under the same method name.
Instead, per repeat we make two run() calls that both use the same --seed/--games (so both see
the identical seed sequence seed+i for i in range(games), matching L0/L1/L2E against L2A
1:1 by seed):
  call ABC: methods=[llm_base, llm_reactive_z, llm_mask], forced_deceive=eligible
  call D:   methods=[llm_mask],                            forced_deceive=always
Then games are pulled from each call's gate1_games.jsonl and matched by "seed", with call D's
llm_mask rows relabeled "llm_mask_always" for the paired analysis (the on-disk file itself is
untouched -- relabeling only happens in this script's in-memory table).

Defender is the discard_tell ResponsiveDefender (tell_weight=1.0) -- the canonical baseline used
by every 200-game Gate1 ladder this session (Gate1_results_v4_threat_gated_ladder_200 /
_ffr_ladder_200) and by aggregate_e5_seeds.py's config_args(). deceive_style="threat" and all
other mask_cfg knobs mirror that same canonical template verbatim.

Usage (GPU1 pinned to dodge the contended GPU0):
  CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 ./py10/bin/python3 run_e2_4arm_ladder.py \
      --games 300 --repeat-seeds 20270707 20270714 20270721
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aggregate_h1_seeds import mean_ci
from experiment_trace import ensure_deterministic_hashing, sign_test_p_value, write_json
from run_gate1_experiments import run

ARMS = ["L0", "L1", "L2E", "L2A"]
ARM_TO_METHOD = {"L0": "llm_base", "L1": "llm_reactive_z", "L2E": "llm_mask", "L2A": "llm_mask"}


def config_args(
    seed: int, games: int, methods: List[str], forced_deceive: str,
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
        mask_forced_deceive=forced_deceive, mask_deceive_style="threat",
        mask_threat_allow_break_ready=False, mask_threat_max_result_shanten=1,
        mask_threat_gate_threshold=0.4, mask_threat_gate_margin=0.12,
        mask_threat_min_delta=0.03, mask_threat_gate_mode="cross",
        mask_threat_tell_window=6, mask_threat_require_real_target=False,
        mask_threat_target_max_shanten=0, mask_threat_target_signal="oracle",
        mask_threat_target_prob_threshold=0.5, mask_threat_max_start_shanten=3,
        mask_threat_allow_exploit_overlap=False, mask_log_counterfactual=True,
        snapshot_oracle_samples=120, snapshot_crn_seeds=2,
        defender_threat_model="discard_tell", defender_tell_weight=1.0,
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


def run_one_repeat(seed: int, games: int, base: argparse.Namespace, out_root: Path) -> Dict[str, Any]:
    abc_dir = out_root / f"seed_{seed}_ABC"
    d_dir = out_root / f"seed_{seed}_D"

    abc_args = config_args(seed, games, ["llm_base", "llm_reactive_z", "llm_mask"], "eligible", abc_dir, base)
    d_args = config_args(seed, games, ["llm_mask"], "always", d_dir, base)

    print(f"\n=== repeat seed={seed}: running L0/L1/L2E (forced_deceive=eligible) ===")
    run(abc_args)
    print(f"=== repeat seed={seed}: running L2A (forced_deceive=always) ===")
    run(d_args)

    abc_games = load_jsonl(abc_dir / "gate1_games.jsonl")
    d_games = load_jsonl(d_dir / "gate1_games.jsonl")

    net_by_arm = {
        "L0": per_seed_net(abc_games, "llm_base"),
        "L1": per_seed_net(abc_games, "llm_reactive_z"),
        "L2E": per_seed_net(abc_games, "llm_mask"),
        "L2A": per_seed_net(d_games, "llm_mask"),
    }

    seeds = sorted(set.intersection(*[set(net_by_arm[a].keys()) for a in ARMS]))
    per_seed_rows = [{"seed": s, **{a: net_by_arm[a][s] for a in ARMS}} for s in seeds]

    arm_agg = {a: mean_ci([r[a] for r in per_seed_rows]) for a in ARMS}

    pairs = [("L1", "L2E"), ("L1", "L2A"), ("L2E", "L2A"), ("L0", "L1")]
    pairwise = {}
    for base_arm, target_arm in pairs:
        deltas = [r[target_arm] - r[base_arm] for r in per_seed_rows]
        diff_agg = mean_ci(deltas)
        pairwise[f"{target_arm}_minus_{base_arm}"] = {
            "n": len(deltas),
            "mean_delta": diff_agg["mean"],
            "ci95_half": diff_agg["ci95_half"],
            "lo": diff_agg["lo"],
            "hi": diff_agg["hi"],
            "sign_test_p": sign_test_p_value(deltas),
        }

    return {
        "seed": seed,
        "n_matched_games": len(per_seed_rows),
        "per_seed": per_seed_rows,
        "arm_avg_net": arm_agg,
        "pairwise": pairwise,
    }


def main() -> None:
    ensure_deterministic_hashing()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat-seeds", type=int, nargs="+", default=[20270707, 20270714, 20270721],
                    help="One base seed per repeat; each repeat runs seed..seed+games-1.")
    p.add_argument("--games", type=int, default=300, help="Games per arm per repeat.")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--sample-every", type=int, default=4)
    p.add_argument("--backend", default="local_qwen", choices=["heuristic_fallback", "local_qwen"])
    p.add_argument("--model-path", default="models/Qwen-Mahjong-V4-GRPO-Merged")
    p.add_argument("--learned-model-path", default="Defender_danger_model/danger_model.pth")
    p.add_argument("--output-dir", type=Path, default=Path("E2_4arm_ladder"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_ns = argparse.Namespace(
        max_steps=args.max_steps, sample_every=args.sample_every, backend=args.backend,
        model_path=args.model_path, learned_model_path=args.learned_model_path,
    )

    repeats: List[Dict[str, Any]] = []
    for seed in args.repeat_seeds:
        repeat_result = run_one_repeat(seed, args.games, base_ns, args.output_dir)
        repeats.append(repeat_result)
        print(f"\n--- repeat seed={seed} done: n={repeat_result['n_matched_games']} ---")
        for arm in ARMS:
            agg = repeat_result["arm_avg_net"][arm]
            print(f"  {arm}: avg_net={agg['mean']:.1f} +/- {agg['ci95_half']:.1f}")
        for key, pw in repeat_result["pairwise"].items():
            print(f"  {key}: delta={pw['mean_delta']:.1f} +/- {pw['ci95_half']:.1f}, p={pw['sign_test_p']}")

    across_repeat_arm_means = {
        a: mean_ci([r["arm_avg_net"][a]["mean"] for r in repeats]) for a in ARMS
    }
    across_repeat_pairwise = {}
    if repeats:
        for key in repeats[0]["pairwise"].keys():
            deltas = [r["pairwise"][key]["mean_delta"] for r in repeats]
            across_repeat_pairwise[key] = {
                "per_repeat_mean_delta": deltas,
                "mean_of_repeat_means": mean_ci(deltas) if len(deltas) > 1 else {"mean": deltas[0] if deltas else None},
            }

    out = {
        "config": {
            "arms": {a: {"method": ARM_TO_METHOD[a], **({"forced_deceive": "eligible"} if a == "L2E" else {"forced_deceive": "always"} if a == "L2A" else {})} for a in ARMS},
            "repeat_seeds": args.repeat_seeds,
            "games_per_arm_per_repeat": args.games,
            "backend": args.backend, "model_path": args.model_path,
            "defender_threat_model": "discard_tell (tell_weight=1.0)",
            "opponent_style": "responsive",
            "mask_deceive_style": "threat",
        },
        "repeats": repeats,
        "across_repeats": {
            "arm_avg_net_mean_of_repeat_means": across_repeat_arm_means,
            "pairwise_mean_of_repeat_deltas": across_repeat_pairwise,
        },
    }
    write_json(args.output_dir / "e2_4arm_ladder.json", out)
    print(f"\nSaved: {args.output_dir / 'e2_4arm_ladder.json'}")


if __name__ == "__main__":
    main()

"""Complete the E5 13-seed ladder by running only the seeds not yet finished.

8/13 seeds (20260627..20261334) already have full baseline+learned gate1_summary.json
on disk in E5_seed_aggregate/ from the earlier run that was stopped early. This script
reuses aggregate_e5_seeds.py's config_args()/pairwise_row() to run just the remaining
seeds (the incomplete 9th seed's baseline is re-run since it has no paired learned arm,
plus 4 new seeds), writing into the same E5_seed_aggregate/ directory layout so the
final reaggregation step can treat all 13 uniformly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from aggregate_e5_seeds import config_args, pairwise_row
from run_gate1_experiments import run

OUT_DIR = Path("E5_seed_aggregate")
BASE_SEED = 20260627
NUM_SEEDS = 13
ALREADY_DONE = {20260627, 20260728, 20260829, 20260930, 20261031, 20261132, 20261233, 20261334}


def base_namespace():
    import argparse
    return argparse.Namespace(
        games=30, max_steps=300, sample_every=4,
        backend="local_qwen", model_path="models/Qwen-Mahjong-V4-GRPO-Merged",
        learned_model_path="Defender_danger_model/danger_model.pth",
    )


def main() -> None:
    base = base_namespace()
    remaining_seeds = [BASE_SEED + k * 101 for k in range(NUM_SEEDS) if (BASE_SEED + k * 101) not in ALREADY_DONE]
    print(f"Remaining seeds to run: {remaining_seeds}")

    for seed in remaining_seeds:
        baseline_dir = OUT_DIR / f"seed_{seed}_baseline"
        learned_dir = OUT_DIR / f"seed_{seed}_learned"
        baseline_sa = config_args(base, seed, "discard_tell", baseline_dir)
        learned_sa = config_args(base, seed, "learned", learned_dir)

        baseline_summary = run(baseline_sa)
        learned_summary = run(learned_sa)

        baseline_pw = pairwise_row(baseline_summary)
        learned_pw = pairwise_row(learned_summary)
        print(f"seed {seed}: baseline avg_delta_net={baseline_pw['avg_delta_net']} "
              f"(p={baseline_pw['net_sign_test_p']}) | "
              f"learned avg_delta_net={learned_pw['avg_delta_net']} "
              f"(p={learned_pw['net_sign_test_p']})")

    print("\nAll remaining seeds complete.")


if __name__ == "__main__":
    main()

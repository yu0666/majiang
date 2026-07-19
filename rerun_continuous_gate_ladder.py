"""Paired L0 / L1 / L2-rule-gated / L2-continuous-gate comparison, same seeds.

heuristic_fallback backend (no LLM) so this runs fast; matches the pattern
already validated this session in rerun_july7_gate1_scale.py. Each seed makes
two run() calls sharing the same --seed so all four arms are matched 1:1 by
seed: call A = [llm_base, llm_reactive_z, llm_mask] with gate_policy=rule
(today's baseline L2), call B = [llm_mask] with gate_policy=continuous (the
new Chapter-5/6 alpha=f(u,rho) gate).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from experiment_trace import ensure_deterministic_hashing, sign_test_p_value, write_json
from run_gate1_experiments import run


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
        "n": n, "mean_diff": mean_diff, "sd_diff": sd_diff, "t": t_value,
        "df": n - 1 if n else None, "t_p_value": p_value,
        "sign_test_p": sign_test_p_value(diffs),
    }


def mean(values):
    values = list(values)
    return stats.mean(values) if values else None


def base_args(
    seed: int, methods: List[str], gate_policy: str, output_dir: Path, games: int = 200,
    forced_deceive: str = "eligible",
) -> SimpleNamespace:
    return SimpleNamespace(
        methods=methods, games=games, seed=seed, opponent_style="responsive",
        max_steps=300, sample_every=4,
        threat_fold_threshold=0.4, oracle_samples=30, oracle_beta=2.0,
        danger_threshold=1, ffr_hand_shanten=1,
        neural_opponent_model_path="Neural_opponent_model/neural_opponent_policy.pth",
        neural_opponent_device="cpu",
        mask_gate_policy=gate_policy, gate_model_path=None, gate_adapter_path=None,
        gate_max_new_tokens=8,
        mask_oracle_samples=30, mask_oracle_beta=2.0, mask_danger_threshold=1,
        mask_dir_ready_threshold=0, mask_deceive_threat_ceiling=0.5,
        # L2_rule needs forced_deceive="eligible" to match the historically
        # healthy rule-gate config (rerun_july7_gate1_scale.py); "continuous"
        # gate_policy ignores forced_deceive entirely (bypasses the whole
        # rule chain in _continuous_gate_action), so this only matters for
        # the rule-gated call.
        mask_forced_deceive=forced_deceive, mask_deceive_style="threat",
        mask_threat_allow_break_ready=False, mask_threat_max_result_shanten=0,
        mask_threat_max_shanten_regret=0, mask_threat_min_ukeire_ratio=1.0,
        mask_threat_gate_threshold=0.4, mask_threat_gate_margin=0.12,
        mask_threat_min_delta=0.03, mask_threat_gate_mode="delta_only",
        mask_threat_response_model="blend", mask_threat_response_tell_weight=0.3,
        mask_threat_tell_window=6, mask_threat_max_start_shanten=3,
        mask_threat_allow_exploit_overlap=False,
        mask_threat_require_real_target=False, mask_threat_target_max_shanten=1,
        mask_threat_target_signal="mc", mask_threat_target_prob_threshold=0.78,
        mask_log_counterfactual=True,
        mask_candidate_reranker=False, mask_candidate_scoring=False,
        mask_reranker_max_candidates=6,
        reranker_model_path=None, reranker_adapter_path=None, reranker_max_new_tokens=16,
        snapshot_oracle_samples=120, snapshot_crn_seeds=1,
        defender_threat_model="blend", defender_tell_weight=0.3, defender_tell_window=6,
        defender_learned_model_path="Defender_danger_model/danger_model.pth",
        backend="heuristic_fallback", model_path=None, adapter_path=None,
        belief_adapter_path=None, max_new_tokens=128, output_dir=output_dir,
    )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_one_seed(seed: int, out_root: Path, force: bool, games: int = 200) -> Dict[str, Any]:
    a_dir = out_root / f"seed_{seed}_A_rule"
    b_dir = out_root / f"seed_{seed}_B_continuous"
    if force or not (a_dir / "gate1_summary.json").exists():
        run(base_args(seed, ["llm_base", "llm_reactive_z", "llm_mask"], "rule", a_dir, games))
    if force or not (b_dir / "gate1_summary.json").exists():
        run(base_args(seed, ["llm_mask"], "continuous", b_dir, games))

    a_games = load_jsonl(a_dir / "gate1_games.jsonl")
    b_games = load_jsonl(b_dir / "gate1_games.jsonl")

    def net_by(games, method):
        return {g["seed"]: g["agent_net"] for g in games if g["method"] == method}

    def rate_by(games, method, key):
        return {g["seed"]: (1.0 if g[key] else 0.0) for g in games if g["method"] == method}

    net = {
        "L0": net_by(a_games, "llm_base"), "L1": net_by(a_games, "llm_reactive_z"),
        "L2_rule": net_by(a_games, "llm_mask"), "L2_continuous": net_by(b_games, "llm_mask"),
    }
    hu = {
        "L0": rate_by(a_games, "llm_base", "agent_hu"), "L1": rate_by(a_games, "llm_reactive_z", "agent_hu"),
        "L2_rule": rate_by(a_games, "llm_mask", "agent_hu"), "L2_continuous": rate_by(b_games, "llm_mask", "agent_hu"),
    }
    dealin = {
        "L0": rate_by(a_games, "llm_base", "agent_dealin"), "L1": rate_by(a_games, "llm_reactive_z", "agent_dealin"),
        "L2_rule": rate_by(a_games, "llm_mask", "agent_dealin"), "L2_continuous": rate_by(b_games, "llm_mask", "agent_dealin"),
    }
    arms = ["L0", "L1", "L2_rule", "L2_continuous"]
    seeds_matched = sorted(set.intersection(*[set(net[a].keys()) for a in arms]))
    rows = [
        {"seed": s, **{f"{a}_net": net[a][s] for a in arms},
         **{f"{a}_hu": hu[a][s] for a in arms}, **{f"{a}_dealin": dealin[a][s] for a in arms}}
        for s in seeds_matched
    ]

    with (a_dir / "gate1_summary.json").open() as f:
        a_summary = json.load(f)
    with (b_dir / "gate1_summary.json").open() as f:
        b_summary = json.load(f)
    a_e2 = a_summary.get("E2_ladder", a_summary)
    b_e2 = b_summary.get("E2_ladder", b_summary)
    dir_counts = {
        "L2_rule": a_e2["llm_mask"].get("DIR_counts"),
        "L2_continuous": b_e2["llm_mask"].get("DIR_counts"),
    }
    mode_counts = {
        "L2_rule": a_e2["llm_mask"].get("mode_counts"),
        "L2_continuous": b_e2["llm_mask"].get("mode_counts"),
    }
    return {
        "seed": seed, "n_matched_games": len(rows), "per_seed": rows,
        "dir_counts": dir_counts, "mode_counts": mode_counts,
    }


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260627, 20261627, 20262627])
    parser.add_argument("--output-dir", type=Path, default=Path("Continuous_Gate_E2_ladder"))
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in args.seeds:
        r = run_one_seed(seed, args.output_dir, args.force, args.games)
        seed_results.append(r)
        print(f"\n--- seed={seed} done: n={r['n_matched_games']} ---")
        print(f"  dir_counts={r['dir_counts']}")
        print(f"  mode_counts={r['mode_counts']}")

    all_rows = [row for r in seed_results for row in r["per_seed"]]
    arms = ["L0", "L1", "L2_rule", "L2_continuous"]
    arm_agg = {
        a: {
            "avg_net": mean(row[f"{a}_net"] for row in all_rows),
            "hu_rate": mean(row[f"{a}_hu"] for row in all_rows),
            "dealin_rate": mean(row[f"{a}_dealin"] for row in all_rows),
        }
        for a in arms
    }
    pairs = [("L2_rule", "L1"), ("L2_continuous", "L1"), ("L2_continuous", "L2_rule"), ("L1", "L0")]
    pairwise = {}
    for base_arm, target_arm in pairs:
        deltas = [row[f"{target_arm}_net"] - row[f"{base_arm}_net"] for row in all_rows]
        diff_agg = paired_stats(
            [row[f"{target_arm}_net"] for row in all_rows],
            [row[f"{base_arm}_net"] for row in all_rows],
        )
        pairwise[f"{target_arm}_minus_{base_arm}"] = diff_agg

    total_dir = {
        a: {
            "induced_dealin": sum(
                (r["dir_counts"].get(a) or {}).get("induced_dealin", 0) for r in seed_results
            ),
            "deceive_windows": sum(
                (r["dir_counts"].get(a) or {}).get("deceive_windows", 0) for r in seed_results
            ),
        }
        for a in ("L2_rule", "L2_continuous")
    }

    out = {
        "config": {"seeds": args.seeds, "games_per_seed_per_arm": 200, "backend": "heuristic_fallback"},
        "per_seed": seed_results,
        "arm_summary": arm_agg,
        "pairwise": pairwise,
        "dir_totals": total_dir,
    }
    write_json(args.output_dir / "continuous_gate_ladder_summary.json", out)
    print("\n=== arm_summary ===")
    print(json.dumps(arm_agg, ensure_ascii=False, indent=2))
    print("=== pairwise ===")
    print(json.dumps(pairwise, ensure_ascii=False, indent=2))
    print("=== dir_totals (induced_dealin / deceive_windows, summed over all seeds) ===")
    print(json.dumps(total_dir, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output_dir / 'continuous_gate_ladder_summary.json'}")


if __name__ == "__main__":
    main()

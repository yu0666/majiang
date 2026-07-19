"""Compare L2 reranking off, direct V2 reranking, and a trained reranker."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from experiment_trace import ensure_deterministic_hashing, write_json
from rerun_v2_e2_ladder_3seeds import args_for_seed, paired_stats, require_local_adapter
from run_gate1_experiments import run


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.mean(values) if values else None


def load_l2(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["E2_ladder"]["llm_mask"]


def aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    reranker = [row.get("candidate_reranker", {}) for row in rows]
    enabled = sum(int(item.get("enabled_states", 0)) for item in reranker)
    used = sum(int(item.get("used_states", 0)) for item in reranker)
    changed = sum(int(item.get("changed_actions", 0)) for item in reranker)
    return {
        "seeds": len(rows),
        "games": sum(int(row.get("games", 0)) for row in rows),
        "avg_net": mean(float(row["avg_net"]) for row in rows),
        "avg_net_seed_std": statistics.stdev(float(row["avg_net"]) for row in rows)
        if len(rows) > 1 else 0.0,
        "hu_rate": mean(float(row["hu_rate"]) for row in rows),
        "dealin_rate": mean(float(row["dealin_rate"]) for row in rows),
        "action_efficiency": {
            "avg_shanten_regret": mean(
                float(row.get("action_efficiency", {}).get("avg_shanten_regret", 0.0))
                for row in rows
            ),
            "positive_shanten_regret_rate": mean(
                float(row.get("action_efficiency", {}).get("positive_shanten_regret_rate", 0.0))
                for row in rows
            ),
        },
        "candidate_reranker": {
            "enabled_states": enabled,
            "used_states": used,
            "changed_actions": changed,
            "use_rate": used / enabled if enabled else 0.0,
            "action_change_rate": changed / used if used else 0.0,
        },
        "decision_latency_ms": {
            key: mean(float(row["decision_latency_ms"][key]) for row in rows)
            for key in ("p50", "p95", "p99")
        },
    }


def evaluate_variant(
    name: str,
    seeds: list[int],
    games: int,
    output_dir: Path,
    model_path: str,
    adapter_path: str,
    reranker_model_path: Optional[str],
    reranker_adapter_path: Optional[str],
    force: bool,
) -> list[Dict[str, Any]]:
    rows = []
    for seed in seeds:
        variant_dir = output_dir / name / f"seed_{seed}"
        summary_path = variant_dir / "gate1_summary.json"
        if summary_path.exists() and not force:
            row = load_l2(summary_path)
        else:
            run_args = args_for_seed(
                seed=seed,
                output_dir=variant_dir,
                model_path=model_path,
                adapter_path=adapter_path,
                games=games,
                max_new_tokens=64,
                candidate_reranker=True,
                candidate_scoring=True,
                reranker_max_candidates=6,
                reranker_model_path=reranker_model_path,
                reranker_adapter_path=reranker_adapter_path,
            )
            run_args.methods = ["llm_mask"]
            row = run(run_args)["E2_ladder"]["llm_mask"]
        rows.append(row)
        print(
            f"[{name}] seed={seed} net={row['avg_net']:.3f} "
            f"change={row.get('candidate_reranker', {}).get('change_rate_when_used', 0.0):.3f} "
            f"p50={row['decision_latency_ms']['p50']:.1f}ms",
            flush=True,
        )
    return rows


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("Reranker_aligned_20260715/ablation"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026080101, 2026081101])
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--trained-reranker-model-path", required=True)
    parser.add_argument("--trained-reranker-adapter-path", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    adapter_path = require_local_adapter(args.adapter_path)
    trained_adapter_path = require_local_adapter(args.trained_reranker_adapter_path)
    per_variant: Dict[str, list[Dict[str, Any]]] = {"off": []}
    for seed in args.seeds:
        baseline_path = args.baseline_dir / f"seed_{seed}" / "gate1_summary.json"
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Missing completed off baseline: {baseline_path}")
        per_variant["off"].append(load_l2(baseline_path))

    per_variant["direct_v2"] = evaluate_variant(
        "direct_v2", args.seeds, args.games, args.output_dir, args.model_path,
        adapter_path, None, None, args.force,
    )
    per_variant["trained"] = evaluate_variant(
        "trained", args.seeds, args.games, args.output_dir, args.model_path,
        adapter_path, args.trained_reranker_model_path, trained_adapter_path, args.force,
    )

    result = {
        "config": vars(args),
        "aggregate": {name: aggregate(rows) for name, rows in per_variant.items()},
        "paired_net": {
            "direct_v2_vs_off": paired_stats(
                [row["avg_net"] for row in per_variant["direct_v2"]],
                [row["avg_net"] for row in per_variant["off"]],
            ),
            "trained_vs_off": paired_stats(
                [row["avg_net"] for row in per_variant["trained"]],
                [row["avg_net"] for row in per_variant["off"]],
            ),
            "trained_vs_direct_v2": paired_stats(
                [row["avg_net"] for row in per_variant["trained"]],
                [row["avg_net"] for row in per_variant["direct_v2"]],
            ),
        },
        "per_seed": {
            name: [dict(seed=seed, **row) for seed, row in zip(args.seeds, rows)]
            for name, rows in per_variant.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "reranker_ablation_summary.json"
    write_json(output, result)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

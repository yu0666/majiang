"""Evaluate deterministic reranker actions against cached rollout rewards."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from evaluate_reranker_sft import parse_raw_action
from experiment_trace import write_json, write_jsonl
from llm_backends import LocalQwenCallable
from mask_llm import legalize_action


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {"examples": 0}
        gains = [row["selected_reward"] - row["reference_reward"] for row in group]
        regrets = [row["best_reward"] - row["selected_reward"] for row in group]
        return {
            "examples": len(group),
            "mean_selected_reward": statistics.mean(row["selected_reward"] for row in group),
            "mean_reference_reward": statistics.mean(row["reference_reward"] for row in group),
            "mean_gain_over_reference": statistics.mean(gains),
            "positive_gain_rate": statistics.mean(gain > 0 for gain in gains),
            "negative_gain_rate": statistics.mean(gain < 0 for gain in gains),
            "best_action_rate": statistics.mean(row["selected_is_best"] for row in group),
            "mean_regret": statistics.mean(regrets),
            "changed_reference_rate": statistics.mean(row["changed_reference"] for row in group),
            "json_parse_rate": statistics.mean(row["json_parsed"] for row in group),
            "raw_legal_rate": statistics.mean(row["raw_legal"] for row in group),
            "mean_latency_ms": statistics.mean(row["latency_ms"] for row in group),
        }

    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    return {
        "overall": group_summary(rows),
        "by_mode": {mode: group_summary(group) for mode, group in sorted(by_mode.items())},
    }


def evaluate(
    model_path: str,
    adapter: str,
    rows: List[Dict[str, Any]],
    max_new_tokens: int,
    candidate_scoring: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    llm = LocalQwenCallable(
        model_path=model_path,
        adapter_path=None if adapter == "base" else adapter,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    outputs = []
    for index, row in enumerate(rows):
        prompt = str(row["prompt"][-1]["content"])
        candidates = list(row["legal_actions"])
        reward_map = json.loads(row["action_rewards_json"])
        reference = str(row["reference_action"])
        started = time.perf_counter()
        if candidate_scoring:
            selected, candidate_scores = llm.rank_candidates(prompt, candidates)
            raw = selected
            raw_action, parsed = selected, True
        else:
            raw = llm(prompt)
            raw_action, parsed = parse_raw_action(raw)
            selected = legalize_action(raw, candidates)
            candidate_scores = None
        latency_ms = (time.perf_counter() - started) * 1000.0
        if selected not in candidates:
            selected = reference
        best_reward = max(float(value) for value in reward_map.values())
        selected_reward = float(reward_map[selected])
        outputs.append(
            {
                "adapter": adapter,
                "index": index,
                "state_id": row.get("state_id"),
                "mode": row["mode"],
                "reference_action": reference,
                "selected_action": selected,
                "raw_action": raw_action,
                "raw": raw,
                "json_parsed": parsed,
                "raw_legal": raw_action in candidates,
                "candidate_scores": candidate_scores,
                "changed_reference": selected != reference,
                "selected_reward": selected_reward,
                "reference_reward": float(reward_map[reference]),
                "best_reward": best_reward,
                "selected_is_best": selected_reward == best_reward,
                "latency_ms": latency_ms,
            }
        )
        if index == 0 or index + 1 == len(rows):
            print(f"[Env reward eval] adapter={adapter} {index + 1}/{len(rows)}", flush=True)

    summary = summarize(outputs)
    del llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return outputs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--adapters", nargs="+", default=["base"])
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Reranker_env_reward_eval"))
    args = parser.parse_args()

    rows = load_rows(args.data_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_outputs = []
    summaries = {}
    for adapter in args.adapters:
        outputs, summary = evaluate(
            args.model_path,
            adapter,
            rows,
            args.max_new_tokens,
            args.candidate_scoring,
        )
        all_outputs.extend(outputs)
        summaries[adapter] = summary
    result = {
        "model_path": args.model_path,
        "data_file": str(args.data_file),
        "examples": len(rows),
        "candidate_scoring": args.candidate_scoring,
        "adapters": summaries,
    }
    write_jsonl(args.output_dir / "env_reward_eval_outputs.jsonl", all_outputs)
    write_json(args.output_dir / "env_reward_eval_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

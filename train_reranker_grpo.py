"""GRPO-train the candidate reranker with cached environment rollout rewards."""

from __future__ import annotations

from unsloth import FastLanguageModel, PatchFastRL

import argparse
import json
import multiprocessing
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer


def completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def parse_completion(completion: Any) -> Tuple[Optional[str], bool]:
    stripped = completion_text(completion).strip()
    if re.fullmatch(r"d\s+[1-9][万条筒]|[hgpn]", stripped):
        return stripped, True
    try:
        parsed = json.loads(stripped)
        return str(parsed.get("action", "")).strip() or None, True
    except json.JSONDecodeError:
        match = re.search(r"d\s+[1-9][万条筒]|[hgpn]", stripped)
        return (match.group(0) if match else None), False


def env_rollout_reward_func(completions, action_rewards_json=None, **kwargs):
    rewards = []
    maps = action_rewards_json if isinstance(action_rewards_json, list) else [action_rewards_json] * len(completions)
    for completion, encoded in zip(completions, maps):
        action, _ = parse_completion(completion)
        reward_map = json.loads(encoded) if isinstance(encoded, str) else dict(encoded or {})
        rewards.append(float(reward_map.get(action, -5.0)))
    return rewards


def legal_format_reward_func(completions, legal_actions=None, **kwargs):
    rewards = []
    legal_lists = legal_actions if isinstance(legal_actions, list) else [legal_actions] * len(completions)
    for completion, legal in zip(completions, legal_lists):
        action, parsed = parse_completion(completion)
        valid = list(legal or [])
        exact_action_only = bool(
            re.fullmatch(r"d\s+[1-9][万条筒]|[hgpn]", completion_text(completion).strip())
        )
        format_reward = 0.5 if exact_action_only else (0.1 if parsed else 0.0)
        rewards.append(format_reward + (0.25 if action in valid else -2.0))
    return rewards


def load_dataset_rows(path: Path, limit: int) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            reward_map = json.loads(row["action_rewards_json"])
            if len(set(reward_map.values())) < 2:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("No non-zero-variance GRPO prompts available")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", default="Reranker_grpo_data/reranker_grpo_env.jsonl")
    parser.add_argument("--output-dir", default="qwen-reranker-grpo-env")
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--max-prompt-length", type=int, default=2900)
    parser.add_argument("--max-completion-length", type=int, default=48)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()

    effective_batch = args.batch_size * args.gradient_accumulation
    if effective_batch % args.num_generations:
        raise ValueError("batch_size * gradient_accumulation must be divisible by num_generations")

    base_dir = Path(__file__).resolve().parent
    model_path = Path(args.model_path)
    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    if not model_path.is_absolute():
        model_path = base_dir / model_path
    if not data_file.is_absolute():
        data_file = base_dir / data_file
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
        max_lora_rank=args.lora_r,
        local_files_only=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    dataset = load_dataset_rows(data_file, args.dataset_limit)
    print(f"[Reranker-GRPO] prompts={len(dataset)} model={model_path}")
    config = GRPOConfig(
        use_vllm=False,
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        logging_steps=args.logging_steps,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[env_rollout_reward_func, legal_format_reward_func],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_log_history.json").open("w", encoding="utf-8") as handle:
        json.dump(trainer.state.log_history, handle, ensure_ascii=False, indent=2)
    final_dir = output_dir / "best_grpo_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[Reranker-GRPO] saved to {final_dir}")


if __name__ == "__main__":
    main()

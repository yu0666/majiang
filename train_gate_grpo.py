"""GRPO-train the learned MASK mode gate with cached mode rewards."""

from __future__ import annotations

from unsloth import FastLanguageModel, PatchFastRL

import argparse
import json
import multiprocessing
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from datasets import Dataset

PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer


def completion_text(value: Any) -> str:
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        return str(value[-1].get("content", ""))
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def parse_mode(value: Any) -> Tuple[Optional[str], bool]:
    text = completion_text(value).strip()
    if text == "explore":
        return "exploit", False
    if text in {"exploit", "safe", "deceive"}:
        return text, True
    match = re.search(r"\b(exploit|safe|deceive)\b", text)
    return (match.group(1) if match else None), False


def mode_reward_func(completions, mode_rewards_json=None, **kwargs):
    maps = mode_rewards_json if isinstance(mode_rewards_json, list) else [mode_rewards_json] * len(completions)
    rewards = []
    for completion, encoded in zip(completions, maps):
        mode, _ = parse_mode(completion)
        reward_map = json.loads(encoded) if isinstance(encoded, str) else dict(encoded or {})
        rewards.append(float(reward_map.get(mode, -5.0)))
    return rewards


def legal_mode_reward_func(completions, legal_modes=None, **kwargs):
    mode_lists = legal_modes if isinstance(legal_modes, list) else [legal_modes] * len(completions)
    rewards = []
    for completion, modes in zip(completions, mode_lists):
        mode, exact = parse_mode(completion)
        rewards.append((0.5 if exact else 0.0) + (0.25 if mode in list(modes or []) else -2.0))
    return rewards


def load_rows(path: Path, limit: int) -> Dataset:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rewards = json.loads(row["mode_rewards_json"])
            if len(set(rewards.values())) < 2:
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("No non-zero-variance gate GRPO rows")
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-limit", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--max-prompt-length", type=int, default=3000)
    parser.add_argument("--max-completion-length", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3419)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    args = parser.parse_args()
    if args.batch_size * args.gradient_accumulation % args.num_generations:
        raise ValueError("effective batch must be divisible by num_generations")

    base = Path(__file__).resolve().parent
    model_path = Path(args.model_path)
    data_file = Path(args.data_file)
    output_dir = Path(args.output_dir)
    if not model_path.is_absolute():
        model_path = base / model_path
    if not data_file.is_absolute():
        data_file = base / data_file
    if not output_dir.is_absolute():
        output_dir = base / output_dir

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
        max_lora_rank=16,
        local_files_only=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    dataset = load_rows(data_file, args.dataset_limit)
    print(f"[Gate-GRPO] prompts={len(dataset)} model={model_path}")
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
        save_total_limit=5,
        logging_steps=5,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[mode_reward_func, legal_mode_reward_func],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_log_history.json").open("w", encoding="utf-8") as handle:
        json.dump(trainer.state.log_history, handle, ensure_ascii=False, indent=2)
    final = output_dir / "best_grpo_adapter"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[Gate-GRPO] saved to {final}")


if __name__ == "__main__":
    main()

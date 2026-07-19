"""Train a LoRA candidate reranker from high-confidence rollout labels."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict


DEFAULT_MODEL = "models/Qwen-Mahjong-V1-Mixed-SFT-Merged"
DEFAULT_DATA = "Reranker_sft_data/reranker_sft_high_confidence.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-file", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default="qwen-v1-candidate-reranker-sft")
    parser.add_argument("--min-examples", type=int, default=500)
    parser.add_argument("--min-modes", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--completion-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def validate_dataset(path: Path, min_examples: int, min_modes: int) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = 0
    modes: Counter[str] = Counter()
    label_types: Counter[str] = Counter()
    hidden_markers = ("可能已听牌(", "真实已听牌", "真实未听牌")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid messages at {path}:{line_number}")
            user_text = str(messages[-2].get("content", ""))
            assistant_text = str(messages[-1].get("content", ""))
            if any(marker in user_text for marker in hidden_markers):
                raise ValueError(f"Hidden-hand marker at {path}:{line_number}")
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            target = str(row.get("target_action") or meta.get("action") or "")
            candidate_match = re.search(r"候选动作:\s*(.*?)(?:\n|$)", user_text)
            if not candidate_match or target not in candidate_match.group(1):
                raise ValueError(f"Target is outside candidates at {path}:{line_number}")
            try:
                parsed = json.loads(assistant_text)
                assistant_action = str(parsed.get("action", "")) if isinstance(parsed, dict) else ""
            except json.JSONDecodeError:
                assistant_action = assistant_text.strip()
            if assistant_action != target:
                raise ValueError(f"Assistant target mismatch at {path}:{line_number}")
            rows += 1
            modes[str(row.get("mode") or meta.get("mode") or "unknown")] += 1
            label_types[str(row.get("label_type") or "teacher_anchor")] += 1

    summary = {
        "rows": rows,
        "modes": dict(modes),
        "label_types": dict(label_types),
        "minimum_examples": min_examples,
        "minimum_modes": min_modes,
        "ready": rows >= min_examples and len(modes) >= min_modes,
    }
    if not summary["ready"]:
        raise RuntimeError(
            "Reranker-SFT dataset is not ready: "
            f"rows={rows} (need {min_examples}), modes={dict(modes)} (need {min_modes} modes)."
        )
    return summary


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    model_path = resolve(base_dir, args.model_path)
    data_file = resolve(base_dir, args.data_file)
    output_dir = resolve(base_dir, args.output_dir)

    summary = validate_dataset(data_file, args.min_examples, args.min_modes)
    print(f"[Reranker-SFT] data validation: {json.dumps(summary, ensure_ascii=False)}")
    if args.validate_only:
        return

    # Unsloth must be imported before transformers/trl.
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_dataset("json", data_files=str(data_file), split="train")
    if args.dataset_limit is not None:
        dataset = dataset.select(range(min(args.dataset_limit, len(dataset))))

    def format_messages(examples):
        prompts = []
        completions = []
        for messages in examples["messages"]:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            completions.append(
                str(messages[-1]["content"]) + tokenizer.eos_token
            )
        return {
            "prompt": prompts,
            "completion": completions,
        }

    dataset = dataset.map(format_messages, batched=True, num_proc=args.dataset_num_proc)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=args.seed,
        report_to=args.report_to,
        max_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
        completion_only_loss=args.completion_only_loss,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[Reranker-SFT] saved adapter/tokenizer to {output_dir}")


if __name__ == "__main__":
    main()

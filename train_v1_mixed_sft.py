"""Train V1 from the original Qwen2.5-1.5B base on mixed Mahjong SFT data.

This is the clean paper route:
  Qwen2.5-1.5B-Instruct -> mixed SFT adapter (V1)

The script intentionally saves a LoRA adapter.  Use merge_lora_adapter.py to
turn the adapter into a full merged model before GRPO if needed.
"""

from __future__ import annotations

# Unsloth must be imported before transformers/trl.
from unsloth import FastLanguageModel

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


DEFAULT_BASE_MODEL = "models/qwen/Qwen2___5-1___5B-Instruct"
DEFAULT_DATA_FILE = "V1_mixed_sft_data/v1_mixed_sft_qwenbase.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE)
    parser.add_argument("--output-dir", default="qwen-v1-mixed-sft-qwen25")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def resolve_path(base_dir: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def print_data_stats(data_file: str) -> None:
    path = Path(data_file)
    rows = 0
    sources = {}
    modes = {}
    if not path.exists():
        print(f"[V1 SFT] data file not found yet: {data_file}")
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = row.get("meta", {})
            src = meta.get("v1_source", "unknown")
            mode = str(meta.get("mode", "none"))
            sources[src] = sources.get(src, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1
    print(f"[V1 SFT] rows={rows} source_counts={sources} mode_counts={modes}")


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = resolve_path(base_dir, args.model_path)
    data_file = resolve_path(base_dir, args.data_file)
    output_dir = resolve_path(base_dir, args.output_dir)

    print(f"[V1 SFT] model={model_path}")
    print(f"[V1 SFT] data={data_file}")
    print(f"[V1 SFT] output={output_dir}")
    print_data_stats(data_file)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
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

    dataset = load_dataset("json", data_files=data_file, split="train")

    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=args.dataset_num_proc)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=args.seed,
        report_to=args.report_to,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[V1 SFT] saved adapter/tokenizer to {output_dir}")


if __name__ == "__main__":
    main()

"""Train a LoRA policy that selects exploit/safe/deceive from public state."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def validate(path: Path, minimum: int) -> dict:
    rows = 0
    targets = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            target = str(row.get("target_mode", ""))
            if target not in row.get("available_modes", []):
                raise ValueError(f"Gate target outside available modes at row {rows + 1}")
            if target not in {"exploit", "safe", "deceive"}:
                raise ValueError(f"Invalid gate target: {target}")
            rows += 1
            targets[target] += 1
    if rows < minimum or len(targets) < 2:
        raise RuntimeError(f"Gate SFT data not ready: rows={rows}, targets={dict(targets)}")
    return {"rows": rows, "targets": dict(targets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-examples", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3417)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

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
    print(f"[Gate-SFT] validation={json.dumps(validate(data_file, args.min_examples), ensure_ascii=False)}")

    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    dataset = load_dataset("json", data_files=str(data_file), split="train")

    def format_messages(examples):
        prompts, completions = [], []
        for messages in examples["messages"]:
            prompts.append(tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
            completions.append(str(messages[-1]["content"]) + tokenizer.eos_token)
        return {"prompt": prompts, "completion": completions}

    dataset = dataset.map(format_messages, batched=True, num_proc=4)
    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=args.save_steps,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=args.seed,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        completion_only_loss=True,
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=config)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[Gate-SFT] saved to {output_dir}")


if __name__ == "__main__":
    main()

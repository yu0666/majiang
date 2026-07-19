"""Merge a LoRA adapter into a full model directory.

Typical V1 merge:
  python merge_lora_adapter.py \
    --base-model models/qwen/Qwen2___5-1___5B-Instruct \
    --adapter qwen-v1-mixed-sft-qwen25 \
    --output-dir models/Qwen-Mahjong-V1-Mixed-SFT-Merged
"""

from __future__ import annotations

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_path(base_dir: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/qwen/Qwen2___5-1___5B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-map", default="cpu", help="Use cpu for safest merge, or auto/cuda for faster merge.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def dtype_from_arg(value: str):
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_model = resolve_path(base_dir, args.base_model)
    adapter = resolve_path(base_dir, args.adapter)
    output_dir = resolve_path(base_dir, args.output_dir)

    print(f"[merge] base={base_model}")
    print(f"[merge] adapter={adapter}")
    print(f"[merge] output={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_from_arg(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[merge] saved merged model to {output_dir}")


if __name__ == "__main__":
    main()

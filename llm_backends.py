"""
Optional LLM callables for MASK experiments.

The default Gate1 path does not load a model.  Use this module only when a run
explicitly requests a local model backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional


LLMCallable = Callable[[str], str]


class LocalQwenCallable:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是四川麻将 MASK 智能体。严格遵守用户要求的输出格式；"
                    "如果用户要求 JSON，只输出 JSON，不要解释。"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated)
        ]
        return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def build_llm_callable(
    backend: str,
    repo_dir: Path,
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    max_new_tokens: int = 128,
) -> Optional[LLMCallable]:
    if backend == "heuristic_fallback":
        return None
    if backend != "local_qwen":
        raise ValueError(f"Unknown LLM backend: {backend}")

    resolved_model = model_path or str(repo_dir / "models" / "Qwen-Mahjong-V3-Merged")
    resolved_adapter = adapter_path or None
    return LocalQwenCallable(
        model_path=resolved_model,
        adapter_path=resolved_adapter,
        max_new_tokens=max_new_tokens,
    )

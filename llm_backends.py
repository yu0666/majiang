"""
Optional LLM callables for MASK experiments.

The default Gate1 path does not load a model.  Use this module only when a run
explicitly requests a local model backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from prompt_builder import ACTION_SYSTEM_PROMPT


LLMCallable = Callable[[str], str]


class LocalQwenCallable:
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or ACTION_SYSTEM_PROMPT

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

    def _render_messages(
        self,
        prompt: str,
        reranker: bool = False,
        system_prompt: Optional[str] = None,
    ) -> str:
        system_content = system_prompt or (
            "你是四川麻将候选动作重排器。模式由外层规则固定，只能从候选动作中选择。"
            if reranker
            else self.system_prompt
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def rank_candidates(
        self,
        prompt: str,
        candidates: list[str],
        system_prompt: Optional[str] = None,
    ) -> tuple[str, dict[str, float]]:
        """Rank constrained actions by mean conditional token log-probability."""
        if not candidates:
            raise ValueError("rank_candidates requires at least one candidate")
        rendered = self._render_messages(prompt, reranker=True, system_prompt=system_prompt)
        prefix_ids = self.tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]
        scores = {}
        for action in candidates:
            action_ids = self.tokenizer(action, add_special_tokens=False)["input_ids"]
            if not action_ids:
                scores[action] = float("-inf")
                continue
            input_ids = self.torch.tensor(
                [prefix_ids + action_ids],
                dtype=self.torch.long,
                device=self.model.device,
            )
            with self.torch.no_grad():
                # The first action token is predicted by the last prefix
                # position. Keep only that row plus the action-token rows.
                logits = self.model(
                    input_ids=input_ids,
                    logits_to_keep=len(action_ids) + 1,
                ).logits[0, : len(action_ids)]
            token_scores = []
            for offset, token_id in enumerate(action_ids):
                token_scores.append(
                    self.torch.log_softmax(logits[offset].float(), dim=-1)[token_id]
                )
            scores[action] = float(self.torch.stack(token_scores).mean().item())
        selected = max(candidates, key=lambda action: scores[action])
        return selected, scores

    def __call__(self, prompt: str) -> str:
        text = self._render_messages(prompt)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                **generation_kwargs,
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
    temperature: float = 0.1,
    system_prompt: Optional[str] = None,
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
        temperature=temperature,
        system_prompt=system_prompt,
    )

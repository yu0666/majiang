"""Train V2 with GRPO from the V1 mixed-SFT model.

Default route:
  Qwen2.5 base -> V1 mixed SFT -> merged V1 -> V2 GRPO adapter

This V2 trainer is prompt-level GRPO with balanced domain rewards.  It is
designed as the bridge before a full environment/rerank GRPO trainer:
  * strict executable-action format reward;
  * legal-action reward from the prompt action mask;
  * missing-suit and shanten-progress discard reward;
  * a teacher action anchor so GRPO does not drift away from V1's SFT behavior;
  * diversified group sampling and an explicit reward-variance diagnostic.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from rule_engine import ShantenCalculator
from tiles import Suit, parse_tile as parse_game_tile

PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer


DEFAULT_MODEL = "models/Qwen-Mahjong-V1-Mixed-SFT-Merged"
DEFAULT_DATA = "V1_mixed_sft_data/v1_mixed_sft_qwenbase.jsonl"
VALID_MODES = {"exploit", "safe", "deceive"}
SUIT_BY_TEXT = {"万": Suit.WAN, "条": Suit.TIAO, "筒": Suit.TONG}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-file", default=DEFAULT_DATA)
    parser.add_argument("--output-dir", default="qwen-v2-grpo-diverse-v1-l2")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1800)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--top-k", type=int, default=100,
                        help="Set <=0 to disable top-k filtering.")
    parser.add_argument("--min-p", type=float, default=0.01,
                        help="Set <=0 to disable min-p filtering.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01,
                        help="Small KL penalty that anchors GRPO to the V1 policy.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--dataset-limit", type=int, default=4000)
    parser.add_argument("--min-legal-actions", type=int, default=2,
                        help="Skip forced states with fewer legal actions; they provide no GRPO preference signal.")
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-completions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--zero-std-warning-threshold", type=float, default=0.5,
                        help="Warn when the mean zero-reward-std group fraction exceeds this value.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=max(1, multiprocessing.cpu_count() - 2))
    return parser.parse_args()


def resolve_path(base_dir: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.join(base_dir, value)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for GRPO.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.min_p > 1:
        raise ValueError("--min-p must be <= 1.")

    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = args.batch_size * args.gradient_accumulation * world_size
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "Effective batch size must be divisible by --num-generations: "
            f"batch_size({args.batch_size}) * gradient_accumulation({args.gradient_accumulation}) "
            f"* world_size({world_size}) = {effective_batch}, num_generations={args.num_generations}."
        )


def summarize_training_signal(log_history: List[Dict[str, Any]], warning_threshold: float) -> Dict[str, Any]:
    zero_std_fractions = [
        float(row["frac_reward_zero_std"])
        for row in log_history
        if isinstance(row.get("frac_reward_zero_std"), (int, float))
    ]
    reward_stds = [
        float(row["reward_std"])
        for row in log_history
        if isinstance(row.get("reward_std"), (int, float))
    ]
    summary = {
        "logged_steps": len(zero_std_fractions),
        "mean_frac_reward_zero_std": (
            sum(zero_std_fractions) / len(zero_std_fractions) if zero_std_fractions else None
        ),
        "mean_reward_std": sum(reward_stds) / len(reward_stds) if reward_stds else None,
        "max_reward_std": max(reward_stds) if reward_stds else None,
        "warning_threshold": warning_threshold,
    }
    print(f"[V2 GRPO] reward diversity: {json.dumps(summary, ensure_ascii=False)}")
    mean_zero = summary["mean_frac_reward_zero_std"]
    if mean_zero is not None and mean_zero >= warning_threshold:
        print(
            "[V2 GRPO][WARNING] Reward diversity is still too low. "
            "Do not treat this adapter as a valid GRPO improvement before rerunning a diversity smoke test."
        )
    return summary


def extract_text(item: Any) -> str:
    if isinstance(item, list) and item and isinstance(item[-1], dict):
        return str(item[-1].get("content", ""))
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(item)


def parse_json_completion(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_action(completion: Any) -> Tuple[str, Optional[str], bool]:
    text = extract_text(completion).strip()
    parsed = parse_json_completion(text)
    if parsed is not None:
        action = str(parsed.get("action", "")).strip()
        mode = str(parsed.get("mode", "")).strip()
        return action, mode, True
    match = re.search(r"(d\s+[1-9][万筒条]|[hgpn])", text)
    if match:
        return match.group(1).replace("d  ", "d "), None, False
    return text, None, False


def legal_actions_from_prompt(prompt_text: str) -> List[str]:
    match = re.search(r"合法动作[:：]\s*(.*?)(?:\n|$)", prompt_text)
    if not match:
        return []
    raw = match.group(1)
    return [x.strip() for x in re.split(r"[,，]", raw) if x.strip()]


def parse_tile_text(tile_str: str) -> Tuple[Optional[int], Optional[str]]:
    match = re.search(r"([1-9])([万筒条])", tile_str)
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def hand_from_prompt(prompt_text: str) -> str:
    match = re.search(r"我的手牌[:：]\s*(.*?)(?:\n|$)", prompt_text)
    return match.group(1).strip() if match else ""


def missing_suit_from_prompt(prompt_text: str) -> Optional[str]:
    match = re.search(r"我的定缺[:：]\s*([万筒条])", prompt_text)
    return match.group(1) if match else None


def mode_hint_from_prompt(prompt_text: str) -> Optional[str]:
    match = re.search(r"['\"]mode_hint['\"]\s*:\s*['\"](exploit|safe|deceive)['\"]", prompt_text)
    return match.group(1) if match else None


def risk_budget_from_prompt(prompt_text: str) -> Optional[float]:
    match = re.search(r"['\"]risk_budget['\"]\s*:\s*([-+]?\d+(?:\.\d+)?)", prompt_text)
    return float(match.group(1)) if match else None


def confidence_values_from_prompt(prompt_text: str) -> List[float]:
    return [
        float(value)
        for value in re.findall(r"['\"]tenpai_confidence['\"]\s*:\s*([-+]?\d+(?:\.\d+)?)", prompt_text)
    ]


def tile_texts_from_hand(hand_tiles_str: str) -> List[str]:
    return re.findall(r"[1-9][万筒条]", hand_tiles_str)


def tile_objects(tile_texts: List[str]):
    tiles = []
    for tile_text in tile_texts:
        try:
            tiles.append(parse_game_tile(tile_text))
        except Exception:
            continue
    return tiles


def shanten_for_tile_texts(tile_texts: List[str], missing_suit_text: Optional[str]) -> Optional[int]:
    tiles = tile_objects(tile_texts)
    if not tiles:
        return None
    missing_suit = SUIT_BY_TEXT.get(missing_suit_text or "")
    return int(ShantenCalculator.calculate_shanten(tiles, missing_suit))


def discard_shanten_profile(prompt_text: str) -> Tuple[Optional[int], Dict[str, int]]:
    hand_tiles = tile_texts_from_hand(hand_from_prompt(prompt_text))
    missing_suit = missing_suit_from_prompt(prompt_text)
    current_shanten = shanten_for_tile_texts(hand_tiles, missing_suit)
    results: Dict[str, int] = {}

    for action in legal_actions_from_prompt(prompt_text):
        if not action.startswith("d "):
            continue
        tile_text = action[2:].strip()
        if tile_text not in hand_tiles:
            continue
        remaining = list(hand_tiles)
        remaining.remove(tile_text)
        result_shanten = shanten_for_tile_texts(remaining, missing_suit)
        if result_shanten is not None:
            results[action] = result_shanten
    return current_shanten, results


def get_optional_list_value(values: Any, index: int) -> Optional[Any]:
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


def tactical_discard_score(played_tile: str, hand_tiles_str: str, prompt_text: str) -> float:
    played_val, played_suit = parse_tile_text(played_tile)
    if played_val is None or played_suit is None:
        return -5.0

    hand_tiles = hand_tiles_str.split()
    same_suit_vals: List[int] = []
    for tile in hand_tiles:
        val, suit = parse_tile_text(tile)
        if suit == played_suit and val is not None:
            same_suit_vals.append(val)

    if played_val in same_suit_vals:
        same_suit_vals.remove(played_val)

    targets = set()
    if played_val in same_suit_vals:
        targets.add(played_val)
    if played_val - 1 in same_suit_vals:
        if played_val - 2 >= 1:
            targets.add(played_val - 2)
        if played_val + 1 <= 9:
            targets.add(played_val + 1)
    if played_val + 1 in same_suit_vals:
        if played_val - 1 >= 1:
            targets.add(played_val - 1)
        if played_val + 2 <= 9:
            targets.add(played_val + 2)
    if played_val - 2 in same_suit_vals:
        targets.add(played_val - 1)
    if played_val + 2 in same_suit_vals:
        targets.add(played_val + 1)

    if not targets:
        return 6.0

    visible_targets = 0
    for value in targets:
        visible_targets += prompt_text.count(f"{value}{played_suit}")
    if visible_targets >= (len(targets) * 4) - 1:
        return 10.0
    quality_penalty = (4 * len(targets) - visible_targets) * 1.5
    return -7.0 - quality_penalty


def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        action, _, _ = extract_action(completion)
        if re.fullmatch(r"(d [1-9][万筒条]|h|g|p|n)", action):
            reward = 2.0
        else:
            reward = -5.0
        rewards.append(reward)
    return rewards


def legal_action_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt_msgs, completion in zip(prompts, completions):
        prompt_text = extract_text(prompt_msgs)
        action, _, _ = extract_action(completion)
        legal = legal_actions_from_prompt(prompt_text)
        if not legal:
            rewards.append(0.0)
        elif action in legal:
            rewards.append(5.0)
        else:
            rewards.append(-12.0)
    return rewards


def mahjong_domain_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt_msgs, completion in zip(prompts, completions):
        prompt_text = extract_text(prompt_msgs)
        action, _, _ = extract_action(completion)
        legal = legal_actions_from_prompt(prompt_text)
        reward = 0.0

        if action == "h":
            reward += 35.0 if "h" in legal else -15.0
        elif action == "g":
            reward += 5.0 if "g" in legal else -8.0
        elif action == "p":
            reward += 3.0 if "p" in legal else -8.0
        elif action == "n":
            if "h" in legal:
                reward -= 30.0
            elif "g" in legal or "p" in legal:
                reward -= 3.0
            else:
                reward += 1.0
        elif action.startswith("d "):
            played_tile = action[2:].strip()
            played_val, played_suit = parse_tile_text(played_tile)
            missing_suit = missing_suit_from_prompt(prompt_text)
            hand_tiles_str = hand_from_prompt(prompt_text)
            if played_val is None or played_suit is None:
                reward -= 5.0
            elif missing_suit and hand_tiles_str:
                has_missing = missing_suit in hand_tiles_str
                if has_missing:
                    reward += 12.0 if played_suit == missing_suit else -30.0
                else:
                    reward += tactical_discard_score(played_tile, hand_tiles_str, prompt_text) * 0.25

            current_shanten, shanten_results = discard_shanten_profile(prompt_text)
            chosen_shanten = shanten_results.get(action)
            if chosen_shanten is not None and shanten_results:
                best_shanten = min(shanten_results.values())
                worst_shanten = max(shanten_results.values())
                if chosen_shanten == best_shanten:
                    reward += 8.0
                else:
                    reward -= 6.0 * (chosen_shanten - best_shanten)

                if current_shanten is not None:
                    delta = current_shanten - chosen_shanten
                    if delta > 0:
                        reward += 5.0 * delta
                    elif delta < 0:
                        reward += 8.0 * delta

                if chosen_shanten <= 0:
                    reward += 7.0
                elif chosen_shanten == 1:
                    reward += 3.0
                elif worst_shanten > best_shanten and chosen_shanten == worst_shanten:
                    reward -= 2.0

        rewards.append(reward)
    return rewards


def teacher_anchor_reward_func(prompts, completions, reference_action=None, reference_mode=None, v1_source=None, **kwargs):
    rewards = []
    for idx, completion in enumerate(completions):
        action, _, _ = extract_action(completion)
        ref_action = get_optional_list_value(reference_action, idx)
        source = str(get_optional_list_value(v1_source, idx) or "")

        reward = 0.0
        if ref_action:
            if action == ref_action:
                reward += 3.0 if source == "mask_l2_teacher" else 2.0
            elif ref_action == "h":
                reward -= 12.0
            elif str(ref_action).startswith("d ") and action.startswith("d "):
                reward -= 1.0
            else:
                reward -= 2.0

        rewards.append(reward)
    return rewards


def load_prompts_dataset(data_file: str, limit: int, min_legal_actions: int) -> Dataset:
    prompts = []
    skipped_forced = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            messages = row["messages"][:2]
            prompt_text = extract_text(messages)
            if len(set(legal_actions_from_prompt(prompt_text))) < min_legal_actions:
                skipped_forced += 1
                continue
            assistant_text = row["messages"][-1].get("content", "")
            reference_action, reference_mode, _ = extract_action(assistant_text)
            meta = row.get("meta", {})
            prompts.append({
                "prompt": messages,
                "reference_action": reference_action,
                "reference_mode": reference_mode or str(meta.get("mode", "")),
                "v1_source": str(meta.get("v1_source", "unknown")),
            })
            if limit > 0 and len(prompts) >= limit:
                break
    print(
        f"[V2 GRPO] dataset kept={len(prompts)}, skipped_forced={skipped_forced}, "
        f"min_legal_actions={min_legal_actions}"
    )
    return Dataset.from_list(prompts)


def main() -> None:
    args = parse_args()
    validate_args(args)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = resolve_path(base_dir, args.model_path)
    data_file = resolve_path(base_dir, args.data_file)
    output_dir = resolve_path(base_dir, args.output_dir)

    print(f"[V2 GRPO] model={model_path}")
    print(f"[V2 GRPO] data={data_file}")
    print(f"[V2 GRPO] output={output_dir}")
    print(
        "[V2 GRPO] sampling="
        f"generations={args.num_generations}, temperature={args.temperature}, "
        f"top_p={args.top_p}, top_k={args.top_k}, min_p={args.min_p}, beta={args.beta}"
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
        max_lora_rank=args.lora_r,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    dataset = load_prompts_dataset(data_file, args.dataset_limit, args.min_legal_actions)
    print(f"[V2 GRPO] prompts={len(dataset)}")

    training_args = GRPOConfig(
        use_vllm=False,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        min_p=args.min_p if args.min_p > 0 else None,
        repetition_penalty=args.repetition_penalty,
        beta=args.beta,
        max_steps=args.max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=10,
        logging_steps=args.logging_steps,
        log_completions=args.log_completions,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            legal_action_reward_func,
            mahjong_domain_reward_func,
            teacher_anchor_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    diversity_summary = summarize_training_signal(
        trainer.state.log_history,
        args.zero_std_warning_threshold,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "reward_diversity_summary.json"), "w", encoding="utf-8") as f:
        json.dump(diversity_summary, f, ensure_ascii=False, indent=2)
    final_dir = os.path.join(output_dir, "best_grpo_adapter")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[V2 GRPO] saved adapter/tokenizer to {final_dir}")


if __name__ == "__main__":
    main()

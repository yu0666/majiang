"""Train a small neural exploit/safe/deceive gate from mode-oracle rollouts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from experiment_trace import write_json
from neural_gate_policy import (
    FEATURE_DIM,
    MODE_TO_INDEX,
    MODES,
    NORMALIZATION_STD_FLOOR,
    NORMALIZED_FEATURE_CLIP,
    NeuralGateNet,
    available_modes_from_row,
    features_from_gate_prompt,
    prepare_normalization_tensors,
    target_from_rewards,
)
from risk_aware_reward import add_reward_arguments, reward_config, score_rollouts


class GateDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], mean: np.ndarray, std: np.ndarray):
        self.rows = rows
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), NORMALIZATION_STD_FLOOR)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        x = (np.asarray(row["features"], dtype=np.float32) - self.mean) / self.std
        x = np.clip(x, -NORMALIZED_FEATURE_CLIP, NORMALIZED_FEATURE_CLIP)
        y = int(row["target_index"])
        reward = np.asarray(row["reward_vector"], dtype=np.float32)
        mask = np.asarray(row["mode_mask"], dtype=np.float32)
        weight = float(row["weight"])
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


def load_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []
    labels: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source = json.loads(line)
            available_modes = available_modes_from_row(source)
            if not available_modes:
                skipped["no_available_modes"] += 1
                continue
            reward_scores: Dict[str, float] = {}
            if isinstance(source.get("mode_rewards"), dict):
                for mode, value in source["mode_rewards"].items():
                    if mode in available_modes:
                        reward_scores[str(mode)] = float(value)
            else:
                for evaluation in source.get("mode_evaluations", []):
                    mode = str(evaluation.get("mode"))
                    if mode not in available_modes:
                        continue
                    scored = score_rollouts(evaluation.get("rollouts", []), args)
                    reward_scores[mode] = float(scored["risk_adjusted_score"])
            if not reward_scores and source.get("target_mode") in available_modes:
                # Pure teacher-distillation rows do not need counterfactual rollout
                # rewards.  Give the teacher label a margin so default filtering
                # keeps these rows.
                target_mode = str(source["target_mode"])
                reward_scores = {
                    mode: (args.teacher_reward_margin if mode == target_mode else 0.0)
                    for mode in available_modes
                }
            if len(reward_scores) < 1:
                skipped["no_rewards"] += 1
                continue
            if len(reward_scores) >= 2 and max(reward_scores.values()) - min(reward_scores.values()) < args.min_reward_range:
                skipped["low_reward_range"] += 1
                continue
            target = target_from_rewards(source, reward_scores)
            if target not in available_modes:
                skipped["target_unavailable"] += 1
                continue
            try:
                if isinstance(source.get("features"), list):
                    features = [float(value) for value in source["features"]]
                    if len(features) != FEATURE_DIM:
                        raise ValueError(f"expected {FEATURE_DIM} online features, got {len(features)}")
                else:
                    features = features_from_gate_prompt(str(source["prompt"]))
            except Exception:
                skipped["feature_parse_error"] += 1
                continue
            reward_vector = [-100.0] * len(MODES)
            mode_mask = [0.0] * len(MODES)
            for mode in available_modes:
                idx = MODE_TO_INDEX[mode]
                mode_mask[idx] = 1.0
                reward_vector[idx] = float(reward_scores.get(mode, min(reward_scores.values())))
            reward_range = max(v for v, m in zip(reward_vector, mode_mask) if m) - min(
                v for v, m in zip(reward_vector, mode_mask) if m
            )
            rows.append(
                {
                    "features": features,
                    "target": target,
                    "target_index": MODE_TO_INDEX[target],
                    "available_modes": available_modes,
                    "reward_vector": reward_vector,
                    "mode_mask": mode_mask,
                    "weight": 1.0 + min(5.0, max(0.0, reward_range) / max(1.0, args.reward_weight_scale)),
                    "state_id": source.get("state_id", f"line_{line_number}"),
                }
            )
            labels[target] += 1
            if args.limit > 0 and len(rows) >= args.limit:
                break
    rng.shuffle(rows)
    summary = {
        "rows": len(rows),
        "labels": dict(labels),
        "skipped": dict(skipped),
        "reward": reward_config(args),
    }
    return rows, summary


def evaluate(model: NeuralGateNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    reward_sum = 0.0
    with torch.no_grad():
        for x, y, reward, mask, weight in loader:
            x, y, reward, mask, weight = x.to(device), y.to(device), reward.to(device), mask.to(device), weight.to(device)
            logits = model(x)
            masked_logits = logits.masked_fill(mask <= 0, -1e9)
            loss = F.cross_entropy(masked_logits, y, reduction="none")
            probs = torch.softmax(masked_logits, dim=-1)
            chosen = torch.argmax(probs, dim=-1)
            chosen_reward = reward.gather(1, chosen.unsqueeze(1)).squeeze(1)
            loss_sum += float((loss * weight).sum().item())
            reward_sum += float(chosen_reward.sum().item())
            correct += int((chosen == y).sum().item())
            total += int(y.numel())
    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "avg_cached_reward": reward_sum / max(1, total),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, data_summary = load_rows(args)
    if len(rows) < args.min_examples:
        raise RuntimeError(f"Not enough neural gate rows: {len(rows)} < {args.min_examples}; summary={data_summary}")

    feature_matrix = np.asarray([row["features"] for row in rows], dtype=np.float32)
    raw_mean = feature_matrix.mean(axis=0)
    raw_std = feature_matrix.std(axis=0)
    if args.normalization_mode == "raw":
        mean_tensor = torch.zeros(FEATURE_DIM, dtype=torch.float32)
        std_tensor = torch.ones(FEATURE_DIM, dtype=torch.float32)
        near_constant = torch.zeros(FEATURE_DIM, dtype=torch.bool)
    else:
        mean_tensor, std_tensor, near_constant = prepare_normalization_tensors(
            torch.tensor(raw_mean, dtype=torch.float32),
            torch.tensor(raw_std, dtype=torch.float32),
        )
    mean = mean_tensor.numpy()
    std = std_tensor.numpy()

    dataset = GateDataset(rows, mean, std)
    eval_size = max(1, int(len(dataset) * args.eval_ratio))
    train_size = len(dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = NeuralGateNet(hidden_dims=tuple(args.hidden_dims)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_metric = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_items = 0
        for x, y, reward, mask, weight in train_loader:
            x, y, reward, mask, weight = x.to(device), y.to(device), reward.to(device), mask.to(device), weight.to(device)
            logits = model(x)
            masked_logits = logits.masked_fill(mask <= 0, -1e9)
            ce_loss = F.cross_entropy(masked_logits, y, reduction="none")
            loss = (ce_loss * weight).mean()
            if args.reward_pg_weight > 0:
                probs = torch.softmax(masked_logits, dim=-1)
                reward_centered = reward - reward.masked_fill(mask <= 0, 0.0).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                expected_reward = (probs * reward_centered * mask).sum(dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-8)) * mask).sum(dim=-1)
                loss = loss - args.reward_pg_weight * expected_reward.mean() - args.entropy_weight * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += float(loss.item()) * int(y.numel())
            train_items += int(y.numel())

        metrics = evaluate(model, eval_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(1, train_items)
        history.append(metrics)
        score = metrics["avg_cached_reward"] + 10.0 * metrics["accuracy"]
        if score > best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[NeuralGate] epoch={epoch:03d} train_loss={metrics['train_loss']:.4f} "
            f"eval_acc={metrics['accuracy']:.3f} eval_reward={metrics['avg_cached_reward']:.3f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.cpu().state_dict(),
        "input_dim": FEATURE_DIM,
        "hidden_dims": list(args.hidden_dims),
        "feature_mean": mean.tolist(),
        "feature_std": np.maximum(std, NORMALIZATION_STD_FLOOR).tolist(),
        "raw_feature_mean": raw_mean.tolist(),
        "raw_feature_std": raw_std.tolist(),
        "near_constant_features": near_constant.numpy().astype(bool).tolist(),
        "near_constant_feature_count": int(near_constant.sum().item()),
        "normalization_std_floor": NORMALIZATION_STD_FLOOR,
        "normalized_feature_clip": NORMALIZED_FEATURE_CLIP,
        "normalization_mode": args.normalization_mode,
        "modes": list(MODES),
        "data_summary": data_summary,
        "history": history,
        "args": vars(args),
    }
    model_path = args.output_dir / "neural_gate_policy.pth"
    torch.save(checkpoint, model_path)
    summary = {
        "model_path": str(model_path),
        "data": data_summary,
        "best_metric": best_metric,
        "final_eval": evaluate(model.to(device), eval_loader, device),
        "history": history,
    }
    write_json(args.output_dir / "neural_gate_training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-examples", type=int, default=200)
    parser.add_argument("--min-reward-range", type=float, default=2.0)
    parser.add_argument("--reward-weight-scale", type=float, default=20.0)
    parser.add_argument("--eval-ratio", type=float, default=0.15)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reward-pg-weight", type=float, default=0.02)
    parser.add_argument("--teacher-reward-margin", type=float, default=10.0)
    parser.add_argument("--normalization-mode", choices=["raw", "standard"], default="raw")
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=3421)
    add_reward_arguments(parser)
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

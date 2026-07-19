"""Train a neural opponent policy from responsive+learned defender imitation data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiment_trace import write_json
from neural_opponent_policy import ACTION_SPACE, ACTION_TO_INDEX, FEATURE_VERSION, NeuralOpponentNet


class OpponentDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = torch.tensor(row["features"], dtype=torch.float32)
        y = torch.tensor(ACTION_TO_INDEX[row["action"]], dtype=torch.long)
        legal = torch.zeros(len(ACTION_SPACE), dtype=torch.bool)
        for action in row["legal_actions"]:
            if action in ACTION_TO_INDEX:
                legal[ACTION_TO_INDEX[action]] = True
        return x, y, legal


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("feature_version") != FEATURE_VERSION:
                raise ValueError(f"Feature version mismatch in {path}: {row.get('feature_version')}")
            if row.get("action") not in ACTION_TO_INDEX:
                continue
            if row.get("action") not in row.get("legal_actions", []):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid rows loaded from {path}")
    dim = len(rows[0]["features"])
    bad = [idx for idx, row in enumerate(rows) if len(row["features"]) != dim]
    if bad:
        raise ValueError(f"Inconsistent feature dimensions; first bad row index={bad[0]}")
    return rows


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    legal_correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for x, y, legal in loader:
            x = x.to(device)
            y = y.to(device)
            legal = legal.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y).item())
            pred = torch.argmax(logits, dim=-1)
            correct += int((pred == y).sum().item())
            masked_logits = logits.masked_fill(~legal, -1e9)
            legal_pred = torch.argmax(masked_logits, dim=-1)
            legal_correct += int((legal_pred == y).sum().item())
            total += int(y.numel())
    return {
        "loss": loss_sum / max(1, total),
        "raw_acc": correct / max(1, total),
        "legal_masked_acc": legal_correct / max(1, total),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    rows = load_rows(args.data_file)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    split = int(len(rows) * (1.0 - args.eval_ratio))
    train_rows = rows[:split]
    eval_rows = rows[split:] or rows[: min(256, len(rows))]
    input_dim = len(rows[0]["features"])

    train_loader = DataLoader(OpponentDataset(train_rows), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(OpponentDataset(eval_rows), batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    hidden_dims = tuple(int(value) for value in args.hidden_dims)
    model = NeuralOpponentNet(input_dim, len(ACTION_SPACE), hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_state = None
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y, _legal in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
        metrics = evaluate(model, eval_loader, device)
        metrics["epoch"] = float(epoch)
        metrics["train_loss"] = total_loss / max(1, total)
        history.append(metrics)
        print(
            f"[train] epoch={epoch}/{args.epochs} train_loss={metrics['train_loss']:.4f} "
            f"eval_loss={metrics['loss']:.4f} legal_acc={metrics['legal_masked_acc']:.4f}",
            flush=True,
        )
        if metrics["legal_masked_acc"] > best_metric:
            best_metric = metrics["legal_masked_acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "neural_opponent_policy.pth"
    torch.save(
        {
            "model_state": best_state or model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "actions": ACTION_SPACE,
            "feature_version": FEATURE_VERSION,
            "source_data": str(args.data_file),
        },
        checkpoint_path,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "data_file": str(args.data_file),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "best_legal_masked_acc": best_metric,
        "history": history,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=Path("Neural_opponent_data/responsive_learned_teacher.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("Neural_opponent_model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    summary = train(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

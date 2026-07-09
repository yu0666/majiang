"""Train the ResponsiveDefender 'learned' danger-perception model.

Trains defender_danger_model.StyleLSTMAttention-based binary classifier on the
per-P0-discard rows from generate_defender_danger_data.py, mirroring
opponent_classifier.OpponentStyleClassifier.train_model()'s Adam+StepLR+
best-val-checkpoint loop. After training, evaluates on the held-out eval split
(same game-level split produced by the data generator, not re-split here) using
experiment_trace.py's AUC/Brier/ECE/sign-test primitives, comparing the learned
model against the discard_tell and mc reference scores recorded on the same rows
-- so the report is directly comparable to the H1 gate (H1_seed_aggregate/).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from experiment_trace import (
    binary_auc,
    brier_score,
    calibration_error,
    ensure_deterministic_hashing,
    sign_test_p_value,
    write_json,
)

from defender_danger_model import DangerSequenceDataset, WINDOW_SIZE, build_model


def load_games(path: Path) -> List[Dict[str, Any]]:
    games = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def games_to_sequences(games: List[Dict[str, Any]]) -> Tuple[List[List[List[float]]], List[List[int]]]:
    feature_sequences = [[s["features"] for s in g["steps"]] for g in games]
    label_sequences = [[s["label"] for s in g["steps"]] for g in games]
    return feature_sequences, label_sequences


def flat_rows(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for g in games for row in g["steps"]]


def paired_sign_test(labels: List[int], learned: List[float], ref: List[float]) -> Dict[str, Any]:
    deltas = [(r - y) ** 2 - (m - y) ** 2 for y, m, r in zip(labels, learned, ref)]
    n = len(deltas)
    return {
        "n": n,
        "mean_brier_gain": (sum(deltas) / n) if n else 0.0,
        "win_rate": (sum(1 for d in deltas if d > 0) / n) if n else 0.0,
        "sign_test_p": sign_test_p_value(deltas),
    }


def evaluate(model, device: str, eval_games: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = flat_rows(eval_games)
    feats, labels = games_to_sequences(eval_games)
    ds = DangerSequenceDataset(feats, labels, window_size=WINDOW_SIZE)
    model.eval()
    learned_probs: List[float] = []
    with torch.no_grad():
        for start in range(0, len(ds), 512):
            batch = torch.stack([ds[i][0] for i in range(start, min(start + 512, len(ds)))]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1].tolist()
            learned_probs.extend(probs)

    y = [row["label"] for row in rows]
    tell = [row["tell_score"] for row in rows]
    mc_rows = [(row["label"], row["mc_score"]) for row in rows if row["mc_score"] is not None]

    report: Dict[str, Any] = {
        "n_eval": len(rows),
        "learned": {
            "auc": binary_auc(y, learned_probs),
            "brier": brier_score(y, learned_probs),
            "ece": calibration_error(y, learned_probs),
        },
        "discard_tell": {
            "auc": binary_auc(y, tell),
            "brier": brier_score(y, tell),
            "ece": calibration_error(y, tell),
        },
        "vs_discard_tell": paired_sign_test(y, learned_probs, tell),
    }
    if mc_rows:
        y_mc = [r[0] for r in mc_rows]
        mc = [r[1] for r in mc_rows]
        learned_for_mc = [learned_probs[i] for i, row in enumerate(rows) if row["mc_score"] is not None]
        report["mc"] = {
            "auc": binary_auc(y_mc, mc),
            "brier": brier_score(y_mc, mc),
            "ece": calibration_error(y_mc, mc),
        }
        report["vs_mc"] = paired_sign_test(y_mc, learned_for_mc, mc)
    return report


def train(
    train_games: List[Dict[str, Any]],
    eval_games: List[Dict[str, Any]],
    epochs: int,
    batch_size: int,
    lr: float,
    save_path: Path,
    device: str,
) -> Dict[str, Any]:
    train_feats, train_labels = games_to_sequences(train_games)
    train_ds = DangerSequenceDataset(train_feats, train_labels, window_size=WINDOW_SIZE)

    eval_feats, eval_labels = games_to_sequences(eval_games)
    val_ds = DangerSequenceDataset(eval_feats, eval_labels, window_size=WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if len(val_ds) else None

    model = build_model(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print(f"[train] train_rows={len(train_ds)} eval_rows={len(val_ds)} device={device}")
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(y_batch)
        scheduler.step()

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_acc = -1.0
        if val_loader is not None:
            model.eval()
            v_correct, v_total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    v_correct += (logits.argmax(1) == y_batch).sum().item()
                    v_total += len(y_batch)
            val_acc = v_correct / max(v_total, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:3d}/{epochs} loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"[train] best val_acc={best_val_acc:.4f}, checkpoint -> {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return {"history": history, "best_val_acc": best_val_acc, "model": model}


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True,
                         help="Base path passed to generate_defender_danger_data.py's --output-file "
                              "(expects <stem>_train.jsonl / <stem>_eval.jsonl next to it).")
    parser.add_argument("--output-dir", type=Path, default=Path("Defender_danger_model"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_path = args.data_file.with_name(args.data_file.stem + "_train.jsonl")
    eval_path = args.data_file.with_name(args.data_file.stem + "_eval.jsonl")
    train_games = load_games(train_path)
    eval_games = load_games(eval_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.output_dir / "danger_model.pth"

    result = train(train_games, eval_games, args.epochs, args.batch_size, args.learning_rate, save_path, device)
    report = evaluate(result["model"], device, eval_games)
    report["config"] = {
        "data_file": str(args.data_file),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "checkpoint": str(save_path),
        "best_val_acc": result["best_val_acc"],
    }
    gate_path = args.output_dir / "danger_model_gate.json"
    write_json(gate_path, report)
    print(f"[eval] report -> {gate_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

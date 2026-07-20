"""Train a small MLP belief surrogate to replace MC sampling in PPO.

Input: 18-dim public features about an opponent
Output: tenpai_prob (scalar in [0,1])

Usage:
  # 1. Collect data first
  python collect_belief_data.py --num-games 1000 --mc-samples 20

  # 2. Train surrogate
  python train_belief_surrogate.py --data belief_surrogate_data.npz --save-path belief_surrogate.pt
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class BeliefSurrogate(nn.Module):
    """Small MLP: 18 → 64 → 32 → 1, outputs tenpai_prob ∈ [0,1]."""

    def __init__(self, input_dim: int = 18, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_belief_surrogate(
    data_path: str,
    save_path: str = "belief_surrogate.pt",
    val_split: float = 0.15,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Tuple[BeliefSurrogate, dict]:
    """Train the belief surrogate and save the best model."""

    # Load data
    data = np.load(data_path)
    X = data["features"]
    y = data["labels"]
    print(f"Loaded {len(y)} samples, feature_dim={X.shape[1]}")

    # Split
    n = len(y)
    idx = np.random.permutation(n)
    n_val = int(n * val_split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"Train: {len(y_train)}, Val: {len(y_val)}")

    # Datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 4)

    # Model
    model = BeliefSurrogate(input_dim=X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    history = []

    t0 = time.time()
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val = val_loss / max(n_val_batches, 1)
        scheduler.step(avg_val)

        history.append({"epoch": epoch + 1, "train_mse": avg_train, "val_mse": avg_val})

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | "
                  f"Best: {best_val_loss:.6f}")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.1f}s")
    print(f"Best val MSE: {best_val_loss:.6f} (RMSE: {best_val_loss**0.5:.4f})")

    # Save
    torch.save({
        "model_state": best_state,
        "input_dim": X.shape[1],
        "metrics": {"best_val_mse": best_val_loss, "epochs": epochs},
    }, save_path)
    print(f"Saved to {save_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="belief_surrogate_data.npz")
    parser.add_argument("--save-path", type=str, default="belief_surrogate.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    train_belief_surrogate(
        data_path=args.data,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()

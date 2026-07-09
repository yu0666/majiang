"""Binary danger-perception model for ResponsiveDefender's 'learned' threat channel.

Reuses opponent_classifier.py's StyleLSTMAttention architecture (LSTM -> attention
-> concat a running-stat vector -> MLP) with num_classes=2 instead of building a
new one from scratch. Unlike that module's per-game style label, the danger label
changes every discard, so DangerSequenceDataset slices one training example per
P0 discard step (a left-padded window of the preceding features -> that step's
oracle within_shanten label) rather than one example per whole game.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from opponent_classifier import ACTION_FEAT_DIM, HIDDEN_DIM, NUM_LAYERS, StyleLSTMAttention

WINDOW_SIZE = 10


class DangerSequenceDataset(Dataset):
    def __init__(
        self,
        feature_sequences: Sequence[Sequence[Sequence[float]]],
        label_sequences: Sequence[Sequence[int]],
        window_size: int = WINDOW_SIZE,
        feat_dim: int = ACTION_FEAT_DIM,
    ):
        self.window_size = window_size
        X: List[np.ndarray] = []
        y: List[int] = []
        for feats, labels in zip(feature_sequences, label_sequences):
            for end in range(1, len(feats) + 1):
                window = feats[max(0, end - window_size):end]
                arr = np.asarray(window, dtype=np.float32)
                if len(arr) < window_size:
                    pad = np.zeros((window_size - len(arr), feat_dim), dtype=np.float32)
                    arr = np.vstack([pad, arr])
                X.append(arr)
                y.append(int(labels[end - 1]))
        self.X = np.asarray(X, dtype=np.float32) if X else np.zeros((0, window_size, feat_dim), dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


def build_model(device: Optional[str] = None) -> StyleLSTMAttention:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return StyleLSTMAttention(
        input_dim=ACTION_FEAT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
        num_classes=2, dropout=0.3,
    ).to(device)


def predict_proba(model: StyleLSTMAttention, window: np.ndarray, device: str) -> float:
    """window: (seq_len, feat_dim) ndarray, already left-padded to WINDOW_SIZE."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return prob


def load_model(checkpoint_path: str, device: Optional[str] = None) -> StyleLSTMAttention:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

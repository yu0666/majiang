"""
对手风格 LSTM 分类器
- 输入：对手最近 N 次动作的特征序列（默认每步 -> 10维 action features）
- 输出：3类风格概率（aggressive / conservative / random）

用法：
    from opponent_classifier import OpponentStyleClassifier, extract_action_features
    clf = OpponentStyleClassifier()
    clf.load("opponent_clf.pth")
    style_probs = clf.predict_online(action_feature_history)  # shape=(3,)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

STYLES = ["aggressive", "conservative", "random"]
STYLE2IDX = {s: i for i, s in enumerate(STYLES)}
TILE_FEAT_DIM = 6       # Single-tile features only.
ACTION_FEAT_DIM = 10    # Per-action features: action type, risk, stats, tile face.
STAT_FEAT_DIM = 5       # Global summary stats used by the classifier head.
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_CLASSES = 3


# =============================================================================
# 特征：单张弃牌 → 6维向量
# =============================================================================

def extract_tile_features(tile) -> np.ndarray:
    """
    将一张弃牌转成 6 维特征：
      [是幺九, 是中张(4-6), 花色万, 花色条, 花色筒, 牌号归一化]
    """
    from tiles import Suit
    n = tile.number
    is_terminal = float(n in (1, 9))
    is_mid = float(n in (4, 5, 6))
    is_wan = float(tile.suit == Suit.WAN)
    is_tiao = float(tile.suit == Suit.TIAO)
    is_tong = float(tile.suit == Suit.TONG)
    norm_num = (n - 1) / 8.0   # 归一化到 [0,1]
    return np.array([is_terminal, is_mid, is_wan, is_tiao, is_tong, norm_num],
                    dtype=np.float32)


def extract_action_features(tile, action_type: str, discard_count: int,
                            cumul_pengs: int, is_safe: bool,
                            meld_count: int) -> np.ndarray:
    """
    将一次弃牌动作转成 10 维特征向量（C组 H_j 矩阵每行）。

    动作类型 one-hot（3维）:
      f0: is_draw_discard   — 正常摸牌后出牌
      f1: is_peng_discard   — 碰牌后出牌
      f2: is_gang_discard   — 杠牌后出牌

    出牌内容（2维）:
      f3: is_dangerous      — 非安全牌 (1 - is_safe)
      f4: attack_score      — 进攻得分 = is_dangerous × (1 - 0.5 × is_terminal)

    进度特征（1维）:
      f5: discard_step_norm — min(discard_count / 20, 1)

    全局累计统计（2维，mem_long来源）:
      f6: peng_rate         — cumul_pengs / max(discard_count, 1)
      f7: meld_count_norm   — meld_count / 4.0

    牌面特征（2维，LSTM 时序模式来源）:
      f8: norm_number       — (tile.number - 1) / 8，归一化牌号
      f9: is_terminal       — 幺九牌标记 (1/9)
    """
    n = tile.number
    is_terminal = float(n in (1, 9))
    is_dangerous = 1.0 - float(is_safe)
    attack_score = is_dangerous * (1.0 - 0.5 * is_terminal)

    is_draw = float(action_type == "draw")
    is_peng = float(action_type == "peng")
    is_gang = float(action_type == "gang")

    step_norm = min(discard_count / 20.0, 1.0)
    peng_rate = cumul_pengs / max(discard_count, 1)
    meld_norm = meld_count / 4.0
    norm_num = (n - 1) / 8.0

    return np.array([
        is_draw, is_peng, is_gang,       # f0-f2: action type
        is_dangerous, attack_score,      # f3-f4: danger / attack
        step_norm,                       # f5: discard progress
        peng_rate, meld_norm,            # f6-f7: global stats (for mem_long)
        norm_num, is_terminal,           # f8-f9: tile identity (for LSTM temporal)
    ], dtype=np.float32)


# =============================================================================
# Dataset：从模拟数据中构建序列样本
# =============================================================================

class DiscardSequenceDataset(Dataset):
    """
    每条样本：最近 window_size 张弃牌的特征序列 + 风格标签
    数据格式：pre_exp1 模拟时保存的弃牌对象序列
    """
    def __init__(self, sequences, labels, window_size=10, feat_dim=None,
                 late_only=False):
        """
        sequences: List[List[Tile]] 或 List[List[np.ndarray]]
        labels:    List[int]
        window_size: int
        feat_dim:  特征维度
        late_only: True 时只从每局后半段取样（行为特征已积累），减少噪声
        """
        self.window_size = window_size
        self.X = []
        self.y = []

        if feat_dim is None:
            for seq in sequences:
                if len(seq) > 0:
                    if isinstance(seq[0], np.ndarray):
                        feat_dim = seq[0].shape[0]
                    else:
                        feat_dim = TILE_FEAT_DIM
                    break
            if feat_dim is None:
                feat_dim = TILE_FEAT_DIM
        self._feat_dim = feat_dim

        for seq, label in zip(sequences, labels):
            if len(seq) < 3:
                continue
            # late_only: 只用后半段（前半段行为特征≈0，是纯噪声）
            start_idx = max(3, len(seq) // 2) if late_only else 3
            for end in range(start_idx, len(seq) + 1):
                window = seq[max(0, end - window_size): end]
                if isinstance(window[0], np.ndarray):
                    feat_seq = np.stack(window)
                else:
                    feat_seq = np.stack([extract_tile_features(t) for t in window])
                if len(feat_seq) < window_size:
                    pad = np.zeros((window_size - len(feat_seq), feat_dim), dtype=np.float32)
                    feat_seq = np.vstack([pad, feat_seq])
                self.X.append(feat_seq)
                self.y.append(label)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# =============================================================================
# LSTM 分类器模型
# =============================================================================

class StyleLSTM(nn.Module):
    """
    Hybrid LSTM 分类器：
      - LSTM 编码弃牌序列的时序模式
      - 最后一步的行为上下文特征（f6-f9）直接拼接到分类器输入
      - 避免 peng_pass_rate 等关键信号经过 LSTM 衰减
    """
    def __init__(self, input_dim=TILE_FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, dropout=0.3,
                 n_ctx_bypass=0):
        """
        n_ctx_bypass: 输入末尾多少维直接 bypass 到分类器（enriched 时=4，纯牌面=0）
        """
        super().__init__()
        self.n_ctx = n_ctx_bypass
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        clf_input_dim = hidden_dim + n_ctx_bypass
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                    # (batch, hidden_dim)
        if self.n_ctx > 0:
            last_ctx = x[:, -1, -self.n_ctx:]    # (batch, n_ctx) 最后一步的行为特征
            combined = torch.cat([last_hidden, last_ctx], dim=1)
        else:
            combined = last_hidden
        return self.classifier(combined)


class StyleLSTMAttention(nn.Module):
    """
    LSTM + Self-Attention + mem_long bypass 风格分类器（H1 C组）。

    架构：
      LSTM(10D 动作序列) → MultiheadAttention(Q=最终隐状态, K/V=所有步) → z_j (64D)
      mem_long (5D) — 从输入窗口实时计算的全局统计量:
        [peng_rate, meld_count_norm, peng_action_ratio, gang_action_ratio, dangerous_rate]
      分类器输入 = concat(z_j, mem_long) = 69D

    对比优势：
      B组: 4D 基础频率统计 → 决策树（碰牌率 + 出牌方差 + 幺九比例 + 副露数）
      C组: z_j（牌面/动作的时序模式）+ 5D mem_long（更丰富的行为聚合）→ 联合分类
      C 比 B 多出：(1) 动作比例统计, (2) 危险牌比例, (3) 时序行为模式
    """
    def __init__(self, input_dim=ACTION_FEAT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES,
                 dropout=0.3, n_heads=4, n_stat=STAT_FEAT_DIM):
        super().__init__()
        self.n_stat = n_stat
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + n_stat, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (B, W, 10)
        lstm_out, (h_n, _) = self.lstm(x)        # lstm_out: (B, W, hidden)
        query = h_n[-1].unsqueeze(1)              # (B, 1, hidden)
        attn_out, _ = self.attention(query, lstm_out, lstm_out)  # (B, 1, hidden)
        z_j = attn_out.squeeze(1)                 # (B, hidden)

        # mem_long: 5D 全局统计量（严格多于 B组的 4D 基础统计）
        # f6=peng_rate, f7=meld_count,  f1=is_peng, f2=is_gang, f3=is_dangerous
        valid_mask = (x[:, :, 0:3].sum(dim=2) > 0.5).float()
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        mem_long = torch.cat([
            x[:, -1, 6:8],                                        # peng_rate, meld_count
            (x[:, :, 1] * valid_mask).sum(dim=1, keepdim=True) / valid_counts,
            (x[:, :, 2] * valid_mask).sum(dim=1, keepdim=True) / valid_counts,
            (x[:, :, 3] * valid_mask).sum(dim=1, keepdim=True) / valid_counts,
        ], dim=1)                                                 # (B, 5)

        combined = torch.cat([z_j, mem_long], dim=1)  # (B, hidden+5)
        return self.classifier(combined)              # (B, num_classes)


# =============================================================================
# 高级封装：训练 + 在线推理
# =============================================================================

class OpponentStyleClassifier:
    def __init__(self, window_size=10, device=None, input_dim=ACTION_FEAT_DIM):
        self.window_size = window_size
        self.input_dim = input_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # For 10D action features, the context dimensions are consumed directly.
        if input_dim == ACTION_FEAT_DIM:
            self.model = StyleLSTMAttention(
                input_dim=input_dim,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS,
                num_classes=NUM_CLASSES,
                dropout=0.0,
            ).to(self.device)
        else:
            n_ctx = (input_dim - TILE_FEAT_DIM) if input_dim > TILE_FEAT_DIM else 0
            self.model = StyleLSTM(input_dim=input_dim, n_ctx_bypass=n_ctx).to(self.device)
        self._buffer = []

    # Training

    def train_model(self, sequences, labels,
                    epochs=30, batch_size=256, lr=1e-3,
                    val_ratio=0.2, save_path="opponent_clf.pth"):
        """
        sequences: List[List[Tile]]
        labels:    List[int]
        """
        # Random train/validation split for standalone classifier usage.
        n = len(sequences)
        idx = np.random.permutation(n)
        split = int(n * (1 - val_ratio))
        train_idx, val_idx = idx[:split], idx[split:]

        train_seqs = [sequences[i] for i in train_idx]
        train_labs = [labels[i] for i in train_idx]
        val_seqs   = [sequences[i] for i in val_idx]
        val_labs   = [labels[i] for i in val_idx]

        train_ds = DiscardSequenceDataset(train_seqs, train_labs, self.window_size,
                                          late_only=True)
        val_ds   = DiscardSequenceDataset(val_seqs,   val_labs,   self.window_size,
                                          late_only=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        history = {"train_loss": [], "train_acc": [], "val_acc": []}

        print(f"[训练] 样本数: 训练={len(train_ds)}, 验证={len(val_ds)}")
        print(f"[训练] 设备: {self.device}")

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(y_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
            scheduler.step()

            train_loss = total_loss / total
            train_acc  = correct / total

            val_acc = self._eval(val_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | loss={train_loss:.4f} "
                      f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)

        print(f"\n[完成] 最佳验证准确率: {best_val_acc:.4f}")
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        return history, best_val_acc

    def _eval(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        return correct / total if total > 0 else 0.0

    # ── 在线推理（游戏中实时调用）────────────────────────────────────────────

    def update_buffer(self, action_or_tile):
        """每次对手弃牌时调用，维护最近 window_size 张弃牌"""
        self._buffer.append(action_or_tile)
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]

    def reset_buffer(self):
        self._buffer = []

    def predict_online(self, discard_history=None) -> np.ndarray:
        """
        实时预测风格概率，返回 shape=(3,) 的 softmax 概率。
        discard_history: 可选，直接传入弃牌列表；否则用 buffer。
        """
        seq = discard_history if discard_history is not None else self._buffer
        if not seq:
            return np.ones(NUM_CLASSES) / NUM_CLASSES

        recent = seq[-self.window_size:]
        if isinstance(recent[0], np.ndarray):
            feat_seq = np.stack(recent)
        else:
            if self.input_dim == ACTION_FEAT_DIM:
                feat_seq = np.stack([
                    extract_action_features(
                        t, "draw", i + 1, cumul_pengs=0,
                        is_safe=False, meld_count=0,
                    )
                    for i, t in enumerate(recent)
                ])
            else:
                feat_seq = np.stack([extract_tile_features(t) for t in recent])
        if len(feat_seq) < self.window_size:
            pad = np.zeros((self.window_size - len(feat_seq), feat_seq.shape[1]), dtype=np.float32)
            feat_seq = np.vstack([pad, feat_seq])

        x = torch.tensor(feat_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs  # [p_aggressive, p_conservative, p_random]

    def predict_style_name(self, discard_history=None) -> str:
        probs = self.predict_online(discard_history)
        return STYLES[probs.argmax()]

    # ── 保存/加载 ─────────────────────────────────────────────────────────────

    def save(self, path="opponent_clf.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"[保存] 分类器: {path}")

    def load(self, path="opponent_clf.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"[加载] 分类器: {path}")

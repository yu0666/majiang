"""Small neural exploit/safe/deceive gate for MASK.

The gate only chooses the high-level mode.  Candidate generation and all hard
constraints stay in MASKLLMAgent, so a bad gate prediction cannot invent an
illegal or unprotected discard.
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from game import MahjongGame, parse_console_tile
from ppo_features import STATE_DIM, extract_state_features
from rule_engine import FanCalculator, HandPattern, ShantenCalculator
from tiles import Suit, Tile


MODES = ("exploit", "safe", "deceive")
MODE_TO_INDEX = {mode: index for index, mode in enumerate(MODES)}
FEATURE_DIM = STATE_DIM
NORMALIZATION_STD_FLOOR = 1e-2
NORMALIZED_FEATURE_CLIP = 5.0


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def prepare_normalization_tensors(
    mean: torch.Tensor,
    std: torch.Tensor,
    std_floor: float = NORMALIZATION_STD_FLOOR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Avoid exploding near-constant bounded features.

    Raw gate features are already bounded to roughly [0, 1] or [-1, 1].  If a
    dimension is near-constant in the collection split, standardizing it with a
    tiny std turns a normal deployment shift into a huge z-score.  Treat those
    dimensions as raw bounded features instead: mean=0, std=1.
    """
    adjusted_mean = mean.clone()
    adjusted_std = std.clone()
    near_constant = adjusted_std < std_floor
    adjusted_mean[near_constant] = 0.0
    adjusted_std[near_constant] = 1.0
    adjusted_std = adjusted_std.clamp_min(std_floor)
    return adjusted_mean, adjusted_std, near_constant


class NeuralGateNet(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(last_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.05)])
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(last_dim, len(MODES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(features))


def _mode_mask(available_modes: Sequence[str], device: torch.device) -> torch.Tensor:
    mask = torch.full((len(MODES),), -1e9, dtype=torch.float32, device=device)
    for mode in available_modes:
        if mode in MODE_TO_INDEX:
            mask[MODE_TO_INDEX[mode]] = 0.0
    return mask


class NeuralGatePolicy:
    def __init__(
        self,
        model: NeuralGateNet,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.std_floor = float(NORMALIZATION_STD_FLOOR)
        self.feature_clip = float(NORMALIZED_FEATURE_CLIP)
        self.normalization_mode = "standard"
        raw_mean = (mean if mean is not None else torch.zeros(FEATURE_DIM)).to(self.device)
        raw_std = (std if std is not None else torch.ones(FEATURE_DIM)).to(self.device)
        self.mean, self.std, self.near_constant_features = prepare_normalization_tensors(
            raw_mean,
            raw_std,
            self.std_floor,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "NeuralGatePolicy":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = NeuralGateNet(
            input_dim=int(checkpoint.get("input_dim", FEATURE_DIM)),
            hidden_dims=tuple(checkpoint.get("hidden_dims", (128, 64))),
        )
        model.load_state_dict(checkpoint["model_state"])
        mean = torch.tensor(checkpoint.get("feature_mean", [0.0] * FEATURE_DIM), dtype=torch.float32)
        std = torch.tensor(checkpoint.get("feature_std", [1.0] * FEATURE_DIM), dtype=torch.float32)
        policy = cls(model=model, mean=mean, std=std, device=device)
        checkpoint_floor = float(checkpoint.get("normalization_std_floor", NORMALIZATION_STD_FLOOR))
        policy.feature_clip = float(checkpoint.get("normalized_feature_clip", NORMALIZED_FEATURE_CLIP))
        policy.normalization_mode = str(checkpoint.get("normalization_mode", "standard"))
        if checkpoint_floor != policy.std_floor:
            policy.std_floor = checkpoint_floor
            policy.mean, policy.std, policy.near_constant_features = prepare_normalization_tensors(
                mean.to(policy.device),
                std.to(policy.device),
                policy.std_floor,
            )
        if isinstance(checkpoint.get("near_constant_features"), list):
            mask = torch.tensor(checkpoint["near_constant_features"], dtype=torch.bool, device=policy.device)
            if mask.numel() == policy.mean.numel():
                policy.near_constant_features = mask
                policy.mean = policy.mean.clone()
                policy.std = policy.std.clone()
                policy.mean[mask] = 0.0
                policy.std[mask] = 1.0
        return policy

    def predict_from_features(self, features: Sequence[float], available_modes: Sequence[str]) -> Tuple[str, Dict[str, Any]]:
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        x = (x - self.mean) / self.std
        max_abs_before_clip = float(torch.max(torch.abs(x)).item()) if x.numel() else 0.0
        clipped = bool(max_abs_before_clip > self.feature_clip)
        x = torch.clamp(x, -self.feature_clip, self.feature_clip)
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0)).squeeze(0)
            masked_logits = logits + _mode_mask(available_modes, self.device)
            probs = torch.softmax(masked_logits, dim=-1)
            index = int(torch.argmax(probs).item())
        scores = {mode: float(probs[MODE_TO_INDEX[mode]].item()) for mode in MODES}
        return MODES[index], {
            "scores": scores,
            "logits": {mode: float(logits[MODE_TO_INDEX[mode]].item()) for mode in MODES},
            "max_abs_normalized_feature": max_abs_before_clip,
            "normalized_feature_clipped": clipped,
            "normalization_std_floor": self.std_floor,
            "normalized_feature_clip": self.feature_clip,
            "normalization_mode": self.normalization_mode,
            "near_constant_feature_count": int(self.near_constant_features.sum().item()),
        }

    def predict(
        self,
        game: MahjongGame,
        player_id: int,
        z_state: Dict[int, Dict[str, Any]],
        beliefs: Dict[str, Any],
        gate: Dict[str, Any],
        available_modes: Sequence[str],
    ) -> Tuple[str, Dict[str, Any]]:
        features = extract_state_features(game, player_id, z_state, beliefs, gate)
        mode, info = self.predict_from_features(features, available_modes)
        info["feature_source"] = "game_state_v1"
        return mode, info


def _parse_tile_list(text: str) -> List[Tile]:
    tiles: List[Tile] = []
    for token in re.findall(r"\d+[万条筒]", text):
        tile = parse_console_tile(token)
        if tile is not None:
            tiles.append(tile)
    return tiles


def _parse_suit(text: str) -> Optional[Suit]:
    mapping = {"万": Suit.WAN, "条": Suit.TIAO, "筒": Suit.TONG}
    return mapping.get(text.strip())


def _parse_open_melds(text: str) -> List[List[Tile]]:
    melds: List[List[Tile]] = []
    for block in re.findall(r"\[([^\]]+)\]", text):
        tiles = _parse_tile_list(block)
        if tiles:
            melds.append(tiles)
    return melds


def _prompt_potential_fan(hand_tiles: List[Tile], missing_suit: Optional[Suit], open_melds: List[List[Tile]]) -> int:
    """Approximate Player.calculate_potential_fan() from parsed prompt fields."""
    if not hand_tiles:
        return 0
    if ShantenCalculator.calculate_shanten(hand_tiles, missing_suit) > 0:
        return 0
    if len(hand_tiles) % 3 != 1:
        return 0

    owned_tiles = hand_tiles + [tile for meld in open_melds for tile in meld]
    owned_counts = Counter((tile.suit, tile.number) for tile in owned_tiles)
    max_fan = 0
    valid_suits = [suit for suit in Suit if suit != missing_suit]
    for suit in valid_suits:
        for number in range(1, 10):
            if owned_counts[(suit, number)] >= 4:
                continue
            waiting_tile = Tile(suit, number)
            test_hand = hand_tiles + [waiting_tile]
            if missing_suit is not None and any(tile.suit == missing_suit for tile in test_hand):
                continue
            if not HandPattern(test_hand).is_winning_hand():
                continue
            fan, _ = FanCalculator.calculate_fan(
                test_hand,
                waiting_tile,
                open_melds=open_melds,
                is_self_drawn=False,
            )
            max_fan = max(max_fan, int(fan))
    return max_fan


def _literal_block(prompt: str, start_marker: str, end_marker: str) -> Any:
    if start_marker not in prompt:
        return None
    tail = prompt.split(start_marker, 1)[1]
    block = tail.split(end_marker, 1)[0].strip()
    try:
        return ast.literal_eval(block)
    except Exception:
        return None


def _public_table(prompt: str) -> str:
    match = re.search(r"【公开牌桌】\n(.*?)\n\n【局势分析】", prompt, re.S)
    return match.group(1) if match else ""


def features_from_gate_prompt(prompt: str) -> List[float]:
    """Best-effort structural parser for historical gate-oracle prompts.

    This intentionally mirrors ppo_features.extract_state_features so the same
    small gate can be used online from game objects.
    """
    table = _public_table(prompt)
    current = re.search(r"【当前视角】\n(.*?)\n\n【决策空间】", prompt, re.S)
    current_text = current.group(1) if current else ""

    missing_match = re.search(r"我的定缺:\s*([万条筒])", current_text)
    missing_suit = _parse_suit(missing_match.group(1)) if missing_match else None

    hand_match = re.search(r"我的手牌:\s*(.*)", current_text)
    hand_tiles = _parse_tile_list(hand_match.group(1) if hand_match else "")
    shanten = ShantenCalculator.calculate_shanten(hand_tiles, missing_suit) if hand_tiles else 6

    meld_match = re.search(r"我的副露:\s*(.*)", current_text)
    open_meld_lists = _parse_open_melds(meld_match.group(1) if meld_match else "")
    open_melds = len(open_meld_lists)
    potential_fan = _prompt_potential_fan(hand_tiles, missing_suit, open_meld_lists)

    remaining_match = re.search(r"剩余牌数:\s*(\d+)", prompt)
    remaining = float(remaining_match.group(1)) if remaining_match else 40.0

    p0_line = next((line for line in table.splitlines() if line.startswith("我:")), "")
    p0_discards_match = re.search(r"弃牌=([^;]+)", p0_line)
    p0_discards = _parse_tile_list(p0_discards_match.group(1) if p0_discards_match else "")
    terminal_count = sum(1 for tile in p0_discards if tile.number in (1, 9))
    tell = _clip(terminal_count / max(1, len(p0_discards)))

    missing_count = sum(1 for tile in hand_tiles if missing_suit is not None and tile.suit == missing_suit)

    visible = Counter((tile.suit, tile.number) for tile in hand_tiles)
    all_discards = []
    balances = []
    for line in table.splitlines():
        discard_match = re.search(r"弃牌=([^;]+)", line)
        if discard_match:
            all_discards.extend(_parse_tile_list(discard_match.group(1)))
        balance_match = re.search(r"余额=([-0-9.]+)", line)
        if balance_match:
            balances.append(float(balance_match.group(1)))
        melds_match = re.search(r"副露=([^;]+)", line)
        if melds_match:
            visible.update((tile.suit, tile.number) for tile in _parse_tile_list(melds_match.group(1)))
    visible.update((tile.suit, tile.number) for tile in all_discards)

    effective_copies = 0
    for suit in Suit:
        if missing_suit is not None and suit == missing_suit:
            continue
        for number in range(1, 10):
            remaining_copies = max(0, 4 - visible[(suit, number)])
            if remaining_copies <= 0:
                continue
            drawn_shanten = ShantenCalculator.calculate_shanten(hand_tiles + [Tile(suit, number)], missing_suit)
            if drawn_shanten < shanten:
                effective_copies += remaining_copies

    beliefs = _literal_block(prompt, "【对手信念估计 B_phi】", "【规则风险摘要】") or {}
    confs = [float(value.get("tenpai_confidence", 0.0)) for value in beliefs.values() if isinstance(value, dict)]
    avg_conf = _clip(sum(confs) / len(confs)) if confs else 0.0
    max_conf = _clip(max(confs)) if confs else 0.0
    min_conf = _clip(min(confs)) if confs else 0.0
    if len(confs) >= 2:
        mean_conf = sum(confs) / len(confs)
        std_conf = _clip(math.sqrt(sum((c - mean_conf) ** 2 for c in confs) / len(confs)))
    else:
        std_conf = 0.0

    z_state = _literal_block(prompt, "【公开对手漂移 z_j(t)】", "【对手信念估计 B_phi】") or {}
    z_values = list(z_state.values()) if isinstance(z_state, dict) else []
    drift_scores = [float(value.get("drift_score", 0.0)) for value in z_values if isinstance(value, dict)]
    cusum_scores = [float(value.get("cusum_score", 0.0)) for value in z_values if isinstance(value, dict)]
    entropy_scores = [float(value.get("entropy_uncertainty", 0.0)) for value in z_values if isinstance(value, dict)]

    gate = _literal_block(prompt, "【规则风险摘要】", "【可选模式】") or {}
    risk_budget = float(gate.get("risk_budget", 0.0)) if isinstance(gate, dict) else 0.0
    uncertainty = float(gate.get("uncertainty", 0.0)) if isinstance(gate, dict) else 0.0
    score_gap = float(gate.get("score_gap", 0.0)) if isinstance(gate, dict) else 0.0
    tiles_left = float(gate.get("tiles_left", remaining)) if isinstance(gate, dict) else remaining

    p0_balance = balances[0] if balances else 10000.0
    opponent_balances = balances[1:4] if len(balances) >= 4 else [10000.0, 10000.0, 10000.0]

    features = [
        _clip(shanten / 6.0),
        _clip(effective_copies / 30.0),
        _clip(potential_fan / 6.0),
        tell,
        _clip(missing_count / 13.0),
        _clip(open_melds / 4.0),
        avg_conf,
        max_conf,
        min_conf,
        std_conf,
        _clip(max(drift_scores)) if drift_scores else 0.0,
        _clip(max(cusum_scores)) if cusum_scores else 0.0,
        _clip(max(entropy_scores)) if entropy_scores else 0.0,
        _clip(len(z_values) / 3.0),
        _clip(risk_budget),
        _clip(uncertainty),
        _clip(-score_gap / 3000.0),
        _clip(tiles_left / 80.0),
        _clip(remaining / 80.0),
        _clip(len(all_discards) / 80.0),
        _clip(p0_balance / 30000.0),
        *[_clip(balance / 30000.0) for balance in opponent_balances[:3]],
        tell,
        tell * 0.9,
        tell * (1.0 - avg_conf),
        (1.0 if shanten <= 1 else 0.0) * (1.0 - avg_conf),
    ]
    if len(features) != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {len(features)}")
    return features


def available_modes_from_row(row: Dict[str, Any]) -> List[str]:
    modes = row.get("available_modes") or []
    return [mode for mode in modes if mode in MODE_TO_INDEX]


def target_from_rewards(row: Dict[str, Any], reward_scores: Optional[Dict[str, float]] = None) -> str:
    if reward_scores:
        best_value = max(reward_scores.values())
        best_modes = sorted(mode for mode, value in reward_scores.items() if value == best_value)
        rule_mode = str(row.get("rule_mode", "exploit"))
        return rule_mode if rule_mode in best_modes else best_modes[0]
    target = row.get("target_mode")
    if target in MODE_TO_INDEX:
        return str(target)
    rule_mode = str(row.get("rule_mode", "exploit"))
    return rule_mode if rule_mode in MODE_TO_INDEX else "exploit"

"""State feature extraction for PPO training of MASK parameters.

Extracts a fixed-dimension feature vector from the game state that captures:
- Hand information (shanten, ukeire, potential fan, tell threat)
- Opponent beliefs (MC oracle confidence)
- z_j tracking (drift, CUSUM, entropy)
- Gate state (risk budget, uncertainty)
- Global info (tiles left, scores)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional

from game import MahjongGame
from rule_engine import ShantenCalculator
from tiles import Suit, Tile


# Feature dimension (fixed)
STATE_DIM = 28


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _hand_shanten(player) -> int:
    return int(ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit))


def _effective_copies(game: MahjongGame, player_id: int) -> int:
    """Count visible effective tile copies for current hand."""
    player = game.players[player_id]
    shanten = _hand_shanten(player)
    visible = Counter((t.suit, t.number) for t in player.hand_tiles)
    for p in game.players:
        visible.update((t.suit, t.number) for t in p.discarded_tiles)
        for meld in p.open_melds:
            visible.update((t.suit, t.number) for t in meld)
    
    copies = 0
    for suit in Suit:
        if suit == player.missing_suit:
            continue
        for num in range(1, 10):
            remaining = max(0, 4 - visible[(suit, num)])
            if remaining == 0:
                continue
            drawn_shanten = ShantenCalculator.calculate_shanten(
                player.hand_tiles + [Tile(suit, num)],
                player.missing_suit,
            )
            if drawn_shanten < shanten:
                copies += remaining
    return copies


def _potential_fan(player) -> int:
    """Estimate potential fan (max possible fan from current hand)."""
    if _hand_shanten(player) > 0:
        return 0
    result = player.calculate_potential_fan()
    if isinstance(result, tuple):
        return result[0] if result else 0
    return int(result) if result else 0


def _tell_threat(player) -> float:
    """Simple tell threat heuristic: how much info do discards reveal."""
    discards = list(player.discarded_tiles)
    if not discards:
        return 0.0
    # Count terminal/honor discards (more = more info revealed)
    terminal_count = sum(1 for t in discards if t.number in (1, 9))
    return _clip(terminal_count / max(1, len(discards)))


def extract_state_features(
    game: MahjongGame,
    player_id: int,
    z_state: Optional[Dict[int, Dict[str, Any]]] = None,
    beliefs: Optional[Dict[str, Any]] = None,
    gate: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Extract state features for PPO policy network.
    
    Returns a list of STATE_DIM floats, all normalized to roughly [0, 1] or [-1, 1].
    """
    player = game.players[player_id]
    features = []
    
    # === Hand information (6 dims) ===
    shanten = _hand_shanten(player)
    features.append(_clip(shanten / 6.0))  # shanten normalized
    
    eff_copies = _effective_copies(game, player_id)
    features.append(_clip(eff_copies / 30.0))  # effective copies normalized
    
    pot_fan = _potential_fan(player)
    features.append(_clip(pot_fan / 6.0))  # potential fan normalized
    
    tell = _tell_threat(player)
    features.append(tell)  # tell threat [0, 1]
    
    # Missing suit progress (how close to clearing)
    missing_count = sum(1 for t in player.hand_tiles if t.suit == player.missing_suit)
    features.append(_clip(missing_count / 13.0))  # missing suit tiles remaining
    
    # Open melds count (progress indicator)
    features.append(_clip(len(player.open_melds) / 4.0))
    
    # === Opponent beliefs (4 dims) ===
    if beliefs:
        confs = [float(b.get("tenpai_confidence", 0.0)) for b in beliefs.values()]
        features.append(_clip(sum(confs) / len(confs)) if confs else 0.0)  # avg conf
        features.append(_clip(max(confs)) if confs else 0.0)  # max conf
        features.append(_clip(min(confs)) if confs else 0.0)  # min conf
        # Belief entropy
        if len(confs) >= 2:
            mean_conf = sum(confs) / len(confs)
            var_conf = sum((c - mean_conf) ** 2 for c in confs) / len(confs)
            features.append(_clip(math.sqrt(var_conf)))  # belief std
        else:
            features.append(0.0)
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # === z_j tracking (4 dims) ===
    if z_state:
        drift_scores = [v.get("drift_score", 0.0) for v in z_state.values()]
        cusum_scores = [v.get("cusum_score", 0.0) for v in z_state.values()]
        entropy_unc = [v.get("entropy_uncertainty", 0.0) for v in z_state.values()]
        features.append(_clip(max(drift_scores)) if drift_scores else 0.0)
        features.append(_clip(max(cusum_scores)) if cusum_scores else 0.0)
        features.append(_clip(max(entropy_unc)) if entropy_unc else 0.0)
        features.append(_clip(len(z_state) / 3.0))  # num opponents
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # === Gate state (4 dims) ===
    if gate:
        features.append(_clip(float(gate.get("risk_budget", 0.0))))
        features.append(_clip(float(gate.get("uncertainty", 0.0))))
        score_gap = float(gate.get("score_gap", 0.0))
        features.append(_clip(-score_gap / 3000.0))  # behind normalized
        tiles_left = float(gate.get("tiles_left", 40.0))
        features.append(_clip(tiles_left / 80.0))
    else:
        remaining = game.deck.remaining_count()
        features.extend([0.0, 0.0, 0.0, _clip(remaining / 80.0)])
    
    # === Global info (6 dims) ===
    remaining = game.deck.remaining_count()
    features.append(_clip(remaining / 80.0))  # tiles left ratio
    
    # Rounds played (approximate from total discards)
    total_discards = sum(len(p.discarded_tiles) for p in game.players)
    features.append(_clip(total_discards / 80.0))
    
    # Agent score normalized
    features.append(_clip(player.balance / 30000.0))
    
    # Opponent scores (3 dims)
    opponents = [p for p in game.players if p.player_id != player_id]
    for i in range(3):
        if i < len(opponents):
            features.append(_clip(opponents[i].balance / 30000.0))
        else:
            features.append(0.0)
    
    # === Deception-related (4 dims) ===
    # tell_before (current tell threat)
    features.append(tell)
    
    # Approximate tell_after (average change if discarding random tile)
    features.append(tell * 0.9)  # rough proxy
    
    # b_term proxy: tell * (1 - belief_conf)
    avg_conf = features[6]  # avg_belief_conf
    features.append(tell * (1.0 - avg_conf))
    
    # d_term proxy: (1 - belief_conf) if close to tenpai
    near_tenpai = 1.0 if shanten <= 1 else 0.0
    features.append(near_tenpai * (1.0 - avg_conf))
    
    assert len(features) == STATE_DIM, f"Expected {STATE_DIM} features, got {len(features)}"
    return features


def batch_extract_features(
    games: List[MahjongGame],
    player_ids: List[int],
    z_states: Optional[List[Dict]] = None,
    beliefs_list: Optional[List[Dict]] = None,
    gates: Optional[List[Dict]] = None,
) -> List[List[float]]:
    """Batch feature extraction for parallel games."""
    features = []
    for i, (game, pid) in enumerate(zip(games, player_ids)):
        z = z_states[i] if z_states else None
        b = beliefs_list[i] if beliefs_list else None
        g = gates[i] if gates else None
        features.append(extract_state_features(game, pid, z, b, g))
    return features

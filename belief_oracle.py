"""
Opponent-view belief oracle for H1 / B_phi supervision and evaluation.

The object H1 must validate is b_j = "what opponent j believes about my hidden
state", NOT my own ground-truth tenpai.  This module estimates, from opponent
j's information set only (j's own hand + all public melds/discards), a Bayesian
posterior over whether the target player is tenpai, by Monte-Carlo sampling the
target's concealed hand from the tiles j cannot see.

Key properties (this is the fix for the previous oracle):
  * It does NOT peek at the target's real hand to build the label.
  * It is per-observer: different j see different tiles, so b_j differs by j.
  * It returns a smooth posterior in [0, 1], so a constant base-rate predictor
    (B0) can no longer trivially win BSE the way it did against the old
    near-degenerate binary label.

A separate ground-truth ``is_tenpai`` (which does look at the real hand) is kept
ONLY for AUC evaluation -- "does the score rank truly-tenpai states higher".
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from rule_engine import ShantenCalculator
from tiles import Suit, Tile


def within_shanten(tiles: List[Tile], missing_suit: Optional[Suit], max_shanten: int = 0) -> bool:
    """Whether the hand is within ``max_shanten`` of a win (count-agnostic).

    max_shanten=0 -> tenpai (or winning); max_shanten=1 -> "danger" (tenpai or
    one away).  Works for 13- and 14-tile hands, unlike is_ready_with_missing_suit.
    """
    return ShantenCalculator.calculate_shanten(tiles, missing_suit) <= max_shanten


def is_tenpai(tiles: List[Tile], missing_suit: Optional[Suit]) -> bool:
    """shanten<=0 == tenpai (or winning).  Thin wrapper over within_shanten(...,0)."""
    return within_shanten(tiles, missing_suit, 0)


def _full_tile_multiset() -> Counter:
    counts: Counter = Counter()
    for suit in Suit:
        for number in range(1, 10):
            counts[(suit, number)] += 4
    return counts


def _visible_to_observer(game, target_pid: int, observer_pid: int, include_observer_hand: bool) -> Counter:
    """Tiles treated as seen when sampling the target's concealed hand.

    Public part (always): every open meld + every discard.

    ``include_observer_hand`` adds observer j's own concealed hand.  That is the
    TRUE rational belief of j (j knows its hand), but it depends on information
    B_phi cannot observe from the public prompt, so it injects noise the model
    cannot predict.  Default False => the label is a pure public-information
    posterior that B_phi can actually estimate (same for all rational observers).
    """
    seen: Counter = Counter()
    for player in game.players:
        for meld in player.open_melds:
            for tile in meld:
                seen[(tile.suit, tile.number)] += 1
        for tile in player.discarded_tiles:
            seen[(tile.suit, tile.number)] += 1
    if include_observer_hand:
        for tile in game.players[observer_pid].hand_tiles:
            seen[(tile.suit, tile.number)] += 1
    return seen


def _concealed_count_13(player) -> int:
    """Resting concealed-hand size (drop the just-drawn tile if mid-turn)."""
    n = len(player.hand_tiles)
    if n % 3 == 2:  # 14, 11, ... -> just drew; the resting hand is one smaller
        n -= 1
    return max(0, n)


def opponent_view_posterior(
    game,
    target_pid: int,
    observer_pid: int,
    num_samples: int = 60,
    rng: Optional[random.Random] = None,
    beta: float = 2.0,
    play_aware: bool = True,
    include_observer_hand: bool = False,
    max_shanten: int = 0,
) -> Dict[str, object]:
    """Monte-Carlo posterior over whether the target is within max_shanten, from public info.

    Samples the target's concealed hand from the unseen pool (respecting the
    target's publicly-declared missing suit).

    A uniform prior over consistent hands is wrong: a random 13-tile hand is
    almost never tenpai, so the posterior collapses to ~0.  ``play_aware`` fixes
    this with importance sampling: each sampled hand is weighted by
    exp(-beta * shanten), reflecting that the target actively draws/discards
    toward a low-shanten hand.  beta=0 recovers the uniform estimate.

    By default the conditioning set is PUBLIC-only (``include_observer_hand=False``)
    so the label is a function of what B_phi can see; conditioning on observer j's
    private hand would make it the true per-j belief but unpredictable from the
    public prompt.
    """
    rng = rng or random.Random(0)
    target = game.players[target_pid]
    missing = target.missing_suit

    seen = _visible_to_observer(game, target_pid, observer_pid, include_observer_hand)
    pool = _full_tile_multiset()
    pool.subtract(seen)
    # Build a flat list of unseen tiles, excluding the target's missing suit
    # (a rational observer assumes the target is clearing/avoiding it).
    unseen: List[Tuple[Suit, int]] = []
    for (suit, number), count in pool.items():
        if count <= 0:
            continue
        if missing is not None and suit == missing:
            continue
        unseen.extend([(suit, number)] * count)

    n_conceal = _concealed_count_13(target)
    if n_conceal <= 0 or len(unseen) < n_conceal:
        return {
            "tenpai_prob": 0.0,
            "num_samples": 0,
            "n_concealed": n_conceal,
            "unseen_pool": len(unseen),
            "note": "insufficient unseen tiles to sample",
        }

    weight_sum = 0.0
    tenpai_weight = 0.0
    for _ in range(num_samples):
        picks = rng.sample(unseen, n_conceal)
        hand = [Tile(suit, number) for suit, number in picks]
        shanten = ShantenCalculator.calculate_shanten(hand, missing)
        weight = math.exp(-beta * max(0, shanten)) if play_aware else 1.0
        weight_sum += weight
        if shanten <= max_shanten:
            tenpai_weight += weight

    prob = (tenpai_weight / weight_sum) if weight_sum > 0 else 0.0
    return {
        "tenpai_prob": prob,
        "num_samples": num_samples,
        "n_concealed": n_conceal,
        "unseen_pool": len(unseen),
        "beta": beta,
        "play_aware": play_aware,
        "note": "importance-weighted MC opponent-view posterior over target tenpai (no hand peek)",
    }


def confidence_label(probability: float) -> str:
    if probability >= 0.65:
        return "yes"
    if probability <= 0.35:
        return "no"
    return "uncertain"

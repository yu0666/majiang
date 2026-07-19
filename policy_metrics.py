"""Public-information tile-efficiency metrics shared by L1 and L2."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Optional

from game import MahjongGame
from rule_engine import ShantenCalculator
from tiles import Suit, Tile


def discard_progress_metrics(
    game: MahjongGame,
    player_id: int,
    action: str,
) -> Optional[Dict[str, int]]:
    """Return post-discard shanten and publicly estimated effective tile copies."""
    if not action.startswith("d "):
        return None
    player = game.players[player_id]
    tile = next((tile for tile in player.hand_tiles if str(tile) == action[2:]), None)
    if tile is None:
        return None

    hand = player.hand_tiles.copy()
    hand.remove(tile)
    shanten = int(ShantenCalculator.calculate_shanten(hand, player.missing_suit))

    visible = Counter((tile.suit, tile.number) for tile in player.hand_tiles)
    for table_player in game.players:
        visible.update((tile.suit, tile.number) for tile in table_player.discarded_tiles)
        for meld in table_player.open_melds:
            visible.update((tile.suit, tile.number) for tile in meld)

    effective_types = 0
    effective_copies = 0
    for suit in Suit:
        if suit == player.missing_suit:
            continue
        for number in range(1, 10):
            remaining = max(0, 4 - visible[(suit, number)])
            if remaining == 0:
                continue
            drawn_shanten = ShantenCalculator.calculate_shanten(
                hand + [Tile(suit, number)],
                player.missing_suit,
            )
            if drawn_shanten < shanten:
                effective_types += 1
                effective_copies += remaining
    return {
        "shanten": shanten,
        "effective_types": effective_types,
        "effective_copies": effective_copies,
    }

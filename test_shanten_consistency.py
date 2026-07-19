from __future__ import annotations

import unittest

from game import MahjongGame, PlayerState
from rule_engine import HandPattern, ShantenCalculator
from policy_metrics import discard_progress_metrics
from run_gate1_experiments import init_game
from tiles import Suit, Tile


def tiles(text: str):
    suit_map = {"w": Suit.WAN, "t": Suit.TIAO, "p": Suit.TONG}
    return [Tile(suit_map[token[-1]], int(token[:-1])) for token in text.split()]


class ShantenConsistencyTest(unittest.TestCase):
    def test_complete_hand_is_minus_one(self):
        hand = tiles("1w 2w 3w 1t 2t 3t 1p 2p 3p 4p 5p 6p 7p 7p")
        self.assertEqual(ShantenCalculator.calculate_shanten(hand), -1)

    def test_closed_ready_hand_is_zero(self):
        hand = tiles("1w 2w 3w 1t 2t 3t 1p 2p 3p 4p 5p 6p 7p")
        self.assertEqual(ShantenCalculator.calculate_shanten(hand), 0)

    def test_ready_hand_with_one_open_meld_is_zero(self):
        concealed = tiles("1w 2w 3w 1t 2t 3t 4p 5p 6p 7p")
        self.assertEqual(ShantenCalculator.calculate_shanten(concealed), 0)

    def test_missing_suit_tile_prevents_ready_label(self):
        hand = tiles("1w 2w 3w 1t 2t 3t 1p 2p 3p 4p 5p 6p 7w")
        self.assertGreater(ShantenCalculator.calculate_shanten(hand, Suit.WAN), 0)

    def test_discard_progress_reports_nonnegative_effective_copies(self):
        game, _, _ = init_game(20260713, "greedy", "progress_metrics")
        action = next(f"d {tile}" for tile in game.players[0].hand_tiles)
        metrics = discard_progress_metrics(game, 0, action)
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics["effective_copies"], 0)

    def test_ready_hand_rejects_impossible_fifth_copy(self):
        hand = tiles("1w 1w 1w 1w 2t 3t 4t 5t 6t 7t 2p 3p 4p")
        player = PlayerState(0, "test")
        player.hand_tiles = hand

        ready, waits = player.is_ready_with_missing_suit()

        self.assertFalse(ready)
        self.assertEqual(waits, [])
        self.assertFalse(HandPattern(hand + [Tile(Suit.WAN, 1)]).is_winning_hand())

    def test_zero_fan_ready_hand_receives_dajiao_base_score(self):
        game = MahjongGame("zero_fan_dajiao", ["P0", "P1", "P2", "P3"])
        ready = game.players[0]
        ready.hand_tiles = tiles("1w 2w 3w 1t 2t 3t 1p 2p 3p 4p 5p 6p 7t")
        not_ready = game.players[1]
        not_ready.hand_tiles = tiles("1w 1w 2w 4w 5w 7t 8t 9t 2p 4p 6p 8p 9p")
        game.players[2].is_hu = True
        game.players[3].is_hu = True

        self.assertTrue(ready.is_ready_with_missing_suit()[0])
        self.assertEqual(ready.calculate_potential_fan()[0], 0)

        game._settle_da_jiao()

        self.assertEqual(ready.balance, 10000 + game.base_score)
        self.assertEqual(not_ready.balance, 10000 - game.base_score)


if __name__ == "__main__":
    unittest.main()

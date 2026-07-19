from __future__ import annotations

import copy
import unittest

from prompt_builder import build_state_prompt
from run_gate1_experiments import init_game


class PublicPromptTest(unittest.TestCase):
    def test_opponent_concealed_hands_do_not_change_public_prompt(self):
        game, _, _ = init_game(20260713, "greedy", "public_prompt_test")
        altered = copy.deepcopy(game)
        altered.players[1].hand_tiles = []
        altered.players[2].hand_tiles = list(reversed(altered.players[2].hand_tiles))

        self.assertEqual(
            build_state_prompt(game, 0, risk_view="public"),
            build_state_prompt(altered, 0, risk_view="public"),
        )

    def test_oracle_prompt_is_explicitly_different(self):
        game, _, _ = init_game(20260713, "greedy", "oracle_prompt_test")
        public_prompt = build_state_prompt(game, 0, risk_view="public")
        oracle_prompt = build_state_prompt(game, 0, risk_view="oracle")
        self.assertNotEqual(public_prompt, oracle_prompt)
        self.assertIn("真实", oracle_prompt)


if __name__ == "__main__":
    unittest.main()

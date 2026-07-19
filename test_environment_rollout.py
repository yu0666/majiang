from __future__ import annotations

import copy
import unittest

from environment_rollout import (
    capture_rollout_snapshot,
    rollout_candidate,
    rule_mask_continuation_policy,
)
from mask_candidates import build_mode_candidates
from mask_llm import MASKLLMAgent
from prompt_builder import get_legal_actions
from run_gate1_experiments import init_game


def game_fingerprint(game):
    return {
        "state": copy.deepcopy(game.get_game_state()),
        "history": copy.deepcopy(game.history),
        "deck": [str(tile) for tile in game.deck.tiles],
        "balances": [player.balance for player in game.players],
    }


class EnvironmentRolloutTest(unittest.TestCase):
    def setUp(self):
        defender_cfg = {
            "threat_threshold": 0.4,
            "oracle_samples": 4,
            "beta": 2.0,
            "danger_threshold": 1,
            "ffr_hand_shanten": 1,
            "threat_model": "blend",
            "tell_weight": 0.3,
            "tell_window": 6,
        }
        self.game, self.opponents, self.defenders = init_game(
            20260713, "responsive", "rollout_test", defender_cfg
        )
        self.agent = MASKLLMAgent(
            player_id=0,
            mc_seed=17,
            mc_oracle_samples=4,
            forced_deceive="eligible",
            deceive_style="threat",
            threat_gate_mode="delta_only",
            threat_require_real_target=False,
        )

    def test_candidate_sets_are_legal(self):
        legal = get_legal_actions(self.game, 0)
        action = self.agent.decide(self.game, legal)
        mode = self.agent.last_decision["mode"]
        candidates = build_mode_candidates(self.agent, self.game, legal, mode)
        self.assertIn(action, legal)
        self.assertTrue(set(candidates.actions).issubset(set(legal)))

    def test_rollout_is_repeatable_and_does_not_mutate_source(self):
        legal = get_legal_actions(self.game, 0)
        action = next(candidate for candidate in legal if candidate.startswith("d "))
        before = game_fingerprint(self.game)
        snapshot = capture_rollout_snapshot(
            game=self.game,
            opponent_funcs=self.opponents,
            defenders=self.defenders,
            mask_agent=self.agent,
            steps=1,
            max_steps=300,
            episode_start_balance=self.game.players[0].balance,
        )
        first = rollout_candidate(snapshot, action, rollout_seed=91001)
        second = rollout_candidate(snapshot, action, rollout_seed=91001)
        self.assertEqual(first, second)
        self.assertEqual(before, game_fingerprint(self.game))
        self.assertTrue(first.settled)

        rule_first = rollout_candidate(
            snapshot,
            action,
            rollout_seed=91002,
            continuation_policy=rule_mask_continuation_policy,
        )
        rule_second = rollout_candidate(
            snapshot,
            action,
            rollout_seed=91002,
            continuation_policy=rule_mask_continuation_policy,
        )
        self.assertEqual(rule_first, rule_second)
        self.assertEqual(before, game_fingerprint(self.game))


if __name__ == "__main__":
    unittest.main()

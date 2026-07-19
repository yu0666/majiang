from __future__ import annotations

import unittest

from mask_llm import MASKLLMAgent, PublicOpponentTracker, RiskGate
from prompt_builder import build_base_decision_prompt, build_reactive_decision_prompt, get_legal_actions
from rule_engine import ShantenCalculator
from run_gate1_experiments import init_game


class CaptureLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        marker = "合法动作: "
        action = prompt.split(marker, 1)[1].splitlines()[0].split(",", 1)[0].strip()
        return action


def discard_result_shanten(game, action: str) -> int:
    player = game.players[0]
    tile = next(tile for tile in player.hand_tiles if str(tile) == action[2:])
    hand = player.hand_tiles.copy()
    hand.remove(tile)
    return int(ShantenCalculator.calculate_shanten(hand, player.missing_suit))


class LadderAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.game, _, _ = init_game(20260713, "greedy", "ladder_alignment")
        tracker = PublicOpponentTracker([1, 2, 3])
        self.z_state = tracker.update_from_game(self.game)
        self.gate = RiskGate().compute(self.game, 0, self.z_state, beliefs={})

    def test_reactive_prompt_keeps_known_state_schema(self):
        prompt = build_reactive_decision_prompt(
            self.game, 0, self.z_state, self.gate, get_legal_actions(self.game, 0)
        )
        self.assertIn("【局势分析】", prompt)
        self.assertNotIn("【公开对手漂移 z_j(t)】", prompt)
        self.assertNotIn("recent_actions", prompt)

    def test_l0_l1_l2_share_base_exploit_prompt(self):
        llm = CaptureLLM()
        agent = MASKLLMAgent(player_id=0, decision_llm=llm, mc_oracle_samples=2)
        legal = get_legal_actions(self.game, 0)
        agent._exploit_action(self.game, legal)
        expected = build_base_decision_prompt(self.game, 0, valid_actions=legal)
        self.assertEqual(llm.prompts[-1], expected)

    def test_threat_discard_has_zero_shanten_regret(self):
        found = False
        for seed in range(20260700, 20260800):
            game, _, _ = init_game(seed, "greedy", f"shanten_guard_{seed}")
            legal = get_legal_actions(game, 0)
            agent = MASKLLMAgent(
                player_id=0,
                forced_deceive="eligible",
                deceive_style="threat",
                mc_danger_threshold=0,
                threat_gate_mode="delta_only",
                threat_min_delta=-1.0,
                threat_max_start_shanten=99,
                threat_require_real_target=False,
                threat_max_shanten_regret=0,
            )
            action = agent._threat_discard(game, legal)
            if not action:
                continue
            all_results = [
                discard_result_shanten(game, candidate)
                for candidate in legal
                if candidate.startswith("d ")
            ]
            self.assertEqual(discard_result_shanten(game, action), min(all_results))
            found = True
            break
        self.assertTrue(found, "test seeds did not produce an eligible threat discard")


if __name__ == "__main__":
    unittest.main()

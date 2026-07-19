from types import SimpleNamespace
import unittest

from risk_aware_reward import score_rollouts


class RiskAwareRewardTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            return_clip=0.0,
            hu_bonus=5.0,
            fan_bonus=3.0,
            dealin_penalty=20.0,
            tail_alpha=0.5,
            tail_risk_weight=0.5,
            catastrophic_loss_threshold=200.0,
            catastrophic_loss_penalty=40.0,
        )

    def test_fan_is_rewarded(self):
        low = [{"continuation_return": 20, "agent_hu": True, "agent_hu_fan": 1, "agent_dealin": False}]
        high = [{"continuation_return": 20, "agent_hu": True, "agent_hu_fan": 4, "agent_dealin": False}]
        self.assertGreater(
            score_rollouts(high, self.args)["risk_adjusted_score"],
            score_rollouts(low, self.args)["risk_adjusted_score"],
        )

    def test_catastrophic_tail_is_penalized(self):
        stable = [
            {"continuation_return": 10, "agent_hu": False, "agent_hu_fan": 0, "agent_dealin": False},
            {"continuation_return": 10, "agent_hu": False, "agent_hu_fan": 0, "agent_dealin": False},
        ]
        tail = [
            {"continuation_return": 230, "agent_hu": True, "agent_hu_fan": 2, "agent_dealin": False},
            {"continuation_return": -210, "agent_hu": False, "agent_hu_fan": 0, "agent_dealin": True},
        ]
        self.assertGreater(
            score_rollouts(stable, self.args)["risk_adjusted_score"],
            score_rollouts(tail, self.args)["risk_adjusted_score"],
        )


if __name__ == "__main__":
    unittest.main()

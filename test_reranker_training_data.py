from collections import Counter
import unittest

from build_reranker_grpo_dataset import candidate_orders
from run_candidate_oracle import collection_modes


class RerankerTrainingDataTest(unittest.TestCase):
    def test_candidate_orders_rotate_every_action_to_the_front(self):
        candidates = ["d 1万", "d 2万", "d 3万"]

        orders = candidate_orders(candidates, count=3)

        self.assertEqual(len(orders), 3)
        self.assertEqual([order[0] for order in orders], sorted(candidates))
        self.assertTrue(all(sorted(order) == sorted(candidates) for order in orders))

    def test_collection_modes_only_augments_unmet_quotas(self):
        counts = Counter({"exploit": 2, "safe": 0, "deceive": 1})
        targets = {"exploit": 2, "safe": 1, "deceive": 2}

        self.assertEqual(
            collection_modes("exploit", counts, targets, augment_modes=False),
            ["exploit"],
        )
        self.assertEqual(
            collection_modes("exploit", counts, targets, augment_modes=True),
            ["exploit", "safe", "deceive"],
        )

if __name__ == "__main__":
    unittest.main()

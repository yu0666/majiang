from __future__ import annotations

import copy
import json
import re
import unittest

from mask_llm import MASKLLMAgent
from prompt_builder import get_legal_actions
from run_gate1_experiments import init_game
from tiles import Suit, Tile


def tiles(text: str):
    suit_map = {"w": Suit.WAN, "t": Suit.TIAO, "p": Suit.TONG}
    return [Tile(suit_map[token[-1]], int(token[:-1])) for token in text.split()]


class LastCandidateLLM:
    def __init__(self):
        self.rerank_calls = 0

    def __call__(self, prompt: str) -> str:
        if "【规则约束候选重排】" in prompt:
            self.rerank_calls += 1
            match = re.search(r"候选动作:\s*(.*?)(?:\n|$)", prompt)
            candidates = [item.strip() for item in match.group(1).split(",")]
            return json.dumps({"action": candidates[-1], "reason": "test"}, ensure_ascii=False)
        match = re.search(r"合法动作:\s*(.*?)(?:\n|$)", prompt)
        action = match.group(1).split(",")[0].strip()
        return json.dumps({"mode": "exploit", "action": action, "reason": "test"}, ensure_ascii=False)


class MASKRerankerTest(unittest.TestCase):
    def setUp(self):
        self.game, _, _ = init_game(20260713, "greedy", "reranker_test")

    def assert_reranker_used(self, agent: MASKLLMAgent, game, expected_mode: str):
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        trace = agent.last_decision["candidate_reranker"]
        self.assertEqual(agent.last_decision["mode"], expected_mode)
        self.assertTrue(trace["used"])
        self.assertIn(action, trace["candidates"])
        self.assertIn(action, legal)
        self.assertEqual(action, trace["selected_action"])

    def test_exploit_candidates_are_reranked(self):
        llm = LastCandidateLLM()
        agent = MASKLLMAgent(
            player_id=0,
            decision_llm=llm,
            mc_oracle_samples=4,
            use_candidate_reranker=True,
        )
        self.assert_reranker_used(agent, self.game, "exploit")
        self.assertEqual(llm.rerank_calls, 1)

    def test_reranker_keeps_the_same_exploit_baseline(self):
        off_game = copy.deepcopy(self.game)
        on_game = copy.deepcopy(self.game)
        off = MASKLLMAgent(
            player_id=0,
            decision_llm=LastCandidateLLM(),
            mc_oracle_samples=4,
            use_candidate_reranker=False,
        )
        on = MASKLLMAgent(
            player_id=0,
            decision_llm=LastCandidateLLM(),
            mc_oracle_samples=4,
            use_candidate_reranker=True,
        )
        off_action = off.decide(off_game, get_legal_actions(off_game, 0))
        on.decide(on_game, get_legal_actions(on_game, 0))

        self.assertEqual(on.last_decision["rule_action"], off_action)

    def test_learned_gate_controls_mode(self):
        class SafeGate:
            def __call__(self, prompt: str) -> str:
                return "safe"

        agent = MASKLLMAgent(
            player_id=0,
            decision_llm=LastCandidateLLM(),
            gate_llm=SafeGate(),
            gate_policy="learned",
            mc_oracle_samples=4,
        )
        action = agent.decide(self.game, get_legal_actions(self.game, 0))

        self.assertEqual(agent.last_decision["mode"], "safe")
        self.assertTrue(agent.last_decision["learned_gate"]["used"])
        self.assertIn(action, get_legal_actions(self.game, 0))

    def test_reranker_preserves_the_same_exploit_baseline(self):
        off_llm = LastCandidateLLM()
        on_llm = LastCandidateLLM()
        off_agent = MASKLLMAgent(
            player_id=0,
            decision_llm=off_llm,
            mc_oracle_samples=4,
            use_candidate_reranker=False,
        )
        on_agent = MASKLLMAgent(
            player_id=0,
            decision_llm=on_llm,
            mc_oracle_samples=4,
            use_candidate_reranker=True,
        )
        off_game = copy.deepcopy(self.game)
        on_game = copy.deepcopy(self.game)
        off_action = off_agent.decide(off_game, get_legal_actions(off_game, 0))
        on_agent.decide(on_game, get_legal_actions(on_game, 0))

        self.assertEqual(on_agent.last_decision["rule_action"], off_action)

    def test_learned_gate_selects_an_available_mode(self):
        class SafeGate:
            def __call__(self, prompt: str) -> str:
                return "safe"

        llm = LastCandidateLLM()
        agent = MASKLLMAgent(
            player_id=0,
            decision_llm=llm,
            gate_llm=SafeGate(),
            gate_policy="learned",
            mc_oracle_samples=4,
        )
        legal = get_legal_actions(self.game, 0)
        action = agent.decide(self.game, legal)

        self.assertIn(action, legal)
        self.assertEqual(agent.last_decision["mode"], "safe")
        self.assertTrue(agent.last_decision["learned_gate"]["used"])

    def test_safe_candidates_are_reranked(self):
        game = copy.deepcopy(self.game)
        for pid in (1, 2):
            tiles = game.players[pid].hand_tiles[:6]
            game.players[pid].open_melds = [tiles[:3], tiles[3:6]]
        llm = LastCandidateLLM()
        agent = MASKLLMAgent(
            player_id=0,
            decision_llm=llm,
            mc_oracle_samples=4,
            use_candidate_reranker=True,
        )
        self.assert_reranker_used(agent, game, "safe")
        self.assertEqual(llm.rerank_calls, 1)

    def test_continuous_gate_returns_a_legal_action(self):
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous", mc_oracle_samples=4)
        legal = get_legal_actions(self.game, 0)
        action = agent.decide(self.game, legal)
        self.assertIn(action, legal)
        self.assertIn(agent.last_decision["mode"], {"exploit", "safe", "deceive"})
        self.assertIn("alpha", agent.last_decision)
        self.assertGreaterEqual(agent.last_decision["alpha"], 0.0)
        self.assertLessEqual(agent.last_decision["alpha"], 1.0)

    def test_continuous_gate_low_alpha_matches_pure_shanten_best(self):
        # Fresh, early-game, on-par-score state -> risk appetite should stay
        # low, so alpha should be small and the winning action should match
        # the pure shanten/ukeire-best discard (no deceive term needed to win).
        game = copy.deepcopy(self.game)
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        # Recompute the pure-Q ranking directly from progress metrics to avoid
        # relying on internal private state; this mirrors what
        # _continuous_gate_action itself uses for best_q_only.
        from policy_metrics import discard_progress_metrics

        progress = {
            a: discard_progress_metrics(game, 0, a)
            for a in legal if a.startswith("d ")
        }
        best_q_only = max(
            progress, key=lambda a: -100.0 * progress[a]["shanten"] + progress[a]["effective_copies"]
        )
        self.assertLess(agent.last_decision["alpha"], 0.5)
        self.assertEqual(action, best_q_only)

    def test_continuous_gate_alpha_rises_when_behind_and_late(self):
        # alpha is now value_gate-multiplied (see _continuous_gate_action's
        # docstring), so both games need a hand that actually reaches tenpai
        # with a real fan this turn -- otherwise value_gate=0 forces alpha=0
        # in both regardless of risk_appetite, and this comparison would be
        # vacuous rather than testing the risk/uncertainty behavior.
        # 全求人-eligible pengpenghu shape (三组刻子 + 一将 waiting on a triplet
        # of 4t), missing the wan suit -- discarding the junk 9w tile leaves
        # a real fan>0 tenpai (碰碰胡, 1番), unlike an arbitrary run-shape
        # tenpai which often prices out at 平胡(0番) and would leave
        # value_gate at 0 just like the docstring's tie-break diagnosis.
        tenpai_hand = tiles("1p 1p 1p 2p 2p 2p 3p 3p 3p 4t 4t 4t 5t 9w")
        fresh_game = copy.deepcopy(self.game)
        behind_game = copy.deepcopy(self.game)
        fresh_game.players[0].hand_tiles = list(tenpai_hand)
        behind_game.players[0].hand_tiles = list(tenpai_hand)
        behind_game.players[0].balance = 5000  # far behind starting balance of 10000
        behind_game.deck.tiles = behind_game.deck.tiles[:6]  # late hand, few tiles left

        fresh_agent = MASKLLMAgent(player_id=0, gate_policy="continuous", mc_oracle_samples=4)
        behind_agent = MASKLLMAgent(player_id=0, gate_policy="continuous", mc_oracle_samples=4)
        fresh_agent.decide(fresh_game, get_legal_actions(fresh_game, 0))
        behind_agent.decide(behind_game, get_legal_actions(behind_game, 0))

        self.assertGreater(behind_agent.last_decision["value_gate"], 0.0)
        self.assertGreater(fresh_agent.last_decision["value_gate"], 0.0)
        self.assertGreater(behind_agent.last_decision["alpha"], fresh_agent.last_decision["alpha"])

    def test_continuous_gate_zero_value_gate_zeroes_alpha_exactly(self):
        # Regression test for the diagnosed bug: an earlier version folded
        # value_gate into the alpha logit additively (-kappa4*(1-value_gate)),
        # which only shrinks alpha asymptotically and never reaches exactly
        # 0 -- so on a worthless (non-tenpai) hand, ANY alpha>0 already let
        # ΔShape win ties against an equally-tied q_base candidate (the
        # crossing alpha for those ties is 0.0), making the "suppression"
        # have zero effect on 1096/1096 real decisions in a smoke test.
        # value_gate must multiply alpha directly so a worthless hand forces
        # alpha to exactly 0.0, not just a small positive number.
        behind_game = copy.deepcopy(self.game)
        behind_game.players[0].balance = 5000
        behind_game.deck.tiles = behind_game.deck.tiles[:6]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous", mc_oracle_samples=4)
        agent.decide(behind_game, get_legal_actions(behind_game, 0))

        self.assertEqual(agent.last_decision["value_gate"], 0.0)
        self.assertEqual(agent.last_decision["alpha"], 0.0)

    def test_hand_shape_fan_direction_rewards_purity_and_pungs(self):
        # Same tile count, but one hand is concentrated in a single suit with
        # several proto-triplets (pairs already formed) -- should score
        # higher than an equally-sized hand scattered across all three suits
        # with no repeated numbers, since the former is closer to
        # 清一色/碰碰胡 (rule_engine.py's FanCalculator) and the latter isn't
        # closer to any scored fan type.
        pure_pung_leaning = tiles("1w 1w 2w 2w 3w 3w 4w 4w 5w 5w 6w 6w 7w 8w")
        scattered_mixed = tiles("1w 2t 3p 4w 5t 6p 7w 8t 9p 1t 2p 3w 4t 5p")
        pure_score = MASKLLMAgent._hand_shape_fan_direction(pure_pung_leaning, [], None)
        mixed_score = MASKLLMAgent._hand_shape_fan_direction(scattered_mixed, [], None)
        self.assertGreater(pure_score, mixed_score)

    def test_continuous_v2_gate_returns_a_legal_action_with_fan_shaping_on(self):
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v2", mc_oracle_samples=4)
        legal = get_legal_actions(self.game, 0)
        action = agent.decide(self.game, legal)
        self.assertIn(action, legal)
        self.assertIn(agent.last_decision["mode"], {"exploit", "safe", "deceive"})
        self.assertTrue(agent.last_decision["fan_shaping"])
        self.assertGreaterEqual(agent.last_decision["alpha"], 0.0)
        self.assertLessEqual(agent.last_decision["alpha"], 1.0)

    def test_continuous_gate_v1_has_fan_shaping_off(self):
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous", mc_oracle_samples=4)
        agent.decide(self.game, get_legal_actions(self.game, 0))
        self.assertFalse(agent.last_decision["fan_shaping"])
        self.assertEqual(agent.last_decision["max_shape_direction"], 0.0)

    def test_continuous_v3_gate_returns_a_legal_action(self):
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v3", mc_oracle_samples=4)
        legal = get_legal_actions(self.game, 0)
        action = agent.decide(self.game, legal)
        self.assertIn(action, legal)
        self.assertIn(agent.last_decision["mode"], {"exploit", "safe", "deceive"})
        self.assertTrue(agent.last_decision["fan_shaping"])
        self.assertTrue(agent.last_decision["early_hu_penalty"])

    def test_continuous_v2_has_early_hu_penalty_off(self):
        # Byte-identical-behavior guarantee: v2 must not gain v3's decline
        # logic just because they share the fan_shaping code path.
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v2", mc_oracle_samples=4)
        agent.decide(self.game, get_legal_actions(self.game, 0))
        self.assertFalse(agent.last_decision["early_hu_penalty"])

    @staticmethod
    def _early_hu_ambiguous_wait_hand():
        # 1t2t3t + 1w2w3w + 7w8w9w + 7p8p + 9p9p (13 tiles), missing_suit=None.
        # Waiting on 6p (completes 6p7p8p run -> 平胡 only, fan=0) or 9p
        # (completes 9p9p9p pung, leaving every group 1-2-3/7-8-9-shaped ->
        # 带幺九, fan=2) -- same ambiguous-wait trick used by
        # test_hand_shape_fan_direction_rewards_purity_and_pungs's sibling
        # fixtures, verified against the real FanCalculator before being
        # hard-coded here (see session notes: fan spread 0 vs 2 on response,
        # 1 vs 3 self-drawn).
        return tiles("1t 2t 3t 1w 2w 3w 7w 8w 9w 7p 8p 9p 9p")

    def test_continuous_v3_declines_a_cheap_response_hu_for_upside(self):
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand()
        game.players[0].missing_suit = None
        game.last_discarded_tile = tiles("6p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v3", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0, response_actions=["hu"])
        action = agent.decide(game, legal)
        self.assertEqual(action, "n")
        self.assertEqual(agent.last_decision["fan_now"], 0)

    def test_continuous_v3_takes_a_valuable_response_hu(self):
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand()
        game.players[0].missing_suit = None
        game.last_discarded_tile = tiles("9p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v3", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0, response_actions=["hu"])
        action = agent.decide(game, legal)
        self.assertEqual(action, "h")
        self.assertEqual(agent.last_decision["fan_now"], 2)

    def test_continuous_v3_declines_a_cheap_self_draw_hu_for_upside(self):
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("6p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("6p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v3", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "d 6筒")
        self.assertEqual(agent.last_decision["fan_now"], 1)

    def test_continuous_v3_takes_a_valuable_self_draw_hu(self):
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("9p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("9p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v3", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "h")
        self.assertEqual(agent.last_decision["fan_now"], 3)

    def test_continuous_v4_gate_returns_a_legal_action(self):
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v4", mc_oracle_samples=4)
        legal = get_legal_actions(self.game, 0)
        action = agent.decide(self.game, legal)
        self.assertIn(action, legal)
        self.assertIn(agent.last_decision["mode"], {"exploit", "safe", "deceive"})
        self.assertTrue(agent.last_decision["fan_shaping"])
        self.assertTrue(agent.last_decision["early_hu_penalty"])
        self.assertTrue(agent.last_decision["early_hu_expected_value"])

    def test_continuous_v3_has_early_hu_expected_value_off(self):
        # Byte-identical-behavior guarantee: v3's already-reported real-eval
        # results must stay reproducible, so v3 must keep using the max-case
        # _early_hu_decline_check, not v4's expected-value variant, even
        # though they share the fan_shaping/early_hu_penalty code path.
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v3", mc_oracle_samples=4)
        agent.decide(self.game, get_legal_actions(self.game, 0))
        self.assertFalse(agent.last_decision["early_hu_expected_value"])

    def test_continuous_v4_takes_a_cheap_self_draw_hu_when_upside_wait_is_dead(self):
        # Root-cause regression test for the v3 diagnosis: the ambiguous
        # wait (6p -> 0 fan, 9p -> 2 fan) has 0 remaining copies of the 2-fan
        # 9p tile -- both of its 2 external copies are already visible in an
        # opponent's discards, on top of the 2 already sitting in this
        # hand's own 9p9p pair. v3's MAX-based upside enumerates 9p as a
        # legal wait regardless of whether any copies are still live, so it
        # would decline chasing a fan value that is now literally
        # unobtainable. v4's expected value only weights waiting tiles with
        # remaining>0, so the dead 9p wait drops out entirely, leaving
        # expected=0.0 (all weight on 6p's 0 fan) -- below fan_now_basis(0)
        # + 0.15, so v4 correctly takes the guaranteed win.
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("6p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("6p")[0]
        game.players[1].discarded_tiles = tiles("9p 9p")
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v4", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "h")
        self.assertEqual(agent.last_decision["fan_now"], 1)

    def test_continuous_v4_still_declines_self_draw_hu_when_expected_upside_clears_margin(self):
        # Unskewed wait (both 6p and 9p fully live: 4 and 2 remaining copies
        # respectively -- 2 of 9p's 4 copies already sit in this hand's own
        # 9p9p pair) still carries a real, well-supported expected edge
        # ((4*0 + 2*2)/6 = 0.667 >= 0 + 0.15), so v4 should still decline
        # here, confirming the retuned margin doesn't disable priority-2
        # outright.
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("6p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("6p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v4", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "d 6筒")
        self.assertEqual(agent.last_decision["fan_now"], 1)

    def test_continuous_v4_takes_a_cheap_response_hu_when_upside_wait_is_dead(self):
        # Response-hu mirror of the dead-wait regression test above: 0
        # remaining copies of the 2-fan 9p tile (both external copies
        # visible in an opponent's discards), so the dead wait drops out of
        # the expected-value weighting entirely and v4 takes the win instead
        # of chasing an unobtainable upgrade.
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand()
        game.players[0].missing_suit = None
        game.last_discarded_tile = tiles("6p")[0]
        game.players[1].discarded_tiles = tiles("9p 9p")
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v4", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0, response_actions=["hu"])
        action = agent.decide(game, legal)
        self.assertEqual(action, "h")
        self.assertEqual(agent.last_decision["fan_now"], 0)

    def test_continuous_v4_still_declines_response_hu_when_expected_upside_clears_margin(self):
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand()
        game.players[0].missing_suit = None
        game.last_discarded_tile = tiles("6p")[0]
        game.players[1].discarded_tiles = tiles("6p 6p")
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v4", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0, response_actions=["hu"])
        action = agent.decide(game, legal)
        self.assertEqual(action, "n")
        self.assertEqual(agent.last_decision["fan_now"], 0)

    def test_continuous_v5_gate_returns_a_legal_action(self):
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v5", mc_oracle_samples=4)
        legal = get_legal_actions(self.game, 0)
        action = agent.decide(self.game, legal)
        self.assertIn(action, legal)
        self.assertIn(agent.last_decision["mode"], {"exploit", "safe", "deceive"})
        self.assertTrue(agent.last_decision["fan_shaping"])
        self.assertTrue(agent.last_decision["early_hu_penalty"])
        self.assertTrue(agent.last_decision["early_hu_expected_value"])
        self.assertTrue(agent.last_decision["early_hu_tightened"])

    def test_continuous_v4_has_early_hu_tightened_off(self):
        # Byte-identical-behavior guarantee: v4's already-reported real-eval
        # results must stay reproducible, so v4 must keep using its own
        # (untightened) gating thresholds, not v5's, even though they share
        # the early_hu_penalty/early_hu_expected_value code path.
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v4", mc_oracle_samples=4)
        agent.decide(self.game, get_legal_actions(self.game, 0))
        self.assertFalse(agent.last_decision["early_hu_tightened"])

    def test_continuous_v5_still_declines_self_draw_hu_when_gate_thresholds_pass(self):
        # Fresh-game gate state (tiles_left=55, risk_budget~0.33, mode_hint
        # "deceive") clears v5's tightened thresholds (tiles_left>=28,
        # rho<=0.5) just as easily as v4's looser ones, so on the unskewed
        # ambiguous wait v5 should still decline exactly like v4 -- the
        # tightening changes *when* the check fires, not the expected-value
        # math once it does.
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("6p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("6p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v5", mc_oracle_samples=4)
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "d 6筒")
        self.assertEqual(agent.last_decision["fan_now"], 1)

    def test_continuous_v5_takes_cheap_hu_when_tiles_left_below_tightened_threshold(self):
        # tiles_left=25 clears v4's EARLY_HU_TILES_LEFT_MIN (20.0) -- v4
        # would decline here (verified separately) -- but falls below v5's
        # tightened 28.0, so v5's gate check exits before ever computing an
        # upside estimate and the agent takes the guaranteed win instead.
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("6p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("6p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v5", mc_oracle_samples=4)
        agent.risk_gate.compute = lambda *a, **kw: {
            "mode_hint": "exploit", "risk_budget": 0.3, "uncertainty": 0.1,
            "score_gap": 0, "tiles_left": 25, "z_drift_flags": {},
        }
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "h")
        self.assertIsNone(agent.last_decision["fan_now"])

    def test_continuous_v5_takes_cheap_hu_when_risk_above_tightened_threshold(self):
        # risk_budget=0.6 clears v4's EARLY_HU_RHO_MAX (0.75) but falls
        # above v5's tightened 0.5, so v5 takes the guaranteed win instead
        # of chasing the upside.
        game = copy.deepcopy(self.game)
        game.players[0].hand_tiles = self._early_hu_ambiguous_wait_hand() + tiles("6p")
        game.players[0].missing_suit = None
        game.players[0].last_drawn_tile = tiles("6p")[0]
        agent = MASKLLMAgent(player_id=0, gate_policy="continuous_v5", mc_oracle_samples=4)
        agent.risk_gate.compute = lambda *a, **kw: {
            "mode_hint": "exploit", "risk_budget": 0.6, "uncertainty": 0.1,
            "score_gap": 0, "tiles_left": 40, "z_drift_flags": {},
        }
        legal = get_legal_actions(game, 0)
        action = agent.decide(game, legal)
        self.assertEqual(action, "h")
        self.assertIsNone(agent.last_decision["fan_now"])

    def test_deceive_candidates_are_reranked(self):
        llm = LastCandidateLLM()
        agent = MASKLLMAgent(
            player_id=0,
            decision_llm=llm,
            mc_oracle_samples=4,
            forced_deceive="always",
            deceive_style="threat",
            threat_allow_break_ready=True,
            threat_require_real_target=False,
            threat_require_non_exploit=False,
            use_candidate_reranker=True,
        )
        self.assert_reranker_used(agent, self.game, "deceive")
        self.assertEqual(llm.rerank_calls, 1)


if __name__ == "__main__":
    unittest.main()

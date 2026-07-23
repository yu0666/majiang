"""
LLM-native MASK skeleton.

The project policy is that future MASK experiments use Chinese prompts and an
LLM for belief estimation and decisions.  This file therefore contains no small
neural classifier.  The default estimator is a deterministic heuristic so E0 and
CI can run without loading a local LLM; a real LLM adapter can be passed in.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from belief_oracle import opponent_view_posterior, within_shanten
from game import MahjongGame
from prompt_builder import (
    build_base_decision_prompt,
    build_belief_prompt,
    build_gate_decision_prompt,
    build_mask_decision_prompt,
    get_legal_actions,
)
from policy_metrics import discard_progress_metrics
from rule_engine import ShantenCalculator, FanCalculator


LLMCallable = Callable[[str], str]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# CUSUM change-point constants for OpponentDriftState (continuous_v7's z_j
# upgrade). _CUSUM_SLACK (k) is the per-step deviation allowance below which
# evidence does not accumulate (ordinary noise); _CUSUM_THRESHOLD (h) is the
# cumulative evidence level that flags a confirmed change point. _EMA_LAMBDA
# smooths the raw per-step feature vector before it feeds the CUSUM statistic,
# so a single noisy step can't swing the reference comparison on its own.
_EMA_LAMBDA = 0.4
_CUSUM_SLACK = 0.05
_CUSUM_THRESHOLD = 0.6
_ACTION_TYPES = ("discard", "peng", "gang", "hu", "pass")


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def legalize_action(action: str, legal_actions: List[str]) -> str:
    action = (action or "").strip()
    if action in legal_actions:
        return action

    # Accept common verbose outputs that contain the action.
    for legal in legal_actions:
        if legal != "n" and legal in action:
            return legal

    discard_match = re.search(r"d\s*[：:]?\s*([1-9][万条筒])", action)
    if discard_match:
        candidate = f"d {discard_match.group(1)}"
        if candidate in legal_actions:
            return candidate

    # Prefer hu/gang/peng when legal; otherwise first legal discard/pass.
    for preferred in ("h", "g", "p", "n"):
        if preferred in legal_actions:
            return preferred
    return legal_actions[0]


@dataclass
class OpponentDriftState:
    player_id: int
    window_size: int = 24
    actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=24))
    last_vector: Dict[str, float] = field(default_factory=dict)
    drift_score: float = 0.0
    drift_flag: bool = False
    # continuous_v7 additions (opt-in downstream; computed unconditionally
    # here since they're cheap and additive, but only *consumed* when
    # RiskGate.compute(..., use_cusum_uncertainty=True) reads them):
    ema_vector: Dict[str, float] = field(default_factory=dict)
    reference_vector: Dict[str, float] = field(default_factory=dict)
    cusum_score: float = 0.0
    cusum_flag: bool = False
    entropy_uncertainty: float = 0.0

    def append_public_action(self, record: Dict[str, Any]) -> None:
        if record.get("pid") != self.player_id:
            return
        self.actions.append(record)
        self._update_drift()

    def _update_drift(self) -> None:
        current = self.as_vector()
        if not self.last_vector:
            self.last_vector = current
            self.drift_score = 0.0
            self.drift_flag = False
        else:
            keys = set(current) | set(self.last_vector)
            delta = math.sqrt(sum((current.get(k, 0.0) - self.last_vector.get(k, 0.0)) ** 2 for k in keys))
            self.drift_score = delta
            self.drift_flag = delta >= 0.45 and len(self.actions) >= 6
            self.last_vector = current
        self.entropy_uncertainty = self._compute_entropy_uncertainty()
        self._update_cusum(current)

    def _update_cusum(self, current: Dict[str, float]) -> None:
        # EMA-smooth the raw per-step vector first (lambda=0.4): this is the
        # tracker's actual per-step state, distinct from the naive single-step
        # `last_vector` diff above.
        if not self.ema_vector:
            self.ema_vector = dict(current)
        else:
            keys = set(current) | set(self.ema_vector)
            self.ema_vector = {
                k: _EMA_LAMBDA * current.get(k, 0.0) + (1.0 - _EMA_LAMBDA) * self.ema_vector.get(k, 0.0)
                for k in keys
            }

        if len(self.actions) < 6:
            # Window still filling: keep sliding the reference so the
            # "normal" baseline reflects settled early behavior, not the
            # very first 1-2 actions.
            self.reference_vector = dict(self.ema_vector)
            self.cusum_score = 0.0
            self.cusum_flag = False
            return
        if not self.reference_vector:
            self.reference_vector = dict(self.ema_vector)

        # Classic one-sided CUSUM: accumulate (deviation - slack), floored at
        # 0, against a *fixed* reference. Unlike drift_score (single-step vs.
        # last update), this accumulates evidence across many small steps, so
        # slow drift away from the opponent's early-game baseline still
        # eventually crosses the threshold.
        keys = set(self.ema_vector) | set(self.reference_vector)
        deviation = math.sqrt(
            sum((self.ema_vector.get(k, 0.0) - self.reference_vector.get(k, 0.0)) ** 2 for k in keys)
        )
        self.cusum_score = max(0.0, self.cusum_score + deviation - _CUSUM_SLACK)
        self.cusum_flag = self.cusum_score >= _CUSUM_THRESHOLD

    def _compute_entropy_uncertainty(self) -> float:
        # Categorical entropy of the action-type histogram, normalized against
        # the fixed 5-type alphabet so it's comparable across windows/opponents
        # (not against however many distinct types happen to appear). High
        # entropy = spread-out/unpredictable behavior mix; low = concentrated.
        total = max(1, len(self.actions))
        counts = Counter(a.get("act", "") for a in self.actions)
        probs = [counts.get(t, 0) / total for t in _ACTION_TYPES if counts.get(t, 0) > 0]
        if len(probs) <= 1:
            return 0.0
        entropy = -sum(p * math.log(p, 2) for p in probs)
        max_entropy = math.log(len(_ACTION_TYPES), 2)
        return _clip(entropy / max_entropy) if max_entropy > 0 else 0.0

    def as_vector(self) -> Dict[str, float]:
        total = max(1, len(self.actions))
        counts = {"discard": 0, "peng": 0, "gang": 0, "hu": 0, "pass": 0}
        terminal_discards = 0
        mid_discards = 0
        suit_counts = {"万": 0, "条": 0, "筒": 0}

        for action in self.actions:
            act = action.get("act", "")
            counts[act] = counts.get(act, 0) + 1
            tile = action.get("tile", "")
            if act == "discard" and len(tile) >= 2:
                try:
                    num = int(tile[0])
                except ValueError:
                    num = 0
                suit = tile[1]
                if num in (1, 9):
                    terminal_discards += 1
                if num in (4, 5, 6):
                    mid_discards += 1
                if suit in suit_counts:
                    suit_counts[suit] += 1

        discard_count = max(1, counts["discard"])
        return {
            "discard_rate": counts["discard"] / total,
            "peng_rate": counts["peng"] / total,
            "gang_rate": counts["gang"] / total,
            "hu_rate": counts["hu"] / total,
            "terminal_discard_rate": terminal_discards / discard_count,
            "mid_discard_rate": mid_discards / discard_count,
            "wan_discard_rate": suit_counts["万"] / discard_count,
            "tiao_discard_rate": suit_counts["条"] / discard_count,
            "tong_discard_rate": suit_counts["筒"] / discard_count,
        }

    def label(self) -> str:
        v = self.as_vector()
        if v["peng_rate"] + v["gang_rate"] > 0.18 or v["mid_discard_rate"] > 0.45:
            return "aggressive_like"
        if v["terminal_discard_rate"] > 0.45 and v["peng_rate"] < 0.08:
            return "conservative_like"
        return "mixed_or_unknown"

    def summary(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "z_label": self.label(),
            "z_public_vector": self.as_vector(),
            "drift_score": round(self.drift_score, 4),
            "drift_flag": self.drift_flag,
            "cusum_score": round(self.cusum_score, 4),
            "cusum_flag": self.cusum_flag,
            "entropy_uncertainty": round(self.entropy_uncertainty, 4),
            "recent_actions": list(self.actions)[-6:],
        }


class PublicOpponentTracker:
    def __init__(self, player_ids: List[int], window_size: int = 24):
        self.states = {pid: OpponentDriftState(pid, window_size=window_size) for pid in player_ids}
        for state in self.states.values():
            state.actions = deque(maxlen=window_size)
        self._seen_history_len = 0

    def update_from_game(self, game: MahjongGame) -> Dict[int, Dict[str, Any]]:
        for record in game.history[self._seen_history_len:]:
            for state in self.states.values():
                state.append_public_action(record)
        self._seen_history_len = len(game.history)
        return self.summary()

    def summary(self) -> Dict[int, Dict[str, Any]]:
        return {pid: state.summary() for pid, state in self.states.items()}


class LLMBeliefEstimator:
    def __init__(self, llm: Optional[LLMCallable] = None):
        self.llm = llm

    def infer(self, game: MahjongGame, player_id: int, target_opponent_id: int) -> Dict[str, Any]:
        prompt = build_belief_prompt(game, player_id, target_opponent_id)
        if self.llm is not None:
            parsed = _safe_json_loads(self.llm(prompt))
            if parsed is not None:
                return self._normalize(parsed, target_opponent_id)
        return self._heuristic(game, player_id, target_opponent_id)

    def infer_all(self, game: MahjongGame, player_id: int) -> Dict[str, Dict[str, Any]]:
        beliefs = {}
        for player in game.players:
            if player.player_id != player_id:
                beliefs[f"P{player.player_id}"] = self.infer(game, player_id, player.player_id)
        return beliefs

    def _normalize(self, data: Dict[str, Any], target_opponent_id: int) -> Dict[str, Any]:
        label = data.get("think_i_am_tenpai", "uncertain")
        if label not in {"yes", "no", "uncertain"}:
            label = "uncertain"
        try:
            conf = float(data.get("tenpai_confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return {
            "target_opponent": data.get("target_opponent", f"P{target_opponent_id}"),
            "think_i_am_tenpai": label,
            "tenpai_confidence": round(_clip(conf), 3),
            "suspected_waits": data.get("suspected_waits", []) or [],
            "suspected_pattern": data.get("suspected_pattern", "unknown"),
            "danger_tiles_for_me": data.get("danger_tiles_for_me", []) or [],
            "reason": data.get("reason", "LLM output"),
        }

    def _heuristic(self, game: MahjongGame, player_id: int, target_opponent_id: int) -> Dict[str, Any]:
        """Public-only fallback used for smoke tests.

        This intentionally avoids reading player.hand_tiles or true shanten.
        It estimates how threatening P_i looks to an opponent from public
        actions only: discards, melds, missing-suit progress, and round phase.
        """
        player = game.players[player_id]
        meld_count = len(player.open_melds)
        own_discards = len(player.discarded_tiles)
        tiles_left = game.deck.remaining_count()
        missing_left_public = 0
        if player.missing_suit:
            missing_left_public = sum(1 for t in player.discarded_tiles if t.suit == player.missing_suit)

        confidence = 0.25
        if meld_count >= 2:
            confidence += 0.22
        elif meld_count == 1:
            confidence += 0.10
        if own_discards >= 9:
            confidence += 0.18
        elif own_discards >= 6:
            confidence += 0.10
        if tiles_left <= 30:
            confidence += 0.14
        if player.missing_suit and own_discards >= 4 and missing_left_public >= 2:
            confidence += 0.08
        if own_discards <= 3:
            confidence -= 0.08
        confidence = _clip(confidence)

        if confidence >= 0.65:
            tenpai_label = "yes"
        elif confidence <= 0.42:
            tenpai_label = "no"
        else:
            tenpai_label = "uncertain"

        return {
            "target_opponent": f"P{target_opponent_id}",
            "think_i_am_tenpai": tenpai_label,
            "tenpai_confidence": round(confidence, 3),
            "suspected_waits": [],
            "suspected_pattern": "unknown",
            "danger_tiles_for_me": [],
            "reason": "public-only heuristic fallback from melds, discards, missing-suit progress, and round phase",
        }


class RiskGate:
    def compute(
        self,
        game: MahjongGame,
        player_id: int,
        z_state: Dict[int, Dict[str, Any]],
        beliefs: Dict[str, Any],
        use_cusum_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        player = game.players[player_id]
        remaining = game.deck.remaining_count()
        start_balance = 10000
        score_gap = player.balance - start_balance
        max_conf = 0.0
        for belief in beliefs.values():
            try:
                max_conf = max(max_conf, float(belief.get("tenpai_confidence", 0.0)))
            except (TypeError, ValueError):
                pass

        drift_uncertainty = max((v.get("drift_score", 0.0) for v in z_state.values()), default=0.0)
        if use_cusum_uncertainty:
            # continuous_v7: blend in the CUSUM/entropy z_j upgrade instead of
            # the naive single-step drift alone. cusum_score is normalized by
            # its own flag threshold so it saturates at 1.0 exactly when a
            # change point is confirmed, keeping the blend in [0, 1].
            cusum_raw = max((v.get("cusum_score", 0.0) for v in z_state.values()), default=0.0)
            cusum_norm = _clip(cusum_raw / _CUSUM_THRESHOLD) if _CUSUM_THRESHOLD else 0.0
            entropy_term = max((v.get("entropy_uncertainty", 0.0) for v in z_state.values()), default=0.0)
            drift_uncertainty = _clip(0.5 * drift_uncertainty + 0.35 * cusum_norm + 0.15 * entropy_term)
        visible_threat = sum(1 for p in game.players if p.player_id != player_id and len(p.open_melds) >= 2)
        risk_budget = 0.45 * visible_threat + 0.35 * max_conf + 0.2 * _clip((40 - remaining) / 40.0)
        uncertainty = _clip(drift_uncertainty + (1.0 - max_conf) * 0.4)

        if risk_budget >= 0.75 or uncertainty >= 0.7:
            mode = "safe"
        elif risk_budget <= 0.45 and uncertainty <= 0.5 and score_gap <= 0:
            mode = "deceive"
        else:
            mode = "exploit"

        return {
            "mode_hint": mode,
            "risk_budget": round(_clip(risk_budget), 3),
            "uncertainty": round(_clip(uncertainty), 3),
            "score_gap": score_gap,
            "tiles_left": remaining,
            "z_drift_flags": {f"P{pid}": info.get("drift_flag", False) for pid, info in z_state.items()},
            "z_cusum_flags": {f"P{pid}": info.get("cusum_flag", False) for pid, info in z_state.items()},
        }


class MASKLLMAgent:
    def __init__(
        self,
        player_id: int = 0,
        llm: Optional[LLMCallable] = None,
        belief_llm: Optional[LLMCallable] = None,
        decision_llm: Optional[LLMCallable] = None,
        reranker_llm: Optional[LLMCallable] = None,
        gate_llm: Optional[LLMCallable] = None,
        neural_gate: Optional[Any] = None,
        gate_policy: str = "rule",
        use_mc_belief: bool = True,
        mc_oracle_samples: int = 30,
        mc_beta: float = 2.0,
        mc_danger_threshold: int = 1,
        dir_ready_threshold: int = 0,
        deceive_threat_ceiling: float = 0.5,
        forced_deceive: str = "off",
        deceive_style: str = "safe",
        threat_allow_break_ready: bool = False,
        threat_max_result_shanten: int = 0,
        threat_max_shanten_regret: int = 0,
        threat_min_ukeire_ratio: float = 1.0,
        threat_gate_threshold: float = 0.4,
        threat_gate_margin: float = 0.12,
        threat_min_delta: float = 0.03,
        threat_gate_mode: str = "cross",
        threat_response_model: str = "tell",
        threat_response_tell_weight: float = 1.0,
        threat_tell_window: int = 6,
        threat_max_start_shanten: int = 3,
        threat_require_non_exploit: bool = True,
        threat_require_real_target: bool = False,
        threat_target_max_shanten: int = 0,
        threat_target_signal: str = "oracle",
        threat_target_prob_threshold: float = 0.5,
        log_counterfactual: bool = False,
        use_candidate_reranker: bool = False,
        use_candidate_scoring: bool = False,
        reranker_max_candidates: int = 6,
        mc_seed: int = 0,
        # PPO-learned parameter overrides (None = use hardcoded defaults)
        ppo_params: Optional[Dict[str, float]] = None,
    ):
        self.player_id = player_id
        self.llm = llm
        self.belief_llm = belief_llm if belief_llm is not None else llm
        self.decision_llm = decision_llm if decision_llm is not None else llm
        self.reranker_llm = reranker_llm if reranker_llm is not None else self.decision_llm
        self.gate_llm = gate_llm
        self.neural_gate = neural_gate
        if gate_policy not in {"rule", "learned", "neural", "continuous", "continuous_v2", "continuous_v3", "continuous_v4", "continuous_v5", "continuous_v6", "continuous_v7"}:
            raise ValueError(f"Unknown gate_policy: {gate_policy}")
        self.gate_policy = gate_policy
        # Store PPO-learned parameters (override hardcoded defaults in _continuous_gate_action)
        self.ppo_params = ppo_params or {}
        self.belief_estimator = LLMBeliefEstimator(llm=self.belief_llm)
        self.risk_gate = RiskGate()
        self.tracker = PublicOpponentTracker([pid for pid in range(4) if pid != player_id])
        # B_phi = MC public-info posterior (the estimator that passed H1; the LLM
        # belief LoRA did not).  It estimates how dangerous each opponent thinks
        # *I* am, which is what the deceive lever tries to move.
        self.use_mc_belief = use_mc_belief
        self.mc_oracle_samples = mc_oracle_samples
        self.mc_beta = mc_beta
        self.mc_danger_threshold = mc_danger_threshold
        self.dir_ready_threshold = dir_ready_threshold
        self.deceive_threat_ceiling = deceive_threat_ceiling
        self.forced_deceive = forced_deceive
        self.deceive_style = deceive_style
        self.threat_allow_break_ready = threat_allow_break_ready
        self.threat_max_result_shanten = threat_max_result_shanten
        self.threat_max_shanten_regret = max(0, int(threat_max_shanten_regret))
        self.threat_min_ukeire_ratio = max(0.0, float(threat_min_ukeire_ratio))
        self.threat_gate_threshold = threat_gate_threshold
        self.threat_gate_margin = threat_gate_margin
        self.threat_min_delta = threat_min_delta
        self.threat_gate_mode = threat_gate_mode
        self.threat_response_model = threat_response_model
        self.threat_response_tell_weight = _clip(float(threat_response_tell_weight))
        self.threat_tell_window = threat_tell_window
        self.threat_max_start_shanten = threat_max_start_shanten
        self.threat_require_non_exploit = threat_require_non_exploit
        # Oracle ablation only: gates threat-style deceive on whether at least
        # one opponent is ACTUALLY close to winning right now (ground-truth
        # hand peek), instead of firing purely on our own projected tell
        # value.  Answers "does deceive pay off if aimed at real threats?" as
        # an upper bound -- not a deployable signal (real MASK can't see
        # opponent hands).
        self.threat_require_real_target = threat_require_real_target
        self.threat_target_max_shanten = threat_target_max_shanten
        # "oracle" = the ground-truth hand-peek gate above (upper bound only).
        # "mc" = deployable substitute: opponent_view_posterior() estimates each
        # opponent's tenpai-ish probability from public info only (melds +
        # discards), same MC machinery _mc_beliefs() already uses in the
        # opposite direction (observer j's belief about me).
        self.threat_target_signal = threat_target_signal
        self.threat_target_prob_threshold = threat_target_prob_threshold
        self.log_counterfactual = log_counterfactual
        self.use_candidate_reranker = use_candidate_reranker
        self.use_candidate_scoring = use_candidate_scoring
        self.reranker_max_candidates = max(2, int(reranker_max_candidates))
        self._mc_cache: Dict[int, Dict[str, Any]] = {}
        self.rng = random.Random(mc_seed)
        self.last_decision: Dict[str, Any] = {}
        self._last_deceive_signal: Dict[str, Any] = {}
        self._last_decision_llm_raw: Optional[str] = None
        self._last_decision_llm_parsed = False
        self._last_reranker: Dict[str, Any] = {}
        self._last_gate: Dict[str, Any] = {}

    def _mc_beliefs(self, game: MahjongGame) -> Dict[str, Any]:
        """Per-opponent: how dangerous does opponent j currently think I am."""
        key = len(game.history)
        if key in self._mc_cache:
            return self._mc_cache[key]
        beliefs: Dict[str, Any] = {}
        for j in range(4):
            if j == self.player_id:
                continue
            post = opponent_view_posterior(
                game, target_pid=self.player_id, observer_pid=j,
                num_samples=self.mc_oracle_samples, rng=self.rng, beta=self.mc_beta,
                max_shanten=self.mc_danger_threshold,
            )
            conf = float(post["tenpai_prob"])
            label = "yes" if conf >= 0.65 else ("no" if conf <= 0.35 else "uncertain")
            beliefs[f"P{j}"] = {"think_i_am_tenpai": label, "tenpai_confidence": round(conf, 3),
                                "source": "mc_posterior"}
        self._mc_cache[key] = beliefs
        return beliefs

    @staticmethod
    def _candidate_potential_fan(player: Any, hand_after_discard: List[Any]) -> int:
        """Best-case fan for a hand that reaches tenpai after this discard.

        Reuses Player.calculate_potential_fan() (max fan across all waiting
        tiles) by temporarily swapping in the post-discard hand; that method
        only reads hand_tiles/open_melds/missing_suit, so this is safe to
        undo in a finally block without leaving the live player mutated.
        """
        original_hand = player.hand_tiles
        player.hand_tiles = hand_after_discard
        try:
            fan, _ = player.calculate_potential_fan()
        finally:
            player.hand_tiles = original_hand
        return fan

    @staticmethod
    def _candidate_expected_fan(game: MahjongGame, player: Any, hand_for_wait: List[Any]) -> Optional[float]:
        """Remaining-copy-weighted expected fan for a tenpai hand's wait, replacing
        _candidate_potential_fan's best-case MAX with a realistic expectation
        (gate_policy="continuous_v4" only).

        Root cause diagnosed from the continuous_v3 real-LLM run (10 seeds x
        1000 games): _candidate_potential_fan takes the MAX fan across all
        waiting tiles of a multi-way wait, so _early_hu_decline_check
        routinely declined a guaranteed cheap win chasing a high-fan
        completion tile that had only 1-2 copies left live, while the cheap
        tile it walked away from often had 3-4 copies live. Traced against
        real game outcomes: decline-games realized avg fan 0.478, LOWER than
        the 0.599 avg fan of no-decline games -- the max-case estimate was
        systematically overselling the upside. This computes
        sum(remaining(t) * fan(t)) / sum(remaining(t)) instead, using the
        same "publicly visible copies" convention as
        policy_metrics.discard_progress_metrics (own hand + all discards +
        all open melds; no hidden-information cheating).
        """
        original_hand = player.hand_tiles
        player.hand_tiles = hand_for_wait
        try:
            is_ready, waiting_tiles = player.is_ready_with_missing_suit()
        finally:
            player.hand_tiles = original_hand
        if not is_ready or not waiting_tiles:
            return None

        visible: Counter = Counter((t.suit, t.number) for t in hand_for_wait)
        for table_player in game.players:
            visible.update((t.suit, t.number) for t in table_player.discarded_tiles)
            for meld in table_player.open_melds:
                visible.update((t.suit, t.number) for t in meld)

        weighted_fan_total = 0.0
        weight_total = 0
        for waiting_tile in waiting_tiles:
            remaining = max(0, 4 - visible[(waiting_tile.suit, waiting_tile.number)])
            if remaining == 0:
                continue
            fan, _ = FanCalculator.calculate_fan(
                hand_for_wait + [waiting_tile], waiting_tile,
                open_melds=player.open_melds, is_self_drawn=False,
                gang_count=player.gang_count, concealed_kong_count=player.concealed_kong_count,
            )
            weighted_fan_total += remaining * fan
            weight_total += remaining

        if weight_total == 0:
            return None
        return weighted_fan_total / weight_total

    @staticmethod
    def _hand_shape_fan_direction(hand_tiles: List[Any], open_melds: List[List[Any]], missing_suit: Any) -> float:
        """Cheap, pre-tenpai proxy for "which fan type is this hand shape drifting
        towards", built from the same fan types FanCalculator.calculate_fan()
        actually scores (rule_engine.py:321-379) so the weights below are not
        invented -- they mirror the real fan payout, just applied as continuous
        progress fractions instead of a binary post-win check:

          - purity fraction (progress towards 清一色, worth 2 fan when complete)
          - proto-triplet fraction: share of number-groups with >=2 copies
            already in hand/melds (progress towards 碰碰胡, worth 1 fan)
          - terminal-tile fraction (progress towards 带幺九, worth 2 fan)

        Unlike Player.calculate_potential_fan() this does not require shanten
        == 0 (no wait enumeration), so it is usable throughout the whole hand-
        building phase, not just on the final tenpai-reaching discard -- see
        _continuous_gate_action's fan_shaping branch for why that reach matters.
        Deliberately excludes 七对/龙七对 (already reflected by
        ShantenCalculator picking the best of standard/qidui shanten on its
        own) and 根 (concealed-kong count, not a discard-time direction choice).
        """
        all_tiles = list(hand_tiles) + [t for meld in open_melds for t in meld]
        counted = [t for t in all_tiles if t.suit != missing_suit]
        if not counted:
            return 0.0
        suit_counts = Counter(t.suit for t in counted)
        purity = max(suit_counts.values()) / len(counted)
        number_counts = Counter((t.suit, t.number) for t in counted)
        total_groups = max(1, len(counted) // 3)
        proto_triplets = sum(1 for c in number_counts.values() if c >= 2)
        pung_ratio = min(1.0, proto_triplets / total_groups)
        terminal_ratio = sum(1 for t in counted if t.number in (1, 9)) / len(counted)
        return 2.0 * purity + 1.0 * pung_ratio + 2.0 * terminal_ratio

    def _early_hu_decline_check(
        self, game: MahjongGame, valid_actions: List[str], gate: Dict[str, Any],
    ) -> Tuple[Optional[int], Optional[str]]:
        """Priority-2 early-hu opportunity-cost check (gate_policy="continuous_v3"
        only, called from _continuous_gate_action before the unconditional hu-take
        below -- default off so "continuous"/"continuous_v2" stay byte-identical).

        Diagnosed pattern this session (10-seed x 1000-game neural+LLM comparison):
        continuous/continuous_v2 win ~75% of hands (vs 33-42% for L0/L1) but at
        avg_hu_fan ~0.60 (vs 1.3-1.4 for L0/L1) -- the unconditional "take any hu"
        branch below cashes out on-table wins the instant they appear regardless of
        value, so hu_rate rises but fan_per_game barely moves. This check declines
        a currently-available win only when ALL of:
          - fan_now <= EARLY_HU_FAN_MAX: the win on the table right now is cheap
            (near-平胡), so the opportunity cost of walking away is small;
          - tiles_left >= EARLY_HU_TILES_LEFT_MIN: still early/mid-game, plenty of
            draws left to actually realize a bigger hand;
          - upside_fan >= fan_now + EARLY_HU_UPSIDE_MARGIN: staying in tenpai (or
            waiting, for a response-hu) has a *concretely* higher best-case fan --
            reuses the exact same calculate_potential_fan() wait enumeration the
            value term in _continuous_gate_action already trusts, not a new
            heuristic;
          - rho <= EARLY_HU_RHO_MAX and mode_hint != "safe": never decline a sure
            win while under threat -- a free win is itself the best defense.

        Returns (fan_now, decline_action). decline_action is None when the hu
        should be taken (any threshold fails, or no legal decline action exists);
        fan_now is still returned (or None if it could not be computed) so callers
        can log it regardless of the final decision.
        """
        EARLY_HU_FAN_MAX = 1
        EARLY_HU_TILES_LEFT_MIN = 20.0
        EARLY_HU_UPSIDE_MARGIN = 1
        EARLY_HU_RHO_MAX = 0.75

        rho = float(gate.get("risk_budget", 0.0))
        tiles_left = float(gate.get("tiles_left", 40.0))
        if tiles_left < EARLY_HU_TILES_LEFT_MIN:
            return None, None
        if rho > EARLY_HU_RHO_MAX or gate.get("mode_hint") == "safe":
            return None, None

        player = game.players[self.player_id]
        is_response = "n" in valid_actions
        if is_response:
            # No discard option on a response window -- the only way to
            # decline is "n" (pass), and the hand is unchanged either way, so
            # upside is whatever this same 13-tile tenpai hand's *other*
            # waiting tiles could still deliver.
            win_tile = game.last_discarded_tile
            if win_tile is None:
                return None, None
            all_tiles = player.hand_tiles + [win_tile]
            fan_now, _ = FanCalculator.calculate_fan(
                all_tiles, win_tile, open_melds=player.open_melds,
                is_self_drawn=False, gang_count=player.gang_count,
                concealed_kong_count=player.concealed_kong_count,
            )
            if fan_now > EARLY_HU_FAN_MAX:
                return fan_now, None
            upside_fan = self._candidate_potential_fan(player, player.hand_tiles)
            if upside_fan < fan_now + EARLY_HU_UPSIDE_MARGIN:
                return fan_now, None
            return fan_now, "n"

        # Self-draw: declining means discarding instead of calling "h". Score
        # every legal discard the same way the main candidate loop below does
        # (discard_progress_metrics for shanten, _candidate_potential_fan for
        # the resulting wait's best-case fan) and take whichever discard keeps
        # tenpai (shanten==0) with the highest upside -- not just the discard
        # that puts the just-drawn tile back, since a multi-wait hand can have
        # a better-value wait sitting on a *different* tile than the one just
        # drawn.
        win_tile = player.last_drawn_tile
        if win_tile is None:
            return None, None
        fan_now, _ = FanCalculator.calculate_fan(
            player.hand_tiles, win_tile, open_melds=player.open_melds,
            is_self_drawn=True, gang_count=player.gang_count,
            concealed_kong_count=player.concealed_kong_count,
        )
        if fan_now > EARLY_HU_FAN_MAX:
            return fan_now, None
        # _candidate_potential_fan always scores via is_self_drawn=False (the
        # 查大叫 convention Player.calculate_potential_fan follows), so it can
        # never reflect the +1 self-draw bonus baked into fan_now above.
        # Compare on the same structural (non-self-draw) basis so the margin
        # check isn't demanding a phantom extra fan that upside_fan can never
        # supply.
        fan_now_basis = max(0, fan_now - 1)

        best_decline_action: Optional[str] = None
        best_upside_fan = -1
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            tile_text = action[2:]
            tile = next((t for t in player.hand_tiles if str(t) == tile_text), None)
            if tile is None:
                continue
            metrics = discard_progress_metrics(game, self.player_id, action)
            if metrics is None or metrics["shanten"] != 0:
                continue
            hand_after_discard = player.hand_tiles.copy()
            hand_after_discard.remove(tile)
            upside_fan = self._candidate_potential_fan(player, hand_after_discard)
            if upside_fan > best_upside_fan:
                best_upside_fan = upside_fan
                best_decline_action = action

        if best_decline_action is None or best_upside_fan < fan_now_basis + EARLY_HU_UPSIDE_MARGIN:
            return fan_now, None
        return fan_now, best_decline_action

    def _early_hu_decline_check_expected(
        self, game: MahjongGame, valid_actions: List[str], gate: Dict[str, Any],
    ) -> Tuple[Optional[int], Optional[str]]:
        """v4 of the priority-2 early-hu opportunity-cost check (gate_policy=
        "continuous_v4" only) -- same gating (fan_now/tiles_left/rho/mode_hint)
        and same decline-vs-take structure as _early_hu_decline_check
        (continuous_v3), but replaces the MAX-case _candidate_potential_fan
        upside estimate with the remaining-copy-weighted expected fan from
        _candidate_expected_fan. See that method's docstring for the real-run
        diagnosis that motivated this. EARLY_HU_UPSIDE_MARGIN is retuned
        down from continuous_v3's 1 full fan to 0.15: an instrumented
        heuristic_fallback smoke probe (300 games, seed 40261627) showed
        real self-draw expected-value margins mostly clustering at 0.0-0.17
        with an occasional 0.33-0.5 -- a 0.5 margin (the first value tried)
        rejected every real decision in that sample, silently reducing v4
        to a no-op. 0.15 still requires a genuine, remaining-copy-weighted
        edge (not just noise) while remaining low enough to actually fire
        sometimes in real play (~21% decline rate observed in the same
        probe, vs continuous_v3's overly permissive ~36%).
        """
        EARLY_HU_FAN_MAX = 1
        EARLY_HU_TILES_LEFT_MIN = 20.0
        EARLY_HU_UPSIDE_MARGIN = 0.15
        EARLY_HU_RHO_MAX = 0.75

        rho = float(gate.get("risk_budget", 0.0))
        tiles_left = float(gate.get("tiles_left", 40.0))
        if tiles_left < EARLY_HU_TILES_LEFT_MIN:
            return None, None
        if rho > EARLY_HU_RHO_MAX or gate.get("mode_hint") == "safe":
            return None, None

        player = game.players[self.player_id]
        is_response = "n" in valid_actions
        if is_response:
            win_tile = game.last_discarded_tile
            if win_tile is None:
                return None, None
            all_tiles = player.hand_tiles + [win_tile]
            fan_now, _ = FanCalculator.calculate_fan(
                all_tiles, win_tile, open_melds=player.open_melds,
                is_self_drawn=False, gang_count=player.gang_count,
                concealed_kong_count=player.concealed_kong_count,
            )
            if fan_now > EARLY_HU_FAN_MAX:
                return fan_now, None
            upside_fan = self._candidate_expected_fan(game, player, player.hand_tiles)
            if upside_fan is None or upside_fan < fan_now + EARLY_HU_UPSIDE_MARGIN:
                return fan_now, None
            return fan_now, "n"

        win_tile = player.last_drawn_tile
        if win_tile is None:
            return None, None
        fan_now, _ = FanCalculator.calculate_fan(
            player.hand_tiles, win_tile, open_melds=player.open_melds,
            is_self_drawn=True, gang_count=player.gang_count,
            concealed_kong_count=player.concealed_kong_count,
        )
        if fan_now > EARLY_HU_FAN_MAX:
            return fan_now, None
        # Same self-draw/查大叫 basis mismatch as continuous_v3 -- see
        # _early_hu_decline_check's comment above.
        fan_now_basis = max(0, fan_now - 1)

        best_decline_action: Optional[str] = None
        best_upside_fan = -1.0
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            tile_text = action[2:]
            tile = next((t for t in player.hand_tiles if str(t) == tile_text), None)
            if tile is None:
                continue
            metrics = discard_progress_metrics(game, self.player_id, action)
            if metrics is None or metrics["shanten"] != 0:
                continue
            hand_after_discard = player.hand_tiles.copy()
            hand_after_discard.remove(tile)
            upside_fan = self._candidate_expected_fan(game, player, hand_after_discard)
            if upside_fan is not None and upside_fan > best_upside_fan:
                best_upside_fan = upside_fan
                best_decline_action = action

        if best_decline_action is None or best_upside_fan < fan_now_basis + EARLY_HU_UPSIDE_MARGIN:
            return fan_now, None
        return fan_now, best_decline_action

    def _early_hu_decline_check_expected_v5(
        self, game: MahjongGame, valid_actions: List[str], gate: Dict[str, Any],
    ) -> Tuple[Optional[int], Optional[str]]:
        """v5 of the priority-2 early-hu opportunity-cost check (gate_policy=
        "continuous_v5" only). Identical decline-vs-take logic to
        _early_hu_decline_check_expected (continuous_v4) -- same
        _candidate_expected_fan upside estimate, same EARLY_HU_UPSIDE_MARGIN
        (0.15) -- but with the two gating thresholds tightened:
        EARLY_HU_TILES_LEFT_MIN 20.0 -> 28.0 and EARLY_HU_RHO_MAX 0.75 ->
        0.5. Motivation: the v4 real-eval (10 seeds x 1000 games,
        50260627 series) showed avg_net/avg_hu_fan/fan_per_game
        significantly improved over v3 but still short of v2 (no
        priority-2 mechanism at all), and a decline-vs-outcome trace found
        v4's decline-games still realized lower avg fan than no-decline
        games (0.528 vs 0.597, gap narrowed from v3's 0.492 vs 0.595 but
        not closed). Since the upside estimator itself was already fixed
        in v4, the remaining lever is *when* the mechanism is allowed to
        fire at all: restricting it to only the safest, earliest-game
        situations (more turns left to actually draw the upgrade tile,
        much lower opponent threat) should cut the downside-heavy decline
        events that dragged v4's decline-game average down, at the cost of
        triggering less often.
        """
        EARLY_HU_FAN_MAX = 1
        EARLY_HU_TILES_LEFT_MIN = 28.0
        EARLY_HU_UPSIDE_MARGIN = 0.15
        EARLY_HU_RHO_MAX = 0.5

        rho = float(gate.get("risk_budget", 0.0))
        tiles_left = float(gate.get("tiles_left", 40.0))
        if tiles_left < EARLY_HU_TILES_LEFT_MIN:
            return None, None
        if rho > EARLY_HU_RHO_MAX or gate.get("mode_hint") == "safe":
            return None, None

        player = game.players[self.player_id]
        is_response = "n" in valid_actions
        if is_response:
            win_tile = game.last_discarded_tile
            if win_tile is None:
                return None, None
            all_tiles = player.hand_tiles + [win_tile]
            fan_now, _ = FanCalculator.calculate_fan(
                all_tiles, win_tile, open_melds=player.open_melds,
                is_self_drawn=False, gang_count=player.gang_count,
                concealed_kong_count=player.concealed_kong_count,
            )
            if fan_now > EARLY_HU_FAN_MAX:
                return fan_now, None
            upside_fan = self._candidate_expected_fan(game, player, player.hand_tiles)
            if upside_fan is None or upside_fan < fan_now + EARLY_HU_UPSIDE_MARGIN:
                return fan_now, None
            return fan_now, "n"

        win_tile = player.last_drawn_tile
        if win_tile is None:
            return None, None
        fan_now, _ = FanCalculator.calculate_fan(
            player.hand_tiles, win_tile, open_melds=player.open_melds,
            is_self_drawn=True, gang_count=player.gang_count,
            concealed_kong_count=player.concealed_kong_count,
        )
        if fan_now > EARLY_HU_FAN_MAX:
            return fan_now, None
        # Same self-draw/查大叫 basis mismatch as continuous_v3/v4.
        fan_now_basis = max(0, fan_now - 1)

        best_decline_action: Optional[str] = None
        best_upside_fan = -1.0
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            tile_text = action[2:]
            tile = next((t for t in player.hand_tiles if str(t) == tile_text), None)
            if tile is None:
                continue
            metrics = discard_progress_metrics(game, self.player_id, action)
            if metrics is None or metrics["shanten"] != 0:
                continue
            hand_after_discard = player.hand_tiles.copy()
            hand_after_discard.remove(tile)
            upside_fan = self._candidate_expected_fan(game, player, hand_after_discard)
            if upside_fan is not None and upside_fan > best_upside_fan:
                best_upside_fan = upside_fan
                best_decline_action = action

        if best_decline_action is None or best_upside_fan < fan_now_basis + EARLY_HU_UPSIDE_MARGIN:
            return fan_now, None
        return fan_now, best_decline_action

    def _continuous_gate_action(
        self,
        game: MahjongGame,
        valid_actions: List[str],
        z_state: Dict[int, Dict[str, Any]],
        beliefs: Dict[str, Any],
        gate: Dict[str, Any],
        fan_shaping: bool = False,
        early_hu_penalty: bool = False,
        early_hu_expected_value: bool = False,
        early_hu_tightened: bool = False,
        belief_shaping: bool = False,
    ) -> str:
        """Chapter-5/6 continuous risk-shaping gate: a_i = argmax_a [Q_base(a) + alpha*DeltaShape(a)],
        s.t. Q_base(a) >= Q* - tau_eff, alpha = f(u, rho).

        Unlike the rule/learned gate_policy branches, there is no discrete
        "is deceive eligible" boolean here: deceive-flavored discards are
        scored for every legal action and win only when their DeltaShape
        term outweighs the shanten/ukeire cost, weighted by alpha. This
        replaces the deceive_ready/gate_ok/require_real_target/
        require_non_exploit boolean chain used by gate_policy in
        {"rule", "learned"} -- see run_e2_shanten_filter_ladder discussion
        this session on why that chain starves training data.

        Q_base also includes a hand-value term: when a candidate discard
        reaches tenpai (shanten == 0), player.calculate_potential_fan() gives
        the exact best-case fan across all waiting tiles, and w_value*fan is
        added to Q_base so tenpai candidates aren't chosen purely on
        shanten/ukeire with no regard for how much the resulting hand is
        worth. Non-tenpai candidates get potential_fan=0 (their eventual fan
        isn't knowable yet) -- this deliberately limits the value term's
        effect to the endgame tenpai decision, where it was diagnosed
        (5-seed x 500-game neural+LLM comparison) that this gate wins far
        more often than L0/L1 (74% vs 32-41% hu_rate) but at roughly 40% of
        their average fan per win, leaving avg_net not ahead despite the
        much higher win/dealin-avoidance rate.

        Follow-up diagnosis (1-seed x 100-game heuristic_fallback smoke test):
        the Q_base value term above only breaks ties between simultaneously-
        tenpai-reaching candidates, and empirically those candidates almost
        always share the same potential_fan (85/85 tie-break opportunities in
        that run had zero fan spread), so it changed 0/100 games outright.
        The real lever is alpha: ΔShape (the tell-shifting deceive term) was
        firing just as often on worthless (0-1 fan) tenpai hands as on
        valuable ones -- 2162 deceive_windows but only 93 induced_dealin in
        the 5x500 run, i.e. most of that tell-exposure risk was spent on
        hands barely worth winning. `value_gate` below suppresses ΔShape's
        influence unless the best fan reachable *this turn* clears a floor,
        so deceive concentrates on hands where inducing a dealin is actually
        worth the exposure.

        Second follow-up diagnosis (instrumented probe, same 100-game run,
        1096 decisions): an earlier version of this gate folded value_gate
        into the alpha logit as an additive term (-kappa4*(1-value_gate)),
        which only pushes alpha asymptotically toward 0 and never reaches it
        exactly. That version produced 0 measurable game-outcome changes
        because every one of the 147 decisions where ΔShape actually flipped
        the pick away from best_q_only turned out to have a *zero* crossing
        alpha -- i.e. those two candidates were themselves tied on q_base
        (the same tie phenomenon as the docstring above, one level up), so
        any alpha > 0 already let ΔShape win regardless of magnitude.
        Suppression has to zero alpha exactly, not just shrink it, to have
        any effect -- hence value_gate multiplies alpha directly below
        instead of feeding the logit.

        belief_shaping (gate_policy="continuous_v6" only, default off so
        "continuous".."continuous_v5" stay byte-identical): `beliefs` is
        already the real B_phi -- decide() calls self._mc_beliefs(game) by
        default (use_mc_belief=True), the MC public-info oracle documented
        above (mask_llm.__init__) as "the estimator that passed H1" -- but
        until now it only ever fed RiskGate.compute()'s max_conf (i.e. alpha
        suppression), never the per-candidate DeltaShape term itself.
        Re-querying the oracle per candidate is too slow (MC sampling; the
        <100ms/decision budget already spends 3 oracle calls/decision on the
        once-per-decision `beliefs`), so this reuses that cached value: it
        scales/decomposes the existing cheap tell-heuristic term by how much
        headroom the oracle says opponents currently have to be surprised
        (avg_belief_conf), instead of leaving DeltaShape blind to B_phi.
        """
        if "h" in valid_actions:
            fan_now: Optional[int] = None
            decline_action: Optional[str] = None
            if early_hu_penalty:
                if early_hu_tightened:
                    fan_now, decline_action = self._early_hu_decline_check_expected_v5(game, valid_actions, gate)
                elif early_hu_expected_value:
                    fan_now, decline_action = self._early_hu_decline_check_expected(game, valid_actions, gate)
                else:
                    fan_now, decline_action = self._early_hu_decline_check(game, valid_actions, gate)
            if decline_action is not None:
                self.last_decision = {
                    "mode": "exploit", "action": decline_action,
                    "reason": f"decline low-value hu (fan_now={fan_now}) for upside",
                    "gate_policy": self.gate_policy, "z_state": z_state,
                    "beliefs": beliefs, "gate": gate, "alpha": 0.0,
                    "fan_now": fan_now, "early_hu_penalty": early_hu_penalty,
                    "early_hu_expected_value": early_hu_expected_value,
                    "early_hu_tightened": early_hu_tightened,
                }
                return decline_action
            self.last_decision = {
                "mode": "exploit", "action": "h", "reason": "take win",
                "gate_policy": self.gate_policy, "z_state": z_state,
                "beliefs": beliefs, "gate": gate, "alpha": 0.0,
                "fan_now": fan_now, "early_hu_penalty": early_hu_penalty,
                "early_hu_expected_value": early_hu_expected_value,
                "early_hu_tightened": early_hu_tightened,
            }
            return "h"

        discard_actions = [a for a in valid_actions if a.startswith("d ")]
        if not discard_actions:
            # No discard to score (e.g. only "g" available) -- fall back to
            # the existing exploit path rather than inventing new handling.
            action, reason = self._exploit_action(game, valid_actions)
            self.last_decision = {
                "mode": "exploit", "action": action, "reason": reason,
                "gate_policy": self.gate_policy, "z_state": z_state,
                "beliefs": beliefs, "gate": gate, "alpha": 0.0,
            }
            return action

        player = game.players[self.player_id]
        progress: Dict[str, Dict[str, int]] = {}
        for action in discard_actions:
            value = discard_progress_metrics(game, self.player_id, action)
            if value is not None:
                progress[action] = value
        if not progress:
            action, reason = self._exploit_action(game, valid_actions)
            self.last_decision = {
                "mode": "exploit", "action": action, "reason": reason,
                "gate_policy": self.gate_policy, "z_state": z_state,
                "beliefs": beliefs, "gate": gate, "alpha": 0.0,
            }
            return action

        best_shanten = min(value["shanten"] for value in progress.values())
        best_effective_copies = max(
            (value["effective_copies"] for value in progress.values() if value["shanten"] == best_shanten),
            default=0,
        )
        # tau_eff feasibility constraint (Q_base(a) >= Q* - tau_eff), reusing
        # the same shanten-regret/ukeire-ratio tolerance knobs already used
        # by the rule-gated candidate generators in mask_candidates.py.
        feasible = [
            action for action, value in progress.items()
            if value["shanten"] <= best_shanten + self.threat_max_shanten_regret
            and value["effective_copies"] >= best_effective_copies * self.threat_min_ukeire_ratio
        ]
        if not feasible:
            feasible = list(progress.keys())

        # Precompute potential_fan per feasible candidate once. Needed before
        # alpha (value_gate uses the best of these) as well as inside Q_base,
        # so this is a single pass shared by both instead of two calls.
        potential_fan_by_action: Dict[str, int] = {}
        tile_by_action: Dict[str, Any] = {}
        for action in feasible:
            value = progress[action]
            tile_text = action[2:]
            tile = next((t for t in player.hand_tiles if str(t) == tile_text), None)
            tile_by_action[action] = tile
            potential_fan = 0
            if tile is not None and value["shanten"] == 0:
                hand_after_discard = player.hand_tiles.copy()
                hand_after_discard.remove(tile)
                potential_fan = self._candidate_potential_fan(player, hand_after_discard)
            potential_fan_by_action[action] = potential_fan
        max_potential_fan = max(potential_fan_by_action.values(), default=0)

        # fan_shaping (gate_policy="continuous_v2" only, default off so the
        # original "continuous" gate_policy stays byte-identical): extends
        # fan-awareness to the whole hand-building phase instead of only the
        # final tenpai-reaching discard. potential_fan above is exact but only
        # non-zero once shanten==0 (no wait enumeration possible earlier);
        # shape_direction_by_action is a cheap proxy usable at any shanten so
        # candidates get scored on which fan type they drift towards well
        # before tenpai, not just tie-broken once every option already ties
        # on realized fan (diagnosed as a 0/100-game no-op in the docstring
        # above for the non-shaped path).
        shape_direction_by_action: Dict[str, float] = {}
        if fan_shaping:
            for action in feasible:
                tile = tile_by_action[action]
                if tile is None:
                    shape_direction_by_action[action] = 0.0
                    continue
                hand_after_discard = player.hand_tiles.copy()
                hand_after_discard.remove(tile)
                shape_direction_by_action[action] = self._hand_shape_fan_direction(
                    hand_after_discard, player.open_melds, player.missing_suit
                )
        max_shape_direction = max(shape_direction_by_action.values(), default=0.0)

        # alpha = f(u, rho) * value_gate: risk appetite rises when behind on
        # score or late in the hand (both already surfaced by
        # RiskGate.compute()); value_gate then multiplies the result so a
        # worthless hand (max_potential_fan == 0) zeroes alpha exactly --
        # not just shrinks it -- fully suppressing ΔShape's pull unless the
        # best fan reachable this turn is worth the tell-exposure a
        # deceive-flavored discard costs. See the docstring above for why
        # this has to be a multiplicative gate rather than an additive logit
        # term.
        rho = float(gate.get("risk_budget", 0.0))
        u = float(gate.get("uncertainty", 0.0))
        score_gap = float(gate.get("score_gap", 0.0))
        tiles_left = float(gate.get("tiles_left", 40.0))
        behind = _clip(-score_gap / 3000.0)
        late = _clip((40.0 - tiles_left) / 40.0)
        risk_appetite = 0.5 * behind + 0.5 * late
        if fan_shaping:
            # Pre-tenpai shape direction is a guess, not a confirmed wait --
            # discount it (0.5x) relative to potential_fan (exact, once
            # tenpai) so early-hand shape drift alone can open the gate at
            # most halfway; only a confirmed fan>=3 tenpai (or a very strong
            # 0.75+ shape read) reaches value_gate==1.0.
            value_gate = _clip(max(max_potential_fan, 0.5 * max_shape_direction) / 3.0)
        else:
            value_gate = _clip(max_potential_fan / 3.0)
        kappa1 = self.ppo_params.get("kappa1", 3.0)
        kappa2 = self.ppo_params.get("kappa2", 3.0)
        kappa3 = self.ppo_params.get("kappa3", 4.0)
        rho_max = self.ppo_params.get("rho_max", 0.75)
        logit = kappa1 * risk_appetite - kappa2 * u - kappa3 * (1.0 if rho > rho_max else 0.0)
        alpha = (1.0 / (1.0 + math.exp(-logit))) * value_gate

        tell_before = self._discard_tell_threat(game)
        avg_belief_conf = 0.0
        if belief_shaping:
            confs = [float(b.get("tenpai_confidence", 0.0)) for b in beliefs.values()]
            avg_belief_conf = _clip(sum(confs) / len(confs)) if confs else 0.0
        w_shanten = self.ppo_params.get("w_shanten", 100.0)
        w_ukeire = self.ppo_params.get("w_ukeire", 1.0)
        w_tell = self.ppo_params.get("w_tell", 100.0)
        w_value = self.ppo_params.get("w_value", 10.0)
        # w_shape sits between w_ukeire (1.0, a minor tie-break today) and
        # w_value (10.0, the realized-fan term) -- big enough to steer
        # direction among near-tied shanten/ukeire candidates, small enough
        # that it never outweighs an actual shanten-level or ukeire-count
        # difference (shanten dominance is exactly what min-shanten play
        # needs to keep; this only biases which of several similarly-fast
        # paths gets picked).
        w_shape = self.ppo_params.get("w_shape", 3.0)
        # belief_shaping weights: w_b matches w_tell's magnitude (the belief-
        # scaled tell term replaces the raw tell term one-for-one); w_d/w_f
        # are a first-pass guess, to be checked against the smoke test's
        # candidate_scores spread before the full 10-seed run.
        w_b = self.ppo_params.get("w_b", 100.0)
        w_d = self.ppo_params.get("w_d", 25.0)
        w_f = self.ppo_params.get("w_f", 25.0)
        scored: Dict[str, Dict[str, float]] = {}
        for action in feasible:
            value = progress[action]
            tile = tile_by_action[action]
            potential_fan = potential_fan_by_action[action]
            q_base = (
                -w_shanten * value["shanten"]
                + w_ukeire * value["effective_copies"]
                + w_value * potential_fan
            )
            if fan_shaping:
                q_base += w_shape * shape_direction_by_action.get(action, 0.0)
            tell_after = self._discard_tell_threat(game, extra_tile=tile) if tile is not None else tell_before
            b_term = d_term = f_term = 0.0
            if belief_shaping:
                b_term = (tell_after - tell_before) * (1.0 - avg_belief_conf)
                hand_after_discard = player.hand_tiles.copy()
                if tile is not None:
                    hand_after_discard.remove(tile)
                    d_term = (1.0 - avg_belief_conf) if within_shanten(hand_after_discard, player.missing_suit, 0) else 0.0
                f_term = tell_after * avg_belief_conf * _clip(value["shanten"] / 3.0)
                delta_shape = w_b * b_term + w_d * d_term + w_f * f_term
            else:
                delta_shape = w_tell * (tell_after - tell_before)
            scored[action] = {
                "q_base": q_base, "delta_shape": delta_shape, "potential_fan": potential_fan,
                "score": q_base + alpha * delta_shape,
                "b_term": b_term, "d_term": d_term, "f_term": f_term,
            }

        best_action = max(scored, key=lambda a: scored[a]["score"])
        # Includes the value term (potential_fan) since that's part of Q_base,
        # not DeltaShape -- "deceive" should only fire when the tell-shift
        # term (not hand value) is what flipped the pick away from this.
        best_q_only = max(feasible, key=lambda a: scored[a]["q_base"])
        if best_action != best_q_only and alpha * scored[best_action]["delta_shape"] > 0:
            mode = "deceive"
        elif rho > rho_max or gate.get("mode_hint") == "safe":
            mode = "safe"
        else:
            mode = "exploit"

        self.last_decision = {
            "mode": mode,
            "action": best_action,
            "reason": f"continuous gate alpha={alpha:.3f} rho={rho:.3f} u={u:.3f}",
            "gate_policy": self.gate_policy,
            "alpha": round(alpha, 4),
            "risk_budget": rho,
            "uncertainty": u,
            "value_gate": round(value_gate, 4),
            "max_potential_fan": max_potential_fan,
            "fan_shaping": fan_shaping,
            "early_hu_penalty": early_hu_penalty,
            "early_hu_expected_value": early_hu_expected_value,
            "early_hu_tightened": early_hu_tightened,
            "max_shape_direction": round(max_shape_direction, 4),
            "q_base": scored[best_action]["q_base"],
            "delta_shape": scored[best_action]["delta_shape"],
            "potential_fan": scored[best_action]["potential_fan"],
            "belief_shaping": belief_shaping,
            "avg_belief_conf": round(avg_belief_conf, 4),
            "b_term": scored[best_action]["b_term"],
            "d_term": scored[best_action]["d_term"],
            "f_term": scored[best_action]["f_term"],
            "candidate_scores": {a: round(v["score"], 3) for a, v in scored.items()},
            "z_state": z_state,
            "beliefs": beliefs,
            "gate": gate,
            "public_gate": gate,
        }
        return best_action

    def decide(self, game: MahjongGame, valid_actions: Optional[List[str]] = None) -> str:
        self._last_decision_llm_raw = None
        self._last_decision_llm_parsed = False
        self._last_reranker = {"enabled": self.use_candidate_reranker, "used": False}
        self._last_gate = {"policy": self.gate_policy, "used": False}
        valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, self.player_id)
        z_state = self.tracker.update_from_game(game)
        beliefs = self._mc_beliefs(game) if self.use_mc_belief else self.belief_estimator.infer_all(game, self.player_id)
        use_cusum_uncertainty = self.gate_policy == "continuous_v7"
        public_gate = self.risk_gate.compute(game, self.player_id, z_state, beliefs={}, use_cusum_uncertainty=use_cusum_uncertainty)
        gate = self.risk_gate.compute(game, self.player_id, z_state, beliefs, use_cusum_uncertainty=use_cusum_uncertainty)

        if "h" in valid_actions:
            self.last_decision = {
                "mode": "exploit",
                "action": "h",
                "reason": "common take-win rule",
                "gate_policy": self.gate_policy,
                "z_state": z_state,
                "beliefs": beliefs,
                "gate": gate,
                "public_gate": public_gate,
                "common_take_win": True,
            }
            return "h"

        if self.gate_policy in ("continuous", "continuous_v2", "continuous_v3", "continuous_v4", "continuous_v5", "continuous_v6", "continuous_v7"):
            return self._continuous_gate_action(
                game, valid_actions, z_state, beliefs, gate,
                fan_shaping=(self.gate_policy in ("continuous_v2", "continuous_v3", "continuous_v4", "continuous_v5", "continuous_v6", "continuous_v7")),
                early_hu_penalty=(self.gate_policy in ("continuous_v3", "continuous_v4", "continuous_v5", "continuous_v6", "continuous_v7")),
                early_hu_expected_value=(self.gate_policy in ("continuous_v4", "continuous_v5", "continuous_v6", "continuous_v7")),
                early_hu_tightened=(self.gate_policy in ("continuous_v5", "continuous_v6", "continuous_v7")),
                belief_shaping=(self.gate_policy in ("continuous_v6", "continuous_v7")),
            )

        player = game.players[self.player_id]
        own_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
        own_danger = own_shanten <= self.mc_danger_threshold
        dir_ready = own_shanten <= self.dir_ready_threshold
        perceived = max((float(b.get("tenpai_confidence", 0.0)) for b in beliefs.values()), default=0.0)
        exploit_cache: Optional[Tuple[str, str]] = None

        def get_exploit_action() -> Tuple[str, str]:
            nonlocal exploit_cache
            if exploit_cache is None:
                # Reranker on/off must share the exact same exploit baseline.
                # Reranking is a post-policy operation only.
                exploit_cache = self._exploit_action(game, valid_actions)
            return exploit_cache

        # B_phi-driven, risk-gated mode:
        #   - take a win;
        #   - defend if opponents look dangerous to me and I have no hand;
        #   - safe/DIR deception triggers only when ready (look safe, invite deal-in);
        #   - threat/FFR deception triggers only when not dangerous (look dangerous,
        #     induce needless folds while the FFR oracle still counts opportunities);
        #   - otherwise exploit (push to win normally via decision_llm / min-shanten).
        ffr_ready = self.deceive_style == "threat" and not own_danger
        deceive_ready = ffr_ready if self.deceive_style == "threat" else dir_ready
        learned_gate_active = self.gate_policy == "learned" and self.gate_llm is not None
        neural_gate_active = self.gate_policy == "neural" and self.neural_gate is not None
        should_attempt_deceive = deceive_ready or self.forced_deceive == "always" or neural_gate_active
        deceive_action = (
            self._deceptive_discard(
                game,
                valid_actions,
                beliefs,
                relaxed_gate=(learned_gate_active or neural_gate_active),
            )
            if should_attempt_deceive
            else None
        )
        deceive_block_reason = None
        if not should_attempt_deceive:
            deceive_block_reason = "not_attempted_by_rule_gate"
        elif not deceive_action:
            deceive_block_reason = (
                (self._last_deceive_signal or {}).get("blocked_reason")
                or "candidate_unavailable"
            )
        elif not deceive_action.startswith("d "):
            deceive_block_reason = "candidate_not_discard"
        can_deceive = deceive_action is not None and deceive_action.startswith("d ")
        counterfactual_exploit_action = None
        disguise_equals_exploit = None
        if can_deceive and self.deceive_style == "threat" and self.threat_require_non_exploit:
            counterfactual_exploit_action, _ = get_exploit_action()
            disguise_equals_exploit = bool(counterfactual_exploit_action == deceive_action)
            if disguise_equals_exploit or not counterfactual_exploit_action.startswith("d "):
                can_deceive = False
                deceive_block_reason = (
                    "same_as_exploit"
                    if disguise_equals_exploit
                    else "counterfactual_exploit_not_discard"
                )
        if can_deceive:
            deceive_block_reason = None
        force_deceive = self.forced_deceive == "always" or (
            self.forced_deceive == "eligible" and deceive_ready and can_deceive
        )
        baseline_action: Optional[str] = None
        safe_override_applied = False
        if "h" in valid_actions:
            action, mode, reason = "h", "exploit", "take win"
        elif learned_gate_active or neural_gate_active:
            baseline_action, baseline_reason = get_exploit_action()
            safe_action = self._safe_discard(
                game,
                valid_actions,
                fallback_action=baseline_action,
            )
            available_modes = ["exploit"]
            if any(item.startswith("d ") for item in valid_actions):
                available_modes.append("safe")
            if can_deceive:
                available_modes.append("deceive")
            if neural_gate_active:
                selected_mode = self._neural_gate_mode(game, z_state, beliefs, gate, available_modes)
            else:
                selected_mode = self._learned_gate_mode(
                    game,
                    z_state,
                    beliefs,
                    gate,
                    available_modes,
                )
            if selected_mode == "deceive" and can_deceive:
                action, mode = deceive_action, "deceive"
            elif selected_mode == "safe":
                action, mode = safe_action, "safe"
                safe_override_applied = action != baseline_action
            else:
                action, mode = baseline_action, "exploit"
            reason = f"learned gate selected {mode}; {baseline_reason}"
        elif self.forced_deceive == "always" and force_deceive and can_deceive:
            action = deceive_action
            mode = "deceive"
            reason = f"forced-{self.forced_deceive} deceive-{self.deceive_style} ablation"
        elif public_gate["mode_hint"] == "safe":
            baseline_action, baseline_reason = get_exploit_action()
            action = self._safe_discard(game, valid_actions, fallback_action=baseline_action)
            safe_override_applied = action != baseline_action
            if action != baseline_action:
                mode, reason = "safe", "L1 public z/risk no-regret safety shield"
            else:
                mode = "safe"
                reason = f"L1 safety gate kept baseline; {baseline_reason}"
        elif force_deceive and can_deceive:
            action = deceive_action
            mode = "deceive"
            reason = f"forced-{self.forced_deceive} deceive-{self.deceive_style} ablation"
        elif (
            deceive_ready
            and can_deceive
            and (self.deceive_style == "threat" or perceived <= self.deceive_threat_ceiling)
        ):
            action = deceive_action
            if self.deceive_style == "threat":
                reason = "not-dangerous + deceive-threat public signal"
            else:
                reason = "ready + deceive-safe public signal"
            mode = "deceive"
        elif own_danger:
            action, reason = get_exploit_action()
            mode = "exploit"
        else:
            action, reason = get_exploit_action()
            mode = "exploit"

        rule_action = action
        if self.use_candidate_reranker and self.reranker_llm is not None:
            action, self._last_reranker = self._rerank_action(
                game,
                valid_actions,
                mode,
                rule_action,
                beliefs,
                gate,
            )
            if self._last_reranker.get("used"):
                reason = f"{reason}; candidate reranker"

        # Counterfactual: what would MASK have discarded if it were NOT disguising
        # (i.e., the exploit action) at this same deceive state?  If the disguise
        # equals the exploit action, deceive changed no public action -> it is a
        # no-op, and this measures that directly instead of inferring it.
        if mode == "deceive" and self.log_counterfactual:
            if counterfactual_exploit_action is None:
                counterfactual_exploit_action, _ = get_exploit_action()
            disguise_equals_exploit = bool(counterfactual_exploit_action == action)

        self.last_decision = {
            "mode": mode,
            "action": action,
            "rule_action": rule_action,
            "reason": reason,
            "baseline_action": baseline_action,
            "safe_override_applied": safe_override_applied,
            "counterfactual_exploit_action": counterfactual_exploit_action,
            "disguise_equals_exploit": disguise_equals_exploit,
            "own_shanten": int(own_shanten),
            "own_danger_threshold": self.mc_danger_threshold,
            "dir_ready_threshold": self.dir_ready_threshold,
            "deceive_ready": bool(deceive_ready),
            "ffr_ready": bool(ffr_ready),
            "deceive_attempted": bool(should_attempt_deceive),
            "deceive_action": deceive_action,
            "can_deceive": bool(can_deceive),
            "deceive_block_reason": deceive_block_reason,
            "perceived_threat_of_me": round(perceived, 3),
            "forced_deceive": self.forced_deceive,
            "deceive_style": self.deceive_style,
            "threat_allow_break_ready": self.threat_allow_break_ready,
            "threat_max_result_shanten": self.threat_max_result_shanten,
            "threat_max_shanten_regret": self.threat_max_shanten_regret,
            "threat_min_ukeire_ratio": self.threat_min_ukeire_ratio,
            "threat_gate_threshold": self.threat_gate_threshold,
            "threat_gate_margin": self.threat_gate_margin,
            "threat_min_delta": self.threat_min_delta,
            "threat_gate_mode": self.threat_gate_mode,
            "threat_response_model": self.threat_response_model,
            "threat_response_tell_weight": self.threat_response_tell_weight,
            "threat_tell_window": self.threat_tell_window,
            "threat_max_start_shanten": self.threat_max_start_shanten,
            "threat_require_non_exploit": self.threat_require_non_exploit,
            "threat_require_real_target": self.threat_require_real_target,
            "threat_target_max_shanten": self.threat_target_max_shanten,
            "threat_target_signal": self.threat_target_signal,
            "threat_target_prob_threshold": self.threat_target_prob_threshold,
            "deceive_signal": self._last_deceive_signal,
            "z_state": z_state,
            "beliefs": beliefs,
            "public_gate": public_gate,
            "gate": gate,
            "decision_llm_raw": self._last_decision_llm_raw,
            "decision_llm_parsed": self._last_decision_llm_parsed,
            "prompt_schema": (
                "shared_base_action_v1"
                if mode == "exploit" and self.decision_llm is not None
                else "rule_selected_mode_action"
            ),
            "candidate_reranker": self._last_reranker,
            "gate_policy": self.gate_policy,
            "learned_gate": self._last_gate,
        }
        return action

    def _learned_gate_mode(
        self,
        game: MahjongGame,
        z_state: Dict[int, Dict[str, Any]],
        beliefs: Dict[str, Any],
        gate: Dict[str, Any],
        available_modes: List[str],
    ) -> str:
        prompt = build_gate_decision_prompt(
            game,
            self.player_id,
            z_state,
            beliefs,
            gate,
            available_modes,
        )
        gate_system = (
            "你是四川麻将 MASK 三态风险门控器。只能选择给定候选中的一个英文标签。"
            "标签含义：exploit=正常进攻/基础策略，safe=防守安全，deceive=信念塑形欺骗。"
            "严禁输出 explore；如果想表达进攻，必须输出 exploit。"
        )
        ranker = getattr(self.gate_llm, "rank_candidates", None) if self.gate_llm is not None else None
        if callable(ranker):
            try:
                selected, scores = ranker(prompt, available_modes, system_prompt=gate_system)
                self._last_gate = {
                    "policy": "learned",
                    "used": True,
                    "available_modes": list(available_modes),
                    "selected_mode": selected,
                    "raw": json.dumps(scores, ensure_ascii=False, sort_keys=True),
                    "parsed": True,
                    "scoring": "candidate_logprob",
                    "candidate_scores": scores,
                }
                return selected
            except Exception as exc:
                self._last_gate = {
                    "policy": "learned",
                    "used": True,
                    "available_modes": list(available_modes),
                    "selected_mode": "exploit",
                    "raw": repr(exc),
                    "parsed": False,
                    "scoring": "candidate_logprob_error",
                }

        raw = self.gate_llm(prompt) if self.gate_llm is not None else "exploit"
        parsed = _safe_json_loads(raw)
        requested = str(parsed.get("mode", "")) if parsed else raw.strip()
        if requested.strip().lower() == "explore":
            requested = "exploit"
        match = re.search(r"\b(exploit|safe|deceive)\b", requested)
        selected = match.group(1) if match and match.group(1) in available_modes else "exploit"
        self._last_gate = {
            "policy": "learned",
            "used": True,
            "available_modes": list(available_modes),
            "selected_mode": selected,
            "raw": raw,
            "parsed": bool(match),
        }
        return selected

    def _neural_gate_mode(
        self,
        game: MahjongGame,
        z_state: Dict[int, Dict[str, Any]],
        beliefs: Dict[str, Any],
        gate: Dict[str, Any],
        available_modes: List[str],
    ) -> str:
        selected = "exploit"
        info: Dict[str, Any] = {}
        parsed = False
        try:
            selected, info = self.neural_gate.predict(
                game,
                self.player_id,
                z_state,
                beliefs,
                gate,
                available_modes,
            )
            parsed = selected in available_modes
        except Exception as exc:
            info = {"error": repr(exc)}
            selected = "exploit"
        if selected not in available_modes:
            selected = "exploit"
            parsed = False
        self._last_gate = {
            "policy": "neural",
            "used": True,
            "parsed": parsed,
            "available_modes": list(available_modes),
            "selected_mode": selected,
            **info,
        }
        return selected

    def _exploit_action(self, game: MahjongGame, valid_actions: List[str]) -> Tuple[str, str]:
        if self.decision_llm is not None:
            prompt = build_base_decision_prompt(game, self.player_id, valid_actions=valid_actions)
            raw = self.decision_llm(prompt)
            parsed = _safe_json_loads(raw)
            self._last_decision_llm_raw = raw
            self._last_decision_llm_parsed = bool(parsed and "action" in parsed)
            if parsed and "action" in parsed:
                return legalize_action(str(parsed.get("action", "")), valid_actions), "exploit (decision LLM)"
            action = legalize_action(raw, valid_actions)
            return action, "exploit (decision LLM text)"
        return self._min_shanten_discard(game, valid_actions), "exploit (min-shanten)"

    def _rerank_action(
        self,
        game: MahjongGame,
        valid_actions: List[str],
        mode: str,
        rule_action: str,
        beliefs: Dict[str, Any],
        gate: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        if not any(action.startswith("d ") for action in valid_actions):
            return rule_action, {
                "enabled": True,
                "used": False,
                "skip_reason": "no_discard_candidates",
            }

        # Local import avoids a module cycle: mask_candidates type-checks this
        # class, while reranking is only needed at runtime after both modules load.
        from mask_candidates import build_mode_candidates

        rng_state = self.rng.getstate()
        try:
            candidate_set = build_mode_candidates(
                self,
                game,
                valid_actions,
                mode,
                mode_already_selected=True,
                beliefs=beliefs,
            )
        finally:
            self.rng.setstate(rng_state)

        candidates = []
        for candidate in candidate_set.actions:
            if candidate in valid_actions and candidate not in candidates:
                candidates.append(candidate)
        if len(candidates) > self.reranker_max_candidates:
            candidates = candidates[: self.reranker_max_candidates]
        if len(candidates) < 2:
            return rule_action, {
                "enabled": True,
                "used": False,
                "skip_reason": "fewer_than_two_candidates",
                "candidates": candidates,
            }
        if self.use_candidate_scoring:
            candidates = sorted(candidates)

        prompt = build_mask_decision_prompt(
            game,
            self.player_id,
            beliefs,
            gate,
            valid_actions=candidates,
            output_format="action" if self.use_candidate_scoring else "json",
        )
        reference_line = "" if self.use_candidate_scoring else f"规则基线动作: {rule_action}\n"
        output_instruction = (
            "只输出一个候选动作本身，例如: d 3万。不要输出解释或 JSON。"
            if self.use_candidate_scoring
            else '严格输出 JSON: {"action": "候选动作", "reason": "一句话依据"}'
        )
        prompt = f"""
{prompt}

【规则约束候选重排】
外层规则已经固定当前模式为: {mode}
{reference_line}候选动作: {", ".join(candidates)}

不要改变模式，只在候选动作中选择预期收益最高的一项。
{output_instruction}
""".strip()
        if self.use_candidate_scoring and hasattr(self.reranker_llm, "rank_candidates"):
            selected, candidate_scores = self.reranker_llm.rank_candidates(prompt, candidates)
            raw = selected
            output_parsed = True
            selection_method = "conditional_logprob"
        else:
            raw = self.reranker_llm(prompt)
            parsed = _safe_json_loads(raw)
            requested = str(parsed.get("action", "")) if parsed else raw
            selected = legalize_action(requested, candidates)
            output_parsed = bool(parsed and "action" in parsed) or raw.strip() in candidates
            candidate_scores = None
            selection_method = "generation"
        return selected, {
            "enabled": True,
            "used": True,
            "mode": mode,
            "rule_action": rule_action,
            "selected_action": selected,
            "changed_action": selected != rule_action,
            "candidates": candidates,
            "candidate_source": candidate_set.metadata.get("candidate_source"),
            "raw": raw,
            "parsed": output_parsed,
            "selection_method": selection_method,
            "candidate_scores": candidate_scores,
        }

    def _safe_discard(
        self,
        game: MahjongGame,
        valid_actions: List[str],
        fallback_action: Optional[str] = None,
    ) -> str:
        if fallback_action in {"h", "g"}:
            return fallback_action
        player = game.players[self.player_id]
        public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        progress = {}
        for action in valid_actions:
            value = discard_progress_metrics(game, self.player_id, action)
            if value is not None:
                progress[action] = value
        best_shanten = min((value["shanten"] for value in progress.values()), default=99)
        best_effective_copies = max(
            (
                value["effective_copies"]
                for value in progress.values()
                if value["shanten"] == best_shanten
            ),
            default=0,
        )
        public_safe = []
        for action, value in progress.items():
            tile = next((tile for tile in player.hand_tiles if str(tile) == action[2:]), None)
            if (
                tile is not None
                and (tile.suit, tile.number) in public
                and value["shanten"] == best_shanten
                and value["effective_copies"] >= best_effective_copies
            ):
                public_safe.append(action)
        if fallback_action in public_safe:
            return fallback_action
        if public_safe:
            return public_safe[0]
        return fallback_action or self._min_shanten_discard(game, valid_actions)

    def _deceptive_discard(
        self,
        game: MahjongGame,
        valid_actions: List[str],
        beliefs: Optional[Dict[str, Any]] = None,
        relaxed_gate: bool = False,
    ) -> str:
        self._last_deceive_signal = {}
        if self.deceive_style == "threat":
            return self._threat_discard(game, valid_actions, beliefs, relaxed_gate=relaxed_gate)
        return self._disguise_discard(game, valid_actions)

    def _discard_tell_threat(self, game: MahjongGame, extra_tile: Any = None) -> float:
        player = game.players[self.player_id]
        recent = list(player.discarded_tiles)
        if extra_tile is not None:
            recent.append(extra_tile)
        recent = recent[-self.threat_tell_window:]
        others = {
            (t.suit, t.number)
            for p in game.players
            if p.player_id != self.player_id
            for t in p.discarded_tiles
        }
        push = fold = 0.0
        for tile in recent:
            if tile.number in (4, 5, 6):
                push += 1.0
            elif tile.number in (1, 9):
                fold += 1.0
            elif (tile.suit, tile.number) in others:
                fold += 0.5
        tell = (push - fold) / max(1, len(recent))
        melds = len(player.open_melds)
        tiles_left = game.deck.remaining_count()
        base = 0.30 + 0.06 * melds + 0.10 * _clip((40 - tiles_left) / 40.0)
        return _clip(base + 0.35 * tell)

    def _disguise_discard(self, game: MahjongGame, valid_actions: List[str]) -> str:
        """Discard a tile that KEEPS tenpai and looks safe (publicly discarded or
        terminal), so opponents read me as folding and keep dealing dangerous tiles."""
        player = game.players[self.player_id]
        public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        keep_safe, keep_any = [], []
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            for tile in player.hand_tiles:
                if str(tile) != action[2:]:
                    continue
                temp = player.hand_tiles.copy()
                temp.remove(tile)
                if within_shanten(temp, player.missing_suit, self.dir_ready_threshold):
                    keep_any.append(action)
                    if (tile.suit, tile.number) in public or tile.number in (1, 9):
                        keep_safe.append(action)
                break
        if keep_safe:
            return keep_safe[0]
        if keep_any:
            return keep_any[0]
        if self.forced_deceive == "always":
            for action in valid_actions:
                if action.startswith("d "):
                    return action
        return self._min_shanten_discard(game, valid_actions)

    def _threat_discard(
        self,
        game: MahjongGame,
        valid_actions: List[str],
        beliefs: Optional[Dict[str, Any]] = None,
        relaxed_gate: bool = False,
    ) -> str:
        """Visible threat signal: prefer middle/non-safe discards.

        This intentionally differs from the safe-disguise policy.  It is used as
        an ablation to test whether a costly, publicly visible deviation can
        move a discard-tell opponent's belief and induce FFR.
        """
        player = game.players[self.player_id]
        public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        current_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
        discard_progress = {}
        for action in valid_actions:
            value = discard_progress_metrics(game, self.player_id, action)
            if value is not None:
                discard_progress[action] = value
        best_result_shanten = min(
            (value["shanten"] for value in discard_progress.values()),
            default=99,
        )
        best_effective_copies = max(
            (
                value["effective_copies"]
                for value in discard_progress.values()
                if value["shanten"] == best_result_shanten
            ),
            default=0,
        )
        ffr_mode = current_shanten > self.mc_danger_threshold and self.forced_deceive != "always"
        if self.threat_require_real_target:
            target_probs: Optional[Dict[int, float]] = None
            if self.threat_target_signal == "mc":
                target_probs = {
                    p.player_id: opponent_view_posterior(
                        game, target_pid=p.player_id, observer_pid=self.player_id,
                        num_samples=self.mc_oracle_samples, rng=self.rng, beta=self.mc_beta,
                        max_shanten=self.threat_target_max_shanten,
                    )["tenpai_prob"]
                    for p in game.players if p.player_id != self.player_id
                }
                has_real_target = any(prob >= self.threat_target_prob_threshold for prob in target_probs.values())
            else:
                has_real_target = any(
                    within_shanten(p.hand_tiles, p.missing_suit, self.threat_target_max_shanten)
                    for p in game.players if p.player_id != self.player_id
                )
            if not has_real_target and not relaxed_gate:
                self._last_deceive_signal = {
                    "signal_model": "discard_tell_gate",
                    "gate_mode": self.threat_gate_mode,
                    "blocked": True,
                    "blocked_reason": "no_real_target",
                    "threat_target_signal": self.threat_target_signal,
                    "threat_target_max_shanten": self.threat_target_max_shanten,
                    **({"threat_target_prob_threshold": self.threat_target_prob_threshold,
                        "threat_target_probs": {f"P{k}": round(v, 3) for k, v in target_probs.items()}}
                       if target_probs is not None else {}),
                }
                return ""
        tell_before = self._discard_tell_threat(game)
        belief_values = {
            key: float(value.get("tenpai_confidence", 0.0))
            for key, value in (beliefs or {}).items()
        }

        def projected_response(tell_value: float) -> Dict[str, float]:
            if self.threat_response_model == "blend" and belief_values:
                weight = self.threat_response_tell_weight
                return {
                    key: weight * tell_value + (1.0 - weight) * confidence
                    for key, confidence in belief_values.items()
                }
            return {"tell": tell_value}

        response_before = projected_response(tell_before)
        ffr_candidates: List[Tuple[float, float, float, str, str, Dict[str, float], int, int]] = []
        relaxed_ffr_candidates: List[Tuple[float, float, float, str, str, Dict[str, float], int, int]] = []
        keep_mid, keep_any, relaxed_mid, relaxed_any, break_mid, break_any = [], [], [], [], [], []
        max_result_shanten = 99 if self.threat_allow_break_ready else self.threat_max_result_shanten
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            for tile in player.hand_tiles:
                if str(tile) != action[2:]:
                    continue
                temp = player.hand_tiles.copy()
                temp.remove(tile)
                action_progress = discard_progress.get(action, {})
                result_shanten = action_progress.get(
                    "shanten", ShantenCalculator.calculate_shanten(temp, player.missing_suit)
                )
                effective_copies = int(action_progress.get("effective_copies", 0))
                keeps_ready = result_shanten <= self.dir_ready_threshold
                within_relaxed_budget = result_shanten <= max_result_shanten
                is_public_safe = (tile.suit, tile.number) in public
                is_mid = tile.number in (4, 5, 6)
                remains_not_dangerous = result_shanten > self.mc_danger_threshold
                within_shanten_regret = (
                    result_shanten <= best_result_shanten + self.threat_max_shanten_regret
                )
                preserves_ukeire = (
                    effective_copies >= best_effective_copies * self.threat_min_ukeire_ratio
                )
                if (
                    remains_not_dangerous
                    and current_shanten <= self.threat_max_start_shanten
                    and within_shanten_regret
                    and preserves_ukeire
                ):
                    tell_after = self._discard_tell_threat(game, extra_tile=tile)
                    tell_delta = tell_after - tell_before
                    response_after = projected_response(tell_after)
                    response_deltas = {
                        key: response_after[key] - response_before[key]
                        for key in response_before
                    }
                    effective_delta = max(response_deltas.values(), default=tell_delta)
                    meaningful_delta = effective_delta >= self.threat_min_delta
                    if self.threat_gate_mode == "delta_only":
                        # No "already near the tipping point" requirement: any
                        # meaningfully-pushing discard counts, so pressure can
                        # build across several turns instead of needing one
                        # single discard to cross the fold threshold outright.
                        gate_ok = meaningful_delta and any(
                            before < self.threat_gate_threshold
                            for before in response_before.values()
                        )
                    else:
                        gate_ok = meaningful_delta and any(
                            self.threat_gate_threshold - self.threat_gate_margin <= response_before[key]
                            < self.threat_gate_threshold <= response_after[key]
                            for key in response_before
                        )
                    relaxed_ok = meaningful_delta or (is_mid and not is_public_safe)
                    if relaxed_gate and relaxed_ok:
                        mid_bonus = 1.0 if is_mid and not is_public_safe else 0.0
                        relaxed_ffr_candidates.append(
                            (
                                mid_bonus,
                                effective_delta,
                                tell_delta,
                                action,
                                str(tile),
                                response_after,
                                result_shanten,
                                effective_copies,
                            )
                        )
                    if gate_ok:
                        mid_bonus = 1.0 if is_mid and not is_public_safe else 0.0
                        ffr_candidates.append(
                            (
                                mid_bonus,
                                effective_delta,
                                tell_delta,
                                action,
                                str(tile),
                                response_after,
                                result_shanten,
                                effective_copies,
                            )
                        )
                if keeps_ready:
                    keep_any.append(action)
                    if is_mid and not is_public_safe:
                        keep_mid.append(action)
                elif within_relaxed_budget:
                    relaxed_any.append(action)
                    if is_mid and not is_public_safe:
                        relaxed_mid.append(action)
                else:
                    break_any.append(action)
                    if is_mid and not is_public_safe:
                        break_mid.append(action)
                break
        if ffr_mode:
            if ffr_candidates:
                ffr_candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
                _, effective_delta, tell_delta, action, tile_text, response_after, chosen_shanten, chosen_copies = ffr_candidates[0]
                self._last_deceive_signal = {
                    "signal_model": "discard_tell_gate",
                    "tell_before": round(tell_before, 4),
                    "tell_after": round(tell_before + tell_delta, 4),
                    "tell_delta": round(tell_delta, 4),
                    "response_before": {key: round(value, 4) for key, value in response_before.items()},
                    "response_after": {key: round(value, 4) for key, value in response_after.items()},
                    "response_delta": round(effective_delta, 4),
                    "response_model": self.threat_response_model,
                    "response_tell_weight": self.threat_response_tell_weight,
                    "gate_threshold": self.threat_gate_threshold,
                    "gate_margin": self.threat_gate_margin,
                    "min_delta": self.threat_min_delta,
                    "gate_mode": self.threat_gate_mode,
                    "max_start_shanten": self.threat_max_start_shanten,
                    "best_result_shanten": best_result_shanten,
                    "chosen_result_shanten": chosen_shanten,
                    "best_effective_copies": best_effective_copies,
                    "chosen_effective_copies": chosen_copies,
                    "min_ukeire_ratio": self.threat_min_ukeire_ratio,
                    "max_shanten_regret": self.threat_max_shanten_regret,
                    "chosen_tile": tile_text,
                }
                return action
            if relaxed_gate and relaxed_ffr_candidates:
                relaxed_ffr_candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
                _, effective_delta, tell_delta, action, tile_text, response_after, chosen_shanten, chosen_copies = relaxed_ffr_candidates[0]
                self._last_deceive_signal = {
                    "signal_model": "discard_tell_candidate",
                    "tell_before": round(tell_before, 4),
                    "tell_after": round(tell_before + tell_delta, 4),
                    "tell_delta": round(tell_delta, 4),
                    "response_before": {key: round(value, 4) for key, value in response_before.items()},
                    "response_after": {key: round(value, 4) for key, value in response_after.items()},
                    "response_delta": round(effective_delta, 4),
                    "response_model": self.threat_response_model,
                    "response_tell_weight": self.threat_response_tell_weight,
                    "gate_threshold": self.threat_gate_threshold,
                    "gate_margin": self.threat_gate_margin,
                    "min_delta": self.threat_min_delta,
                    "gate_mode": self.threat_gate_mode,
                    "relaxed_gate": True,
                    "max_start_shanten": self.threat_max_start_shanten,
                    "best_result_shanten": best_result_shanten,
                    "chosen_result_shanten": chosen_shanten,
                    "best_effective_copies": best_effective_copies,
                    "chosen_effective_copies": chosen_copies,
                    "min_ukeire_ratio": self.threat_min_ukeire_ratio,
                    "max_shanten_regret": self.threat_max_shanten_regret,
                    "chosen_tile": tile_text,
                }
                return action
            self._last_deceive_signal = {
                "signal_model": "discard_tell_gate",
                "tell_before": round(tell_before, 4),
                "response_before": {key: round(value, 4) for key, value in response_before.items()},
                "response_model": self.threat_response_model,
                "response_tell_weight": self.threat_response_tell_weight,
                "gate_mode": self.threat_gate_mode,
                "gate_threshold": self.threat_gate_threshold,
                "gate_margin": self.threat_gate_margin,
                "min_delta": self.threat_min_delta,
                "max_start_shanten": self.threat_max_start_shanten,
                "best_result_shanten": best_result_shanten,
                "best_effective_copies": best_effective_copies,
                "min_ukeire_ratio": self.threat_min_ukeire_ratio,
                "max_shanten_regret": self.threat_max_shanten_regret,
                "blocked": True,
                "blocked_reason": "no_ffr_candidate",
            }
            return ""
        candidate_groups = [keep_mid, keep_any, relaxed_mid, relaxed_any]
        if self.threat_allow_break_ready or self.forced_deceive == "always":
            candidate_groups.extend([break_mid, break_any])
        for candidates in candidate_groups:
            if candidates:
                return candidates[0]
        return self._min_shanten_discard(game, valid_actions)

    def _min_shanten_discard(self, game: MahjongGame, valid_actions: List[str]) -> str:
        return self._heuristic_action(game, valid_actions, "exploit", {})

    def _heuristic_action(self, game: MahjongGame, valid_actions: List[str], mode: str, beliefs: Dict[str, Any]) -> str:
        player = game.players[self.player_id]
        if "h" in valid_actions:
            return "h"
        if mode == "safe":
            safe_discards = []
            public_discards = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
            for action in valid_actions:
                if not action.startswith("d "):
                    continue
                tile_text = action[2:]
                for tile in player.hand_tiles:
                    if str(tile) == tile_text and (tile.suit, tile.number) in public_discards:
                        safe_discards.append(action)
            if safe_discards:
                return safe_discards[0]

        # In deceive mode, prefer discarding a tile that does not break tenpai if possible.
        if mode == "deceive":
            ready, waits = player.is_ready_with_missing_suit()
            if ready:
                for action in valid_actions:
                    if action.startswith("d "):
                        return action

        best_action = valid_actions[0]
        best_shanten = 99
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            tile_text = action[2:]
            for tile in player.hand_tiles:
                if str(tile) != tile_text:
                    continue
                temp = player.hand_tiles.copy()
                temp.remove(tile)
                shanten = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
                if shanten < best_shanten:
                    best_shanten = shanten
                    best_action = action
                break
        return best_action

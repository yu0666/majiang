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
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from belief_oracle import opponent_view_posterior, within_shanten
from game import MahjongGame
from prompt_builder import build_belief_prompt, build_mask_decision_prompt, get_legal_actions
from rule_engine import ShantenCalculator


LLMCallable = Callable[[str], str]


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


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
            return

        keys = set(current) | set(self.last_vector)
        delta = math.sqrt(sum((current.get(k, 0.0) - self.last_vector.get(k, 0.0)) ** 2 for k in keys))
        self.drift_score = delta
        self.drift_flag = delta >= 0.45 and len(self.actions) >= 6
        self.last_vector = current

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
    def compute(self, game: MahjongGame, player_id: int, z_state: Dict[int, Dict[str, Any]], beliefs: Dict[str, Any]) -> Dict[str, Any]:
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
        }


class MASKLLMAgent:
    def __init__(
        self,
        player_id: int = 0,
        llm: Optional[LLMCallable] = None,
        belief_llm: Optional[LLMCallable] = None,
        decision_llm: Optional[LLMCallable] = None,
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
        threat_gate_threshold: float = 0.4,
        threat_gate_margin: float = 0.12,
        threat_min_delta: float = 0.03,
        threat_gate_mode: str = "cross",
        threat_tell_window: int = 6,
        threat_max_start_shanten: int = 3,
        threat_require_non_exploit: bool = True,
        threat_require_real_target: bool = False,
        threat_target_max_shanten: int = 0,
        threat_target_signal: str = "oracle",
        threat_target_prob_threshold: float = 0.5,
        log_counterfactual: bool = False,
        mc_seed: int = 0,
    ):
        self.player_id = player_id
        self.llm = llm
        self.belief_llm = belief_llm if belief_llm is not None else llm
        self.decision_llm = decision_llm if decision_llm is not None else llm
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
        self.threat_gate_threshold = threat_gate_threshold
        self.threat_gate_margin = threat_gate_margin
        self.threat_min_delta = threat_min_delta
        self.threat_gate_mode = threat_gate_mode
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
        self._mc_cache: Dict[int, Dict[str, Any]] = {}
        self.rng = random.Random(mc_seed)
        self.last_decision: Dict[str, Any] = {}
        self._last_deceive_signal: Dict[str, Any] = {}

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

    def decide(self, game: MahjongGame, valid_actions: Optional[List[str]] = None) -> str:
        valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, self.player_id)
        z_state = self.tracker.update_from_game(game)
        beliefs = self._mc_beliefs(game) if self.use_mc_belief else self.belief_estimator.infer_all(game, self.player_id)
        gate = self.risk_gate.compute(game, self.player_id, z_state, beliefs)

        player = game.players[self.player_id]
        own_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
        own_danger = own_shanten <= self.mc_danger_threshold
        dir_ready = own_shanten <= self.dir_ready_threshold
        perceived = max((float(b.get("tenpai_confidence", 0.0)) for b in beliefs.values()), default=0.0)
        visible_threat = sum(1 for p in game.players if p.player_id != self.player_id and len(p.open_melds) >= 2)
        exploit_cache: Optional[Tuple[str, str]] = None

        def get_exploit_action() -> Tuple[str, str]:
            nonlocal exploit_cache
            if exploit_cache is None:
                exploit_cache = self._exploit_action(game, valid_actions, beliefs, gate)
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
        deceive_action = (
            self._deceptive_discard(game, valid_actions)
            if (deceive_ready or self.forced_deceive == "always")
            else None
        )
        can_deceive = deceive_action is not None and deceive_action.startswith("d ")
        counterfactual_exploit_action = None
        disguise_equals_exploit = None
        if can_deceive and self.deceive_style == "threat" and self.threat_require_non_exploit:
            counterfactual_exploit_action, _ = get_exploit_action()
            disguise_equals_exploit = bool(counterfactual_exploit_action == deceive_action)
            if disguise_equals_exploit:
                can_deceive = False
        force_deceive = self.forced_deceive == "always" or (
            self.forced_deceive == "eligible" and deceive_ready and can_deceive
        )
        if "h" in valid_actions:
            action, mode, reason = "h", "exploit", "take win"
        elif force_deceive and can_deceive:
            action = deceive_action
            mode = "deceive"
            reason = f"forced-{self.forced_deceive} deceive-{self.deceive_style} ablation"
        elif visible_threat >= 2 and not own_danger:
            action = self._safe_discard(game, valid_actions)
            mode, reason = "safe", "opponents threatening, no hand -> fold safe"
        elif deceive_ready and perceived <= self.deceive_threat_ceiling and can_deceive:
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
            "reason": reason,
            "counterfactual_exploit_action": counterfactual_exploit_action,
            "disguise_equals_exploit": disguise_equals_exploit,
            "own_shanten": int(own_shanten),
            "own_danger_threshold": self.mc_danger_threshold,
            "dir_ready_threshold": self.dir_ready_threshold,
            "deceive_ready": bool(deceive_ready),
            "ffr_ready": bool(ffr_ready),
            "perceived_threat_of_me": round(perceived, 3),
            "forced_deceive": self.forced_deceive,
            "deceive_style": self.deceive_style,
            "threat_allow_break_ready": self.threat_allow_break_ready,
            "threat_max_result_shanten": self.threat_max_result_shanten,
            "threat_gate_threshold": self.threat_gate_threshold,
            "threat_gate_margin": self.threat_gate_margin,
            "threat_min_delta": self.threat_min_delta,
            "threat_gate_mode": self.threat_gate_mode,
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
            "gate": gate,
        }
        return action

    def _exploit_action(self, game: MahjongGame, valid_actions: List[str],
                        beliefs: Dict[str, Any], gate: Dict[str, Any]) -> Tuple[str, str]:
        if self.decision_llm is not None:
            prompt = build_mask_decision_prompt(
                game, self.player_id, beliefs, gate, valid_actions=valid_actions,
            )
            raw = self.decision_llm(prompt)
            parsed = _safe_json_loads(raw)
            if parsed and "action" in parsed:
                return legalize_action(str(parsed.get("action", "")), valid_actions), "exploit (decision LLM)"
            action = legalize_action(raw, valid_actions)
            return action, "exploit (decision LLM text)"
        return self._min_shanten_discard(game, valid_actions), "exploit (min-shanten)"

    def _safe_discard(self, game: MahjongGame, valid_actions: List[str]) -> str:
        player = game.players[self.player_id]
        public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            for tile in player.hand_tiles:
                if str(tile) == action[2:] and (tile.suit, tile.number) in public:
                    return action
        return self._min_shanten_discard(game, valid_actions)

    def _deceptive_discard(self, game: MahjongGame, valid_actions: List[str]) -> str:
        self._last_deceive_signal = {}
        if self.deceive_style == "threat":
            return self._threat_discard(game, valid_actions)
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

    def _threat_discard(self, game: MahjongGame, valid_actions: List[str]) -> str:
        """Visible threat signal: prefer middle/non-safe discards.

        This intentionally differs from the safe-disguise policy.  It is used as
        an ablation to test whether a costly, publicly visible deviation can
        move a discard-tell opponent's belief and induce FFR.
        """
        player = game.players[self.player_id]
        public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        current_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
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
            if not has_real_target:
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
        ffr_candidates: List[Tuple[float, float, str, str]] = []
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
                result_shanten = ShantenCalculator.calculate_shanten(temp, player.missing_suit)
                keeps_ready = result_shanten <= self.dir_ready_threshold
                within_relaxed_budget = result_shanten <= max_result_shanten
                is_public_safe = (tile.suit, tile.number) in public
                is_mid = tile.number in (4, 5, 6)
                remains_not_dangerous = result_shanten > self.mc_danger_threshold
                if remains_not_dangerous and current_shanten <= self.threat_max_start_shanten:
                    tell_after = self._discard_tell_threat(game, extra_tile=tile)
                    tell_delta = tell_after - tell_before
                    meaningful_delta = tell_delta >= self.threat_min_delta
                    if self.threat_gate_mode == "delta_only":
                        # No "already near the tipping point" requirement: any
                        # meaningfully-pushing discard counts, so pressure can
                        # build across several turns instead of needing one
                        # single discard to cross the fold threshold outright.
                        gate_ok = meaningful_delta
                    else:
                        near_gate = self.threat_gate_threshold - self.threat_gate_margin <= tell_before < self.threat_gate_threshold
                        crosses_gate = tell_after >= self.threat_gate_threshold
                        gate_ok = near_gate and crosses_gate and meaningful_delta
                    if gate_ok:
                        mid_bonus = 1.0 if is_mid and not is_public_safe else 0.0
                        ffr_candidates.append((mid_bonus, tell_delta, action, str(tile)))
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
                _, tell_delta, action, tile_text = ffr_candidates[0]
                self._last_deceive_signal = {
                    "signal_model": "discard_tell_gate",
                    "tell_before": round(tell_before, 4),
                    "tell_after": round(tell_before + tell_delta, 4),
                    "tell_delta": round(tell_delta, 4),
                    "gate_threshold": self.threat_gate_threshold,
                    "gate_margin": self.threat_gate_margin,
                    "min_delta": self.threat_min_delta,
                    "gate_mode": self.threat_gate_mode,
                    "max_start_shanten": self.threat_max_start_shanten,
                    "chosen_tile": tile_text,
                }
                return action
            self._last_deceive_signal = {
                "signal_model": "discard_tell_gate",
                "tell_before": round(tell_before, 4),
                "gate_mode": self.threat_gate_mode,
                "gate_threshold": self.threat_gate_threshold,
                "gate_margin": self.threat_gate_margin,
                "min_delta": self.threat_min_delta,
                "max_start_shanten": self.threat_max_start_shanten,
                "blocked": True,
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

"""
Gate1 minimal experiments for LLM-native MASK.

This is the first executable bridge from the W0 skeleton to W1/W2 evidence:

E1: coarse B_phi validity
    Does the belief estimator's tenpai confidence track whether P0 is actually
    ready?  This is only a coarse oracle, not the final opponent-posterior
    oracle described in the paper plan.

E2: L0/L1/L2 ladder
    L0 = llm_base       : public table + legal action heuristic.
    L1 = llm_reactive_z : L0 plus public opponent drift z_j(t).
    L2 = llm_mask       : z_j(t) + B_phi + risk gate.

Current default backend is heuristic fallback so the pipeline can run without a
local LLM.  Reports carry that backend label explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from belief_oracle import opponent_view_posterior, within_shanten
from experiment_trace import (
    ensure_deterministic_hashing,
    paired_method_comparison,
    summarize_belief_samples,
    summarize_by_method,
    write_json,
    write_jsonl,
)
from game import (
    STYLE_BOTS,
    MahjongGame,
    bot_decide_exchange,
    bot_decide_missing_suit,
    bot_decide_response,
    bot_decide_turn_action,
    parse_console_tile,
)
from llm_backends import build_llm_callable
from mask_llm import LLMBeliefEstimator, MASKLLMAgent, PublicOpponentTracker, RiskGate, legalize_action
from prompt_builder import build_state_prompt, get_legal_actions
from rule_engine import ShantenCalculator


RESULTS_DIR = Path("Gate1_results")
METHODS = ("llm_base", "llm_reactive_z", "llm_mask")
OPPONENT_STYLES = tuple(sorted(STYLE_BOTS.keys())) + ("mixed", "responsive")
LLMCallable = Callable[[str], str]


def init_game(
    seed: int, opponent_style: str, game_id: str,
    defender_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[MahjongGame, Dict[int, Tuple], Dict[int, ResponsiveDefender]]:
    random.seed(seed)
    game = MahjongGame(game_id, ["Agent", "B1", "B2", "B3"], bots=[False, True, True, True])
    defenders: Dict[int, ResponsiveDefender] = {}
    if opponent_style == "responsive":
        cfg = defender_cfg or {}
        for pid in (1, 2, 3):
            defenders[pid] = ResponsiveDefender(
                observer_pid=pid,
                threat_threshold=cfg.get("threat_threshold", 0.4),
                oracle_samples=cfg.get("oracle_samples", 30),
                beta=cfg.get("beta", 2.0),
                danger_threshold=cfg.get("danger_threshold", 1),
                ffr_hand_shanten=cfg.get("ffr_hand_shanten", 1),
                rng=random.Random(seed * 100 + pid),
                threat_model=cfg.get("threat_model", "mc"),
                tell_weight=cfg.get("tell_weight", 1.0),
                tell_window=cfg.get("tell_window", 6),
            )
        opponent_funcs = {pid: (defenders[pid].turn, None) for pid in (1, 2, 3)}
    elif opponent_style == "mixed":
        styles = ["aggressive", "conservative", "random"]
        opponent_funcs = {pid: STYLE_BOTS[styles[pid - 1]] for pid in (1, 2, 3)}
    else:
        opponent_funcs = {pid: STYLE_BOTS.get(opponent_style, STYLE_BOTS["greedy"]) for pid in (1, 2, 3)}
    game.start_game()

    for player in game.players:
        game.select_exchange_tiles(player.player_id, bot_decide_exchange(player))

    for player in game.players:
        game.set_missing_suit(player.player_id, bot_decide_missing_suit(player))

    return game, opponent_funcs, defenders


def true_tenpai_label(game: MahjongGame, player_id: int = 0) -> Tuple[bool, List[str], int]:
    player = game.players[player_id]
    ready, waits = player.is_ready_with_missing_suit()
    shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
    return bool(ready), [str(t) for t in waits], int(shanten)


def choose_min_shanten_action(game: MahjongGame, player_id: int, valid_actions: Optional[List[str]] = None) -> str:
    valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, player_id)
    player = game.players[player_id]

    if "h" in valid_actions:
        return "h"
    if "g" in valid_actions:
        return "g"

    best_action = valid_actions[0] if valid_actions else "n"
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


def defensive_discard_action(game: MahjongGame, pid: int, valid_actions: List[str]) -> str:
    """Safest discard: a tile already publicly discarded (won't deal in), else a
    terminal (1/9), else the first legal discard.  Used when folding under threat."""
    player = game.players[pid]
    discards = [a for a in valid_actions if a.startswith("d ")]
    if not discards:
        return valid_actions[0] if valid_actions else "n"
    public = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
    for action in discards:
        tile_text = action[2:]
        for tile in player.hand_tiles:
            if str(tile) == tile_text and (tile.suit, tile.number) in public:
                return action
    for action in discards:
        try:
            num = int(action[2])
        except (ValueError, IndexError):
            num = 5
        if num in (1, 9):
            return action
    return discards[0]


def discard_tell_threat(game: MahjongGame, target_pid: int = 0, window: int = 6) -> float:
    """Deceivable, deterministic threat from P0's RECENT discard pattern.

    Unlike the pool-based MC posterior (which P0 can barely move), this reads the
    classic tell and IS manipulable by P0's discard choice:
      * middle tiles (4-6) discarded  -> looks like pushing  -> threat UP;
      * terminals (1/9) or already-public-safe tiles -> looks like folding -> threat DOWN.
    A structural base (melds + tile depletion) anchors it. No sampling -> the
    before/after threat-delta from this model is noise-free.
    """
    p0 = game.players[target_pid]
    recent = list(p0.discarded_tiles)[-window:]
    others = {(t.suit, t.number) for pl in game.players if pl.player_id != target_pid for t in pl.discarded_tiles}
    push = fold = 0.0
    for t in recent:
        if t.number in (4, 5, 6):
            push += 1.0
        elif t.number in (1, 9):
            fold += 1.0
        elif (t.suit, t.number) in others:
            fold += 0.5
        # 2,3,7,8 are neutral
    tell = (push - fold) / max(1, len(recent))           # roughly [-1.5, 1]
    melds = len(p0.open_melds)
    tiles_left = game.deck.remaining_count()
    base = 0.30 + 0.06 * melds + 0.10 * max(0.0, min(1.0, (40 - tiles_left) / 40.0))
    return max(0.0, min(1.0, base + 0.35 * tell))


class ResponsiveDefender:
    """Experiment-only, rule-based, deceivable opponent.

    Reads only PUBLIC info to estimate how dangerous P0 looks (cached MC
    posterior), then:
      * turn policy: threat low  -> greedy push (may deal into P0 -> feeds DIR);
                     threat high -> defensive safe discard (folds a good hand
                     when it wrongly believes P0 is dangerous -> feeds FFR).
      * response policy: threat high -> decline peng/gang (commitment); always
                     take a hu (declining a win is not realistic).

    FFR is instrumented at the TURN phase with an oracle for "had a real hand
    and P0 was not actually dangerous", so a fold only counts as *false* when it
    was needless.  The bot's own hand is its own information (legitimate); the P0
    oracle peek is for measurement only, never for the decision.
    """

    def __init__(self, observer_pid: int, threat_threshold: float, oracle_samples: int,
                 beta: float, danger_threshold: int, ffr_hand_shanten: int, rng: random.Random,
                 threat_model: str = "mc", tell_weight: float = 1.0, tell_window: int = 6):
        self.pid = observer_pid
        self.threshold = threat_threshold
        self.oracle_samples = oracle_samples
        self.beta = beta
        self.danger_threshold = danger_threshold
        self.ffr_hand_shanten = ffr_hand_shanten
        self.rng = rng
        self.threat_model = threat_model      # "mc" | "discard_tell" | "blend"
        self.tell_weight = tell_weight        # blend: tell_weight*tell + (1-tell_weight)*mc
        self.tell_window = tell_window
        self._cache: Dict[int, float] = {}
        self.ff_opportunities = 0   # had a real hand AND P0 not actually dangerous
        self.ff_false = 0           # ... and folded anyway (induced false fold)
        self.turn_folds = 0
        self.response_declines = 0
        self.ffr_events: List[Dict[str, Any]] = []
        self.false_fold_events: List[Dict[str, Any]] = []

    def _mc_threat(self, game: MahjongGame, num_samples: int, rng: random.Random) -> float:
        post = opponent_view_posterior(
            game, target_pid=0, observer_pid=self.pid,
            num_samples=num_samples, rng=rng, beta=self.beta, max_shanten=self.danger_threshold,
        )
        return float(post["tenpai_prob"])

    def _combine(self, game: MahjongGame, num_samples: int, rng: random.Random) -> float:
        if self.threat_model == "discard_tell":
            return discard_tell_threat(game, 0, self.tell_window)
        if self.threat_model == "blend":
            tell = discard_tell_threat(game, 0, self.tell_window)
            mc = self._mc_threat(game, num_samples, rng)
            return self.tell_weight * tell + (1.0 - self.tell_weight) * mc
        return self._mc_threat(game, num_samples, rng)  # "mc"

    def threat(self, game: MahjongGame) -> float:
        key = len(game.history)  # public state advances with history; cache per state
        if key not in self._cache:
            self._cache[key] = self._combine(game, self.oracle_samples, self.rng)
        return self._cache[key]

    def threat_crn(self, game: MahjongGame, seeds: List[int], num_samples: int) -> float:
        """Measurement-only threat with common random numbers (paired sampling).

        Pass the SAME seeds for the before- and after-action snapshot so MC
        sampling noise largely cancels and the delta reflects P0's action, not
        sampling jitter.  The discard_tell model is deterministic, so its delta is
        exact.  Does not affect the bot's policy (which still uses threat())."""
        vals = [self._combine(game, num_samples, random.Random(s)) for s in seeds]
        return sum(vals) / len(vals) if vals else 0.0

    def turn(
        self,
        player,
        game: MahjongGame,
        step: int = -1,
        last_p0_state: Optional[Dict[str, Any]] = None,
        in_deceive_window: bool = False,
    ) -> str:
        valid = get_legal_actions(game, self.pid)
        if "h" in valid:
            return "h"  # take a win when offered
        threat = self.threat(game)
        push = choose_min_shanten_action(game, self.pid, valid)
        high = threat >= self.threshold

        own_shanten = ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit)
        p0 = game.players[0]
        p0_dangerous = within_shanten(p0.hand_tiles, p0.missing_suit, self.danger_threshold)  # oracle (measurement only)
        # A "false-fold opportunity": this bot has a real hand worth pushing AND
        # P0 is not actually dangerous, so folding here would be needless.
        opportunity = (own_shanten <= self.ffr_hand_shanten) and (not p0_dangerous)

        if high:
            action = defensive_discard_action(game, self.pid, valid)
            folded = action != push
        else:
            action = push
            folded = False

        if folded:
            self.turn_folds += 1
        if opportunity:
            self.ff_opportunities += 1
            event = {
                "step": step,
                "defender_pid": self.pid,
                "threat_at_fold": round(threat, 4),
                "threat_threshold": self.threshold,
                "own_shanten": int(own_shanten),
                "ffr_hand_shanten": self.ffr_hand_shanten,
                "p0_dangerous": bool(p0_dangerous),
                "danger_threshold": self.danger_threshold,
                "push_action": push,
                "folded_action": action if folded else None,
                "chosen_action": action,
                "false_fold": bool(folded),
                "last_p0_step": (last_p0_state or {}).get("step"),
                "last_p0_mode": (last_p0_state or {}).get("mode"),
                "last_p0_action": (last_p0_state or {}).get("action"),
                "last_p0_reason": (last_p0_state or {}).get("reason"),
                "last_p0_threat_before": (last_p0_state or {}).get("threat_before"),
                "last_p0_threat_after": (last_p0_state or {}).get("threat_after"),
                "last_p0_threat_delta": (last_p0_state or {}).get("threat_delta"),
                "in_deceive_window": bool(in_deceive_window),
            }
            self.ffr_events.append(event)
            if folded:
                self.ff_false += 1
                self.false_fold_events.append(event)
        return action

    def response(self, player, acts, game: MahjongGame) -> str:
        if "hu" in acts:
            return "h"
        if self.threat(game) >= self.threshold:
            self.response_declines += 1
            return "n"  # decline peng/gang under perceived threat (commitment risk)
        if "gang" in acts:
            return "g"
        if "peng" in acts:
            return "p"
        return "n"


def choose_llm_base_action(
    game: MahjongGame,
    player_id: int,
    valid_actions: List[str],
    llm: Optional[LLMCallable],
) -> Tuple[str, Dict[str, Any]]:
    if llm is None:
        return choose_min_shanten_action(game, player_id, valid_actions), {
            "mode": "exploit",
            "reason": "L0 base min shanten fallback",
        }

    prompt = build_state_prompt(
        game,
        player_id,
        valid_actions=valid_actions,
        objective="只根据当前手牌和公开牌桌选择一个合法动作，不使用对手信念塑形",
    )
    raw = llm(prompt)
    return legalize_action(raw, valid_actions), {
        "mode": "exploit",
        "reason": "L0 local LLM base decision",
        "llm_raw": raw,
    }


def choose_reactive_z_action(
    game: MahjongGame,
    player_id: int,
    tracker: PublicOpponentTracker,
    risk_gate: RiskGate,
    llm: Optional[LLMCallable],
    valid_actions: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, player_id)
    z_state = tracker.update_from_game(game)
    gate = risk_gate.compute(game, player_id, z_state, beliefs={})

    if "h" in valid_actions:
        return "h", {"z_state": z_state, "gate": gate, "mode": "exploit", "reason": "hu legal"}

    if llm is not None:
        base_prompt = build_state_prompt(
            game,
            player_id,
            valid_actions=valid_actions,
            objective="结合公开对手漂移 z_j(t) 选择一个合法动作，但不要做主动信念塑形",
        )
        prompt = f"""
{base_prompt}

【公开对手漂移 z_j(t)】
{json.dumps(z_state, ensure_ascii=False)}

请只输出一个合法动作。
""".strip()
        raw = llm(prompt)
        return legalize_action(raw, valid_actions), {
            "z_state": z_state,
            "gate": gate,
            "mode": gate["mode_hint"] if gate["mode_hint"] != "deceive" else "exploit",
            "reason": "L1 local LLM reactive-z decision",
            "llm_raw": raw,
        }

    if gate["mode_hint"] == "safe":
        public_discards = {(t.suit, t.number) for p in game.players for t in p.discarded_tiles}
        player = game.players[player_id]
        for action in valid_actions:
            if not action.startswith("d "):
                continue
            tile_text = action[2:]
            for tile in player.hand_tiles:
                if str(tile) == tile_text and (tile.suit, tile.number) in public_discards:
                    return action, {
                        "z_state": z_state,
                        "gate": gate,
                        "mode": "safe",
                        "reason": "reactive-z safe public discard",
                    }

    action = choose_min_shanten_action(game, player_id, valid_actions)
    return action, {"z_state": z_state, "gate": gate, "mode": "exploit", "reason": "reactive-z min shanten"}


def execute_action(game: MahjongGame, pid: int, action: str, drawn_tile=None) -> Optional[object]:
    player = game.players[pid]
    action = (action or "").strip()
    discarded_tile = None

    if action == "h" and player.can_hu():
        win_tile = drawn_tile if drawn_tile else (player.hand_tiles[-1] if player.hand_tiles else None)
        if win_tile is not None:
            game.hu(pid, win_tile, True)
            game.check_game_over()
        return None

    if action == "g":
        gang_info = game.can_self_gang(pid)
        if gang_info.get("can_gang"):
            game.gang(pid, gang_info["gang_tiles"][0])
            return None

    if action.startswith("d "):
        tile = parse_console_tile(action[2:])
        if tile and game.discard_tile(pid, tile):
            discarded_tile = tile

    if discarded_tile is None:
        legal_discards = [a for a in get_legal_actions(game, pid) if a.startswith("d ")]
        if legal_discards:
            tile = parse_console_tile(legal_discards[0][2:])
            if tile and game.discard_tile(pid, tile):
                discarded_tile = tile

    return discarded_tile


def resolve_responses(
    game: MahjongGame, discarded_tile, acting_pid: int, agent: Optional[MASKLLMAgent],
    defenders: Optional[Dict[int, "ResponsiveDefender"]] = None,
) -> Dict[str, Any]:
    defenders = defenders or {}
    responses = game.check_responses(discarded_tile, acting_pid)
    response_trace: List[Dict[str, Any]] = []
    responded = False
    agent_won_by_discard = False
    agent_dealt_in = False
    response_false_fold_opportunities = 0
    response_false_folds = 0
    last_p0_state: Dict[str, Any] = {}

    for rid, acts in responses.items():
        if responded:
            break

        player = game.players[rid]
        if rid == 0 and agent is not None:
            valid = get_legal_actions(game, rid, response_actions=acts)
            response_action = "h" if "h" in valid else agent.decide(game, valid)
        elif rid in defenders:
            response_action = defenders[rid].response(player, acts, game)
        else:
            response_action = bot_decide_response(player, acts)

        # Legacy response-phase "decline to hu" counter (near-zero: declining a
        # win is not realistic). The meaningful FFR is the turn-phase one below.
        if "hu" in acts and rid != 0:
            response_false_fold_opportunities += 1
            if response_action != "h":
                response_false_folds += 1

        response_trace.append(
            {
                "responder": rid,
                "available": list(acts),
                "chosen": response_action,
            }
        )

        if response_action == "h" and "hu" in acts:
            game.hu(rid, discarded_tile, False, acting_pid)
            game.check_game_over()
            responded = True
            agent_won_by_discard = rid == 0
            agent_dealt_in = acting_pid == 0 and rid != 0
        elif response_action == "g" and "gang" in acts:
            game.gang(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True
        elif response_action == "p" and "peng" in acts:
            game.peng(rid, discarded_tile, acting_pid)
            game.current_player_id = rid
            responded = True

    return {
        "responded": responded,
        "agent_won_by_discard": agent_won_by_discard,
        "agent_dealt_in": agent_dealt_in,
        "response_false_fold_opportunities": response_false_fold_opportunities,
        "response_false_folds": response_false_folds,
        "responses": response_trace,
    }


def make_step_trace(
    game: MahjongGame,
    method: str,
    seed: int,
    step: int,
    pid: int,
    action: str,
    legal_actions: List[str],
    decision_ms: float,
    decision_state: Dict[str, Any],
    backend: str,
) -> Dict[str, Any]:
    player = game.players[0]
    true_ready, true_waits, true_shanten = true_tenpai_label(game, 0)
    return {
        "game_id": game.game_id,
        "seed": seed,
        "method": method,
        "backend": backend,
        "step": step,
        "actor": pid,
        "action": action,
        "legal_actions": legal_actions,
        "mode": decision_state.get("mode", "none"),
        "z_state": decision_state.get("z_state", {}),
        "belief_json": decision_state.get("beliefs", {}),
        "risk_budget": decision_state.get("gate", {}).get("risk_budget"),
        "gate": decision_state.get("gate", {}),
        "score_delta": player.balance - 10000,
        "true_tenpai": true_ready,
        "true_waits": true_waits,
        "true_shanten": true_shanten,
        "dealin": False,
        "induced_dealin": False,
        "false_fold": False,
        "latency_ms": decision_ms,
        "reason": decision_state.get("reason", ""),
        "own_shanten": decision_state.get("own_shanten"),
        "own_danger_threshold": decision_state.get("own_danger_threshold"),
        "dir_ready_threshold": decision_state.get("dir_ready_threshold"),
        "deceive_ready": decision_state.get("deceive_ready"),
        "ffr_ready": decision_state.get("ffr_ready"),
        "deceive_signal": decision_state.get("deceive_signal", {}),
        "counterfactual_exploit_action": decision_state.get("counterfactual_exploit_action"),
        "disguise_equals_exploit": decision_state.get("disguise_equals_exploit"),
    }


def collect_belief_samples(game: MahjongGame, method: str, seed: int, step: int, estimator: LLMBeliefEstimator, backend: str) -> List[Dict[str, Any]]:
    true_ready, true_waits, true_shanten = true_tenpai_label(game, 0)
    rows = []
    for target_pid in (1, 2, 3):
        belief = estimator.infer(game, 0, target_pid)
        rows.append(
            {
                "game_id": game.game_id,
                "seed": seed,
                "method": method,
                "backend": backend,
                "step": step,
                "target_opponent": f"P{target_pid}",
                "think_i_am_tenpai": belief.get("think_i_am_tenpai"),
                "tenpai_confidence": float(belief.get("tenpai_confidence", 0.5)),
                "suspected_waits": belief.get("suspected_waits", []),
                "danger_tiles_for_me": belief.get("danger_tiles_for_me", []),
                "true_tenpai": true_ready,
                "true_waits": true_waits,
                "true_shanten": true_shanten,
                "label_scope": "coarse true P0 tenpai",
            }
        )
    return rows


def summarize_false_fold_attribution(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {
            "ffr_opportunity_events": 0,
            "note": "No turn-phase FFR opportunity events were recorded.",
        }

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        by_mode.setdefault(str(event.get("last_p0_mode") or "none"), []).append(event)

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: Dict[str, Any] = {
        "ffr_opportunity_events": len(events),
        "false_fold_events": sum(1 for event in events if event.get("false_fold")),
        "ffr": sum(1 for event in events if event.get("false_fold")) / len(events),
        "opportunities_in_deceive_window": sum(1 for event in events if event.get("in_deceive_window")),
        "false_folds_in_deceive_window": sum(
            1 for event in events if event.get("in_deceive_window") and event.get("false_fold")
        ),
        "by_last_p0_mode": {},
    }
    for mode, rows in sorted(by_mode.items()):
        deltas = []
        threat_at_fold = []
        for row in rows:
            threat_at_fold.append(float(row.get("threat_at_fold", 0.0)))
            delta_map = row.get("last_p0_threat_delta") or {}
            if isinstance(delta_map, dict):
                key = f"P{row.get('defender_pid')}"
                if key in delta_map:
                    try:
                        deltas.append(float(delta_map[key]))
                    except (TypeError, ValueError):
                        pass
        summary["by_last_p0_mode"][mode] = {
            "opportunities": len(rows),
            "false_folds": sum(1 for row in rows if row.get("false_fold")),
            "ffr": sum(1 for row in rows if row.get("false_fold")) / len(rows),
            "avg_threat_at_fold": round(avg(threat_at_fold), 4),
            "avg_last_p0_threat_delta_for_defender": round(avg(deltas), 4),
            "positive_delta_rate": (
                sum(1 for value in deltas if value > 0.0) / len(deltas)
                if deltas else None
            ),
        }
    return summary


def play_one_game(
    method: str,
    seed: int,
    opponent_style: str,
    max_steps: int,
    sample_every: int,
    backend: str,
    llm: Optional[LLMCallable],
    belief_llm: Optional[LLMCallable],
    defender_cfg: Optional[Dict[str, Any]] = None,
    mask_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    game, opponent_funcs, defenders = init_game(seed, opponent_style, f"Gate1_{method}_{seed}", defender_cfg)
    start_balances = [p.balance for p in game.players]
    mask_cfg = mask_cfg or {}
    snapshot_samples = int(mask_cfg.get("snapshot_samples", 120))
    snapshot_crn_seeds = max(1, int(mask_cfg.get("snapshot_crn_seeds", 1)))
    mask_agent = (
        MASKLLMAgent(
            player_id=0,
            decision_llm=llm,
            belief_llm=belief_llm,
            mc_seed=seed * 13 + 1,
            mc_oracle_samples=mask_cfg.get("oracle_samples", 30),
            mc_beta=mask_cfg.get("beta", 2.0),
            mc_danger_threshold=mask_cfg.get("danger_threshold", 1),
            dir_ready_threshold=mask_cfg.get("dir_ready_threshold", 0),
            deceive_threat_ceiling=mask_cfg.get("deceive_threat_ceiling", 0.5),
            forced_deceive=mask_cfg.get("forced_deceive", "off"),
            deceive_style=mask_cfg.get("deceive_style", "safe"),
            threat_allow_break_ready=mask_cfg.get("threat_allow_break_ready", False),
            threat_max_result_shanten=mask_cfg.get("threat_max_result_shanten", 0),
            threat_gate_threshold=mask_cfg.get("threat_gate_threshold", 0.4),
            threat_gate_margin=mask_cfg.get("threat_gate_margin", 0.12),
            threat_min_delta=mask_cfg.get("threat_min_delta", 0.03),
            threat_gate_mode=mask_cfg.get("threat_gate_mode", "cross"),
            threat_tell_window=mask_cfg.get("threat_tell_window", 6),
            threat_max_start_shanten=mask_cfg.get("threat_max_start_shanten", 3),
            threat_require_non_exploit=mask_cfg.get("threat_require_non_exploit", True),
            threat_require_real_target=mask_cfg.get("threat_require_real_target", False),
            threat_target_max_shanten=mask_cfg.get("threat_target_max_shanten", 0),
            log_counterfactual=mask_cfg.get("log_counterfactual", False),
        )
        if method == "llm_mask" else None
    )
    z_tracker = PublicOpponentTracker([1, 2, 3])
    risk_gate = RiskGate()
    belief_estimator = LLMBeliefEstimator(llm=belief_llm)

    skip_draw = True
    steps = 0
    decision_times: List[float] = []
    mode_counts: Counter[str] = Counter()
    step_rows: List[Dict[str, Any]] = []
    belief_rows: List[Dict[str, Any]] = []
    deceive_active_until = -1
    deceive_windows = 0
    induced_dealin = 0
    response_false_fold_opportunities = 0
    response_false_folds = 0
    last_p0_state: Dict[str, Any] = {}

    while not game.is_game_over and steps < max_steps:
        steps += 1
        if game.deck.remaining_count() == 0 or sum(1 for p in game.players if p.is_hu) >= 3:
            game.check_game_over()
            break

        pid = game.current_player_id
        player = game.players[pid]
        if player.is_hu:
            game.next_player()
            skip_draw = False
            continue

        drawn_tile = None
        if not skip_draw:
            drawn_tile = game.draw_tile(pid)
            if not drawn_tile:
                game.check_game_over()
                break
        else:
            skip_draw = False

        decision_state: Dict[str, Any] = {}
        decision_ms = 0.0
        legal_actions: List[str] = []

        if pid == 0:
            legal_actions = get_legal_actions(game, 0)
            if sample_every > 0 and (steps == 1 or steps % sample_every == 0):
                belief_rows.extend(collect_belief_samples(game, method, seed, steps, belief_estimator, backend))

            # Common-random-numbers snapshot for causal threat-delta (paired
            # before/after, fixed seeds, high samples) -- measurement only.
            snap_seeds = [seed * 100003 + steps * 31 + k for k in range(snapshot_crn_seeds)]
            threat_before = {
                f"P{rid}": defenders[rid].threat_crn(game, snap_seeds, snapshot_samples) for rid in defenders
            }
            t0 = time.perf_counter()
            if method == "llm_base":
                action, decision_state = choose_llm_base_action(game, 0, legal_actions, llm)
            elif method == "llm_reactive_z":
                action, decision_state = choose_reactive_z_action(game, 0, z_tracker, risk_gate, llm, legal_actions)
            elif method == "llm_mask":
                assert mask_agent is not None
                action = mask_agent.decide(game, legal_actions)
                decision_state = dict(mask_agent.last_decision)
            else:
                raise ValueError(f"Unknown method: {method}")
            decision_ms = (time.perf_counter() - t0) * 1000.0
            decision_times.append(decision_ms)

            mode = decision_state.get("mode", "exploit")
            mode_counts[mode] += 1
            if mode == "deceive":
                deceive_active_until = steps + 4
                deceive_windows += 1

            step_rows.append(
                make_step_trace(game, method, seed, steps, pid, action, legal_actions, decision_ms, decision_state, backend)
            )
        else:
            if pid in defenders:
                action = defenders[pid].turn(
                    player,
                    game,
                    step=steps,
                    last_p0_state=last_p0_state,
                    in_deceive_window=deceive_active_until >= steps,
                )
            else:
                action = opponent_funcs[pid][0](player, game)

        discarded_tile = execute_action(game, pid, action, drawn_tile)
        if game.is_game_over:
            break

        if pid == 0:
            # SAME snap_seeds as the before-snapshot (paired CRN): only the
            # post-action pool change drives the delta, not sampling noise.
            threat_after = {
                f"P{rid}": defenders[rid].threat_crn(game, snap_seeds, snapshot_samples) for rid in defenders
            }
            threat_delta = {
                key: round(threat_after.get(key, 0.0) - threat_before.get(key, 0.0), 4)
                for key in set(threat_before) | set(threat_after)
            }
            last_p0_state = {
                "step": steps,
                "mode": decision_state.get("mode", "none"),
                "action": action,
                "reason": decision_state.get("reason", ""),
                "threat_before": {k: round(v, 4) for k, v in threat_before.items()},
                "threat_after": {k: round(v, 4) for k, v in threat_after.items()},
                "threat_delta": threat_delta,
            }
            if step_rows and step_rows[-1]["actor"] == 0:
                step_rows[-1]["threat_before"] = last_p0_state["threat_before"]
                step_rows[-1]["threat_after"] = last_p0_state["threat_after"]
                step_rows[-1]["threat_delta"] = last_p0_state["threat_delta"]

        if discarded_tile is None:
            if action == "g":
                skip_draw = True
                continue
            game.next_player()
            skip_draw = False
            continue

        response_info = resolve_responses(game, discarded_tile, pid, mask_agent, defenders)
        response_false_fold_opportunities += response_info["response_false_fold_opportunities"]
        response_false_folds += response_info["response_false_folds"]

        if response_info["agent_dealt_in"]:
            for row in reversed(step_rows):
                if row["actor"] == 0:
                    row["dealin"] = True
                    break

        if response_info["agent_won_by_discard"] and deceive_active_until >= steps:
            induced_dealin += 1
            for row in reversed(step_rows):
                if row.get("mode") == "deceive":
                    row["induced_dealin"] = True
                    break

        if game.is_game_over:
            break
        if response_info["responded"]:
            skip_draw = True
        else:
            game.next_player()
            skip_draw = False

    if not game.is_game_over:
        game.check_game_over()

    end_balances = [p.balance for p in game.players]
    net = [end_balances[i] - start_balances[i] for i in range(4)]
    agent_player = game.players[0]
    agent_dealin = any(p.is_hu and not p.hu_is_self_drawn and p.hu_discard_player_id == 0 for p in game.players)

    # Meaningful FFR (turn phase, oracle-grounded): opponents folded a real hand
    # while P0 was not actually dangerous -> a fold induced by (false) perceived threat.
    turn_ff_opportunities = sum(d.ff_opportunities for d in defenders.values())
    turn_ff_false = sum(d.ff_false for d in defenders.values())
    response_declines = sum(d.response_declines for d in defenders.values())
    ffr_events = [event for defender in defenders.values() for event in defender.ffr_events]
    false_fold_events = [event for defender in defenders.values() for event in defender.false_fold_events]
    for event in ffr_events:
        event["game_id"] = game.game_id
        event["seed"] = seed
        event["method"] = method
        event["backend"] = backend
    false_folds_after_deceive = sum(1 for event in false_fold_events if event.get("last_p0_mode") == "deceive")
    false_folds_after_exploit = sum(1 for event in false_fold_events if event.get("last_p0_mode") == "exploit")
    false_folds_in_deceive_window = sum(1 for event in false_fold_events if event.get("in_deceive_window"))

    # Tag steps where an opponent false-folded its hand this game (for trace).
    if turn_ff_false > 0:
        for row in step_rows:
            if row["actor"] == 0:
                row["false_fold_round"] = True

    game_row = {
        "seed": seed,
        "method": method,
        "backend": backend,
        "opponent_style": opponent_style,
        "steps": steps,
        "agent_hu": bool(agent_player.is_hu),
        "agent_net": net[0],
        "agent_dealin": bool(agent_dealin),
        "winners": list(game.winners),
        "decision_ms": decision_times,
        "mode_counts": dict(mode_counts),
        "deceive_windows": deceive_windows,
        "induced_dealin": induced_dealin,
        # primary FFR = turn-phase oracle-grounded false fold (drives summarize_games FFR)
        "false_fold_opportunities": turn_ff_opportunities,
        "false_folds": turn_ff_false,
        "false_folds_after_deceive": false_folds_after_deceive,
        "false_folds_after_exploit": false_folds_after_exploit,
        "false_folds_in_deceive_window": false_folds_in_deceive_window,
        "ffr_events": ffr_events,
        "false_fold_events": false_fold_events,
        "opponent_response_declines": response_declines,
        # legacy response-phase "decline to hu" (near-zero), kept for reference
        "response_false_fold_opportunities": response_false_fold_opportunities,
        "response_false_folds": response_false_folds,
        "history_len": len(game.history),
    }
    return game_row, step_rows, belief_rows


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo_dir = Path(__file__).resolve().parent
    llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    belief_llm = build_llm_callable(
        backend=args.backend,
        repo_dir=repo_dir,
        model_path=args.model_path,
        adapter_path=args.belief_adapter_path if args.belief_adapter_path else args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )

    defender_cfg = {
        "threat_threshold": args.threat_fold_threshold,
        "oracle_samples": args.oracle_samples,
        "beta": args.oracle_beta,
        "danger_threshold": args.danger_threshold,
        "ffr_hand_shanten": args.ffr_hand_shanten,
        "threat_model": args.defender_threat_model,
        "tell_weight": args.defender_tell_weight,
        "tell_window": args.defender_tell_window,
    }
    mask_cfg = {
        "oracle_samples": args.mask_oracle_samples,
        "beta": args.mask_oracle_beta,
        "danger_threshold": args.mask_danger_threshold,
        "dir_ready_threshold": args.mask_dir_ready_threshold,
        "deceive_threat_ceiling": args.mask_deceive_threat_ceiling,
        "forced_deceive": args.mask_forced_deceive,
        "deceive_style": args.mask_deceive_style,
        "threat_allow_break_ready": args.mask_threat_allow_break_ready,
        "threat_max_result_shanten": args.mask_threat_max_result_shanten,
        "threat_gate_threshold": args.mask_threat_gate_threshold,
        "threat_gate_margin": args.mask_threat_gate_margin,
        "threat_min_delta": args.mask_threat_min_delta,
        "threat_gate_mode": args.mask_threat_gate_mode,
        "threat_tell_window": args.mask_threat_tell_window,
        "threat_max_start_shanten": args.mask_threat_max_start_shanten,
        "threat_require_non_exploit": not args.mask_threat_allow_exploit_overlap,
        "threat_require_real_target": args.mask_threat_require_real_target,
        "threat_target_max_shanten": args.mask_threat_target_max_shanten,
        "log_counterfactual": args.mask_log_counterfactual,
        "snapshot_samples": args.snapshot_oracle_samples,
        "snapshot_crn_seeds": args.snapshot_crn_seeds,
    }

    all_games: List[Dict[str, Any]] = []
    all_steps: List[Dict[str, Any]] = []
    all_beliefs: List[Dict[str, Any]] = []
    all_ffr_events: List[Dict[str, Any]] = []
    all_false_folds: List[Dict[str, Any]] = []

    for method in args.methods:
        for i in range(args.games):
            seed = args.seed + i
            game_row, step_rows, belief_rows = play_one_game(
                method=method,
                seed=seed,
                opponent_style=args.opponent_style,
                max_steps=args.max_steps,
                sample_every=args.sample_every,
                backend=args.backend,
                llm=llm,
                belief_llm=belief_llm,
                defender_cfg=defender_cfg,
                mask_cfg=mask_cfg,
            )
            all_games.append(game_row)
            all_steps.extend(step_rows)
            all_beliefs.extend(belief_rows)
            all_ffr_events.extend(game_row.get("ffr_events", []))
            all_false_folds.extend(game_row.get("false_fold_events", []))

    cf_rows = [r for r in all_steps if r.get("disguise_equals_exploit") is not None]
    accepted_cf_rows = [r for r in cf_rows if r.get("mode") == "deceive"]
    n_cf = len(cf_rows)
    n_eq = sum(1 for r in cf_rows if r.get("disguise_equals_exploit"))
    n_accepted_cf = len(accepted_cf_rows)
    n_accepted_eq = sum(1 for r in accepted_cf_rows if r.get("disguise_equals_exploit"))
    deceive_counterfactual = {
        "candidate_decisions_logged": n_cf,
        "candidate_disguise_equals_exploit": n_eq,
        "candidate_overlap_rate": (n_eq / n_cf) if n_cf else None,
        "accepted_deceive_logged": n_accepted_cf,
        "accepted_disguise_equals_exploit": n_accepted_eq,
        "accepted_overlap_rate": (n_accepted_eq / n_accepted_cf) if n_accepted_cf else None,
        "note": ("candidate_* includes rejected threat-gate candidates; accepted_* is the "
                 "actual executed deceive action overlap. High accepted overlap means the "
                 "disguise is a no-op at the action level. Requires --mask-log-counterfactual."),
    }

    summary = {
        "created_for": "Gate1 minimal H1/H2 pipeline",
        "backend": args.backend,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "belief_adapter_path": args.belief_adapter_path,
        "opponent_style": args.opponent_style,
        "methods": list(args.methods),
        "games_per_method": args.games,
        "opponent_style": args.opponent_style,
        "defender_cfg": defender_cfg if args.opponent_style == "responsive" else None,
        "mask_cfg": mask_cfg,
        "E1_belief": summarize_belief_samples(all_beliefs),
        "E2_ladder": summarize_by_method(all_games),
        "FFR_attribution": summarize_false_fold_attribution(all_ffr_events),
        "deceive_counterfactual": deceive_counterfactual,
        "Gate1_pairwise": paired_method_comparison(all_games, "llm_reactive_z", "llm_mask"),
        "ffr_definition": (
            "FFR = turn-phase false folds / opportunities. Opportunity = a responsive opponent "
            "had a real hand (shanten<=ffr_hand_shanten) AND P0 was not actually dangerous "
            "(oracle); false fold = it nonetheless played safe because its public threat estimate "
            "of P0 was high. Only meaningful with --opponent-style responsive."
        ),
        "limitations": [
            "If backend=heuristic_fallback, numbers are pipeline checks rather than trained LLM conclusions.",
            "E1 label is coarse true P0 tenpai; final paper needs opponent-view posterior oracle.",
            "FFR/DIR are only nonzero against belief-responsive opponents (--opponent-style responsive); "
            "the default rule bots (greedy/aggressive/...) do not fold on perceived threat.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "gate1_summary.json", summary)
    write_jsonl(args.output_dir / "gate1_games.jsonl", all_games)
    write_jsonl(args.output_dir / "gate1_steps.jsonl", all_steps)
    write_jsonl(args.output_dir / "gate1_beliefs.jsonl", all_beliefs)
    write_jsonl(args.output_dir / "gate1_ffr_events.jsonl", all_ffr_events)
    write_jsonl(args.output_dir / "gate1_false_folds.jsonl", all_false_folds)
    return summary


def main() -> None:
    ensure_deterministic_hashing()
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--opponent-style", default="greedy", choices=list(OPPONENT_STYLES))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--sample-every", type=int, default=4)
    # ResponsiveDefender knobs (only used when --opponent-style responsive).
    parser.add_argument("--threat-fold-threshold", type=float, default=0.4,
                        help="Opponent folds/declines when its public threat estimate of P0 >= this.")
    parser.add_argument("--oracle-samples", type=int, default=30,
                        help="MC samples for the opponent's public threat estimate (cached per state).")
    parser.add_argument("--oracle-beta", type=float, default=2.0)
    parser.add_argument("--danger-threshold", type=int, default=1,
                        help="P0 'dangerous' = shanten<=this (also the threat-posterior target).")
    parser.add_argument("--ffr-hand-shanten", type=int, default=1,
                        help="Opponent has a 'real hand' worth pushing when its shanten<=this.")
    # MASK-side MC-B_phi and deceive controls.
    parser.add_argument("--mask-oracle-samples", type=int, default=30,
                        help="MC samples for MASK's own B_phi estimate.")
    parser.add_argument("--mask-oracle-beta", type=float, default=2.0)
    parser.add_argument("--mask-danger-threshold", type=int, default=1,
                        help="MASK belief target: P0 dangerous when shanten<=this.")
    parser.add_argument("--mask-dir-ready-threshold", type=int, default=0,
                        help="Strict readiness required for disguise/DIR actions.")
    parser.add_argument("--mask-deceive-threat-ceiling", type=float, default=0.5,
                        help="Natural deceive triggers only when perceived threat <= this.")
    parser.add_argument("--mask-forced-deceive", choices=["off", "eligible", "always"], default="off",
                        help=("Ablation switch. eligible uses style-specific eligibility: "
                              "safe/DIR when ready; threat/FFR when P0 is not dangerous."))
    parser.add_argument("--mask-deceive-style", choices=["safe", "threat"], default="safe",
                        help="safe=old ready-preserving safe discard; threat=visible middle/non-safe discard for FFR ablation.")
    parser.add_argument("--mask-threat-allow-break-ready", action="store_true",
                        help="Allow deceive-threat to break ready. Default is safer: only ready-preserving threat signals.")
    parser.add_argument("--mask-threat-max-result-shanten", type=int, default=0,
                        help=("For deceive-threat from a dangerous hand, allow resulting hand up to this shanten. "
                              "When threat/FFR starts from a non-dangerous hand, the action instead preserves "
                              "non-dangerous status so FFR opportunities remain countable."))
    parser.add_argument("--mask-threat-gate-threshold", type=float, default=0.4,
                        help="Deceive-threat only fires when its projected discard-tell threat crosses this fold threshold.")
    parser.add_argument("--mask-threat-gate-margin", type=float, default=0.12,
                        help="Deceive-threat only considers states with tell threat in [threshold-margin, threshold).")
    parser.add_argument("--mask-threat-min-delta", type=float, default=0.03,
                        help="Minimum projected discard-tell threat increase required for deceive-threat.")
    parser.add_argument("--mask-threat-gate-mode", choices=["cross", "delta_only"], default="cross",
                        help=("cross (default): only fire when tell_before is already near the fold "
                              "threshold AND this single discard crosses it (narrow, can collapse trigger "
                              "rate). delta_only: fire on any meaningfully-pushing discard regardless of "
                              "distance to the threshold, letting pressure build across turns."))
    parser.add_argument("--mask-threat-tell-window", type=int, default=6,
                        help="Recent P0 discard window used by MASK's public tell gate.")
    parser.add_argument("--mask-threat-require-real-target", action="store_true",
                        help=("ORACLE ABLATION (upper bound, not a deployable signal): only allow "
                              "deceive-threat to fire when at least one opponent's true hand "
                              "(ground-truth peek) is within --mask-threat-target-max-shanten. Tests "
                              "whether deceiving only genuinely-close opponents converts FFR into net "
                              "benefit, vs. the default which fires on MASK's own tell projection alone."))
    parser.add_argument("--mask-threat-target-max-shanten", type=int, default=0,
                        help="Oracle threshold for --mask-threat-require-real-target (0 = opponent must be tenpai).")
    parser.add_argument("--mask-threat-max-start-shanten", type=int, default=3,
                        help="Do not use FFR deceive-threat from hopeless hands above this starting shanten.")
    parser.add_argument("--mask-threat-allow-exploit-overlap", action="store_true",
                        help="Allow deceive-threat even if the chosen disguise equals the exploit action. Default requires a real action change.")
    parser.add_argument("--mask-log-counterfactual", action="store_true",
                        help="On deceive decisions, also compute the exploit action and log whether disguise==exploit (measures no-op).")
    parser.add_argument("--snapshot-oracle-samples", type=int, default=120,
                        help="MC samples for the CRN threat before/after snapshot (measurement only; higher = less noise).")
    parser.add_argument("--snapshot-crn-seeds", type=int, default=1,
                        help="Number of common-random-number seeds to average for the threat-delta snapshot.")
    parser.add_argument("--defender-threat-model", choices=["mc", "discard_tell", "blend"], default="mc",
                        help="Opponent threat model. discard_tell is deceivable by P0's discard choice (middle=push/up, terminal/safe=fold/down).")
    parser.add_argument("--defender-tell-weight", type=float, default=1.0,
                        help="blend model: tell_weight*discard_tell + (1-tell_weight)*mc.")
    parser.add_argument("--defender-tell-window", type=int, default=6,
                        help="How many of P0's most recent discards the tell reads.")
    parser.add_argument("--backend", default="heuristic_fallback", choices=["heuristic_fallback", "local_qwen"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--belief-adapter-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved Gate1 outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()

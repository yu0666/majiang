"""Rule-constrained candidate generation for MASK action selection.

The existing MASK agent directly chooses one action in safe/deceive branches.
This module exposes the corresponding candidate sets without changing the
deployed decision path.  It is used by counterfactual rollout experiments and
will later serve as the action mask for a learned reranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from belief_oracle import opponent_view_posterior, within_shanten
from game import MahjongGame
from mask_llm import MASKLLMAgent
from policy_metrics import discard_progress_metrics
from rule_engine import ShantenCalculator


@dataclass
class ModeCandidateSet:
    mode: str
    actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _unique(actions: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for action in actions:
        if action and action not in seen:
            result.append(action)
            seen.add(action)
    return result


def _discard_tile(game: MahjongGame, player_id: int, action: str):
    if not action.startswith("d "):
        return None
    tile_text = action[2:]
    for tile in game.players[player_id].hand_tiles:
        if str(tile) == tile_text:
            return tile
    return None


def _result_shanten(game: MahjongGame, player_id: int, action: str) -> Optional[int]:
    tile = _discard_tile(game, player_id, action)
    if tile is None:
        return None
    player = game.players[player_id]
    remaining = player.hand_tiles.copy()
    remaining.remove(tile)
    return int(ShantenCalculator.calculate_shanten(remaining, player.missing_suit))


def _admissible_progress(
    agent: MASKLLMAgent,
    game: MahjongGame,
    valid_actions: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, int]], int, int]:
    progress: Dict[str, Dict[str, int]] = {}
    for action in valid_actions:
        value = discard_progress_metrics(game, agent.player_id, action)
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
    admissible = [
        action
        for action, value in progress.items()
        if value["shanten"] <= best_shanten + agent.threat_max_shanten_regret
        and value["effective_copies"]
        >= best_effective_copies * agent.threat_min_ukeire_ratio
    ]
    return admissible, progress, best_shanten, best_effective_copies


def exploit_candidates(
    agent: MASKLLMAgent,
    game: MahjongGame,
    valid_actions: List[str],
) -> ModeCandidateSet:
    if "h" in valid_actions:
        return ModeCandidateSet(mode="exploit", actions=["h"], metadata={"forced_win": True})

    admissible, progress, best_shanten, best_effective_copies = _admissible_progress(
        agent, game, valid_actions
    )
    actions = sorted(
        admissible,
        key=lambda action: (
            progress[action]["shanten"],
            -progress[action]["effective_copies"],
            valid_actions.index(action),
        ),
    )
    if "g" in valid_actions:
        actions.insert(0, "g")
    return ModeCandidateSet(
        mode="exploit",
        actions=_unique(actions or valid_actions),
        metadata={
            "candidate_source": "shanten_ranked",
            "result_shanten": {action: value["shanten"] for action, value in progress.items()},
            "effective_copies": {action: value["effective_copies"] for action, value in progress.items()},
            "best_result_shanten": best_shanten,
            "best_effective_copies": best_effective_copies,
        },
    )


def safe_candidates(agent: MASKLLMAgent, game: MahjongGame, valid_actions: List[str]) -> ModeCandidateSet:
    if "h" in valid_actions:
        return ModeCandidateSet(mode="safe", actions=["h"], metadata={"forced_win": True})

    admissible, progress, best_shanten, best_effective_copies = _admissible_progress(
        agent, game, valid_actions
    )
    discards = [action for action in admissible if action.startswith("d ")]
    public = {
        (tile.suit, tile.number)
        for player in game.players
        for tile in player.discarded_tiles
    }
    public_safe = []
    terminals = []
    for action in discards:
        tile = _discard_tile(game, agent.player_id, action)
        if tile is None:
            continue
        if (tile.suit, tile.number) in public:
            public_safe.append(action)
        if tile.number in (1, 9):
            terminals.append(action)

    if public_safe:
        actions = public_safe
        source = "public_safe"
    elif terminals:
        actions = terminals
        source = "terminal_fallback"
    else:
        actions = discards or valid_actions
        source = "legal_fallback"
    return ModeCandidateSet(
        mode="safe",
        actions=_unique(actions),
        metadata={
            "candidate_source": source,
            "best_result_shanten": best_shanten,
            "best_effective_copies": best_effective_copies,
            "action_progress": progress,
        },
    )


def disguise_candidates(agent: MASKLLMAgent, game: MahjongGame, valid_actions: List[str]) -> ModeCandidateSet:
    player = game.players[agent.player_id]
    public = {
        (tile.suit, tile.number)
        for table_player in game.players
        for tile in table_player.discarded_tiles
    }
    keep_safe: List[str] = []
    keep_any: List[str] = []
    admissible, progress, best_shanten, best_effective_copies = _admissible_progress(
        agent, game, valid_actions
    )
    for action in admissible:
        tile = _discard_tile(game, agent.player_id, action)
        if tile is None:
            continue
        remaining = player.hand_tiles.copy()
        remaining.remove(tile)
        if within_shanten(remaining, player.missing_suit, agent.dir_ready_threshold):
            keep_any.append(action)
            if (tile.suit, tile.number) in public or tile.number in (1, 9):
                keep_safe.append(action)

    if keep_safe:
        actions = keep_safe
        source = "keep_ready_public_safe"
    elif keep_any:
        actions = keep_any
        source = "keep_ready"
    elif agent.forced_deceive == "always":
        actions = [action for action in valid_actions if action.startswith("d ")]
        source = "forced_any_discard"
    else:
        actions = []
        source = "blocked_no_ready_discard"
    return ModeCandidateSet(
        mode="deceive",
        actions=_unique(actions),
        metadata={
            "style": "safe",
            "candidate_source": source,
            "best_result_shanten": best_shanten,
            "best_effective_copies": best_effective_copies,
            "action_progress": progress,
        },
    )


def _target_gate(agent: MASKLLMAgent, game: MahjongGame) -> Tuple[bool, Dict[str, Any]]:
    if not agent.threat_require_real_target:
        return True, {"required": False}

    if agent.threat_target_signal == "mc":
        probabilities = {
            player.player_id: float(
                opponent_view_posterior(
                    game,
                    target_pid=player.player_id,
                    observer_pid=agent.player_id,
                    num_samples=agent.mc_oracle_samples,
                    rng=agent.rng,
                    beta=agent.mc_beta,
                    max_shanten=agent.threat_target_max_shanten,
                )["tenpai_prob"]
            )
            for player in game.players
            if player.player_id != agent.player_id
        }
        allowed = any(
            probability >= agent.threat_target_prob_threshold
            for probability in probabilities.values()
        )
        return allowed, {
            "required": True,
            "signal": "mc",
            "threshold": agent.threat_target_prob_threshold,
            "probabilities": {f"P{pid}": round(value, 4) for pid, value in probabilities.items()},
        }

    allowed = any(
        within_shanten(player.hand_tiles, player.missing_suit, agent.threat_target_max_shanten)
        for player in game.players
        if player.player_id != agent.player_id
    )
    return allowed, {
        "required": True,
        "signal": "oracle",
        "max_shanten": agent.threat_target_max_shanten,
    }


def threat_candidates(
    agent: MASKLLMAgent,
    game: MahjongGame,
    valid_actions: List[str],
    mode_already_selected: bool = False,
    beliefs: Optional[Dict[str, Any]] = None,
) -> ModeCandidateSet:
    if "h" in valid_actions:
        return ModeCandidateSet(mode="deceive", actions=["h"], metadata={"forced_win": True})

    if mode_already_selected:
        target_allowed, target_metadata = True, {
            "required": agent.threat_require_real_target,
            "skipped": True,
            "reason": "outer mode gate already accepted deceive",
        }
    else:
        target_allowed, target_metadata = _target_gate(agent, game)
    if not target_allowed:
        return ModeCandidateSet(
            mode="deceive",
            actions=[],
            metadata={"style": "threat", "blocked_reason": "no_real_target", **target_metadata},
        )

    player = game.players[agent.player_id]
    current_shanten = int(ShantenCalculator.calculate_shanten(player.hand_tiles, player.missing_suit))
    ffr_mode = current_shanten > agent.mc_danger_threshold and agent.forced_deceive != "always"
    public = {
        (tile.suit, tile.number)
        for table_player in game.players
        for tile in table_player.discarded_tiles
    }
    tell_before = float(agent._discard_tell_threat(game))
    belief_values = {
        key: float(value.get("tenpai_confidence", 0.0))
        for key, value in (beliefs or {}).items()
    }

    def projected_response(tell_value: float) -> Dict[str, float]:
        if agent.threat_response_model == "blend" and belief_values:
            weight = agent.threat_response_tell_weight
            return {
                key: weight * tell_value + (1.0 - weight) * confidence
                for key, confidence in belief_values.items()
            }
        return {"tell": tell_value}

    response_before = projected_response(tell_before)
    max_result_shanten = 99 if agent.threat_allow_break_ready else agent.threat_max_result_shanten
    admissible, progress, best_shanten, best_effective_copies = _admissible_progress(
        agent, game, valid_actions
    )

    ffr_rows: List[Tuple[float, float, str, int]] = []
    groups: Dict[str, List[str]] = {
        "keep_mid": [],
        "keep_any": [],
        "relaxed_mid": [],
        "relaxed_any": [],
        "break_mid": [],
        "break_any": [],
    }
    action_metadata: Dict[str, Dict[str, Any]] = {}

    for action in admissible:
        tile = _discard_tile(game, agent.player_id, action)
        if tile is None:
            continue
        result_shanten = progress[action]["shanten"]
        effective_copies = progress[action]["effective_copies"]
        keeps_ready = result_shanten <= agent.dir_ready_threshold
        within_relaxed_budget = result_shanten <= max_result_shanten
        is_public_safe = (tile.suit, tile.number) in public
        is_mid = tile.number in (4, 5, 6)
        remains_not_dangerous = result_shanten > agent.mc_danger_threshold
        tell_after = float(agent._discard_tell_threat(game, extra_tile=tile))
        tell_delta = tell_after - tell_before
        response_after = projected_response(tell_after)
        response_deltas = {
            key: response_after[key] - response_before[key]
            for key in response_before
        }
        effective_delta = max(response_deltas.values(), default=tell_delta)
        meaningful_delta = effective_delta >= agent.threat_min_delta
        if agent.threat_gate_mode == "delta_only":
            gate_ok = meaningful_delta and any(
                before < agent.threat_gate_threshold for before in response_before.values()
            )
        else:
            gate_ok = meaningful_delta and any(
                agent.threat_gate_threshold - agent.threat_gate_margin <= response_before[key]
                < agent.threat_gate_threshold <= response_after[key]
                for key in response_before
            )

        if (
            remains_not_dangerous
            and current_shanten <= agent.threat_max_start_shanten
            and gate_ok
        ):
            mid_bonus = 1.0 if is_mid and not is_public_safe else 0.0
            ffr_rows.append((mid_bonus, effective_delta, action, result_shanten))

        if keeps_ready:
            groups["keep_any"].append(action)
            if is_mid and not is_public_safe:
                groups["keep_mid"].append(action)
        elif within_relaxed_budget:
            groups["relaxed_any"].append(action)
            if is_mid and not is_public_safe:
                groups["relaxed_mid"].append(action)
        else:
            groups["break_any"].append(action)
            if is_mid and not is_public_safe:
                groups["break_mid"].append(action)

        action_metadata[action] = {
            "result_shanten": result_shanten,
            "tell_before": round(tell_before, 4),
            "tell_after": round(tell_after, 4),
            "tell_delta": round(tell_delta, 4),
            "effective_response_delta": round(effective_delta, 4),
            "effective_copies": effective_copies,
            "is_mid": is_mid,
            "is_public_safe": is_public_safe,
            "gate_ok": gate_ok,
        }

    if ffr_mode:
        ffr_rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
        actions = [row[2] for row in ffr_rows]
        source = "ffr_gate"
    else:
        source = "none"
        actions = []
        # The deployed rule prefers middle tiles inside each admissible
        # shanten tier. A reranker needs every action in that same tier, not
        # only the rule's preferred subgroup, otherwise it cannot learn when
        # preserving hand quality outweighs a stronger visible tell.
        for group_name, candidate_group in (
            ("keep_ready", groups["keep_any"]),
            ("relaxed", groups["relaxed_any"]),
            ("break_ready", groups["break_any"]),
        ):
            if candidate_group and (
                group_name != "break_ready"
                or agent.threat_allow_break_ready
                or agent.forced_deceive == "always"
            ):
                actions = candidate_group
                source = group_name
                break

    return ModeCandidateSet(
        mode="deceive",
        actions=_unique(actions),
        metadata={
            "style": "threat",
            "candidate_source": source,
            "current_shanten": current_shanten,
            "ffr_mode": ffr_mode,
            "action_metadata": action_metadata,
            "target_gate": target_metadata,
            "best_result_shanten": best_shanten,
            "best_effective_copies": best_effective_copies,
            "min_ukeire_ratio": agent.threat_min_ukeire_ratio,
            "max_shanten_regret": agent.threat_max_shanten_regret,
        },
    )


def build_mode_candidates(
    agent: MASKLLMAgent,
    game: MahjongGame,
    valid_actions: List[str],
    mode: str,
    mode_already_selected: bool = False,
    beliefs: Optional[Dict[str, Any]] = None,
) -> ModeCandidateSet:
    if mode == "safe":
        return safe_candidates(agent, game, valid_actions)
    if mode == "deceive":
        if agent.deceive_style == "threat":
            return threat_candidates(
                agent,
                game,
                valid_actions,
                mode_already_selected=mode_already_selected,
                beliefs=beliefs,
            )
        return disguise_candidates(agent, game, valid_actions)
    return exploit_candidates(agent, game, valid_actions)

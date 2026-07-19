"""Counterfactual rollouts from an in-progress Gate1 Mahjong state.

The rollout engine is intentionally separate from the production evaluator. It
clones game state, defender state, MASK trackers, and RNG state so candidate
actions can be compared without mutating the live episode.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from game import MahjongGame
from mask_llm import MASKLLMAgent
from prompt_builder import get_legal_actions
from run_gate1_experiments import (
    ResponsiveDefender,
    choose_min_shanten_action,
    execute_action,
    resolve_responses,
)


ContinuationPolicy = Callable[[MahjongGame, int, list[str], Optional[MASKLLMAgent]], str]


@dataclass
class RolloutSnapshot:
    game: MahjongGame
    opponent_funcs: Dict[int, Tuple]
    defenders: Dict[int, ResponsiveDefender]
    mask_agent: Optional[MASKLLMAgent]
    steps: int
    max_steps: int
    episode_start_balance: float
    drawn_tile_text: Optional[str]
    last_p0_state: Dict[str, Any]
    deceive_active_until: int
    random_state: object


@dataclass
class RolloutResult:
    action: str
    rollout_seed: int
    continuation_return: float
    episode_net: float
    terminal_balance: float
    agent_hu: bool
    agent_hu_fan: int
    agent_dealin: bool
    settled: bool
    steps: int
    winners: list[int]


def heuristic_continuation_policy(
    game: MahjongGame,
    player_id: int,
    valid_actions: list[str],
    agent: Optional[MASKLLMAgent],
) -> str:
    return choose_min_shanten_action(game, player_id, valid_actions)


def mask_continuation_policy(
    game: MahjongGame,
    player_id: int,
    valid_actions: list[str],
    agent: Optional[MASKLLMAgent],
) -> str:
    if agent is None:
        return choose_min_shanten_action(game, player_id, valid_actions)
    return agent.decide(game, valid_actions)


def rule_mask_continuation_policy(
    game: MahjongGame,
    player_id: int,
    valid_actions: list[str],
    agent: Optional[MASKLLMAgent],
) -> str:
    """Continue with MASK mode/risk rules and a min-shanten exploit backend."""
    if agent is None:
        return choose_min_shanten_action(game, player_id, valid_actions)
    decision_llm = agent.decision_llm
    reranker_llm = agent.reranker_llm
    gate_llm = agent.gate_llm
    gate_policy = agent.gate_policy
    try:
        agent.decision_llm = None
        agent.reranker_llm = None
        agent.gate_llm = None
        agent.gate_policy = "rule"
        return agent.decide(game, valid_actions)
    finally:
        agent.decision_llm = decision_llm
        agent.reranker_llm = reranker_llm
        agent.gate_llm = gate_llm
        agent.gate_policy = gate_policy


def clone_mask_agent(agent: Optional[MASKLLMAgent]) -> Optional[MASKLLMAgent]:
    if agent is None:
        return None
    memo: Dict[int, Any] = {}
    for shared in (
        getattr(agent, "decision_llm", None),
        getattr(agent, "reranker_llm", None),
        getattr(agent, "gate_llm", None),
        getattr(getattr(agent, "belief_estimator", None), "llm", None),
    ):
        if shared is not None:
            memo[id(shared)] = shared
    return copy.deepcopy(agent, memo)


def _clone_defenders(
    defenders: Dict[int, ResponsiveDefender],
) -> Dict[int, ResponsiveDefender]:
    memo: Dict[int, Any] = {}
    for defender in defenders.values():
        learned_model = getattr(defender, "_learned_model", None)
        if learned_model is not None:
            memo[id(learned_model)] = learned_model
    return copy.deepcopy(defenders, memo)


def capture_rollout_snapshot(
    game: MahjongGame,
    opponent_funcs: Dict[int, Tuple],
    defenders: Dict[int, ResponsiveDefender],
    mask_agent: Optional[MASKLLMAgent],
    steps: int,
    max_steps: int,
    episode_start_balance: float,
    drawn_tile=None,
    last_p0_state: Optional[Dict[str, Any]] = None,
    deceive_active_until: int = -1,
) -> RolloutSnapshot:
    return RolloutSnapshot(
        game=copy.deepcopy(game),
        opponent_funcs=dict(opponent_funcs),
        defenders=_clone_defenders(defenders),
        mask_agent=clone_mask_agent(mask_agent),
        steps=steps,
        max_steps=max_steps,
        episode_start_balance=episode_start_balance,
        drawn_tile_text=str(drawn_tile) if drawn_tile is not None else None,
        last_p0_state=copy.deepcopy(last_p0_state or {}),
        deceive_active_until=deceive_active_until,
        random_state=copy.deepcopy(random.getstate()),
    )


def clone_rollout_snapshot(snapshot: RolloutSnapshot) -> RolloutSnapshot:
    return RolloutSnapshot(
        game=copy.deepcopy(snapshot.game),
        opponent_funcs=dict(snapshot.opponent_funcs),
        defenders=_clone_defenders(snapshot.defenders),
        mask_agent=clone_mask_agent(snapshot.mask_agent),
        steps=snapshot.steps,
        max_steps=snapshot.max_steps,
        episode_start_balance=snapshot.episode_start_balance,
        drawn_tile_text=snapshot.drawn_tile_text,
        last_p0_state=copy.deepcopy(snapshot.last_p0_state),
        deceive_active_until=snapshot.deceive_active_until,
        random_state=copy.deepcopy(snapshot.random_state),
    )


def _find_drawn_tile(game: MahjongGame, player_id: int, tile_text: Optional[str]):
    if tile_text is None:
        return None
    player = game.players[player_id]
    last_drawn = getattr(player, "last_drawn_tile", None)
    if last_drawn is not None and str(last_drawn) == tile_text:
        return last_drawn
    for tile in reversed(player.hand_tiles):
        if str(tile) == tile_text:
            return tile
    return None


def _reseed_rollout(snapshot: RolloutSnapshot, rollout_seed: int) -> None:
    random.seed(rollout_seed)
    # Current concealed hands stay fixed, while unknown future draws are
    # resampled. Using a local RNG gives every candidate the same shuffled wall
    # for the same rollout_seed (common random numbers).
    random.Random(rollout_seed).shuffle(snapshot.game.deck.tiles)
    for pid, defender in snapshot.defenders.items():
        defender.rng.seed(rollout_seed * 100 + pid)
        defender._cache.clear()
    if snapshot.mask_agent is not None:
        snapshot.mask_agent.rng.seed(rollout_seed * 100 + snapshot.mask_agent.player_id)
        snapshot.mask_agent._mc_cache.clear()


def _opponent_action(snapshot: RolloutSnapshot, pid: int, in_deceive_window: bool) -> str:
    player = snapshot.game.players[pid]
    if pid in snapshot.defenders:
        return snapshot.defenders[pid].turn(
            player,
            snapshot.game,
            step=snapshot.steps,
            last_p0_state=snapshot.last_p0_state,
            in_deceive_window=in_deceive_window,
        )
    return snapshot.opponent_funcs[pid][0](player, snapshot.game)


def _advance_after_action(
    snapshot: RolloutSnapshot,
    pid: int,
    action: str,
    drawn_tile,
    use_agent_for_responses: bool,
) -> bool:
    """Execute one action and return the next loop's skip_draw value."""
    game = snapshot.game
    discarded_tile = execute_action(game, pid, action, drawn_tile)
    if game.is_game_over:
        return False

    if discarded_tile is None:
        if action == "g":
            return True
        game.next_player()
        return False

    response_agent = snapshot.mask_agent if use_agent_for_responses else None
    response_info = resolve_responses(
        game,
        discarded_tile,
        pid,
        response_agent,
        snapshot.defenders,
    )
    if game.is_game_over:
        return False
    if response_info["responded"]:
        return True
    game.next_player()
    return False


def _run_to_terminal(
    snapshot: RolloutSnapshot,
    skip_draw: bool,
    continuation_policy: ContinuationPolicy,
    use_agent_for_responses: bool,
) -> None:
    game = snapshot.game
    while not game.is_game_over and snapshot.steps < snapshot.max_steps:
        snapshot.steps += 1
        if game.deck.remaining_count() == 0 or sum(1 for player in game.players if player.is_hu) >= 3:
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
            if drawn_tile is None:
                game.check_game_over()
                break
        else:
            skip_draw = False

        if pid == 0:
            valid_actions = get_legal_actions(game, 0)
            action = continuation_policy(game, 0, valid_actions, snapshot.mask_agent)
            mode = "exploit"
            if snapshot.mask_agent is not None and snapshot.mask_agent.last_decision:
                mode = str(snapshot.mask_agent.last_decision.get("mode", mode))
            snapshot.last_p0_state = {
                "step": snapshot.steps,
                "mode": mode,
                "action": action,
                "reason": "rollout continuation",
            }
            if mode == "deceive":
                snapshot.deceive_active_until = snapshot.steps + 4
        else:
            action = _opponent_action(
                snapshot,
                pid,
                in_deceive_window=snapshot.deceive_active_until >= snapshot.steps,
            )

        skip_draw = _advance_after_action(
            snapshot,
            pid,
            action,
            drawn_tile,
            use_agent_for_responses,
        )

    if not game.is_game_over:
        game.check_game_over()


def rollout_candidate(
    snapshot: RolloutSnapshot,
    action: str,
    rollout_seed: int,
    continuation_policy: ContinuationPolicy = heuristic_continuation_policy,
    use_agent_for_responses: bool = False,
    initial_mode: Optional[str] = None,
) -> RolloutResult:
    rollout = clone_rollout_snapshot(snapshot)
    game = rollout.game
    legal_actions = get_legal_actions(game, 0)
    if action not in legal_actions:
        raise ValueError(f"Candidate action is not legal in snapshot: {action}; legal={legal_actions}")

    original_random_state = random.getstate()
    try:
        _reseed_rollout(rollout, rollout_seed)
        snapshot_balance = float(game.players[0].balance)
        drawn_tile = _find_drawn_tile(game, 0, rollout.drawn_tile_text)
        if initial_mode is not None:
            rollout.last_p0_state = {
                "step": rollout.steps,
                "mode": initial_mode,
                "action": action,
                "reason": "counterfactual learned-gate rollout",
            }
            if initial_mode == "deceive":
                rollout.deceive_active_until = rollout.steps + 4
        skip_draw = _advance_after_action(
            rollout,
            0,
            action,
            drawn_tile,
            use_agent_for_responses,
        )
        _run_to_terminal(
            rollout,
            skip_draw,
            continuation_policy,
            use_agent_for_responses,
        )
    finally:
        random.setstate(original_random_state)

    agent = game.players[0]
    terminal_balance = float(agent.balance)
    agent_dealin = any(
        player.is_hu
        and not player.hu_is_self_drawn
        and player.hu_discard_player_id == 0
        for player in game.players
    )
    return RolloutResult(
        action=action,
        rollout_seed=rollout_seed,
        continuation_return=terminal_balance - snapshot_balance,
        episode_net=terminal_balance - float(rollout.episode_start_balance),
        terminal_balance=terminal_balance,
        agent_hu=bool(agent.is_hu),
        agent_hu_fan=int(agent.hu_fan if agent.is_hu else 0),
        agent_dealin=bool(agent_dealin),
        settled=bool(game.is_game_over),
        steps=rollout.steps,
        winners=list(game.winners),
    )

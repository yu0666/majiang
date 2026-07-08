"""
LLM-native prompt construction for the MASK mahjong agent.

This module is the single place for Chinese observation text used by SFT,
GRPO, belief estimation, and evaluation.  It deliberately returns text and
JSON-friendly metadata instead of numeric features.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from game import MahjongGame


def format_tiles(tiles: Iterable) -> str:
    values = [str(t) for t in tiles]
    return " ".join(values) if values else "无"


def format_melds(player) -> str:
    if not player.open_melds:
        return "无"
    return " ".join("[" + " ".join(str(t) for t in meld) + "]" for meld in player.open_melds)


def get_legal_actions(game: MahjongGame, player_id: int, response_actions: Optional[List[str]] = None) -> List[str]:
    player = game.players[player_id]
    if response_actions is not None:
        actions = ["n"]
        if "hu" in response_actions:
            actions.append("h")
        if "gang" in response_actions:
            actions.append("g")
        if "peng" in response_actions:
            actions.append("p")
        return actions

    actions: List[str] = []
    if player.can_hu():
        actions.append("h")

    gang_info = game.can_self_gang(player_id)
    if gang_info.get("can_gang"):
        actions.append("g")

    has_missing = any(t.suit == player.missing_suit for t in player.hand_tiles) if player.missing_suit else False
    seen = set()
    for tile in player.hand_tiles:
        if has_missing and tile.suit != player.missing_suit:
            continue
        text = str(tile)
        if text not in seen:
            actions.append(f"d {text}")
            seen.add(text)

    return actions or ["n"]


def get_public_table_text(game: MahjongGame, viewer_id: int) -> str:
    lines = []
    for player in game.players:
        missing = player.missing_suit.value if player.missing_suit else "未定"
        melds = format_melds(player)
        discards = format_tiles(player.discarded_tiles)
        balance = player.balance
        hu = "已胡" if player.is_hu else "未胡"
        marker = "我" if player.player_id == viewer_id else f"P{player.player_id}"
        lines.append(
            f"{marker}: 定缺={missing}; 副露={melds}; 弃牌={discards}; 余额={balance}; 状态={hu}"
        )
    return "\n".join(lines)


def get_risk_analysis(game: MahjongGame, my_pid: int) -> str:
    risks = []
    for player in game.players:
        if player.player_id == my_pid:
            continue
        risk_level = "安全"
        note = "观察"
        if len(player.open_melds) >= 3:
            risk_level = "极高"
            note = "可能单钓/清一色"
        elif len(player.open_melds) == 2:
            risk_level = "中等"

        if player.discarded_tiles and player.missing_suit and player.discarded_tiles[-1].suit == player.missing_suit:
            note += ", 正在清缺"

        ready, waits = player.is_ready_with_missing_suit()
        if ready:
            note += f", 可能已听牌({format_tiles(waits[:5])})"
        risks.append(f"P{player.player_id}({risk_level}): {note}")
    return " | ".join(risks) if risks else "无"


def build_state_prompt(
    game: MahjongGame,
    player_id: int,
    valid_actions: Optional[List[str]] = None,
    history_k: int = 15,
    objective: str = "最大化收益，同时控制点炮风险",
) -> str:
    player = game.players[player_id]
    valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, player_id)
    missing = player.missing_suit.value if player.missing_suit else "未定"

    return f"""
【战局记忆】
{game.get_history_text(k=history_k)}

【公开牌桌】
{get_public_table_text(game, player_id)}

【局势分析】
剩余牌数: {game.deck.remaining_count()}
对手状态: {get_risk_analysis(game, player_id)}

【当前视角】
我是 P{player_id}
我的定缺: {missing}
我的副露: {format_melds(player)}
我的手牌: {format_tiles(player.hand_tiles)}

【决策空间】
合法动作: {", ".join(valid_actions)}

目标: {objective}
请只输出一个合法动作指令。
""".strip()


def get_observer_footprint(game: MahjongGame, observer_id: int) -> str:
    """Observer j's own public footprint, so the belief prompt differs per j.

    j's melds/discards/missing-suit progress are exactly the j-specific evidence
    that makes 'how would P{j} read me' a per-observer question instead of a
    nominal one.  j's concealed hand is never shown (no leak).
    """
    p = game.players[observer_id]
    missing = p.missing_suit.value if p.missing_suit else "未定"
    melds = format_melds(p)
    discards = format_tiles(p.discarded_tiles)
    recent = format_tiles(p.discarded_tiles[-5:]) if p.discarded_tiles else "无"
    state = "已胡" if p.is_hu else "未胡"
    return (
        f"P{observer_id} 定缺={missing}; 副露={melds}; 近期弃牌={recent}; "
        f"全部弃牌={discards}; 状态={state}"
    )


def build_belief_prompt(game: MahjongGame, player_id: int, target_opponent_id: int, history_k: int = 20) -> str:
    return f"""
你是 MASK 的对手信念估计器 B_phi。下面的“我”指我方 P{player_id}。请基于公开信息，推断对手 P{target_opponent_id} 此刻会如何判断我（P{player_id}）的听牌情况——即“站在 P{target_opponent_id} 的位置看我”。

【战局记忆】
{game.get_history_text(k=history_k)}

【公开牌桌】
{get_public_table_text(game, player_id)}

【对手 P{target_opponent_id} 的公开行为（其视角线索）】
{get_observer_footprint(game, target_opponent_id)}

【任务】
只根据公开信息（我方 P{player_id} 的公开动作 + 对手 P{target_opponent_id} 的公开行为），估计“P{target_opponent_id} 会以为我（P{player_id}）是否听牌/听什么/哪些牌对我危险”。注意这是 P{target_opponent_id} 的主观判断，可能与真实情况不同。

请严格输出 JSON，字段如下:
{{
  "target_opponent": "P{target_opponent_id}",
  "think_i_am_tenpai": "yes|no|uncertain",
  "tenpai_confidence": 0.0,
  "suspected_waits": [],
  "suspected_pattern": "unknown|normal|qingyise|pengpenghu|qidui",
  "danger_tiles_for_me": [],
  "reason": "一句话依据"
}}
""".strip()


def build_mask_decision_prompt(
    game: MahjongGame,
    player_id: int,
    belief_state: Dict,
    gate_state: Dict,
    valid_actions: Optional[List[str]] = None,
    history_k: int = 15,
) -> str:
    state_prompt = build_state_prompt(
        game,
        player_id,
        valid_actions=valid_actions,
        history_k=history_k,
        objective="在 exploit/safe/deceive 三态中选择合适模式，并输出一个合法动作",
    )
    return f"""
{state_prompt}

【对手信念估计 B_phi】
{belief_state}

【风险门控】
{gate_state}

请严格输出 JSON:
{{"mode": "exploit|safe|deceive", "action": "合法动作", "reason": "一句话理由"}}
""".strip()


def build_response_prompt(
    game: MahjongGame,
    player_id: int,
    discard_player_id: int,
    discarded_tile,
    response_actions: List[str],
    history_k: int = 15,
) -> str:
    valid_actions = get_legal_actions(game, player_id, response_actions=response_actions)
    base = build_state_prompt(game, player_id, valid_actions=valid_actions, history_k=history_k)
    return f"""
{base}

【突发事件】
P{discard_player_id} 刚打出 {discarded_tile}，我有响应机会。
请只输出一个响应动作: {", ".join(valid_actions)}
""".strip()

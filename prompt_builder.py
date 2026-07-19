"""
LLM-native prompt construction for the MASK mahjong agent.

This module is the single place for Chinese observation text used by SFT,
GRPO, belief estimation, and evaluation.  It deliberately returns text and
JSON-friendly metadata instead of numeric features.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from game import MahjongGame


ACTION_SYSTEM_PROMPT = (
    "你是四川麻将决策智能体。请根据当前手牌、公开牌桌、历史记录和合法动作空间，"
    "选择预期收益最高的合法动作。严格遵守用户要求的输出格式，不要输出无关解释。"
)


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


def get_public_risk_analysis(game: MahjongGame, my_pid: int) -> str:
    """Estimate opponent pressure from information visible to ``my_pid``."""
    risks = []
    for player in game.players:
        if player.player_id == my_pid:
            continue
        risk_level = "安全"
        note = "观察"
        if len(player.open_melds) >= 3:
            risk_level = "极高"
            note = "三组以上副露"
        elif len(player.open_melds) == 2:
            risk_level = "中等"
            note = "两组副露"

        if player.discarded_tiles and player.missing_suit and player.discarded_tiles[-1].suit == player.missing_suit:
            note += ", 正在清缺"
        if game.deck.remaining_count() <= 20:
            note += ", 牌局后期"
        risks.append(f"P{player.player_id}({risk_level}): {note}")
    return " | ".join(risks) if risks else "无"


def get_oracle_risk_analysis(game: MahjongGame, my_pid: int) -> str:
    """Ground-truth hand-peek risk analysis for explicitly labeled ablations."""
    risks = []
    for player in game.players:
        if player.player_id == my_pid:
            continue
        ready, waits = player.is_ready_with_missing_suit()
        status = f"真实已听牌({format_tiles(waits[:5])})" if ready else "真实未听牌"
        risks.append(f"P{player.player_id}: {status}")
    return " | ".join(risks) if risks else "无"


def get_risk_analysis(game: MahjongGame, my_pid: int) -> str:
    """Backward-compatible name for the deployable public-only analysis."""
    return get_public_risk_analysis(game, my_pid)


def build_state_prompt(
    game: MahjongGame,
    player_id: int,
    valid_actions: Optional[List[str]] = None,
    history_k: int = 15,
    objective: str = "最大化收益，同时控制点炮风险",
    risk_view: str = "public",
) -> str:
    player = game.players[player_id]
    valid_actions = valid_actions if valid_actions is not None else get_legal_actions(game, player_id)
    missing = player.missing_suit.value if player.missing_suit else "未定"
    if risk_view == "public":
        risk_analysis = get_public_risk_analysis(game, player_id)
    elif risk_view == "oracle":
        risk_analysis = get_oracle_risk_analysis(game, player_id)
    else:
        raise ValueError(f"Unknown risk_view: {risk_view}")

    return f"""
【战局记忆】
{game.get_history_text(k=history_k)}

【公开牌桌】
{get_public_table_text(game, player_id)}

【局势分析】
剩余牌数: {game.deck.remaining_count()}
对手状态: {risk_analysis}

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


def build_base_decision_prompt(
    game: MahjongGame,
    player_id: int,
    valid_actions: Optional[List[str]] = None,
    history_k: int = 15,
) -> str:
    """Canonical action prompt shared by the L0/L1/L2 policy backbone."""
    return build_state_prompt(
        game,
        player_id,
        valid_actions=valid_actions,
        history_k=history_k,
        objective="只根据当前手牌和公开牌桌选择一个合法动作，不使用对手信念塑形",
    )


def build_reactive_decision_prompt(
    game: MahjongGame,
    player_id: int,
    z_state: Dict[int, Dict[str, Any]],
    gate_state: Dict[str, Any],
    valid_actions: Optional[List[str]] = None,
    history_k: int = 15,
) -> str:
    """Build the shared L1/L2 exploit prompt without a novel JSON block.

    V1 never saw the old standalone ``z_j(t)`` JSON section during SFT.  Keep
    the observation in the familiar state-prompt schema and summarize only the
    public drift/risk signals that the policy can act on.
    """
    labels = {
        "aggressive_like": "偏激进",
        "conservative_like": "偏保守",
        "mixed_or_unknown": "混合或未知",
    }
    opponents = []
    for pid, state in sorted(z_state.items()):
        label = labels.get(str(state.get("z_label", "")), "混合或未知")
        changed = "近期行为突变" if state.get("drift_flag") else "近期行为稳定"
        opponents.append(f"P{pid}{label}且{changed}")
    drift_text = "；".join(opponents) if opponents else "暂无有效变化"

    mode_text = {
        "safe": "风险偏高，避免明显危险动作",
        "deceive": "风险较低，但仍优先保持牌效",
        "exploit": "正常推进胡牌",
    }.get(str(gate_state.get("mode_hint", "exploit")), "正常推进胡牌")
    risk_budget = gate_state.get("risk_budget", 0.0)
    uncertainty = gate_state.get("uncertainty", 0.0)
    objective = (
        "优先保持或降低向听数并最大化最终收益；"
        f"公开对手行为判断为：{drift_text}；"
        f"当前公开风险预算={risk_budget}、不确定度={uncertainty}，建议={mode_text}。"
        "这些信号只用于风险调整，不得忽略合法动作和基础牌效。"
    )
    return build_state_prompt(
        game,
        player_id,
        valid_actions=valid_actions,
        history_k=history_k,
        objective=objective,
    )


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
    output_format: str = "json",
) -> str:
    if output_format not in {"json", "action"}:
        raise ValueError(f"Unknown MASK output format: {output_format}")
    state_prompt = build_state_prompt(
        game,
        player_id,
        valid_actions=valid_actions,
        history_k=history_k,
        objective=(
            "在 exploit/safe/deceive 三态中选择合适模式，并输出一个合法动作"
            if output_format == "json"
            else "外层规则已固定决策模式，只从后续候选集合选择预期收益最高的合法动作"
        ),
    )
    output_instruction = (
        '请严格输出 JSON:\n{"mode": "exploit|safe|deceive", "action": "合法动作", "reason": "一句话理由"}'
        if output_format == "json"
        else "请只输出一个合法动作本身，不要输出解释或 JSON。"
    )
    return f"""
{state_prompt}

【对手信念估计 B_phi】
{belief_state}

【风险门控】
{gate_state}

{output_instruction}
""".strip()


def _format_z_state_text(z_state: Dict) -> str:
    """Render PublicOpponentTracker.summary() as clean structured text
    instead of a raw Python dict repr (which buries the signal under
    nested braces, floats, and unix timestamps)."""
    if not z_state:
        return "(无对手漂移数据)"
    lines = []
    for pid, info in sorted(z_state.items(), key=lambda kv: str(kv[0])):
        vec = info.get("z_public_vector", {}) or {}
        drift_flag = "是" if info.get("drift_flag") else "否"
        recent = info.get("recent_actions", []) or []
        recent_text = "→".join(
            f"{a.get('desc') or a.get('act', '?')}{a.get('tile', '')}".strip()
            for a in recent
        ) or "(无)"
        lines.append(
            f"P{info.get('player_id', pid)}: 风格={info.get('z_label', 'unknown')}, "
            f"漂移分={info.get('drift_score', 0.0):.2f}, 漂移突变={drift_flag}, "
            f"弃牌率={vec.get('discard_rate', 0.0):.2f}, 碰率={vec.get('peng_rate', 0.0):.2f}, "
            f"杠率={vec.get('gang_rate', 0.0):.2f}, 中张弃牌率={vec.get('mid_discard_rate', 0.0):.2f}, "
            f"幺九弃牌率={vec.get('terminal_discard_rate', 0.0):.2f}, "
            f"最近动作: {recent_text}"
        )
    return "\n".join(lines)


def _format_belief_state_text(belief_state: Dict) -> str:
    """Render the belief-estimator output as clean structured text."""
    if not belief_state:
        return "(无对手信念数据)"
    lines = []
    for key, info in sorted(belief_state.items(), key=lambda kv: str(kv[0])):
        label = {"yes": "是", "no": "否", "uncertain": "不确定"}.get(
            info.get("think_i_am_tenpai"), str(info.get("think_i_am_tenpai"))
        )
        lines.append(
            f"{key}: 认为我听牌={label}, 置信度={info.get('tenpai_confidence', 0.0):.2f}"
        )
    return "\n".join(lines)


def _format_gate_state_text(gate_state: Dict) -> str:
    """Render RiskGate.compute() output as clean structured text."""
    if not gate_state:
        return "(无规则风险摘要)"
    drift_flags = gate_state.get("z_drift_flags", {}) or {}
    drift_text = ", ".join(f"{k}={'是' if v else '否'}" for k, v in sorted(drift_flags.items())) or "(无)"
    return (
        f"建议模式={gate_state.get('mode_hint', 'unknown')}, "
        f"风险预算={gate_state.get('risk_budget', 0.0):.2f}, "
        f"不确定性={gate_state.get('uncertainty', 0.0):.2f}, "
        f"分数差={gate_state.get('score_gap', 0)}, "
        f"剩余牌数={gate_state.get('tiles_left', 0)}, "
        f"对手漂移突变: {drift_text}"
    )


def build_gate_decision_prompt(
    game: MahjongGame,
    player_id: int,
    z_state: Dict,
    belief_state: Dict,
    gate_state: Dict,
    available_modes: List[str],
    history_k: int = 15,
) -> str:
    """Build the public-information prompt for a learned MASK mode gate."""
    state_prompt = build_state_prompt(
        game,
        player_id,
        history_k=history_k,
        objective=(
            "根据牌效、公开对手行为、对手信念和风险预算，"
            "选择预期结算收益最高且尾部风险可控的决策模式"
        ),
    )
    return f"""
{state_prompt}

【公开对手漂移 z_j(t)】
{_format_z_state_text(z_state)}

【对手信念估计 B_phi】
{_format_belief_state_text(belief_state)}

【规则风险摘要】
{_format_gate_state_text(gate_state)}

【可选模式】
{", ".join(available_modes)}

只输出一个可选模式本身：exploit、safe 或 deceive。不要输出动作、解释或 JSON。
注意：正常进攻模式的标签是 exploit，不是 explore；严禁输出 explore。
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

import json
import os
import random
from tqdm import tqdm
from game import (
    MahjongGame, 
    bot_decide_exchange, 
    bot_decide_missing_suit, 
    bot_decide_turn_action, 
    bot_decide_response, 
    parse_console_tile
)

# ================= 🏆 精英数据配置 =================
NUM_GAMES = 150        # 模拟总局数 (筛选后可能只有 5000-8000 条，但都是精品)
OUTPUT_FILE = "sft_data_elite.jsonl2" 
# ===============================================

def get_risk_analysis(game, my_pid):
    """
    【对手建模预处理】
    基于规则生成简单的局势分析，教会 AI 关注对手状态。
    这也为后续训练 '对手建模网络' 提供了 Ground Truth 数据结构。
    """
    risks = []
    for p in game.players:
        if p.player_id == my_pid: continue
        
        # 风险判断逻辑
        risk_level = "安全"
        note = "观察"
        
        # 1. 副露判断
        if len(p.open_melds) >= 3:
            risk_level = "极高"
            note = "可能单钓/清一色"
        elif len(p.open_melds) == 2:
            risk_level = "中等"
        
        # 2. 缺门判断 (四川麻将核心)
        if p.discarded_tiles and p.discarded_tiles[-1].suit == p.missing_suit:
             note += ", 正在清缺"
        
        risks.append(f"P{p.player_id}({risk_level}): {note}")
    
    return " | ".join(risks)

def build_elite_prompt(game, player_id, valid_actions_str):
    """
    【记忆模块预处理】
    构建包含 '记忆' 和 '感知' 的结构化 Prompt
    """
    player = game.players[player_id]
    
    # --- 1. 记忆模块 (History & Context) ---
    # 提取最近的关键动作，不仅仅是流水账
    history_raw = game.get_history_text(k=15)
    
    # --- 2. 对手建模 (Opponent Modeling) ---
    risk_context = get_risk_analysis(game, player_id)
    
    # --- 3. 感知模块 (Current State) ---
    hand_str = " ".join([str(t) for t in player.hand_tiles])
    melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
    missing = player.missing_suit.value if player.missing_suit else "未定"
    
    # 剩余牌墙 (宏观局势)
    tiles_left = game.deck.remaining_count()
    
    # 构造更符合思维链 (CoT) 的 Prompt
    prompt = f"""
【战局记忆】
{history_raw}

【局势分析】
剩余牌数: {tiles_left}
对手状态: {risk_context}

【当前视角】
我是 P{player_id}
我的定缺: {missing}
我的副露: {melds_str}
我的手牌: {hand_str}

【决策空间】
合法动作: {valid_actions_str}

基于以上信息，为了最大化收益，请给出最佳决策（只输出动作指令）：
""".strip()
    return prompt

def generate_dataset():
    print(f"🚀 开始生成【精英收益版】数据...")
    print(f"   - 筛选标准: 必须胡牌 (Hu) 且 净收益 > 0")
    print(f"   - 数据增强: 注入局势分析与合法动作掩码")

    total_saved = 0
    buffer_data = {0: [], 1: [], 2: [], 3: []} # 暂存每局每个人的数据
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for game_idx in tqdm(range(NUM_GAMES), desc="模拟对局"):
            game = MahjongGame(f"GEN_{game_idx}", ["Bot0", "Bot1", "Bot2", "Bot3"], bots=[True]*4)
            
            # 记录初始资金
            start_balances = [p.balance for p in game.players]
            
            game.start_game()
            
            # 清空上一局的暂存
            for pid in range(4): buffer_data[pid] = []
            
            # --- 快速模拟开局 ---
            game.phase = game.phase.EXCHANGE
            for p in game.players:
                game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))
            game._execute_exchange()
            
            game.phase = game.phase.CHOOSE_MISSING
            for p in game.players:
                game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))
            game.phase = game.phase.PLAYING
            
            skip_draw = True
            
            # --- 游戏主循环 ---
            while not game.is_game_over:
                if game.deck.remaining_count() == 0 or sum(1 for p in game.players if p.is_hu) >= 3: break
                pid = game.current_player_id
                player = game.players[pid]

                if player.is_hu:
                    game.next_player(); skip_draw = False; continue

                drawn = None
                if not skip_draw:
                    drawn = game.draw_tile(pid)
                    if not drawn: game.check_game_over(); break
                else: skip_draw = False

                # === 1. 记录由于 "出牌" 产生的决策 ===
                # 计算合法动作 (Action Masking) - 写入训练数据，教会模型只看合法动作
                valid_actions = []
                if player.can_hu(): valid_actions.append("h")
                gang_info = game.can_self_gang(pid)
                if gang_info['can_gang']: valid_actions.append("g")
                
                has_missing = any(t.suit == player.missing_suit for t in player.hand_tiles)
                seen_discard = set()
                for t in player.hand_tiles:
                    if has_missing and t.suit != player.missing_suit: continue
                    t_str = str(t)
                    if t_str not in seen_discard:
                        valid_actions.append(f"d {t_str}")
                        seen_discard.add(t_str)
                valid_actions_str = ", ".join(valid_actions)

                # Bot 决策
                action = bot_decide_turn_action(player, game)
                
                # 构建 Prompt
                prompt = build_elite_prompt(game, pid, valid_actions_str)
                
                # 存入暂存区
                record = {
                    "messages": [
                        {"role": "system", "content": "你是一个四川麻将高手。你会根据局势分析风险，并严格遵守合法动作规则，以最大化收益为目标。"},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": action}
                    ]
                }
                buffer_data[pid].append(record)

                # 执行动作
                turn_end = False
                if action == 'h':
                    game.hu(pid, drawn if drawn else player.hand_tiles[-1], True)
                    if game.check_game_over(True): break
                    game.next_player(); skip_draw = False; turn_end = True
                elif action == 'g':
                    g_info = game.can_self_gang(pid)
                    if g_info['can_gang']:
                        game.gang(pid, g_info['gang_tiles'][0])
                        continue
                elif action.startswith('d '):
                    t = parse_console_tile(action[2:])
                    if t and game.discard_tile(pid, t):
                        res = game.check_responses(t, pid)
                        responded = False
                        if res:
                            for r_pid, acts in res.items():
                                if responded: break
                                r_player = game.players[r_pid]
                                r_action = bot_decide_response(r_player, acts)
                                
                                # === 2. 记录 "响应" 决策 (碰/杠/胡) ===
                                valid_resps = ['n']
                                if 'hu' in acts: valid_resps.append('h')
                                if 'gang' in acts: valid_resps.append('g')
                                if 'peng' in acts: valid_resps.append('p')
                                r_valid_str = ", ".join(valid_resps)
                                
                                r_prompt = build_elite_prompt(game, r_pid, r_valid_str)
                                r_prompt += f"\n【突发事件】\n对手 P{pid} 打出了 {t}，触发响应机会。"
                                
                                r_record = {
                                    "messages": [
                                        {"role": "system", "content": "你是一个四川麻将高手。"},
                                        {"role": "user", "content": r_prompt},
                                        {"role": "assistant", "content": r_action}
                                    ]
                                }
                                buffer_data[r_pid].append(r_record)

                                if r_action != 'n':
                                    if r_action == 'h':
                                        game.hu(r_pid, t, False, pid)
                                        responded = True
                                        if game.check_game_over(True): break
                                    elif r_action == 'g':
                                        game.gang(r_pid, t, pid)
                                        game.current_player_id = r_pid
                                        responded = True; skip_draw = True
                                    elif r_action == 'p':
                                        game.peng(r_pid, t, pid)
                                        game.current_player_id = r_pid
                                        responded = True; skip_draw = True
                        
                        if game.is_game_over: break
                        if not responded: game.next_player(); skip_draw = False
                        turn_end = True
                    else: turn_end = True

            # === 游戏结束，核心筛选逻辑 ===
            for p in game.players:
                net_income = p.balance - start_balances[p.player_id]
                
                # 💎 精英筛选标准 💎
                # 1. 必须胡牌 (is_hu)
                # 2. 必须赚钱 (net_income > 0)
                # 这会过滤掉那些 "屁胡但点炮输钱" 的低质量数据
                if p.is_hu and net_income > 0:
                    for rec in buffer_data[p.player_id]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_saved += 1

    print(f"\n✅ 精英数据生成完毕！")
    print(f"   - 输出文件: {OUTPUT_FILE}")
    print(f"   - 有效样本数: {total_saved}")
    print(f"   - (样本虽然变少了，但每一条都是通往胜利且盈利的正确示范)")

if __name__ == "__main__":
    generate_dataset()
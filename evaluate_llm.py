import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os
import sys
import contextlib

# 导入游戏核心模块
from game import (
    MahjongGame, 
    bot_decide_exchange, 
    bot_decide_missing_suit, 
    bot_decide_turn_action, 
    bot_decide_response,
    parse_console_tile
)

# 尝试导入本地 LLM Agent
try:
    from local_llm_agent import LocalLLMAgent
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️ Warning: local_llm_agent.py 未找到，将使用 Bot 替代 LLM 进行评估。")

# ================= 🔧 评估配置 =================
NUM_EPISODES = 10       # 评估局数
INITIAL_BALANCE = 10000  # 初始资金
LLM_PLAYER_ID = 0        # AI 所在的玩家 ID
VERBOSE = True          # 【开启/关闭】打印详细过程(上帝视角)
# ==============================================

# ============================================================
# 🛠️ 辅助函数：局势分析 & Prompt 构建
# ============================================================
def get_risk_analysis(game, my_pid):
    risks = []
    for p in game.players:
        if p.player_id == my_pid: continue
        risk_level = "安全"
        note = "观察"
        if len(p.open_melds) >= 3:
            risk_level = "极高"
            note = "可能单钓/清一色"
        elif len(p.open_melds) == 2:
            risk_level = "中等"
        if p.discarded_tiles and p.discarded_tiles[-1].suit == p.missing_suit:
             note += ", 正在清缺"
        risks.append(f"P{p.player_id}({risk_level}): {note}")
    return " | ".join(risks)

def build_observation_prompt(game: MahjongGame, player_id: int, valid_actions: list = None) -> str:
    player = game.players[player_id]
    history_raw = game.get_history_text(k=15)
    risk_context = get_risk_analysis(game, player_id)
    hand_str = " ".join([str(t) for t in player.hand_tiles])
    melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
    missing = player.missing_suit.value if player.missing_suit else "未定"
    tiles_left = game.deck.remaining_count()
    valid_str = ", ".join(valid_actions) if valid_actions else "无限制"

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
合法动作: {valid_str}

基于以上信息，为了最快胡牌，请给出最佳决策（只输出动作指令）：
"""
    return prompt.strip()

@contextlib.contextmanager
def suppress_stdout():
    if VERBOSE: yield
    else:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try: yield
            finally: sys.stdout = old_stdout

def print_game_snapshot(game, current_pid, drawn_tile=None):
    if not VERBOSE: return
    print("\n" + "-"*30 + f" 🀄 剩余牌墙: {game.deck.remaining_count()} " + "-"*30)
    for p in game.players:
        marker = "👉" if p.player_id == current_pid else "  "
        hu_mark = "🎉[已胡]" if p.is_hu else ""
        hand_str = " ".join([str(t) for t in p.hand_tiles])
        if p.player_id == current_pid and drawn_tile and not p.is_hu:
            hand_str += f" + 摸[{drawn_tile}]"
        meld_str = ""
        if p.open_melds:
            meld_str = " | 副露: " + " ".join([f"[{str(m[0])}x{len(m)}]" for m in p.open_melds])
        print(f"{marker} P{p.player_id} [{p.name}]: {hand_str}{meld_str} | 缺: {p.missing_suit.value if p.missing_suit else 'None'} {hu_mark}")
    print("-" * 75)

# ============================================================
# 🌟 原生详细结算打印 (来自提供的代码)
# ============================================================
def print_detailed_settlement(game: MahjongGame, start_balances: list):
    if not VERBOSE: return
    
    print("\n" + "█" * 30 + " 本局详细结算 " + "█" * 30)
    
    dian_pao_players = {}
    for p in game.players:
        if p.is_hu and not p.hu_is_self_drawn and p.hu_discard_player_id is not None:
            dian_pao_players[p.player_id] = p.hu_discard_player_id

    for i, p in enumerate(game.players):
        net_score = p.balance - start_balances[i]
        score_str = f"+{net_score}" if net_score >= 0 else f"{net_score}"
        
        role = "[BOT]" if p.is_bot else "[YOU]"
        status_tags = []
        details = []

        if p.is_hu:
            status_tags.append("【胡牌】")
            if p.hu_is_self_drawn:
                method = "自摸"
            else:
                loser_id = p.hu_discard_player_id
                method = f"捉 玩家{loser_id} 炮"
            fan_str = ",".join(p.hu_fan_types)
            details.append(f"{method} | {fan_str} | 共{p.hu_fan}番")
        else:
            has_missing = any(t.suit == p.missing_suit for t in p.hand_tiles)
            if has_missing:
                status_tags.append("【花猪】")
                details.append("定缺牌未打完，赔付所有非花猪玩家满番")
            else:
                max_fan, _ = p.calculate_potential_fan()
                if max_fan == 0:
                    status_tags.append("【无叫】")
                    if game.deck.remaining_count() == 0:
                        details.append("流局未听牌，赔付听牌玩家")
                else:
                    status_tags.append("【听牌】")
                    if game.deck.remaining_count() == 0:
                        details.append(f"手握{max_fan}番，理论最大番")

            pao_targets = [str(wid) for wid, lid in dian_pao_players.items() if lid == p.player_id]
            if pao_targets:
                status_tags.append("【点炮】")
                details.append(f"点炮给 -> 玩家 {','.join(pao_targets)}")

        print(f"P{p.player_id} {role} {score_str} {' '.join(status_tags)}")
        for d in details: print(f"   └─ {d}")
        
    print("-" * 76)
    print(f"当前余额: ", end="")
    for p in game.players: print(f"P{p.player_id}:{p.balance}  ", end="")
    print("\n" + "█" * 76 + "\n")

# ============================================================
# 🚀 主评估逻辑
# ============================================================
def run_evaluation():
    global HAS_LLM 

    print(f"\n🚀 开始评估: 微调大模型 (Qwen) vs 规则Bot")
    print(f"   - 局数: {NUM_EPISODES}")
    print(f"   - 详细模式: {'✅ 开启' if VERBOSE else '❌ 关闭'}")
    
    llm_agent = None
    if HAS_LLM:
        try:
            print("⏳ 正在初始化 LLM Agent...")
            llm_agent = LocalLLMAgent() 
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            HAS_LLM = False

    player_names = ["Qwen-SFT", "Bot-1", "Bot-2", "Bot-3"]
    bots_config = [False, True, True, True] 
    
    global_balances = [INITIAL_BALANCE] * 4
    balance_history = [[INITIAL_BALANCE] * 4] 
    
    # 统计数据：新增 hu_fan_ge_1 和 dianpao_count
    stats = { 
        "hu_count": [0] * 4, 
        "hu_fan_ge_1": [0] * 4,
        "dianpao_count": [0] * 4, 
        "total_fan": [0] * 4 
    }

    iterator = range(NUM_EPISODES) if VERBOSE else tqdm(range(NUM_EPISODES), desc="对战进度")
    
    for ep in iterator:
        if VERBOSE: print(f"\n📢 >>>>>> 第 {ep+1} 局开始 <<<<<<")
        
        game = MahjongGame(f"EVAL_{ep}", player_names, bots=bots_config)
        for i, p in enumerate(game.players): p.balance = global_balances[i]
        round_start_balances = [p.balance for p in game.players]
        
        game.start_game()
        
        game.phase = game.phase.EXCHANGE
        for p in game.players:
            game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))
        game.phase = game.phase.CHOOSE_MISSING
        for p in game.players:
            game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))
            
        game.phase = game.phase.PLAYING
        skip_draw = True
        game_step_count = 0 

        while not game.is_game_over:
            game_step_count += 1
            if game_step_count > 300 or sum(1 for p in game.players if p.is_hu) >= 3:
                game.is_game_over = True; break

            pid = game.current_player_id
            player = game.players[pid]

            if player.is_hu:
                game.next_player(); skip_draw = False; continue

            drawn = None
            if not skip_draw:
                drawn = game.draw_tile(pid)
                if not drawn: 
                    # 🌟 游戏核心机制：摸完牌触发 check_game_over，底层自动查叫、查花猪 🌟
                    game.check_game_over()
                    break
            else: skip_draw = False

            print_game_snapshot(game, pid, drawn)

            turn_end = False
            loop_attempts = 0 

            while not turn_end:
                loop_attempts += 1
                action = ""
                force_bot = loop_attempts > 3 
                
                if HAS_LLM and pid == LLM_PLAYER_ID and not force_bot:
                    try:
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
                        
                        prompt = build_observation_prompt(game, pid, valid_actions)
                        
                        if VERBOSE:
                            print(f"🤖 [AI 思考] (可选: {len(valid_actions)}个动作)...")
                            action = llm_agent.decide(prompt)
                            print(f"🤖 [AI 决策]: {action}")
                        else:
                            with suppress_stdout():
                                action = llm_agent.decide(prompt)
                                
                        if action not in valid_actions:
                            found = False
                            for va in valid_actions:
                                if va in action: action = va; found = True; break
                            if not found and valid_actions: action = valid_actions[0]

                    except Exception as e:
                        if VERBOSE: print(f"❌ AI 出错: {e}")
                        action = bot_decide_turn_action(player, game)
                else:
                    action = bot_decide_turn_action(player, game)
                    if VERBOSE and pid != LLM_PLAYER_ID:
                        print(f"🤖 [Bot P{pid}]: {action}")

                if action == 'h':
                    if player.can_hu():
                        win_card = drawn if drawn else player.hand_tiles[-1]
                        if VERBOSE: print(f"🎉 P{pid} 自摸胡牌！")
                        game.hu(pid, win_card, True)
                        game.check_game_over() 
                        if game.is_game_over: turn_end = True; break
                        turn_end = True; game.next_player(); skip_draw = False
                    else: pass 
                elif action == 'g':
                    g_info = game.can_self_gang(pid)
                    if g_info['can_gang']:
                        if VERBOSE: print(f"💥 P{pid} 杠牌！")
                        game.gang(pid, g_info['gang_tiles'][0])
                        continue
                elif action.startswith('d '):
                    t = parse_console_tile(action[2:])
                    if t and game.discard_tile(pid, t):
                        if VERBOSE: print(f"👉 P{pid} 打出: {t}")
                        res = game.check_responses(t, pid)
                        someone_responded = False
                        if res:
                            for r_id, acts in res.items():
                                if someone_responded: break
                                responder = game.players[r_id]
                                choice = 'n'
                                
                                if HAS_LLM and r_id == LLM_PLAYER_ID:
                                    try:
                                        valid_resps = ['n']
                                        if 'hu' in acts: valid_resps.append('h')
                                        if 'gang' in acts: valid_resps.append('g')
                                        if 'peng' in acts: valid_resps.append('p')
                                        
                                        prompt = build_observation_prompt(game, r_id, valid_resps)
                                        prompt += f"\n【突发事件】\n对手 P{pid} 打出了 {t}，触发响应机会。"
                                        
                                        if VERBOSE:
                                            print(f"⚡ [AI 响应思考] 对手打出 {t} (可选: {valid_resps})...")
                                            choice = llm_agent.decide(prompt)
                                            print(f"⚡ [AI 响应]: {choice}")
                                        else:
                                            with suppress_stdout():
                                                choice = llm_agent.decide(prompt)
                                        
                                        if choice not in valid_resps:
                                            if 'h' in choice and 'h' in valid_resps: choice = 'h'
                                            elif 'p' in choice and 'p' in valid_resps: choice = 'p'
                                            elif 'g' in choice and 'g' in valid_resps: choice = 'g'
                                            else: choice = 'n'
                                    except: choice = 'n'
                                else:
                                    choice = bot_decide_response(responder, acts)
                                
                                if choice == 'h' and 'hu' in acts:
                                    if VERBOSE: print(f"🎉 P{r_id} 食胡！点炮者: P{pid}")
                                    game.hu(r_id, t, False, pid)
                                    # 记录点炮数
                                    stats["dianpao_count"][pid] += 1 
                                    game.check_game_over()
                                    someone_responded = True
                                    if game.is_game_over: turn_end = True; break
                                elif choice == 'g' and 'gang' in acts:
                                    if VERBOSE: print(f"💥 P{r_id} 明杠: {t}")
                                    game.gang(r_id, t, pid)
                                    game.current_player_id = r_id
                                    turn_end = True; someone_responded = True; skip_draw = True
                                elif choice == 'p' and 'peng' in acts:
                                    if VERBOSE: print(f"🤜 P{r_id} 碰牌: {t}")
                                    game.peng(r_id, t, pid)
                                    game.current_player_id = r_id
                                    turn_end = True; someone_responded = True; skip_draw = True
                        
                        if game.is_game_over: break
                        if not someone_responded:
                            game.next_player(); skip_draw = False
                        turn_end = True
                    else: pass

        game.check_game_over()
        print_detailed_settlement(game, round_start_balances)
        
        # --- 局后数据收集更新 ---
        for i, p in enumerate(game.players):
            global_balances[i] = p.balance
            if p.is_hu:
                stats["hu_count"][i] += 1
                stats["total_fan"][i] += p.hu_fan
                if p.hu_fan >= 2:
                    stats["hu_fan_ge_1"][i] += 1
        
        balance_history.append(global_balances.copy())

    print("\n" + "="*95)
    print(f"📊 最终战报 (共{NUM_EPISODES}局)")
    print("="*95)
    print(f"{'ID':<4} {'Role':<15} {'Wins':<8} {'Win(>=2F)':<12} {'DianPao':<10} {'TotalFan':<10} {'Balance':<12} {'Net':<10}")
    print("-" * 95)
    
    for i in range(4):
        role = player_names[i]
        net = global_balances[i] - INITIAL_BALANCE
        net_str = f"+{net}" if net > 0 else f"{net}"
        print(f"{i:<4} {role:<15} {stats['hu_count'][i]:<8} {stats['hu_fan_ge_1'][i]:<12} {stats['dianpao_count'][i]:<10} {stats['total_fan'][i]:<10} {global_balances[i]:<12} {net_str:<10}")

    print("-" * 95)
    plot_balance_curve(balance_history)

def plot_balance_curve(history):
    history = np.array(history)
    episodes = range(len(history))
    plt.figure(figsize=(12, 7))
    plt.plot(episodes, history[:, 0], label="Qwen-SFT", color='#FF4444', linewidth=3, marker='o', markersize=4)
    plt.plot(episodes, history[:, 1], label="Bot-1", color='#FFDD44', linewidth=1.5, linestyle='--')
    plt.plot(episodes, history[:, 2], label="Bot-2", color='#44AAFF', linewidth=1.5, linestyle='--')
    plt.plot(episodes, history[:, 3], label="Bot-3", color='#44FF44', linewidth=1.5, linestyle='--')
    plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle=':', alpha=0.5)
    plt.title(f"Evaluation Result")
    plt.xlabel("Episodes")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = "local_llm_result.png"
    plt.savefig(filename, dpi=150)
    print(f"\n📈 趋势图已保存: {filename}")

if __name__ == "__main__":
    run_evaluation()
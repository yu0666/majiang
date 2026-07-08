# # bot vs 3 llm
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm
# import os
# import sys
# import contextlib

# # 导入游戏核心模块
# from game import (
#     MahjongGame, 
#     bot_decide_exchange, 
#     bot_decide_missing_suit, 
#     bot_decide_turn_action, 
#     bot_decide_response,
#     parse_console_tile
# )

# # 🌟 尝试导入 API 版的 LLM Agent
# try:
#     from llm_agent import LLMAgent
#     HAS_LLM = True
# except ImportError:
#     HAS_LLM = False
#     print("⚠️ Warning: llm_agent.py 未找到，将使用 Bot 替代 LLM 进行评估。")

# # ================= 🔧 评估配置 =================
# NUM_EPISODES = 100       # ⚠️ API 调用需要网络请求，3个大模型调用量是原来的3倍，建议先设为 50-100 局测试
# INITIAL_BALANCE = 10000  # 初始资金
# VERBOSE = False          # 关闭/开启详细模式(上帝视角)

# # ================= 🌐 API 模型配置 =================
# API_KEY = os.getenv("DASHSCOPE_API_KEY", "") # 填入你的通义/混元 API Key
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# MODEL_NAME = "qwen-plus" # 也可以换成 qwen-max
# # ==============================================


# @contextlib.contextmanager
# def suppress_stdout():
#     """详细模式下不屏蔽输出，否则屏蔽"""
#     if VERBOSE: yield
#     else:
#         with open(os.devnull, "w") as devnull:
#             old_stdout = sys.stdout
#             sys.stdout = devnull
#             try: yield
#             finally: sys.stdout = old_stdout

# # ============================================================
# # 🛠️ 可视化打印函数
# # ============================================================
# def print_game_snapshot(game, current_pid, drawn_tile=None):
#     if not VERBOSE: return
    
#     print("\n" + "-"*30 + f" 🀄 剩余牌墙: {game.deck.remaining_count()} " + "-"*30)
#     for p in game.players:
#         marker = "👉" if p.player_id == current_pid else "  "
#         hu_mark = "🎉[已胡]" if p.is_hu else ""
        
#         hand_str = " ".join([str(t) for t in p.hand_tiles])
#         if p.player_id == current_pid and drawn_tile and not p.is_hu:
#             hand_str += f" + 摸[{drawn_tile}]"
            
#         meld_str = ""
#         if p.open_melds:
#             meld_str = " | 副露: " + " ".join([f"[{str(m[0])}x{len(m)}]" for m in p.open_melds])
            
#         print(f"{marker} P{p.player_id} [{p.name}]: {hand_str}{meld_str} | 缺: {p.missing_suit.value if p.missing_suit else 'None'} {hu_mark}")
#     print("-" * 75)

# def print_settlement(game, start_balances):
#     if not VERBOSE: return
    
#     print("\n" + "="*25 + " 📊 本局结算 " + "="*25)
#     print(f"{'玩家':<10} {'状态':<8} {'番数':<6} {'本局盈亏':<10} {'当前资金':<10} {'最终手牌'}")
#     print("-" * 80)
    
#     for p in game.players:
#         status = "胡牌" if p.is_hu else "未胡"
#         diff = p.balance - start_balances[p.player_id]
#         diff_str = f"+{diff}" if diff > 0 else str(diff)
#         hand_final = " ".join([str(t) for t in p.hand_tiles])
        
#         print(f"P{p.player_id:<4} {p.name:<15} {status:<6} {p.hu_fan:<6} {diff_str:<10} {p.balance:<10} {hand_final}")
#     print("=" * 80 + "\n")

# # ============================================================
# # 🚀 主评估逻辑
# # ============================================================
# def run_evaluation():
#     global HAS_LLM 

#     print(f"\n🚀 开始评估: 3x 云端 API ({MODEL_NAME}) vs 1x 规则Bot")
#     print(f"   - 局数: {NUM_EPISODES}")
#     print(f"   - 详细模式: {'✅ 开启' if VERBOSE else '❌ 关闭'}")
    
#     llm_agent = None
#     if HAS_LLM:
#         try:
#             print(f"⏳ 正在初始化 API LLM Agent [{MODEL_NAME}]...")
#             llm_agent = LLMAgent(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)
#             print("✅ 模型连接成功！")
#         except Exception as e:
#             print(f"❌ 模型初始化失败: {e}")
#             HAS_LLM = False

#     # 🌟 核心修改：3个大模型，1个Bot
#     player_names = [f"API-{MODEL_NAME}-A", f"API-{MODEL_NAME}-B", f"API-{MODEL_NAME}-C", "Bot-1"]
#     bots_config = [False, False, False, True] 
    
#     global_balances = [INITIAL_BALANCE] * 4
#     balance_history = [[INITIAL_BALANCE] * 4] 
    
#     stats = { "hu_count": [0] * 4, "total_fan": [0] * 4, "dianpao_count": [0] * 4 }

#     iterator = range(NUM_EPISODES) if VERBOSE else tqdm(range(NUM_EPISODES), desc="对战进度")
    
#     for ep in iterator:
#         if VERBOSE: print(f"\n📢 >>>>>> 第 {ep+1} 局开始 <<<<<<")
        
#         game = MahjongGame(f"EVAL_{ep}", player_names, bots=bots_config)
#         for i, p in enumerate(game.players): p.balance = global_balances[i]
#         round_start_balances = [p.balance for p in game.players]
        
#         game.start_game()
        
#         game.phase = game.phase.EXCHANGE
#         for p in game.players:
#             game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))
#         game.phase = game.phase.CHOOSE_MISSING
#         for p in game.players:
#             game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))
            
#         game.phase = game.phase.PLAYING
#         skip_draw = True
#         game_step_count = 0 

#         while not game.is_game_over:
#             game_step_count += 1
#             if game_step_count > 300 or sum(1 for p in game.players if p.is_hu) >= 3:
#                 game.is_game_over = True; break

#             pid = game.current_player_id
#             player = game.players[pid]

#             if player.is_hu:
#                 game.next_player(); skip_draw = False; continue

#             drawn = None
#             if not skip_draw:
#                 drawn = game.draw_tile(pid)
#                 if not drawn: game.check_game_over(); break
#             else: skip_draw = False

#             print_game_snapshot(game, pid, drawn)

#             turn_end = False
#             loop_attempts = 0 

#             while not turn_end:
#                 loop_attempts += 1
#                 action = ""
#                 force_bot = loop_attempts > 3 
                
#                 # === 决策 ===
#                 # 🌟 核心修改：只要不是 bot，就调用大模型 API
#                 if HAS_LLM and not bots_config[pid] and not force_bot:
#                     try:
#                         valid_actions = []
#                         if player.can_hu(): valid_actions.append("h")
#                         gang_info = game.can_self_gang(pid)
#                         if gang_info['can_gang']: valid_actions.append("g")
                        
#                         has_missing = any(t.suit == player.missing_suit for t in player.hand_tiles)
#                         seen_discard = set()
#                         for t in player.hand_tiles:
#                             if has_missing and t.suit != player.missing_suit: continue
#                             t_str = str(t)
#                             if t_str not in seen_discard:
#                                 valid_actions.append(f"d {t_str}")
#                                 seen_discard.add(t_str)
                        
#                         if VERBOSE:
#                             print(f"🤖 [P{pid} API 思考] (可选: {len(valid_actions)}个动作)...")
#                             action = llm_agent.get_action(player, game, valid_actions)
#                         else:
#                             with suppress_stdout():
#                                 action = llm_agent.get_action(player, game, valid_actions)
                                
#                         if action not in valid_actions:
#                             found = False
#                             for va in valid_actions:
#                                 if va in action: action = va; found = True; break
#                             if not found and valid_actions: action = valid_actions[0]

#                     except Exception as e:
#                         if VERBOSE: print(f"❌ API 出错: {e}")
#                         action = bot_decide_turn_action(player, game)
#                 else:
#                     action = bot_decide_turn_action(player, game)
#                     if VERBOSE and bots_config[pid]:
#                         print(f"🤖 [Bot P{pid}]: {action}")

#                 # === 执行动作 ===
#                 if action == 'h':
#                     if player.can_hu():
#                         win_card = drawn if drawn else player.hand_tiles[-1]
#                         if VERBOSE: print(f"🎉 P{pid} 自摸胡牌！")
#                         game.hu(pid, win_card, True)
#                         game.check_game_over() 
#                         if game.is_game_over: turn_end = True; break
#                         turn_end = True; game.next_player(); skip_draw = False
#                     else: pass 
#                 elif action == 'g':
#                     g_info = game.can_self_gang(pid)
#                     if g_info['can_gang']:
#                         if VERBOSE: print(f"💥 P{pid} 杠牌！")
#                         game.gang(pid, g_info['gang_tiles'][0])
#                         continue
#                 elif action.startswith('d '):
#                     t = parse_console_tile(action[2:])
#                     if t and game.discard_tile(pid, t):
#                         if VERBOSE: print(f"👉 P{pid} 打出: {t}")
#                         res = game.check_responses(t, pid)
#                         someone_responded = False
#                         if res:
#                             for r_id, acts in res.items():
#                                 if someone_responded: break
#                                 responder = game.players[r_id]
#                                 choice = 'n'
                                
#                                 # 响应决策
#                                 # 🌟 核心修改：只要不是 bot，就让对应的大模型去思考响应
#                                 if HAS_LLM and not bots_config[r_id]:
#                                     try:
#                                         valid_resps = ['n']
#                                         if 'hu' in acts: valid_resps.append('h')
#                                         if 'gang' in acts: valid_resps.append('g')
#                                         if 'peng' in acts: valid_resps.append('p')
                                        
#                                         if VERBOSE:
#                                             print(f"⚡ [P{r_id} API 响应思考] 针对 {t}...")
#                                             choice = llm_agent.get_action(responder, game, valid_resps)
#                                         else:
#                                             with suppress_stdout():
#                                                 choice = llm_agent.get_action(responder, game, valid_resps)
                                        
#                                         if choice not in valid_resps:
#                                             if 'h' in choice and 'h' in valid_resps: choice = 'h'
#                                             elif 'p' in choice and 'p' in valid_resps: choice = 'p'
#                                             elif 'g' in choice and 'g' in valid_resps: choice = 'g'
#                                             else: choice = 'n'
#                                     except: choice = 'n'
#                                 else:
#                                     choice = bot_decide_response(responder, acts)
                                
#                                 # 执行响应
#                                 if choice == 'h' and 'hu' in acts:
#                                     if VERBOSE: print(f"🎉 P{r_id} 食胡！点炮者: P{pid}")
#                                     game.hu(r_id, t, False, pid)
#                                     stats["dianpao_count"][pid] += 1
#                                     game.check_game_over()
#                                     someone_responded = True
#                                     if game.is_game_over: turn_end = True; break
#                                 elif choice == 'g' and 'gang' in acts:
#                                     if VERBOSE: print(f"💥 P{r_id} 明杠: {t}")
#                                     game.gang(r_id, t, pid)
#                                     game.current_player_id = r_id
#                                     turn_end = True; someone_responded = True; skip_draw = True
#                                 elif choice == 'p' and 'peng' in acts:
#                                     if VERBOSE: print(f"🤜 P{r_id} 碰牌: {t}")
#                                     game.peng(r_id, t, pid)
#                                     game.current_player_id = r_id
#                                     turn_end = True; someone_responded = True; skip_draw = True
                        
#                         if game.is_game_over: break
#                         if not someone_responded:
#                             game.next_player(); skip_draw = False
#                         turn_end = True
#                     else: pass

#         game.check_game_over()
#         print_settlement(game, round_start_balances)
        
#         for i, p in enumerate(game.players):
#             global_balances[i] = p.balance
#             if p.is_hu:
#                 stats["hu_count"][i] += 1
#                 stats["total_fan"][i] += p.hu_fan
        
#         balance_history.append(global_balances.copy())

#     print("\n" + "="*85)
#     print(f"📊 最终战报 (共{NUM_EPISODES}局)")
#     print("="*85)
#     print(f"{'ID':<4} {'Role':<18} {'Wins(胡)':<10} {'DianPao(炮)':<12} {'TotalFan':<10} {'Balance':<12} {'Net':<10}")
#     print("-" * 85)
    
#     for i in range(4):
#         role = player_names[i]
#         net = global_balances[i] - INITIAL_BALANCE
#         net_str = f"+{net}" if net > 0 else f"{net}"
#         print(f"{i:<4} {role:<18} {stats['hu_count'][i]:<10} {stats['dianpao_count'][i]:<12} {stats['total_fan'][i]:<10} {global_balances[i]:<12} {net_str:<10}")

#     print("-" * 85)
#     plot_balance_curve(balance_history, player_names)

# def plot_balance_curve(history, player_names):
#     history = np.array(history)
#     episodes = range(len(history))
#     plt.figure(figsize=(12, 7))
    
#     # 🌟 修改画图逻辑，3 个模型 1 个 Bot，用不同的颜色标识
#     colors = ['#FF4444', '#FF8844', '#FFCC44', '#44AAFF']
#     styles = ['-', '-', '-', '--']
#     widths = [2.5, 2.5, 2.5, 1.5]
#     markers = ['o', 's', '^', '']
    
#     for i in range(4):
#         plt.plot(episodes, history[:, i], label=player_names[i], 
#                  color=colors[i], linewidth=widths[i], linestyle=styles[i], 
#                  marker=markers[i], markersize=4)

#     plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle=':', alpha=0.5)
#     plt.title("3x API Models vs 1x Bot Evaluation Result")
#     plt.xlabel("Episodes")
#     plt.ylabel("Balance")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     filename = "api_llm_3v1_result.png"
#     plt.savefig(filename, dpi=150)
#     print(f"\n📈 趋势图已保存: {filename}")

# if __name__ == "__main__":
#     run_evaluation()


# bot vs 3 local llm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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

# 🌟 导入本地的 LLM Agent (挂载了你训练好的模型)
try:
    from local_llm_agent import LocalLLMAgent
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️ Warning: local_llm_agent.py 未找到，将使用 Bot 替代 LLM 进行评估。")

# ================= 🔧 评估配置 =================
NUM_EPISODES = 300       # 本地推理速度较快，可以设为 200~500 局
INITIAL_BALANCE = 10000  # 初始资金
VERBOSE = False          # 关闭/开启详细模式(上帝视角)
# ==============================================


@contextlib.contextmanager
def suppress_stdout():
    """详细模式下不屏蔽输出，否则屏蔽"""
    if VERBOSE: yield
    else:
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try: yield
            finally: sys.stdout = old_stdout

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

基于以上信息，为了最大化收益，请给出最佳决策（只输出动作指令）：
"""
    return prompt.strip()

# ============================================================
# 🛠️ 可视化打印函数
# ============================================================
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

def print_settlement(game, start_balances):
    if not VERBOSE: return
    
    print("\n" + "="*25 + " 📊 本局结算 " + "="*25)
    print(f"{'玩家':<10} {'状态':<8} {'番数':<6} {'本局盈亏':<10} {'当前资金':<10} {'最终手牌'}")
    print("-" * 80)
    
    for p in game.players:
        status = "胡牌" if p.is_hu else "未胡"
        diff = p.balance - start_balances[p.player_id]
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        hand_final = " ".join([str(t) for t in p.hand_tiles])
        
        print(f"P{p.player_id:<4} {p.name:<15} {status:<6} {p.hu_fan:<6} {diff_str:<10} {p.balance:<10} {hand_final}")
    print("=" * 80 + "\n")

# ============================================================
# 🚀 主评估逻辑
# ============================================================
def run_evaluation():
    global HAS_LLM 

    print(f"\n🚀 开始评估: 3x 本地大模型 (Local LLM) vs 1x 规则Bot")
    print(f"   - 局数: {NUM_EPISODES}")
    print(f"   - 详细模式: {'✅ 开启' if VERBOSE else '❌ 关闭'}")
    
    llm_agent = None
    if HAS_LLM:
        try:
            print(f"⏳ 正在初始化本地 LLM Agent (只需加载一次权重，三家共用)...")
            llm_agent = LocalLLMAgent()
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型初始化失败: {e}")
            HAS_LLM = False

    # 🌟 核心修改：3个本地大模型，1个Bot
    player_names = ["Local-LLM-A", "Bot-1", "Bot-2", "Bot-3"]
    bots_config = [False, True, True, True] 
    
    global_balances = [INITIAL_BALANCE] * 4
    balance_history = [[INITIAL_BALANCE] * 4] 
    
    stats = { "hu_count": [0] * 4, "total_fan": [0] * 4, "dianpao_count": [0] * 4 }

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
                if not drawn: game.check_game_over(); break
            else: skip_draw = False

            print_game_snapshot(game, pid, drawn)

            turn_end = False
            loop_attempts = 0 

            while not turn_end:
                loop_attempts += 1
                action = ""
                force_bot = loop_attempts > 3 
                
                # === 决策 ===
                # 🌟 只要不是 bot，就调用本地大模型的 decide 方法
                if HAS_LLM and not bots_config[pid] and not force_bot:
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
                            print(f"🤖 [P{pid} 本地模型思考] (可选: {len(valid_actions)}个动作)...")
                            action = llm_agent.decide(prompt)
                        else:
                            with suppress_stdout():
                                action = llm_agent.decide(prompt)
                                
                        if action not in valid_actions:
                            found = False
                            for va in valid_actions:
                                if va in action: action = va; found = True; break
                            if not found and valid_actions: action = valid_actions[0]

                    except Exception as e:
                        if VERBOSE: print(f"❌ 本地大模型出错: {e}")
                        action = bot_decide_turn_action(player, game)
                else:
                    action = bot_decide_turn_action(player, game)
                    if VERBOSE and bots_config[pid]:
                        print(f"🤖 [Bot P{pid}]: {action}")

                # === 执行动作 ===
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
                                
                                # 响应决策
                                # 🌟 只要不是 bot，就让本地大模型去思考响应
                                if HAS_LLM and not bots_config[r_id]:
                                    try:
                                        valid_resps = ['n']
                                        if 'hu' in acts: valid_resps.append('h')
                                        if 'gang' in acts: valid_resps.append('g')
                                        if 'peng' in acts: valid_resps.append('p')
                                        
                                        prompt = build_observation_prompt(game, r_id, valid_resps)
                                        prompt += f"\n【突发事件】\n对手 P{pid} 打出了 {t}，触发响应机会。"

                                        if VERBOSE:
                                            print(f"⚡ [P{r_id} 本地模型响应思考] 针对 {t}...")
                                            choice = llm_agent.decide(prompt)
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
                                
                                # 执行响应
                                if choice == 'h' and 'hu' in acts:
                                    if VERBOSE: print(f"🎉 P{r_id} 食胡！点炮者: P{pid}")
                                    game.hu(r_id, t, False, pid)
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
        print_settlement(game, round_start_balances)
        
        for i, p in enumerate(game.players):
            global_balances[i] = p.balance
            if p.is_hu:
                stats["hu_count"][i] += 1
                stats["total_fan"][i] += p.hu_fan
        
        balance_history.append(global_balances.copy())

    print("\n" + "="*85)
    print(f"📊 最终战报 (共{NUM_EPISODES}局)")
    print("="*85)
    print(f"{'ID':<4} {'Role':<18} {'Wins(胡)':<10} {'DianPao(炮)':<12} {'TotalFan':<10} {'Balance':<12} {'Net':<10}")
    print("-" * 85)
    
    for i in range(4):
        role = player_names[i]
        net = global_balances[i] - INITIAL_BALANCE
        net_str = f"+{net}" if net > 0 else f"{net}"
        print(f"{i:<4} {role:<18} {stats['hu_count'][i]:<10} {stats['dianpao_count'][i]:<12} {stats['total_fan'][i]:<10} {global_balances[i]:<12} {net_str:<10}")

    print("-" * 85)
    plot_balance_curve(balance_history, player_names)

def plot_balance_curve(history, player_names):
    history = np.array(history)
    episodes = range(len(history))
    plt.figure(figsize=(12, 7))
    
    # 用不同的颜色标识三个大模型和一个普通Bot
    colors = ['#FF4444', '#FF8844', '#FFCC44', '#44AAFF']
    styles = ['-', '-', '-', '--']
    widths = [2.5, 2.5, 2.5, 1.5]
    markers = ['o', 's', '^', '']
    
    for i in range(4):
        plt.plot(episodes, history[:, i], label=player_names[i], 
                 color=colors[i], linewidth=widths[i], linestyle=styles[i], 
                 marker=markers[i], markersize=4)

    plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle=':', alpha=0.5)
    plt.title("3x Local LLM Models vs 1x Bot Evaluation Result")
    plt.xlabel("Episodes")
    plt.ylabel("Balance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = "local_llm_3v1_result.png"
    plt.savefig(filename, dpi=150)
    print(f"\n📈 趋势图已保存: {filename}")

if __name__ == "__main__":
    run_evaluation()

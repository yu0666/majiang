import json
import os
import random
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

# 导入游戏核心模块
from game import (
    MahjongGame, 
    bot_decide_exchange, 
    bot_decide_missing_suit, 
    parse_console_tile
)

# ================= 🏆 Self-Play 配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")
OUTPUT_FILE = "sft_data_selfplay_v3.jsonl" 

NUM_GAMES = 2000         
TEMPERATURE = 0.7        
TOP_P = 0.9              
# ====================================================

# === 1. 局势分析与 Prompt 构建 ===
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

# def build_observation_prompt(game, player_id, valid_actions: list = None):
#     player = game.players[player_id]
#     history_raw = game.get_history_text(k=15)
#     risk_context = get_risk_analysis(game, player_id)
#     hand_str = " ".join([str(t) for t in player.hand_tiles])
#     melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
#     missing = player.missing_suit.value if player.missing_suit else "未定"
#     tiles_left = game.deck.remaining_count()
#     valid_str = ", ".join(valid_actions) if valid_actions else "无限制"

#     prompt = f"""
# 【战局记忆】
# {history_raw}

# 【局势分析】
# 剩余牌数: {tiles_left}
# 对手状态: {risk_context}

# 【当前视角】
# 我是 P{player_id}
# 我的定缺: {missing}
# 我的副露: {melds_str}
# 我的手牌: {hand_str}

# 【决策空间】
# 合法动作: {valid_str}

# 基于以上信息，为了最大化收益，请给出最佳决策（只输出动作指令）：
# """.strip()
#     return prompt

def build_observation_prompt(game, player_id, valid_actions: list = None):
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
""".strip()
    return prompt

# === 2. 加载模型引擎 ===
print(f"🚀 正在加载 AI 引擎: {MODEL_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True, 
)
FastLanguageModel.for_inference(model)
print("✅ AI 引擎已就绪！")

def llm_decide(prompt):
    messages = [
        # {"role": "system", "content": "你是一个四川麻将高手。你会根据局势分析风险，并严格遵守合法动作规则，以最大化收益为目标。"},
        {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最快胡牌的最佳动作指令。"},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    # 【修复 1】生成并传入 Attention Mask，消除控制台黄字警告
    attention_mask = torch.ones_like(inputs)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs, 
            attention_mask=attention_mask,
            max_new_tokens=10, 
            temperature=TEMPERATURE,  
            top_p=TOP_P,
            do_sample=True,           
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id
        )
    resp = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
    return resp

# === 3. 主循环生成数据 ===
def run_self_play():
    total_saved = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for game_idx in tqdm(range(NUM_GAMES), desc="AI 左右互搏中"):
            game = MahjongGame(f"SP_{game_idx}", ["AI-0", "AI-1", "AI-2", "AI-3"], bots=[False]*4)
            start_balances = [p.balance for p in game.players]
            game.start_game()
            
            buffer_data = {0: [], 1: [], 2: [], 3: []} 
            is_trajectory_clean = {0: True, 1: True, 2: True, 3: True} 
            
            game.phase = game.phase.EXCHANGE
            for p in game.players: game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))
            game._execute_exchange()
            game.phase = game.phase.CHOOSE_MISSING
            for p in game.players: game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))
            game.phase = game.phase.PLAYING
            
            skip_draw = True
            
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

                # 【1. 主动出牌决策】
                valid_actions = []
                if player.can_hu(): valid_actions.append("h")
                g_info = game.can_self_gang(pid)
                if g_info['can_gang']: valid_actions.append("g")
                
                has_mis = any(t.suit == player.missing_suit for t in player.hand_tiles)
                seen = set()
                for t in player.hand_tiles:
                    if has_mis and t.suit != player.missing_suit: continue
                    if str(t) not in seen: valid_actions.append(f"d {t}"); seen.add(str(t))

                # 【修复 2】极端情况保护：如果引擎由于某种Bug判定无牌可打，直接跳过，防止 random.choice 报错崩溃
                if not valid_actions:
                    is_trajectory_clean[pid] = False
                    game.next_player()
                    skip_draw = False
                    continue

                prompt = build_observation_prompt(game, pid, valid_actions)
                action = llm_decide(prompt)
                
                buffer_data[pid].append({
                    "messages": [
                        # {"role": "system", "content": "你是一个四川麻将高手。你会根据局势分析风险，并严格遵守合法动作规则，以最大化收益为目标。"},
                        {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最快胡牌的最佳动作指令。"},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": action}
                    ]
                })

                # 幻觉校验
                if action not in valid_actions:
                    is_trajectory_clean[pid] = False 
                    found = False
                    for va in valid_actions:
                        if va in action: action = va; found = True; break
                    if not found: action = random.choice(valid_actions) # 因为上面加了保护，这里绝不会报错了

                # 执行动作
                turn_end = False
                if action == 'h':
                    game.hu(pid, drawn if drawn else player.hand_tiles[-1], True)
                    if game.check_game_over(True): break
                    game.next_player(); skip_draw = False; turn_end = True
                elif action == 'g':
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
                                
                                # 【2. 被动响应决策】
                                valid_resps = ['n']
                                if 'hu' in acts: valid_resps.append('h')
                                if 'gang' in acts: valid_resps.append('g')
                                if 'peng' in acts: valid_resps.append('p')
                                
                                r_prompt = build_observation_prompt(game, r_pid, valid_resps)
                                r_prompt += f"\n【突发事件】\n对手 P{pid} 打出了 {t}，触发响应机会。"
                                r_action = llm_decide(r_prompt)
                                
                                buffer_data[r_pid].append({
                                    "messages": [
                                        # {"role": "system", "content": "你是一个四川麻将高手。你会根据局势分析风险，并严格遵守合法动作规则，以最大化收益为目标。"},
                                        {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最快胡牌的最佳动作指令。"},
                                        {"role": "user", "content": r_prompt},
                                        {"role": "assistant", "content": r_action}
                                    ]
                                })

                                if r_action not in valid_resps:
                                    is_trajectory_clean[r_pid] = False 
                                    if 'h' in r_action and 'h' in valid_resps: r_action = 'h'
                                    elif 'p' in r_action and 'p' in valid_resps: r_action = 'p'
                                    elif 'g' in r_action and 'g' in valid_resps: r_action = 'g'
                                    else: r_action = 'n'

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
                    else: 
                        turn_end = True
                else:
                    # 【修复 3】终极兜底防死循环
                    turn_end = True
                    game.next_player()
                    skip_draw = False

            # === 游戏结束，RFT 严格筛选逻辑 ===
            for p in game.players:
                net_income = p.balance - start_balances[p.player_id]
                if p.is_hu and net_income > 0 and is_trajectory_clean[p.player_id]:
                    for rec in buffer_data[p.player_id]:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_saved += 1

    print(f"\n✅ Self-Play 高质量轨迹提炼完毕！")
    print(f"   - 输出文件: {OUTPUT_FILE}")
    print(f"   - 提炼出的专家决策数: {total_saved}")

if __name__ == "__main__":
    run_self_play()
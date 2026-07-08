# ==========================================
# ⚠️ 必须放在最前面！Unsloth 内核优化补丁
# ==========================================
from unsloth import FastLanguageModel 
import os
import torch
import numpy as np
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from tqdm import tqdm

# 引入游戏逻辑 (确保 game.py 在同级目录)
from game import (
    MahjongGame, 
    bot_decide_exchange, 
    bot_decide_missing_suit, 
    bot_decide_turn_action, 
    bot_decide_response, 
    parse_console_tile
)

# # ================= 🚀 Unsloth 极速训练配置 =================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen", "Qwen2___5-1___5B-Instruct")
# DATA_FILE = "sft_data_elite.jsonl"   
# OUTPUT_DIR = "qwen-sft-elite-earlystop2" 

# # ⚡ 4090 优化参数 ⚡
# MAX_SEQ_LENGTH = 2048 
# BATCH_SIZE = 32       
# GRADIENT_ACCUMULATION = 2
# NUM_EPOCHS = 3
# LEARNING_RATE = 2e-4  

# # 🛑 早停配置
# EVAL_STEPS = 1000      # 每训练 200 步评估一次
# EVAL_GAMES = 30       # 每次评估打 30 局 (数量少点为了速度)
# PATIENCE = 3          # 容忍度：连续 3 次没创新高就停止
# # =========================================================

# ================= 🚀 Unsloth 极速训练配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 【关键修改】基座模型换成你刚才合并出来的 V1 版本！
# 这意味着它是“站在巨人的肩膀上”继续学，而不是从零开始
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V1-Merged")

# 2. 【关键修改】数据换成刚才生成的自我博弈数据
DATA_FILE = "sft_data_selfplay_v1.jsonl"   

# 3. 【关键修改】输出文件夹换一个新名字
OUTPUT_DIR = "qwen-sft-selfplay-v3" 

# ⚡ 4090 优化参数 ⚡
MAX_SEQ_LENGTH = 2048 
BATCH_SIZE = 16       
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4  # 稍微调小一点，因为是在已有基础上微调，防遗忘

# 4. 【关键修改】因为数据量只有 6000 多条，参数需要适配
NUM_EPOCHS = 5        # 数据变少了，多看几遍加深印象
EVAL_STEPS = 1000       # 数据少，总步数大概才几百步，缩短到每 50 步考试一次
EVAL_GAMES = 30       # 每次评估打 30 局
PATIENCE = 3          # 容忍度：连续 3 次没创新高就停止
# =========================================================


# === 复制评估用的 Prompt 构建逻辑 (保持一致性) ===
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

def build_prompt(game, player_id, valid_actions):
    player = game.players[player_id]
    history_raw = game.get_history_text(k=15)
    risk_context = get_risk_analysis(game, player_id)
    hand_str = " ".join([str(t) for t in player.hand_tiles])
    melds_str = " ".join([f"[{str(m[0])}x{len(m)}]" for m in player.open_melds]) if player.open_melds else "无"
    missing = player.missing_suit.value if player.missing_suit else "未定"
    valid_str = ", ".join(valid_actions) if valid_actions else "无限制"

    return f"""
【战局记忆】
{history_raw}

【局势分析】
剩余牌数: {game.deck.remaining_count()}
对手状态: {risk_context}

【当前视角】
我是 P{player_id}
我的定缺: {missing}
我的副露: {melds_str}
我的手牌: {hand_str}

【决策空间】
合法动作: {valid_str}

基于以上信息，为了最大化收益，请给出最佳决策（只输出动作指令）：
""".strip()

# === 定义自定义回调函数 ===
class MahjongEarlyStoppingCallback(TrainerCallback):
    def __init__(self, tokenizer, output_dir):
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.best_win_rate = -1.0
        self.no_improve_count = 0
        
    def on_step_end(self, args, state, control, model, **kwargs):
        # 每 EVAL_STEPS 步触发一次，且跳过前 50 步
        if state.global_step > 50 and state.global_step % EVAL_STEPS == 0:
            print(f"\n\n🛑 [早停检查] 正在进行期中考试 (Step {state.global_step})...")
            
            # 1. 切换到推理模式 (Unsloth 加速)
            FastLanguageModel.for_inference(model)
            
            # 2. 跑评估
            win_rate = self.evaluate_win_rate(model)
            print(f"📊 当前胡牌率: {win_rate:.1f}% (历史最佳: {self.best_win_rate:.1f}%)")
            
            # 3. 比较与保存
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.no_improve_count = 0
                
                save_path = os.path.join(self.output_dir, "best_model_hu_rate")
                print(f"🔥 创新高！保存最佳模型到: {save_path}")
                model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
            else:
                self.no_improve_count += 1
                print(f"❄️ 未创新高 ({self.no_improve_count}/{PATIENCE})")
            
            # 4. 判断是否早停
            if self.no_improve_count >= PATIENCE:
                print(f"🛑 触发早停机制！训练结束。")
                control.should_training_stop = True
            
            # 5. 切回训练模式
            FastLanguageModel.for_training(model)
            print("🚀 继续训练...\n")

    def evaluate_win_rate(self, model):
        """内置的一个迷你评估循环"""
        wins = 0
        pbar = tqdm(range(EVAL_GAMES), desc="评估中", leave=False)
        
        for _ in pbar:
            game = MahjongGame("EVAL_INTERNAL", ["AI", "B1", "B2", "B3"], bots=[False, True, True, True])
            game.start_game()
            
            # 快速开局
            game.phase = game.phase.EXCHANGE
            for p in game.players: game.select_exchange_tiles(p.player_id, bot_decide_exchange(p))
            game.phase = game.phase.CHOOSE_MISSING
            for p in game.players: game.set_missing_suit(p.player_id, bot_decide_missing_suit(p))
            game.phase = game.phase.PLAYING
            
            skip = True
            while not game.is_game_over:
                if sum(1 for p in game.players if p.is_hu) >= 3: break
                pid = game.current_player_id
                player = game.players[pid]
                if player.is_hu: 
                    game.next_player(); skip = False; continue
                
                if not skip: 
                    if not game.draw_tile(pid): break
                else: skip = False

                if pid == 0: # AI 回合
                    # 1. Action Masking
                    valid = []
                    if player.can_hu(): valid.append("h")
                    if game.can_self_gang(pid)['can_gang']: valid.append("g")
                    has_mis = any(t.suit == player.missing_suit for t in player.hand_tiles)
                    seen = set()
                    for t in player.hand_tiles:
                        if has_mis and t.suit != player.missing_suit: continue
                        if str(t) not in seen: valid.append(f"d {t}"); seen.add(str(t))
                    
                    # 2. 推理
                    prompt = build_prompt(game, pid, valid)
                    inputs = self.tokenizer.apply_chat_template([
                        {"role": "system", "content": "你是一个四川麻将高手。"},
                        {"role": "user", "content": prompt}
                    ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
                    
                    with torch.no_grad():
                        out = model.generate(input_ids=inputs, max_new_tokens=10, temperature=0.1)
                    resp = self.tokenizer.decode(out[0][len(inputs[0]):], skip_special_tokens=True).strip()
                    
                    # 3. 修正
                    action = valid[0]
                    for v in valid:
                        if v in resp: action = v; break
                else:
                    action = bot_decide_turn_action(player, game)

                # 执行 (简化版，只处理打牌和胡)
                if action == 'h': 
                    game.hu(pid, player.hand_tiles[-1], True)
                elif action.startswith('d '):
                    t = parse_console_tile(action[2:])
                    if game.discard_tile(pid, t):
                        game.next_player(); skip = False

            if game.players[0].is_hu:
                wins += 1
                
        return (wins / EVAL_GAMES) * 100

def train():
    print(f"🚀 启动 Unsloth 极速训练 (含胡牌率早停机制)...")
    
    # 1. 加载模型 
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, 
        load_in_4bit = True, 
    )

    # 2. LoRA 配置
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 3. 数据处理
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    
    def formatting_prompts_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        warmup_steps = 100,
        num_train_epochs = NUM_EPOCHS,
        learning_rate = LEARNING_RATE,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 20,
        save_steps = 100, # 常规保存
        optim = "adamw_8bit", 
        weight_decay = 0.01,
        seed = 3407,
        report_to = "tensorboard",
    )

    # 5. 注入回调
    # 初始化 Early Stopping Callback
    early_stop_callback = MahjongEarlyStoppingCallback(tokenizer, OUTPUT_DIR)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",     
        max_seq_length = MAX_SEQ_LENGTH, 
        dataset_num_proc = 8,            
        args = training_args,
        packing = True, 
        callbacks = [early_stop_callback] # 【关键】加入回调
    )

    print(f"🔥 训练开始！(每 {EVAL_STEPS} 步评估一次胡牌率)")
    trainer.train()
    
    # 训练结束后，确保最佳模型被保留（如果没触发早停）
    print(f"💾 训练结束。最佳模型保存在: {os.path.join(OUTPUT_DIR, 'best_model_hu_rate')}")

if __name__ == "__main__":
    train()
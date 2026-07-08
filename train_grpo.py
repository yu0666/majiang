import os
import re
import json
import torch
import multiprocessing
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOConfig, GRPOTrainer

# ================= 🔧 GRPO 训练配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")
DATA_FILE = "sft_data_selfplay_v3.jsonl"
OUTPUT_DIR = "qwen-grpo-fast-hu-expert6" # 专家级稠密奖励版

MAX_SEQ_LENGTH = 2048
NUM_GENERATIONS = 4 
NUM_PROC = max(1, multiprocessing.cpu_count() - 2)
# ====================================================

# ================= 🧠 核心：盘面透视与向听分析引擎 =================

def extract_text(item):
    if isinstance(item, list) and len(item) > 0 and isinstance(item[-1], dict):
        return item[-1].get("content", "")
    return str(item)

def parse_tile(tile_str):
    """解析牌的数字和花色，例如 '3万' -> (3, '万')"""
    match = re.match(r"([1-9])([万筒条])", tile_str)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

def evaluate_tactics(played_tile_str, hand_tiles_str, prompt_text):
    """
    专家级战术评估引擎 (向听数、死叫、听牌质量)
    """
    played_val, played_suit = parse_tile(played_tile_str)
    if not played_val: return 0.0

    # 1. 提取手里同花色的所有数字
    hand_tiles = hand_tiles_str.split()
    same_suit_vals = []
    for t in hand_tiles:
        v, s = parse_tile(t)
        if s == played_suit and v is not None:
            same_suit_vals.append(v)
            
    # 从手里模拟移除这张刚打出的牌
    if played_val in same_suit_vals:
        same_suit_vals.remove(played_val)

    # 2. 寻找与这张牌相关的“搭子” (进张目标)
    targets = set()
    if played_val in same_suit_vals: # 这是一个对子
        targets.add(played_val)
    if played_val - 1 in same_suit_vals: # 有 4, 打 5 -> 进张 3, 6
        if played_val - 2 >= 1: targets.add(played_val - 2)
        if played_val + 1 <= 9: targets.add(played_val + 1)
    if played_val + 1 in same_suit_vals: # 有 6, 打 5 -> 进张 4, 7
        if played_val - 1 >= 1: targets.add(played_val - 1)
        if played_val + 2 <= 9: targets.add(played_val + 2)
    if played_val - 2 in same_suit_vals: # 有 3, 打 5 -> 嵌张 4
        targets.add(played_val - 1)
    if played_val + 2 in same_suit_vals: # 有 7, 打 5 -> 嵌张 6
        targets.add(played_val + 1)

    # 3. 孤张判定 (向听数逻辑)
    is_isolated = len(targets) == 0

    # 4. 死叫 / 绝张判定 (极其关键)
    is_dead_wait = False
    visible_targets_count = 0
    if not is_isolated:
        # 如果不是孤张，说明打掉它就是“拆搭子”。我们看看这个搭子要进的牌是不是已经死光了。
        total_needed = len(targets) * 4 # 每种牌最多4张
        total_visible = 0
        for t_val in targets:
            target_str = f"{t_val}{played_suit}"
            # 全局视野：直接扫描历史记录和副露中出现了几次这张牌！
            total_visible += prompt_text.count(target_str)
        
        visible_targets_count = total_visible
        # 如果需要的牌在场上已经出现了 >= 3张 (几乎绝张)，判定为死搭子/死叫
        if total_visible >= (len(targets) * 4) - 1: 
            is_dead_wait = True

    # ================= 打分逻辑 =================
    try:
        if is_isolated:
            # 【逻辑 2】打出无用孤张，向听数减少，离听牌更近
            return 8.0 
        elif is_dead_wait:
            # 【逻辑 3】果断拆掉死叫/死搭子，尝试换叫，巨额奖励！
            return 15.0 
        else:
            # 【逻辑 3 反面】拆散了还能进张的活搭子，导致向听数倒退，严厉惩罚！
            # 【逻辑 4】听牌质量：活搭子如果外面出现的少，说明极其优质，拆了扣分更重！
            quality_penalty = (4 * len(targets) - visible_targets_count) * 2.0
            return -10.0 - quality_penalty
    except:
        return 0.0

# ================= 👨‍⚖️ 裁判：规则奖励函数 =================

def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        action = extract_text(completion).strip()
        if re.match(r"^(d [1-9][万筒条]|h|g|p|n)$", action):
            rewards.append(2.0)  
        else:
            rewards.append(-5.0) 
    return rewards

def expert_mahjong_reward_func(prompts, completions, **kwargs):
    """融合了四大铁律的专家级打分器"""
    rewards = []
    for prompt_msgs, completion in zip(prompts, completions):
        prompt_text = extract_text(prompt_msgs)
        action = extract_text(completion).strip()
        reward = 0.0
        
        if action == "h":
            reward += 50.0  # 胡牌绝对正义，给最高分
        elif action in ["p", "g"]:
            reward += 3.0   # 鼓励吃碰杠加速
            
        elif action.startswith("d "):
            played_tile = action.split(" ")[1] if len(action.split(" ")) > 1 else ""
            played_val, played_suit = parse_tile(played_tile)
            
            missing_match = re.search(r"我的定缺:\s*([万筒条])", prompt_text)
            hand_match = re.search(r"我的手牌:\s*(.*?)(?:\n|$)", prompt_text)
            
            if missing_match and hand_match and played_suit:
                missing_suit = missing_match.group(1)
                hand_tiles_str = hand_match.group(1).strip()
                
                # 【逻辑 1】绝对红线：先打缺门
                has_missing_in_hand = missing_suit in hand_tiles_str
                
                if has_missing_in_hand:
                    if played_suit == missing_suit:
                        reward += 10.0  # 乖乖清缺门，重赏！
                    else:
                        reward -= 30.0  # 致命错误！有缺不打，给一个毁灭性的惩罚，让它长记性！
                else:
                    # 缺门清完后，启动高级盘面分析（向听数、死叫、听牌质量）
                    tactical_score = evaluate_tactics(played_tile, hand_tiles_str, prompt_text)
                    reward += tactical_score
                    
                    # 检查是否保留了死叫（如果打出了孤张，但手里还抱着死搭子，轻微扣分）
                    # 这是一个近似逻辑，防止它一直抱着死叫不放
                    if tactical_score > 0 and played_tile in prompt_text:
                         if prompt_text.count(played_tile) >= 3:
                             reward += 5.0 # 打出的刚好也是场上不要的死牌，安全！
                             
        rewards.append(reward)
    return rewards

# ================= 🚀 训练主流程 =================

def load_prompts_dataset():
    print(f"🔄 正在启动 {NUM_PROC} 核心分布式数据处理...")
    prompts_list = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            messages = data["messages"][:2] 
            prompts_list.append({"prompt": messages})
            if len(prompts_list) >= 3000: 
                break
    return Dataset.from_list(prompts_list)

def train():
    print(f"🚀 启动分布式 GRPO [专家稠密逻辑战神版]...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_PATH,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = True, 
        fast_inference = False, 
        max_lora_rank = 16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    dataset = load_prompts_dataset()
    print(f"📚 成功构建题库。")

    training_args = GRPOConfig(
        use_vllm = False, 
        output_dir = OUTPUT_DIR,
        learning_rate = 1.5e-5, 
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_prompt_length = int(MAX_SEQ_LENGTH * 0.9),
        max_completion_length = 20, 
        num_generations = NUM_GENERATIONS, 
        max_steps = 1000, 
        
        # 🌟 多进程分布式配置
        dataloader_num_workers = NUM_PROC, 
        dataloader_prefetch_factor = 2,    
        
        save_strategy = "steps",   
        save_steps = 100,           
        save_total_limit = 20,      
        
        logging_steps = 5,
        report_to = "none",
    )

    trainer = GRPOTrainer(
        model = model,
        reward_funcs = [format_reward_func, expert_mahjong_reward_func],
        args = training_args,
        train_dataset = dataset,
    )

    print("🔥 开始专家级强化学习训练！请紧盯 rewards 增长...")
    trainer.train()
    
    model.save_pretrained(os.path.join(OUTPUT_DIR, "best_grpo_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_grpo_adapter"))
    print("✅ GRPO 训练完成！具备大局观的【终极麻将战神】诞生！")

if __name__ == "__main__":
    train()
import os
import json
import re
import torch
import torch.nn.functional as F
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= 🔧 训练配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")
DATASET_PATH = os.path.join(BASE_DIR, "sft_data_selfplay_v1.jsonl") 

G = 4            
LR = 1.5e-5      # 稍微调高学习率，强化记忆
BETA = 0.02      
EPSILON = 0.2
# ===============================================

class OfflineGRPOTrainer:
    def __init__(self):
        print(f"🚀 正在加载 GRPO 离线强化学习引擎 (纯血大牌战神 2.0)...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, 
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=LR)
        
        print("✅ 引擎就绪！极端奖励法案已颁布！")

    def _get_reward(self, action_text: str, prompt: str) -> float:
        """🌟 核心：极端大牌惩罚/奖励机制 🌟"""
        action = action_text.strip().lower()
        if not action: return -10.0 

        que_suit = ""
        que_match = re.search(r"(?:缺:|定缺[:：]|缺)\s*([万筒条wtd])", prompt)
        if que_match:
            que_suit = que_match.group(1).replace('万','w').replace('筒','t').replace('条','d')

        match = re.search(r"手牌[:：]\s*(.+?)(?:\||\n|$)", prompt)
        if not match: return 0.0
        raw_hand = match.group(1).replace('万', 'w').replace('筒', 't').replace('条', 'd')
        hand_tiles = re.findall(r'[1-9][wtd]', raw_hand)
        
        if not hand_tiles: return 0.0

        if action == 'h': return 50.0  
        if action == 'g': return 10.0  
        if action == 'p': return 5.0   
        if action in ['n', 'pass']: return 0.0

        if action.startswith('d '):
            discard_str = action.split(' ')[-1].replace('万','w').replace('筒','t').replace('条','d')
            if len(discard_str) != 2 or discard_str not in hand_tiles:
                return -20.0 # 幻觉重罚
                
            discard_val, discard_suit = discard_str[0], discard_str[1]
            
            # 【铁律 1】：必须打缺门！
            has_que = any(t[1] == que_suit for t in hand_tiles)
            if has_que:
                if discard_suit == que_suit: return 20.0 
                else: return -50.0 # 极刑：有缺不打！

            # 【铁律 2】：清一色强制引导
            suit_counts = {'w': 0, 't': 0, 'd': 0}
            for t in hand_tiles: suit_counts[t[1]] += 1
            dominant_suit = max(suit_counts, key=suit_counts.get) 
            
            reward = 0.0
            
            # 如果手里还有杂色牌
            if suit_counts[dominant_suit] < len(hand_tiles):
                if discard_suit != dominant_suit:
                    reward += 15.0 # 奖励清杂牌
                else:
                    reward -= 40.0 # 极刑：敢扔主花色！
            else:
                reward += 5.0
                
            # 【铁律 3】：保护对子
            count_in_hand = hand_tiles.count(discard_str)
            if count_in_hand >= 2:
                reward -= 15.0 # 严惩拆对子
                
            return reward

        return -5.0

    def train_on_dataset(self):
        if not os.path.exists(DATASET_PATH): return

        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"📚 开始训练，共 {len(lines)} 条离线数据...")
        pbar = tqdm(enumerate(lines), total=len(lines), desc="🔥 训练大牌流", dynamic_ncols=True)
        
        for step, line in pbar:
            try:
                data = json.loads(line)
                prompt = data['instruction'] if 'instruction' in data else data['messages'][0]['content']
            except:
                continue

            self.optimizer.zero_grad()
            
            # =========================================================
            # 🌟 核心修改 1：阶段性保存前置！绝对不会被 skip 吞掉！
            # =========================================================
            if step > 0 and step % 1000 == 0:
                save_dir = os.path.join(BASE_DIR, "qwen-grpo-da-hu-expert", f"checkpoint-{step}")
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                tqdm.write(f"💾 [Step {step}] 检查点已安全保存至: {save_dir}")

            # =========================================================
            # 🌟 核心修改 2：使用和 evaluate_llm 一模一样的提示词
            # =========================================================
            messages = [{"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最快胡牌的最佳动作指令。"},
                        {"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            prompt_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                # 🌟 核心修改 3：暴力加温，强制探索不同的动作，拒绝 0 梯度！
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    temperature=1.5,   # 原来是 1.2
                    top_p=0.9,         # 添加核采样，保持动作的合理性
                    do_sample=True,
                    num_return_sequences=G,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            rewards = []
            for i in range(G):
                gen_ids = outputs[i][prompt_length:]
                action_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                rewards.append(self._get_reward(action_text, prompt))
                
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.model.device)

            if rewards_tensor.std() > 0:
                advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            else:
                advantages = torch.zeros_like(rewards_tensor)
            
            if (advantages == 0).all():
                pbar.set_postfix({"Status": "跳过 (无优势)"})
                continue

            outputs_forward = self.model(outputs)
            logits = outputs_forward.logits[:, prompt_length-1:-1, :] 
            target_ids = outputs[:, prompt_length:]
            log_probs = torch.gather(F.log_softmax(logits, dim=-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
            seq_log_probs = log_probs.sum(dim=-1) 

            with torch.no_grad():
                with self.model.disable_adapter():
                    ref_outputs = self.model(outputs)
                    ref_logits = ref_outputs.logits[:, prompt_length-1:-1, :]
                    ref_log_probs = torch.gather(F.log_softmax(ref_logits, dim=-1), 2, target_ids.unsqueeze(-1)).squeeze(-1)
                    ref_seq_log_probs = ref_log_probs.sum(dim=-1)

            ratio = torch.exp(seq_log_probs - ref_seq_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * advantages
            kl_div = ref_seq_log_probs.detach() - seq_log_probs
            loss = -torch.mean(torch.min(surr1, surr2) - BETA * kl_div)

            loss.backward()
            self.optimizer.step()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "KL": f"{kl_div.mean().item():.4f}"})

        # 循环结束的最终保存
        final_save_dir = os.path.join(BASE_DIR, "qwen-grpo-da-hu-expert", "final_model")
        self.model.save_pretrained(final_save_dir)
        self.tokenizer.save_pretrained(final_save_dir)
        print(f"\n🎉 训练全部结束！最终战神适配器已保存至: {final_save_dir}")

if __name__ == "__main__":
    trainer = OfflineGRPOTrainer()
    trainer.train_on_dataset()
# import os
# import torch
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel  # 【关键】引入 PEFT 用来加载适配器

# # # ================= 🔧 模型配置 =================
# # # 1. 基础模型路径 (和训练时保持一致)
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "qwen", "Qwen2___5-1___5B-Instruct")

# # # 2. 适配器路径 (指向您的 checkpoint-60)
# # # 注意：如果您想用最后保存的完整模型，可以改为 "qwen-sft-mahjong-v1"
# # ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-sft-elite-earlystop2", "checkpoint-2500")

# import os
# # ================= 🔧 模型配置 =================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 1. 基础模型 (V3 合并版)
# BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V1-Merged")

# # 2. GRPO 适配器 (刚刚练出来的极速胡牌流)
# ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-sft-selfplay-v3", "checkpoint-500")

# # ==============================================

# class LocalLLMAgent:
#     def __init__(self):
#         print(f"🚀 正在加载基座模型: {BASE_MODEL_PATH}")
#         print(f"🧩 正在挂载适配器: {ADAPTER_PATH}")

#         # 1. 量化配置 (必须与训练时一致，使用 4-bit)
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.float32,
#         )

#         # 2. 加载基座模型
#         self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
#         base_model = AutoModelForCausalLM.from_pretrained(
#             BASE_MODEL_PATH,
#             quantization_config=bnb_config,
#             device_map="auto",
#             trust_remote_code=True
#         )

#         # 3. 【核心】加载并合并 LoRA 适配器
#         # 这步会把您训练的“麻将知识”注入到模型中
#         self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#         self.model.eval() # 切换到推理模式

#         print("✅ 模型加载完毕！准备战斗！")

#     def decide(self, prompt: str) -> str:
#         # 构造对话格式
#         messages = [
#             {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最佳动作指令。"},
#             {"role": "user", "content": prompt}
#         ]
        
#         # 应用模版
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

#         # 生成回复
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens=16, # 动作通常很短，比如 "d 5w" 或 "peng"
#                 temperature=0.1,   # 低温度，让决策更稳定
#                 do_sample=True
#             )
            
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]
        
#         response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         # 清理输出，防止模型啰嗦
#         response = response.strip()
#         print(f"🤖 AI 决策: {response}")
#         return response

# if __name__ == "__main__":
#     # 简单测试
#     agent = LocalLLMAgent()
#     test_prompt = "我的手牌: 1m 2m 3m. 此时对手打出 1m, 我该碰吗?"
#     agent.decide(test_prompt)



import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 🔧 模型配置 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 【核心修改 1】直接指向合并后的新模型路径
MERGED_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V4-GRPO-Merged")
# MERGED_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")
# ==============================================

class LocalLLMAgent:
    def __init__(self):
        print(f"🚀 正在加载全新合并的大模型: {MERGED_MODEL_PATH}")

        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 【核心修改 2】直接加载合并后的全量模型
        # 因为在 4090 上 1.5B 模型非常小，直接使用 bfloat16 加载，速度会飞快且无精度损失
        self.model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        self.model.eval() # 切换到推理模式

        print("✅ 全新模型加载完毕！准备战斗！")

    def decide(self, prompt: str) -> str:
        # 构造对话格式 (保持原样不变)
        messages = [
            {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最佳动作指令。"},
            {"role": "user", "content": prompt}
        ]
        
        # 应用模版 (保持原样不变)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 生成回复 (保持原样不变)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=16, # 动作通常很短，比如 "d 5w" 或 "peng"
                temperature=0.1,   # 低温度，让决策更稳定
                do_sample=True
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return response



# import os
# import torch
# import json
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel  # 👈 重新引入 PeftModel 用来挂载 GRPO 补丁

# # ================= 🔧 模型配置 =================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 1. 基础模型 (V3 合并版)
# BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")

# # 2. GRPO 适配器 (刚刚练出来的极速胡牌流)
# ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-grpo-fast-hu-expert3", "checkpoint-800")
# # ==============================================

# class LocalLLMAgent:
#     def __init__(self):
#         print(f"🚀 正在加载基座模型: {BASE_MODEL_PATH}")

#         # 1. 加载 Tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         # 2. 加载基座模型 (强制使用 GPU)
#         base_model = AutoModelForCausalLM.from_pretrained(
#             BASE_MODEL_PATH,
#             device_map="cuda:0",  
#             torch_dtype=torch.bfloat16, 
#             trust_remote_code=True
#         )
        
#         print(f"🧩 正在挂载 GRPO 极速流补丁: {ADAPTER_PATH}")
#         # 3. 挂载 GRPO LoRA
#         self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#         self.model.eval() # 切换到推理模式

#         print("✅ 极速胡牌战神加载完毕！准备战斗！")

#     def decide(self, prompt: str) -> str:
#         # 构造对话格式
#         messages = [
#             {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最佳动作指令。"},
#             {"role": "user", "content": prompt}
#         ]
        
#         # 应用模版
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

#         # 生成回复
#         with torch.no_grad():
#             generated_ids = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens=16, 
#                 temperature=0.1,   
#                 do_sample=True
#             )
            
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]
        
#         response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
#         return response



# import os
# import re
# import torch
# import numpy as np
# import warnings
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# warnings.filterwarnings("ignore")

# try:
#     from explainable_ai import DecisionVisualizer
# except ImportError:
#     DecisionVisualizer = None

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")
# ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-grpo-fast-hu-expert2", "checkpoint-500")

# class LocalLLMAgent:
#     def __init__(self, enable_xai=False): 
#         print(f"🚀 正在加载基座模型: {BASE_MODEL_PATH}")

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             BASE_MODEL_PATH, 
#             trust_remote_code=True,
#             fix_mistral_regex=True 
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         base_model = AutoModelForCausalLM.from_pretrained(
#             BASE_MODEL_PATH,
#             device_map="cuda:0",  
#             dtype=torch.bfloat16, 
#             trust_remote_code=True,
#             attn_implementation="eager" 
#         )
        
#         print(f"🧩 正在挂载 GRPO 补丁: {ADAPTER_PATH}")
#         self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
#         self.model.eval() 

#         self.enable_xai = enable_xai
#         if self.enable_xai and DecisionVisualizer is not None:
#             self.visualizer = DecisionVisualizer()
#             print("👁️ [XAI] 终极透视引擎已开启：字节完美缝合，乱码彻底抹除！")
#         else:
#             self.visualizer = None
            
#         self.step_counter = 0

#     def decide(self, prompt: str) -> str:
#         messages = [
#             {"role": "system", "content": "你是一个四川麻将高手。根据历史记录和当前手牌，输出最快胡牌的最佳动作指令。"},
#             {"role": "user", "content": prompt}
#         ]
        
#         text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **model_inputs,
#                 max_new_tokens=16, 
#                 temperature=1.0,   
#                 do_sample=True,
#                 return_dict_in_generate=True,
#                 output_scores=self.enable_xai,       
#                 output_attentions=self.enable_xai    
#             )
            
#         generated_ids = outputs.sequences[0][len(model_inputs.input_ids[0]):]
#         response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

#         # ======== 🧠 XAI 深度提取 ========
#         if self.enable_xai and self.visualizer is not None:
#             self.step_counter += 1
            
#             try:
#                 valid_actions = []
#                 match = re.search(r"合法动作:\s*(.+)", prompt)
#                 if match:
#                     valid_actions = [a.strip() for a in match.group(1).split(',')]
                    
#                 is_discard_phase = any(a.startswith('d ') for a in valid_actions)

#                 # 锁定纠结的瞬间 (出牌找数字，响应找字母)
#                 step_to_analyze = 0
#                 if is_discard_phase:
#                     for i, tid in enumerate(generated_ids):
#                         t_str = self.tokenizer.decode([tid])
#                         if any(c.isdigit() for c in t_str):
#                             step_to_analyze = i
#                             break

#                 scores_float = outputs.scores[step_to_analyze][0].float()
#                 probs = torch.nn.functional.softmax(scores_float, dim=-1)
#                 top_k_probs, top_k_indices = torch.topk(probs, 8) 
                
#                 action_probs = {}
#                 suit_map = {"万": "w", "筒": "t", "条": "d"}
                
#                 for prob, idx in zip(top_k_probs, top_k_indices):
#                     token_str = self.tokenizer.decode([idx]).strip().lower()
#                     if not token_str or not token_str.isprintable() or prob.item() < 0.005: 
#                         continue
                    
#                     if token_str in ['p', 'g', 'h', 'n', 'pass']:
#                         beautiful_name = {'p': 'Peng', 'g': 'Gang', 'h': 'Hu', 'n': 'Pass', 'pass': 'Pass'}[token_str]
#                         action_probs[beautiful_name] = action_probs.get(beautiful_name, 0) + prob.item()
#                         continue

#                     matches = [va for va in valid_actions if token_str in va.lower()]
#                     if len(matches) == 1:
#                         act_key = matches[0]
#                         for cn_suit, en_suit in suit_map.items():
#                             act_key = act_key.replace(cn_suit, en_suit)
#                         if act_key.startswith('d '):
#                             act_key = f"Discard {act_key[2:]}"
#                         action_probs[act_key] = action_probs.get(act_key, 0) + prob.item()
#                     elif len(matches) > 1 and is_discard_phase:
#                         # 用实际 response 消歧义：找出模型真正选的那个动作
#                         resolved = None
#                         resp_lower = response.lower()
#                         for m in matches:
#                             if m.lower() in resp_lower:
#                                 resolved = m
#                                 break
#                         if resolved:
#                             # 把全部概率归给真实选中的动作
#                             act_key = resolved
#                             for cn_suit, en_suit in suit_map.items():
#                                 act_key = act_key.replace(cn_suit, en_suit)
#                             if act_key.startswith('d '):
#                                 act_key = f"Discard {act_key[2:]}"
#                             action_probs[act_key] = action_probs.get(act_key, 0) + prob.item()
#                         else:
#                             # 无法消歧义时均分概率，保证总和不超过1
#                             split_prob = prob.item() / len(matches)
#                             for m in matches:
#                                 act_key = m
#                                 for cn_suit, en_suit in suit_map.items():
#                                     act_key = act_key.replace(cn_suit, en_suit)
#                                 if act_key.startswith('d '):
#                                     act_key = f"Discard {act_key[2:]}"
#                                 action_probs[act_key] = action_probs.get(act_key, 0) + split_prob
#                     else:
#                         act_key = token_str
#                         action_probs[act_key] = action_probs.get(act_key, 0) + prob.item()
                
#                 if action_probs:
#                     action_probs = dict(sorted(action_probs.items(), key=lambda item: item[1], reverse=True))
#                     self.visualizer.plot_action_confidence(self.step_counter, action_probs)

#                 # =====================================================================
#                 # 🌟 终极修复：完美字节缝合与 100% 防 Linux 乱码引擎
#                 # =====================================================================
#                 last_layer_attention = outputs.attentions[step_to_analyze][-1].float() 
#                 avg_head_attention = last_layer_attention[0].mean(dim=0) 
                
#                 input_length = model_inputs.input_ids.shape[1]
#                 attention_to_prompt = avg_head_attention[-1, :input_length].cpu().numpy()
#                 ids = model_inputs.input_ids[0].tolist()
                
#                 # 第一阶段：用 Unicode 乱码符 '\ufffd' 精准判断字节是否闭合！
#                 decoded_chars = []
#                 decoded_atts = []
                
#                 temp_ids = []
#                 temp_weight = 0.0
                
#                 for tid, weight in zip(ids, attention_to_prompt):
#                     temp_ids.append(tid)
#                     temp_weight += weight
                    
#                     text_check = self.tokenizer.decode(temp_ids, errors='replace')
                    
#                     # 只要没有  (U+FFFD)，就说明碎片凑成了一个完整的字！
#                     if '\ufffd' not in text_check:
#                         # 1. 翻译核心花色
#                         clean = text_check.replace('万', 'w').replace('筒', 't').replace('条', 'd')
#                         # 2. 把其他无法在图表显示的中文变成 *，彻底消灭字体报错
#                         clean = "".join(c if ord(c) < 128 else '*' for c in clean)
#                         # 3. 清理换行和空格
#                         clean = clean.replace('\n', '↵').replace(' ', '').strip()
                        
#                         decoded_chars.append(clean if clean else "_")
#                         decoded_atts.append(temp_weight)
                        
#                         temp_ids = []
#                         temp_weight = 0.0
                
#                 # 第二阶段：把 "8" 和 "w" 吸附成 "8w"
#                 prompt_tokens = []
#                 final_attentions = []
                
#                 i = 0
#                 while i < len(decoded_chars):
#                     c = decoded_chars[i]
#                     w = decoded_atts[i]
                    
#                     if i + 1 < len(decoded_chars):
#                         next_c = decoded_chars[i+1]
#                         if c in '123456789' and next_c in ['w', 't', 'd']:
#                             prompt_tokens.append(c + next_c) # 合并成 8w
#                             final_attentions.append(w + decoded_atts[i+1]) # 权重相加
#                             i += 2
#                             continue
                            
#                     prompt_tokens.append(c)
#                     final_attentions.append(w)
#                     i += 1
                        
#                 final_attentions = np.array(final_attentions)
#                 self.visualizer.plot_attention_heatmap(self.step_counter, prompt_tokens, final_attentions, top_k=25)
            
#             except Exception as e:
#                 pass 

#         return response

# # 补充缺失的核心导入
# import os
# import torch
# from typing import List
# from transformers import AutoModelForCausalLM, AutoTokenizer
# # 确保游戏核心模块能正确导入
# from game import MahjongGame, PlayerState
# from rule_engine import ShantenCalculator
# import json

# # 定义基础目录（解决BASE_DIR未定义问题）
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# class LocalLLMAgent:
#     # 修复初始化参数：设置默认路径+兼容外部传入路径
#     def __init__(self, model_path=None):
#         # 优先使用外部传入的路径，否则用默认路径
#         if model_path is None:
#             model_path = os.path.join(BASE_DIR, "models", "qwen", "Qwen2___5-1___5B-Instruct")
        
#         print(f"正在加载本地模型: {model_path} ...")
        
#         # 1. 加载分词器（补充pad_token处理，避免推理报错）
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         # 修复部分Qwen模型pad_token缺失问题
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # 2. 加载模型（优化设备配置，适配4090）
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map="cuda:0" if torch.cuda.is_available() else "cpu",  # 优先用第一张GPU
#             torch_dtype=torch.float16,  # 半精度适配4090
#             trust_remote_code=True,
#             attn_implementation="eager"  # 兼容更多环境
#         ).eval()  # 评估模式，关闭dropout
        
#         print("本地模型加载完成！")

#     def get_action(self, player: PlayerState, game: MahjongGame, valid_actions: List[str]) -> str:
#         """核心决策函数：输入玩家状态+游戏状态，输出最佳动作"""
#         # 1. 牌理分析（向听数计算）
#         analysis = self._analyze_shanten(player)
        
#         # 2. 构建Prompt（适配麻将决策场景）
#         prompt = self._construct_prompt(player, game, valid_actions, analysis)
        
#         # 3. 本地模型推理
#         response = self._generate_response(prompt)
        
#         # 4. 解析响应（保证返回合法动作）
#         return self._parse_response(response, valid_actions)

#     def _generate_response(self, prompt: str) -> str:
#         """模型推理核心函数：生成决策响应"""
#         # 构建标准化Chat模板
#         messages = [
#             {"role": "system", "content": "你是四川麻将高手，精通定缺、向听数计算。仅输出动作指令（如'd 1万'/'h'/'p'/'g'/'n'），无需任何解释！"},
#             {"role": "user", "content": prompt}
#         ]
        
#         # 应用Qwen的Chat模板
#         text = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
        
#         # 分词并移到GPU
#         inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
#         # 推理（无梯度计算，提
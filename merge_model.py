# import os
# import torch
# from unsloth import FastLanguageModel

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # 你的最佳 checkpoint 路径 (请确认路径正确)
# ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-sft-elite-earlystop2", "checkpoint-3500") 
# # 合并后的新模型输出路径
# MERGED_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V1-Merged")

# def merge_and_save():
#     print(f"🚀 正在加载带有 LoRA 的模型: {ADAPTER_PATH}")
    
#     # 加载模型 (注意这里 load_in_4bit=False，合并模型需要全精度或半精度)
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = ADAPTER_PATH,
#         max_seq_length = 2048,
#         dtype = torch.bfloat16, # 使用 bfloat16 精度合并
#         load_in_4bit = False,   # 必须是 False 才能保存合并权重
#     )
    
#     print("🧩 正在将 LoRA 权重合并到基座模型中 (这可能需要几分钟)...")
#     # 保存为 HuggingFace 格式的标准模型
#     model.save_pretrained_merged(MERGED_MODEL_DIR, tokenizer, save_method = "merged_16bit")
    
#     print(f"✅ 合并完成！全新的强大基座已保存至: {MERGED_MODEL_DIR}")

# if __name__ == "__main__":
#     merge_and_save()

# import os
# import torch
# from unsloth import FastLanguageModel

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # ⚠️ 注意路径，换成你刚刚测出 +9440 的那个 V3 权重！
# ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-sft-selfplay-v3", "checkpoint-500") 
# MERGED_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")

# def merge_and_save():
#     print(f"🚀 正在加载带有 LoRA 的 V3 模型: {ADAPTER_PATH}")
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name = ADAPTER_PATH,
#         max_seq_length = 2048,
#         dtype = torch.bfloat16,
#         load_in_4bit = False,
#     )
    
#     print("🧩 正在将 V3 融合为独立大模型...")
#     model.save_pretrained_merged(MERGED_MODEL_DIR, tokenizer, save_method = "merged_16bit")
#     print(f"✅ 合并完成！终极基座已保存至: {MERGED_MODEL_DIR}")

# if __name__ == "__main__":
#     merge_and_save()
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# # ================= 🔧 终极安全合并配置 =================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # 1. ⚠️ 明确指定地基：必须是你之前的 V1 合并版
# BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V1-Merged") 

# # 2. V3 的 LoRA 补丁路径
# ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-sft-selfplay-v3", "checkpoint-500")

# # 3. 最终输出路径
# MERGED_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V3-Merged")
# # ====================================================

# def merge_and_save():
#     print("==================================================")
#     print("🛠️  启动高精度 CPU 原生合并模式 (防止 GPU 精度丢失)")
#     print("==================================================")
    
#     print(f"🚀 [1/4] 正在加载基础模型 Tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
#     print(f"🚀 [2/4] 正在 CPU 上加载 V1 基础模型 (这可能会稍慢，请耐心等待)...")
#     # 强制在 CPU 上加载，使用 bfloat16 保持原始大模型的精度
#     base_model = AutoModelForCausalLM.from_pretrained(
#         BASE_MODEL_PATH,
#         device_map="cpu", 
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True
#     )
    
#     print(f"🧩 [3/4] 正在挂载 V3 LoRA 适配器: {ADAPTER_PATH}")
#     model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
#     print("🔥 [4/4] 正在进行高精度数学合并 (Merge and Unload)...")
#     # 核心操作：在 CPU 上安全地进行矩阵相加，彻底把灵魂注入肉体
#     merged_model = model.merge_and_unload()
    
#     print(f"💾 正在保存无损完全体模型至: {MERGED_MODEL_DIR}")
#     # safe_serialization=True 会保存为更安全的 safetensors 格式
#     merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
#     tokenizer.save_pretrained(MERGED_MODEL_DIR)
    
#     print("✅ 合并彻底完成！真正的 V3 完全体已诞生！")

# if __name__ == "__main__":
#     merge_and_save()
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. ⚠️ 【核心防坑】必须明确指定地基！(V3 是在 V1 基础上训练的)
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V1-Merged") 

# 2. V3 的 LoRA 补丁路径
ADAPTER_PATH = os.path.join(BASE_DIR, "qwen-sft-selfplay-v3", "checkpoint-500")

# 3. 最终输出路径
MERGED_MODEL_DIR = os.path.join(BASE_DIR, "models", "Qwen-Mahjong-V4-Merged")

def merge_and_save():
    print(f"🚀 步骤 1: 正在加载基础模型 (V1肉体): {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 强制在 CPU 上加载和合并，防止爆显存，同时也防显卡掉线
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
        trust_remote_code=True
    )
    
    print(f"🧩 步骤 2: 正在挂载 V3 适配器 (V3灵魂): {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("🔥 步骤 3: 正在将灵魂永久注入肉体 (Merge and Unload)...")
    # 这一步把 LoRA 的权重彻底加到基座里
    merged_model = model.merge_and_unload()
    
    print(f"💾 步骤 4: 正在保存终极合并模型到: {MERGED_MODEL_DIR}")
    merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)
    
    print("✅ 合并彻底完成！真正的 V3 完全体已诞生！")

if __name__ == "__main__":
    merge_and_save()
# 【关键修改】从 transformers 库导入，而不是 modelscope
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 设置模型路径 (这是你刚刚下载成功的路径)
model_path = r"C:\Users\31801\.cache\modelscope\hub\models\qwen\Qwen2___5-1___5B-Instruct"

print(f"正在从本地加载模型: {model_path} ...")

# 2. 加载分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer 加载成功")
except Exception as e:
    print(f"Tokenizer 加载失败: {e}")
    exit()

# 3. 加载模型 (使用 GPU)
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",          # 自动分配到 GPU
        torch_dtype=torch.float16,  # 半精度加载，显存占用更小，速度更快
        trust_remote_code=True
    )
    print("Model 加载成功！准备测试对话...")
except Exception as e:
    print(f"Model 加载失败: {e}")
    exit()

# 4. 构造 Prompt (模拟麻将场景)
prompt = "你是四川麻将高手。现在手牌是：1万 2万 3万 5筒 6筒。请问我该打哪张？"
messages = [
    {"role": "system", "content": "你是一个麻将助手。"},
    {"role": "user", "content": prompt}
]

# 5. 格式化输入
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 6. 推理生成
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=50
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\n" + "="*30)
print(f"问: {prompt}")
print(f"答: {response}")
print("="*30)
# download_deepseek.py
from modelscope import snapshot_download

print("正在下载 qwen/Qwen2.5-1.5B-Instruct ...")
model_dir = snapshot_download(
    'qwen/Qwen2.5-1.5B-Instruct', 
    cache_dir='./models'  # 指定下载到当前目录下的 models 文件夹，方便管理
)
print(f"✅ 模型下载完成！路径: {model_dir}")
import os
from dotenv import load_dotenv
from google import genai

# 加载你的环境变量
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 向谷歌服务器请求你这个账号目前所有可用的模型
print("正在查询你账号专属的可用模型列表...\n")
for model in client.models.list():
    # 我们只过滤出包含 "flash" 的模型，方便你查找
    if "flash" in model.name.lower():
        print(model.name)
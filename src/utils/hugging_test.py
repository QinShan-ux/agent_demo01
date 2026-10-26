from dotenv import load_dotenv

# 必须在导入 transformers 之前加载，huggingface_hub 会自动读取 HF_TOKEN 环境变量
load_dotenv()

from transformers import pipeline

# 加载一个情感分析模型（第一次运行会自动从 Hub 下载）
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english",local_files_only=True )

# 直接输入文本进行预测
result = classifier("I hunger")
print(result)
# 输出类似: [{'label': 'POSITIVE', 'score': 0.9998}]
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# 加载 .env 文件到环境变量
load_dotenv()
key = os.environ['API_KEY']
url = os.environ['API_URL']
text_model = os.environ['MODEL']


def create_model():
    return ChatOpenAI(
        api_key=key,
        base_url=url,
        model='gpt-5.6-luna',
        temperature=0
    )

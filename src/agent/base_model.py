"""LLM 网关封装：进程内只造一个 ChatOpenAI，其他模块通过 create_model() 拿。

密钥与 base_url 来自 core.config（读 .env 的 API_KEY / API_URL），这里不再自己 load_dotenv。
模型名暂时写死：换模型是件会影响回答质量的事，不该被 .env 里一个残留值悄悄改掉。
"""

from langchain_openai import ChatOpenAI

from core import config


def create_model() -> ChatOpenAI:
    """建一个走 OpenAI 兼容网关的对话模型。"""
    return ChatOpenAI(
        api_key=config.API_KEY,
        base_url=config.API_URL,
        model="deepseek-v4-flash",
        temperature=0,
        # 网关偶发挂起/空回，给它一个上限：超时就快速抛错，由调用方转成 error 事件，
        # 而不是让 SSE 连接无限悬着（那样前端会一直卡在"流式"状态，看起来像没返回）。
        # max_retries=0：不重试，避免退避把"无响应"拖得更久。
        timeout=25,
        max_retries=0,
    )

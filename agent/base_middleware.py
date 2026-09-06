from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage

SENSITIVE_WORDS = ["密码", "银行卡号", "身份证号"]
@before_model
def content_filter(state,runtime):
    """检查用户消息是否包含敏感词，如果包含则拦截"""
    messages = state.get("messages", [])
    if not messages:
        return None
    last_msg = messages[-1]
    content = str(last_msg.content) if hasattr(last_msg, 'content') else ""

    for word in SENSITIVE_WORDS:
        if word in content:
            print(f"[拦截] 检测到敏感词: {word}")
            # jump_to="end" 直接结束，不让模型回复
            return {
                "jump_to": "end",
                "messages": [
                    HumanMessage(content=f"抱歉，为了安全，不能处理包含「{word}」的请求。")
                ]
            }

    return None


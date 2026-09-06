from typing import Dict, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model
import json


class ToolConfig(BaseModel):
    """工具配置模型"""
    name: str
    description: str
    endpoint: str
    method: str = "GET"
    parameters: Dict[str, Any] = {}


def create_tool_from_config(config: dict):
    """根据配置创建工具"""

    # 1. 动态创建参数模型
    param_fields = {}
    for param_name, param_info in config.get("parameters", {}).items():
        param_type = str  # 默认字符串
        if param_info.get("type") == "int":
            param_type = int
        elif param_info.get("type") == "float":
            param_type = float
        elif param_info.get("type") == "bool":
            param_type = bool

        # 关键：使用 Field 定义，并设置默认值 ... 表示该字段为必填
        param_fields[param_name] = (
            param_type,
            Field(description=param_info.get("description", ""), default=...)
        )

    # 使用 create_model 动态创建 Pydantic 模型（推荐方式）
    ArgsSchema = create_model(
        f"{config['name']}Args",
        **param_fields
    )

    # 2. 创建执行函数
    def execute_func(**kwargs) -> str:
        import requests
        url = config["endpoint"]

        if config.get("method") == "GET":
            response = requests.get(url, params=kwargs)
        else:
            response = requests.post(url, json=kwargs)

        return response.text

    # 3. 创建工具
    return StructuredTool(
        name=config["name"],
        description=config["description"],
        func=execute_func,
        args_schema=ArgsSchema
    )


# 从数据库加载配置
configs = [
    {
        "name": "get_weather",
        "description": """你是一个助手，可以使用以下工具：

{tool_desc}

请严格按照以下格式回答。注意：每次只输出 Thought + Action + Action Input，然后停下来等待工具返回结果，不要自己编造 Observation。

格式示例：

Question: 用户提出的问题？
Thought: 用户想了解什么，然后去做什么。
Action: 判断是否有符合条件的工具
Action Input: {{"city": "上海"}}
Observation: 上海 天气 晴 , 25 摄氏度
Thought: 分析出用户问题的答案，给用户提供方案。
Final Answer: 最终的方案。

注意：
- 如果你不需要工具，直接输出 Final Answer
- Action 必须是工具名称，不要加引号
- Action Input 必须是合法的 JSON
- 每次只输出到 Action Input 为止，不要输出 Observation，Observation 由系统自动填充

现在开始：

Question: {input}
Thought: {agent_scratchpad}""",
        "endpoint": "https://uapis.cn/api/v1/misc/weather?city={city}",
        "parameters": {
            "city": {"type": "str", "description": "城市名称"}
        }
    }
]

# 动态生成所有工具
tools = [create_tool_from_config(cfg) for cfg in configs]
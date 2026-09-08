# agent_client.py
import asyncio
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import create_model


async def main():
    # 1. 配置 MCP 服务器连接
    #    方式A：连接到本地 stdio 服务器（你的 math_server.py）

    start = time.time()
    client = MultiServerMCPClient({
        "math_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["/Users/linan/project/python/agent/demo01/mcp_demo.py"],  # 你的服务器脚本路径
        }
    })

    print(f"⏱️ 客户端初始化: {time.time() - start:.2f}s")

    # 测试工具发现耗时
    start = time.time()
    # 方式B：如果服务器是 HTTP 方式运行
    # client = MultiServerMCPClient({
    #     "math_server": {
    #         "transport": "http",
    #         "url": "http://localhost:8000/mcp",
    #     }
    # })

    # 2. 从 MCP 服务器拉取工具（自动发现）
    tools = await client.get_tools()
    tool_map = {tool.name:tool for tool in tools}

    # 直接计算，无LLM
    r1 = await tool_map["add"].ainvoke({"a": 3, "b": 5})
    print(f'结果为 {r1}')
    print(f"⏱️ 工具发现 (list_tools): {time.time() - start:.2f}s")
    print(f"✅ 发现 {len(tools)} 个工具: {[tool.name for tool in tools]}")

    #  测试单次工具调用耗时
    start = time.time()

    # 3. 创建 LangChain Agent（使用你部署的模型）
    model = create_model()
    agent = create_agent(
        model,
        tools,
        system_prompt="直接计算，只返回数字结果，不解释。")

    # 4. 与 Agent 对话
    user_query = "3 加 5 等于多少"
    print(f"👤 用户: {user_query}")

    response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_query}]}
    )

    print(f"🤖 Agent: {response['messages'][-1].content}")
    print(f"⏱️ 单次工具调用: {time.time() - start:.2f}s")


if __name__ == "__main__":
    time1 = datetime.now()
    asyncio.run(main())
    time2 = datetime.now()
    d = time2 - time1
    print(f'流程总共耗时 {d.seconds}')

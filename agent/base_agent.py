from typing import Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from starlette.middleware import Middleware


def train_agent(model: BaseChatModel,
                tools: Optional[list[BaseTool]] = None,
                middleware: Optional[list[AgentMiddleware]] = None ):
    tools = tools or []
    middleware = middleware or []
    agent = create_agent(
        model=model,
        tools = tools,
        middleware=middleware,
        )
    return agent



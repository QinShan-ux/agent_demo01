from langchain.agents.middleware import AgentMiddleware


class ToolCallTrackingMiddleware(AgentMiddleware):
    """为所有工具调用生成并管理可信的 tool_call_id，并记录到数据库。"""

    def __init__(self, db_client):
        """
            Args:
                db_client: 数据库客户端，用于记录工具调用日志
        """
        self.db_client = db_client

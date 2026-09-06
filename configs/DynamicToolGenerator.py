import json
import aiomysql
import requests
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool, BaseTool
import logging

logger = logging.getLogger(__name__)


class DynamicToolGenerator:
    """MySQL版本的动态工具生成器"""

    def __init__(self, pool: aiomysql.Pool):
        self.pool = pool
        self._tool_cache = {}

    async def load_tool_from_db(self, tool_id: str) -> Optional[BaseTool]:
        """从数据库加载工具"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT * FROM tool_definitions WHERE id = %s AND is_active = TRUE",
                    (tool_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    return None
                return self._create_tool_from_config(row)

    async def load_all_active_tools(self) -> List[BaseTool]:
        """加载所有启用的工具"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT * FROM tool_definitions WHERE is_active = TRUE"
                )
                rows = await cursor.fetchall()

        tools = []
        for row in rows:
            tool = self._create_tool_from_config(row)
            if tool:
                tools.append(tool)
        return tools

    def _create_tool_from_config(self, config: Dict) -> Optional[BaseTool]:
        """根据配置创建工具"""
        try:
            tool_type = config.get('type', 'api')

            if tool_type == 'api':
                return self._create_api_tool(config)
            elif tool_type == 'sql':
                return self._create_sql_tool(config)
            elif tool_type == 'custom':
                return self._create_custom_tool(config)
            else:
                logger.warning(f"未知工具类型: {tool_type}")
                return None
        except Exception as e:
            logger.error(f"创建工具失败: {config.get('name')}, 错误: {e}")
            return None

    def _create_api_tool(self, config: Dict) -> BaseTool:
        """创建API调用工具"""

        # 1. 解析参数
        params = json.loads(config.get('parameters', '[]'))
        param_definitions = {}

        for param in params:
            param_name = param['name']
            param_type = self._get_python_type(param.get('type', 'string'))
            param_desc = param.get('description', '')
            is_required = param.get('required', True)

            if is_required:
                param_definitions[param_name] = (
                    param_type,
                    Field(description=param_desc)
                )
            else:
                param_definitions[param_name] = (
                    Optional[param_type],
                    Field(default=None, description=param_desc)
                )

        # 动态创建参数模型
        ArgsSchema = create_model(
            f"{config['name']}Args",
            **param_definitions
        )

        # 2. 获取API配置
        api_config = json.loads(config.get('api_config', '{}'))
        url = api_config.get('url')
        method = api_config.get('method', 'GET').upper()
        timeout = api_config.get('timeout', 10)

        # 3. 执行函数
        def execute_func(**kwargs) -> Dict[str, Any]:
            try:
                if method == 'GET':
                    response = requests.get(url, params=kwargs, timeout=timeout)
                else:
                    response = requests.post(url, json=kwargs, timeout=timeout)

                response.raise_for_status()
                data = response.json()
                return data.get('data', data)

            except requests.exceptions.Timeout:
                return {"error": f"请求超时（{timeout}秒）"}
            except requests.exceptions.ConnectionError:
                return {"error": "网络连接失败"}
            except requests.exceptions.HTTPError as e:
                return {"error": f"HTTP错误: {e.response.status_code}"}
            except Exception as e:
                return {"error": f"请求失败: {str(e)}"}

        # 4. 创建工具
        return StructuredTool(
            name=config['name'],
            description=config.get('description', ''),
            func=execute_func,
            args_schema=ArgsSchema
        )

    def _get_python_type(self, type_str: str):
        """字符串类型转Python类型"""
        type_map = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        return type_map.get(type_str, str)

    def clear_cache(self):
        """清空缓存"""
        self._tool_cache.clear()
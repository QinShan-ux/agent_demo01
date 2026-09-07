import json

import aiomysql
import requests
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, create_model, ValidationError
from langchain_core.tools import StructuredTool, BaseTool
import logging

from configs.ToolError import ToolConfigError

logger = logging.getLogger(__name__)


class RobustDynamicToolGenerator:
    """带异常处理的动态工具生成器"""

    def __init__(self, pool):
        self.pool = pool
        self._tool_cache = {}
        self._failed_tools = {}  # 记录失败的工具

    async def load_all_active_tools(self) -> List[BaseTool]:
        """加载所有启用的工具（跳过有问题的工具）"""
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(
                    "SELECT * FROM tool_definitions WHERE is_active = TRUE"
                )
                rows = await cursor.fetchall()

        tools = []
        for row in rows:
            try:
                tool = await self._safe_create_tool(row)
                if tool:
                    tools.append(tool)
            except Exception as e:
                # 记录失败但不中断整体流程
                logger.error(f"加载工具失败: {row.get('name', 'unknown')}, 错误: {e}")
                self._failed_tools[row.get('id')] = str(e)
                continue

        logger.info(f"成功加载 {len(tools)} 个工具，失败 {len(self._failed_tools)} 个")
        return tools

    async def _safe_create_tool(self, config: Dict) -> Optional[BaseTool]:
        """安全创建工具，捕获所有异常"""
        try:
            return self._create_tool_from_config(config)
        except ToolConfigError as e:
            logger.warning(f"工具配置错误: {config.get('name')}, {e}")
            return None
        except ValidationError as e:
            logger.warning(f"工具参数验证失败: {config.get('name')}, {e}")
            return None
        except Exception as e:
            logger.error(f"创建工具时发生未知错误: {config.get('name')}, {e}")
            return None

    def _create_tool_from_config(self, config: Dict) -> Optional[BaseTool]:
        """根据配置创建工具"""
        tool_type = config.get('type', 'api')

        # 验证必要字段
        required_fields = ['name', 'description', 'parameters']
        for field in required_fields:
            if not config.get(field):
                raise ToolConfigError(f"缺少必要字段: {field}")

        try:
            if tool_type == 'api':
                return self._create_api_tool(config)
            elif tool_type == 'sql':
                return self._create_sql_tool(config)
            elif tool_type == 'custom':
                return self._create_custom_tool(config)
            else:
                raise ToolConfigError(f"未知工具类型: {tool_type}")
        except Exception as e:
            raise ToolConfigError(f"创建工具失败: {e}")

    def _create_api_tool(self, config: Dict) -> BaseTool:
        """创建API调用工具（带执行异常处理）"""

        # 1. 解析参数
        try:
            params = json.loads(config.get('parameters', '[]'))
            api_config = json.loads(config.get('api_config', '{}'))
        except json.JSONDecodeError as e:
            raise ToolConfigError(f"JSON解析失败: {e}")

        # 2. 验证API配置
        url = api_config.get('url')
        if not url:
            raise ToolConfigError("API URL不能为空")

        # 3. 创建参数模型
        try:
            param_definitions = {}
            for param in params:
                param_name = param.get('name')
                if not param_name:
                    continue
                param_type = self._get_python_type(param.get('type', 'string'))
                param_desc = param.get('description', '')
                is_required = param.get('required', True)

                if is_required:
                    param_definitions[param_name] = (param_type, Field(description=param_desc))
                else:
                    param_definitions[param_name] = (Optional[param_type], Field(default=None, description=param_desc))

            ArgsSchema = create_model(
                f"{config['name']}Args",
                **param_definitions
            )
        except Exception as e:
            raise ToolConfigError(f"创建参数模型失败: {e}")

        # 4. 创建执行函数（带执行异常处理）
        def execute_func(**kwargs) -> Dict[str, Any]:
            """执行API调用，捕获所有执行异常"""
            try:
                # 参数验证
                if not kwargs:
                    return {"error": "缺少必要参数"}

                # 发送请求
                method = api_config.get('method', 'GET').upper()
                timeout = api_config.get('timeout', 10)

                try:
                    if method == 'GET':
                        response = requests.get(url, params=kwargs, timeout=timeout)
                    else:
                        response = requests.post(url, json=kwargs, timeout=timeout)

                    response.raise_for_status()
                    data = response.json()

                    # 检查业务状态码
                    if data.get('code') != 0 and data.get('code') is not None:
                        return {
                            "error": f"API业务错误: {data.get('msg', '未知错误')}",
                            "code": data.get('code')
                        }

                    return data.get('data', data)

                except requests.exceptions.Timeout:
                    return {"error": f"请求超时（{timeout}秒）"}
                except requests.exceptions.ConnectionError:
                    return {"error": "网络连接失败"}
                except requests.exceptions.HTTPError as e:
                    return {"error": f"HTTP错误: {e.response.status_code}"}
                except requests.exceptions.JSONDecodeError:
                    return {"error": "响应数据格式异常"}
                except Exception as e:
                    return {"error": f"请求失败: {str(e)}"}

            except Exception as e:
                return {"error": f"执行工具时发生错误: {str(e)}"}

        # 5. 创建工具
        try:
            return StructuredTool(
                name=config['name'],
                description=config.get('description', ''),
                func=execute_func,
                args_schema=ArgsSchema
            )
        except Exception as e:
            raise ToolConfigError(f"创建StructuredTool失败: {e}")

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

    def get_failed_tools(self) -> Dict:
        """获取失败的工具列表"""
        return self._failed_tools

    def clear_cache(self):
        """清空缓存"""
        self._tool_cache.clear()
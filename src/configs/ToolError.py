from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ToolGenerationError(Exception):
    """工具生成异常"""
    pass

class ToolConfigError(ToolGenerationError):
    """工具配置异常"""
    pass

class ToolExecutionError(ToolGenerationError):
    """工具执行异常"""
    pass
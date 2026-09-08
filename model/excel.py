from typing import Literal

from pycparser.c_ast import Enum
from pydantic import BaseModel, Field


class TypeEnum(Enum):
    """数据类型枚举"""
    TEXT = "text"
    NUMERIC = "numeric"
    DATE = "date"
    FUNC = "function"


class ExcelTitle(BaseModel):
    title: str = Field(..., description="列标题名称")
    type: Literal["text", "numeric", "date", "function"] = Field(
        ...,
        description="数据类型：text-文本，numeric-数字，date-日期，function-函数"
    )

    model_config = {
        "use_enum_values": True,
        "arbitrary_types_allowed": True,
    }


"""excel 文件解析与预览的共用工具，供 FastAPI 路由层/上层逻辑复用。

file 形参约定：传入具备 `.filename` 属性、且 `.read()` 为协程的对象
（例如 fastapi 的 UploadFile）。此处不依赖 fastapi，便于在其他场景复用。
"""
import io
from typing import List, Optional

import pandas as pd

# 允许上传的excel文件后缀
ALLOWED_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}

# 默认保留的列（前端未传时使用）
DEFAULT_DESIRED_COLUMNS = ["序号", "姓名", "性别", "身份证号", "民族", "联系电话", "全住宿", "半走读", "班级", "宿舍号",
                           "床铺号"]

__all__ = [
    "ALLOWED_EXCEL_SUFFIXES",
    "DEFAULT_DESIRED_COLUMNS",
    "get_suffix",
    "read_excel_sheets",
    "analyze_excel",
]


def get_suffix(filename: str) -> str:
    """获取文件小写后缀，无后缀时返回空字符串"""
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


async def read_excel_sheets(file) -> dict:
    """读取excel文件全部sheet，返回解析结果。

    校验失败/解析失败时返回 {"code": 1, "message": ..., "data": None}；
    成功时返回 {"sheets": {sheet名: DataFrame}, "filename": str, "content": bytes}。
    """
    filename = file.filename or ""
    suffix = get_suffix(filename)
    if suffix not in ALLOWED_EXCEL_SUFFIXES:
        return {"code": 1, "message": f"不支持的文件类型: {suffix or '未知'}，仅支持 {sorted(ALLOWED_EXCEL_SUFFIXES)}",
                "data": None}

    content = await file.read()

    try:
        # sheet_name=None 同时解析所有sheet
        sheets: dict = pd.read_excel(io.BytesIO(content), sheet_name=None)
        return {
            "sheets": sheets,
            "filename": filename,
            "content": content
        }
    except Exception as e:
        return {"code": 1, "message": f"excel文件解析失败: {e}", "data": None}


async def analyze_excel(file, desired_columns: Optional[List[str]] = None) -> dict:
    """接收excel文件，解析所有sheet并返回预览数据。

    返回 {"code": 0, "message": "success", "data": {...}}，
    每个sheet包含 sheet_name / rows / columns / preview 信息。
    """
    if desired_columns is None:
        desired_columns = DEFAULT_DESIRED_COLUMNS

    result = await read_excel_sheets(file)

    # 检查是否成功
    if result.get("code") == 1:
        return result

    # 从字典中获取数据（使用键访问）
    sheets = result["sheets"]  # 而不是 result.sheets
    filename = result["filename"]
    content = result["content"]

    data = {
        "filename": filename,
        "size": len(content),
        "sheets": [
            {
                "sheet_name": name,
                "rows": len(df),
                "columns": [col for col in desired_columns if col in df.columns],  # 只显示存在的列
                "preview": df.head(5).reindex(columns=desired_columns).fillna("").to_dict(orient="records")
            }
            for name, df in sheets.items()
        ],
    }
    return {"code": 0, "message": "success", "data": data}

"""excel 格式化导出工具：过滤 desired_columns 列，并对 columns_to_fill 处理合并单元格空值。

返回约定：读取/解析失败或无匹配列时返回 {"code": 1, "message": ..., "data": ...}；
成功时返回 {"new_sheets": {sheet名: DataFrame}, "filename": str}，由调用方决定如何输出文件。
"""
from typing import List

import pandas as pd

from .analyze_excel import read_excel_sheets

# 默认需要填充的列（处理合并单元格导致的空值）
DEFAULT_COLUMNS_TO_FILL = ["宿舍号"]

__all__ = [
    "DEFAULT_COLUMNS_TO_FILL",
    "format_excel",
]


async def format_excel(
        file,
        desired_columns: List[str],
        columns_to_fill: List[str],
) -> dict:
    """根据解析结果生成包含 desired_columns 且填充 columns_to_fill 的新表格数据"""
    result = await read_excel_sheets(file)

    # 检查是否成功
    if result.get("code") == 1:
        return result

    # 从字典中获取数据（使用键访问）
    sheets = result["sheets"]  # 而不是 result.sheets
    filename = result["filename"]

    # 创建新表格
    new_sheets = {}
    for name, df in sheets.items():
        # 只获取上传列和表格中真实存在的列
        existing_columns = [col for col in desired_columns if col in df.columns]
        if existing_columns:
            # 先过滤出需要的列
            df_filtered = df[existing_columns].copy()
            # 只对前端指定的列进行填充（处理合并单元格导致的空值）
            for col in columns_to_fill:
                if col in df_filtered.columns:
                    # 向前填充
                    df_filtered[col] = df_filtered[col].ffill()
                    # 如果第一行是空，向上填充
                    if pd.isna(df_filtered[col].iloc[0]):
                        df_filtered[col] = df_filtered[col].bfill()

            new_sheets[name] = df_filtered

    if not new_sheets:
        # 返回实际收到的列名和表格中真实的列名，便于前端排查不匹配的原因
        return {
            "code": 1,
            "message": "没有找到匹配的列，无法生成新表格",
            "data": {
                "received_desired_columns": desired_columns,
                "sheet_columns": {name: list(df.columns) for name, df in sheets.items()},
            },
        }
    return {"new_sheets": new_sheets, "filename": filename}

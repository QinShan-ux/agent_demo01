from typing import List

import io
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse

from model import *

router = APIRouter(prefix="/excel", tags=["excel"])

# 允许上传的excel文件后缀
ALLOWED_EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}

desired_columns = ["序号", "姓名", "性别", "身份证号", "民族", "联系电话", "全住宿", "半走读", "班级", "宿舍号",
                       "床铺号"]
def get_suffix(filename: str) -> str:
    """获取文件小写后缀，无后缀时返回空字符串"""
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/created_titles")
async def created_titles(titles: RequestBody[List[ExcelTitle]]):
    """获取excel的表头， 用于约束excel"""
    print(titles)
    return {"code": 0, "message": "pong", "data": None}


@router.post("/upload")
async def upload_excel(file: UploadFile = File(..., description="excel文件")):
    """接收excel文件，解析所有sheet并返回预览数据"""
    # 校验文件后缀
    # 调用 _file 函数
    result = await _file(file)

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

@router.post("/format-excel")
async def upload_excel(file: UploadFile = File(..., description="excel文件")):
    result = await _file(file)

    # 检查是否成功
    if result.get("code") == 1:
        return result

    # 从字典中获取数据（使用键访问）
    sheets = result["sheets"]  # 而不是 result.sheets
    filename = result["filename"]

    # 创建新表格
    new_sheets = {}
    for name, df in sheets.items():
        # 先处理合并单元格：向前填充空值
        # 对于每一列，如果是合并单元格导致的空值，用前面的值填充
        # df_filled = df.ffill()  # 向下填充（forward fill）
        # df_filled = df_filled.bfill()  # 向上填充（backward fill）

        existing_columns = [col for col in desired_columns if col in df.columns]
        if existing_columns:
            # 先过滤出需要的列
            df_filtered = df[existing_columns].copy()
            # 只对特定列进行填充（处理合并单元格）
            # 例如：只填充 "姓名" 和 "班级" 列
            columns_to_fill = ["宿舍号"]  # 指定需要填充的列

            for col in columns_to_fill:
                if col in df_filtered.columns:
                    # 向前填充
                    df_filtered[col] = df_filtered[col].ffill()
                    # 如果第一行是空，向上填充
                    if pd.isna(df_filtered[col].iloc[0]):
                        df_filtered[col] = df_filtered[col].bfill()

            new_sheets[name] = df_filtered

    if not new_sheets:
        return {"code": 1, "message": "没有找到匹配的列，无法生成新表格", "data": None}

    # 将新表格写入内存
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, df in new_sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    output.seek(0)  # 重置指针到开头

    # 生成新文件名
    new_filename = f"filtered_{filename}"

    # 对中文文件名进行编码（RFC 5987）
    encoded_filename = quote(new_filename)

    # 返回文件流
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )

@router.post("/merge")
async def merge_excel(files: List[UploadFile] = File(..., description="多个excel文件")):
    """接收多个excel文件，按sheet名合并为一个excel并返回合并后的文件

    合并规则：所有文件中同名的sheet纵向拼接（行合并），列不一致时自动补齐；
    某个文件独有的sheet单独保留。
    """
    if not files:
        return {"code": 1, "message": "请至少上传一个excel文件", "data": None}

    # 校验文件后缀
    invalid = [f.filename for f in files if get_suffix(f.filename or "") not in ALLOWED_EXCEL_SUFFIXES]
    if invalid:
        return {"code": 1, "message": f"以下文件不是支持的excel类型: {invalid}，仅支持 {sorted(ALLOWED_EXCEL_SUFFIXES)}",
                "data": None}

    # 按sheet名归集所有文件的表格数据
    sheet_frames: dict = {}
    for f in files:
        content = await f.read()
        try:
            # sheet_name=None 同时解析所有sheet
            sheets: dict = pd.read_excel(io.BytesIO(content), sheet_name=None)
        except Exception as e:
            return {"code": 1, "message": f"excel文件解析失败: {f.filename}: {e}", "data": None}
        for name, df in sheets.items():
            sheet_frames.setdefault(name, []).append(df)

    if not sheet_frames:
        return {"code": 1, "message": "上传的文件中没有可合并的数据", "data": None}

    # 同名sheet纵向拼接，列不一致时自动补齐
    merged = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in sheet_frames.items()
    }

    # 写出到内存中的xlsx，返回文件下载
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in merged.items():
            # excel的sheet名最长31字符
            df.to_excel(writer, sheet_name=name[:31], index=False)
    buf.seek(0)

    filename = f"merged_{len(files)}个文件.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


async def _file(file: UploadFile):
    """接收excel文件，解析所有sheet并返回预览数据"""
    # 校验文件后缀
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


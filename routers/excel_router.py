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


def get_suffix(filename: str) -> str:
    """获取文件小写后缀，无后缀时返回空字符串"""
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


@router.post("/created_titles")
async def created_titles(titles:RequestBody[List[ExcelTitle]]):
    """获取excel的表头， 用于约束excel"""
    print(titles)
    return {"code": 0, "message": "pong", "data": None}


@router.post("/upload")
async def upload_excel(file: UploadFile = File(..., description="excel文件")):
    """接收excel文件，解析所有sheet并返回预览数据"""
    # 校验文件后缀
    filename = file.filename or ""
    suffix = get_suffix(filename)
    if suffix not in ALLOWED_EXCEL_SUFFIXES:
        return {"code": 1, "message": f"不支持的文件类型: {suffix or '未知'}，仅支持 {sorted(ALLOWED_EXCEL_SUFFIXES)}", "data": None}

    content = await file.read()
    try:
        # sheet_name=None 同时解析所有sheet
        sheets: dict = pd.read_excel(io.BytesIO(content), sheet_name=None)
    except Exception as e:
        return {"code": 1, "message": f"excel文件解析失败: {e}", "data": None}

    data = {
        "filename": filename,
        "size": len(content),
        "sheets": [
            {
                "sheet_name": name,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(5).fillna("").to_dict(orient="records"),
            }
            for name, df in sheets.items()
        ],
    }
    return {"code": 0, "message": "success", "data": data}


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
        return {"code": 1, "message": f"以下文件不是支持的excel类型: {invalid}，仅支持 {sorted(ALLOWED_EXCEL_SUFFIXES)}", "data": None}

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
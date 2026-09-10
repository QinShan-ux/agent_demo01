from typing import List

import io
import json
import re
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from model import *
from utils.excel import (
    ALLOWED_EXCEL_SUFFIXES,
    DEFAULT_COLUMNS_TO_FILL,
    DEFAULT_DESIRED_COLUMNS,
    analyze_excel,
    format_excel as excel_format,
    get_suffix,
    read_excel_sheets,
)

router = APIRouter(prefix="/excel", tags=["excel"])


def parse_column_list(raw: List[str]) -> List[str]:
    """解析前端传入的列名列表，兼容三种传法并去除每项首尾空白：
    1. 同名字段重复提交: desired_columns=姓名&desired_columns=班级
    2. 逗号分隔的字符串: desired_columns=姓名,班级
    3. 整体序列化为一个JSON字符串: desired_columns=["姓名","班级"]
    """
    items: List[str] = []
    for value in raw:
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items.extend(str(item) for item in parsed)
                    continue
            except json.JSONDecodeError:
                pass
        # 逗号分隔（兼容中文逗号）
        items.extend(re.split(r"[,，]", s))
    return [item.strip() for item in items if item.strip()]


@router.post("/created_titles")
async def created_titles(titles: RequestBody[List[ExcelTitle]]):
    """获取excel的表头， 用于约束excel"""
    print(titles)
    return {"code": 0, "message": "pong", "data": None}


@router.post("/upload")
async def upload_excel(file: UploadFile = File(..., description="excel文件")):
    """接收excel文件，解析所有sheet并返回预览数据"""
    return await analyze_excel(file)

@router.post("/format-excel")
async def format_excel(
        file: UploadFile = File(..., description="excel文件"),
        desired_columns: List[str] = Form(default=DEFAULT_DESIRED_COLUMNS, description="需要保留的列名列表"),
        columns_to_fill: List[str] = Form(default=DEFAULT_COLUMNS_TO_FILL,
                                          description="需要向前填充的列名列表（处理合并单元格）"),
):
    """上传文件，输出desired_columns的列，同时将columns_to_fill的列进行填充，最后输出新的文件

    desired_columns / columns_to_fill 由前端通过表单字段传入，同名字段重复提交即可传列表；
    未传时使用默认列。
    """
    desired_columns = parse_column_list(desired_columns)
    columns_to_fill = parse_column_list(columns_to_fill)

    res = await excel_format(file, desired_columns, columns_to_fill)

    # 读取/解析失败、无匹配列等错误直接返回
    if res.get("code") == 1:
        return res

    new_sheets = res["new_sheets"]
    filename = res["filename"]

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


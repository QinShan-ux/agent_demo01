from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/test", tags=["测试"])


class TestRequest(BaseModel):
    """测试接口请求体"""
    message: str


@router.get("/ping")
async def ping():
    """健康检查测试接口"""
    return {"code": 0, "message": "pong", "data": None}


@router.post("/echo")
async def echo(req: TestRequest):
    """回显测试接口：返回请求中的内容"""
    return {"code": 0, "message": "success", "data": {"message": req.message}}

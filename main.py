"""启动入口：只负责把 api/app.py 组装好的应用跑起来。

命令行与 IDE 的用法：
    PYTHONPATH=src .venv/Scripts/python.exe -m uvicorn main:app
    PYTHONPATH=src .venv/Scripts/python.exe main.py
或直接双击仓库根的 start.bat（后端 8000 + 前端 5173 各一个常驻窗口）。

单进程启动（reload=False），**不要改回 reload=True**：reload 会起 reloader 父进程 +
worker 子进程，父进程被杀后 worker 常残留占用 8000 端口，表现为「进程表里查不到，
但接口还是旧代码」。
"""

import logging

from api.app import app

logger = logging.getLogger(__name__)

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

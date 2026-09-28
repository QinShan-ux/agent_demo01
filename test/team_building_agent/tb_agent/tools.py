"""工具层：统一失败语义 + 重试策略 + 调用审计。

铁律：
1. 工具永远返回 ToolResult，异常在注册表边界被翻译成 FAILED，不允许炸穿主循环。
2. EMPTY 与 FAILED 必须区分：只有 FAILED/TIMEOUT 才重试；EMPTY 交给上层换参数。
3. 重试**不消耗步骤预算**。步骤只花在"推进任务"上，不花在网络抖动上。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Protocol

from .models import ToolResult, ToolStatus


class Tool(Protocol):
    name: str

    def __call__(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
    def __init__(self, retries: int = 2, backoff_s: float = 0.0) -> None:
        self._tools: dict[str, Callable[..., ToolResult]] = {}
        self.retries = retries
        self.backoff_s = backoff_s
        self.call_count: dict[str, int] = {}
        self.retry_count: dict[str, int] = {}

    def register(self, name: str, fn: Callable[..., ToolResult]) -> "ToolRegistry":
        self._tools[name] = fn
        return self

    def has(self, name: str) -> bool:
        return name in self._tools

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(ToolStatus.FAILED, name, error=f"未注册的工具: {name}")

        last: ToolResult | None = None
        for i in range(self.retries + 1):
            attempts = i + 1
            t0 = time.perf_counter()
            try:
                res = self._tools[name](**kwargs)
            except Exception as exc:  # noqa: BLE001 - 外部世界的异常不许炸穿主循环
                res = ToolResult(ToolStatus.FAILED, name,
                                 error=f"{type(exc).__name__}: {exc}")
            res.attempts = attempts
            res.latency_ms = int((time.perf_counter() - t0) * 1000)

            self.call_count[name] = self.call_count.get(name, 0) + 1
            if attempts > 1:
                self.retry_count[name] = self.retry_count.get(name, 0) + 1

            last = res
            if res.ok or res.status is ToolStatus.EMPTY:
                break  # EMPTY 不重试
            if i < self.retries and self.backoff_s:
                time.sleep(self.backoff_s)

        assert last is not None
        return last

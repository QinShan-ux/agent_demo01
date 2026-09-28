"""团建场地调研 Agent：调研 -> 比价 -> 发邮件。

设计主线（对应需求）：
  1. 工具失败/结果数量不定 -> ToolResult 显式区分 OK/EMPTY/FAILED/TIMEOUT，重试不算步数
  2. 6-8 步               -> 确定性 Planner + 双预算（每版需求 8 步 / 全场 14 步）
  3. 防止假装比价          -> Evidence 必须携带原文子串；比价门槛 = >=2 条可核验报价且 >=2 个独立来源
  4. 中途改需求            -> Intent 版本号 + 字段依赖标签，只失效"依赖被改字段"的证据
"""

from .agent import AgentConfig, Action, BuildingAgent, RunResult
from .extractors import GroundingValidator, HallucinatingExtractor, LLMExtractor, RuleExtractor, parse_price
from .mock_tools import build_world
from .models import Evidence, Intent, RunState, RevisionEvent, Step, ToolResult, ToolStatus, Venue
from .reporting import ComparisonReport, Offer, build_report, draft_email, render_report, safe_intro
from .tools import ToolRegistry
from .verifier import audit_report, find_price_tokens, verify_email

__all__ = [
    "AgentConfig", "Action", "BuildingAgent", "RunResult",
    "GroundingValidator", "HallucinatingExtractor", "LLMExtractor", "RuleExtractor", "parse_price",
    "build_world",
    "Evidence", "Intent", "RunState", "RevisionEvent", "Step", "ToolResult", "ToolStatus", "Venue",
    "ComparisonReport", "Offer", "build_report", "draft_email", "render_report", "safe_intro",
    "ToolRegistry", "audit_report", "find_price_tokens", "verify_email",
]

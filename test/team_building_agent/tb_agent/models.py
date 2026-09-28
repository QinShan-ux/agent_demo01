"""领域模型：意图 / 证据 / 候选场地 / 工具结果 / 运行状态。

三条不变式（整个 Agent 的可信度都建立在这上面）：

1. ToolResult 里 FAILED 与 EMPTY 是两种状态。失败可以重试，空结果重试没有意义
   （参数一样，再打一次还是空），只能换参数 —— 这是"结果数量不定"的正确处理姿势。

2. Evidence 只允许携带 value_span，且 value_span 必须是 quote 的**连续子串**。
   任何无法在来源正文里逐字命中的价格，在进入比价表之前就会被丢弃。
   这一条是"防止假装比价"的地基：模型可以胡说，但胡说进不了数据结构。

3. Intent 带版本号 + 字段依赖标签（FIELD_DEPS）。用户中途改需求时，只失效
   "依赖了被改字段"的证据，既不继续混用旧数据，也不无脑清空整轮结果。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------- 工具结果


class ToolStatus(StrEnum):
    OK = "ok"            # 成功且有结果
    EMPTY = "empty"      # 成功但结果为空 —— 不是错误，不要用同样参数重试
    FAILED = "failed"    # 工具自身报错
    TIMEOUT = "timeout"  # 超时


@dataclass
class ToolResult:
    status: ToolStatus
    tool: str
    data: Any = None
    error: str | None = None
    attempts: int = 1
    latency_ms: int = 0
    tool_call_id: str = field(default_factory=lambda: "tc_" + uuid.uuid4().hex[:8])

    @property
    def ok(self) -> bool:
        return self.status is ToolStatus.OK

    @property
    def retryable(self) -> bool:
        return self.status in (ToolStatus.FAILED, ToolStatus.TIMEOUT)

    def brief(self) -> str:
        if isinstance(self.data, (list, tuple)):
            n = f"n={len(self.data)}"
        elif self.data is None:
            n = "n=-"
        else:
            n = "n=1"
        s = f"{self.tool}:{self.status.value}({n},try{self.attempts})"
        return s + (f" err={self.error}" if self.error else "")


# ------------------------------------------------------------------------ 意图

# 字段 -> 依赖标签：改动某个字段时，只会失效依赖了对应标签的证据
FIELD_DEPS: dict[str, frozenset[str]] = {
    "city": frozenset({"location"}),
    "district": frozenset({"location"}),
    "activity": frozenset({"theme"}),
    "date": frozenset({"time"}),
    "headcount": frozenset({"scale"}),
    "budget_per_person": frozenset({"scale"}),
}

DEP_LABEL = {
    "location": "场地位置",
    "theme": "活动类型",
    "time": "日期",
    "scale": "人数/预算",
}


@dataclass
class RevisionEvent:
    version: int
    changed: dict[str, Any]
    touched_deps: frozenset[str]
    reason: str
    at: str

    def describe(self) -> str:
        chg = "，".join(f"{k}={v}" for k, v in self.changed.items()) or "无实质变化"
        deps = "、".join(DEP_LABEL.get(d, d) for d in sorted(self.touched_deps)) or "无"
        return f"需求升到 v{self.version}（{chg}，原因：{self.reason}）；受影响维度：{deps}"


@dataclass
class Intent:
    city: str = "合肥"
    district: str | None = None
    activity: str = "团建"
    date: str | None = None
    headcount: int = 40
    budget_per_person: int | None = 200
    raw_utterance: str = ""

    version: int = 0
    history: list[RevisionEvent] = field(default_factory=list)

    def digest(self, deps) -> str:
        """按依赖标签把 intent 压成稳定指纹；证据带着它出生，用它判断是否过期。"""
        deps = set(deps)
        payload = {k: getattr(self, k) for k, d in FIELD_DEPS.items() if d & deps}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def revise(self, patch: dict[str, Any], reason: str = "用户中途改需求") -> RevisionEvent:
        changed = {
            k: v for k, v in patch.items()
            if k in FIELD_DEPS and getattr(self, k, None) != v
        }
        for k, v in changed.items():
            setattr(self, k, v)
        self.version += 1
        touched: set[str] = set()
        for k in changed:
            touched |= FIELD_DEPS[k]
        ev = RevisionEvent(self.version, changed, frozenset(touched), reason, now_iso())
        self.history.append(ev)
        return ev

    def brief(self) -> str:
        return (f"{self.city}{self.district or ''} · {self.activity} · {self.headcount} 人 · "
                f"人均预算 {self.budget_per_person or '不限'} 元")


# ------------------------------------------------------------------ 候选与证据


@dataclass(frozen=True)
class Venue:
    key: str
    name: str
    url: str
    district: str
    deps: frozenset[str]
    dep_digest: str
    tool_call_id: str

    @staticmethod
    def make(name: str, url: str, district: str, intent: Intent,
             deps, tool_call_id: str) -> "Venue":
        key = "v_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        return Venue(key, name, url, district, frozenset(deps), intent.digest(deps), tool_call_id)


@dataclass(frozen=True)
class Evidence:
    """一条可核验的断言。value_span 必须逐字出现在 quote 里。"""

    evidence_id: str
    venue_key: str
    venue_name: str
    field_name: str
    value_span: str
    quote: str
    source_url: str
    fetched_at: str
    deps: frozenset[str]
    dep_digest: str
    tool_call_id: str

    def grounded(self) -> bool:
        return bool(self.value_span) and self.value_span in self.quote

    def brief(self) -> str:
        return f"{self.venue_name} {self.field_name}={self.value_span!r} @{self.source_url}"


# ---------------------------------------------------------------- 步骤与状态


@dataclass
class Step:
    idx: int
    action: str
    detail: str
    results: list[ToolResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        body = " | ".join(r.brief() for r in self.results) or "-"
        line = f"[{self.idx:>2}] {self.action:<13} {self.detail}\n       => {body}"
        for n in self.notes:
            line += f"\n       · {n}"
        return line


@dataclass
class RunState:
    intent: Intent
    candidates: list[Venue] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    retired: list[str] = field(default_factory=list)
    fetched_keys: set[str] = field(default_factory=set)
    failed_fetch_keys: set[str] = field(default_factory=set)
    log: list[Step] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    search_rounds: int = 0
    last_search_empty: bool = False

    # ---- 依赖失效：只认"指纹仍然匹配"的候选与证据 ----

    def live_candidates(self) -> list[Venue]:
        return [c for c in self.candidates
                if c.dep_digest == self.intent.digest(c.deps)]

    def live_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence
                if e.dep_digest == self.intent.digest(e.deps)]

    def audit_lines(self) -> list[str]:
        out = []
        for s in self.log:
            out.append(s.render())
        if self.retired:
            out.append("— 已作废产物 —")
            out += [f"  × {r}" for r in self.retired]
        return out

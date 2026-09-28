"""Agent 主循环：确定性 Planner + 预算 + 放行闸门。

为什么控制流不交给 LLM：
  "下一步做什么"是可以用代码写死的有限状态机（搜 -> 取 -> 比 -> 写 -> 发）。
  让模型参与决策只会引入两类故障：跳步（没取证就比价）和绕圈（反复重搜）。
  模型只在两个地方出现：抽取报价候选（必须过逐字校验）、写邮件引言（必须过数字核对）。

预算模型：
  · 每版需求 8 步（用户改需求 = 新子目标，重新给 8 步）
  · 全场硬上限 14 步（防止用户反复改需求把任务拖死）
  · 工具重试不占步数
  预算耗尽 -> 降级交付：交出"已覆盖的部分 + 明确缺口"，绝不补齐、绝不编造。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum

from .extractors import GroundingValidator
from .models import (Evidence, Intent, RevisionEvent, RunState, Step, ToolStatus,
                     Venue, now_iso)
from .reporting import (ComparisonReport, IntroFn, build_report, draft_email,
                        render_report, safe_intro)
from .verifier import audit_report, find_price_tokens, verify_email

# 候选场地与报价都绑定"位置 + 活动类型"。
# 如果某些套餐是按人数分档报价的，把 "scale" 加进来即可让旧报价自动失效。
CANDIDATE_DEPS = frozenset({"location", "theme"})
PRICE_DEPS = frozenset({"location", "theme"})

TERMINAL = {"done", "give_up", "verifier_blocked", "awaiting_confirmation",
            "degraded", "send_failed"}


class Action(StrEnum):
    SEARCH = "search"
    RELAX_SEARCH = "relax_search"
    FETCH = "fetch"
    COMPARE = "compare"
    DRAFT = "draft"
    SEND = "send"
    GIVE_UP = "give_up"
    DONE = "done"


@dataclass
class AgentConfig:
    max_steps_per_version: int = 8   # 单个子目标（一版需求）的预算
    max_total_steps: int = 14        # 全场硬上限
    fetch_batch: int = 2             # 每个 fetch 步骤抓几个页面
    max_fetch_steps: int = 2         # 最多花几步取证
    mailbox: str = "hr-team@example.com"
    require_send_confirmation: bool = False   # 常态下自动发；低置信度结果强制人工确认


@dataclass
class RunResult:
    status: str
    intent_version: int
    steps_total: int
    report: ComparisonReport | None
    email: dict | None
    violations: list[str]
    problems: list[str]
    notes: list[str]
    state: RunState

    def audit_text(self) -> str:
        L = [f"■ 运行状态：{self.status}｜需求版本 v{self.intent_version}｜总步数 {self.steps_total}"]
        L.append("■ 步骤轨迹")
        L += self.state.audit_lines()
        if self.notes:
            L.append("■ 关键说明")
            L += [f"  · {n}" for n in self.notes]
        if self.violations or self.problems:
            L.append("■ 拦截记录")
            L += [f"  ✗ {v}" for v in self.violations + self.problems]
        return "\n".join(L)


class BuildingAgent:
    def __init__(self, tools, extractor, config: AgentConfig | None = None,
                 intro_fn: IntroFn | None = None) -> None:
        self.tools = tools
        self.extractor = extractor
        self.config = config or AgentConfig()
        self.intro_fn: IntroFn = intro_fn or safe_intro

        self.state = RunState(intent=Intent())
        self.report: ComparisonReport | None = None
        self.email: dict | None = None
        self.violations: list[str] = []
        self.problems: list[str] = []
        self.status = "init"
        self.intro_degraded = False

        self.total_steps = 0
        self._version_steps = 0
        self._fetch_steps = 0
        self._draft_attempts = 0
        self._confirmed = False

    # ------------------------------------------------------------ 对外接口

    def set_intent(self, reason: str = "初始需求", **patch) -> None:
        self.state.intent.revise(patch, reason)

    def revise(self, reason: str = "用户中途改需求", **patch) -> str:
        """用户中途改需求：按依赖标签作废旧产物，然后重新给预算。"""
        ev: RevisionEvent = self.state.intent.revise(patch, reason)
        if not ev.changed:
            return "需求无变化，计划保持不变"

        dead_c = [c for c in self.state.candidates if not self._alive(c.deps, c.dep_digest)]
        dead_e = [e for e in self.state.evidence if not self._alive(e.deps, e.dep_digest)]
        self.state.candidates = [c for c in self.state.candidates if self._alive(c.deps, c.dep_digest)]
        self.state.evidence = [e for e in self.state.evidence if self._alive(e.deps, e.dep_digest)]

        for c in dead_c:
            self.state.retired.append(f"候选场地作废：{c.name}（依赖 {'/'.join(sorted(c.deps))}）")
        for e in dead_e:
            self.state.retired.append(f"证据作废：{e.venue_name} {e.value_span!r}（依赖 {'/'.join(sorted(e.deps))}）")

        if "location" in ev.touched_deps:
            self._fetch_steps = 0
            self.state.fetched_keys.clear()
            self.state.failed_fetch_keys.clear()
            self.state.search_rounds = 0
            self.state.last_search_empty = False

        self._version_steps = 0          # 新子目标，重新给 8 步
        self._draft_attempts = 0
        self.report = None
        self.email = None
        self.violations = []
        self.status = "revised"

        note = f"{ev.describe()}；作废候选 {len(dead_c)} 个、证据 {len(dead_e)} 条"
        self.state.notes.append(note)
        return note

    def confirm_send(self) -> None:
        """低置信度结果的放行动作（人工确认）。确认后允许继续 run()。"""
        self._confirmed = True
        if self.status == "awaiting_confirmation":
            self.status = "confirmed"   # 解除挂起，下一轮循环会走到 SEND

    # ------------------------------------------------------------ 主循环

    def run(self, max_loops: int = 60) -> RunResult:
        for _ in range(max_loops):
            if self.status in TERMINAL:
                break
            self.step_once()
        return self.result()

    def step_once(self) -> Action:
        if self.status in TERMINAL:
            return Action.DONE
        action = self._decide()
        self._execute(action)
        return action

    def result(self) -> RunResult:
        return RunResult(self.status, self.state.intent.version, self.total_steps,
                         self.report, self.email, self.violations, self.problems,
                         list(self.state.notes), self.state)

    # ------------------------------------------------------------ 决策

    def _alive(self, deps, digest) -> bool:
        return digest == self.state.intent.digest(deps)

    def _afford(self, k: int) -> bool:
        return (self.config.max_steps_per_version - self._version_steps >= k
                and self.total_steps + k <= self.config.max_total_steps)

    def _pending(self) -> list[Venue]:
        return [c for c in self.state.live_candidates() if c.key not in self.state.fetched_keys]

    def _decide(self) -> Action:
        s = self.state
        if not s.candidates:
            if s.search_rounds == 0:
                return Action.SEARCH
            if s.search_rounds == 1 and self._afford(5):
                return Action.RELAX_SEARCH      # 换参数再试一次，而不是重试同样参数
            return Action.GIVE_UP
        if self._pending() and self._fetch_steps < self.config.max_fetch_steps and self._afford(4):
            return Action.FETCH
        if self.report is None:
            return Action.COMPARE
        if self.email is None:
            return Action.DRAFT
        if self.status != "sent":
            return Action.SEND
        return Action.DONE

    def _execute(self, action: Action) -> None:
        if self._version_steps >= self.config.max_steps_per_version:
            self._degrade("本版需求步数预算用尽")
            return
        if self.total_steps >= self.config.max_total_steps:
            self._degrade("全场步数上限用尽")
            return
        {
            Action.SEARCH: self._h_search,
            Action.RELAX_SEARCH: self._h_search,
            Action.FETCH: self._h_fetch,
            Action.COMPARE: self._h_compare,
            Action.DRAFT: self._h_draft,
            Action.SEND: self._h_send,
            Action.GIVE_UP: self._h_give_up,
            Action.DONE: lambda: None,   # 终态由各 handler 自己设置，这里不动状态
        }[action]()

    def _begin(self, action: Action, detail: str) -> Step:
        step = Step(self.total_steps + 1, action.value, detail)
        self.state.log.append(step)
        return step

    def _end(self, step: Step, results=None, notes=None) -> None:
        step.results = results or []
        step.notes = notes or []
        self.total_steps += 1
        self._version_steps += 1

    # ------------------------------------------------------------ 各步骤

    def _h_search(self) -> None:
        s = self.state
        wide = s.search_rounds >= 1
        action = Action.RELAX_SEARCH if wide else Action.SEARCH
        detail = (f"放宽条件重搜（去掉活动词）" if wide else
                  f"检索 {s.intent.city} · {s.intent.activity}")
        step = self._begin(action, detail)
        res = self.tools.invoke("search_venues", city=s.intent.city,
                                keyword="" if wide else s.intent.activity, wide=wide)
        s.search_rounds += 1
        notes: list[str] = []

        if res.ok:
            seen = {c.url for c in s.candidates}
            fresh: list[Venue] = []
            for row in res.data:
                if row["url"] in seen:
                    continue
                seen.add(row["url"])
                fresh.append(Venue.make(row["name"], row["url"], row.get("district", ""),
                                        s.intent, CANDIDATE_DEPS, res.tool_call_id))
            s.candidates.extend(fresh)
            s.last_search_empty = False
            notes.append(f"命中 {len(res.data)} 条（条数不定），去重后新增候选 {len(fresh)} 个")
        elif res.status is ToolStatus.EMPTY:
            s.last_search_empty = True
            notes.append("检索成功但结果为空 → 不重试同参数，改走放宽条件分支")
        else:
            s.last_search_empty = True
            notes.append(f"检索失败（内部已重试 {res.attempts} 次）：{res.error} → 不猜测，"
                         f"改用放宽条件兜底；仍失败则如实上报")
        self._end(step, [res], notes)

    def _h_fetch(self) -> None:
        s = self.state
        batch = self._pending()[:self.config.fetch_batch]
        step = self._begin(Action.FETCH,
                           f"抓正文 + 取证 {len(batch)} 个场地（批大小 {self.config.fetch_batch}）")
        results, notes = [], []

        for c in batch:
            res = self.tools.invoke("fetch_page", url=c.url)
            results.append(res)
            s.fetched_keys.add(c.key)   # 无论成败都记为"已尝试"，避免同一场地反复烧步数

            if res.ok:
                text = res.data.get("text", "")
                cands = self.extractor.extract(text, c.name)
                accepted, rejected = GroundingValidator.validate(cands, text)
                if rejected:
                    notes.append(f"{c.name}：丢弃 {len(rejected)} 条无法在原文逐字命中的片段"
                                 f"（疑似编造）→ {[r['span'] for r in rejected]}")
                for k, cand in enumerate(accepted):
                    s.evidence.append(Evidence(
                        evidence_id=f"ev_{c.key[2:]}_{k}",
                        venue_key=c.key, venue_name=c.name,
                        field_name=cand["field"], value_span=cand["span"],
                        quote=text, source_url=c.url, fetched_at=now_iso(),
                        deps=PRICE_DEPS, dep_digest=s.intent.digest(PRICE_DEPS),
                        tool_call_id=res.tool_call_id))
                notes.append(f"{c.name}：采信 {len(accepted)} 条带出处的报价"
                             if accepted else
                             f"{c.name}：正文中没有报价 → 标为未取得报价（不估算）")
            elif res.status is ToolStatus.EMPTY:
                notes.append(f"{c.name}：页面无有效内容（EMPTY，不重试）")
            else:
                s.failed_fetch_keys.add(c.key)
                notes.append(f"{c.name}：抓取失败（重试 {res.attempts} 次）{res.error} → 标为未取证")

        self._fetch_steps += 1
        self._end(step, results, notes)

    def _h_compare(self) -> None:
        step = self._begin(Action.COMPARE, "装配比价表（只采信可核验报价）")
        self.report = build_report(self.state, self.state.intent)
        r = self.report
        notes: list[str] = []
        if r.can_compare:
            lo = min(o.price for o in r.verified)   # type: ignore[type-var]
            hi = max(o.price for o in r.verified)   # type: ignore[type-var]
            notes.append(f"可核验报价 {len(r.verified)} 条、独立来源 {len(r.sources)} 个 "
                         f"→ 达到比价门槛，人均 {lo}–{hi} 元")
        else:
            notes.append("未达到比价门槛 → 不产出任何排序或推荐：" + "；".join(r.blocked_reasons))
        if r.degraded:
            notes.append(f"覆盖度不完整：未覆盖 {r.not_fetched} 个、抓取失败 {r.fetch_failed} 个 "
                         f"→ 报告与邮件必须标注结论不完整")
        self._end(step, [], notes)

    def _h_draft(self) -> None:
        assert self.report is not None
        step = self._begin(Action.DRAFT, "生成邮件（模型只写引言，表格由代码渲染）")
        notes: list[str] = []
        intent = self.state.intent

        subject, body = draft_email(self.report, intent, self.intro_fn)
        v = verify_email(self.report, subject, body, intent)

        if v and self._draft_attempts < 1:
            self._draft_attempts += 1
            notes.append(f"引言核对不通过 {len(v)} 处，带批注回炉重写一次：{v[0]}")
            subject, body = draft_email(self.report, intent, self.intro_fn, critique=v)
            v = verify_email(self.report, subject, body, intent)

        if v:
            notes.append(f"回炉后仍有 {len(v)} 处不实 → 丢弃模型引言，改用代码模板引言")
            self.intro_degraded = True
            subject, body = draft_email(self.report, intent, self.intro_fn, forced_template=True)
            v = verify_email(self.report, subject, body, intent)

        self.problems = audit_report(self.report)
        self.violations = v
        if v or self.problems:
            self.email = None
            self.status = "verifier_blocked"
            notes.append("最终核对仍未通过 → 拒绝生成邮件（宁可不发，也不发带无出处数字的邮件）")
            notes += [f"✗ {x}" for x in (v + self.problems)]
        else:
            self.email = {"to": self.config.mailbox, "subject": subject, "body": body}
            notes.append(f"核对通过：正文 {len(find_price_tokens(body))} 处报价数字全部可溯源"
                         + ("（引言已降级为模板）" if self.intro_degraded else ""))
        self._end(step, [], notes)

    def _h_send(self) -> None:
        assert self.report is not None and self.email is not None
        step = self._begin(Action.SEND, f"发送邮件 → {self.config.mailbox}")
        notes: list[str] = []
        intent = self.state.intent

        # 不对称策略：不可比价的低置信度结果，必须人工确认后才允许外发
        need_confirm = (not self.report.can_compare) or self.config.require_send_confirmation
        if need_confirm and not self._confirmed:
            self.status = "awaiting_confirmation"
            notes.append("本轮结论置信度不足（未形成比价）→ 按策略挂起，等待人工确认后再发")
            notes.append(f"待发主题：{self.email['subject']}")
            self._end(step, [], notes)
            return

        v = verify_email(self.report, self.email["subject"], self.email["body"], intent)
        if v:
            self.violations = v
            self.status = "verifier_blocked"
            notes.append("发送前二次核对不通过 → 拒发")
            notes += [f"✗ {x}" for x in v]
            self._end(step, [], notes)
            return

        res = self.tools.invoke("send_email", **self.email)
        if res.ok:
            self.status = "sent"
            notes.append(f"已发送 message_id={res.data['message_id']}（共尝试 {res.attempts} 次）")
        else:
            self.status = "send_failed"
            notes.append(f"发送失败（重试 {res.attempts} 次）：{res.error} "
                         f"→ 不谎报送达，草稿已落盘可人工取回")
            path = self._save_draft()
            notes.append(f"草稿路径：{path}")
        self._end(step, [res], notes)

    def _h_give_up(self) -> None:
        step = self._begin(Action.GIVE_UP, "无可用候选场地，终止")
        reasons = [f"检索轮次 {self.state.search_rounds} 次（含放宽条件）均未取到候选场地"]
        if self.tools.call_count:
            reasons.append("工具调用统计：" + "，".join(
                f"{k}×{v}" for k, v in self.tools.call_count.items()))
        reasons.append("不做任何推测性输出：没有候选就没有报价，没有报价就没有比价")
        self.status = "give_up"
        self._end(step, [], reasons)

    # ------------------------------------------------------------ 降级与落盘

    def _degrade(self, why: str) -> None:
        step = self._begin(Action.DONE, f"降级交付：{why}")
        notes = [f"{why}（本版已用 {self._version_steps}/{self.config.max_steps_per_version} 步，"
                 f"全场 {self.total_steps}/{self.config.max_total_steps} 步）"]
        if self.report is None and self.state.candidates:
            self.report = build_report(self.state, self.state.intent)
            notes.append("按已获得的数据装配报告：覆盖率与缺口如实标注，未取证场地不参与排序")
        self.status = "degraded"
        self._end(step, [], notes)

    def _save_draft(self) -> str:
        outdir = os.path.join(os.getcwd(), "outbox")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"draft_v{self.state.intent.version}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"To: {self.email['to']}\nSubject: {self.email['subject']}\n\n{self.email['body']}")
        return path

    # ------------------------------------------------------------ 只读视图

    def render_audit_report(self) -> str:
        if self.report is None:
            return "（尚未生成报告）"
        return render_report(self.report, self.state.intent)

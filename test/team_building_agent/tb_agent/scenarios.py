"""5 类边界场景的可执行验证：跑一遍就知道防护是否真的生效。

S1 正常路径        —— 工具抖动（超时后自愈）+ 覆盖不完整但如实标注
S2 抽取层幻觉      —— 页面上没有价格，抽取器"读"出一个价格 → 被逐字校验丢弃 → 比价被拒
S3 引言层幻觉      —— 表格是真的，但模型在引言里编了个更便宜的价格 → 被拦 → 降级模板
S4 中途改需求      —— 合肥 → 南京（位置依赖全废）；人数 40 → 60（报价证据保留）
S5 工具故障与空结果 —— 检索全挂 → 如实放弃；抓取全挂 → 降级但不编；结果为空 → 换参数而非重试
"""
from __future__ import annotations

from .agent import AgentConfig, BuildingAgent
from .extractors import HallucinatingExtractor, RuleExtractor
from .mock_tools import FaultInjector, build_world
from .models import ToolStatus

BAR = "=" * 78
SUB = "-" * 78


def braggy_intro(report, intent, critique=None):
    """一个"嘴上很能说"的模型：无视批注，坚持编价格。"""
    return (f"这轮把{intent.city}的团建场地都摸了一遍：巢湖半汤温泉度假区性价比最高，"
            f"人均 158 元就能拿下，全场最便宜，建议直接定这家。")


def _show(log, title, agent, result, *, mode="full", extra=None) -> None:
    log(BAR)
    log(f"◆ {title}")
    log(BAR)
    if extra:
        log(extra)
    log(result.audit_text())
    log(SUB)
    log(agent.render_audit_report())
    if result.email and mode != "brief":
        log(SUB)
        log("【邮件预览】")
        log(f"To: {result.email['to']}")
        log(f"Subject: {result.email['subject']}")
        log(result.email["body"])
    elif result.email:
        log(SUB)
        log(f"【邮件】Subject: {result.email['subject']}（正文略）")
    if result.status in ("awaiting_confirmation", "send_failed", "verifier_blocked", "give_up"):
        log(SUB)
        log(f"⚠ 未完成外发，运行状态 = {result.status}")
    log("")


# ------------------------------------------------------------------ S1


def s1_normal(log) -> None:
    faults = FaultInjector(script={"search_venues": [ToolStatus.TIMEOUT]})
    registry, outbox = build_world(faults, retries=2)
    agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
    agent.set_intent(city="合肥", activity="团建", headcount=40, budget_per_person=200)
    result = agent.run()
    _show(log, "S1 正常路径：工具超时自愈 + 覆盖度如实标注（6 步 / 8 步预算）",
          agent, result)
    log(f"（发件箱：{len(outbox)} 封；工具调用统计 {registry.call_count}，"
        f"其中重试 {registry.retry_count}）\n")


# ------------------------------------------------------------------ S2


def s2_extraction_hallucination(log) -> None:
    # 两个有报价的页面抓取失败；两个没有报价的页面抓取成功 —— 这正是幻觉最容易发生的缺口
    faults = FaultInjector(script={"fetch_page": [ToolStatus.FAILED, ToolStatus.FAILED]})
    registry, outbox = build_world(faults, retries=0)
    agent = BuildingAgent(registry, HallucinatingExtractor(), AgentConfig())
    agent.set_intent(city="合肥", activity="团建", headcount=60, budget_per_person=200)

    result = agent.run()
    log(BAR)
    log("◆ S2 抽取层幻觉：模型给「没有报价的页面」读出了价格")
    log(BAR)
    log(result.audit_text())
    log(SUB)
    log(agent.render_audit_report())
    log(SUB)
    log(f"→ 运行状态 = {result.status}：低置信度结果不允许自动外发")

    # 人工确认后放行（发出的是一封"未形成比价"的诚实邮件，而不是编造的比价表）
    log(f"（确认前发件箱：{len(outbox)} 封 —— 低置信度结果不允许自动外发）")
    agent.confirm_send()
    result2 = agent.run()
    log(SUB)
    log(f"人工确认后：状态 {result2.status}，发件箱 {len(outbox)} 封")
    log(f"  Subject: {result2.email['subject']}")
    log("  ---- 正文 ----")
    log(result2.email["body"])
    log("")


# ------------------------------------------------------------------ S3


def s3_email_hallucination(log) -> None:
    registry, outbox = build_world(FaultInjector(), retries=2)
    agent = BuildingAgent(registry, RuleExtractor(), AgentConfig(), intro_fn=braggy_intro)
    agent.set_intent(city="合肥", activity="团建", headcount=40, budget_per_person=200)
    result = agent.run()

    log(BAR)
    log("◆ S3 引言层幻觉：表格是真的，模型在引言里把 168 说成 158 并声称全场最便宜")
    log(BAR)
    log(result.audit_text())
    log(SUB)
    log("最终外发邮件（引言已被丢弃，改用代码模板）：")
    log(f"Subject: {result.email['subject'] if result.email else '（未生成）'}")
    if result.email:
        log("---- 正文 ----")
        log(result.email["body"])
    log("")


# ------------------------------------------------------------------ S4


def s4_revision(log) -> None:
    registry, outbox = build_world(FaultInjector(), retries=2)
    agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
    agent.set_intent(city="合肥", activity="团建", headcount=40, budget_per_person=200)

    for _ in range(3):          # 已经搜完 + 取完 4 家证
        agent.step_once()
    log(BAR)
    log("◆ S4 中途改需求：跑到第 3 步时用户说「算了，改成南京」")
    log(BAR)
    log("改需求前已积累：")
    log(f"  候选场地 {len(agent.state.candidates)} 个，证据 {len(agent.state.evidence)} 条")

    note = agent.revise(city="南京", reason="算了，改成南京")
    log(SUB)
    log("失效处理：")
    log(f"  {note}")
    log(f"  作废后立即生效的数据：候选 {len(agent.state.candidates)} 个，证据 {len(agent.state.evidence)} 条")
    log(SUB)

    result = agent.run()
    log(result.audit_text())
    log(SUB)
    log(agent.render_audit_report())
    log(SUB)
    log("【邮件预览】")
    log(f"Subject: {result.email['subject']}")
    log(result.email["body"])
    log(SUB)

    # 对照组：只改人数 —— 报价证据不该被作废
    log("对照组：只改人数（40 → 60），报价证据应当保留：")
    note2 = agent.revise(headcount=60, reason="人数改成 60 人")
    log(f"  {note2}")
    log(f"  候选 {len(agent.state.candidates)} 个，证据 {len(agent.state.evidence)} 条（未被作废）")
    result2 = agent.run()
    log(f"  → 直接用旧证据重出报告：{result2.status}，总步数 {result2.steps_total}")
    log("")


# ------------------------------------------------------------------ S5


def s5_failures(log) -> None:
    log(BAR)
    log("◆ S5 工具故障与空结果：三种边界")
    log(BAR)

    # 5a 检索彻底挂掉
    faults = FaultInjector(script={"search_venues": [ToolStatus.FAILED] * 6})
    registry, _ = build_world(faults, retries=2)
    agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
    agent.set_intent(city="合肥", activity="团建")
    result = agent.run()
    log("▸ 5a 检索连续失败（首轮 + 放宽各重试 3 次均失败）")
    log(result.audit_text())
    log(f"  → 状态 {result.status}；报告？{'有' if result.report else '没有'}；"
        f"邮件？{'有' if result.email else '没有'}")
    log("")

    # 5b 抓取全挂
    faults = FaultInjector(script={"fetch_page": [ToolStatus.FAILED] * 12})
    registry, outbox = build_world(faults, retries=2)
    agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
    agent.set_intent(city="合肥", activity="团建")
    result = agent.run()
    log("▸ 5b 检索成功但所有页面抓取失败")
    log(result.audit_text())
    log(SUB)
    log(agent.render_audit_report())
    log(f"（确认前发件箱：{len(outbox)} 封）")
    agent.confirm_send()
    result = agent.run()
    log(SUB)
    log(f"人工确认后：状态 {result.status}，发件箱 {len(outbox)} 封")
    if result.email:
        log("外发的邮件（没有编造任何报价）：")
        log(f"Subject: {result.email['subject']}")
    log("")

    # 5c 结果为空 -> 放宽条件，而不是重试同样参数
    registry, _ = build_world(FaultInjector(), retries=2)
    agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
    agent.set_intent(city="合肥", activity="高端年会")
    result = agent.run()
    log("▸ 5c 检索结果为空（活动词太窄）：不重试同参数，改走放宽分支")
    log(result.audit_text())
    log(f"  → 状态 {result.status}，重试统计 {registry.retry_count or '无重试'}")
    log("")


def run_all() -> str:
    lines: list[str] = []

    def log(s: str = "") -> None:
        lines.append(str(s))

    log(BAR)
    log("团建场地调研 Agent —— 边界场景验证")
    log("需求：调研场地 → 比价 → 发邮件；要求处理工具失败、结果数量不定、防幻觉、支持中途改需求")
    log(BAR)
    log("")
    s1_normal(log)
    s2_extraction_hallucination(log)
    s3_email_hallucination(log)
    s4_revision(log)
    s5_failures(log)
    return "\n".join(lines)

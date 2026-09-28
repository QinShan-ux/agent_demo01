"""报告层：把证据装配成比价表，并生成邮件。

比价门槛（硬规则，写在代码里而不是写给模型看）：
  can_compare = 可核验报价 >= 2 条  且  来源 >= 2 个独立 URL

不满足就不产出任何"最优/最便宜"结论，只在报告里列出缺口和原因。
未取得报价的场地一律显示"未取证"，绝不用估算值填空。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .extractors import parse_price
from .models import Evidence, Intent, RunState, Venue, now_iso

CONF_LABEL = {
    "verified": "已核验",
    "price_not_found": "页面无公开报价",
    "fetch_failed": "抓取失败，未取证",
    "not_fetched": "未取证（步数预算内未覆盖）",
}


@dataclass
class Offer:
    venue: Venue
    price: int | None
    confidence: str
    evidence_ids: tuple[str, ...] = ()
    spans: tuple[str, ...] = ()


@dataclass
class ComparisonReport:
    version: int
    city: str
    offers: list[Offer]
    verified: list[Offer]
    sources: set[str]
    can_compare: bool
    blocked_reasons: list[str]
    cheapest: Offer | None
    within_budget: list[Offer]
    not_fetched: int
    fetch_failed: int
    price_not_found: int
    generated_at: str = field(default_factory=now_iso)

    @property
    def degraded(self) -> bool:
        return (self.not_fetched + self.fetch_failed) > 0


def build_report(state: RunState, intent: Intent) -> ComparisonReport:
    live_c = state.live_candidates()
    live_e = [e for e in state.live_evidence() if e.field_name == "price_per_person"]

    by_venue: dict[str, list[Evidence]] = {}
    for e in live_e:
        by_venue.setdefault(e.venue_key, []).append(e)

    offers: list[Offer] = []
    for c in live_c:
        evs = by_venue.get(c.key, [])
        prices = [p for p in (parse_price(e.value_span) for e in evs) if p is not None]
        if evs and prices:
            offers.append(Offer(c, min(prices), "verified",
                                tuple(e.evidence_id for e in evs),
                                tuple(e.value_span for e in evs)))
        elif c.key in state.failed_fetch_keys:
            offers.append(Offer(c, None, "fetch_failed"))
        elif c.key in state.fetched_keys:
            offers.append(Offer(c, None, "price_not_found"))
        else:
            offers.append(Offer(c, None, "not_fetched"))

    verified = [o for o in offers if o.confidence == "verified"]
    ev_index = {e.evidence_id: e for e in live_e}
    sources = {ev_index[i].source_url for o in verified for i in o.evidence_ids if i in ev_index}

    reasons: list[str] = []
    if len(verified) < 2:
        reasons.append(f"取得可核验报价的场地只有 {len(verified)} 个（构成比价需要 ≥2 个）")
    if len(sources) < 2:
        reasons.append(f"可核验报价只来自 {len(sources)} 个独立来源（需要 ≥2 个，"
                       f"避免同一个页面被当成多家比价）")
    can_compare = len(verified) >= 2 and len(sources) >= 2

    cheapest = min(verified, key=lambda o: o.price) if can_compare else None  # type: ignore[arg-type]
    within = ([o for o in verified if o.price is not None
               and (intent.budget_per_person is None or o.price <= intent.budget_per_person)]
              if can_compare else [])

    return ComparisonReport(
        version=intent.version,
        city=intent.city,
        offers=offers,
        verified=verified,
        sources=sources,
        can_compare=can_compare,
        blocked_reasons=reasons,
        cheapest=cheapest,
        within_budget=within,
        not_fetched=sum(1 for o in offers if o.confidence == "not_fetched"),
        fetch_failed=sum(1 for o in offers if o.confidence == "fetch_failed"),
        price_not_found=sum(1 for o in offers if o.confidence == "price_not_found"),
    )


# ----------------------------------------------------------------- 文本渲染


def render_report(report: ComparisonReport, intent: Intent) -> str:
    """本地审计版（含完整原文引用）。"""
    L: list[str] = []
    L.append(f"# {intent.city}{intent.activity}场地调研 v{intent.version}")
    L.append(f"需求：{intent.brief()}｜生成时间 {report.generated_at}")
    if report.can_compare:
        lo = min(o.price for o in report.verified)  # type: ignore[type-var]
        hi = max(o.price for o in report.verified)  # type: ignore[type-var]
        L.append(f"结论：可核验报价 {len(report.verified)} 条 / {len(report.sources)} 个来源，"
                 f"人均 {lo}–{hi} 元；预算内 {len(report.within_budget)} 个。")
    else:
        L.append("结论：本轮**未形成比价**（不给出最优/最便宜）。原因：")
        L += [f"  - {r}" for r in report.blocked_reasons]
    L.append("")
    L.append("| # | 场地 | 区域 | 人均报价 | 核验状态 | 来源 |")
    L.append("|---|------|------|----------|----------|------|")
    for i, o in enumerate(report.offers, 1):
        price = f"{o.price} 元" if o.price is not None else "—"
        L.append(f"| {i} | {o.venue.name} | {o.venue.district} | {price} | "
                 f"{CONF_LABEL[o.confidence]} | {o.venue.url} |")
    L.append("")
    L.append("## 报价出处（原文片段）")
    if not report.verified:
        L.append("- （无：没有任何一条报价同时满足『原文逐字命中』）")
    for o in report.verified:
        for span in o.spans:
            L.append(f"- {o.venue.name}：`{span}`")
    L.append("")
    L.append("## 覆盖度与缺口")
    L.append(f"- 候选场地 {len(report.offers)} 个；已核验 {len(report.verified)} 个、"
             f"页面无报价 {report.price_not_found} 个、抓取失败 {report.fetch_failed} 个、"
             f"未覆盖 {report.not_fetched} 个")
    if report.degraded:
        L.append("- ⚠ 本轮结论**不完整**：存在未取证的场地，不代表这些场地更贵或更便宜。")
    return "\n".join(L)


def render_email_table(report: ComparisonReport) -> str:
    L = ["| # | 场地 | 区域 | 人均报价 | 核验状态 | 来源 |",
         "|---|------|------|----------|----------|------|"]
    for i, o in enumerate(report.offers, 1):
        price = f"{o.price} 元" if o.price is not None else "—"
        L.append(f"| {i} | {o.venue.name} | {o.venue.district} | {price} | "
                 f"{CONF_LABEL[o.confidence]} | {o.venue.url} |")
    return "\n".join(L)


def render_gaps(report: ComparisonReport, intent: Intent) -> str:
    L = ["【报价出处】"]
    if report.verified:
        for o in report.verified:
            L.append(f"- {o.venue.name}：" + "、".join(f"`{s}`" for s in o.spans)
                     + f"（{o.venue.url}）")
    else:
        L.append("- 无。本轮没有任何报价能在来源原文中逐字命中。")
    L.append("")
    L.append("【缺口说明】")
    missing = [o for o in report.offers if o.confidence != "verified"]
    if missing:
        for o in missing:
            L.append(f"- {o.venue.name}：{CONF_LABEL[o.confidence]}，无可用报价，不做估算")
    else:
        L.append("- 全部候选场地均已取得可核验报价。")
    L.append("")
    L.append(f"共 {len(report.offers)} 个候选、{len(report.verified)} 个可核验报价；"
             f"未取得报价的场地不参与任何排序，也不代表价格更高或更低。")
    return "\n".join(L)


# ------------------------------------------------------------------ 邮件起草


def safe_intro(report: ComparisonReport, intent: Intent, critique: list[str] | None = None) -> str:
    """代码模板引言：只使用报告里已经存在的数字，因此天然能过校验。"""
    if report.can_compare:
        lo = min(o.price for o in report.verified)  # type: ignore[type-var]
        hi = max(o.price for o in report.verified)  # type: ignore[type-var]
        return (f"关于{intent.city}{intent.activity}（{intent.headcount} 人），"
                f"本轮核验到 {len(report.verified)} 个带原文出处的报价，人均区间 {lo}–{hi} 元，"
                f"其中 {len(report.within_budget)} 个落在人均预算 "
                f"{intent.budget_per_person} 元以内。下表每条报价都附了来源；"
                f"未取得报价的场地单独标注，不做估算。")
    return (f"关于{intent.city}{intent.activity}（{intent.headcount} 人），"
            f"本轮**未能形成比价**：取得可核验报价的场地不足，无法给出任何排序或推荐。"
            f"下表列出已覆盖的场地与缺口原因，建议对无公开报价的场地直接电话询价。")


IntroFn = Callable[[ComparisonReport, Intent, list[str] | None], str]


def draft_email(report: ComparisonReport, intent: Intent, intro_fn: IntroFn,
                critique: list[str] | None = None, *, forced_template: bool = False
                ) -> tuple[str, str]:
    """生成邮件。表格永远由代码渲染，模型只能写引言。"""
    if report.can_compare:
        lo = min(o.price for o in report.verified)  # type: ignore[type-var]
        hi = max(o.price for o in report.verified)  # type: ignore[type-var]
        subject = (f"【{intent.city}{intent.activity}场地比价】"
                   f"{len(report.verified)} 个可核验报价，人均 {lo}-{hi} 元")
    else:
        subject = f"【{intent.city}{intent.activity}场地调研】未形成比价，附缺口清单"

    intro = safe_intro(report, intent, critique) if forced_template else intro_fn(report, intent, critique)

    lines = [intro, ""]
    if report.degraded:
        lines.append("（说明：本轮存在未取证的场地，结论不完整。）")
        lines.append("")
    lines += [render_email_table(report), "", render_gaps(report, intent), "",
              "—", "本邮件由调研 Agent 生成：表中每条报价均可追溯到来源原文；"
                   "未取得报价的场地不做估算或推测。"]
    return subject, "\n".join(lines)

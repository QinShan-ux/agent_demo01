"""校验层：出门前的最后一道闸。

三个检查互相独立，任一处不过就不放行：
  1. 报告自检      —— 数据结构内部一致性（"已核验"必须有证据，不可比价就不许有最优推荐）
  2. 禁用语检查    —— 不可比价时正文里不许出现"最便宜/最低价/性价比最高"这类结论词
  3. 数字溯源检查  —— 正文里每个报价数字都必须能对上某条证据；且
                     同一句里"场地名 + 价格"的配对必须与证据一致
"""
from __future__ import annotations

import re

from .extractors import parse_price
from .models import Intent
from .reporting import ComparisonReport

_NUM = r"\d[\d,]*(?:\.\d+)?"

# 报价类数字：带货币符号的，或带"元/块"量词的（"40 人""3000 平方米""2026-09-28"都不算）
PRICE_TOKEN_RE = re.compile(
    rf"[¥￥]\s?{_NUM}(?:\s*[-–~至]\s*[¥￥]?\s*{_NUM})?"
    rf"|{_NUM}(?:\s*[-–~至]\s*{_NUM})?\s*(?:元|块钱|块)"
)

FORBIDDEN_WHEN_NO_COMPARE = (
    "最便宜", "最低价", "性价比最高", "比价结论", "最优选择", "最划算", "全场最优",
)

SUPERLATIVE_WORDS = (
    "最便宜", "最低价", "性价比最高", "最优选择", "最划算", "全场最优", "推荐这家",
)

_SENTENCE_SPLIT = re.compile(r"[。；;\n]")


def find_price_tokens(text: str) -> list[str]:
    return [m.group(0).strip() for m in PRICE_TOKEN_RE.finditer(text)]


def numbers_in(token: str) -> list[int]:
    out: list[int] = []
    for m in re.finditer(_NUM, token):
        try:
            out.append(int(float(m.group(0).replace(",", ""))))
        except ValueError:
            pass
    return out


def _allowed_numbers(report: ComparisonReport, intent: Intent) -> set[int]:
    allowed: set[int] = set()
    for o in report.offers:
        for span in o.spans:
            p = parse_price(span)
            if p is not None:
                allowed.add(p)
    if intent.headcount:
        allowed.add(int(intent.headcount))
    if intent.budget_per_person:
        allowed.add(int(intent.budget_per_person))
    return allowed


def audit_report(report: ComparisonReport) -> list[str]:
    """报告自检：不允许出现"没有证据的价格"或"不可比价却有推荐"。"""
    problems: list[str] = []
    verified = [o for o in report.offers if o.confidence == "verified"]
    for o in verified:
        if o.price is None:
            problems.append(f"自检失败：{o.venue.name} 标记为已核验但没有价格")
        if not o.evidence_ids:
            problems.append(f"自检失败：{o.venue.name} 标记为已核验但没有证据编号")
    if report.can_compare and len(verified) < 2:
        problems.append("自检失败：判定为可比价，但可核验报价不足 2 条")
    if not report.can_compare and report.cheapest is not None:
        problems.append("自检失败：判定为不可比价，却给出了最优推荐")
    return problems


def verify_email(report: ComparisonReport, subject: str, body: str, intent: Intent) -> list[str]:
    violations: list[str] = []
    text = f"{subject}\n{body}"

    # 1) 不可比价时不许下结论
    if not report.can_compare:
        for word in FORBIDDEN_WHEN_NO_COMPARE:
            if word in text:
                violations.append(f"不可比价却出现结论性表述「{word}」")

    # 2) 每个报价数字都要有出处
    allowed = _allowed_numbers(report, intent)
    for tok in find_price_tokens(text):
        for n in numbers_in(tok):
            if n not in allowed:
                violations.append(f"正文出现无出处的报价数字 {n}（片段「{tok}」不在任何证据中）")

    # 3) 同句内的"场地名 + 价格"必须与证据配对一致
    price_of = {o.venue.name: o.price for o in report.offers if o.confidence == "verified"}
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        toks = find_price_tokens(sentence)
        if not toks:
            continue
        for name, price in price_of.items():
            if name not in sentence:
                continue
            for tok in toks:
                for n in numbers_in(tok):
                    if price is not None and n != price:
                        violations.append(
                            f"配对错误：「{name}」的证据报价是 {price}，正文写作 {n}")

    # 4) 声称"最优/最便宜"时，必须指向报告里真正的最优项
    if report.can_compare and report.cheapest is not None:
        best = report.cheapest.venue.name
        for sentence in _SENTENCE_SPLIT.split(text):
            if not any(w in sentence for w in SUPERLATIVE_WORDS):
                continue
            named = [n for n in price_of if n in sentence]
            if named and best not in named:
                violations.append(
                    f"结论与证据不符：正文称「{named[0]}」最优，但可核验报价中的最优是「{best}」")
            elif not named:
                violations.append(
                    f"结论缺少主语：出现最优表述但未指明场地，无法与证据核对"
                    f"（真正最优：{best}）")
    return violations

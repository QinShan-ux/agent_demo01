"""抽取层：把页面正文变成结构化报价候选。

关键契约（比价可信与否全在这里）：
抽取器**只能返回原文里逐字存在的片段**（span）。返回之后还要过 GroundingValidator，
校验 span 是不是 page_text 的连续子串。不是就丢掉。

所以这个层是"LLM 可以很蠢、可以幻觉，但幻觉进不了下游"的保险丝。
真实接入 LLM 时用 LLMExtractor，把模型输出当**候选**而不是**事实**。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Protocol

# 只用来"找候选"，最终是否采信由逐字校验决定
PRICE_SPAN_PATTERNS = [
    r"人均\s*[¥￥]?\s*\d[\d,]*(?:\.\d+)?\s*元?(?:\s*/\s*人)?",
    r"[¥￥]\s*\d[\d,]*(?:\.\d+)?\s*(?:\d[\d,]*\s*)?(?:元)?\s*/?\s*(?:人|每人|每位)?",
    r"\d[\d,]*(?:\.\d+)?\s*元\s*/?\s*人",
    r"\d[\d,]*(?:\.\d+)?\s*元\s*每人",
]

_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def parse_price(span: str) -> int | None:
    """从片段里取第一个数字（'人均 ¥ 280 元起' -> 280）。"""
    m = _ANY_NUMBER.search(span or "")
    if not m:
        return None
    try:
        return int(float(m.group(0).replace(",", "")))
    except ValueError:
        return None


class Extractor(Protocol):
    name: str

    def extract(self, page_text: str, venue_name: str) -> list[dict[str, Any]]:
        """返回 [{"field": "price_per_person", "span": "<原文连续子串>"}]"""
        ...


def _dedup(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in cands:
        span = (c.get("span") or "").strip()
        if not span or span in seen:
            continue
        seen.add(span)
        out.append({"field": c.get("field", "price_per_person"), "span": span})
    return out


class RuleExtractor:
    """离线可用：正则抽候选。返回的一定是原文子串。"""

    name = "rule"

    def extract(self, page_text: str, venue_name: str) -> list[dict[str, Any]]:
        cands: list[dict[str, Any]] = []
        for pat in PRICE_SPAN_PATTERNS:
            for m in re.finditer(pat, page_text):
                span = m.group(0).strip()
                if span in page_text:
                    cands.append({"field": "price_per_person", "span": span})
        return _dedup(cands)[:3]


class HallucinatingExtractor:
    """故意编造的抽取器，用来证明防幻觉闸门有效。

    注意它返回的 span 完全合法（格式正确、数字合理），唯一的问题是不在原文里。
    GroundingValidator 会全部丢掉。
    """

    name = "hallucinating"

    def __init__(self, fake_span: str = "人均 288 元") -> None:
        self.fake_span = fake_span

    def extract(self, page_text: str, venue_name: str) -> list[dict[str, Any]]:
        base = RuleExtractor().extract(page_text, venue_name)
        if base:
            return base
        # 页面上本来没有价格 —— 这里"读"出一个价格来
        return [{"field": "price_per_person", "span": self.fake_span}]


class LLMExtractor:
    """接真实 LLM 的抽取器。

    对 LLM 的要求只有一条：span 必须是输入文本的子串（原样复制，不要改写、不要换算、
    不要补单位）。这条要求仍然**不信任模型的自律** —— 返回后统一过逐字校验。
    """

    name = "llm"

    PROMPT = """从下面的网页正文中找出"人均团建报价"。
只输出 JSON 数组，每项形如 {{"field": "price_per_person", "span": "<原文片段>"}}。
span 必须原样复制正文里的连续文字，禁止改写、换算、补全或推测。
找不到就输出 []。

场地：{venue_name}
正文：
{page_text}"""

    def __init__(self, llm_call: Callable[[str], str]) -> None:
        self.llm_call = llm_call

    def extract(self, page_text: str, venue_name: str) -> list[dict[str, Any]]:
        raw = self.llm_call(self.PROMPT.format(venue_name=venue_name, page_text=page_text))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return _dedup([c for c in data if isinstance(c, dict)])


class GroundingValidator:
    """唯一的采信入口：span 必须是原文连续子串，否则判定为编造。"""

    @staticmethod
    def validate(candidates: list[dict[str, Any]],
                 page_text: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for c in candidates:
            span = (c.get("span") or "").strip()
            if span and span in page_text:
                accepted.append(c)
            else:
                rejected.append({**c, "reason": "span 不是来源正文的连续子串 → 判定为编造"})
        return accepted, rejected

"""护栏单元测试：python -m unittest discover -s tests -t . （在 team_building_agent 目录下）

重点验证 5 条不变式，而不是验证"能跑通"。
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from tb_agent.agent import AgentConfig, BuildingAgent  # noqa: E402
from tb_agent.extractors import (GroundingValidator, HallucinatingExtractor,  # noqa: E402
                                 RuleExtractor, parse_price)
from tb_agent.mock_tools import FaultInjector, build_world  # noqa: E402
from tb_agent.models import Evidence, Intent, ToolResult, ToolStatus, Venue  # noqa: E402
from tb_agent.reporting import build_report, draft_email, safe_intro  # noqa: E402
from tb_agent.verifier import audit_report, find_price_tokens, verify_email  # noqa: E402


class TestToolSemantics(unittest.TestCase):
    def test_empty_is_not_failed(self):
        """EMPTY 不应被当成失败，也不应被重试。"""
        empty = ToolResult(ToolStatus.EMPTY, "t", data=[])
        failed = ToolResult(ToolStatus.FAILED, "t", error="boom")
        self.assertFalse(empty.ok)
        self.assertFalse(empty.retryable, "空结果不重试：换参数才有意义")
        self.assertTrue(failed.retryable, "失败才重试")
        self.assertIsNot(empty.status, failed.status)

    def test_exception_becomes_failed_not_crash(self):
        reg = build_world(retries=1)[0]

        def boom(**kwargs):
            raise RuntimeError("网络炸了")

        reg.register("boom", boom)
        res = reg.invoke("boom")
        self.assertIs(res.status, ToolStatus.FAILED)
        self.assertEqual(res.attempts, 2, "重试 1 次 -> 共 2 次尝试")
        self.assertIn("网络炸了", res.error or "")

    def test_unregistered_tool_is_failed_not_ok(self):
        reg = build_world()[0]
        res = reg.invoke("not_exists")
        self.assertIs(res.status, ToolStatus.FAILED)


class TestGrounding(unittest.TestCase):
    PAGE = "紫蓬山国家森林公园团建拓展基地，场地可容纳 300 人。费用请致电咨询，暂不公布套餐价格。"

    def test_grounding_blocks_hallucinated_span(self):
        cands = HallucinatingExtractor(fake_span="人均 288 元").extract(self.PAGE, "紫蓬山")
        accepted, rejected = GroundingValidator.validate(cands, self.PAGE)
        self.assertEqual(accepted, [], "原文没有价格，任何价格都必须被丢弃")
        self.assertEqual(len(rejected), 1)
        self.assertIn("编造", rejected[0]["reason"])

    def test_grounding_accepts_real_span(self):
        page = "团建套餐 168 元/人，含温泉门票。"
        cands = RuleExtractor().extract(page, "X")
        accepted, rejected = GroundingValidator.validate(cands, page)
        self.assertTrue(accepted)
        self.assertEqual(rejected, [])
        self.assertEqual(parse_price(accepted[0]["span"]), 168)

    def test_parse_price_handles_currency_forms(self):
        self.assertEqual(parse_price("人均 ¥ 280 元起"), 280)
        self.assertEqual(parse_price("¥ 220 /人"), 220)
        self.assertEqual(parse_price("报价面议"), None)


class TestComparisonGate(unittest.TestCase):
    def _state_with(self, urls: list[str]):
        from tb_agent.models import RunState

        intent = Intent(city="合肥")
        state = RunState(intent=intent)
        for i, url in enumerate(urls):
            v = Venue.make(f"场地{i}", url, "某区", intent, {"location", "theme"}, "tc_x")
            state.candidates.append(v)
            state.fetched_keys.add(v.key)
            state.evidence.append(Evidence(
                evidence_id=f"ev{i}", venue_key=v.key, venue_name=v.name,
                field_name="price_per_person", value_span=f"{100 + i} 元/人",
                quote=f"人均 {100 + i} 元/人。", source_url=url, fetched_at="now",
                deps=frozenset({"location", "theme"}),
                dep_digest=intent.digest({"location", "theme"}), tool_call_id="tc_x"))
        return state, intent

    def test_compare_needs_two_independent_sources(self):
        """两条报价来自同一个 URL 时，不构成比价。"""
        state, intent = self._state_with(["https://x.com/a", "https://x.com/a"])
        report = build_report(state, intent)
        self.assertFalse(report.can_compare, "同一来源的两行数字不能当两家比价")
        self.assertIsNone(report.cheapest)
        self.assertTrue(any("独立来源" in r for r in report.blocked_reasons))

    def test_single_price_cannot_compare(self):
        state, intent = self._state_with(["https://x.com/a"])
        report = build_report(state, intent)
        self.assertFalse(report.can_compare)
        self.assertIsNone(report.cheapest)

    def test_two_sources_can_compare(self):
        state, intent = self._state_with(["https://x.com/a", "https://x.com/b"])
        report = build_report(state, intent)
        self.assertTrue(report.can_compare)
        self.assertEqual(report.cheapest.price, 100)  # type: ignore[union-attr]

    def test_unfetched_venue_gets_no_price(self):
        state, intent = self._state_with(["https://x.com/a"])
        v2 = Venue.make("未覆盖场地", "https://x.com/c", "某区", intent,
                        {"location", "theme"}, "tc_x")
        state.candidates.append(v2)
        report = build_report(state, intent)
        offer = [o for o in report.offers if o.venue.name == "未覆盖场地"][0]
        self.assertIsNone(offer.price, "未取证场地绝不能有价格")
        self.assertEqual(offer.confidence, "not_fetched")
        self.assertTrue(report.degraded)


class TestEmailVerifier(unittest.TestCase):
    def _report(self):
        from tb_agent.models import RunState

        intent = Intent(city="合肥", headcount=40, budget_per_person=200)
        state = RunState(intent=intent)
        for i, (name, url, price) in enumerate([
            ("A 场地", "https://x.com/a", 168),
            ("B 场地", "https://x.com/b", 128),
        ]):
            v = Venue.make(name, url, "某区", intent, {"location", "theme"}, "tc_x")
            state.candidates.append(v)
            state.fetched_keys.add(v.key)
            state.evidence.append(Evidence(
                evidence_id=f"ev{i}", venue_key=v.key, venue_name=name,
                field_name="price_per_person", value_span=f"{price} 元/人",
                quote=f"人均 {price} 元/人。", source_url=url, fetched_at="now",
                deps=frozenset({"location", "theme"}),
                dep_digest=intent.digest({"location", "theme"}), tool_call_id="tc_x"))
        return build_report(state, intent), intent

    def test_clean_email_passes(self):
        report, intent = self._report()
        subject, body = draft_email(report, intent, safe_intro)
        self.assertEqual(verify_email(report, subject, body, intent), [])
        self.assertEqual(audit_report(report), [])

    def test_unsourced_price_is_blocked(self):
        report, intent = self._report()
        subject, body = draft_email(report, intent, safe_intro)
        body = body.replace("关于合肥", "关于合肥（A 场地人均 99 元超划算）")
        v = verify_email(report, subject, body, intent)
        self.assertTrue(any("99" in x for x in v), f"无出处数字必须被拦：{v}")

    def test_wrong_superlative_is_blocked(self):
        report, intent = self._report()
        subject, body = draft_email(report, intent, safe_intro)
        body = body.replace("关于合肥", "关于合肥：A 场地性价比最高，建议直接定")
        v = verify_email(report, subject, body, intent)
        self.assertTrue(any("最优" in x or "不符" in x for x in v), f"{v}")

    def test_claim_blocked_when_cannot_compare(self):
        report, intent = self._report()
        report.can_compare = False          # 人为制造"不可比价"
        report.cheapest = None
        subject, body = draft_email(report, intent, safe_intro)
        body += "\n结论：B 场地最便宜。"
        v = verify_email(report, subject, body, intent)
        self.assertTrue(any("最便宜" in x for x in v), f"{v}")

    def test_price_like_tokens_ignore_headcount(self):
        self.assertEqual(find_price_tokens("团队 40 人，场地 3000 平方米"), [])
        self.assertEqual(find_price_tokens("人均 168 元"), ["168 元"])
        self.assertEqual(find_price_tokens("人均区间 168–280 元"), ["168–280 元"])


class TestRevisionInvalidation(unittest.TestCase):
    def _agent(self):
        registry, _ = build_world(FaultInjector(), retries=0)
        agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
        agent.set_intent(city="合肥", activity="团建", headcount=40)
        for _ in range(3):
            agent.step_once()
        return agent

    def test_location_change_invalidates_everything(self):
        agent = self._agent()
        before = len(agent.state.evidence)
        self.assertGreater(before, 0)
        note = agent.revise(city="南京", reason="算了，改成南京")
        self.assertEqual(agent.state.evidence, [], "换城市后旧报价必须全废")
        self.assertEqual(agent.state.candidates, [])
        self.assertIn("作废", note)
        self.assertTrue(agent.state.retired)

    def test_scale_change_keeps_evidence(self):
        agent = self._agent()
        before = len(agent.state.evidence)
        agent.revise(headcount=60, reason="人数改成 60")
        self.assertEqual(len(agent.state.evidence), before,
                         "只改人数不该作废报价证据（报价按人均计价）")
        self.assertEqual(agent.state.retired, [])

    def test_revision_grants_fresh_budget(self):
        agent = self._agent()
        used = agent._version_steps
        self.assertGreater(used, 0)
        agent.revise(city="南京")
        self.assertEqual(agent._version_steps, 0, "新子目标重新给 8 步预算")


class TestBudgetAndHonesty(unittest.TestCase):
    def test_budget_exhaustion_never_fabricates(self):
        registry, _ = build_world(FaultInjector(), retries=0)
        agent = BuildingAgent(registry, RuleExtractor(),
                              AgentConfig(max_steps_per_version=3, max_fetch_steps=9,
                                          fetch_batch=6))
        agent.set_intent(city="合肥", activity="团建")
        result = agent.run()
        self.assertEqual(result.status, "degraded")
        self.assertIsNotNone(result.report)
        for offer in result.report.offers:            # type: ignore[union-attr]
            if offer.price is not None:
                self.assertTrue(offer.evidence_ids, "任何价格都必须有证据编号")

    def test_search_total_failure_gives_up_cleanly(self):
        faults = FaultInjector(script={"search_venues": [ToolStatus.FAILED] * 6})
        registry, _ = build_world(faults, retries=2)
        agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
        agent.set_intent(city="合肥", activity="团建")
        result = agent.run()
        self.assertEqual(result.status, "give_up")
        self.assertIsNone(result.report)
        self.assertIsNone(result.email, "没有任何数据时不许发邮件")

    def test_fetch_failure_degrades_without_inventing_prices(self):
        faults = FaultInjector(script={"fetch_page": [ToolStatus.FAILED] * 12})
        registry, _ = build_world(faults, retries=2)
        agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
        agent.set_intent(city="合肥", activity="团建")
        result = agent.run()
        self.assertFalse(result.report.can_compare)   # type: ignore[union-attr]
        self.assertIsNone(result.report.cheapest)     # type: ignore[union-attr]
        self.assertEqual(result.report.fetch_failed, 4)  # type: ignore[union-attr]

    def test_empty_search_relaxes_instead_of_retrying(self):
        registry, _ = build_world(FaultInjector(), retries=2)
        agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
        agent.set_intent(city="合肥", activity="高端年会")
        result = agent.run()
        self.assertEqual(result.status, "sent")
        self.assertEqual(agent.state.search_rounds, 2, "首轮为空 + 放宽一轮")
        self.assertEqual(registry.retry_count, {}, "空结果不应触发重试")

    def test_low_confidence_result_requires_human_confirmation(self):
        faults = FaultInjector(script={"fetch_page": [ToolStatus.FAILED] * 12})
        registry, outbox = build_world(faults, retries=2)
        agent = BuildingAgent(registry, RuleExtractor(), AgentConfig())
        agent.set_intent(city="合肥", activity="团建")
        result = agent.run()
        self.assertEqual(result.status, "awaiting_confirmation")
        self.assertEqual(outbox, [], "未确认前不许真的发出去")
        agent.confirm_send()
        agent.run()
        self.assertEqual(len(outbox), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""离线可跑的工具实现（带可控故障注入）。

真实接入时把这三个类换成 HTTP/SMTP 客户端即可，ToolResult 契约不变：
联网检索、抓页面、发邮件。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import ToolResult, ToolStatus
from .tools import ToolRegistry

# --------------------------------------------------------------------- 语料

DB: dict[str, list[dict]] = {
    "合肥": [
        {
            "name": "巢湖半汤温泉度假区", "district": "巢湖市",
            "url": "https://example.com/hefei/bantang",
            "page": ("合肥巢湖半汤温泉度假区承接企业团建。团建套餐 168 元/人，含温泉门票、"
                     "自助午餐与会议室 2 小时。可同时容纳 120 人用餐。"),
        },
        {
            "name": "合肥融创万达文华酒店", "district": "包河区",
            "url": "https://example.com/hefei/wanda",
            "page": ("会议+团建打包方案：人均 ¥ 280 元起（40 人以上团队），含接驳车与团建教练。"
                     "另可加购宴会厅。"),
        },
        {
            "name": "紫蓬山拓展基地", "district": "肥西县",
            "url": "https://example.com/hefei/zipeng",
            "page": ("紫蓬山国家森林公园团建拓展基地，提供高空断桥、信任背摔等项目，"
                     "场地可容纳 300 人。费用请致电 0551-88886666 咨询，暂不公布套餐价格。"),
        },
        {
            "name": "三十岗桃蹊农场", "district": "庐阳区",
            "url": "https://example.com/hefei/taoxi",
            "page": "桃蹊农场团建烧烤套餐 128 元每人，含食材、炉具与场地布置，30 人起订。",
        },
        {
            "name": "大圩生态农业园", "district": "包河区",
            "url": "https://example.com/hefei/daxu",
            "page": "大圩生态园团建：人均 98 元起（农事体验 + 土灶饭）。团队规模 20-200 人。",
        },
        {
            "name": "滨湖会展中心团建馆", "district": "滨湖新区",
            "url": "https://example.com/hefei/binhu",
            "page": ("合肥滨湖国际会展中心 8 号馆，3000 平方米室内团建场地，可承接年会与运动会。"
                     "报价面议。"),
        },
    ],
    "南京": [
        {
            "name": "钟山体育公园拓展基地", "district": "玄武区",
            "url": "https://example.com/nanjing/zhongshan",
            "page": "南京钟山体育公园团建拓展基地，人均 150 元/人，含教练、器材与保险。",
        },
        {
            "name": "汤山温泉团建基地", "district": "江宁区",
            "url": "https://example.com/nanjing/tangshan",
            "page": "汤山温泉团建套餐：¥ 220 /人，含温泉、自助餐与会议室半天。",
        },
        {
            "name": "老山国家森林公园", "district": "浦口区",
            "url": "https://example.com/nanjing/laoshan",
            "page": "老山国家森林公园团建场地，可烧烤、徒步、露营。费用详询景区，无公开报价。",
        },
        {
            "name": "银杏湖乐园", "district": "江宁区",
            "url": "https://example.com/nanjing/yinxinghu",
            "page": "银杏湖乐园企业团建套票 168 元/人，含园区通票与团餐。",
        },
    ],
}


@dataclass
class FaultInjector:
    """按调用次序注入故障：script["search_venues"] = [TIMEOUT, OK, ...]"""

    script: dict[str, list[ToolStatus]] = field(default_factory=dict)
    _cursor: dict[str, int] = field(default_factory=dict)

    def take(self, tool: str, fallback: ToolStatus = ToolStatus.OK) -> ToolStatus:
        seq = self.script.get(tool)
        if not seq:
            return fallback
        i = self._cursor.get(tool, 0)
        status = seq[i] if i < len(seq) else fallback
        self._cursor[tool] = i + 1
        return status


# --------------------------------------------------------------------- 工具


class SearchTool:
    """联网检索：返回候选场地列表（数量不定，可能为 0）。"""

    name = "search_venues"

    def __init__(self, db: dict[str, list[dict]] | None = None,
                 faults: FaultInjector | None = None) -> None:
        self.db = db or DB
        self.faults = faults or FaultInjector()

    def __call__(self, city: str = "", keyword: str = "", wide: bool = False,
                 limit: int = 10) -> ToolResult:
        status = self.faults.take(self.name)
        if status is not ToolStatus.OK:
            return ToolResult(status, self.name, error=f"注入故障: {status.value}")

        rows = list(self.db.get(city, []))
        if keyword:
            rows = [r for r in rows if keyword in r["name"] or keyword in r["page"]]
        # 放宽条件时不再按关键词过滤，直接扩大召回
        rows = rows[:limit] if not wide else rows

        payload = [{"name": r["name"], "url": r["url"], "district": r["district"]} for r in rows]
        if not payload:
            return ToolResult(ToolStatus.EMPTY, self.name, data=[],
                              error=f"0 条候选（city={city}, keyword={keyword!r}）")
        return ToolResult(ToolStatus.OK, self.name, data=payload)


class FetchTool:
    """抓取场地详情页正文。"""

    name = "fetch_page"

    def __init__(self, db: dict[str, list[dict]] | None = None,
                 faults: FaultInjector | None = None) -> None:
        self.db = db or DB
        self.faults = faults or FaultInjector()
        self._index = {r["url"]: r["page"] for rows in self.db.values() for r in rows}

    def __call__(self, url: str = "") -> ToolResult:
        status = self.faults.take(self.name)
        if status is ToolStatus.OK:
            text = self._index.get(url)
            if not text:
                return ToolResult(ToolStatus.EMPTY, self.name, data={"url": url, "text": ""},
                                  error="页面存在但正文为空")
            return ToolResult(ToolStatus.OK, self.name, data={"url": url, "text": text})
        return ToolResult(status, self.name, error=f"注入故障: {status.value}")


class EmailTool:
    """发送邮件。失败时绝不返回成功。"""

    name = "send_email"

    def __init__(self, faults: FaultInjector | None = None,
                 sink: list[dict] | None = None) -> None:
        self.faults = faults or FaultInjector()
        self.sink = sink if sink is not None else []

    def __call__(self, to: str = "", subject: str = "", body: str = "") -> ToolResult:
        status = self.faults.take(self.name)
        if status is ToolStatus.OK:
            msg_id = f"<msg-{len(self.sink) + 1:03d}@example.com>"
            self.sink.append({"to": to, "subject": subject, "body": body, "message_id": msg_id})
            return ToolResult(ToolStatus.OK, self.name, data={"message_id": msg_id})
        return ToolResult(status, self.name, error=f"SMTP 异常: {status.value}")


def build_world(faults: FaultInjector | None = None, retries: int = 2):
    """组装：注册表 + 三个工具 + 邮件发件箱。"""
    faults = faults or FaultInjector()
    registry = ToolRegistry(retries=retries)
    search = SearchTool(faults=faults)
    fetch = FetchTool(faults=faults)
    outbox: list[dict] = []
    email = EmailTool(faults=faults, sink=outbox)
    registry.register(SearchTool.name, search)
    registry.register(FetchTool.name, fetch)
    registry.register(EmailTool.name, email)
    return registry, outbox

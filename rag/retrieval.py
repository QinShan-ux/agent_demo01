"""高级召回：BM25 关键词召回、RRF 混合融合、rerank 交叉编码器重排、多 query 改写。

依赖是单向的（vector_store / cli -> retrieval）：本模块只操作 sections（切片 dict 列表）、
hit 列表和 HTTP，不 import faiss、不 import embed，因此不会和 vector_store 循环引用。

切片 dict 约定（由 chunking 产出、vector_store 持久化）：
    {"text", "title", "headings"}                            heading 策略
    {"text", "title", "headings", "parent_idx", "parent"}     parent_child 策略

hit dict 约定（统一由 make_hit 构造）：
    {"score", "doc_idx", "title", "text", "headings"}
    parent_child 额外带 "parent_idx" 与 "child_text"（此时 "text" 已被换成父块全文）
    "score" 始终表示「当前的排序依据」，各路子分数挂在独立字段上，便于横向对比。
"""

import logging
import math
import os
import re
from collections import Counter, defaultdict

import jieba
import requests
from dotenv import load_dotenv

load_dotenv()

jieba.setLogLevel(logging.WARNING)  # 关掉首次构建词典缓存的日志，避免污染检索输出

K1 = 1.5  # BM25 词频饱和参数：越大，词频继续增长带来的收益越持久
B = 0.75  # BM25 长度归一化强度：0 完全不归一化，1 完全归一化
RRF_K = 60  # RRF 常数：越大头部越平缓，越不容易让某一路的第 1 名直接胜出
# 实测区分度最好的 rerank 模型：相关 0.897 / 不相关 0.000007
RERANK_MODEL = "rerank-multilingual-v3.0"
RERANK_TIMEOUT = 30

# 只要含字母/数字/汉字就算有效 token，用来滤掉 jieba 切出的纯标点和空白
_WORD_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """jieba 分词，丢掉纯标点/空白 token。统一小写，让英文词和缩写也能字面匹配。"""
    return [token for token in jieba.lcut(text.lower()) if _WORD_RE.search(token)]


class BM25:
    """内存版 Okapi BM25，用倒排表做稀疏打分。

    score(q,d) = Σ IDF(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
    IDF(t)     = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )

    语料只有几百条切片，每次进程启动重建一次（几十毫秒），因此不落盘缓存：
    多一个磁盘文件就要和 chunks.pkl、分词器版本保持同步失效，静默错位的代价远大于收益。
    """

    def __init__(self, sections: list[dict], k1: float = K1, b: float = B):
        self.k1, self.b = k1, b
        self.n = len(sections)
        self.doc_len: list[int] = []
        self.postings: dict[str, dict[int, int]] = {}  # term -> {doc_idx: tf}
        for doc_idx, section in enumerate(sections):
            tokens = tokenize(section["text"])
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings.setdefault(term, {})[doc_idx] = tf
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0

    def _idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        # 用 ln(1 + x) 形式：词出现在过半文档时 IDF 仍为正，不会把命中反噬成负贡献
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """返回 (doc_idx, score) 按分数降序。只累加查询词命中的文档，不做全量打分。"""
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_idx, tf in postings.items():
                denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += idf * tf * (self.k1 + 1) / denominator
        return sorted(scores.items(), key=lambda item: -item[1])[:top_k]


def make_hit(chunk: dict, doc_idx: int, score: float) -> dict:
    """hit 的唯一构造出口，保证向量和 BM25 两条召回路径产出的结构完全一致。

    parent_child 策略下 chunk["text"] 是被索引的子块，返回时替换成父块全文并保留命中的子块，
    所以下游必须读 hit["child_text"] 而不是 hit["text"] 来给「真正被召回的那段」打分/重排。
    """
    hit = {
        "score": float(score),
        "doc_idx": doc_idx,
        "title": chunk.get("title", ""),
        "text": chunk["text"],
        "headings": chunk.get("headings", []),
    }
    if "parent" in chunk:  # 父子块策略：命中子块，返回父块完整上下文
        hit["text"] = chunk["parent"]
        hit["child_text"] = chunk["text"]
        hit["parent_idx"] = chunk["parent_idx"]
    return hit


def bm25_search(
    query: str, bm25: BM25, sections: list[dict], top_k: int = 20
) -> list[dict]:
    """BM25 关键词召回。sections 必须和建 BM25 时是同一个列表，doc_idx 才能对得上。"""
    hits: list[dict] = []
    for doc_idx, score in bm25.search(query, top_k):
        hit = make_hit(sections[doc_idx], doc_idx, score)
        hit["bm25_score"] = float(score)
        hits.append(hit)
    return hits


def rrf_fuse(rankings: list[list[dict]], k: int = RRF_K, top_k: int = 20) -> list[dict]:
    """RRF 倒数排名融合：score(d) = Σ 1/(k + rank_r(d))，某一路未召回则贡献 0。

    只用排名不用分数：两路分数量纲根本不可比（余弦被 L2 归一化后挤在 0.3~0.7 的窄区间，
    BM25 无上界且随 query 浮动）。加权融合要先归一化，而归一化结果依赖候选池大小和 query；
    RRF 天然免疫量纲。代价是丢掉了幅度信息，所以各路子分数都保留下来供对比。
    """
    fused: dict[int, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            entry = fused.setdefault(
                hit["doc_idx"], {"hit": hit, "best_rank": rank, "rrf": 0.0, "subs": {}}
            )
            entry["rrf"] += 1.0 / (k + rank)
            if rank < entry["best_rank"]:  # 取排名更好的那次作代表，避免丢掉 parent 上下文
                entry["hit"], entry["best_rank"] = hit, rank
            for key, value in hit.items():
                if key.endswith("_score"):  # 两路子分数都留下，打印时能看到每路各排第几
                    entry["subs"][key] = value

    hits: list[dict] = []
    for entry in fused.values():
        hit = dict(entry["hit"])
        hit.update(entry["subs"])
        hit["rrf_score"] = entry["rrf"]
        hit["score"] = entry["rrf"]
        hits.append(hit)
    hits.sort(key=lambda item: -item["rrf_score"])
    return hits[:top_k]


def collapse_by_parent(hits: list[dict], top_k: int = 5, score_key: str = "score") -> list[dict]:
    """同一父块的多个子块只保留分最高的一个。

    用 max 而不是 sum 聚合：sum 会让子块多的大章节被系统性抬高（长度偏置）。
    heading 策略没有 parent_idx，退化成按切片自身去重，两种策略共用一套代码。
    """
    best: dict[int, dict] = {}
    for hit in hits:
        key = hit.get("parent_idx", hit["doc_idx"])
        if key not in best or hit[score_key] > best[key][score_key]:
            best[key] = hit
    ranked = sorted(best.values(), key=lambda item: -item[score_key])
    return ranked[:top_k]


def rerank(
    query: str, hits: list[dict], top_n: int = 5, model: str = RERANK_MODEL
) -> list[dict]:
    """调网关 /rerank 交叉编码器重排候选；失败时原样返回，保证召回结果不丢。

    送子块（child_text）而不是父块：交叉编码器窗口约 512 token，父块约 1000 字符会被截断，
    且截断点是任意的——真正命中的那句话可能正好被切掉，重排结果反而比融合更差。
    子块 ≤200 字符，稳稳在窗口内，且正是两路召回打分的那一单元。
    """
    if not hits:
        return hits

    documents = [hit.get("child_text", hit["text"]) for hit in hits]
    try:
        resp = requests.post(
            f"{os.environ['API_URL'].rstrip('/')}/rerank",
            headers={"Authorization": f"Bearer {os.environ['API_KEY']}"},
            json={"model": model, "query": query, "documents": documents},
            timeout=RERANK_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:  # 网关对不支持的模型返回 {"error": "unsupported model"}
            raise ValueError(payload["error"])
        results = payload["results"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        # rerank 是增强项不是硬依赖：不加 retry 和熔断，回退到融合排序即可
        print(f"  [warn] rerank 调用失败，回退到融合排序：{exc}")
        return hits[:top_n]

    reranked: list[dict] = []
    seen: set[int] = set()
    for item in results:
        # index 只是本次请求 documents 数组里的下标：网关可能少回/重复回/越界，一律按它对齐
        index, score = item.get("index"), item.get("relevance_score")
        if not isinstance(index, int) or score is None:
            continue
        if not 0 <= index < len(hits) or index in seen:
            continue
        seen.add(index)
        hit = dict(hits[index])
        hit["rerank_score"] = float(score)
        hit["score"] = hit["rerank_score"]
        reranked.append(hit)

    # 网关只回部分结果时，剩下的按原顺序补在后面，保证条数稳定
    reranked += [hit for index, hit in enumerate(hits) if index not in seen]
    return reranked[:top_n]


def generate_multi_queries(query: str, llm, num_queries: int = 3) -> list[str]:
    """用 LLM 把一个问题改写成多个视角的查询，提高召回覆盖（返回时保留原 query 在首位）。

    llm 由调用方传入，本模块不 import agent，避免把检索逻辑和 agent 脚手架绑在一起。
    """
    prompt = f"""你是一个AI助手，负责生成多个不同视角的搜索查询。
给定一个用户问题，生成{num_queries}个不同但相关的查询，以帮助检索更全面的信息。

原始问题: {query}

请直接输出{num_queries}个查询，每行一个，不要编号和其他内容:"""

    response = llm.invoke(prompt)
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return [query] + queries[:num_queries]

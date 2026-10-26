"""检索 demo 入口：建/加载索引，然后并排对比几种召回方式的效果。

召回方式（--mode）：
  vector  纯向量召回
  hybrid  向量 + BM25，RRF 倒数排名融合
  rerank  hybrid 之后再走交叉编码器重排

各模块分工：
  rag/chunking.py      切片策略（按标题切 / 父子块）
  rag/vector_store.py  embedding + faiss 索引构建与持久化 + 向量召回
  rag/retrieval.py     BM25 / RRF / rerank 高级召回 + 多 query 改写
  rag/pipeline.py      完整问答链路与回答模板，HTTP 接口（routers/rag_router.py）共用同一份

BM25 直接从缓存的切片现建，改召回方式不需要重建向量索引。重建：--force

每个 query 打印完各召回结果后，取当前召回方式的前 3 条切片组装进回答模板，交给 LLM 生成带引用的回答。
召回与 RagPipeline 同阈值口径（余弦 / BM25 各 0.6），过滤后无命中时打印拒答话术、不调 LLM。

依赖：pip install faiss-cpu jieba
用法：python -m rag.cli [--strategy parent_child] [--mode all]
"""

import argparse

from agent import create_model
from rag.pipeline import ANSWER_TOP_N, NO_HIT_ANSWER, RagPipeline, build_answer_prompt
from rag.retrieval import (
    BM25,
    RERANK_MODEL,
    RRF_K,
    bm25_search,
    collapse_by_parent,
    generate_multi_queries,
    rerank,
    rrf_fuse,
)
from rag.vector_store import build_index, dense_search, embed


def _print_hits(label: str, hits: list[dict], min_score: float = None) -> None:
    """打印一路召回结果。那行子分数（dense/bm25/rrf/rerank）是重点：它解释排序为什么变。"""
    print(f"\n---------- {label} ----------")
    shown = [h for h in hits if min_score is None or h["score"] >= min_score]
    if not shown:
        print("  （无结果）")
        return
    for rank, hit in enumerate(shown, start=1):
        print(f"\n{rank}. [{hit['score']:.4f}] {hit['title']}\n{hit['text']}")
        if "child_text" in hit:
            print(f"  └ 命中的子块：{hit['child_text']}")
        detail = [
            f"{key.removesuffix('_score')}={hit[key]:.4f}"
            for key in ("dense_score", "bm25_score", "rrf_score", "rerank_score")
            if key in hit
        ]
        if len(detail) > 1:  # 单路召回时子分数和方括号里的分数重复，不打印
            print(f"  {' '.join(detail)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="markdown -> faiss 向量索引 + 高级召回对比")
    parser.add_argument("-f", "--force", action="store_true", help="忽略已有索引，重新切片并向量化")
    parser.add_argument("-q", "--query", default="如何坚持和发展中国特色社会主义", help="用来验证检索的 query")
    parser.add_argument(
        "--strategy",
        choices=["heading", "parent_child"],
        default="parent_child",
        help="切片策略：heading=按标题切，parent_child=父子块（子块检索、父块返回）",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "vector", "hybrid", "rerank"],
        default="rerank",
        help="对比哪些召回方式：vector=纯向量，hybrid=向量+BM25，rerank=hybrid 后再重排",
    )
    parser.add_argument("--top-k", type=int, default=5, help="每种模式最终返回条数")
    parser.add_argument("--candidates", type=int, default=20, help="每路召回深度，也是 rerank 的候选池")
    parser.add_argument("--rrf-k", type=int, default=RRF_K, help="RRF 常数，越大头部越平缓")
    parser.add_argument("--rerank-model", default=RERANK_MODEL, help="网关 rerank 模型 id")
    parser.add_argument(
        "--min-score", type=float, default=None, help="按当前排序依据过滤，默认不过滤（RRF 分数很小，别沿用余弦阈值）"
    )
    parser.add_argument(
        "--multi-query", action="store_true", help="开启 LLM 多 query 改写（默认关，输出块数会翻几倍）"
    )
    args = parser.parse_args()

    # 构建向量
    index, sections = build_index(force=args.force, strategy=args.strategy)
    bm25 = BM25(sections)  # 一次进程建一次，多 query 复用
    llm = create_model()  # 多 query 改写和最终回答共用一个实例

    queries = [args.query]
    if args.multi_query:
        queries = generate_multi_queries(args.query, llm, num_queries=3)

    for query in queries:
        print(f"\n{'=' * 20} {query} {'=' * 20}")
        # query embed
        query_vec = embed([query], verbose=False)  # 几种模式共用一次 query 编码
        dense = dense_search(query, index, sections, args.candidates, query_vec=query_vec)
        bm25_hits = bm25_search(query, bm25, sections, args.candidates)
        # 与 RagPipeline.retrieve 同一阈值口径：余弦 / BM25 先过滤再融合，
        # 弱命中进不了融合结果，也就进不了回答 —— 不打阈值的话后面的拒答闸形同虚设
        dense = [d for d in dense if d["dense_score"] > RagPipeline.DENSE_MIN_SCORE]
        bm25_hits = [b for b in bm25_hits if b["bm25_score"] > RagPipeline.BM25_MIN_SCORE]
        fused = rrf_fuse(
            [dense, bm25_hits],
            k=args.rrf_k,
            top_k=args.candidates,
        )

        if args.mode in ("all", "vector"):
            _print_hits("vector 纯向量召回", collapse_by_parent(dense, args.top_k), args.min_score)
        if args.mode in ("all", "hybrid"):
            _print_hits(
                "hybrid 向量 + BM25（RRF 融合）", collapse_by_parent(fused, args.top_k), args.min_score
            )
        if args.mode in ("all", "rerank"):
            reranked = rerank(query, fused, top_n=args.candidates, model=args.rerank_model)
            _print_hits(
                "hybrid + rerank 交叉编码器重排", collapse_by_parent(reranked, args.top_k), args.min_score
            )

        # 用当前召回方式的排名组装 prompt：all 时取整条链路最强的那一路
        if args.mode in ("all", "rerank"):
            ranked = reranked
        elif args.mode == "hybrid":
            ranked = fused
        else:
            ranked = dense
        # 先按返回单元折叠再截前 3：父子块下直接取前 3 条可能全是同一个父块的子块
        top_hits = collapse_by_parent(ranked, ANSWER_TOP_N)
        if not top_hits:  # 与 stream_ask 同一道闸：阈值过滤后无命中绝不调 LLM，防瞎编
            print(f"\n---------- 拒答 ----------\n{NO_HIT_ANSWER}")
            continue
        prompt = build_answer_prompt(query, top_hits)
        print(f"\n---------- LLM 回答（送入 {len(top_hits)} 条切片） ----------")
        print(llm.invoke(prompt).content)

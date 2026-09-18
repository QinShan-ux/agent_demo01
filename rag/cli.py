"""检索 demo 入口：建/加载索引，然后并排对比几种召回方式的效果。

召回方式（--mode）：
  vector  纯向量召回
  hybrid  向量 + BM25，RRF 倒数排名融合
  rerank  hybrid 之后再走交叉编码器重排

各模块分工：
  rag/chunking.py      切片策略（按标题切 / 父子块）
  rag/vector_store.py  embedding + faiss 索引构建与持久化 + 向量召回
  rag/retrieval.py     BM25 / RRF / rerank 高级召回 + 多 query 改写

BM25 直接从缓存的切片现建，改召回方式不需要重建向量索引。重建：--force

依赖：pip install faiss-cpu jieba
用法：python -m rag.cli [--strategy parent_child] [--mode all]
"""

import argparse

from agent import create_model
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
    parser.add_argument("-q", "--query", default="资本主义经济危机爆发的原因？", help="用来验证检索的 query")
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

    index, sections = build_index(force=args.force, strategy=args.strategy)
    bm25 = BM25(sections)  # 一次进程建一次，多 query 复用

    queries = [args.query]
    if args.multi_query:
        queries = generate_multi_queries(args.query, create_model(), num_queries=3)

    for query in queries:
        print(f"\n{'=' * 20} {query} {'=' * 20}")
        # query embed
        query_vec = embed([query], verbose=False)  # 几种模式共用一次 query 编码
        dense = dense_search(query, index, sections, args.candidates, query_vec=query_vec)
        fused = rrf_fuse(
            [dense, bm25_search(query, bm25, sections, args.candidates)],
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

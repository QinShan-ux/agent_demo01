"""RAG 检索包：切片 -> 向量化 -> 建索引 -> 召回 -> 融合 -> 重排。

各模块分工：
    chunking.py     切片策略（按标题切 / 父子块），纯文本处理，只依赖 re
    vector_store.py embedding + faiss 索引构建与持久化 + 向量召回
    retrieval.py    BM25 / RRF / rerank 高级召回 + 多 query 改写
    cli.py          检索效果对比的命令行入口，`python -m rag.cli`

这里刻意不做 re-export：`import rag` 不该连带把 faiss、jieba、agent 全部拉起来。
需要什么就显式 `from rag.vector_store import build_index`。
"""

"""RAG 检索能力包：切片 -> 向量化 -> 建索引 -> 召回 -> 融合 -> 重排 -> 生成。

各模块分工：
    chunking.py          切片策略（按标题切 / 父子块），纯文本处理，只依赖 re
    ppt_chunking.py      PPT 讲义的切片与表格文本（课件来源）
    vector_store.py      embedding + faiss 索引构建与持久化 + 向量召回
    retrieval.py         BM25 / RRF / rerank 高级召回 + 多 query 改写
    pipeline.py          完整问答链路（prompt 的模板 / 路由 / 拆解在 prompting 包）
    correction_memory.py 纠错原问题的向量库（MySQL 是对账基准）
    cli.py               检索效果对比的命令行入口，`python -m rag.cli`

这是**能力包**：不含 HTTP，也不含表定义与业务编排。表在 models/tables/，库的读写
在 repositories/，一次提问的编排在 services/rag_service.py，入口在 api/routers/rag_router.py。

这里刻意不做 re-export：`import rag` 不该连带把 faiss、jieba、agent 全部拉起来。
需要什么就显式 `from rag.vector_store import build_index`。
"""

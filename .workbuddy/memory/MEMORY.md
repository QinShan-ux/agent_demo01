# 项目约定 — agent_demo01

## rag 包结构

`切片 -> 向量化 -> 建索引 -> 召回 -> 融合 -> 重排`，模块单向依赖：
`chunking` / `acl`（不依赖 faiss）→ `vector_store` → `retrieval` → `cli`。
`retrieval.py` 只操作 sections / hit 列表和 HTTP，不 import faiss 也不 import embed，避免与 vector_store 循环引用。

## 数据分层（data/faiss/）

- `index.faiss`：只存向量，无任何可读信息
- `chunks.pkl`：切片清单（406 条），第 i 条按位置对应第 i 行向量，靠顺序对齐，不建 id
- `meta.json`：索引级清单，**只有 1 条**，作用是缓存失效判定（比 embed_model / strategy / source_sha256）。不是切片元数据

切片字段约定（两条策略一致，改动需同步 `chunking.py` 与 `retrieval.py` 的 docstring）：
`{"text", "title", "headings"}` + parent_child 额外 `{"parent_idx", "parent"}`。
`text` 开头的标题路径是从 `headings` 拼进去的，改 `title` 必须同步 `text` 前缀。

## 踩坑记录

- **`meta.json` 的指纹只跟源 markdown 走，不含切片逻辑**。改了 `chunking.py` 必须手动 `--force`，否则静默命中旧缓存。

## 权限控制（设计过但已回滚，代码未落地）

2026-09-18 实现过一版基于元数据的角色过滤，当天按用户要求全部撤销。以下是讨论中确定的设计结论，将来要重做时直接照此实现：

- 角色 `ROLES = ("user", "admin")` 由低到高按等级比较；`default: "admin"`（用户明确要求），fail-closed
- **标签不进 `text`**：拼进 text 就得重烧 embedding，权限要频繁调，必须外挂（如 acl.json）
- **粒度 ≥ 返回粒度**：parent_child 返回父块全文，判权必须按返回单元，不能按子块
- **过滤位置：两路召回之后、`rrf_fuse` 之前**（`retrieval.rerank` 会把候选正文发给外部网关）
- rules 的 key 用标题路径，逐级回退 = 最长前缀优先，配一级标题即覆盖整章

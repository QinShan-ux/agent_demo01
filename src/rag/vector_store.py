"""源文本 -> embedding -> faiss 索引：切片、向量化、按策略持久化，以及最基础的向量召回。

切片策略在 rag/chunking.py，BM25 / RRF / rerank 等高级召回在 rag/retrieval.py。
本模块只管「源文本 -> 索引」这条链路，外加向量召回 dense_search。

索引按策略持久化到各自目录：heading -> data/faiss/lesson1，parent_child -> data/faiss/lesson1_parent_child。
两者互不覆盖，切换策略时直接加载对应目录的缓存，无需重新向量化。每个目录里三样东西：
    index.faiss    向量（IndexIDMap，id = 切片下标）
    chunks.pkl     切片清单（二进制，给机器读）
    chunks.json    同一份切片清单的 JSON 版，给肉眼看切分效果
    meta.json      缓存失效判定用的指纹

目录名必须是纯 ASCII：faiss 的 C++ 实现在 Windows 上用 ANSI 编码 fopen 路径，
中文目录会被改写成乱码文件名（甚至把路径分隔符吃掉）或直接报 ENOENT。

数据不在本包内：源 markdown 在 data/pdf/，索引在 data/faiss/。

依赖：pip install faiss-cpu
"""

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

from core import config
from rag.chunking import split_markdown, split_parent_child
from rag.ppt_chunking import embed_text, split_ppt_parent_child
from rag.retrieval import make_hit
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)
MD_PATH = config.DATA_DIR / "pdf" / "必修markidown.md"
# 两种切片策略分目录保存，切换时直接加载各自缓存，无需重新向量化
# 目录名保持 ASCII：faiss 在 Windows 上按 ANSI 解码路径，中文目录写不进去
OUT_DIR = config.FAISS_DIR / "lesson1"  # heading（按标题）索引目录
PARENT_CHILD_OUT_DIR = config.FAISS_DIR / "lesson1_parent_child"  # parent_child（父子块）索引目录
PPT_DIR = config.DATA_DIR / "ppt" / "ppt_images"  # PPT 讲义源 markdown 目录（每份讲义一个 .md）
PPT_OUT_DIR = config.FAISS_DIR / "ppt"  # PPT 讲义索引目录，与教材索引互不覆盖

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"
CHUNKS_JSON_FILE = "chunks.json"
META_FILE = "meta.json"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 32
EMBED_DIM = 1024

client = OpenAI(api_key=config.API_KEY, base_url=config.API_URL)


@retry(
    retry=retry_if_exception_type((
            APIConnectionError, APITimeoutError, RateLimitError
    )),
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    reraise=True,
)
def _embed_batch(batch: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model=EMBED_MODEL, input=batch, dimensions=EMBED_DIM
    )
    items = sorted(resp.data, key=lambda x: x.index)
    return [it.embedding for it in items]


def embed(texts: list[str], verbose: bool = True) -> np.ndarray:
    """批量取向量，并做 L2 归一化（配合 IndexFlatIP 等价于余弦相似度）。"""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    vectors: list[list[float]] = []
    total = len(texts)
    for i in range(0, total, BATCH_SIZE):
        vectors.extend(_embed_batch(texts[i: i + BATCH_SIZE]))
        if verbose:
            logger.info("已向量化 %d/%d", min(i + BATCH_SIZE, total), total)

    arr = np.ascontiguousarray(vectors, dtype="float32")
    if arr.shape[1] != EMBED_DIM:
        raise ValueError(f"返回维度 {arr.shape[1]} != EMBED_DIM={EMBED_DIM}")
    if not np.isfinite(arr).all():
        raise ValueError("embedding 含 nan/inf")
    if np.any(np.linalg.norm(arr, axis=1) == 0):
        raise ValueError("存在零向量，检查 embedding 是否异常")

    faiss.normalize_L2(arr)
    return arr


def new_index(dim: int):
    """IndexIDMap 包住 IndexFlatIP，向量自带显式 id。

    id 用切片在 sections 里的位置下标，不是另抽的哈希：doc_idx 是向量召回与 BM25 召回
    共用的对齐键，BM25 侧给的就是位置下标，id 换成别的值两条路径就对不上、融合去重会错。
    包这一层换来的是能按 id 定向删改（remove_ids），不必为动一条切片重建整个索引。
    """
    return faiss.IndexIDMap(faiss.IndexFlatIP(dim))


def _fingerprint(text: str) -> dict:
    return {
        "source_size": len(text),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def write_chunks_json(out_dir: Path, sections: list[dict], meta: dict) -> Path:
    """把切片清单写成索引目录下的 chunks.json，用来肉眼审阅切分效果。

    chunks.pkl 是给机器读的，没法直接翻；这份 JSON 与它同源同序，就是同一个列表，
    所以 "id" 等于切片下标，也就等于 faiss 里的向量 id —— 检索命中的 id、纠错记录里的 id
    都能对到这里。字段原样保留（parent_child 的 "parent" 不缩掉）：一条子块被召回时到底
    返回什么，正是浏览切分效果时要看的东西。
    """
    payload = {
        "strategy": meta["strategy"],
        "source": meta["source"],
        "embed_model": meta["embed_model"],
        "dim": meta["dim"],
        "count": meta["count"],
        "created_at": meta["created_at"],
        "chunks": [{"id": i, **section} for i, section in enumerate(sections)],
    }
    path = out_dir / CHUNKS_JSON_FILE
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _out_dir_for(strategy: str) -> Path:
    """切片策略对应的索引保存目录：heading 与 parent_child 分目录，互不覆盖。"""
    return PARENT_CHILD_OUT_DIR if strategy == "parent_child" else OUT_DIR


def _ensure_ascii_dir(path: Path) -> None:
    """faiss 底层用 ANSI 编码 fopen 路径，中文目录会被改写成乱码文件名或直接 ENOENT，提前拦掉。"""
    if not str(path).isascii():
        raise ValueError(f"faiss 索引目录必须是纯 ASCII，请改用英文目录名：{path}")


def index_info(strategy: str) -> dict:
    """只读索引现状（不触发构建、不加载向量），供接口向前端列出可选项。

    ready 为 False 表示这个策略的索引还没建过：选它会触发一次同步构建（要重新向量化整份
    文档），所以前端应当把这类选项禁用，别让用户点了干等。
    """
    out_dir = _out_dir_for(strategy)
    try:
        meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ready": False, "count": 0, "created_at": None}
    return {
        "ready": True,
        "count": int(meta.get("count", 0)),
        "created_at": meta.get("created_at"),
    }


def load_index(md_path: Path = MD_PATH, out_dir: Path = None, strategy: str = "heading"):
    """索引存在且源文件、embedding 模型、切片策略都没变时直接加载，返回 (index, sections)，否则返回 None。"""
    out_dir = _out_dir_for(strategy) if out_dir is None else Path(out_dir)
    _ensure_ascii_dir(out_dir)
    try:
        meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
        index = faiss.read_index(str(out_dir / INDEX_FILE))
        with open(out_dir / CHUNKS_FILE, "rb") as f:
            sections = pickle.load(f)
    except (OSError, ValueError, KeyError, RuntimeError, pickle.UnpicklingError):
        return None

    if not isinstance(index, faiss.IndexIDMap):
        # meta.json 的指纹只跟源 markdown 走，不含索引结构，所以这里按实际索引类型判定。
        # 旧的无 id 索引仍能搜，但没有可寻址的 id，直接判为需要重建。
        logger.warning("索引未使用 IndexIDMap，需要重建")
        return None

    if meta.get("embed_model") != EMBED_MODEL:
        logger.warning("embedding 模型已从 %s 变为 %s，需要重建", meta.get("embed_model"), EMBED_MODEL)
        return None

    if meta.get("strategy", "heading") != strategy:
        logger.warning("切片策略已从 %s 变为 %s，需要重建", meta.get("strategy", "heading"), strategy)
        return None

    current = _fingerprint(Path(md_path).read_text(encoding="utf-8"))
    if meta.get("source_sha256") != current["source_sha256"]:
        logger.warning("源文件已变更，需要重建")
        return None

    return index, sections


def build_index(md_path: Path = MD_PATH, out_dir: Path = None, force: bool = False, strategy: str = "heading"):
    out_dir = _out_dir_for(strategy) if out_dir is None else Path(out_dir)
    _ensure_ascii_dir(out_dir)
    if not force:
        # 如果已经向量化了，就获取现有的缓存
        cached = load_index(md_path, out_dir, strategy)
        if cached is not None:
            index, sections = cached
            logger.info("命中已有索引，直接加载：%d 段，共 %d 条向量", len(sections), index.ntotal)
            return cached

    start = time.perf_counter()
    md_path = Path(md_path)
    # read file
    text = md_path.read_text(encoding="utf-8")

    # father and child chunk strategy
    if strategy == "parent_child":
        parents, sections = split_parent_child(text)
        logger.info("切片完成（父子块）：%d 个父块，%d 个子块", len(parents), len(sections))
    else:
        sections = split_markdown(text)
        logger.info("切片完成：%d 段", len(sections))

    # 来源标记随切片一起持久化（chunks.pkl / chunks.json）：课本 book，课件 ppt
    for section in sections:
        section["from"] = "book"

    vectors = embed([s["text"] for s in sections])
    index = new_index(vectors.shape[1])
    index.add_with_ids(vectors, np.arange(len(sections), dtype="int64"))

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    with open(out_dir / CHUNKS_FILE, "wb") as f:
        pickle.dump(sections, f)

    meta = {
        "source": str(md_path),
        "embed_model": EMBED_MODEL,
        "strategy": strategy,
        "dim": int(vectors.shape[1]),
        "count": int(index.ntotal),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **_fingerprint(text),
    }
    (out_dir / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    json_path = write_chunks_json(out_dir, sections, meta)

    logger.info(
        "索引已保存到 %s（dim=%d，共 %d 条），切片清单见 %s，重建用时 %.1fs",
        out_dir, vectors.shape[1], index.ntotal, json_path.name, time.perf_counter() - start,
    )
    return index, sections


def _ppt_fingerprint(files: list[Path]) -> dict:
    """PPT 讲义的源指纹：逐个文件累积 文件名+内容 的 sha256。"""
    digest = hashlib.sha256()
    total = 0
    for p in files:
        content = p.read_text(encoding="utf-8")
        digest.update(p.name.encode("utf-8"))
        digest.update(content.encode("utf-8"))
        total += len(content)
    return {"source_size": total, "source_sha256": digest.hexdigest()}


def build_ppt_index(force: bool = False):
    """PPT 讲义（data/ppt/ppt_images/*.md）单独建索引，输出到 data/faiss/ppt/。

    与教材索引完全独立：源是多个 md，切片走 ppt_chunking（表格感知），
    向量化文本 = 子块 text + 序列化表格（embed_text），保证表格内容可被检索。
    缓存判定指纹 = 全部 md 文件的文件名+内容；任一讲义改动即整库重建。
    chunks.json 里的切片就是 chunks.pkl 的同一列表，id = 下标 = faiss 向量 id。
    """
    md_files = sorted(PPT_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"{PPT_DIR} 下没有 .md 文件")

    if not force:
        try:
            meta = json.loads((PPT_OUT_DIR / META_FILE).read_text(encoding="utf-8"))
            index = faiss.read_index(str(PPT_OUT_DIR / INDEX_FILE))
            with open(PPT_OUT_DIR / CHUNKS_FILE, "rb") as f:
                sections = pickle.load(f)
            if (
                meta.get("strategy") == "ppt"
                and meta.get("embed_model") == EMBED_MODEL
                and meta.get("source_sha256") == _ppt_fingerprint(md_files)["source_sha256"]
            ):
                logger.info("命中已有 PPT 索引，直接加载：%d 段，共 %d 条向量", len(sections), index.ntotal)
                return index, sections
        except (OSError, ValueError, RuntimeError, pickle.UnpicklingError):
            pass  # 索引不存在或指纹不符，走下面的重建

    start = time.perf_counter()
    sections: list[dict] = []
    total_parents = 0
    table_chunks = 0
    for path in md_files:
        parents, children = split_ppt_parent_child(path.read_text(encoding="utf-8"))
        total_parents += len(parents)
        table_chunks += sum(1 for c in children if c.get("has_table"))
        sections.extend(children)
        logger.info("%s：切片完成，%d 个父块，%d 个子块", path.name, len(parents), len(children))

    # 来源标记随切片一起持久化（chunks.pkl / chunks.json）：课本 book，课件 ppt
    for section in sections:
        section["from"] = "ppt"

    vectors = embed([embed_text(s) for s in sections])
    index = new_index(vectors.shape[1])
    index.add_with_ids(vectors, np.arange(len(sections), dtype="int64"))

    PPT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(PPT_OUT_DIR / INDEX_FILE))
    with open(PPT_OUT_DIR / CHUNKS_FILE, "wb") as f:
        pickle.dump(sections, f)

    meta = {
        "source": str(PPT_DIR),
        "embed_model": EMBED_MODEL,
        "strategy": "ppt",
        "dim": int(vectors.shape[1]),
        "count": int(index.ntotal),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **_ppt_fingerprint(md_files),
    }
    (PPT_OUT_DIR / META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    json_path = write_chunks_json(PPT_OUT_DIR, sections, meta)

    logger.info(
        "PPT 索引已保存到 %s（%d 份讲义，%d 个父块，%d 个子块，其中含表格 %d 条，dim=%d），"
        "切片清单见 %s，重建用时 %.1fs",
        PPT_OUT_DIR, len(md_files), total_parents, len(sections), table_chunks,
        vectors.shape[1], json_path.name, time.perf_counter() - start,
    )
    return index, sections


def dense_search(
        query: str, index, sections: list[dict], top_k: int = 5,
        query_vec: np.ndarray = None,
) -> list[dict]:
    """向量召回，走 make_hit 构造 hit（与 BM25 召回结构一致）。

    query_vec 可传入复用：多种召回方式并排对比时，query 只编码一次。
    索引是 IndexIDMap，返回的 id 就是切片在 sections 里的位置下标，可直接取 sections[id]。
    来源标记 from 由切片自带（make_hit 从 chunk 带出），这里不重复设置。
    """
    if query_vec is None:
        query_vec = embed([query], verbose=False)
    scores, ids = index.search(query_vec, top_k)
    hits: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:  # top_k > 索引条数时，faiss 用 -1 补齐
            continue
        hit = make_hit(sections[int(idx)], int(idx), float(score))
        hit["dense_score"] = float(score)
        hits.append(hit)
    return hits

"""源文本 -> embedding -> faiss 索引：切片、向量化、按策略持久化，以及最基础的向量召回。

切片策略在 rag/chunking.py，BM25 / RRF / rerank 等高级召回在 rag/retrieval.py。
本模块只管「源文本 -> 索引」这条链路，外加向量召回 dense_search。

索引按策略持久化到各自目录：heading -> data/faiss/第一课，parent_child -> data/faiss/第一课_parent_child。
两者互不覆盖，切换策略时直接加载对应目录的缓存，无需重新向量化。

数据不在本包内：源 markdown 在 utils/clent_text/，索引在顶层 data/faiss/。

依赖：pip install faiss-cpu
"""

import hashlib
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError

from rag.chunking import split_markdown, split_parent_child
from rag.retrieval import make_hit
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

load_dotenv()
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
MD_PATH = BASE_DIR / "utils" / "clent_text" / "第一课.clean.md"
# 两种切片策略分目录保存，切换时直接加载各自缓存，无需重新向量化
OUT_DIR = BASE_DIR / "data" / "faiss" / "第一课"  # heading（按标题）索引目录
PARENT_CHILD_OUT_DIR = BASE_DIR / "data" / "faiss" / "第一课_parent_child"  # parent_child（父子块）索引目录

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"
META_FILE = "meta.json"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 32
EMBED_DIM = 1024

client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["API_URL"])


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


def _fingerprint(text: str) -> dict:
    return {
        "source_size": len(text),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _out_dir_for(strategy: str) -> Path:
    """切片策略对应的索引保存目录：heading 与 parent_child 分目录，互不覆盖。"""
    return PARENT_CHILD_OUT_DIR if strategy == "parent_child" else OUT_DIR


def load_index(md_path: Path = MD_PATH, out_dir: Path = None, strategy: str = "heading"):
    """索引存在且源文件、embedding 模型、切片策略都没变时直接加载，返回 (index, sections)，否则返回 None。"""
    out_dir = _out_dir_for(strategy) if out_dir is None else Path(out_dir)
    try:
        meta = json.loads((out_dir / META_FILE).read_text(encoding="utf-8"))
        index = faiss.read_index(str(out_dir / INDEX_FILE))
        with open(out_dir / CHUNKS_FILE, "rb") as f:
            sections = pickle.load(f)
    except (OSError, ValueError, KeyError, RuntimeError, pickle.UnpicklingError):
        return None

    if meta.get("embed_model") != EMBED_MODEL:
        print(f"embedding 模型已从 {meta.get('embed_model')} 变为 {EMBED_MODEL}，需要重建")
        return None

    if meta.get("strategy", "heading") != strategy:
        print(f"切片策略已从 {meta.get('strategy', 'heading')} 变为 {strategy}，需要重建")
        return None

    current = _fingerprint(Path(md_path).read_text(encoding="utf-8"))
    if meta.get("source_sha256") != current["source_sha256"]:
        print("源文件已变更，需要重建")
        return None

    return index, sections


def build_index(md_path: Path = MD_PATH, out_dir: Path = None, force: bool = False, strategy: str = "heading"):
    out_dir = _out_dir_for(strategy) if out_dir is None else Path(out_dir)
    if not force:
        cached = load_index(md_path, out_dir, strategy)
        if cached is not None:
            index, sections = cached
            print(f"命中已有索引，直接加载：{len(sections)} 段，共 {index.ntotal} 条向量")
            return cached

    md_path = Path(md_path)
    # read file
    text = md_path.read_text(encoding="utf-8")

    # father and child chunk strategy
    if strategy == "parent_child":
        parents, sections = split_parent_child(text)
        print(f"切片完成（父子块）：{len(parents)} 个父块，{len(sections)} 个子块")
    else:
        sections = split_markdown(text)
        print(f"切片完成：{len(sections)} 段")

    vectors = embed([s["text"] for s in sections])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

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

    print(f"索引已保存到 {out_dir}（dim={vectors.shape[1]}，共 {index.ntotal} 条）")
    return index, sections


def dense_search(
        query: str, index, sections: list[dict], top_k: int = 5, query_vec: np.ndarray = None
) -> list[dict]:
    """向量召回，走 make_hit 构造 hit（与 BM25 召回结构一致）。

    query_vec 可传入复用：多种召回方式并排对比时，query 只编码一次。
    """
    if query_vec is None:
        query_vec = embed([query], verbose=False)
    scores, ids = index.search(query_vec, top_k)
    hits: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:  # top_k > 索引条数时，faiss 用 -1 补齐
            continue
        hit = make_hit(sections[idx], int(idx), float(score))
        hit["dense_score"] = float(score)
        hits.append(hit)
    return hits

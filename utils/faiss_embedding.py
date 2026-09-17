"""把 markdown 切片，调用 OpenAI 兼容的 embedding 接口向量化，并写入 faiss 索引。

支持两种切片策略（--strategy）：
  heading       按标题层次切（默认）
  parent_child  父子块：父块按标题切、子块按句子聚合；子块用于检索，命中后返回父块完整上下文

索引按策略持久化到各自目录：heading -> data/faiss/第一课，parent_child -> data/faiss/第一课_parent_child。
两者互不覆盖，切换策略时直接加载对应目录的缓存，无需重新向量化。
强制重建：python utils/faiss_embedding.py --force

依赖：pip install faiss-cpu
用法：python utils/faiss_embedding.py [--strategy parent_child]
"""

import argparse
import hashlib
import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
MAX_CHARS = 1000  # 单个切片的最大字符数，超长章节按段落再切
CHILD_MAX_CHARS = 200  # 父子块策略中子块（段落）的最大字符数，越小检索越精准

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 句末标点或换行作为句子边界，标点/换行保留在句子里，聚合时原样拼接
SENTENCE_RE = re.compile(r"[^。！？；!?;\n]*(?:[。！？；!?;\n]|$)")

client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["API_URL"])


def _split_long(content: str, max_chars: int = MAX_CHARS) -> list[str]:
    """按句子切分后聚合到不超过 max_chars 的片段，避免超过 embedding 接口的长度上限。

    先按句末标点与换行切出句子（标点/换行保留在句子里），再把句子顺序聚合，
    使每个片段尽量接近 max_chars 且不截断句子；单个句子超长时按字符硬切兜底。
    """
    if len(content) <= max_chars:
        return [content]

    sentences = SENTENCE_RE.findall(content)
    pieces, buf, size = [], [], 0
    for sent in sentences:
        if not sent:
            continue
        # 单个句子超长：先把已聚合的 buf 落盘，再按字符硬切
        while len(sent) > max_chars:
            if buf:
                pieces.append("".join(buf))
                buf, size = [], 0
            pieces.append(sent[:max_chars])
            sent = sent[max_chars:]
        if not sent:
            continue
        if buf and size + len(sent) > max_chars:
            pieces.append("".join(buf))
            buf, size = [], 0
        buf.append(sent)
        size += len(sent)
    if buf:
        pieces.append("".join(buf))
    return pieces


def _emit(sections: list[dict], lines: list[str], headings: list[str]) -> None:
    content = "\n".join(lines).strip()
    if not content:
        return
    for piece in _split_long(content):
        # 把标题路径拼进正文，检索时更容易命中上下文
        text = f"{' > '.join(headings)}\n{piece}" if headings else piece
        sections.append(
            {
                "text": text,
                "title": headings[-1] if headings else "",
                "headings": list(headings),
            }
        )


def split_markdown(text: str) -> list[dict]:
    """按 # / ## / ### 等标题切分，保留每段的标题层级路径。"""
    sections: list[dict] = []
    lines: list[str] = []
    headings: list[str] = []

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            _emit(sections, lines, headings)
            lines = []
            level, title = len(match.group(1)), match.group(2).strip()
            headings = headings[: level - 1] + [title]
            continue
        lines.append(line)

    _emit(sections, lines, headings)
    return sections


def split_parent_child(text: str) -> tuple[list[dict], list[dict]]:
    """父子块策略：父块按标题切（保留完整章节上下文），子块再把父块正文按句子聚合。

    返回 (parents, children)。子块用于向量化和检索（更精准），命中后返回父块完整上下文（更完整）。
    每个子块带 parent_idx 和 parent 全文，检索命中时可直接取父块。
    """
    parents = split_markdown(text)
    children: list[dict] = []
    for pid, parent in enumerate(parents):
        prefix = " > ".join(parent["headings"])
        # parent["text"] 形如 f"{prefix}\n{body}"（无标题时就是 body），去掉前缀得到正文再切
        body = parent["text"][len(prefix) + 1 :] if prefix else parent["text"]
        for piece in _split_long(body, CHILD_MAX_CHARS):
            child_text = f"{prefix}\n{piece}" if prefix else piece
            children.append(
                {
                    "text": child_text,
                    "title": parent["title"],
                    "headings": list(parent["headings"]),
                    "parent_idx": pid,
                    "parent": parent["text"],
                }
            )
    return parents, children


def embed(texts: list[str], verbose: bool = True) -> np.ndarray:
    """批量取向量，并做 L2 归一化（配合 IndexFlatIP 等价于余弦相似度）。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
        if verbose:
            print(f"  已向量化 {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    arr = np.array(vectors, dtype="float32")
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
    text = md_path.read_text(encoding="utf-8")

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


def search(query: str, index, sections: list[dict], top_k: int = 5) -> list[dict]:
    query_vec = embed([query], verbose=False)
    scores, ids = index.search(query_vec, top_k)
    hits: list[dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        chunk = sections[idx]
        hit = {
            "score": float(score),
            "title": chunk.get("title", ""),
            "text": chunk["text"],
            "headings": chunk.get("headings", []),
        }
        if "parent" in chunk:  # 父子块策略：命中子块，返回父块完整上下文
            hit["text"] = chunk["parent"]
            hit["child_text"] = chunk["text"]
        hits.append(hit)
    return hits

def test(text:list[str]):
    print(text[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="markdown -> faiss 向量索引")
    parser.add_argument("-f", "--force", action="store_true", help="忽略已有索引，重新切片并向量化")
    parser.add_argument("-q", "--query", default="资本主义经济危机爆发的原因？", help="建好索引后用来验证检索的 query")
    parser.add_argument(
        "--strategy",
        choices=["heading", "parent_child"],
        default="parent_child",
        help="切片策略：heading=按标题切，parent_child=父子块（子块检索、父块返回）",
    )
    args = parser.parse_args()

    index, sections = build_index(force=args.force, strategy=args.strategy)

    if args.query:
        for hit in search(args.query, index, sections):
            if  hit['score'] > 0.5:
                print(f"\n[{hit['score']:.4f}] {hit['title']}\n{hit['text']}")
            if "child_text" in hit:
                print(f"\n[命中的子块]\n{hit['child_text']}")

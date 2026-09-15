"""把 markdown 按标题层次切片，调用 OpenAI 兼容的 embedding 接口向量化，并写入 faiss 索引。

索引会持久化在 OUT_DIR：下次运行时若源文件没变、模型没变，直接加载，不重复切片和调接口。
强制重建：python utils/faiss_embedding.py --force

依赖：pip install faiss-cpu
用法：python utils/faiss_embedding.py
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
OUT_DIR = BASE_DIR / "data" / "faiss" / "第一课"

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.pkl"
META_FILE = "meta.json"

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 32
MAX_CHARS = 1000  # 单个切片的最大字符数，超长章节按段落再切

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["API_URL"])


def _split_long(content: str) -> list[str]:
    """超长章节按空行（段落）再切，避免超过 embedding 接口的长度上限。"""
    if len(content) <= MAX_CHARS:
        return [content]

    pieces, buf, size = [], [], 0
    for para in content.split("\n\n"):
        if buf and size + len(para) > MAX_CHARS:
            pieces.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        pieces.append("\n\n".join(buf))
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


def load_index(md_path: Path = MD_PATH, out_dir: Path = OUT_DIR):
    """索引存在且源文件、embedding 模型都没变时直接加载，返回 (index, sections)，否则返回 None。"""
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

    current = _fingerprint(Path(md_path).read_text(encoding="utf-8"))
    if meta.get("source_sha256") != current["source_sha256"]:
        print("源文件已变更，需要重建")
        return None

    return index, sections


def build_index(md_path: Path = MD_PATH, out_dir: Path = OUT_DIR, force: bool = False):
    if not force:
        cached = load_index(md_path, out_dir)
        if cached is not None:
            index, sections = cached
            print(f"命中已有索引，直接加载：{len(sections)} 段，共 {index.ntotal} 条向量")
            return cached

    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")
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


def search(query: str, index, sections: list[dict], top_k: int = 3) -> list[dict]:
    query_vec = embed([query], verbose=False)
    scores, ids = index.search(query_vec, top_k)
    return [
        {"score": float(score), **sections[idx]}
        for score, idx in zip(scores[0], ids[0])
        if idx != -1
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="markdown -> faiss 向量索引")
    parser.add_argument("-f", "--force", action="store_true", help="忽略已有索引，重新切片并向量化")
    parser.add_argument("-q", "--query", default="资本主义经济危机为什么会爆发", help="建好索引后用来验证检索的 query")
    args = parser.parse_args()

    index, sections = build_index(force=args.force)

    if args.query:
        for hit in search(args.query, index, sections):
            print(f"\n[{hit['score']:.4f}] {hit['title']}\n{hit['text']}...")

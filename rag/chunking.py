"""markdown 切片策略：按标题层次切，以及父子块（子块检索、父块返回）。

只依赖 re，输入输出都是纯文本和字典列表，不碰 embedding、faiss 和文件 IO。

切片 dict 约定（下游 vector_store / retrieval 都按这个结构消费）：
    {"text", "title", "headings"}                          heading 策略
    {"text", "title", "headings", "parent_idx", "parent"}   parent_child 策略
"""

import re

MAX_CHARS = 1000  # 单个切片的最大字符数，超长章节按段落再切
CHILD_MAX_CHARS = 200  # 父子块策略中子块（段落）的最大字符数，越小检索越精准

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 句末标点或换行作为句子边界，标点/换行保留在句子里，聚合时原样拼接
SENTENCE_RE = re.compile(r"[^。！？；!?;\n]*(?:[。！？；!?;\n]|$)")


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

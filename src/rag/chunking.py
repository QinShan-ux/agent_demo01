"""markdown 切片策略：按标题层次切，以及父子块（子块检索、父块返回）。

只依赖 re，输入输出都是纯文本和字典列表，不碰 embedding、faiss 和文件 IO。

切片 dict 约定（下游 vector_store / retrieval 都按这个结构消费）：
    {"text", "title", "headings"}                          heading 策略
    {"text", "title", "headings", "parent_idx", "parent"}   parent_child 策略

栏目块（探究与分享 / 相关链接 / 专家点评）单独成切片，字段结构与上面一致，
只是 title 是栏目名、headings 末尾多一层栏目名、并带 "module": True 标记。
栏目块在两条策略下都不切分（材料与设问不能散），详见 MODULE_RE 与 _emit_module。
"""

import re
import logging

MAX_CHARS = 1000  # 单个切片的最大字符数，超长章节按段落再切（栏目块除外，见 _emit_module）
CHILD_MAX_CHARS = 200  # 父子块策略中子块（段落）的最大字符数，越小检索越精准

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 教材栏目在源文档里写成 "> **栏目名**"：标记行以 ">" 起头，栏目正文紧跟其后（不带 ">"）。
# 全文档只有这 100 处标记行以 ">" 开始，正文从不使用 "> "，所以这个信号不会误判正文。
MODULE_RE = re.compile(r"^>\s*\*\*(.+?)\*\*\s*$")
# 句末标点或换行作为句子边界，标点/换行保留在句子里，聚合时原样拼接。
# 注意：进这里的文本已经过了 _unwrap，\n 只剩段落边界，物理断行造成的断句已不存在。
SENTENCE_RE = re.compile(r"[^。！？；!?;\n]*(?:[。！？；!?;\n]|$)")
# 字段的上限，现在是 当字段大于 HARD_LIMIT 时必须切分 ， 正式项目根据选择的模型来设置 HARD_LIMIT 也可以收纳柜tiktoken来计算
HARD_LIMIT = 500


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

        # large sentence
        if len(sent) > max_chars:
            if buf:
                pieces.append("".join(buf))
                buf, size = [], 0
            if len(sent) < HARD_LIMIT:
                pieces.append("".join(sent))
            else:
                logging.warning("超长句 %d 字符，超过 HARD_LIMIT，强制切分", len(sent))
                for i in range(0, len(sent),HARD_LIMIT):
                    pieces.append(sent[i:i + HARD_LIMIT])

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


def _unwrap(text: str) -> str:
    """把 PDF 物理断行合并成逻辑段落。

    源 markdown 是 PDF 转的，按行宽断行，会把字从中间切开（如「我们从哪↵儿来」），
    物理换行不是段落边界。真正的段落边界只有三种：空行、标题行（#）、栏目标记行
    （> **栏目**）。连续的物理行直接拼接（PDF 断行处没有空格可保），结构行与空行
    原样保留，后续切片逻辑照旧。
    """
    rows = text.splitlines()
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append("".join(buf))
            buf.clear()

    for row in rows:
        s = row.strip()
        if not s or HEADING_RE.match(s) or MODULE_RE.match(s):
            flush()
            out.append(row)
        else:
            buf.append(s)
    flush()
    return "\n".join(out)


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


def _emit_module(
    sections: list[dict], lines: list[str], headings: list[str], name: str
) -> None:
    """把栏目块（探究与分享 / 相关链接 / 专家点评）单独切成一条切片。

    栏目内容与正文不是一回事：它常是引文、案例或思考题，混在正文里既会把正文段落
    从中间截断（如「…建立在科学认识社会发展客观规律的基础上。」被引文隔开），
    也会让栏目自己的材料与设问落到不同切片里，检索回来答非所问。所以整块独立成条。

    不作 _split_long：一个栏目就是一条切片，切开了材料与设问就散了。实测 100 个栏目
    中位 238 字、最长 1239 字，直接用栏目名接在标题路径末尾（与 _emit 的约定一致，
    title = headings[-1]），检索时能看出它属于哪一节的哪个栏目。
    "module": True 是给 split_parent_child 的标记：父子块策略下栏目同样整块不切。
    """
    content = "\n".join(lines).strip()
    if not content:
        return
    path = headings + [name]
    sections.append(
        {
            "text": f"{' > '.join(path)}\n{content}",
            "title": name,
            "headings": path,
            "module": True,
        }
    )


def split_markdown(text: str) -> list[dict]:
    """按 # / ## / ### 等标题切分，保留每段的标题层级路径。

    先 _unwrap 把 PDF 物理断行合并成逻辑段落（段落边界 = 空行/标题/栏目），再按标题切。
    以 "> **栏目名**" 起头的栏目块整体切成独立切片，不与其前后的正文合并；正文照旧按标题切。
    """
    sections: list[dict] = []
    lines: list[str] = []
    headings: list[str] = []
    rows = _unwrap(text).splitlines()

    i = 0
    while i < len(rows):
        stripped = rows[i].strip()
        module = MODULE_RE.match(stripped)
        heading = HEADING_RE.match(stripped)

        if module:
            _emit(sections, lines, headings)  # 栏目前的正文先收口，别和栏目内容并到一条
            lines = []
            i += 1
            body: list[str] = []
            # 栏目正文到空行为止；再遇标记行或标题也停，避免没有空行的写法把后面的内容吞进来
            while (
                i < len(rows)
                and rows[i].strip()
                and not MODULE_RE.match(rows[i].strip())
                and not HEADING_RE.match(rows[i].strip())
            ):
                body.append(rows[i])
                i += 1
            _emit_module(sections, body, headings, module.group(1).strip())
            continue

        if heading:
            _emit(sections, lines, headings)
            lines = []
            level, title = len(heading.group(1)), heading.group(2).strip()
            headings = headings[: level - 1] + [title]
        else:
            lines.append(rows[i])
        i += 1

    _emit(sections, lines, headings)
    return sections


def split_parent_child(text: str) -> tuple[list[dict], list[dict]]:
    """父子块策略：父块按标题切（保留完整章节上下文），子块再把父块正文按句子聚合。

    返回 (parents, children)。子块用于向量化和检索（更精准），命中后返回父块完整上下文（更完整）。
    每个子块带 parent_idx 和 parent 全文，检索命中时可直接取父块。
    """
    # split father chunk
    parents = split_markdown(text)
    children: list[dict] = []
    for pid, parent in enumerate(parents):
        prefix = " > ".join(parent["headings"])
        # parent["text"] 形如 f"{prefix}\n{body}"（无标题时就是 body），去掉前缀得到正文再切
        body = parent["text"][len(prefix) + 1:] if prefix else parent["text"]
        # 栏目块不切：材料与设问切散了语义就断了，整块作为一个子块。
        # 实测最长栏目 text 1299 字符（约 2.2k token），远低于 embedding 上限，无需兜底。
        pieces = [body] if parent.get("module") else _split_long(body, CHILD_MAX_CHARS)
        for piece in pieces:
            child_text = f"{prefix}\n{piece}" if prefix else piece
            child = {
                "text": child_text,
                "title": parent["title"],
                "headings": list(parent["headings"]),
                "parent_idx": pid,
                "parent": parent["text"],
            }
            if parent.get("module"):
                child["module"] = True
            children.append(child)
    return parents, children

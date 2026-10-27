
from agent import create_model
from prompting.registry import render
from rag.pipeline import _parse_rewrite


def test_rewrite_promote():
    llm = create_model()
    res = llm.invoke(render("intent_split.j2", query="改革开发的进程，按照对外对内分层，每一阶段放到一起"))
    content = res.content if isinstance(res.content, str) else str(res.content)
    question, requirements = _parse_rewrite(content)
    print(question)


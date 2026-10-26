
import pymupdf4llm

PATH = '../../data/pdf/必修1.pdf'

def conv_pymupdf4llm():
    md_text = pymupdf4llm.to_markdown(PATH)
    with open("必修一pymupdf4llm.md", "w", encoding="utf-8") as f:
        f.write(md_text)


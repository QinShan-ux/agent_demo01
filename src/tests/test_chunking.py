from rag.chunking import *
import json

def test_split_parent_child():
    text = '''
        # 第一章
第一章内容
## 1.1
1.1内容
# 第二章
第二章内容
## 2.1
2.1内容
    '''
    section = split_markdown(text)
    _f = {
        "sections":section
    }
    res = json.dumps(_f,ensure_ascii=False)

    print(res)
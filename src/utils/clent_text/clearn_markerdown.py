from torch._higher_order_ops import wrap

PATH = '../../data/pdf/必修markidown.md'

def remove_pagesize():
    with open(PATH, "r+", encoding="utf-8") as f:
        content = f.read()
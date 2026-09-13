import re

def is_special(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith('#'):
        return True
    if re.match(r'^[-*+]\s', s):
        return True
    if re.match(r'^\d+[.、]\s', s):
        return True
    if s.startswith('【'):
        return True
    return False

def clean_stream(input_path: str, output_path: str):
    buffer = []
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        def flush():
            if buffer:
                paragraph = ''.join(buffer).strip()
                if paragraph:
                    fout.write(paragraph + '\n\n')
                buffer.clear()

        for line in fin:
            # 1. 去图片
            line = re.sub(r'!\[\]\([^)]*\)', '', line)
            # 2. 去页码
            if re.fullmatch(r'\s*\d+\s*', line):
                continue

            line = re.sub('◆+','',line)
            stripped = line.strip()

            if not stripped:
                flush()
                continue

            if is_special(line):
                flush()
                fout.write(stripped + '\n\n')
                continue

            buffer.append(stripped)
            # 句末标点 → 段落结束
            if re.search(r'[。！？；：”」）]$', stripped):
                flush()

        flush()

clean_stream('./../../data/markdown/2026年秋季版必修/2026年秋季版必修1.md', '第一课.clean.md')
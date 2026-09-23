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
                    fout.write(paragraph + '\n')
                buffer.clear()

        for line in fin:
            # 1. 去图片
            line = re.sub(r'^!\[.*?\]\(.*?\)[ \t]*\n?$', '', line)
            # 2. 去页码
            if re.fullmatch(r'\s*\d+\s*', line):
                continue

            # 人民出版社去除
            if re.fullmatch(r'(#+) 人民教育出版社\s*', line):
                continue

            if line.strip() == '':
                continue
            if line.strip() == '## 探究与分享':
                line = line.replace('## 探究与分享','#### 相关链接')

            if line.strip() == '## 相关链接':
                line = line.replace('## 相关链接', '#### 相关链接')

            if line.strip() == '## 专家点评':
                line = line.replace('## 专家点评', '#### 专家点评')

            line = re.sub('◆+','',line)
            stripped = line.strip()

            if not stripped:
                flush()
                continue

            if is_special(line):
                flush()
                fout.write(stripped + '\n')
                continue

            buffer.append(stripped)
            # 句末标点 → 段落结束
            if re.search(r'[。！？；：”」）]$', stripped):
                flush()

        flush()

clean_stream('./../../data/pdf/2026年秋季版必修1_2.md', '第一课.md')
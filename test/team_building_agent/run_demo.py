"""入口：python run_demo.py

零第三方依赖，Python 3.12 直接跑。结果同时打印到终端并写到 demo_output.txt。
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台中文
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from tb_agent.scenarios import run_all  # noqa: E402


def main() -> None:
    text = run_all()
    print(text)
    with open(os.path.join(HERE, "demo_output.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n[已写出] {os.path.join(HERE, 'demo_output.txt')}")


if __name__ == "__main__":
    main()

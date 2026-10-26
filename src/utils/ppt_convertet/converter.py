import os
import shutil
import subprocess
import tempfile

import pymupdf

# 以当前文件位置为基准，避免受工作目录影响
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PPT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "data", "ppt"))
OUTPUT_ROOT = os.path.join(PPT_DIR, "ppt_images")

ZOOM = 2.0  # 缩放系数，2.0 约等于 144 DPI，可根据需要调整

# LibreOffice 可执行文件候选路径（新版安装到 Program Files，旧版可能在 x86）
SOFFICE_CANDIDATES = [
    os.environ.get("SOFFICE_PATH", ""),
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def find_soffice() -> str:
    for path in SOFFICE_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "未找到 soffice.exe，请安装 LibreOffice 或设置环境变量 SOFFICE_PATH 指向 soffice.exe"
    )


def clean_env() -> dict:
    """系统 PATH 中可能存在与 LibreOffice 冲突的同名 DLL，
    导致 soffice 启动报 0xC0000135，因此用最小 PATH 启动"""
    env = os.environ.copy()
    env["PATH"] = r"C:\Windows\System32;C:\Windows"
    return env


def ppt_to_pdf(soffice: str, ppt_path: str, pdf_dir: str) -> str:
    """调用 LibreOffice 无头模式将 ppt 转为 pdf，返回 pdf 路径"""
    subprocess.run(
        [soffice, "--headless", "--norestore", "--convert-to", "pdf", "--outdir", pdf_dir, ppt_path],
        check=True,
        env=clean_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pdf_path = os.path.join(pdf_dir, os.path.splitext(os.path.basename(ppt_path))[0] + ".pdf")
    if not os.path.isfile(pdf_path):
        raise RuntimeError(f"LibreOffice 转换失败，未生成 pdf：{ppt_path}")
    return pdf_path


def convert_ppt(soffice: str, ppt_path: str, output_dir: str, tmp_dir: str) -> None:
    """ppt -> pdf -> 逐页渲染为图片，保存到 output_dir"""
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = ppt_to_pdf(soffice, ppt_path, tmp_dir)

    doc = pymupdf.open(pdf_path)
    page_count = doc.page_count
    mat = pymupdf.Matrix(ZOOM, ZOOM)

    for page_num in range(page_count):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        pix.save(os.path.join(output_dir, f"page_{page_num + 1:03d}.png"))

    doc.close()
    print(f"[完成] {os.path.basename(ppt_path)} -> {output_dir}（{page_count} 页）")


def main() -> None:
    soffice = find_soffice()
    with tempfile.TemporaryDirectory() as tmp_dir:
        for name in sorted(os.listdir(PPT_DIR)):
            path = os.path.join(PPT_DIR, name)
            if os.path.isfile(path) and name.lower().endswith((".ppt", ".pptx")):
                # 每个 ppt 一个独立文件夹，文件夹名为去掉扩展名的文件名
                output_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(name)[0])
                convert_ppt(soffice, path, output_dir, tmp_dir)


if __name__ == "__main__":
    main()

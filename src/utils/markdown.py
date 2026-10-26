from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered, save_output
import os

if __name__ == '__main__':
    output_dir = "C:\\X\\project\\python\\agent\\agent_demo01\\data\\markdown"
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "output_format": "markdown",
        "output_dir": output_dir,
        "num_workers": 2,
        "disable_image_extraction": False,
    }
    config_parser = ConfigParser(config)

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )

    rendered = converter("C:\\X\\project\\python\\agent\\agent_demo01\\data\\pdf\\2026年秋季版必修1.pdf")

    # 提取结果
    text, ext, images = text_from_rendered(rendered)

    # ✅ 关键步骤：保存到磁盘
    # fname_base 是输出文件的基础名（不带扩展名）
    save_output(rendered, output_dir=output_dir, fname_base="2026年秋季版必修1")

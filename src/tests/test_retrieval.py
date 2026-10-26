import os

import requests

from dotenv import load_dotenv
load_dotenv()
def test_rerank():
    try:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 未设置")

        resp = requests.post(
            "https://maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen3.7-text-rerank",
                "input": {
                    "query": '什么是文本排序模型',
                    "documents": [
                        "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
                        "量子计算是计算科学的一个前沿领域",
                        "预训练语言模型的发展给文本排序模型带来了新的进展",
                    ],
                },
                "parameters": {
                    "top_n": 5,
                    "return_documents": True,
                },
            },
            timeout=30,   # 注意：原来是 3000 秒，太夸张了
        )
        resp.raise_for_status()
        payload = resp.json()
        # print(payload)
        if "error" in payload:
            raise ValueError(payload["error"])
        print(payload["output"]["results"])
    except (requests.RequestException, KeyError, ValueError) as exc:
        import traceback
        traceback.print_exc()   # 调试用，定位后删掉
        print(f"  [warn] rerank 调用失败，回退到融合排序：{exc}")
        return ""
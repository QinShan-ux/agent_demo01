import os

import requests

from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.environ.get("GLMKEY")
url = os.environ.get("GLMURL")

def test_ocr():
    url = "https://open.bigmodel.cn/api/paas/v4/files/ocr"

    path = os.path.join(os.path.dirname(__file__), "..", "data", "png", "test.png")
    files = {
        "file": (os.path.basename(path), open(path, "rb"))
    }
    payload = {
        "tool_type": "hand_write",
        "probability": "true"
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(url, data=payload, files=files, headers=headers)
    data = json.loads(response.text)
    word = [item['words'] for item in data['words_result']]
    print('\n')
    print(word)


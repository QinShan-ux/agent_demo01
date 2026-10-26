from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://api.agicto.cn/v1")
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好",
        }
    ], model="gpt-image-2.5-flare", )
print(chat_completion)

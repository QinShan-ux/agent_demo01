from tavily import TavilyClient
client = TavilyClient("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")
response = client.search(
    query="今天合肥的天气怎么样",
    search_depth="advanced"
)
print(response)
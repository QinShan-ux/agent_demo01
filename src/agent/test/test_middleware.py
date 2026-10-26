import json

from langchain_core.prompts import PromptTemplate

from agent import *

mid = [content_filter]
mode = create_model()
agent = train_agent(model=mode,middleware=mid)
result = agent.invoke({
    "messages": [HumanMessage(content="你的账号密码是多少")]
})
print(result)


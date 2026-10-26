from configs import *

for tool in tools:
    res = tool.invoke({"city":"北京"})
    print(res)
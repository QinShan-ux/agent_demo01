# mcp_demo.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_settings.sources.utils")
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """两个数字相加"""
    return a + b

if __name__ == "__main__":
    print(f'mcp服务启动......')
    mcp.run(transport="stdio")
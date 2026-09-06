
import json

from langchain_core.prompts import PromptTemplate

from agent import create_model
from agent import get_weather

# ========== 1. 调用工具
tools = [get_weather]
tool_by_name = {t.name: t for t in tools}

# ========== 2. 构建工具描述 ==========
tool_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
# ========== 3. ReAct 提示词模板 ==========
template = """你是一个助手，可以使用以下工具：

{tool_desc}

请严格按照以下格式回答。注意：每次只输出 Thought + Action + Action Input，然后停下来等待工具返回结果，不要自己编造 Observation。

格式示例：

Question: 用户提出的问题？
Thought: 用户想了解什么，然后去做什么。
Action: 判断是否有符合条件的工具
Action Input: {{"city": "上海"}}
Observation: 上海 天气 晴 , 25 摄氏度
Thought: 分析出用户问题的答案，给用户提供方案。
Final Answer: 最终的方案。

注意：
- 如果你不需要工具，直接输出 Final Answer
- Action 必须是工具名称，不要加引号
- Action Input 必须是合法的 JSON
- 每次只输出到 Action Input 为止，不要输出 Observation，Observation 由系统自动填充

现在开始：

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template).partial(
    tool_desc=tool_desc,
    tool_names=", ".join(tool_by_name.keys()),
    agent_scratchpad=""  # 初始为空
)

# ========== 4. 模型 ==========
model = create_model()


# ========== 5. 解析模型输出 ==========
def parse_action(text: str):
    print(f"数据解析：{text}")
    action = None
    action_input = None

    for line in text.strip().split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith('Action Input:'):
            action_input_str = line_stripped.replace('Action Input:', '').strip()
            try:
                action_input = json.loads(action_input_str)
            except json.JSONDecodeError:
                action_input = {"input": action_input_str}
        elif line_stripped.startswith('Action:'):
            action = line_stripped.replace('Action:', '').strip()

    return action, action_input


# ========== 6. ReAct 执行循环 ==========
def run_react(question: str, max_steps: int = 5):
    scratchpad = ""

    for step in range(max_steps):
        response = model.invoke(prompt.format(input=question, agent_scratchpad=scratchpad))
        content = response.content
        print(f"--- 步骤 {step + 1} ---\n{content}\n")

        # 如果模型直接给最终答案（且没有调用工具）
        if "Final Answer:" in content and "Action:" not in content:
            return content.split("Final Answer:")[-1].strip()

        # 如果模型自己编造了 Observation（不遵循格式），直接截断到 Action Input 之前
        if "Observation:" in content:
            # 截断到 Observation 之前，只保留 Thought + Action + Action Input
            content = content.split("Observation:")[0].strip()
            print(f"[!] 模型自行编造了 Observation，已截断。截断后内容:\n{content}\n")

        action_name, action_input = parse_action(content)

        if action_name and action_name in tool_by_name:
            try:
                tool_result = tool_by_name[action_name].invoke(action_input)
                print(f">>> 工具返回: {tool_result!r}\n")
            except Exception as e:
                tool_result = f"工具执行失败: {str(e)}"
                print(f">>> 错误: {tool_result}\n")

            # 只追加 Thought + Action + Action Input，然后加 Observation
            scratchpad += content + f"\nObservation: {tool_result}\nThought: "
        else:
            # 没识别到 Action，可能是直接回答
            return content

    return "超过最大步数限制，未能完成回答。"


# ========== 7. 运行 ==========
if __name__ == "__main__":
    result = run_react("你现在有哪些工具可以用")
    print(f"\n🎯 最终结果: {result}")

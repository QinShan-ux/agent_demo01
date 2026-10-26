from agent import create_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

llm = create_model()
llm.invoke([SystemMessage()])
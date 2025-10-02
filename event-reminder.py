import grp
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from datetime import datetime
from typing_extensions import TypedDict
from langchain_gigachat import GigaChat
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.globals import set_debug
import os

load_dotenv()
# set_debug(True)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define a tool for current date
@tool
def get_current_date() -> str:
    """Returns today's current date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")

# Setup LLM
llm = GigaChat(
    credentials=os.getenv("GIGACHAT_CREDENTIALS"),
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    model="GigaChat-2",
    temperature=0,
)

# Setup tools
tools = [get_current_date]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    """Chatbot node that processes messages and decides whether to use tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

# Add conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Add edges
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()

print(graph.get_graph().draw_ascii())

# Run the agent
user_input = "What is the exact current date?"
response = graph.invoke({"messages": [{"role": "user", "content": user_input}]})

print("Agent response:", response["messages"][-1].content)
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from datetime import datetime
from typing_extensions import TypedDict
from langchain_gigachat import GigaChat
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.globals import set_debug
from lxml import etree
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
# llm = GigaChat(
#     credentials=os.getenv("GIGACHAT_CREDENTIALS"),
#     verify_ssl_certs=False,
#     scope="GIGACHAT_API_PERS",
#     model="GigaChat-Pro",
#     temperature=0,
# )

llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)

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

# Print the graph in ASCII (optional)
# print(graph.get_graph().draw_ascii())

# Prepare data
xml_file = "test-events.xml"
tree = etree.parse(xml_file)
xml_text = etree.tostring(tree, pretty_print=True, encoding="unicode")

# Prepare prompts and initial state
system_prompt = """
Ты эксперт в анализе XML документов.
"""

days = 7

user_prompt = f"""
Проанализируй XML документ:\n\n{xml_text}\n\n.

В нем описаны события. 
Каждое событие представлено в виде элемента <event>.
Элемент <event> содержит следующие атрибуты:
- name - название события
- date - дата события
- person - имя или название группы

При анализе даты события учитывай только месяц и день, не учитывай год.

Выведи список событий на предстоящие {days} дней.
Не выводи события, выходящие за пределы предстоящих {days} дней.

Событие должно быть строго в формате: <дата> # <имя> # <название>

Примеры:
5 октября # Anna White # Birthday
12 октября # City Residents # Festival

Дата должна быть в формате "день месяца", например "12 октября" или "22 ноября".

Отсортируй события по дате (сначала ближайшие).

Не выводи других комментариев, кроме списка событий.

Если событий нет, то не выводи ничего (верни пустую строку).
"""

initial_state = {
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
}

# Run the agent
response = graph.invoke(initial_state)

print(response["messages"][-1].content)
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from datetime import datetime
from typing_extensions import TypedDict
from models import grok, gpt_oss, groq, deepseek, gigachat_pro, gigachat
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.globals import set_debug
from lxml import etree
import requests
import os
import argparse

load_dotenv()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Event Reminder Bot')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output for debugging')
    return parser.parse_args()

args = parse_args()

# Enable debug mode if verbose flag is set
if (args.verbose):
    set_debug(True)

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define a tool for current date
@tool
def get_current_date() -> str:
    """Возвращает сегодняшнюю дату в формате YYYY-MM-DD."""
    return datetime.now().strftime("%Y-%m-%d")

# Define a tool for sending Telegram messages
@tool
def send_telegram_message(message: str) -> str:
    """Отправляет сообщение в Telegram чат.
    
    Параметры:
        message: Текст сообщения для отправки
        
    Возвращает:
        Сообщение об успешной отправке или ошибке
    """
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token:
        return "Error: TELEGRAM_BOT_TOKEN environment variable not set"
    if not chat_id:
        return "Error: TELEGRAM_CHAT_ID environment variable not set"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message
        }
        response = requests.post(url, data=data)
        response.raise_for_status()
        return f"Message sent successfully to chat {chat_id}"        
    except Exception as e:
        return f"Error sending message: {str(e)}"

# Setup LLM
llm = grok

# Setup tools
tools = [get_current_date, send_telegram_message]
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

# Print the graph in ASCII if verbose flag is set
if (args.verbose):
    print(graph.get_graph().draw_ascii())

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
Каждое событие содержит следующие параметры:
- дата
- имя
- название

При анализе даты события учитывай только месяц и день, не учитывай год.

Сначала получи текущую дату с помощью доступного инструмента, чтобы определить дни попадают в предстоящие {days} дней.

Сформируй список событий на предстоящие {days} дней.
Не добавляй события, выходящие за пределы предстоящих {days} дней.

Событие должно быть строго в формате: <дата> # <имя> # <название>

Примеры:
5 октября # Anna White # Birthday
12 октября # City Residents # Festival

Дата должна быть в формате "день месяца", например "12 октября" или "22 ноября".

Отсортируй события по дате (сначала ближайшие).

Не добавляй других комментариев, кроме списка событий.

Результатом должен быть список событий в виде строки.
Если событий нет, результатом должна быть пустая строка.

Если результат не пустой, то отправь его в чат в Telegram с помощью доступного инструмента.

В итоге верни результат в виде строки.
"""

initial_state = {
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
}

# Run the agent
response = graph.invoke(initial_state, config={"recursion_limit": 10})

print(response["messages"][-1].content)
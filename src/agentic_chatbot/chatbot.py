import os

import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver

from agentic_chatbot.tools import search_tool, get_current_weather, calculator, get_stock_price


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpoint = SqliteSaver(connection)


class ChatState(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]


def extract_message_text(message: BaseMessage | object) -> str:
  if message is None:
    return ""

  content = getattr(message, "content", "")
  if isinstance(content, str):
    return content

  if isinstance(content, list):
    parts: list[str] = []
    for item in content:
      if isinstance(item, str):
        parts.append(item)
      elif isinstance(item, dict):
        text = item.get("text") or item.get("content")
        if text:
          parts.append(str(text))
      elif item is not None:
        parts.append(str(item))
    return "".join(parts)

  if isinstance(content, dict):
    text = content.get("text") or content.get("content")
    return str(text) if text is not None else ""

  if content is None:
    return ""

  return str(content)


def collect_stream_text(stream):
  chunks: list[str] = []
  for message, _ in stream:
    text = extract_message_text(message)
    if text:
      chunks.append(text)
  return "".join(chunks)


llm = ChatOpenAI(
  model='openrouter/free',
  base_url='https://openrouter.ai/api/v1',
  api_key=OPENROUTER_API_KEY,
)

tools = [tool for tool in [search_tool, get_current_weather, calculator, get_stock_price] if tool is not None]
llm_with_tools = llm.bind_tools(tools)


def chat(state: ChatState):
  messages = state['messages']
  response = llm_with_tools.invoke(messages)
  return {'messages': [response]}

graph = StateGraph(ChatState)

graph.add_node('chat', chat)
graph.add_node('tools', ToolNode(tools))

graph.add_edge(START, 'chat')
graph.add_conditional_edges('chat', tools_condition)
graph.add_edge('tools', 'chat')

chatbot = graph.compile(checkpointer=checkpoint)

def get_thread_IDs():
  try:
    cursor = connection.execute("select distinct thread_id from checkpoints")
    ids = cursor.fetchall()
    ids = [i[0] for i in ids]
    return ids
  except:
    return [] 
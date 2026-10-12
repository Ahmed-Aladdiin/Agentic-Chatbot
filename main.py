from agentic_chatbot.chatbot import chatbot, get_thread_IDs
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

import uuid

def generate_id():
  return str(uuid.uuid4())

st.title("Agentic Chatbot")

################################ Init session state

if 'chat_messages' not in st.session_state:
  st.session_state['chat_messages'] = []

if 'current_thread' not in st.session_state:
  st.session_state['current_thread'] = generate_id()

if 'chat_threads' not in st.session_state:
  st.session_state['chat_threads'] = get_thread_IDs() + [st.session_state['current_thread']]

################################ Chat area

for message in st.session_state['chat_messages']:
  with st.chat_message(message['role']):
    st.text(message['content'])


input = st.chat_input("Type here")

if input:
  st.session_state['chat_messages'].append({'role': 'user', 'content': input})
  with st.chat_message('user'):
    st.text(input)

  with st.chat_message('ai'):
    response = st.write_stream(
      message.content for message, _ in chatbot.stream(
        {'messages': [HumanMessage(content=input)]},
        config={'configurable': {'thread_id': st.session_state['current_thread']}},
        stream_mode='messages'
      )
    )
  
  st.session_state['chat_messages'].append({'role': 'ai', 'content': response})

################################ Resitting methods

def add_new_thread(thread_id):
  if thread_id not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(thread_id)

def reset_chat():
  new_thread = generate_id()
  st.session_state['current_thread'] = new_thread
  st.session_state['chat_messages'] = []
  add_new_thread(new_thread)

def get_thread_messages(thread_id):
  state = chatbot.get_state(
      config={
          "configurable": {
              "thread_id": thread_id
          }
      }
  )
  return state.values.get("messages", [])

def convert_messages(messages):
  converted = []
  for msg in messages:
    if isinstance(msg, HumanMessage):
      converted.append({'role': 'user', 'content': msg.content})
    elif isinstance(msg, AIMessage):
      converted.append({'role': 'ai', 'content': msg.content})
  return converted

def load_thread(thread_id):
  st.session_state['current_thread'] = thread_id
  msgs = get_thread_messages(thread_id)
  st.session_state['chat_messages'] = convert_messages(msgs)
  st.rerun()

################################ Sidebar

st.sidebar.title("Conversations")

if st.sidebar.button("New Chat"):
  reset_chat()
  st.rerun()

for thread_id in st.session_state['chat_threads'][::-1]:
  if st.sidebar.button(str(thread_id), key=thread_id):
    load_thread(thread_id)



from agentic_chatbot.chatbot import chatbot, get_thread_IDs, extract_message_text
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid
import json


def generate_id():
    return str(uuid.uuid4())


def normalize_message_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    pieces.append(str(text))
            elif item is not None:
                pieces.append(str(item))
        return "".join(pieces)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return str(text) if text is not None else ""
    return str(content)


def display_value(value):
    """Format tool arguments/results for readable display."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def collect_stream_response(stream, config):
    response_parts = []
    tool_statuses = {}
    tool_events = []

    for update in stream:
        # Each update contains the node that just ran and its state changes.
        for node_name, node_update in update.items():
            messages = node_update.get("messages", [])

            for message in messages:
                # The LLM requested one or more tools.
                if isinstance(message, AIMessage) and message.tool_calls:
                    for call in message.tool_calls:
                        call_id = call.get("id") or str(uuid.uuid4())
                        tool_name = call.get("name", "Unknown tool")
                        tool_args = call.get("args", {})

                        status = st.status(
                            f"🔧 Using tool: `{tool_name}`",
                            expanded=True,
                        )
                        status.write("**Input**")
                        status.code(display_value(tool_args))

                        tool_statuses[call_id] = {
                            "name": tool_name,
                            "args": tool_args,
                            "status": status,
                        }
                        tool_events.append(call_id)

                # The tool has returned its result.
                elif isinstance(message, ToolMessage):
                    call_id = message.tool_call_id
                    result = normalize_message_content(message.content)

                    event = tool_statuses.get(call_id)

                    if event is None:
                        # Fallback for messages without a matching call ID.
                        event = {
                            "name": getattr(message, "name", "Unknown tool"),
                            "status": st.status(
                                "🔧 Tool result",
                                expanded=True,
                            ),
                        }
                        tool_statuses[call_id or str(uuid.uuid4())] = event

                    event["result"] = result
                    event["status"].write("**Result**")
                    event["status"].code(display_value(result))
                    event["status"].update(
                        label=f"✅ Tool finished: `{event['name']}`",
                        state="complete",
                        expanded=False,
                    )

                # Collect the assistant's final response, not tool-call messages.
                elif isinstance(message, AIMessage):
                    text = extract_message_text(message)
                    if text:
                        response_parts.append(text)

    return "".join(response_parts)


st.title("Agentic Chatbot")

# -------------------- Initialize session state --------------------

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

if "current_thread" not in st.session_state:
    st.session_state["current_thread"] = generate_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list(
        dict.fromkeys(
            get_thread_IDs() + [st.session_state["current_thread"]]
        )
    )

# -------------------- Chat area --------------------

for message in st.session_state["chat_messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["chat_messages"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        config = {
            "configurable": {
                "thread_id": st.session_state["current_thread"]
            }
        }

        response = collect_stream_response(
            chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="updates",
            ),
            config,
        )

        st.markdown(response)

    st.session_state["chat_messages"].append(
        {"role": "ai", "content": response}
    )

# -------------------- Conversation management --------------------

def add_new_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    new_thread = generate_id()
    st.session_state["current_thread"] = new_thread
    st.session_state["chat_messages"] = []
    add_new_thread(new_thread)


def get_thread_messages(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    if state is None or state.values is None:
        return []

    return state.values.get("messages", [])


def convert_messages(messages):
    converted = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            converted.append({
                "role": "user",
                "content": normalize_message_content(msg.content),
            })
        elif isinstance(msg, AIMessage):
            # Do not show intermediate assistant messages that only call tools.
            if msg.tool_calls:
                continue

            content = normalize_message_content(msg.content)
            if content:
                converted.append({"role": "ai", "content": content})

    return converted


def load_thread(thread_id):
    st.session_state["current_thread"] = thread_id
    st.session_state["chat_messages"] = convert_messages(
        get_thread_messages(thread_id)
    )
    st.rerun()


# -------------------- Sidebar --------------------

st.sidebar.title("Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=thread_id):
        load_thread(thread_id)

import os
import json
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

HISTORY_FILE = "chat_history.json"

# ✅ Function to save chat history to file
def save_history(memory):
    messages = [
        {"role": "user", "content": msg.content}
        if isinstance(msg, HumanMessage)
        else {"role": "ai", "content": msg.content}
        for msg in memory.chat_memory.messages
    ]
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f)

# ✅ Function to load chat history from file
def load_history(memory):
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return  # Empty file, skip loading
                messages = json.loads(content)
                for msg in messages:
                    if msg["role"] == "user":
                        memory.chat_memory.add_user_message(msg["content"])
                    else:
                        memory.chat_memory.add_ai_message(msg["content"])
        except (json.JSONDecodeError, ValueError):
            # If file is corrupted, ignore loading
            pass


# ✅ Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# ✅ Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    # ✅ Load previous chat history
    load_history(st.session_state.memory)

memory = st.session_state.memory

st.title("🤖 My Chatbot")

# Container to display chat messages
chat_container = st.container()

# ✅ Function to process user input
def process_input():
    user_message = st.session_state.user_input
    if user_message:
        # Add user message to memory
        memory.chat_memory.add_user_message(user_message)

        # Get AI response
        response_text = llm.invoke(memory.chat_memory.messages)

        # Add AI response to memory
        memory.chat_memory.add_ai_message(response_text)

        # ✅ Save chat history after each exchange
        save_history(memory)

        # Clear input
        st.session_state.user_input = ""

# Text input with callback
st.text_input("You:", key="user_input", on_change=process_input)

# ✅ Display chat history
with chat_container:
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# Initialize memory in session state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

memory = st.session_state.memory

st.title("🤖 My Chatbot")

# Container to display chat messages
chat_container = st.container()

# Function to process user input
def process_input():
    user_message = st.session_state.user_input
    if user_message:
        # Add user message to memory
        memory.chat_memory.add_user_message(user_message)

        # Get AI response
        response_text = llm.invoke(memory.chat_memory.messages)

        # Add AI response to memory
        memory.chat_memory.add_ai_message(response_text)

        # Clear input safely via session_state
        st.session_state.user_input = ""

# Reactive input with callback
st.text_input("You:", key="user_input", on_change=process_input)

# Display chat history
with chat_container:
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**Bot:** {msg.content}")

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.schema import HumanMessage, AIMessage

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),  # Set this in .env
    model_name="mixtral-instruct-7b"         # Free model
)

# In-memory session storage
session_history = []

# Function to provide session history
def get_session_history():
    return session_history

# Base runnable to process messages
def process_message(message: HumanMessage) -> str:
    return llm.invoke(message)

# Runnable with memory
conversation = RunnableWithMessageHistory(
    runnable=RunnableLambda(process_message),
    get_session_history=get_session_history
)

def chat():
    print("🤖 Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        human_message = HumanMessage(content=user_input)
        response_text = conversation.invoke(human_message)
        ai_message = AIMessage(content=response_text)

        # Append messages to memory
        session_history.append(human_message)
        session_history.append(ai_message)

        print("Bot:", response_text)

if __name__ == "__main__":
    chat()

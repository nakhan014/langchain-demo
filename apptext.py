import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def simple_chat():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    question = "What is "
    response = llm.invoke(question)
    print("Response Noor:", response.content)

if __name__ == "__main__":
    simple_chat()

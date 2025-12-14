import os
import sys
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import settings

def test_chat():
    print(f"Testing Chat Model: {settings.CHAT_MODEL} at {settings.OPENAI_BASE_URL}")
    try:
        chat = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.CHAT_MODEL,
            temperature=0
        )
        response = chat.invoke([HumanMessage(content="Hello, are you working?")])
        print("Chat Success!")
        print(f"Response: {response.content}\n")
    except Exception as e:
        print(f"Chat Failed: {e}\n")

def test_embedding():
    print(f"Testing Embedding Model: {settings.EMBEDDING_MODEL} at {settings.OPENAI_BASE_URL}")
    try:
        embeddings = OpenAIEmbeddings(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.EMBEDDING_MODEL
        )
        # Try to embed a simple string
        vector = embeddings.embed_query("This is a test sentence.")
        print("Embedding Success!")
        print(f"Vector length: {len(vector)}")
        print(f"First 5 values: {vector[:5]}\n")
    except Exception as e:
        print(f"Embedding Failed: {e}\n")

if __name__ == "__main__":
    if not settings.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is missing in .env file")
    else:
        test_chat()
        test_embedding()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import sys
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# Add parent directory to path to import rag modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import settings

app = FastAPI()

# Initialize Clients (Lazy initialization or global)
# We initialize here to fail fast if keys are missing, or we can do it per request.
# For serverless (Vercel), global scope is executed on cold start.

try:
    print("Initializing clients...")
    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is missing")
    if not settings.PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY is missing")
        
    embeddings_client = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.EMBEDDING_MODEL
    )
    chat_client = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.CHAT_MODEL,
        temperature=1
    )
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX_NAME)
    print("Clients initialized successfully")
except Exception as e:
    print(f"Error initializing clients: {str(e)}")
    embeddings_client = None
    chat_client = None
    index = None

class PromptRequest(BaseModel):
    question: str

class ContextItem(BaseModel):
    talk_id: str
    title: str
    chunk: str
    score: float
    speaker_1: Optional[str] = None
    url: Optional[str] = None
    chunk_index: Optional[int] = None

class AugmentedPrompt(BaseModel):
    System: str
    User: str

class PromptResponse(BaseModel):
    response: str
    context: List[Dict[str, Any]]
    Augmented_prompt: AugmentedPrompt

class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int

@app.post("/api/prompt", response_model=PromptResponse)
async def prompt(request: PromptRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if not embeddings_client or not chat_client or not index:
        raise HTTPException(status_code=503, detail="Service unavailable (clients not initialized)")

    # 1. Embed Question
    try:
        query_vector = embeddings_client.embed_query(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # 2. Query Pinecone
    try:
        query_response = index.query(
            vector=query_vector,
            top_k=settings.TOP_K,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

    # 3. Build Context
    context_items = []
    context_text_parts = []
    
    for match in query_response.matches:
        metadata = match.metadata
        chunk_text = metadata.get("chunk_text", "")
        
        # Format context for the LLM
        context_str = f"Title: {metadata.get('title', 'Unknown')}\nSpeaker: {metadata.get('speaker_1', 'Unknown')}\nText: {chunk_text}\n"
        context_text_parts.append(context_str)
        
        # Format context for response
        context_items.append({
            "talk_id": metadata.get("talk_id", ""),
            "title": metadata.get("title", ""),
            "chunk": chunk_text,
            "score": match.score
        })

    full_context_str = "\n---\n".join(context_text_parts)

    # 4. Build Prompts
    system_prompt = (
        "You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). "
        "You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. "
        "If the answer cannot be determined from the provided context, respond: 'I don't know based on the provided TED data.' "
        "Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful."
    )
    
    user_prompt = f"Context:\n{full_context_str}\n\nQuestion: {request.question}"

    # 5. Call LLM
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        chat_response = chat_client.invoke(messages)
        answer = chat_response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    return PromptResponse(
        response=answer,
        context=context_items,
        Augmented_prompt=AugmentedPrompt(
            System=system_prompt,
            User=user_prompt
        )
    )

@app.get("/api/stats", response_model=StatsResponse)
async def stats():
    return StatsResponse(
        chunk_size=settings.CHUNK_SIZE,
        overlap_ratio=settings.OVERLAP_RATIO,
        top_k=settings.TOP_K
    )

# For local running
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

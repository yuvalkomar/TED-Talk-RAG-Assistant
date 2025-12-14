import os
import sys
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
import ast
import time

# Add parent directory to path to import rag modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import settings, utils

def ingest(limit: int = None):
    print("Starting ingestion...")
    
    # Check API keys
    if not settings.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set.")
        return
    if not settings.PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY not set.")
        return

    # Initialize Clients
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        model=settings.EMBEDDING_MODEL
    )
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)

    # Check/Create Index
    index_name = settings.PINECONE_INDEX_NAME
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating index {index_name}...")
        try:
            pc.create_index(
                name=index_name,
                dimension=settings.EMBEDDING_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        except Exception as e:
            print(f"Error creating index: {e}")
            # Fallback or continue if it exists but list failed (rare)
    
    index = pc.Index(index_name)
    
    # Load Dataset
    csv_path = "ted_talks_en.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    if limit:
        df = df.head(limit)
        
    print(f"Processing {len(df)} talks...")
    
    batch_size = 100 # Pinecone upsert batch size
    vectors_to_upsert = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        talk_id = str(row['talk_id'])
        transcript = row['transcript']
        
        # Clean transcript
        cleaned_transcript = utils.clean_text(transcript)
        if not cleaned_transcript:
            continue
            
        # Chunk transcript
        chunks = utils.chunk_text(cleaned_transcript, settings.CHUNK_SIZE, settings.OVERLAP_RATIO)
        
        # Prepare metadata
        # Handle potential parsing errors for stringified lists
        topics = row['topics']
        try:
            # If it looks like a list string, try to parse it, otherwise keep as string
            if isinstance(topics, str) and topics.startswith("["):
                topics = str(ast.literal_eval(topics))
        except:
            pass

        base_metadata = {
            "talk_id": talk_id,
            "title": utils.clean_text(str(row['title'])),
            "speaker_1": utils.clean_text(str(row['speaker_1'])),
            "url": str(row['url']),
            "topics": str(topics),
            "published_date": str(row['published_date'])
        }
        
        # Embed chunks
        for chunk in chunks:
            chunk_id = f"{talk_id}:{chunk['chunk_index']}"
            chunk_text = chunk['text']
            
            # Create embedding
            try:
                # Add a small delay to avoid rate limits or server overload
                time.sleep(0.1)
                embedding = embeddings.embed_query(chunk_text)
                
                # Add chunk specific metadata
                metadata = base_metadata.copy()
                metadata["chunk_text"] = chunk_text
                metadata["chunk_index"] = chunk['chunk_index']
                
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
            except Exception as e:
                print(f"Error embedding chunk {chunk_id}: {e}")
                continue
                
            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                try:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                except Exception as e:
                    print(f"Error upserting batch: {e}")
                    # Retry logic could go here, but keeping it simple
                    
    # Upsert remaining
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"Error upserting final batch: {e}")

    print("Ingestion complete.")

if __name__ == "__main__":
    # Optional: allow passing limit via args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Number of talks to ingest")
    args = parser.parse_args()
    
    ingest(limit=args.limit)

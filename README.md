# TED Talk RAG Assistant

A Retrieval-Augmented Generation (RAG) system for TED Talks, built with FastAPI, Pinecone, and OpenAI.

## Project Structure

- `app/main.py`: FastAPI application with `/api/prompt` and `/api/stats` endpoints.
- `scripts/ingest.py`: Script to ingest TED talks from CSV to Pinecone.
- `rag/`: Shared modules for configuration and utilities.
- `requirements.txt`: Python dependencies.
- `vercel.json`: Vercel deployment configuration.

## Setup

1.  **Clone the repository**
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    Set the following environment variables (in `.env` or your shell):
    ```bash
    export OPENAI_API_KEY="your_openai_key"
    export PINECONE_API_KEY="your_pinecone_key"
    export PINECONE_INDEX="ted-talks" # Optional, defaults to ted-talks
    # PINECONE_HOST is often needed for serverless indexes if not using the index name directly with the new SDK, 
    # but the code handles creation/retrieval by name.
    ```

## Ingestion

To ingest the dataset (`ted_talks_en.csv` must be in the root directory):

```bash
python scripts/ingest.py
# Or to test with a small subset:
python scripts/ingest.py --limit 10
```

This script will:
1.  Create the Pinecone index if it doesn't exist.
2.  Read the CSV.
3.  Clean and chunk transcripts (Chunk size: 1000, Overlap: 15%).
4.  Generate embeddings (Model: RPRTHPB-text-embedding-3-small).
5.  Upsert vectors to Pinecone.

## Running Locally

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## Deployment on Vercel

1.  Install Vercel CLI: `npm i -g vercel`
2.  Run `vercel` in the project root.
3.  Set the environment variables in the Vercel dashboard (Settings -> Environment Variables).

## API Usage

### POST /api/prompt

Query the assistant.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/prompt" \
     -H "Content-Type: application/json" \
     -d '{"question": "What does the speaker say about fear?"}'
```

**Response:**
```json
{
  "response": "...",
  "context": [...],
  "Augmented_prompt": {
    "System": "...",
    "User": "..."
  }
}
```

### GET /api/stats

Get RAG configuration.

**Request:**
```bash
curl "http://localhost:8000/api/stats"
```

**Response:**
```json
{
  "chunk_size": 1000,
  "overlap_ratio": 0.15,
  "top_k": 5
}
```

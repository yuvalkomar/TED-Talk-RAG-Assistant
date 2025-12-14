import os
from dotenv import load_dotenv

load_dotenv()

# Hyperparameters
CHUNK_SIZE = 1000
OVERLAP_RATIO = 0.15
TOP_K = 5

# Models
EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
EMBEDDING_DIMENSIONS = 1536
OPENAI_BASE_URL = "https://api.llmod.ai/v1"

# Environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "ted-talks")

import tiktoken

def clean_text(text: str) -> str:
    """Returns text as is."""
    if not isinstance(text, str):
        return ""
    return text

def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[dict]:
    """Chunks text by tokens."""
    if not text:
        return []
        
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Fallback
        encoding = tiktoken.get_encoding("gpt2")
        
    tokens = encoding.encode(text)
    total_tokens = len(tokens)
    overlap = int(chunk_size * overlap_ratio)
    step = chunk_size - overlap
    
    if step <= 0:
        step = 1 # Prevent infinite loop if overlap is too high
        
    chunks = []
    chunk_index = 0
    
    if total_tokens <= chunk_size:
        chunks.append({
            "text": text,
            "chunk_index": 0,
            "token_count": total_tokens
        })
        return chunks

    for start in range(0, total_tokens, step):
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_str = encoding.decode(chunk_tokens)
        
        chunks.append({
            "text": chunk_text_str,
            "chunk_index": chunk_index,
            "token_count": len(chunk_tokens)
        })
        chunk_index += 1
        
        if end >= total_tokens:
            break
            
    return chunks

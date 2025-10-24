# chunker/semantic_chunker.py
import re
from typing import List, Dict

def semantic_chunk(text: str, max_tokens: int = 250, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of ~max_tokens words.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def create_chunk_objects(doc_id: str, topic: str, text: str, max_tokens: int = 250) -> List[Dict]:
    parts = semantic_chunk(text, max_tokens=max_tokens)
    chunk_objs = []
    for i, c in enumerate(parts):
        chunk_objs.append({
            "chunk_id": f"{doc_id}_{i+1:04d}",
            "doc_id": doc_id,
            "text": c,
            "metadata": {"topic": topic, "chunk_index": i+1}
        })
    return chunk_objs

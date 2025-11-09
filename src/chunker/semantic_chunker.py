# chunker/semantic_chunker.py
# import re
# from typing import List, Dict

# def semantic_chunk(text: str, max_tokens: int = 250, overlap: int = 50) -> List[str]:
#     """
#     Split text into overlapping chunks of ~max_tokens words.
#     """
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + max_tokens
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start += max_tokens - overlap
#     return chunks

# def create_chunk_objects(doc_id: str, topic: str, text: str, max_tokens: int = 250) -> List[Dict]:
#     parts = semantic_chunk(text, max_tokens=max_tokens)
#     chunk_objs = []
#     for i, c in enumerate(parts):
#         chunk_objs.append({
#             "chunk_id": f"{doc_id}_{i+1:04d}",
#             "doc_id": doc_id,
#             "text": c,
#             "metadata": {"topic": topic, "chunk_index": i+1}
#         })
#     return chunk_objs


# chunker/semantic_chunker.py
"""
LangChain-based splitter wrapper.

Produces the same chunk object shape as before:
{
  "chunk_id": "...",
  "doc_id": "...",
  "text": "...",
  "metadata": {...}
}

Requirements (add to requirements.txt):
  langchain
  tiktoken (optional but recommended)
"""

import logging
import re
from typing import List, Dict
logger = logging.getLogger(__name__)

# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
_HAS_LANGCHAIN = True


import tiktoken
_HAS_TIKTOKEN = True


def _approx_tokens(text: str, encoding_name: str = "gpt2") -> int:
    if _HAS_TIKTOKEN:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    return max(1, len(text.split()))

def _paragraphs_with_offsets(text: str):
    parts = []
    cursor = 0
    for part in re.split(r"\n{2,}|\r\n{2,}", text):
        if not part:
            cursor += 2
            continue
        start = text.find(part, cursor)
        end = start + len(part)
        parts.append((part.strip(), start, end))
        cursor = end
    if not parts:
        parts = [(text.strip(), 0, len(text))]
    return parts

def create_chunk_objects(
    doc_id: str,
    topic: str,
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    chunk_size_is_tokens: bool = True,
    encoding_name: str = "gpt2",
    use_langchain: bool = True,
) -> List[Dict]:
    """
    Create chunk objects (list of dicts) using langchain splitters when available.
    If langchain is unavailable, falls back to a simple paragraph/sentence sliding window.
    """
    if use_langchain and _HAS_LANGCHAIN:
        if chunk_size_is_tokens:
            try:
                splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    encoding_name=encoding_name
                )
                parts = splitter.split_text(text)
            except Exception:
                # fallback to recursive character splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n"],
                )
                parts = splitter.split_text(text)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n"],
            )
            parts = splitter.split_text(text)
    else:
        # fallback splitter: paragraphs -> headings -> sliding windows
        paragraphs = _paragraphs_with_offsets(text)
        candidate_parts = []
        heading_pat = re.compile(r"(?m)^(#{1,6}\s+.+|(?:\d+\.)+\s+.+|^[A-Z][A-Za-z\s]{3,}$\n[-=]{3,})", re.MULTILINE)
        for p, _, _ in paragraphs:
            subs = []
            last = 0
            for m in heading_pat.finditer(p):
                if m.start() > last:
                    subs.append(p[last:m.start()].strip())
                subs.append(p[m.start():m.end()].strip())
                last = m.end()
            if last < len(p):
                subs.append(p[last:].strip())
            if not subs:
                subs = [p]
            candidate_parts.extend([s for s in subs if s])

        parts = []
        for part in candidate_parts:
            if chunk_size_is_tokens:
                tokens = part.split()
                stride = max(1, chunk_size - chunk_overlap)
                i = 0
                while i < len(tokens):
                    window = tokens[i:i + chunk_size]
                    parts.append(" ".join(window).strip())
                    if i + chunk_size >= len(tokens):
                        break
                    i += stride
            else:
                stride = max(1, chunk_size - chunk_overlap)
                i = 0
                while i < len(part):
                    parts.append(part[i:i + chunk_size].strip())
                    i += stride

    chunk_objs = []
    cursor = 0
    for i, p_text in enumerate(parts):
        if not p_text:
            continue
        sidx = text.find(p_text, cursor)
        if sidx == -1:
            sidx = text.find(p_text)
        if sidx == -1:
            sidx = cursor
        eidx = sidx + len(p_text)
        approx_tokens = _approx_tokens(p_text, encoding_name)
        sentence_count = max(0, len(re.findall(r'[.!?]\s', p_text)) + (1 if p_text.strip() else 0))
        chunk_objs.append({
            "chunk_id": f"{doc_id}_{i+1:06d}",
            "doc_id": doc_id,
            "text": p_text,
            "metadata": {
                "topic": topic,
                "chunk_index": i+1,
                "char_start": int(sidx),
                "char_end": int(eidx),
                "approx_tokens": int(approx_tokens),
                "sentence_count": int(sentence_count),
            }
        })
        cursor = eidx
    return chunk_objs

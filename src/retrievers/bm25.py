# retrievers/bm25.py
from pathlib import Path
from typing import List, Dict
import re
from rank_bm25 import BM25Okapi

def _tokenize(text: str):
    # simple regex tokenizer — lowercased words only
    return re.findall(r"\b\w+\b", text.lower())

class BM25Retriever:
    """
    Build BM25 index on text files or chunk texts.
    Expects data/processed/texts/<topic>/*.txt or fallback to scanning chunk JSONs.
    """
    def __init__(self, docs_root: str = "data/processed/texts"):
        self.docs_root = Path(docs_root)
        self.doc_ids = []
        self.docs = []
        self._build_index()

    def _build_index(self):
        # iterate topic subdirs and read .txt files
        for topic_dir in sorted(self.docs_root.iterdir()) if self.docs_root.exists() else []:
            if not topic_dir.is_dir():
                continue
            for txt in sorted(topic_dir.glob("*.txt")):
                doc_id = f"{topic_dir.name}/{txt.stem}"
                text = txt.read_text(encoding="utf-8")
                self.doc_ids.append(doc_id)
                self.docs.append(text)
        # if no files found, try to load chunk JSONs (fast fallback)
        if not self.docs:
            chunk_dir = Path("data/processed/chunks")
            if chunk_dir.exists():
                for topic_dir in sorted(chunk_dir.iterdir()):
                    if not topic_dir.is_dir():
                        continue
                    for f in sorted(topic_dir.glob("*_chunks.json")):
                        try:
                            import json
                            arr = json.loads(f.read_text(encoding="utf-8"))
                            for chunk in arr:
                                cid = chunk.get("chunk_id") or f"{topic_dir.name}/{f.stem}"
                                txt = chunk.get("text","")
                                self.doc_ids.append(cid)
                                self.docs.append(txt)
                        except Exception:
                            continue
        tokenized = [_tokenize(d) for d in self.docs]
        if tokenized:
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if not getattr(self, "bm25", None):
            return []
        qtok = _tokenize(query)
        scores = self.bm25.get_scores(qtok)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        out = []
        for idx, score in ranked:
            out.append({
                "id": self.doc_ids[idx],
                "metadata": {"source": self.doc_ids[idx]},
                "text": self.docs[idx],
                "score": float(score)
            })
        return out

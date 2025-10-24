# retrievers/reranker.py
from typing import List, Dict
from sentence_transformers import CrossEncoder

class Reranker:
    """
    Cross-encoder reranker. Heavier; instantiate only if needed.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = None) -> List[Dict]:
        # candidates: list of {"id","text","metadata",...}
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked

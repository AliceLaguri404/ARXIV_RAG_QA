# retrievers/hybrid.py
from typing import List, Dict
from src.retrievers.bm25 import BM25Retriever
from src.retrievers.dense import DenseRetriever
from src.retrievers.reranker import Reranker
import logging
logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Combine BM25 and Dense: get top-N from both, normalize and merge.
    """
    def __init__(self, bm25_k: int = 50, dense_k: int = 50, alpha: float = 0.5, rerank_model: str = None):
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.alpha = alpha
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.reranker = Reranker(rerank_model) if rerank_model else None

    def _normalize(self, scores):
        if not scores:
            return []
        mn, mx = min(scores), max(scores)
        if mx - mn < 1e-9:
            return [1.0 for _ in scores]
        return [(s - mn) / (mx - mn) for s in scores]

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        bm = self.bm25.retrieve(query, k=self.bm25_k)
        dn = self.dense.retrieve(query, k=self.dense_k)
        candidates = {}
        # normalize BM25 scores
        bm_scores = [c.get("score", 0.0) for c in bm]
        bm_norm = self._normalize(bm_scores)
        for c, s in zip(bm, bm_norm):
            candidates[c["id"]] = {"id": c["id"], "text": c["text"], "metadata": c.get("metadata", {}), "score": self.alpha * s}
        # dense: score already similarity-like (0..1), if missing treat as 1
        dn_scores = [c.get("score", 1.0) for c in dn]
        dn_norm = self._normalize(dn_scores)
        for c, s in zip(dn, dn_norm):
            if c["id"] in candidates:
                candidates[c["id"]]["score"] += (1.0 - self.alpha) * s
            else:
                candidates[c["id"]] = {"id": c["id"], "text": c["text"], "metadata": c.get("metadata", {}), "score": (1.0 - self.alpha) * s}
        cand_list = list(candidates.values())
        # optional rerank with cross-encoder
        if self.reranker:
            try:
                cand_list = self.reranker.rerank(query, cand_list, top_k=k)
                return cand_list
            except Exception as e:
                logger.exception("Reranker failed: %s", e)
        # otherwise sort by score
        return sorted(cand_list, key=lambda x: x.get("score", 0.0), reverse=True)[:k]

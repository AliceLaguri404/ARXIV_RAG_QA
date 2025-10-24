# retrievers/manager.py
import logging
logger = logging.getLogger(__name__)

# lazy imports to avoid heavy models loading on import if not needed
from typing import Dict

class RetrieverManager:
    def __init__(self, config: Dict = None):
        cfg = config or {}
        logger.info("Initializing RetrieverManager...")
        # instantiate only lightweight ones first; heavy ones instantiated here but you can change
        from retrievers.dense import DenseRetriever
        from retrievers.hybrid import HybridRetriever
        # optional heavy components; instantiate if you plan to use
        from retrievers.bm25 import BM25Retriever
        from retrievers.hyde import HyDERetriever
        # from retrievers.reranker import Reranker

        self._instances = {
            "dense": DenseRetriever(**cfg.get("dense", {})),
            "bm25": BM25Retriever(**cfg.get("bm25", {})),
            "hyde": HyDERetriever(**cfg.get("hyde", {})),
            "hybrid": HybridRetriever(**cfg.get("hybrid", {}))
            # "reranker": Reranker(**cfg.get("reranker", {}))
        }

    def list(self):
        return list(self._instances.keys())

    def get(self, name: str):
        name = (name or "hybrid").lower()
        if name not in self._instances:
            raise KeyError(f"Unknown retriever '{name}'. Available: {self.list()}")
        return self._instances[name]

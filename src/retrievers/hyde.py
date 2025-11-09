# retrievers/hyde.py
from typing import List, Dict
from src.embeddings.encoder import Embedder  # your embeddings/encoder.py
from src.qa.llm_runner import LLMRunner
import chromadb
import logging

logger = logging.getLogger(__name__)

class HyDERetriever:
    """
    HyDE: generate a hypothetical doc via LLM, embed it, and do dense retrieval using that embedding.
    """
    def __init__(self, embed_model: str = "all-MiniLM-L6-v2", db_path: str = "data/vectorstore", collection_name: str = "rag_docs"):
        self.embedder = Embedder(model_name=embed_model)
        # LLMRunner will be defined in qa/llm_runner.py (REST-based)
        self.hyde_llm = LLMRunner()
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def _generate_hypo(self, query: str) -> str:
        prompt = f"Write a concise hypothetical document (1-2 short paragraphs) that would answer the question: {query}"
        # hyde uses LLMRunner.answer(prompt, context) — supply empty context
        return self.hyde_llm.answer(prompt, "")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        hypo = self._generate_hypo(query)
        if not hypo:
            hypo = query
        emb = self.embedder.encode([hypo], normalize_embeddings=True)
        resp = self.collection.query(query_embeddings=emb.tolist(), n_results=k)
        docs = resp.get("documents", [[]])[0]
        metas = resp.get("metadatas", [[]])[0]
        ids = resp.get("ids", [[]])[0]
        return [{"id": i, "metadata": m or {}, "text": d} for i,m,d in zip(ids, metas, docs)]

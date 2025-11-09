# # retrievers/dense.py
# from typing import List, Dict, Optional
# from sentence_transformers import SentenceTransformer
# import chromadb
# import logging

# logger = logging.getLogger(__name__)

# class DenseRetriever:
#     def __init__(self,
#                  db_path: str = "data/vectorstore",
#                  collection_name: str = "rag_docs",
#                  model_name: str = "all-MiniLM-L6-v2"):
#         """
#         Dense retriever using sentence-transformers + Chroma.
#         Returns list of {"id","metadata","text","score"}.
#         """
#         logger.info("Initializing DenseRetriever: model=%s db=%s", model_name, db_path)
#         self.model = SentenceTransformer(model_name)
#         self.client = chromadb.PersistentClient(path=db_path)
#         # get_collection will raise if missing; get_or_create_collection is safer:
#         try:
#             self.collection = self.client.get_collection(collection_name)
#         except Exception:
#             self.collection = self.client.get_or_create_collection(collection_name)

#     def retrieve(self, query: str, k: int = 5) -> List[Dict]:
#         q_emb = self.model.encode([query], normalize_embeddings=True).tolist()
#         resp = self.collection.query(query_embeddings=q_emb, n_results=k)
#         docs = resp.get("documents", [[]])[0]
#         metas = resp.get("metadatas", [[]])[0]
#         ids = resp.get("ids", [[]])[0]
#         distances = resp.get("distances", [[]])[0] if resp.get("distances") else [None]*len(ids)
#         results = []
#         for _id, meta, text, dist in zip(ids, metas, docs, distances):
#             results.append({
#                 "id": _id,
#                 "metadata": meta or {},
#                 "text": text,
#                 # convert distance to similarity-ish score if present (smaller dist -> larger score)
#                 "score": (1.0/(1.0+dist)) if dist is not None else None
#             })
#         return results
    
# retrievers/retriever.py
import chromadb
from src.embeddings.encoder import Embedder

class DenseRetriever:
    def __init__(self, db_path="data/vectorstore", collection_name="rag_docs", model_name="all-MiniLM-L6-v2"):

        self.model = Embedder()
        self.client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.get_or_create_collection(collection_name)

    def retrieve(self, query: str, k: int = 5):
        query_embedding = self.model.encode([query])
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        return [
            {"id": i, "metadata": m, "text": d}
            for i, m, d in zip(ids, metas, docs)
        ]


# vector_store/chromadb_client.py
import chromadb
from chromadb.config import Settings

class ChromaDBClient:
    def __init__(self, persist_path="data/vectorstore", collection_name="rag_docs"):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_chunks(self, chunk_objs, embeddings):
        ids = [c["chunk_id"] for c in chunk_objs]
        texts = [c["text"] for c in chunk_objs]
        metadatas = [c["metadata"] for c in chunk_objs]
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())

    def query(self, query_embedding, n_results=3):
        return self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)

    def count(self):
        return self.collection.count()

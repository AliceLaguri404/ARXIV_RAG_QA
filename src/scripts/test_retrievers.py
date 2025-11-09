# scripts/test_retrievers.py
import time
from src.retrievers.manager import RetrieverManager

mgr = RetrieverManager()
queries = [
    "What is retrieval augmented generation?",
    "How does knowledge distillation reduce model size?",
    "What is HyDE in retrieval?"
]

for name in mgr.list():
    r = mgr.get(name)
    print("=== Testing retriever:", name)
    for q in queries:
        t0 = time.time()
        try:
            res = r.retrieve(q, k=5)
            took = (time.time() - t0) * 1000
            print(f"Q: {q[:40]}... took {int(took)}ms, results: {[x['id'] for x in res]}")
        except Exception as e:
            print("Error for", name, e)
    print()

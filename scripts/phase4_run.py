# scripts/phase4_run.py
import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
import logging
import time
from retrievers.manager import RetrieverManager
from qa.llm_runner import LLMRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_context(results, per_snippet_chars=4000):
    """Build short context string from retrieved results (use summary if available)."""
    parts = []
    for r in results:
        meta = r.get("metadata", {}) or {}
        summary = meta.get("summary")
        if summary and len(summary.strip()) > 20:
            snippet = summary
        else:
            text = r.get("text", "")
            snippet = text[:per_snippet_chars] + "..." if len(text) > per_snippet_chars else text
        parts.append(f"[{r['id']}]\n{snippet}")
    return "\n\n---\n\n".join(parts)

def main():
    logger.info("Initializing retrievers and LLM...")
    mgr = RetrieverManager()
    llm = LLMRunner()
    # Testing
    retriever_name = "hybrid"
    k = 5
    logger.info("Phase 4 RAG CLI ready. Use commands:")

    while True:
        try:
            query = input(f"\n[{retriever_name}][k={k}] Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in (":exit", ":quit", "exit", "quit"):
            break
        if query.startswith(":list"):
            print("Available retrievers:", ", ".join(mgr.list()))
            continue
        if query.startswith(":use"):
            _, _, new_name = query.partition(" ")
            new_name = new_name.strip()
            if not new_name:
                print("Usage: :use <name>")
                continue
            try:
                _ = mgr.get(new_name)
                retriever_name = new_name
                print(f"✅ Switched retriever to {retriever_name}")
            except KeyError:
                print(f"Unknown retriever: {new_name}")
            continue
        if query.startswith(":k"):
            try:
                k = int(query.split()[1])
                print(f"✅ k set to {k}")
            except Exception:
                print("Usage: :k <num>")
            continue

        retriever = mgr.get(retriever_name)
        t0 = time.time()
        results = retriever.retrieve(query, k=k)
        t1 = time.time()
        context = build_context(results)
        t2 = time.time()
        answer = llm.answer(query, context)
        t3 = time.time()

        print("\n📚 Retrieved Sources:")
        for i, r in enumerate(results, 1):
            topic = r.get("metadata", {}).get("topic", "-")
            score = r.get("score")
            print(f"{i}. {r['id']} (topic={topic}, score={score})")

        print("\n🤖 Answer:\n", answer)
        print(f"\n⏱ Timings (ms): retrieve={int((t1-t0)*1000)}, build_ctx={int((t2-t1)*1000)}, llm={int((t3-t2)*1000)}")

if __name__ == "__main__":
    main()

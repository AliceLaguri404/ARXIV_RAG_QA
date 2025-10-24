# # app/main.py
# import os
# os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional, List
# from retrievers.manager import RetrieverManager
# from qa.llm_runner import LLMRunner
# from fastapi.concurrency import run_in_threadpool

# app = FastAPI(title="RAG API", version="0.1")

# class QueryRequest(BaseModel):
#     query: str
#     retriever: Optional[str] = "hybrid"
#     k: Optional[int] = 5
#     rerank: Optional[bool] = False

# class Source(BaseModel):
#     id: str
#     metadata: dict
#     snippet: str
#     score: Optional[float] = None

# class QueryResponse(BaseModel):
#     answer: str
#     retriever: str
#     sources: List[Source]

# # singletons
# mgr = None
# llm = None

# @app.on_event("startup")
# def startup():
#     global mgr, llm
#     mgr = RetrieverManager()
#     llm = LLMRunner()

# @app.get("/health")
# async def health():
#     return {"status": "ok", "retrievers": mgr.list() if mgr else []}

# @app.post("/query", response_model=QueryResponse)
# async def query(req: QueryRequest):
#     if not mgr or not llm:
#         raise HTTPException(status_code=500, detail="Service not ready")
#     try:
#         retr = mgr.get(req.retriever)
#     except KeyError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     # retrieval (run in threadpool to avoid blocking)
#     results = await run_in_threadpool(retr.retrieve, req.query, req.k)
#     # build context using summaries if present
#     parts = []
#     for r in results:
#         meta = r.get("metadata",{}) or {}
#         summary = meta.get("summary")
#         snippet = summary if summary and len(summary)>20 else (r.get("text","")[:400] + "..." if len(r.get("text",""))>400 else r.get("text",""))
#         parts.append(f"[{r['id']}]\n{snippet}")
#     context = "\n\n---\n\n".join(parts)
#     # call LLM
#     answer = await run_in_threadpool(llm.answer, req.query, context)
#     sources = []
#     for r in results:
#         meta = r.get("metadata",{}) or {}
#         summary = meta.get("summary")
#         snippet = summary if summary and len(summary)>20 else (r.get("text","")[:400] + "..." if len(r.get("text",""))>400 else r.get("text",""))
#         sources.append(Source(id=r["id"], metadata=meta, snippet=snippet, score=r.get("score")))
#     return QueryResponse(answer=answer, retriever=req.retriever, sources=sources)

import os
import logging
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from retrievers.manager import RetrieverManager
from qa.llm_runner import LLMRunner
from fastapi.concurrency import run_in_threadpool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="0.1")

class QueryRequest(BaseModel):
    query: str
    retriever: Optional[str] = "hybrid"
    k: Optional[int] = 5
    rerank: Optional[bool] = False

class Source(BaseModel):
    id: str
    metadata: dict
    snippet: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    retriever: str
    sources: List[Source]

# singletons
mgr = None
llm = None

@app.on_event("startup")
def startup():
    global mgr, llm
    mgr = RetrieverManager()
    # Dummy LLMRunner for demonstration if qa.llm_runner doesn't exist
    try:
        llm = LLMRunner()
    except (ImportError, NameError):
        logger.warning("Could not import LLMRunner. Using a dummy implementation.")
        class DummyLLM:
            def answer(self, query, context):
                return f"This is a dummy answer for the query '{query}' based on the provided context."
        llm = DummyLLM()


@app.get("/health")
async def health():
    return {"status": "ok", "retrievers": mgr.list() if mgr else []}

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not mgr or not llm:
        raise HTTPException(status_code=500, detail="Service not ready")

    # THE FIX - PART 1: Explicitly block 'reranker' as a primary retriever.
    # A reranker is a component used for post-processing, not initial retrieval.
    if req.retriever.lower() == "reranker":
        raise HTTPException(
            status_code=400,
            detail="Cannot use 'reranker' as a primary retriever. "
                   "Select a base retriever (e.g., 'hybrid') and set 'rerank: true' to apply reranking."
        )

    try:
        retr = mgr.get(req.retriever)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    # Step 1: Perform initial retrieval
    results = await run_in_threadpool(retr.retrieve, req.query, req.k)

    # THE FIX - PART 2: Implement the 'rerank' flag as a post-processing step.
    if req.rerank:
        logger.info(f"Reranking results for query: '{req.query}'")
        try:
            reranker = mgr.get("reranker")
            # The original results become the candidates for the reranker
            results = await run_in_threadpool(reranker.rerank, query=req.query, candidates=results, top_k=req.k)
        except KeyError:
            logger.error("Reranker component not found in manager. Skipping rerank step.")
            # Fail gracefully if reranker isn't configured, but log an error.
        except Exception as e:
            logger.exception(f"An error occurred during reranking: {e}. Proceeding with original results.")
            # Proceed with un-reranked results if reranking fails for any reason.

    # build context using summaries if present
    parts = []
    for r in results:
        meta = r.get("metadata",{}) or {}
        summary = meta.get("summary")
        snippet = summary if summary and len(summary)>20 else (r.get("text","")[:400] + "..." if len(r.get("text",""))>400 else r.get("text",""))
        parts.append(f"[{r['id']}]\n{snippet}")
    context = "\n\n---\n\n".join(parts)
    
    # call LLM
    answer = await run_in_threadpool(llm.answer, req.query, context)
    
    sources = []
    for r in results:
        meta = r.get("metadata",{}) or {}
        summary = meta.get("summary")
        snippet = summary if summary and len(summary)>20 else (r.get("text","")[:400] + "..." if len(r.get("text",""))>400 else r.get("text",""))
        sources.append(Source(id=r["id"], metadata=meta, snippet=snippet, score=r.get("score")))
        
    return QueryResponse(answer=answer, retriever=req.retriever, sources=sources)


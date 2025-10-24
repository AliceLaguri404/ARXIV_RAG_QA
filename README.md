# RAG Pipeline — Retrieval-Augmented Generation System

A modular RAG system: fetch arXiv papers → extract text → chunk → embed → index → retrieve (dense, BM25, hybrid, HyDE, rerank) → LLM answer → FastAPI + Gradio UI.

## Features
- Multi-retriever support: Dense, BM25, Hybrid, HyDE, Cross-encoder reranker
- Vector store: ChromaDB + optional FAISS backend
- LLM integration via Groq (REST) or any compatible backend
- FastAPI endpoints and Gradio interactive UI
- Dockerized for reproducible deployment

## Requirements
See `requirements.txt` (Linux / Docker). macOS users: use `requirements.mac.txt` and install torch/faiss via conda as noted.

## Environment (.env)
Create `.env` with:

GROQ_API_KEY = sk-...
GROQ_MODEL = llama-3.1-8b-instant
PUBLIC_URL_BASE = http://127.0.0.1:8000 OR http://localhost:8000
FOR UI: http://127.0.0.1:7860


## Run locally (development)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env   # or create .env manually

Phases:
1. Fetch from Arvix: python3 -m scripts.phase1_run
2. Extract Text: python3 -m scripts.phase2_run
3. Chunk and Embed: python3 -m scripts.phase3_run
4. Query CLI: python3 -m scripts.phase4_run
5. FastAPI: uvicorn app.main:app --reload --port 8000
6. UI: python3 ui/gradio_app.py
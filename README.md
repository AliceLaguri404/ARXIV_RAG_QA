# 🧠 ARXIV RAG QA — Retrieval-Augmented Generation System

> **A modular RAG pipeline** that fetches, parses, and indexes research papers from **arXiv**, then answers questions using hybrid retrieval (dense + BM25 + HyDE + reranking) and an integrated **LLM backend** (Groq or compatible).
>
> 🚀 Deployable via **FastAPI** or **Gradio UI**, and fully containerized for reproducible setups.

---

## 🧩 Architecture Overview

```text
arxiv → PDF/Text extraction → Chunking → Embedding → Vector store
→ Multi-retriever search → Reranking → LLM → API/UI output
```

**Core Components:**

* 📄 `phase1`: Fetch papers from arXiv
* 🧹 `phase2`: Extract and clean text
* 🧩 `phase3`: Chunk + Embed + Index
* 🔎 `phase4`: Query (CLI or API)
* ⚙️ `FastAPI` backend — REST endpoints
* 🎨 `Gradio` interface — interactive Q&A

---

## 🧰 Tech Stack

| Category         | Tools                                        |
| ---------------- | -------------------------------------------- |
| **Embedding**    | Sentence-Transformers (`all-mpnet-base-v2`)  |
| **Vector Store** | ChromaDB (persistent client)                 |
| **Retriever**    | Dense, BM25, Hybrid, HyDE, Cross-Encoder     |
| **Backend**      | FastAPI, Uvicorn                             |
| **UI**           | Gradio                                       |
| **Infra**        | Docker, Python 3.11                          |
| **LLM**          | Groq API (or any OpenAI-compatible endpoint) |

---

## ⚙️ Setup — Local Development

### 1️⃣ Clone the repository

```bash
git clone https://github.com/AliceLaguri404/ARXIV_RAG_QA.git
cd ARXIV_RAG_QA
```

### 2️⃣ Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Set up your `.env`

Create `.env` in the project root (or copy from `.env.sample`):

```bash
GROQ_API_KEY=sk-xxxxxxxxxxxx
GROQ_MODEL=llama-3.1-8b-instant
PUBLIC_URL_BASE=http://127.0.0.1:8000
```

---

## 🧮 Pipeline Phases (Local Run)

Run from project **root** (each as a module under `src/`):

```bash
# 1️⃣ Fetch from arXiv
python -m src.scripts.phase1_run

# 2️⃣ Extract and clean text
python -m src.scripts.phase2_run

# 3️⃣ Chunk + Embed + Index
python -m src.scripts.phase3_run

# 4️⃣ Query CLI
python -m src.scripts.phase4_run
```

---

## ⚡ API & UI (Local Run)

### ▶️ Run FastAPI

```bash
uvicorn src.app.main:app --reload --port 8000
```

Then open: [http://127.0.0.1:8000/docs]

### 🎨 Run Gradio UI

```bash
python -m src.ui.gradio_app
```

Then open: [http://127.0.0.1:7860]

---

## 🐳 Docker Deployment

### 🔧 1. Build the image

```bash
docker build -t arxiv_rag_qa:latest .
```

### 📦 2. Run FastAPI (default)

```bash
mkdir -p ./data ./cache
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/cache:/cache" \
  -e CHROMA_PERSIST_PATH=/data/vectorstore \
  arxiv_rag_qa:latest
```

→ Open: [http://127.0.0.1:8000]

### 🖥 3. Run Gradio UI

```bash
docker run --rm -it \
  -p 7860:7860 \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/cache:/cache" \
  -e RUN_MODE=gradio \
  arxiv_rag_qa:latest
```

→ Open: [http://127.0.0.1:7860]

### 🧠 4. Run ingestion pipeline (inside container)

```bash
docker run --rm -it \
  -v "$(pwd)/data:/data" \
  -v "$(pwd)/cache:/cache" \
  arxiv_rag_qa:latest \
  python -m src.scripts.phase3_run
```

---

## 📁 Project Structure

```
src/
 ├── app/                  # FastAPI backend
 ├── ui/                   # Gradio interface
 ├── scripts/              # Pipeline phases
 ├── chunker/              # Semantic/recursive chunking
 ├── embeddings/           # SentenceTransformer-based encoder
 ├── vector_store/         # ChromaDB client
 └── qa/                   # LLM runner + retriever logic
data/
 ├── raw/                  # Downloaded PDFs
 ├── processed/            # Extracted and chunked text
 └── vectorstore/          # Chroma persistent storage
docker-entrypoint.sh
Dockerfile
requirements.txt
.env
```

---

## 🧩 Environment Variables (Key)

| Variable                        | Description                | Example                |
| ------------------------------- | -------------------------- | ---------------------- |
| `GROQ_API_KEY`                  | API key for LLM backend    | `sk-xxxxxx`            |
| `GROQ_MODEL`                    | Model name                 | `llama-3.1-8b-instant` |
| `CHROMA_PERSIST_PATH`           | ChromaDB storage directory | `/data/vectorstore`    |
| `RUN_MODE`                      | `api` or `gradio`          | `gradio`               |
| `HF_HOME`, `TRANSFORMERS_CACHE` | Model cache path           | `/cache/huggingface`   |

---

## 🚀 Quick Commands

| Task           | Command                                                          |
| -------------- | ---------------------------------------------------------------- |
| Rebuild image  | `docker build --no-cache -t arxiv_rag_qa:latest .`               |
| Run API        | `docker run -p 8000:8000 arxiv_rag_qa:latest`                    |
| Run Gradio     | `docker run -p 7860:7860 -e RUN_MODE=gradio arxiv_rag_qa:latest` |
| Ingest papers  | `python -m src.scripts.phase3_run`                               |
| Run all phases | `make pipeline` *(optional if you add a Makefile)*               |

---

## 🧑‍💻 Future Enhancements

* [ ] Vector compression with FAISS + IVF
* [ ] Caching for re-embedding skip
* [ ] Multi-LLM evaluation metrics
* [ ] CI/CD pipeline with Docker Compose
* [ ] Auto-sync with arXiv RSS feeds

---

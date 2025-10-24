# ui/gradio_app.py
import os
import time
import requests
import html
from typing import List, Dict

import gradio as gr

# Configure this to point at your running backend
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/query")

# Helper: call FastAPI /query
def call_query_api(query: str, retriever: str, k: int, rerank: bool):
    payload = {"query": query, "retriever": retriever, "k": k, "rerank": rerank}
    start = time.time()
    try:
        r = requests.post(API_URL, json=payload, timeout=60)
        latency = int((time.time() - start) * 1000)
    except Exception as e:
        return {"error": f"Network error contacting API: {e}", "latency_ms": None, "data": None}
    if r.status_code != 200:
        return {"error": f"API error {r.status_code}: {r.text}", "latency_ms": latency, "data": None}
    try:
        data = r.json()
    except Exception as e:
        return {"error": f"Failed to parse API JSON: {e}", "latency_ms": latency, "data": None}
    return {"error": None, "latency_ms": latency, "data": data}

# Helper: format sources into markdown with clickable PDF or source id
def format_sources_md(sources: List[Dict]) -> str:
    if not sources:
        return "_No sources returned._"
    lines = []
    for i, s in enumerate(sources, start=1):
        meta = s.get("metadata", {}) or {}
        # prefer title or topic if available
        title = meta.get("title") or meta.get("topic") or meta.get("source") or s.get("id")
        snippet = s.get("snippet") or meta.get("summary") or ""
        # if pdf_path exists, create link (assumes server can serve files or it's a URL)
        pdf_path = meta.get("pdf_path") or meta.get("pdf_url") or None
        if pdf_path:
            # escape and ensure clickable
            link = f'<a href="{html.escape(pdf_path)}" target="_blank">{html.escape(title)}</a>'
        else:
            link = html.escape(title)
        score = s.get("score")
        score_txt = f" — score: {score:.3f}" if score is not None else ""
        # snippet escape & truncation
        snippet_trunc = (snippet[:350] + "...") if len(snippet) > 350 else snippet
        lines.append(f"**{i}. {link}**{score_txt}\n\n> {html.escape(snippet_trunc)}\n")
    return "\n\n".join(lines)

# Gradio function
def ask(query: str, retriever: str, k: int, rerank: bool):
    if not query or query.strip() == "":
        return "Please enter a query.", "", ""
    result = call_query_api(query=query, retriever=retriever, k=k, rerank=rerank)
    if result["error"]:
        return f"Error: {result['error']}", "", ""
    data = result["data"]
    answer = data.get("answer") or "[no answer]"
    sources = data.get("sources") or []
    # build sources markdown
    sources_md = format_sources_md(sources)
    meta = f"Retriever: **{data.get('retriever','unknown')}** • {len(sources)} sources • latency {result['latency_ms']} ms"
    return answer, meta, sources_md

# available retriever names (keep in sync with retriever manager)
RETRIEVERS = ["dense", "bm25", "hybrid", "hyde", "reranker"]

with gr.Blocks(title="RAG QA Explorer") as demo:
    gr.Markdown("# Retrieval-Augmented Generation — Explorer")
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(lines=4, placeholder="Enter your question here...", label="Question")
            with gr.Row():
                retriever_dropdown = gr.Dropdown(RETRIEVERS, value="hybrid", label="Retriever")
                k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Top-K")
                rerank_checkbox = gr.Checkbox(label="Post-rank with cross-encoder", value=False)
            ask_btn = gr.Button("Ask")
        with gr.Column(scale=2):
            answer_box = gr.Textbox(lines=12, label="Answer")
            meta_box = gr.Markdown("", label="Metadata")

    gr.Markdown("### Sources:")
    sources_md = gr.Markdown("", elem_id="sources_md")

    # connect
    ask_btn.click(fn=ask,
                  inputs=[query_input, retriever_dropdown, k_slider, rerank_checkbox],
                  outputs=[answer_box, meta_box, sources_md])

if __name__ == "__main__":
    # run on port 7860
    demo.launch(server_name="127.0.0.1", server_port=int(os.getenv("GRADIO_PORT", 7860)))

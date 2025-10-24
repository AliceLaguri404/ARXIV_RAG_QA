# fetch/arxiv_fetcher.py
# requests + feedparser based; saves files named by topic for easy inspection
import requests
import feedparser
import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
PDF_BASE = DATA_DIR / "pdfs"
META_BASE = DATA_DIR / "meta"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_BASE.mkdir(parents=True, exist_ok=True)
META_BASE.mkdir(parents=True, exist_ok=True)

ARXIV_API_BASE = "https://export.arxiv.org/api/query"  # use HTTPS explicitly
HEADERS = {"User-Agent": "rag-pipeline/1.0 (mailto:your-email@example.com)"}  # replace email if you like

def _build_query_url(query: str, start: int = 0, max_results: int = 10, sort_by: str = "relevance"):
    q = quote_plus(query)
    return f"{ARXIV_API_BASE}?search_query={q}&start={start}&max_results={max_results}&sortBy={sort_by}&sortOrder=descending"

def _download_file(url: str, dest: Path, timeout: int = 30):
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=timeout, headers=HEADERS, allow_redirects=True) as r:
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            logger.warning("Download attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(1 + attempt)
    logger.error("Failed to download %s after retries", url)
    return False

def _sanitize_topic(topic: str) -> str:
    """
    Make a filesystem-safe topic name: lowercase, underscores, alphanum and hyphens only.
    """
    t = topic.strip().lower()
    # replace spaces and slashes with underscore
    t = re.sub(r"[\/\s]+", "_", t)
    # remove characters not alphanumeric, underscore, or hyphen
    t = re.sub(r"[^a-z0-9_\-]", "", t)
    # collapse multiple underscores
    t = re.sub(r"_+", "_", t)
    return t or "topic"

def _slugify_title(title: str, max_length: int = 120) -> str:
    """
    Turn a paper title into a safe filename fragment:
      - lowercase, ascii-ish, replace spaces with underscores
      - remove characters not alnum/_/-
      - collapse underscores, trim length
    """
    if not title:
        return "paper"
    t = title.strip().lower()
    # replace spaces and slashes with underscore
    t = re.sub(r"[\/\s]+", "_", t)
    # remove non-ascii by transliteration fallback (best-effort)
    try:
        import unicodedata
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    # keep alnum, underscore, hyphen
    t = re.sub(r"[^a-z0-9_\-]", "", t)
    t = re.sub(r"_+", "_", t)
    # trim
    if len(t) > max_length:
        t = t[:max_length].rstrip("_")
    return t or "paper"


def fetch_arxiv_papers_for_query_topic(topic_name: str, query: str, max_results: int = 5, sort_by: str = "relevance") -> List[Dict]:
    """
    Query arXiv for a single topic and save PDFs + metadata using the topic name for filenames.
    Saves to:
      data/raw/pdfs/<topic_name>/{topic}_{i}.pdf
      data/raw/meta/<topic_name>/{topic}_{i}.json
    Returns list of metadata dicts (one per entry).
    """
    safe_topic = _sanitize_topic(topic_name)
    logger.info("Searching arXiv for topic='%s' (query=%s) max_results=%d", safe_topic, query, max_results)

    # create topic-level dirs
    topic_pdf_dir = PDF_BASE / safe_topic
    topic_meta_dir = META_BASE / safe_topic
    topic_pdf_dir.mkdir(parents=True, exist_ok=True)
    topic_meta_dir.mkdir(parents=True, exist_ok=True)

    url = _build_query_url(query, start=0, max_results=max_results, sort_by=sort_by)
    logger.debug("Query URL: %s", url)

    resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    resp.raise_for_status()
    feed = feedparser.parse(resp.text)

    results = []
    if not feed.entries:
        logger.info("No entries returned for topic '%s' (query: %s)", safe_topic, query)
        return results

    # iterate entries and save as topic_{i}.pdf / topic_{i}.json
    for idx, entry in enumerate(feed.entries, start=1):
        try:
            entry_id = entry.get("id", "")
            paper_id = entry_id.rstrip("/").split("/")[-1] if entry_id else f"paper_{abs(hash(entry.get('title','')))}"

            # find pdf link
            pdf_url = None
            for link in entry.get("links", []):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break
                href = link.get("href", "")
                if href.endswith(".pdf"):
                    pdf_url = href
                    break
            if not pdf_url and entry_id:
                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

            # target filenames using topic name + index
            # use slugified title for filenames, fallback to topic_index if necessary
            title = entry.get("title") or ""
            slug = _slugify_title(title)
            # ensure uniqueness per topic: append index
            pdf_filename = f"{slug}_{idx}.pdf"
            meta_filename = f"{slug}_{idx}.json"
            pdf_path = topic_pdf_dir / pdf_filename
            meta_path = topic_meta_dir / meta_filename

            metadata = {
                "saved_filename": pdf_filename,
                "saved_metadata": meta_filename,
                "id": paper_id,
                "title": entry.get("title"),
                "authors": [a.get("name") for a in entry.get("authors", [])] if entry.get("authors") else [],
                "summary": entry.get("summary"),
                "published": entry.get("published"),
                "updated": entry.get("updated"),
                "pdf_url": pdf_url,
                "entry_id": entry_id,
                "topic": topic_name,
                "topic_safe": safe_topic,
                "topic_index": idx
            }

            # download pdf if possible
            if pdf_url:
                if not pdf_path.exists():
                    ok = _download_file(pdf_url, pdf_path)
                    if not ok:
                        logger.warning("Could not download PDF for %s (%s). Will still write metadata.", paper_id, pdf_url)
                else:
                    logger.info("PDF already exists: %s", pdf_path)
                metadata["pdf_path"] = str(pdf_path) if pdf_path.exists() else None
            else:
                logger.warning("No PDF URL found for %s", paper_id)
                metadata["pdf_path"] = None

            # write metadata to disk (overwrites if same topic+index existed)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info("Saved: %s (meta: %s)", pdf_path if metadata["pdf_path"] else '<no-pdf>', meta_path)
            results.append(metadata)
        except Exception as e:
            logger.exception("Failed to handle entry for topic %s: %s", safe_topic, e)

    logger.info("Fetched %d papers for topic %s", len(results), safe_topic)
    return results

def fetch_many_queries(queries: Dict[str, str], per_topic: int = 3) -> Dict[str, List[Dict]]:
    all_results = {}
    for topic_name, query in queries.items():
        logger.info("=== Fetching topic: %s", topic_name)
        try:
            metas = fetch_arxiv_papers_for_query_topic(topic_name, query, max_results=per_topic)
            all_results[topic_name] = metas
            time.sleep(0.5)  # be polite
        except Exception as e:
            logger.exception("Failed fetching topic %s: %s", topic_name, e)
            all_results[topic_name] = []
    return all_results

if __name__ == "__main__":
    # quick test
    res = fetch_arxiv_papers_for_query_topic("retrieval_augmented_generation", '"retrieval augmented generation" OR RAG AND cat:cs.CL', max_results=2)
    print("Got", len(res))

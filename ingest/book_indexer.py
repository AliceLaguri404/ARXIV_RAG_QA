# ingest/book_indexer.py
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
MANUAL_META_DIR = DATA_DIR / "manual_meta"
MANUAL_META_DIR.mkdir(parents=True, exist_ok=True)

def index_local_book(pdf_path: str, title: Optional[str] = None, author: Optional[str] = None, extra: Optional[Dict] = None):
    """
    Index a local PDF by writing a metadata JSON. Does NOT move the PDF.
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    doc_id = pdf.stem
    meta = {
        "id": doc_id,
        "title": title or doc_id,
        "author": author or "Unknown",
        "path": str(pdf.resolve()),
        "added_on": datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        meta.update(extra)

    out_path = MANUAL_META_DIR / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("Indexed local book: %s -> %s", pdf, out_path)
    return meta


if __name__ == "__main__":
    # Example usage:
    print(index_local_book("data/raw/pdfs/sample.pdf", title="Sample Paper", author="Jane Doe"))

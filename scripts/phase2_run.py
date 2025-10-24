# scripts/phase2_run.py
import logging
from pathlib import Path
import json
from extract.pdf_extractor import PDFExtractor
from extract.cleaner.header_footer_remover import remove_headers_footers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PDF_DIR = Path("data/raw/pdfs")
PROCESSED_DIR = Path("data/processed/texts")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_topic(topic_dir: Path):
    extractor = PDFExtractor()
    topic = topic_dir.name
    out_dir = PROCESSED_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(topic_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs in %s", topic_dir)
        return

    for pdf in pdf_files:
        out_txt = out_dir / (pdf.stem + ".txt")
        try:
            logger.info("Extracting %s...", pdf.name)
            text = extractor.extract_text(pdf)
            if not text:
                continue
            clean_text = remove_headers_footers(text)
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(clean_text)
            logger.info("✅ Saved cleaned text: %s", out_txt)
        except Exception as e:
            logger.exception("Failed to process %s: %s", pdf, e)

def main():
    logger.info("Starting Phase 2: Extract + Clean")
    for topic_dir in RAW_PDF_DIR.iterdir():
        if topic_dir.is_dir():
            process_topic(topic_dir)
    logger.info("Phase 2 complete — cleaned text in data/processed/texts/")

if __name__ == "__main__":
    main()

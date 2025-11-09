# extract/pdf_extractor.py
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self):
        pass

    def has_text_layer(self, page: fitz.Page) -> bool:
        """Return True if page has selectable text."""
        text = page.get_text("text")
        return bool(text.strip())

    def extract_text(self, pdf_path: Path, min_len: int = 500) -> str:
        """
        Extract main text (Abstract → Conclusion/References) from PDF.
        """
        doc = fitz.open(pdf_path)
        text_blocks = []
        for page in doc:
            if self.has_text_layer(page):
                text_blocks.append(page.get_text("text"))
            else:
                # minimal fallback (skip heavy OCR for simplicity)
                logger.warning("Page without text layer in %s", pdf_path.name)
        doc.close()

        full_text = "\n".join(text_blocks)

        # Heuristic crop between Abstract and Conclusion/References
        lower = full_text.lower()
        start = re.search(r"(abstract[\s:]+)", lower)
        end = re.search(r"(references|bibliography|conclusion[\s:]+)", lower)

        start_i = start.start() if start else 0
        end_i = end.start() if end else len(full_text)
        body = full_text[start_i:end_i].strip()

        if len(body) < min_len:
            logger.warning("Too little text extracted from %s (%d chars)", pdf_path.name, len(body))
            return ""
        return body

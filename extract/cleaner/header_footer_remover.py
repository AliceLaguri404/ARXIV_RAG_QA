# extract/cleaner/header_footer_remover.py
import re
from collections import Counter

def remove_headers_footers(text: str, repetition_threshold: int = 3) -> str:
    """
    Remove repeating headers/footers and page numbers.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = Counter(lines)
    common = {l for l, c in counts.items() if c >= repetition_threshold}

    cleaned = [l for l in lines if l not in common]
    cleaned = [re.sub(r"^\d+$", "", l) for l in cleaned]
    cleaned = [l for l in cleaned if l.strip()]
    return "\n".join(cleaned)

# # scripts/phase3_run.py
# import json
# import logging
# from pathlib import Path
# from chunker.semantic_chunker import create_chunk_objects
# from embeddings.encoder import Embedder
# from vector_store.chromadb_client import ChromaDBClient
# import os
# os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# TEXT_DIR = Path("data/processed/texts")
# CHUNK_DIR = Path("data/processed/chunks")
# CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# def process_topic(topic_dir: Path, embedder, db):
#     topic = topic_dir.name
#     logger.info("Processing topic: %s", topic)
#     out_dir = CHUNK_DIR / topic
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for txt_file in sorted(topic_dir.glob("*.txt")):
#         doc_id = txt_file.stem
#         logger.info("Chunking %s", txt_file.name)
#         text = txt_file.read_text(encoding="utf-8")
#         chunk_objs = create_chunk_objects(doc_id, topic, text)

#         if not chunk_objs:
#             logger.warning("No chunks produced for %s", txt_file)
#             continue

#         embeddings = embedder.encode([c["text"] for c in chunk_objs])
#         db.add_chunks(chunk_objs, embeddings)

#         out_path = out_dir / f"{doc_id}_chunks.json"
#         with open(out_path, "w", encoding="utf-8") as f:
#             json.dump(chunk_objs, f, indent=2, ensure_ascii=False)

#         logger.info("✅ Saved %d chunks for %s", len(chunk_objs), doc_id)

# def main():
#     logger.info("Starting Phase 3: Chunk + Embed")
#     embedder = Embedder()
#     db = ChromaDBClient()

#     for topic_dir in TEXT_DIR.iterdir():
#         if topic_dir.is_dir():
#             process_topic(topic_dir, embedder, db)

#     logger.info("Phase 3 complete. Total embeddings in DB: %d", db.count())

# if __name__ == "__main__":
#     main()

# scripts/phase3_run.py
import json
import logging
from pathlib import Path
from src.chunker.semantic_chunker import create_chunk_objects
from src.embeddings.encoder import Embedder
from src.vector_store.chromadb_client import ChromaDBClient
import os

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEXT_DIR = Path("src/data/processed/texts")
CHUNK_DIR = Path("src/data/processed/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

def process_topic(topic_dir: Path, embedder, db):
    topic = topic_dir.name
    logger.info("Processing topic: %s", topic)
    out_dir = CHUNK_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(topic_dir.glob("*.txt")):
        doc_id = txt_file.stem
        logger.info("Chunking %s", txt_file.name)
        text = txt_file.read_text(encoding="utf-8")
        chunk_objs = create_chunk_objects(doc_id, topic, text)

        if not chunk_objs:
            logger.warning("No chunks produced for %s", txt_file)
            continue

        embeddings = embedder.encode([c["text"] for c in chunk_objs])
        db.add_chunks(chunk_objs, embeddings)

        out_path = out_dir / f"{doc_id}_chunks.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk_objs, f, indent=2, ensure_ascii=False)

        logger.info("✅ Saved %d chunks for %s", len(chunk_objs), doc_id)

def main():
    logger.info("Starting Phase 3: Chunk + Embed")
    embedder = Embedder()
    db = ChromaDBClient()

    for topic_dir in TEXT_DIR.iterdir():
        if topic_dir.is_dir():
            process_topic(topic_dir, embedder, db)

    logger.info("Phase 3 complete. Total embeddings in DB: %d", db.count())

if __name__ == "__main__":
    main()

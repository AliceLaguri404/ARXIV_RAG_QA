# embeddings/encoder.py
# from sentence_transformers import SentenceTransformer
# import numpy as np

# class Embedder:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)

#     def encode(self, texts):
#         emb = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
#         return np.array(emb)

# embeddings/encoder.py
from typing import List, Iterable, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    _HAS_SBT = True
except Exception:
    _HAS_SBT = False

class Embedder:
    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = None,
        batch_size: int = 64,
        normalize: bool = True,
    ):
        if not _HAS_SBT:
            raise ImportError("sentence-transformers and torch are required. pip install sentence-transformers torch")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        logger.info("Loading embedder %s on %s", model_name, device)
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Union[List[str], Iterable[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        all_embs = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            emb = self.model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            all_embs.append(emb)
        embs = np.vstack(all_embs).astype(np.float32)
        if self.normalize:
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embs = embs / norms
        return embs

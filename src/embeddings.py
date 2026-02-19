from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(embeddings, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(embedding, dtype="float32")

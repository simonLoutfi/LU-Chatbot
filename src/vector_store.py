from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np


class VectorStore:
    def __init__(self, index, records: List[dict], embedding_dim: int):
        self.index = index
        self.records = records
        self.embedding_dim = embedding_dim

    @classmethod
    def build(cls, embeddings: np.ndarray, records: List[dict]) -> "VectorStore":
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)
        return cls(index=index, records=records, embedding_dim=embedding_dim)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        index_path = index_dir / "index.faiss"
        chunks_path = index_dir / "chunks.json"
        meta_path = index_dir / "meta.json"

        index = faiss.read_index(str(index_path))
        records = json.loads(chunks_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(index=index, records=records, embedding_dim=meta["embedding_dim"])

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        (index_dir / "chunks.json").write_text(
            json.dumps(self.records, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        meta = {"embedding_dim": self.embedding_dim, "count": len(self.records)}
        (index_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[dict]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            record = self.records[index]
            results.append({"score": float(score), **record})
        return results

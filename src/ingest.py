from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from . import config
from .chunking import chunk_text
from .embeddings import EmbeddingModel
from .vector_store import VectorStore


def load_documents(docs_dir: Path) -> List[Dict]:
    documents = []
    for path in sorted(docs_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        document_id = path.stem
        last_updated = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        documents.append(
            {
                "document_id": document_id,
                "source_file": str(path),
                "last_updated": last_updated,
                "text": text,
            }
        )
    return documents


def build_records(documents: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    records = []
    for document in documents:
        chunks = chunk_text(document["text"], chunk_size, chunk_overlap)
        for chunk in chunks:
            chunk_id = f"{document['document_id']}:{chunk['chunk_index']}"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": {
                        "document_id": document["document_id"],
                        "source_file": document["source_file"],
                        "section_title": chunk.get("section_title"),
                        "last_updated": document["last_updated"],
                        "chunk_index": chunk["chunk_index"],
                        "token_count": chunk["token_count"],
                    },
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the admissions vector index.")
    parser.add_argument("--docs_dir", type=Path, default=config.DOCS_DIR)
    parser.add_argument("--out_dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--chunk_size", type=int, default=config.CHUNK_SIZE_TOKENS)
    parser.add_argument(
        "--chunk_overlap", type=int, default=config.CHUNK_OVERLAP_TOKENS
    )
    parser.add_argument("--embed_model", type=str, default=config.EMBEDDING_MODEL)
    args = parser.parse_args()

    documents = load_documents(args.docs_dir)
    if not documents:
        raise SystemExit(f"No .txt documents found in {args.docs_dir}")

    records = build_records(documents, args.chunk_size, args.chunk_overlap)
    texts = [record["text"] for record in records]

    model = EmbeddingModel(args.embed_model)
    embeddings = model.embed_texts(texts)

    vector_store = VectorStore.build(embeddings, records)
    vector_store.save(args.out_dir)

    print(f"Indexed {len(records)} chunks into {args.out_dir}")


if __name__ == "__main__":
    main()

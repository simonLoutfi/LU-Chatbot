from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .embeddings import EmbeddingModel
from .llm import LLMUnavailableError, generate_answer
from .vector_store import VectorStore


def format_retrieval_results(results):
    lines = []
    for idx, result in enumerate(results, start=1):
        meta = result["metadata"]
        title = meta.get("section_title") or meta.get("document_id")
        lines.append(f"[{idx}] score={result['score']:.3f} {title}")
        lines.append(result["text"])
        lines.append("")
    return "\n".join(lines).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the admissions index.")
    parser.add_argument("--index_dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--threshold", type=float, default=config.SIMILARITY_THRESHOLD
    )
    parser.add_argument("--embed_model", type=str, default=config.EMBEDDING_MODEL)
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        query = input("Ask a question: ").strip()
    if not query:
        raise SystemExit("Query is required.")

    store = VectorStore.load(args.index_dir)
    model = EmbeddingModel(args.embed_model)
    query_embedding = model.embed_query(query)
    results = store.search(query_embedding, args.top_k)
    results = [r for r in results if r["score"] >= args.threshold]

    if not results:
        print("No relevant admissions content found for this query.")
        return

    if args.no_llm:
        print(format_retrieval_results(results))
        return

    try:
        answer = generate_answer(query, results)
    except LLMUnavailableError as exc:
        print(f"LLM unavailable: {exc}")
        print(format_retrieval_results(results))
        return

    if not answer:
        print("LLM returned an empty response.")
        print(format_retrieval_results(results))
        return

    print(answer)


if __name__ == "__main__":
    main()

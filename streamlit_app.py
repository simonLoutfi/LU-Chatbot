from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config
from src.embeddings import EmbeddingModel
from src.llm import LLMUnavailableError, generate_answer
from src.token_utils import take_last_tokens
from src.vector_store import VectorStore

INDEX_FILES = ("index.faiss", "chunks.json", "meta.json")


def index_status(index_dir: Path) -> list[str]:
    missing = []
    for name in INDEX_FILES:
        if not (index_dir / name).exists():
            missing.append(name)
    return missing


@st.cache_resource(show_spinner="Loading admissions index...")
def load_store(index_dir: str) -> VectorStore:
    return VectorStore.load(Path(index_dir))


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name)


def retrieve_chunks(
    question: str,
    index_dir: str,
    top_k: int,
    threshold: float,
    embed_model: str,
    history: Optional[list[dict]] = None,
) -> list[dict]:
    store = load_store(index_dir)
    model = load_model(embed_model)
    search_query = build_retrieval_query(question, history)
    query_embedding = model.embed_query(search_query)
    results = store.search(query_embedding, top_k)
    return [r for r in results if r["score"] >= threshold]


def render_sources(results: list[dict]) -> None:
    for idx, result in enumerate(results, start=1):
        meta = result.get("metadata", {})
        title = meta.get("section_title") or meta.get("document_id") or "Admissions"
        label = f"[{idx}] {title} (score {result['score']:.3f})"
        with st.expander(label, expanded=False):
            st.write(result.get("text", ""))
            if meta:
                st.caption(
                    " | ".join(
                        f"{key}: {value}" for key, value in meta.items() if value
                    )
                )


def answer_question(
    question: str,
    index_dir: str,
    top_k: int,
    threshold: float,
    embed_model: str,
    use_llm: bool,
    history: Optional[list[dict]] = None,
) -> tuple[str, list[dict]]:
    results = retrieve_chunks(
        question, index_dir, top_k, threshold, embed_model, history=history
    )
    if not results:
        return "No relevant admissions content found for this query.", []

    if not use_llm:
        return "Here are the most relevant admissions passages.", results

    try:
        answer = generate_answer(question, results, history=history)
    except LLMUnavailableError as exc:
        return f"LLM unavailable: {exc}\n\nShowing retrieved passages instead.", results

    if not answer:
        return "LLM returned an empty response. Showing retrieved passages instead.", results

    return answer, results


def build_retrieval_query(question: str, history: Optional[list[dict]]) -> str:
    if not history:
        return question

    user_messages = [
        message.get("content", "")
        for message in history
        if message.get("role") == "user" and message.get("content")
    ]
    if not user_messages:
        return question

    if config.RETRIEVAL_HISTORY_MESSAGES > 0:
        user_messages = user_messages[-config.RETRIEVAL_HISTORY_MESSAGES:]

    combined = " ".join(user_messages + [question]).strip()
    if not combined:
        return question

    if config.RETRIEVAL_HISTORY_TOKENS > 0:
        combined = take_last_tokens(combined, config.RETRIEVAL_HISTORY_TOKENS)

    return combined


st.set_page_config(page_title="LU Admissions Chatbot", layout="centered")

st.title("LU Admissions Chatbot")
st.write(
    "Ask questions about Lebanese University admissions. Answers are grounded in the indexed admissions documents."
)

with st.sidebar:
    st.subheader("Settings")
    index_dir = st.text_input("Index directory", value=str(config.INDEX_DIR))
    top_k = st.slider("Top K", min_value=1, max_value=10, value=config.TOP_K)
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(config.SIMILARITY_THRESHOLD),
        step=0.01,
    )
    use_llm = st.toggle("Use LLM", value=True)
    show_sources = st.toggle("Show sources", value=True)

    col_left, col_right = st.columns(2)
    with col_left:
        if st.button("Reload index"):
            load_store.clear()
    with col_right:
        if st.button("Clear chat"):
            st.session_state.messages = []

    st.caption(f"Embedding model: {config.EMBEDDING_MODEL}")
    st.caption(f"LLM model: {config.OLLAMA_MODEL}")

index_dir_path = Path(index_dir)
missing_files = index_status(index_dir_path)
if missing_files:
    st.warning(
        "Index files are missing: "
        + ", ".join(missing_files)
        + ". Build the index first with `python -m src.ingest --docs_dir . --out_dir ./index`."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources") and show_sources:
            render_sources(message["sources"])

question = st.chat_input("Ask about admissions requirements, deadlines, or documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    history = st.session_state.messages[:-1]

    with st.chat_message("assistant"):
        if missing_files:
            response_text = (
                "The admissions index is not available yet. "
                "Run `python -m src.ingest --docs_dir . --out_dir ./index` first."
            )
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            with st.spinner("Searching admissions content..."):
                try:
                    response_text, sources = answer_question(
                        question,
                        index_dir=str(index_dir_path),
                        top_k=top_k,
                        threshold=threshold,
                        embed_model=config.EMBEDDING_MODEL,
                        use_llm=use_llm,
                        history=history,
                    )
                except Exception as exc:
                    response_text = f"Something went wrong while querying the index: {exc}"
                    sources = []

            st.write(response_text)
            if sources and show_sources:
                render_sources(sources)

            st.session_state.messages.append(
                {"role": "assistant", "content": response_text, "sources": sources}
            )

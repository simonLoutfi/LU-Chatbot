# LU Admissions Chatbot (RAG)

This project indexes Lebanese university admissions text documents and answers
queries using retrieval-augmented generation grounded in the official content.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Build the index

```powershell
python -m src.ingest --docs_dir "." --out_dir "./index"
```

## Query the index

```powershell
python -m src.query --index_dir "./index" --query "What are the admission requirements for engineering?"
```

If you want retrieval-only output without calling an LLM:

```powershell
python -m src.query --index_dir "./index" --query "What documents are required?" --no_llm
```

## Streamlit interface

Build the index first, then launch the UI:

```powershell
streamlit run streamlit_app.py
```

## LLM configuration

The default implementation uses a local Ollama model. Make sure the Ollama
server is running and the model is pulled (for example,
`ollama pull deepseek-coder:6.7b`). Environment variables are loaded from `.env` or
`src/.env` automatically if present. You can override the model with
`OLLAMA_MODEL`. The integration uses the `ollama` Python package.

## Configuration

Environment variables (optional):

- `LU_DOCS_DIR`: Directory containing `.txt` admissions documents.
- `LU_INDEX_DIR`: Output directory for the FAISS index and metadata.
- `LU_CHUNK_SIZE`: Chunk size in tokens (default 384).
- `LU_CHUNK_OVERLAP`: Chunk overlap in tokens (default 80).
- `LU_TOP_K`: Retrieval top-k (default 4).
- `LU_SIM_THRESHOLD`: Similarity cutoff (default 0.2).
- `LU_MAX_CONTEXT_TOKENS`: Max tokens sent to the LLM (default 1200).
- `LU_MAX_CHUNK_TOKENS`: Max tokens per chunk sent to the LLM (default 220).
- `LU_MAX_CONTEXT_CHUNKS`: Max chunks sent to the LLM (default 3).
- `LU_MAX_HISTORY_MESSAGES`: Max chat messages to keep in LLM memory (default 6).
- `LU_MAX_HISTORY_TOKENS`: Max tokens from chat history sent to the LLM (default 400).
- `LU_RETRIEVAL_HISTORY_MESSAGES`: Max user messages to append to retrieval queries (default 2).
- `LU_RETRIEVAL_HISTORY_TOKENS`: Max tokens from chat history for retrieval queries (default 120).
- `LU_EMBED_MODEL`: Sentence transformer model name.
- `OLLAMA_MODEL`: Ollama model name (default `deepseek-coder:6.7b`).
- `OLLAMA_API_KEY`: Required for Ollama cloud models (if you see 401 errors).
- `OLLAMA_HOST`: Optional Ollama server URL (default `http://localhost:11434`). For
  cloud, set this to `https://ollama.com`.

## Files

- `src/ingest.py`: Document ingestion, chunking, embeddings, index build.
- `src/query.py`: Query embedding, retrieval, and answer generation.

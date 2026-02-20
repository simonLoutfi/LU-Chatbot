from __future__ import annotations

import os
from pathlib import Path

# Project root directory (one level above src/).
ROOT_DIR = Path(__file__).resolve().parents[1]

def _load_env_file(path: Path) -> None:
    # Load key=value pairs from a .env file into the environment.
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load environment overrides (root first, then src/).
_load_env_file(ROOT_DIR / ".env")
_load_env_file(ROOT_DIR / "src" / ".env")

# Directory containing source documents to index.
DOCS_DIR = Path(os.environ.get("LU_DOCS_DIR", ROOT_DIR))
# Directory where the vector index is stored.
INDEX_DIR = Path(os.environ.get("LU_INDEX_DIR", ROOT_DIR / "index"))

# Target chunk size for splitting documents (in tokens).
CHUNK_SIZE_TOKENS = int(os.environ.get("LU_CHUNK_SIZE", "384"))
# Token overlap between adjacent chunks.
CHUNK_OVERLAP_TOKENS = int(os.environ.get("LU_CHUNK_OVERLAP", "80"))

# Embedding model name or path.
EMBEDDING_MODEL = os.environ.get(
    "LU_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Number of top matches to retrieve.
TOP_K = int(os.environ.get("LU_TOP_K", "4"))
# Minimum similarity score to accept a match.
SIMILARITY_THRESHOLD = float(os.environ.get("LU_SIM_THRESHOLD", "0.2"))

# Max tokens allowed in assembled context for the model.
MAX_CONTEXT_TOKENS = int(os.environ.get("LU_MAX_CONTEXT_TOKENS", "1200"))
# Hard cap on any single chunk size used in context.
MAX_CHUNK_TOKENS = int(os.environ.get("LU_MAX_CHUNK_TOKENS", "220"))
# Max number of chunks to include in context.
MAX_CONTEXT_CHUNKS = int(os.environ.get("LU_MAX_CONTEXT_CHUNKS", "3"))
# Max number of chat messages to keep in memory.
MAX_HISTORY_MESSAGES = int(os.environ.get("LU_MAX_HISTORY_MESSAGES", "6"))
# Max tokens from chat history to include.
MAX_HISTORY_TOKENS = int(os.environ.get("LU_MAX_HISTORY_TOKENS", "400"))
# Messages from history considered for retrieval.
RETRIEVAL_HISTORY_MESSAGES = int(os.environ.get("LU_RETRIEVAL_HISTORY_MESSAGES", "2"))
# Tokens from history considered for retrieval.
RETRIEVAL_HISTORY_TOKENS = int(os.environ.get("LU_RETRIEVAL_HISTORY_TOKENS", "120"))

# LLM model name served by Ollama.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-coder:6.7b")

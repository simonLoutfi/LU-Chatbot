from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def _load_env_file(path: Path) -> None:
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


_load_env_file(ROOT_DIR / ".env")
_load_env_file(ROOT_DIR / "src" / ".env")

DOCS_DIR = Path(os.environ.get("LU_DOCS_DIR", ROOT_DIR))
INDEX_DIR = Path(os.environ.get("LU_INDEX_DIR", ROOT_DIR / "index"))

CHUNK_SIZE_TOKENS = int(os.environ.get("LU_CHUNK_SIZE", "384"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("LU_CHUNK_OVERLAP", "80"))

EMBEDDING_MODEL = os.environ.get(
    "LU_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = int(os.environ.get("LU_TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.environ.get("LU_SIM_THRESHOLD", "0.2"))

MAX_CONTEXT_TOKENS = int(os.environ.get("LU_MAX_CONTEXT_TOKENS", "1200"))
MAX_CHUNK_TOKENS = int(os.environ.get("LU_MAX_CHUNK_TOKENS", "220"))
MAX_CONTEXT_CHUNKS = int(os.environ.get("LU_MAX_CONTEXT_CHUNKS", "3"))
MAX_HISTORY_MESSAGES = int(os.environ.get("LU_MAX_HISTORY_MESSAGES", "6"))
MAX_HISTORY_TOKENS = int(os.environ.get("LU_MAX_HISTORY_TOKENS", "400"))
RETRIEVAL_HISTORY_MESSAGES = int(os.environ.get("LU_RETRIEVAL_HISTORY_MESSAGES", "2"))
RETRIEVAL_HISTORY_TOKENS = int(os.environ.get("LU_RETRIEVAL_HISTORY_TOKENS", "120"))

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-coder:6.7b")

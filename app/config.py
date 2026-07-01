from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent.parent


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _path_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    if value.is_absolute():
        return value
    return ROOT_DIR / value


class Settings:
    def __init__(self) -> None:
        self.root_dir = ROOT_DIR
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.catalog_path = _path_env("CATALOG_PATH", "data/catalog.json")

        # Hybrid retriever paths (populated by scripts/build_index.py)
        self.faiss_index_path = _path_env("FAISS_INDEX_PATH", "data/faiss.index")
        self.bm25_path = _path_env("BM25_PATH", "data/bm25.pkl")
        self.embed_model = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

        self.max_conversation_turns = int(os.getenv("MAX_CONVERSATION_TURNS", "8"))
        self.top_k_candidates = int(os.getenv("TOP_K_CANDIDATES", "10"))

        # LLM settings
        self.llm_enabled = _bool_env("LLM_ENABLED", True)
        self.llm_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        self.llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "5"))

        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()


settings = Settings()

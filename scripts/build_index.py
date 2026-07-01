"""
scripts/build_index.py
One-time script: embeds catalog.json into FAISS + BM25 indexes.

Usage:
    python scripts/build_index.py

Outputs:
    data/faiss.index  — FAISS IndexFlatIP (cosine on normalised vecs)
    data/bm25.pkl     — BM25Okapi serialised with pickle
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

# ── Allow running from repo root ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

CATALOG_PATH = ROOT / "data" / "catalog.json"
INDEX_PATH   = ROOT / "data" / "faiss.index"
BM25_PATH    = ROOT / "data" / "bm25.pkl"
EMBED_MODEL  = "all-MiniLM-L6-v2"

TEST_TYPE_LABELS = {
    "A": "ability aptitude cognitive reasoning",
    "B": "biodata situational judgement",
    "C": "competency behavioural",
    "D": "development 360 feedback",
    "E": "exercise simulation assessment center",
    "K": "knowledge skills technical",
    "P": "personality behavior questionnaire",
    "S": "simulation interactive",
}


def make_document(item: dict) -> str:
    """Create rich text document for embedding/BM25."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keywords", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(TEST_TYPE_LABELS.get(t, "") for t in item.get("test_types", [])),
    ]
    return " ".join(p for p in parts if p).strip()


def build(catalog_path: Path = CATALOG_PATH,
          index_path: Path = INDEX_PATH,
          bm25_path: Path  = BM25_PATH,
          model_name: str  = EMBED_MODEL) -> None:

    if not catalog_path.exists():
        print(f"[ERROR] Catalog not found: {catalog_path}", file=sys.stderr)
        sys.exit(1)

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = [item for item in catalog if item.get("name") and item.get("url")]
    print(f"Loaded {len(catalog)} catalog items from {catalog_path}")

    docs = [make_document(item) for item in catalog]

    # ── FAISS ────────────────────────────────────────────────────────────────
    print(f"Embedding with '{model_name}'…")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64,
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine on unit vecs
    index.add(embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    print(f"FAISS index saved -> {index_path}  ({index.ntotal} vectors, dim={dim})")

    # ── BM25 ─────────────────────────────────────────────────────────────────
    tokenized = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized)
    bm25_path.write_bytes(pickle.dumps(bm25))
    print(f"BM25 index  saved -> {bm25_path}")

    print("\nDone. Run the server normally — hybrid retrieval will be used automatically.")


if __name__ == "__main__":
    build()

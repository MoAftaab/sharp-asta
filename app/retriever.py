"""
app/retriever.py
Hybrid FAISS + BM25 retriever with Reciprocal Rank Fusion (RRF).
Falls back gracefully to the lexical scorer when indexes are not available.

Build indexes first:
    python scripts/build_index.py
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from typing import Any

from app.catalog import TEST_TYPE_LABELS, load_catalog, primary_test_type
from app.config import settings
from app.models import ChatMessage

logger = logging.getLogger(__name__)

# ── Hybrid retriever state (populated by warm_up_hybrid()) ───────────────────
_hybrid: dict[str, Any] = {
    "available": False,
    "faiss": None,
    "bm25": None,
    "model": None,
}

# ── Tokenisation helpers ─────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from",
    "hire", "hiring", "i", "in", "is", "it", "level", "levels",
    "mid", "need", "of", "on", "or", "our", "that", "the",
    "to", "with", "who", "work", "works",
    "hello", "hi", "hey", "dear", "deaar", "dearr", "baby", "babt",
    "u", "you", "wbu", "sup", "there", "how", "whom", "hwo",
}

VAGUE_TERMS = {"assessment", "test", "tests", "something", "help", "anything", "best", "recommend"}

QUERY_EXPANSIONS: dict[str, set[str]] = {
    "dev":          {"developer", "software", "programming"},
    "developer":    {"software", "programming", "technical"},
    "engineer":     {"technical", "software", "programming"},
    "frontend":     {"front-end", "javascript", "react", "angular", "ui"},
    "front":        {"frontend", "javascript", "react", "angular"},
    "backend":      {"back-end", "java", "python", "sql", "api"},
    "stakeholder":  {"communication", "business", "collaboration"},
    "stakeholders": {"stakeholder", "communication", "business", "collaboration"},
    "lead":         {"leadership", "manager"},
    "manager":      {"leadership", "management", "situational"},
    "data":         {"analytics", "analysis", "sql", "excel"},
    "analyst":      {"analysis", "analytics", "sql", "excel"},
    "sales":        {"selling", "revenue", "customer"},
    "support":      {"customer", "contact", "service"},
    "call":         {"contact", "customer", "service"},
    "finance":      {"accounting", "financial", "numerical"},
    "graduate":     {"entry", "early", "potential"},
    "entry":        {"graduate", "junior"},
    "devops":       {"docker", "kubernetes", "aws", "cloud", "infrastructure"},
    "cloud":        {"aws", "infrastructure", "devops"},
}

TYPE_INTENTS: dict[str, str] = {
    "personality":  "P",
    "behavior":     "P",
    "behaviour":    "P",
    "motivation":   "P",
    "cognitive":    "A",
    "ability":      "A",
    "aptitude":     "A",
    "reasoning":    "A",
    "technical":    "K",
    "coding":       "K",
    "programming":  "K",
    "knowledge":    "K",
    "skills":       "K",
    "simulation":   "S",
    "simulations":  "S",
    "interactive":  "S",
    "situational":  "B",
    "judgment":     "B",
    "judgement":    "B",
    "competency":   "C",
    "competencies": "C",
    "360":          "D",
    "feedback":     "D",
}

SENIORITY_TERMS: dict[str, str] = {
    "entry":        "Entry",
    "graduate":     "Entry",
    "junior":       "Entry",
    "mid":          "Professional",
    "middle":       "Professional",
    "professional": "Professional",
    "senior":       "Manager",
    "lead":         "Manager",
    "manager":      "Manager",
    "executive":    "Executive",
    "director":     "Executive",
}

ALIASES: dict[str, str] = {
    "opq":        "OPQ32r",
    "opq32":      "OPQ32r",
    "g+":         "Verify G+",
    "g plus":     "Verify G+",
    "gsa":        "Verify G+",
    "java":       "Java 8 (New)",
    "python":     "Python (New)",
    "javascript": "JavaScript (New)",
    "react":      "React JS",
    "sql":        "SQL (New)",
}

_DEFAULT_NAMES = [
    "Verify G+", "OPQ32r", "Verify Numerical Reasoning", "Verify Verbal Reasoning",
]


# ── Public API: warm up hybrid at startup ─────────────────────────────────────

def warm_up_hybrid() -> bool:
    """
    Called from app/main.py lifespan on startup.
    Loads FAISS index, BM25 pickle, and SentenceTransformer model.
    Returns True if hybrid is now available, False otherwise.
    """
    try:
        import pickle

        import faiss
        import numpy as np  # noqa: F401  (imported here to check availability)
        from sentence_transformers import SentenceTransformer

        idx_path = settings.faiss_index_path
        bm25_path = settings.bm25_path

        if not (idx_path.exists() and bm25_path.exists()):
            logger.info(
                "Hybrid indexes not found (%s, %s). "
                "Run 'python scripts/build_index.py' to enable FAISS+BM25 retrieval.",
                idx_path, bm25_path,
            )
            return False

        _hybrid["faiss"] = faiss.read_index(str(idx_path))
        _hybrid["bm25"] = pickle.loads(bm25_path.read_bytes())
        _hybrid["model"] = SentenceTransformer(settings.embed_model)
        _hybrid["available"] = True
        logger.info(
            "Hybrid FAISS+BM25 retriever ready — %d vectors, model=%s",
            _hybrid["faiss"].ntotal, settings.embed_model,
        )
        return True

    except ImportError as exc:
        logger.warning(
            "Hybrid retriever unavailable (missing dependency: %s). "
            "Using lexical fallback.", exc,
        )
    except Exception as exc:
        logger.warning("Hybrid retriever failed to load: %s. Using lexical fallback.", exc)

    return False


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def expanded_tokens(text: str) -> set[str]:
    tokens = set(tokenize(text))
    # Naive stemming: strip trailing 's'
    for token in list(tokens):
        if token.endswith("s") and len(token) > 4:
            tokens.add(token[:-1])
    # Domain expansions
    for token in list(tokens):
        tokens.update(QUERY_EXPANSIONS.get(token, set()))
    # Compound detection
    if "front" in tokens and "end" in tokens:
        tokens.update({"frontend", "javascript", "react"})
    if "back" in tokens and "end" in tokens:
        tokens.update({"backend", "java", "python", "sql"})
    return tokens


def _duration_limit(text: str) -> int | None:
    patterns = [
        r"(?:under|below|less than|no more than|within|max(?:imum)?|<=)\s*(\d{1,3})\s*(?:min|mins|minutes)?",
        r"(\d{1,3})\s*(?:min|mins|minutes)\s*(?:or less|max)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return None


# ── Slot extraction ───────────────────────────────────────────────────────────

def extract_slots(messages: list[ChatMessage]) -> dict[str, Any]:
    user_text = " ".join(m.content for m in messages if m.role == "user")
    latest_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    lowered = user_text.lower()
    tokens = expanded_tokens(user_text)

    requested_types = {code for term, code in TYPE_INTENTS.items() if term in lowered}
    seniority = {label for term, label in SENIORITY_TERMS.items() if term in lowered}

    named: list[str] = []
    for item in load_catalog():
        name = item["name"].lower()
        if name in lowered:
            named.append(item["name"])
    for alias, canonical in ALIASES.items():
        if alias in lowered and canonical not in named:
            named.append(canonical)

    return {
        "query":             user_text,
        "latest_user":       latest_user,
        "tokens":            tokens,
        "requested_types":   requested_types,
        "seniority":         seniority,
        "max_duration":      _duration_limit(user_text),
        "remote_required":   "remote" in tokens,
        "adaptive_required": "adaptive" in tokens or "irt" in tokens,
        "named_assessments": named,
    }


def has_enough_context(slots: dict[str, Any]) -> bool:
    if slots["named_assessments"]:
        return True
    useful = slots["tokens"] - VAGUE_TERMS
    if slots["requested_types"] and len(useful) >= 2:
        return True
    if useful & {
        "java", "python", "javascript", "react", "angular", "sql",
        "data", "analyst", "developer", "engineer", "manager", "sales",
        "support", "customer", "finance", "graduate", "administrative",
        "leadership", "docker", "kubernetes", "aws", "devops",
    }:
        return True
    return len(useful) >= 4


# ── Indexed catalog (lexical scorer) ─────────────────────────────────────────

@lru_cache(maxsize=1)
def _indexed_catalog() -> list[dict[str, Any]]:
    indexed = []
    for idx, item in enumerate(load_catalog()):
        fields = " ".join([
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keywords", [])),
            " ".join(item.get("job_levels", [])),
            " ".join(TEST_TYPE_LABELS.get(t, "") for t in item.get("test_types", [])),
        ])
        indexed.append({
            "idx":         idx,
            "item":        item,
            "text":        fields.lower(),
            "name_tokens": set(tokenize(item.get("name", ""))),
            "keywords":    set(tokenize(" ".join(item.get("keywords", [])))),
            "all_tokens":  set(tokenize(fields)),
        })
    return indexed


def _score_item(indexed: dict[str, Any], slots: dict[str, Any]) -> float:
    item = indexed["item"]
    query = slots["query"].lower()
    latest = slots["latest_user"].lower()
    tokens = slots["tokens"]
    item_types = set(item.get("test_types", []))

    score = 0.0
    name_lower = item["name"].lower()

    # Exact name match
    if name_lower in query:
        score += 50

    # Alias match
    for alias, canonical in ALIASES.items():
        if alias in query and item["name"] == canonical:
            score += 22

    # Token overlap
    for token in tokens:
        if token in indexed["name_tokens"]:
            score += 6
        if token in indexed["keywords"]:
            score += 5
        if token in indexed["all_tokens"]:
            score += 1.5

    # Keyword phrase match
    for keyword in item.get("keywords", []):
        kw = keyword.lower()
        if len(kw) > 3 and kw in query:
            score += 4 + min(len(kw.split()), 3)

    # Requested type boost/penalty
    requested_types = slots["requested_types"]
    if requested_types:
        if item_types & requested_types:
            score += 13
        else:
            score -= 3

    # Duration filter
    max_duration = slots["max_duration"]
    duration = item.get("duration_minutes")
    if max_duration and duration:
        if duration <= max_duration:
            score += 4
        else:
            score -= min(18, math.log(duration - max_duration + 1) * 6)

    # Remote / adaptive flags
    if slots["remote_required"]:
        score += 2 if item.get("remote_testing") else -6
    if slots["adaptive_required"]:
        score += 3 if item.get("adaptive_irt") else -4

    # Seniority match
    if slots["seniority"]:
        levels = set(item.get("job_levels", []))
        if levels & slots["seniority"]:
            score += 3

    # Inline personality refinement signal
    if any(term in latest for term in ("add personality", "also personality", "personality tests")):
        score += 18 if "P" in item_types else -2

    # Stakeholder / communication signals → prefer P/C/B
    if any(term in query for term in ("stakeholder", "communication", "collaboration")):
        if item_types & {"P", "C", "B"}:
            score += 4
        if indexed["keywords"] & {"business", "stakeholder", "communication"}:
            score += 4

    # Job description keyword density
    if "job description" in query or "jd" in tokens:
        score += len(tokens & indexed["all_tokens"]) * 0.5

    # Suppress development-type items unless explicitly requested
    if primary_test_type(item) == "D" and "development" not in tokens and "360" not in tokens:
        score -= 2

    return score


# ── Hybrid retrieval (FAISS + BM25 + RRF) ───────────────────────────────────

def _rrf_fuse(bm25_ranks: list[int], faiss_ranks: list[int], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(faiss_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def _hybrid_candidates(query: str, top_k: int) -> list[dict[str, Any]]:
    """Returns catalog items ranked by FAISS+BM25 RRF fusion."""
    import numpy as np

    catalog = load_catalog()
    faiss_index = _hybrid["faiss"]
    bm25 = _hybrid["bm25"]
    model = _hybrid["model"]
    n = len(catalog)
    k = min(top_k * 2, n)

    # FAISS semantic search
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    _, faiss_idxs = faiss_index.search(q_emb, k)
    faiss_ranks = [int(i) for i in faiss_idxs[0] if 0 <= i < n]

    # BM25 keyword search
    tokens = re.sub(r"[^\w\s]", " ", query.lower()).split()
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranks = list(np.argsort(-bm25_scores)[:k])

    # RRF fusion
    fused = _rrf_fuse(bm25_ranks, faiss_ranks)
    return [catalog[idx] for idx, _ in fused[:top_k] if 0 <= idx < n]


# ── Main public retrieve functions ────────────────────────────────────────────

def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Convenience wrapper: create slots from a single query string."""
    top_k = top_k or settings.top_k_candidates
    fake_msg = ChatMessage(role="user", content=query)
    slots = extract_slots([fake_msg])
    return retrieve_from_slots(slots, top_k=top_k)


def retrieve_from_slots(slots: dict[str, Any], top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Main retrieval entry-point.
    If hybrid (FAISS+BM25) is loaded, uses RRF candidates + lexical re-scorer.
    Otherwise falls back to pure lexical scoring.
    """
    top_k = top_k or settings.top_k_candidates
    indexed = _indexed_catalog()

    scored: list[tuple[float, int, dict[str, Any]]] = []
    seen_idxs: set[int] = set()

    if _hybrid["available"]:
        # ── Hybrid path ───────────────────────────────────────────────────────
        hybrid_items = _hybrid_candidates(slots["query"], top_k * 3)
        hybrid_urls = {item.get("url") for item in hybrid_items}

        # Build URL → indexed-item lookup
        url_to_idx_item: dict[str, tuple[int, dict[str, Any]]] = {
            idx_item["item"].get("url"): (i, idx_item)
            for i, idx_item in enumerate(indexed)
        }

        # Score hybrid candidates with lexical scorer
        for item in hybrid_items:
            url = item.get("url")
            if url and url in url_to_idx_item:
                i, idx_item = url_to_idx_item[url]
                if i not in seen_idxs:
                    score = _score_item(idx_item, slots)
                    scored.append((score, idx_item["idx"], idx_item["item"]))
                    seen_idxs.add(i)

        # Also include strong lexical matches not surfaced by hybrid
        for i, idx_item in enumerate(indexed):
            if i not in seen_idxs:
                url = idx_item["item"].get("url")
                if url not in hybrid_urls:
                    score = _score_item(idx_item, slots)
                    if score >= 15:
                        scored.append((score, idx_item["idx"], idx_item["item"]))
                        seen_idxs.add(i)

    else:
        # ── Pure lexical path ─────────────────────────────────────────────────
        for idx_item in indexed:
            score = _score_item(idx_item, slots)
            if score > 0:
                scored.append((score, idx_item["idx"], idx_item["item"]))

    # No results → return safe defaults
    if not scored:
        by_name = {item["name"]: item for item in load_catalog()}
        return [by_name[name] for name in _DEFAULT_NAMES if name in by_name][:top_k]

    scored.sort(key=lambda row: (-row[0], row[1]))
    ranked = [item for _, _, item in scored]

    # Type forcing: guarantee at least one item per requested type in top-K
    requested_types = slots["requested_types"]
    if requested_types:
        for code in requested_types:
            if not any(code in item.get("test_types", []) for item in ranked[:top_k]):
                first = next((item for item in ranked if code in item.get("test_types", [])), None)
                if first:
                    ranked = [first] + [item for item in ranked if item.get("url") != first.get("url")]

    return ranked[:top_k]

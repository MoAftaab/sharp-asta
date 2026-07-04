from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from app.config import settings
from app.models import Recommendation


TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# --- Live SHL catalog pages (verified reachable -- anti-404 guard) -----------
# Older per-product SHL URLs (e.g. /products/opq/, /products/verify/) now 404.
# We route every assessment to the closest *live* public SHL page by its primary
# test type. This keeps the "no hallucinated / broken links" PRD guardrail true
# while giving the user a page that actually opens.
CATALOG_ROOT = "https://www.shl.com/solutions/products/product-catalog/"

TEST_TYPE_URLS = {
    "A": "https://www.shl.com/solutions/products/assessments/cognitive-assessments/",
    "B": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/",
    "C": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/",
    "D": "https://www.shl.com/solutions/products/assessments/behavioral-assessments/",
    "E": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/",
    "K": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/",
    "P": "https://www.shl.com/solutions/products/assessments/personality-assessment/",
    "S": "https://www.shl.com/solutions/products/assessments/skills-and-simulations/",
}


def canonical_url(item: dict[str, Any]) -> str:
    """Map an assessment to a verified-live SHL page based on its primary type."""
    code = ""
    types = item.get("test_types") or []
    if isinstance(types, str):
        code = types
    elif types:
        code = str(types[0])
    return TEST_TYPE_URLS.get(code, CATALOG_ROOT)


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    raw = json.loads(settings.catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Catalog must be a JSON list.")
    items = [item for item in raw if item.get("name") and item.get("url")]
    # Normalise every URL to a live SHL page so recommendation links never 404.
    for item in items:
        item["url"] = canonical_url(item)
    return items


@lru_cache(maxsize=1)
def valid_urls() -> set[str]:
    return {item["url"] for item in load_catalog()}


@lru_cache(maxsize=1)
def catalog_by_name() -> dict[str, dict[str, Any]]:
    return {item["name"].lower(): item for item in load_catalog()}


def primary_test_type(item: dict[str, Any]) -> str:
    types = item.get("test_types") or []
    if isinstance(types, str):
        return types
    return str(types[0]) if types else ""


def to_recommendation(item: dict[str, Any]) -> Recommendation | None:
    if item.get("url") not in valid_urls():
        return None
    test_type = primary_test_type(item)
    if not test_type:
        return None
    return Recommendation(name=item["name"], url=item["url"], test_type=test_type)


def validate_recommendations(items: list[dict[str, Any]], limit: int = 10) -> list[Recommendation]:
    recs: list[Recommendation] = []
    seen_names: set[str] = set()
    for item in items:
        # De-duplicate by assessment NAME, not URL: many distinct assessments
        # share the same live category URL, so URL-dedup would silently collapse
        # a rich shortlist down to a single item.
        name_key = str(item.get("name", "")).strip().lower()
        if not name_key or name_key in seen_names:
            continue
        rec = to_recommendation(item)
        if rec is None:
            continue
        seen_names.add(name_key)
        recs.append(rec)
        if len(recs) >= limit:
            break
    return recs


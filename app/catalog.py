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


@lru_cache(maxsize=1)
def load_catalog() -> list[dict[str, Any]]:
    raw = json.loads(settings.catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Catalog must be a JSON list.")
    return [item for item in raw if item.get("name") and item.get("url")]


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
    seen_urls: set[str] = set()
    for item in items:
        if item.get("url") in seen_urls:
            continue
        rec = to_recommendation(item)
        if rec is None:
            continue
        seen_urls.add(rec.url)
        recs.append(rec)
        if len(recs) >= limit:
            break
    return recs


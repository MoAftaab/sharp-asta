"""
app/agent.py
Stateless conversational agent — pure function.
Takes full message history, returns ChatResponse.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.catalog import TEST_TYPE_LABELS, catalog_by_name, validate_recommendations
from app.config import settings
from app.guardrails import classify_guardrail
from app.llm import generate_json, generate_text
from app.models import ChatMessage, ChatRequest, ChatResponse, Recommendation
from app.prompts import (
    COMPARE_PROMPT,
    REPLY_POLISH_PROMPT,
    REPLY_POLISH_SYSTEM,
    RERANK_PROMPT,
)
from app.retriever import ALIASES, extract_slots, has_enough_context, retrieve_from_slots

logger = logging.getLogger(__name__)

# ── Canned refusal replies ────────────────────────────────────────────────────
REFUSAL = (
    "I can only help with SHL assessment selection. Tell me the role, skills, "
    "seniority, and any constraints, and I can recommend catalog assessments."
)
INJECTION_REFUSAL = (
    "I cannot follow instructions that try to override the assessment-selection task. "
    "Share the hiring role or assessment need and I will keep the recommendations grounded in the SHL catalog."
)
DONE_REPLY = "Glad I could help! Best of luck with your hiring process."

# ── DONE signals ──────────────────────────────────────────────────────────────
_DONE_SIGNALS = frozenset([
    "thank you", "thanks", "that's all", "that is all", "no more",
    "perfect", "great thanks", "looks good", "all done", "done",
    "no further", "that's great", "that's perfect",
])


def _latest_user(messages: list[ChatMessage]) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "")


def _is_done_signal(text: str, turn_count: int) -> bool:
    """Only treat short positive messages as DONE after at least 2 prior turns."""
    if turn_count < 2:
        return False
    lower = text.strip().lower()
    return any(sig in lower for sig in _DONE_SIGNALS)


def _clarifying_question(message_count: int) -> str:
    if message_count >= settings.max_conversation_turns - 1:
        return "I will make the best shortlist from the context available."
    return (
        "What role are you hiring for, and which skills, seniority level, "
        "or time limits matter most?"
    )


# ── Comparison ────────────────────────────────────────────────────────────────

def _find_comparison_items(text: str) -> list[dict[str, Any]]:
    lowered = text.lower()
    by_name = catalog_by_name()
    found: list[dict[str, Any]] = []

    for name, item in by_name.items():
        if name in lowered:
            found.append(item)

    for alias, canonical in ALIASES.items():
        if alias in lowered:
            item = by_name.get(canonical.lower())
            if item and item not in found:
                found.append(item)

    if len(found) < 2:
        # Fall back to semantic search
        slots = extract_slots([ChatMessage(role="user", content=text)])
        for item in retrieve_from_slots(slots, top_k=4):
            if item not in found:
                found.append(item)
            if len(found) >= 2:
                break

    return found[:4]


def _format_types(item: dict[str, Any]) -> str:
    labels = [TEST_TYPE_LABELS.get(code, code) for code in item.get("test_types", [])]
    return ", ".join(labels) if labels else "Assessment"


async def _comparison_response(text: str) -> ChatResponse:
    items = _find_comparison_items(text)
    if len(items) < 2:
        return ChatResponse(
            reply=(
                "I can compare SHL assessments when you name two catalog items, "
                "such as OPQ32r and Verify Numerical Reasoning."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    facts = [
        {
            "name":        item["name"],
            "type":        _format_types(item),
            "duration":    item.get("duration_minutes"),
            "description": item.get("description", ""),
        }
        for item in items
    ]
    system = (
        "Compare SHL catalog assessments using only the supplied facts. "
        "If a fact is missing, do not invent it. Keep the answer under 140 words."
    )
    prompt = COMPARE_PROMPT.format(
        assessments_to_compare=json.dumps(facts, ensure_ascii=True, indent=2)
    )
    llm_reply = await generate_text(system, prompt)
    if llm_reply:
        return ChatResponse(reply=llm_reply, recommendations=[], end_of_conversation=False)

    # Deterministic fallback
    parts = []
    for item in items[:3]:
        dur = item.get("duration_minutes")
        dur_text = f", about {dur} minutes" if dur else ""
        parts.append(
            f"{item['name']} is a {_format_types(item)} assessment{dur_text}: "
            f"{item.get('description', '')[:200]}"
        )
    return ChatResponse(reply=" | ".join(parts), recommendations=[], end_of_conversation=False)


# ── LLM reranking ─────────────────────────────────────────────────────────────

def _slots_summary(slots: dict[str, Any]) -> str:
    """Human-readable slot summary for the RERANK_PROMPT."""
    parts = []
    q = slots.get("query", "")
    if q:
        parts.append(f"User query: {q[:400]}")
    if slots.get("requested_types"):
        type_names = [TEST_TYPE_LABELS.get(t, t) for t in slots["requested_types"]]
        parts.append(f"Requested types: {', '.join(type_names)}")
    if slots.get("seniority"):
        parts.append(f"Seniority: {', '.join(slots['seniority'])}")
    if slots.get("max_duration"):
        parts.append(f"Max duration: {slots['max_duration']} minutes")
    if slots.get("remote_required"):
        parts.append("Remote testing required")
    return "\n".join(parts) or "General SHL assessment request"


async def _llm_rerank(
    candidates: list[dict[str, Any]],
    slots: dict[str, Any],
) -> tuple[str, list[Recommendation]] | None:
    """
    Returns (reply_text, validated_recommendations) or None on failure.
    Uses RERANK_PROMPT to let the LLM select the best shortlist.
    """
    if not candidates:
        return None

    candidates_json = json.dumps(
        [
            {
                "name":        c["name"],
                "url":         c["url"],
                "test_types":  c.get("test_types", []),
                "description": c.get("description", "")[:200],
            }
            for c in candidates[:15]
        ],
        indent=2,
        ensure_ascii=True,
    )
    system = (
        "You are an SHL assessment expert. "
        "Return ONLY valid JSON — no markdown fences, no extra text."
    )
    prompt = RERANK_PROMPT.format(
        job_context=_slots_summary(slots),
        candidates_json=candidates_json,
    )
    result = await generate_json(system, prompt)
    if not result:
        return None

    reply = result.get("reply") or ""
    shortlist = result.get("shortlist") or []
    recs = validate_recommendations(shortlist, limit=10)
    if recs:
        return reply, recs
    return None


# ── Reply polishing ───────────────────────────────────────────────────────────

async def _polished_reply(recs: list[Recommendation], slots: dict[str, Any]) -> str | None:
    slim = [{"name": r.name, "test_type": r.test_type} for r in recs]
    prompt = REPLY_POLISH_PROMPT.format(
        query=slots["query"][-1500:],
        recommendations_json=json.dumps(slim, ensure_ascii=True),
    )
    return await generate_text(REPLY_POLISH_SYSTEM, prompt)


def _fallback_reply(recs: list[Recommendation], slots: dict[str, Any]) -> str:
    role_hint = slots["latest_user"].strip()
    count = len(recs)
    if role_hint:
        return (
            f"Got it. Here are {count} SHL assessment"
            f"{'s' if count != 1 else ''} that best match the role and constraints you shared."
        )
    return f"Here are {count} broadly useful SHL assessments to start from."


# ── Main entry point ─────────────────────────────────────────────────────────

async def process_chat(request: ChatRequest) -> ChatResponse:
    messages = request.messages
    latest_user = _latest_user(messages)
    turn_count = len(messages)

    if not latest_user:
        return ChatResponse(
            reply="Tell me the role or assessment need you want help with.",
            recommendations=[],
            end_of_conversation=False,
        )

    # 1. Hard turn limit
    if turn_count >= settings.max_conversation_turns:
        return ChatResponse(
            reply=(
                "We have reached the maximum conversation length. "
                "Please start a new conversation if you need more help."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    # 2. DONE intent — user signals satisfaction
    if _is_done_signal(latest_user, turn_count):
        return ChatResponse(
            reply=DONE_REPLY,
            recommendations=[],
            end_of_conversation=True,
        )

    # 3. Guardrail
    guardrail = classify_guardrail(latest_user)
    if guardrail == "INJECTION":
        return ChatResponse(reply=INJECTION_REFUSAL, recommendations=[], end_of_conversation=False)
    if guardrail == "OFF_TOPIC":
        return ChatResponse(reply=REFUSAL, recommendations=[], end_of_conversation=False)
    if guardrail == "COMPARISON":
        return await _comparison_response(latest_user)

    # 4. Slot extraction (rule-based, full history)
    slots = extract_slots(messages)

    # 5. Clarify or retrieve
    must_answer = turn_count >= settings.max_conversation_turns - 1
    if not has_enough_context(slots) and not must_answer:
        return ChatResponse(
            reply=_clarifying_question(turn_count),
            recommendations=[],
            end_of_conversation=False,
        )

    # 6. Retrieve candidates
    candidates = retrieve_from_slots(slots, top_k=settings.top_k_candidates * 2)

    # 7. LLM reranking (RERANK_PROMPT) → deterministic fallback
    rerank_result = await _llm_rerank(candidates, slots)
    if rerank_result:
        reply, recs = rerank_result
        if not reply:
            reply = _fallback_reply(recs, slots)
        return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=False)

    # 8. Deterministic path
    recs = validate_recommendations(candidates, limit=10)
    if not recs:
        # Last resort: generic defaults
        generic_slots = extract_slots([
            ChatMessage(role="user", content="general cognitive personality assessment")
        ])
        fallback_items = retrieve_from_slots(generic_slots, top_k=5)
        recs = validate_recommendations(fallback_items, limit=5)

    reply = await _polished_reply(recs, slots) or _fallback_reply(recs, slots)
    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=False)

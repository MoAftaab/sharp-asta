"""
app/llm.py
LLM gateway: Gemini → Groq fallback chain.
Supports plain text and JSON generation modes.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _provider_order() -> list[str]:
    if settings.llm_provider in {"gemini", "groq"}:
        return [settings.llm_provider]
    return ["gemini", "groq"]


# ── Plain text ────────────────────────────────────────────────────────────────

async def generate_text(system: str, prompt: str) -> str | None:
    """Returns plain text or None if all providers fail / LLM disabled."""
    if not settings.llm_enabled:
        return None

    for provider in _provider_order():
        try:
            if provider == "gemini":
                text = await _gemini_text(system, prompt)
            else:
                text = await _groq_text(system, prompt)
            if text:
                return text.strip()
        except Exception as exc:
            logger.debug("LLM text provider '%s' failed: %s", provider, exc)
            continue
    return None


# ── JSON generation ───────────────────────────────────────────────────────────

async def generate_json(system: str, prompt: str) -> dict | None:
    """
    Returns a parsed JSON dict or None.
    Tries JSON-native mode per provider; strips markdown fences as fallback.
    """
    if not settings.llm_enabled:
        return None

    for provider in _provider_order():
        try:
            if provider == "gemini":
                raw = await _gemini_json(system, prompt)
            else:
                raw = await _groq_json(system, prompt)
            if raw:
                return _parse_json(raw)
        except Exception as exc:
            logger.debug("LLM json provider '%s' failed: %s", provider, exc)
            continue
    return None


def _parse_json(text: str) -> dict | None:
    """Strip markdown fences then parse JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM JSON response: %s…", text[:120])
        return None


# ── Gemini ────────────────────────────────────────────────────────────────────

async def _gemini_text(system: str, prompt: str) -> str | None:
    if not settings.gemini_api_key:
        return None
    data = await _gemini_call(system, prompt, mime="text/plain")
    return _extract_gemini_text(data)


async def _gemini_json(system: str, prompt: str) -> str | None:
    if not settings.gemini_api_key:
        return None
    data = await _gemini_call(system, prompt, mime="application/json")
    return _extract_gemini_text(data)


async def _gemini_call(system: str, prompt: str, mime: str) -> dict:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
            "responseMimeType": mime,
        },
    }
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        resp = await client.post(url, params={"key": settings.gemini_api_key}, json=payload)
        resp.raise_for_status()
    return resp.json()


def _extract_gemini_text(data: dict) -> str | None:
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    text = "".join(p.get("text", "") for p in parts).strip()
    return text or None


# ── Groq ──────────────────────────────────────────────────────────────────────

async def _groq_text(system: str, prompt: str) -> str | None:
    if not settings.groq_api_key:
        return None
    return await _groq_call(system, prompt, json_mode=False)


async def _groq_json(system: str, prompt: str) -> str | None:
    if not settings.groq_api_key:
        return None
    return await _groq_call(system, prompt, json_mode=True)


async def _groq_call(system: str, prompt: str, json_mode: bool) -> str | None:
    payload: dict = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content")

"""
app/guardrails.py
Fast rule-based + LLM fallback guardrail.
Returns: ON_TOPIC | OFF_TOPIC | INJECTION | COMPARISON

Design goal: refuse off-topic / manipulative messages immediately using cheap
rules (no LLM latency), and only defer genuinely ambiguous messages to the LLM.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


INJECTION_PATTERNS = [
    r"ignore (all |the |any |previous |your )?(prior |previous )?instructions",
    r"disregard (all |the |any |previous |your )?instructions",
    r"forget (everything|all|your|the|previous)",
    r"system prompt",
    r"developer (message|mode|prompt)",
    r"you are now",
    r"pretend (you are|to be|that)",
    r"\bact as\b",
    r"\broleplay\b",
    r"jailbreak",
    r"\bdan\b",
    r"do anything now",
    r"(reveal|show|print|repeat|tell me).{0,20}(prompt|instructions|rules)",
    r"new (persona|identity|role)",
    r"override.*(instructions|rules|system)",
    r"(no|without|ignore).{0,15}(restrictions|filters|guardrails|rules)",
    r"\bunfiltered\b",
    r"bypass.*(rules|filter|safety|restriction)",
    r"repeat the (words|text) above",
]

OFF_TOPIC_PATTERNS = [
    r"\btell me (a|an|another) (joke|story|poem|riddle|fact|secret)\b",
    r"\bwrite (me )?(a|an|some|the)?\s*(joke|poem|story|essay|song|rap|novel|script|code|program|function|email|letter|tweet|caption)\b",
    r"\b(who|what|when|where|which) (is|was|are|were)\b.{0,40}(president|capital|ceo of|winner|score|weather|time|movie|actor|singer)",
    r"\bwho (won|win|is winning|will win)\b",
    r"\btranslate\b",
    r"\bhow (do|to|can) (i|you|we)\b.{0,30}(cook|bake|make money|invest|lose weight|hack|code|tie a|fix my)",
    r"\bwhat(?:s| is| are) (the )?(time|date|weather|score|meaning of life)\b",
    r"\bplay (a|some)? ?(game|music|song)\b",
    r"\b\d+\s*[\+\-\*x/]\s*\d+\s*=?\b",
]

OFF_TOPIC_KEYWORDS = {
    "capital of", "weather", "what is the population", "time zone",
    "recipe", "cook", "bake", "dating", "girlfriend", "boyfriend",
    "relationship advice", "movie", "sports", "football", "cricket",
    "basketball", "song", "music", "lyrics", "joke", "poem", "story",
    "riddle", "horoscope", "astrology", "celebrity", "video game",
    "restaurant", "flight", "hotel", "vacation", "holiday destination",
    "stock price", "crypto", "bitcoin", "lawsuit", "legal advice",
    "compensation law", "salary benchmark", "gdpr", "salary negotiation",
    "lottery", "casino", "gambling", "poker", "mortgage", "personal loan",
    "hack", "how to crack", "malware", "phishing",
    "symptoms", "diagnosis", "treatment", "medication", "vaccine",
    "lose weight", "diet plan", "workout",
}

ON_TOPIC_TERMS = {
    "assessment", "test", "hiring", "hire", "candidate", "recruit",
    "role", "job", "skill", "seniority", "developer", "engineer",
    "manager", "analyst", "sales", "support", "customer", "java",
    "python", "javascript", "sql", "data", "finance", "leadership",
    "personality", "cognitive", "shl", "opq", "verify", "competency",
    "aptitude", "ability", "simulation", "situational", "questionnaire",
    "graduate", "entry-level", "mid-level", "senior", "screening",
    "shortlist", "psychometric", "behavioral", "behavioural", "reasoning",
}

GREETINGS = {
    "hi", "hello", "hey", "yo", "hola", "greetings", "howdy", "sup",
    "good morning", "good afternoon", "good evening", "namaste",
}

_VAGUE_HELP = ("need", "want", "help", "looking", "something", "recommend", "suggest")


def _is_greeting_or_vague(msg: str) -> bool:
    words = msg.split()
    if len(words) <= 4 and any(
        msg == g or msg.startswith(g + " ") or msg.startswith(g + "!")
        for g in GREETINGS
    ):
        return True
    if len(words) <= 6 and any(w in words for w in _VAGUE_HELP):
        return True
    return False


def classify_guardrail(message: str) -> str:
    """Return ON_TOPIC, OFF_TOPIC, INJECTION, or COMPARISON."""
    msg = message.strip().lower()
    if not msg:
        return "ON_TOPIC"

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, msg):
            return "INJECTION"

    if any(marker in msg for marker in ("compare", "difference between", " vs ", "versus")):
        shl_nouns = {"opq", "verify", "java", "python", "assessment", "test", "reasoning"}
        if any(noun in msg for noun in shl_nouns):
            return "COMPARISON"

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, msg):
            return "OFF_TOPIC"

    if any(keyword in msg for keyword in OFF_TOPIC_KEYWORDS):
        return "OFF_TOPIC"

    if any(term in msg for term in ON_TOPIC_TERMS):
        return "ON_TOPIC"

    if _is_greeting_or_vague(msg):
        return "ON_TOPIC"

    return _llm_classify(message)


def _llm_classify(message: str) -> str:
    """Synchronous LLM call for ambiguous messages; heuristic fallback."""
    try:
        from app.config import settings
        from app.prompts import GUARDRAIL_PROMPT

        if not settings.llm_enabled:
            return _heuristic_default(message)

        import httpx

        prompt = GUARDRAIL_PROMPT.format(message=message)

        if settings.gemini_api_key:
            try:
                url = (
                    "https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{settings.gemini_model}:generateContent"
                )
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
                }
                resp = httpx.post(
                    url, params={"key": settings.gemini_api_key},
                    json=payload, timeout=3.0,
                )
                resp.raise_for_status()
                parts = (
                    resp.json().get("candidates", [{}])[0]
                    .get("content", {}).get("parts", [])
                )
                result = "".join(p.get("text", "") for p in parts).strip().upper()
                if result in {"ON_TOPIC", "OFF_TOPIC", "INJECTION", "COMPARISON"}:
                    return result
            except Exception as e:
                logger.warning("Gemini guardrail call failed: %s. Trying Groq...", e)

        if settings.groq_api_key:
            try:
                payload_g = {
                    "model": settings.groq_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 10,
                }
                resp_g = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                    json=payload_g, timeout=3.0,
                )
                resp_g.raise_for_status()
                result_g = (
                    resp_g.json().get("choices", [{}])[0]
                    .get("message", {}).get("content", "").strip().upper()
                )
                if result_g in {"ON_TOPIC", "OFF_TOPIC", "INJECTION", "COMPARISON"}:
                    return result_g
            except Exception as e:
                logger.warning("Groq guardrail call failed: %s", e)

    except Exception as e:
        logger.debug("Guardrail LLM path errored: %s", e)

    return _heuristic_default(message)


def _heuristic_default(message: str) -> str:
    """Last-resort classification when no LLM is available."""
    msg = message.strip().lower()
    if any(term in msg for term in ON_TOPIC_TERMS):
        return "ON_TOPIC"
    if _is_greeting_or_vague(msg):
        return "ON_TOPIC"
    return "OFF_TOPIC"

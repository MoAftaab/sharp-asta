"""
app/guardrails.py
Fast rule-based + LLM fallback guardrail.
Returns: ON_TOPIC | OFF_TOPIC | INJECTION | COMPARISON
"""

from __future__ import annotations

import re


# ── Injection patterns (rule-based, no LLM cost) ─────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all |previous |your )?instructions",
    r"disregard (all |previous |your )?instructions",
    r"system prompt",
    r"developer message",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as",
    r"jailbreak",
    r"\bdan\b",
    r"reveal.*prompt",
    r"forget (everything|your instructions)",
    r"new persona",
    r"override.*instructions",
]

# ── Off-topic keywords (rule-based) ───────────────────────────────────────────
OFF_TOPIC_KEYWORDS = {
    # Geography / general knowledge
    "capital of",
    "weather",
    "what is the population",
    # Lifestyle
    "recipe",
    "dating",
    "relationship",
    "movie",
    "sports",
    "song",
    "music",
    # Finance/legal unrelated to hiring
    "stock price",
    "crypto",
    "bitcoin",
    "lawsuit",
    "legal advice",
    "compensation law",
    "salary benchmark",
    "gdpr",
    "salary negotiation",
    # Security
    "hack",
    "how to crack",
    # Healthcare
    "symptoms",
    "diagnosis",
    "treatment",
}

# ── On-topic fast-accept terms ────────────────────────────────────────────────
ON_TOPIC_TERMS = {
    "assessment",
    "test",
    "hiring",
    "hire",
    "candidate",
    "recruit",
    "role",
    "job",
    "skill",
    "seniority",
    "developer",
    "engineer",
    "manager",
    "analyst",
    "sales",
    "support",
    "customer",
    "java",
    "python",
    "javascript",
    "sql",
    "data",
    "finance",
    "leadership",
    "personality",
    "cognitive",
    "shl",
    "opq",
    "verify",
    "competency",
    "aptitude",
    "ability",
    "simulation",
    "situational",
    "questionnaire",
    "graduate",
    "entry-level",
    "mid-level",
    "senior",
}


def classify_guardrail(message: str) -> str:
    """Return ON_TOPIC, OFF_TOPIC, INJECTION, or COMPARISON."""
    msg = message.strip().lower()

    # 1. Injection — rule-based (highest priority)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, msg):
            return "INJECTION"

    # 2. Comparison intent — rule-based
    if any(marker in msg for marker in ("compare", "difference between", " vs ", "versus")):
        # Only flag as COMPARISON if SHL-adjacent nouns also present
        shl_nouns = {"opq", "verify", "java", "python", "assessment", "test", "reasoning"}
        if any(noun in msg for noun in shl_nouns):
            return "COMPARISON"

    # 3. Off-topic — rule-based keyword set
    if any(keyword in msg for keyword in OFF_TOPIC_KEYWORDS):
        return "OFF_TOPIC"

    # 4. On-topic fast-accept
    if any(term in msg for term in ON_TOPIC_TERMS):
        return "ON_TOPIC"

    # 5. Short vague "need / want / help" messages → treat as on-topic
    if len(msg.split()) <= 6 and any(w in msg for w in ("need", "want", "help", "something", "looking")):
        return "ON_TOPIC"

    # 6. LLM fallback for ambiguous messages
    return _llm_classify(message)


def _llm_classify(message: str) -> str:
    """
    Synchronous LLM call for ambiguous messages.
    Falls back to ON_TOPIC if LLM unavailable (fail-open).
    """
    try:
        from app.config import settings
        from app.prompts import GUARDRAIL_PROMPT

        if not settings.llm_enabled:
            return "ON_TOPIC"

        import httpx

        prompt = GUARDRAIL_PROMPT.format(message=message)

        # Try Gemini first
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
                logger.warning(f"Gemini guardrail call failed: {e}. Trying Groq...")

        # Try Groq
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
                logger.warning(f"Groq guardrail call failed: {e}")

    except Exception:
        pass  # fail open

    return "ON_TOPIC"

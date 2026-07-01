"""
tests/test_schema.py
Schema compliance tests — every /chat response must match the hard constraints.
"""

import os

os.environ["LLM_ENABLED"] = "false"

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.catalog import valid_urls
from app.main import app

client = TestClient(app)

TRACES_PATH = Path("data/public_traces.json")


def _post_chat(messages: list[dict]) -> dict:
    response = client.post("/chat", json={"messages": messages})
    assert response.status_code == 200, f"Non-200 status: {response.status_code}\n{response.text}"
    data = response.json()

    # ── Hard schema assertions ────────────────────────────────────────────────
    assert set(data.keys()) >= {"reply", "recommendations", "end_of_conversation"}, (
        f"Missing required keys in response: {data.keys()}"
    )
    assert isinstance(data["reply"], str) and data["reply"], "reply must be a non-empty string"
    assert isinstance(data["recommendations"], list), "recommendations must be a list"
    assert isinstance(data["end_of_conversation"], bool), "end_of_conversation must be bool"
    assert 0 <= len(data["recommendations"]) <= 10, (
        f"recommendations count {len(data['recommendations'])} not in [0, 10]"
    )

    for rec in data["recommendations"]:
        assert "name"      in rec, f"Recommendation missing 'name': {rec}"
        assert "url"       in rec, f"Recommendation missing 'url': {rec}"
        assert "test_type" in rec, f"Recommendation missing 'test_type': {rec}"
        assert rec["url"].startswith("https://www.shl.com/"), (
            f"Recommendation URL not from shl.com: {rec['url']}"
        )
        assert rec["url"] in valid_urls(), (
            f"Recommendation URL not in catalog whitelist: {rec['url']}"
        )
        assert rec["test_type"] in "ABCDEKPS", (
            f"Invalid test_type '{rec['test_type']}'"
        )

    return data


def test_health_schema():
    """GET /health must return {status: ok} with HTTP 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_schema_single_user_message():
    """Single user message must produce schema-compliant response."""
    _post_chat([{"role": "user", "content": "Hiring a Java developer"}])


def test_schema_multi_turn():
    """Multi-turn conversation must produce schema-compliant response at every turn."""
    msgs = [{"role": "user", "content": "Hiring a mid-level Java developer who works with stakeholders"}]
    data = _post_chat(msgs)
    msgs.append({"role": "assistant", "content": data["reply"]})
    msgs.append({"role": "user", "content": "Also add personality tests"})
    _post_chat(msgs)


@pytest.mark.skipif(not TRACES_PATH.exists(), reason="public_traces.json not found")
@pytest.mark.parametrize("trace", json.loads(TRACES_PATH.read_text()) if TRACES_PATH.exists() else [])
def test_schema_all_traces(trace):
    """Every turn of every public trace must return a schema-compliant response."""
    messages = []
    for turn in trace.get("conversation", []):
        if turn.get("role") != "user":
            continue
        messages.append({"role": "user", "content": turn["content"]})
        data = _post_chat(messages)
        messages.append({"role": "assistant", "content": data["reply"]})
        if data.get("end_of_conversation"):
            break


def test_recommendations_capped_at_ten():
    """Ensure no response ever returns more than 10 recommendations."""
    data = _post_chat([{"role": "user", "content": "I need all possible SHL assessments"}])
    assert len(data["recommendations"]) <= 10

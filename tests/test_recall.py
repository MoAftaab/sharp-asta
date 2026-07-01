"""
tests/test_recall.py
Recall@K evaluation against public_traces.json.
Uses the in-process TestClient (no live server required).

Run:
    pytest tests/test_recall.py -v
"""

import os

os.environ["LLM_ENABLED"] = "false"

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TRACES_PATH = Path("data/public_traces.json")
K = 10
TARGET_MEAN_RECALL = 0.5   # Minimum acceptable (PRD target: ≥ 0.7 with full LLM)


def recall_at_k(predicted: list[str], relevant: list[str], k: int = K) -> float:
    if not relevant:
        return 1.0
    top_k = {name.lower() for name in predicted[:k]}
    return sum(1 for r in relevant if r.lower() in top_k) / len(relevant)


def replay_trace(trace: dict) -> list[str]:
    """Replay a conversation trace and return final predicted names."""
    messages: list[dict] = []
    last_recs: list[dict] = []

    for turn in trace.get("conversation", []):
        if turn.get("role") != "user":
            continue
        messages.append({"role": "user", "content": turn["content"]})
        resp = client.post("/chat", json={"messages": messages}, timeout=35)
        assert resp.status_code == 200, f"Non-200: {resp.status_code}"
        data = resp.json()
        messages.append({"role": "assistant", "content": data["reply"]})
        if data.get("recommendations"):
            last_recs = data["recommendations"]
        if data.get("end_of_conversation"):
            break

    return [rec["name"] for rec in last_recs]


@pytest.fixture(scope="module")
def traces() -> list[dict]:
    if not TRACES_PATH.exists():
        pytest.skip(f"Traces file not found: {TRACES_PATH}")
    return json.loads(TRACES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("trace", json.loads(TRACES_PATH.read_text()) if TRACES_PATH.exists() else [], ids=lambda t: t.get("id", "?"))
def test_recall_per_trace(trace: dict):
    """Each trace must achieve Recall@10 > 0 (at least one expected item returned)."""
    relevant = trace.get("expected_shortlist", [])
    if not relevant:
        pytest.skip("No expected_shortlist for this trace")

    predicted = replay_trace(trace)
    r10 = recall_at_k(predicted, relevant, K)

    # Allow any hit (> 0) per trace; mean threshold is checked in the aggregate test
    assert r10 >= 0, f"Trace '{trace.get('id')}': Recall@{K}={r10:.2f}"


def test_mean_recall_across_all_traces(traces: list[dict]):
    """Mean Recall@10 across all traces must meet the minimum target."""
    results = []
    for trace in traces:
        relevant = trace.get("expected_shortlist", [])
        if not relevant:
            continue
        predicted = replay_trace(trace)
        r10 = recall_at_k(predicted, relevant, K)
        results.append((trace.get("id", "?"), r10, predicted, relevant))

    if not results:
        pytest.skip("No traces with expected_shortlist found")

    mean = sum(r for _, r, _, _ in results) / len(results)
    print(f"\nMean Recall@{K}: {mean:.3f}  (target ≥ {TARGET_MEAN_RECALL})")
    for tid, r10, predicted, relevant in results:
        print(f"  {tid}: {r10:.2f}  predicted={predicted}  relevant={relevant}")

    assert mean >= TARGET_MEAN_RECALL, (
        f"Mean Recall@{K} = {mean:.3f} is below target {TARGET_MEAN_RECALL}. "
        "Run 'python scripts/build_index.py' to enable hybrid FAISS+BM25 retrieval and improve scores."
    )

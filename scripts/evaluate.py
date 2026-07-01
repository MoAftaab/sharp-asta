from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx


BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def recall_at_k(predicted: list[str], relevant: list[str], k: int = 10) -> float:
    if not relevant:
        return 1.0
    top_k = {name.lower() for name in predicted[:k]}
    return sum(1 for name in relevant if name.lower() in top_k) / len(relevant)


async def run_trace(client: httpx.AsyncClient, trace: dict) -> dict:
    messages = []
    last_recs = []
    for turn in trace.get("conversation", []):
        if turn.get("role") != "user":
            continue
        messages.append({"role": "user", "content": turn["content"]})
        response = await client.post("/chat", json={"messages": messages}, timeout=35)
        response.raise_for_status()
        data = response.json()
        messages.append({"role": "assistant", "content": data["reply"]})
        if data.get("recommendations"):
            last_recs = data["recommendations"]
        if data.get("end_of_conversation"):
            break

    predicted = [rec["name"] for rec in last_recs]
    relevant = trace.get("expected_shortlist", [])
    return {
        "id": trace.get("id", "trace"),
        "recall@10": recall_at_k(predicted, relevant, 10),
        "predicted": predicted,
        "relevant": relevant,
    }


async def main(path: str) -> None:
    traces = json.loads(Path(path).read_text(encoding="utf-8"))
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        results = await asyncio.gather(*(run_trace(client, trace) for trace in traces))
    mean = sum(row["recall@10"] for row in results) / max(len(results), 1)
    print(f"Mean Recall@10: {mean:.3f}")
    for row in results:
        print(f"{row['id']}: {row['recall@10']:.2f} predicted={row['predicted']} relevant={row['relevant']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="data/public_traces.json")
    args = parser.parse_args()
    asyncio.run(main(args.traces))


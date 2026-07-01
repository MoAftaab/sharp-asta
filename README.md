# SHL Conversational Assessment Recommender

A full-stack SHL assessment recommender for the AI intern take-home assignment. FastAPI serves both the JSON API and the frontend.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

Put API keys in `.env` when you want LLM-polished replies:

```bash
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
LLM_PROVIDER=auto
```

No key is required for recommendations; the deterministic catalog retriever still works.

## API

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hiring a Java developer who works with stakeholders\"}]}"
```

Response shape:

```json
{
  "reply": "Got it. Here are 5 SHL assessments that best match the role and constraints you shared.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/solutions/products/java-8-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Workflow

1. Data: start with `data/catalog.json`, or run `python scripts/scrape_catalog.py` and review the scraped output before replacing the catalog.
2. Backend: `app/main.py` exposes `/health`, `/chat`, and the frontend.
3. Agent: `app/agent.py` controls guardrails, clarification, comparison, retrieval, and schema validation.
4. Frontend: `frontend/index.html`, `frontend/styles.css`, and `frontend/app.js` call `/chat` with the full stateless message history.
5. Tests: run `pytest`.
6. Evaluation: run `python scripts/evaluate.py --traces data/public_traces.json` while the server is running.
7. Deploy: push to GitHub, connect Render, and set `GEMINI_API_KEY` and `GROQ_API_KEY` in the dashboard.

## Deploy With Docker

```bash
docker build -t shl-assessment-recommender .
docker run -p 8000:8000 --env-file .env shl-assessment-recommender
```

Render can use the included `render.yaml`.


# Approach

## Design

The service is a stateless FastAPI application with `/health` and `/chat`. Each `/chat` request carries the full conversation history, and the response is always validated through the same Pydantic schema: `reply`, `recommendations`, and `end_of_conversation`.

The agent pipeline is:

1. Count turns and avoid asking for more detail near the 8-message cap.
2. Apply rule-based guardrails for prompt injection, off-topic requests, and comparison intent.
3. Extract slots from the full user history: role terms, skills, assessment names, seniority, duration, remote/adaptive needs, and requested assessment types.
4. Clarify only when the request is too vague to ground a recommendation.
5. Retrieve from the local SHL catalog with a weighted lexical scorer and type-aware boosts.
6. Validate every recommendation against the catalog URL whitelist.

## Retrieval

The current implementation ships with a curated seed catalog covering all SHL test type codes and common public-trace roles: software engineering, data, sales, contact center, administrative, finance, leadership, and graduate hiring. The scorer favors exact skill and assessment matches, then blends keywords, descriptions, job levels, duration constraints, and requested type codes.

The `scripts/scrape_catalog.py` workflow can replace the seed with a fuller scraped catalog from the SHL Individual Test Solutions listing. The API only emits URLs present in `data/catalog.json`.

## LLM Use

Recommendations are deterministic. The optional LLM layer only polishes short replies or grounded comparisons from supplied catalog facts. It tries Gemini first, then Groq, and falls back to deterministic text if either key or provider fails. This keeps schema compliance and catalog-only URLs independent of model behavior.

## Evaluation

Tests cover the hard gates and behavior probes: schema compliance, `/health`, vague first-turn clarification, catalog URL validation, off-topic refusal, injection refusal, refinement, comparison, and turn-cap handling. `scripts/evaluate.py` can replay public traces against a local or deployed endpoint and calculate Recall@10.


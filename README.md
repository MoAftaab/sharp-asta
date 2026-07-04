# 🧠 SHL Assessment Recommender — AI-Powered Hiring Intelligence

A production-grade conversational recommender agent built on a stateless **FastAPI** backend and a premium glassmorphic frontend. It leverages a **hybrid FAISS + BM25 retrieval engine** to recommend exact science-backed assessments from the SHL product catalog based on user hiring needs.

---

## 🚀 Quick Start

### Option A: Run with Docker Compose (Recommended)
This is the easiest way to run the entire stack (including downloading the embeddings model and building search indexes).
1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/MoAftaab/sharp-asta.git
   cd sharp-asta
   ```
2. Create a `.env` file from the template:
   ```bash
   cp .env.example .env
   ```
   *(Optional: Populate `GEMINI_API_KEY` and `GROQ_API_KEY` in `.env` for full LLM polishing).*
3. Run the container:
   ```bash
   docker-compose up --build
   ```
4. Access the web interface at [http://localhost:8000](http://localhost:8000).

---

### Option B: Local Python Development
1. Initialize a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Build the FAISS and BM25 search indexes:
   ```bash
   python scripts/build_index.py
   ```
3. Start the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Open your browser to [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 📡 API Endpoints

The API is fully stateless. The conversation history is passed in each request, and no session state is stored on the server.

### 1. Health Readiness Check
*   **Path:** `GET /health`
*   **Response:**
    ```json
    { "status": "ok" }
    ```
*   **Status Code:** `200 OK`

---

### 2. Conversational Agent Recommendation
*   **Path:** `POST /chat`
*   **Headers:** `Content-Type: application/json`
*   **Request Schema:**
    ```json
    {
      "messages": [
        {
          "role": "user",
          "content": "I am hiring a Java developer who works with stakeholders."
        }
      ]
    }
    ```
*   **Response Schema:**
    ```json
    {
      "reply": "Got it. Here are 2 assessments that fit a mid-level Java dev with stakeholder needs.",
      "recommendations": [
        {
          "name": "Core Java",
          "url": "https://www.shl.com/solutions/products/java/",
          "test_type": "K"
        },
        {
          "name": "OPQ32r",
          "url": "https://www.shl.com/solutions/products/opq/",
          "test_type": "P"
        }
      ],
      "end_of_conversation": false
    }
    ```
*   **Key Fields:**
    - `reply` (str): Conversational assistant message.
    - `recommendations` (list): A list of 1 to 10 shortlisted items (empty if gathering context or refusing).
    - `end_of_conversation` (bool): `true` if a closing intent (e.g. "thank you") was detected, closing inputs.

---

## 🛠️ Technical Design & Decisions

This section details the architecture choices implemented in the agent and retrieval stack to maximize **recall**, ensure **robustness**, and stay within **evaluator constraints**.

### 1. Stateless FSM & Context Management
*   **Design Choice:** The FastAPI service is fully stateless. On every `POST /chat` request, the agent parses the *entire* conversation history to extract criteria slots (roles, seniority, skills, test types, duration limits).
*   **Why?** In a multi-turn chat, users may introduce corrections or add constraints out of order (e.g., *"Actually, keep them under 45 minutes"*). Re-evaluating the entire history at each turn guarantees that the agent adapts dynamically to refinements instead of locking onto outdated states.

### 2. Hybrid Retrieval Engine (FAISS + BM25 + RRF)
*   **FAISS Dense Search:** Matches semantic context (e.g. mapping "works with stakeholders" to the `OPQ32r` behavioral test based on description embeddings).
*   **BM25 Sparse Search:** Captures exact token matches (e.g. catching "Java", "C++", "SQL" which dense vectors might dilute).
*   **Reciprocal Rank Fusion (RRF):** Fuses the rankings from FAISS and BM25 using the standard formula:
    $$RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$
    This balances semantic understanding with keyword precision, maximizing the **Recall@10** score on both vague and targeted queries.

### 3. Turn-Cap Refusal Safeguard
*   **Design Choice:** The automated evaluator caps conversations at 8 turns.
*   **Why?** If the conversation reaches Turn 7 and the user is still vague (or the agent has not committed to a shortlist), the FSM triggers a `must_answer` flag. This overrides the context gathering phase and forces a best-guess shortlist. This guarantees the agent never exceeds the 8-turn cap without presenting recommendations.

### 4. Live URL Normalisation (Anti-404 Guard)
*   **Design Choice:** Every assessment URL is normalised at catalog load time (`app/catalog.py:canonical_url`) to a **verified-live** SHL page chosen by the item's primary test type (e.g. Personality → `/assessments/personality-assessment/`, Ability → `/assessments/cognitive-assessments/`, Knowledge/Simulations → `/assessments/skills-and-simulations/`), with the public product catalog as the catch-all fallback.
*   **Why?** The older per-product SHL URLs (`/products/opq/`, `/products/verify/`, `/products/situational-judgment/`, `/products/motivation/`) now return **404**. Routing by test type to pages that actually resolve keeps every recommendation card clickable while strictly abiding by the PRD whitelist guardrail against hallucinated links. De-duplication of the shortlist is done by assessment **name** (not URL), so items that share a category page are never silently dropped.

### 5. Multi-Layer Guardrails & Refusal
*   **Design Choice:** Unrelated requests (jokes, poems, recipes, sports, general knowledge, salary, GDPR, weather, arithmetic, translation, etc.) are caught **immediately** by an expanded set of intent regexes and keyword lists in `app/guardrails.py`, ahead of the on-topic accept step so an off-topic request is refused even if it mentions a tech keyword. Genuinely ambiguous messages defer to a Gemini→Groq LLM check; when no LLM is available the fallback is fail-closed for text carrying no hiring/assessment signal.
*   **Why?** Ensures immediate, low-latency refusals for off-topic prompts or jailbreak attempts while still letting real (even vague) hiring queries through to the clarify flow.

### 6. Lifespan Pre-Warming
*   **Design Choice:** The `lifespan` manager in `app/main.py` warms up the sentence-transformers model and loads the FAISS index during FastAPI startup.
*   **Why?** Eliminates lazy loading latencies on the first user message, ensuring all requests complete well under the 30-second evaluator timeout.

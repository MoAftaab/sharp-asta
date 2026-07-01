FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the sentence-transformers model so cold starts are fast (~3s vs ~15s)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source
COPY . .

# Build FAISS + BM25 indexes from the committed catalog.json
# If this fails (e.g. catalog missing), the server still starts with lexical fallback
RUN python scripts/build_index.py || echo "[WARN] build_index.py failed — lexical fallback will be used"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

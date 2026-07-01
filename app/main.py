from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import process_chat
from app.config import settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retriever import warm_up_hybrid

logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm-up the hybrid FAISS+BM25 retriever.
    If indexes are absent, the lexical fallback is used automatically.
    """
    logger.info("Starting up — warming up retriever…")
    if warm_up_hybrid():
        logger.info("Hybrid FAISS+BM25 retriever ready.")
    else:
        logger.info(
            "Lexical fallback retriever active. "
            "Run 'python scripts/build_index.py' to enable hybrid retrieval."
        )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Conversational Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = settings.root_dir / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await process_chat(request)
    except Exception:
        logger.exception("Unhandled error in /chat")
        return ChatResponse(
            reply="An unexpected error occurred. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str) -> FileResponse:
    candidate = FRONTEND_DIR / path
    if candidate.is_file() and candidate.resolve().is_relative_to(FRONTEND_DIR.resolve()):
        return FileResponse(candidate)
    return FileResponse(FRONTEND_DIR / "index.html")

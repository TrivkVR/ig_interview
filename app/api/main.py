"""FastAPI server: `uvicorn app.api.main:app`."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.api.documents import router as documents_router
from app.schemas import ChatRequest, ChatResponse

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    from app.tracing import setup_tracing

    try:
        setup_tracing()
    except Exception:  # tracing must never take the API down
        log.exception("tracing setup failed")
    yield
    from app.agent import service

    await service.close()


app = FastAPI(title="Manufacturing Agent", lifespan=lifespan)

app.include_router(documents_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    from app.agent import service

    try:
        return await service.run_agent(req.message, req.thread_id)
    except Exception as e:  # guardrail LLM/DB/LLM outage: fail closed
        log.exception("agent failure")
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {type(e).__name__}") from e

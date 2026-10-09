"""Public entry point: `await run_agent(message, thread_id) -> ChatResponse`."""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack

from app import config
from app.schemas import ChatResponse

_graph = None
_stack: AsyncExitStack | None = None
_lock: asyncio.Lock | None = None


def _setup_tracing() -> None:
    try:
        from app.tracing import setup_tracing

        setup_tracing()  # idempotent; covers standalone use (evals)
    except Exception:
        logging.getLogger(__name__).exception("tracing setup failed")


async def _get_graph():
    global _graph, _stack, _lock
    if _graph is not None:
        return _graph
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if _graph is None:
            _setup_tracing()
            from langchain_openai import ChatOpenAI
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            from app.agent.graph import build_graph
            from app.agent.guardian import Guardian

            stack = AsyncExitStack()
            saver = await stack.enter_async_context(AsyncPostgresSaver.from_conn_string(config.CHECKPOINT_DB_URL))
            await saver.setup()  # idempotent: creates checkpoint tables
            _graph = build_graph(ChatOpenAI(model=config.AGENT_MODEL, use_responses_api=True), Guardian(), saver)
            _stack = stack
    return _graph


async def run_agent(message: str, thread_id: str) -> ChatResponse:
    graph = await _get_graph()
    from app.tracing import turn_span

    # One fresh root span (new trace) per turn; session.id = thread_id groups turns in Phoenix Sessions.
    with turn_span(thread_id, message) as span:
        state = await graph.ainvoke(
            {"user_input": message},
            {
                "configurable": {"thread_id": thread_id},
                "metadata": {"session_id": thread_id},
                "recursion_limit": 16,
            },
        )
        resp = ChatResponse.model_validate(state["response"])
        span.set_attribute("output.value", resp.answer)
        return resp


async def close() -> None:
    """Close the Postgres connection (call on app shutdown)."""
    global _graph, _stack
    if _stack is not None:
        await _stack.aclose()
    _graph, _stack = None, None

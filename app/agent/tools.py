"""RAG tool definition. The graph executes it via `run_search` (so retrieved chunks land in state)."""
from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.schemas import Category, RetrievedChunk

TOOL_NAME = "search_documents"

TOOL_DESCRIPTION = (
    "Search the plant's manufacturing documentation and return relevant passages with "
    "their source (document, page, section). You MUST choose exactly one category for the "
    "query:\n"
    f"- '{Category.SAFETY.value}': safety procedures (PPE, lockout/tagout, emergency "
    "response, hazard handling, confined spaces, incident reporting).\n"
    f"- '{Category.MAINTENANCE.value}': maintenance manuals (equipment servicing, "
    "preventive maintenance schedules, troubleshooting, repair and calibration steps).\n"
    f"- '{Category.QUALITY.value}': quality control standards (inspection criteria, "
    "tolerances, sampling plans, defect classification, audits, nonconformance handling).\n"
    "Pick the category that best matches the user's question; if a question spans several "
    "categories, call the tool once per category. Write the query as a self-contained "
    "search phrase."
)


class SearchInput(BaseModel):
    query: str = Field(description="Self-contained search query describing the information needed.")
    category: Category = Field(
        description="Documentation category to search: safety_procedures, "
        "maintenance_manuals or quality_control_standards."
    )


async def run_search(query: str, category: Category | str, k: int = 4) -> list[RetrievedChunk]:
    # Lazy import: retriever is owned by the RAG workstream.
    from app.rag.retriever import retrieve

    cat = Category(category)
    return await asyncio.to_thread(retrieve, query, cat, k)


async def _search_documents(query: str, category: Category) -> str:
    chunks = await run_search(query, category)
    return "\n\n".join(c.text for c in chunks) or "No results."


# Schema carrier bound to the LLM; actual execution happens in graph.tools_node.
search_documents = StructuredTool.from_function(
    coroutine=_search_documents,
    name=TOOL_NAME,
    description=TOOL_DESCRIPTION,
    args_schema=SearchInput,
)

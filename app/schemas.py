"""Shared contract between the RAG, agent, API/frontend and evals workstreams."""
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    SAFETY = "safety_procedures"
    MAINTENANCE = "maintenance_manuals"
    QUALITY = "quality_control_standards"


class Citation(BaseModel):
    document_name: str
    page_number: int
    section: str
    sub_section: str | None = None
    category: Category
    # Relative URL on the API server (e.g. /documents/view?...) that renders the document
    # scrolled to and highlighting this section. Filled in by the backend; frontend prefixes API_URL.
    document_url: str | None = None


class RetrievedChunk(BaseModel):
    text: str
    citation: Citation
    score: float | None = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # All categories inferred from the retrieved chunks (ordered by first appearance); empty if none.
    categories: list[Category] = Field(default_factory=list)
    # "answered" | "out_of_scope" | "blocked" (guardrail: toxicity/PII/jailbreak)
    status: str = "answered"
    retrieved_contexts: list[str] = Field(default_factory=list)  # used by evals

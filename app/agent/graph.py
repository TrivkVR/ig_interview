"""LangGraph: guard_input -> agent <-> tools -> finalize -> guard_output."""
from __future__ import annotations

from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.agent.guardian import Guardian
from app.api.documents import document_url
from app.agent.tools import TOOL_NAME, run_search, search_documents
from app.schemas import Category, ChatResponse, Citation, RetrievedChunk

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with questions about safety procedures, maintenance manuals or "
    "quality control standards. Please ask a question within one of these categories."
)
BLOCKED_MESSAGE = (
    "Your message could not be processed because it was flagged by our content guardrails "
    "(toxicity, personal information or an attempt to manipulate the assistant). Please "
    "rephrase and ask about safety procedures, maintenance manuals or quality control standards."
)
BLOCKED_OUTPUT_MESSAGE = (
    "The generated answer was withheld by our content guardrails. Please rephrase your question "
    "about safety procedures, maintenance manuals or quality control standards."
)

SYSTEM_PROMPT = (
    "You are an assistant for floor supervisors in a manufacturing plant. You answer ONLY "
    "questions about safety procedures, maintenance manuals and quality control standards, "
    "using the `search_documents` tool. Always search before answering, choose the correct "
    "category, and base the answer strictly on the retrieved passages (each is tagged with an "
    "[id]). If the passages do not contain the answer, say so. If the question is unrelated to "
    "these three areas, do not call the tool; reply briefly that it is out of scope."
)

FINALIZE_PROMPT = (
    "Now produce the final structured answer. in_scope=false if the user's question is not about "
    "safety procedures, maintenance manuals or quality control standards. cited_chunk_ids must list "
    "the [id] of every retrieved passage actually used."
)

MAX_TOOL_ROUNDS_RECURSION = 16


class AgentOutput(BaseModel):
    answer: str = Field(description="Final answer for the supervisor, grounded in the retrieved passages.")
    in_scope: bool = Field(description="False if the question is not about the three supported categories.")
    cited_chunk_ids: list[int] = Field(default_factory=list, description="[id]s of passages used.")


def _replace(_old: Any, new: Any) -> Any:
    return new


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    user_input: str
    chunks: Annotated[list, _replace]  # list[RetrievedChunk] retrieved this turn
    response: Annotated[dict | None, _replace]  # ChatResponse dump once final


def _fmt_chunk(i: int, c: RetrievedChunk) -> str:
    ct = c.citation
    loc = f"{ct.document_name}, p.{ct.page_number}, {ct.section}" + (f" > {ct.sub_section}" if ct.sub_section else "")
    return f"[{i}] ({loc})\n{c.text}"


def build_graph(
    llm,
    guardian: Guardian,
    checkpointer=None,
    search: Callable[..., Awaitable[list[RetrievedChunk]]] = run_search,
):
    """llm: a LangChain chat model (supports bind_tools / with_structured_output)."""
    agent_llm = llm.bind_tools([search_documents])
    final_llm = llm.with_structured_output(AgentOutput)

    async def guard_input(state: AgentState):
        text = state["user_input"]
        res = await guardian.check_input(text)
        if res.blocked:
            return {"response": ChatResponse(answer=BLOCKED_MESSAGE, status="blocked").model_dump(mode="json")}
        return {"messages": [HumanMessage(text)], "chunks": [], "response": None}

    def after_guard_input(state: AgentState):
        return END if state.get("response") else "agent"

    async def agent(state: AgentState):
        msg = await agent_llm.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [msg]}

    def after_agent(state: AgentState):
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "finalize"

    async def tools_node(state: AgentState):
        chunks = list(state.get("chunks") or [])
        out = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] != TOOL_NAME:
                out.append(ToolMessage(f"Unknown tool {call['name']}", tool_call_id=call["id"]))
                continue
            try:
                found = await search(call["args"]["query"], call["args"]["category"])
            except Exception as e:  # bad category, retriever failure -> let LLM see it
                out.append(ToolMessage(f"Search failed: {e}", tool_call_id=call["id"]))
                continue
            start = len(chunks)
            chunks.extend(found)
            text = "\n\n".join(_fmt_chunk(start + j, c) for j, c in enumerate(found)) or "No results."
            out.append(ToolMessage(text, tool_call_id=call["id"]))
        return {"messages": out, "chunks": chunks}

    async def finalize(state: AgentState):
        chunks: list[RetrievedChunk] = state.get("chunks") or []
        result: AgentOutput = await final_llm.ainvoke(
            [SystemMessage(SYSTEM_PROMPT), *state["messages"], HumanMessage(FINALIZE_PROMPT)]
        )
        if not result.in_scope:
            resp = ChatResponse(answer=OUT_OF_SCOPE_MESSAGE, status="out_of_scope")
            return {"response": resp.model_dump(mode="json"), "messages": [AIMessage(OUT_OF_SCOPE_MESSAGE)]}
        ids = [i for i in dict.fromkeys(result.cited_chunk_ids) if 0 <= i < len(chunks)] or list(range(len(chunks)))
        citations: list[Citation] = []
        for i in ids:
            ct = chunks[i].citation
            ct = ct.model_copy(update={"document_url": document_url(
                ct.document_name, ct.section, ct.sub_section, ct.page_number)})
            if ct not in citations:
                citations.append(ct)
        categories = list(dict.fromkeys(c.citation.category for c in chunks))
        resp = ChatResponse(
            answer=result.answer,
            citations=citations,
            categories=categories,
            status="answered",
            retrieved_contexts=[c.text for c in chunks],
        )
        return {"response": resp.model_dump(mode="json"), "messages": [AIMessage(result.answer)]}

    async def guard_output(state: AgentState):
        resp = state["response"]
        if resp["status"] != "answered":
            return {}
        res = await guardian.check_output(state["user_input"], resp["answer"])
        if res.blocked:
            return {"response": ChatResponse(answer=BLOCKED_OUTPUT_MESSAGE, status="blocked").model_dump(mode="json")}
        return {}

    g = StateGraph(AgentState)
    g.add_node("guard_input", guard_input)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize)
    g.add_node("guard_output", guard_output)
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", after_guard_input, ["agent", END])
    g.add_conditional_edges("agent", after_agent, ["tools", "finalize"])
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "guard_output")
    g.add_edge("guard_output", END)
    return g.compile(checkpointer=checkpointer)

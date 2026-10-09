import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import (BLOCKED_MESSAGE, OUT_OF_SCOPE_MESSAGE, AgentOutput, build_graph)
from app.agent.guardian import GuardResult
from app.schemas import Category, Citation, RetrievedChunk

pytestmark = pytest.mark.asyncio


class FakeGuardian:
    def __init__(self, block_in=False, block_out=False):
        self.block_in, self.block_out = block_in, block_out

    async def check_input(self, t):
        return GuardResult(self.block_in)

    async def check_output(self, u, a):
        return GuardResult(self.block_out)


class FakeLLM:
    def __init__(self, calls_tool, out):
        self.calls_tool, self.out, self.n = calls_tool, out, 0

    def bind_tools(self, tools):
        assert tools[0].name == "search_documents"
        return self

    def with_structured_output(self, schema):
        assert schema is AgentOutput
        return self

    async def ainvoke(self, msgs):
        if isinstance(self.out, AgentOutput) and msgs[-1].content.startswith("Now produce"):
            return self.out
        self.n += 1
        if self.calls_tool and self.n == 1:
            return AIMessage("", tool_calls=[{"name": "search_documents", "id": "c1",
                             "args": {"query": "lockout", "category": "safety_procedures"}}])
        return AIMessage("draft")


CIT = Citation(document_name="LOTO.pdf", page_number=3, section="2", sub_section="2.1",
               category=Category.SAFETY)


async def fake_search(q, c, k=4):
    assert q == "lockout" and Category(c) == Category.SAFETY
    return [RetrievedChunk(text="Apply lock.", citation=CIT),
            RetrievedChunk(text="Other.", citation=CIT.model_copy(update={"page_number": 9}))]


async def run(llm, guardian, msg="q", tid="t"):
    g = build_graph(llm, guardian, InMemorySaver(), fake_search)
    return (await g.ainvoke({"user_input": msg}, {"configurable": {"thread_id": tid}}))["response"]


async def test_answered():
    llm = FakeLLM(True, AgentOutput(answer="Apply lock.", in_scope=True,
                                    cited_chunk_ids=[0]))
    r = await run(llm, FakeGuardian())
    assert r["status"] == "answered" and r["citations"][0]["page_number"] == 3
    assert r["categories"] == ["safety_procedures"]
    assert r["citations"][0]["document_url"].startswith("/documents/view?document=LOTO.pdf")
    assert len(r["citations"]) == 1 and r["retrieved_contexts"] == ["Apply lock.", "Other."]


async def test_out_of_scope():
    r = await run(FakeLLM(False, AgentOutput(answer="x", in_scope=False)), FakeGuardian())
    assert r["status"] == "out_of_scope" and r["answer"] == OUT_OF_SCOPE_MESSAGE


async def test_blocked_input_skips_llm():
    llm = FakeLLM(True, None)
    r = await run(llm, FakeGuardian(block_in=True))
    assert r["status"] == "blocked" and r["answer"] == BLOCKED_MESSAGE and llm.n == 0


async def test_blocked_output():
    llm = FakeLLM(True, AgentOutput(answer="bad", in_scope=True, cited_chunk_ids=[0]))
    assert (await run(llm, FakeGuardian(block_out=True)))["status"] == "blocked"


async def test_api(monkeypatch):
    from fastapi.testclient import TestClient

    from app.agent import service
    from app.api.main import app
    from app.schemas import ChatResponse

    async def fake(m, t):
        return ChatResponse(answer=f"{m}/{t}")

    monkeypatch.setattr(service, "run_agent", fake)
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert c.post("/chat", json={"message": "a", "thread_id": "b"}).json()["answer"] == "a/b"


def test_guardian_config_loads():
    from nemoguardrails import RailsConfig

    from app.agent.guardian import GUARDRAILS_DIR
    cfg = RailsConfig.from_path(str(GUARDRAILS_DIR))
    assert cfg.rails.input.flows == ["self check input"]


async def test_categories_from_all_retrieved_chunks():
    qual = CIT.model_copy(update={"category": Category.QUALITY})
    maint = CIT.model_copy(update={"category": Category.MAINTENANCE})

    async def search(q, c, k=4):
        return [RetrievedChunk(text="a", citation=qual), RetrievedChunk(text="b", citation=CIT),
                RetrievedChunk(text="c", citation=qual), RetrievedChunk(text="d", citation=maint)]

    llm = FakeLLM(True, AgentOutput(answer="x", in_scope=True, cited_chunk_ids=[1]))
    g = build_graph(llm, FakeGuardian(), InMemorySaver(), search)
    r = (await g.ainvoke({"user_input": "q"}, {"configurable": {"thread_id": "t"}}))["response"]
    assert r["categories"] == ["quality_control_standards", "safety_procedures", "maintenance_manuals"]


def test_documents_view():
    from fastapi.testclient import TestClient

    from app.api.documents import document_url, get_index
    from app.api.main import app

    title = "Belt Conveyor System Maintenance Manual MNT-CV10"
    assert title in get_index()
    url = document_url(title, "2. Belt and Drive", "2.2 Belt Tension", 1)
    c = TestClient(app)
    r = c.get(url)
    assert r.status_code == 200
    assert r.text.count("blk highlight") == 1 and 'class="page-marker"' in r.text
    hl = r.text.split('blk highlight"')[1].split("</div>")[0]
    assert "2.2 Belt Tension" in hl
    assert c.get("/documents/view", params={"document": "../../etc/passwd"}).status_code == 404
    assert c.get("/documents/view", params={"document": "nope"}).status_code == 404


def test_turn_spans_are_separate_traces():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from app.tracing import turn_span

    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace.set_tracer_provider(prov)
    with turn_span("thr", "one"):
        pass
    with turn_span("thr", "two"):
        pass
    a, b = exp.get_finished_spans()
    assert a.context.trace_id != b.context.trace_id and a.parent is None and b.parent is None
    assert a.attributes["session.id"] == b.attributes["session.id"] == "thr"

"""Ragas eval harness: response relevancy, faithfulness, context precision.

Usage (from project root):
    uv run python -m evals.run_evals            # real agent + Gemini judge (needs GOOGLE_API_KEY, services up)
    uv run python -m evals.run_evals --mock     # canned responses; judge is skipped unless --judge is given
    uv run python -m evals.run_evals --limit 5

Judge: JUDGE_MODEL (gemini-3.5-flash-lite) via langchain-google-genai, wrapped for Ragas.
Embeddings (needed by answer_relevancy): Gemini embeddings (EVAL_EMBEDDING_MODEL).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import JUDGE_MODEL  # noqa: E402
from app.schemas import Category, ChatResponse, Citation  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "gemini-embedding-001")
METRIC_NAMES = ["answer_relevancy", "faithfulness", "llm_context_precision_with_reference"]


def load_dataset(path: Path = DATASET_PATH, limit: int | None = None) -> list[dict]:
    samples = json.loads(path.read_text())["samples"]
    return samples[:limit] if limit else samples


def mock_response(sample: dict) -> ChatResponse:
    """Canned response: echoes the reference as the answer with one fake context."""
    cat = Category(sample["category"])
    return ChatResponse(
        answer=sample["reference"],
        citations=[Citation(document_name="mock_doc.pdf", page_number=1, section="Mock", category=cat)],
        category=cat,
        status="answered",
        retrieved_contexts=[sample["reference"], "Unrelated filler context."],
    )


async def collect_responses(samples: list[dict], mock: bool, concurrency: int = 4) -> list[ChatResponse]:
    if mock:
        return [mock_response(s) for s in samples]
    from app.agent.service import run_agent  # lazy: built by another workstream
    from app.tracing import setup_tracing

    setup_tracing()  # send eval runs to Phoenix

    sem = asyncio.Semaphore(concurrency)

    async def one(s: dict) -> ChatResponse:
        async with sem:
            return await run_agent(s["question"], f"eval-{uuid.uuid4()}")

    return list(await asyncio.gather(*(one(s) for s in samples)))


def build_rows(samples: list[dict], responses: list[ChatResponse]) -> list[dict]:
    return [
        {
            "user_input": s["question"],
            "response": r.answer,
            "retrieved_contexts": list(r.retrieved_contexts),
            "reference": s["reference"],
        }
        for s, r in zip(samples, responses)
    ]


def _patch_ragas_compat() -> None:
    """ragas 0.4.x imports langchain_community.chat_models.vertexai, removed in langchain-community 0.4.
    Register a stub so the import succeeds (Vertex is not used here)."""
    import importlib.util
    import types

    name = "langchain_community.chat_models.vertexai"
    try:
        if importlib.util.find_spec(name):
            return
    except ModuleNotFoundError:
        pass
    stub = types.ModuleType(name)
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[name] = stub


def score(rows: list[dict]) -> list[dict]:
    """Run the 3 Ragas metrics; returns one dict of scores per row."""
    _patch_ragas_compat()
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GOOGLE_API_KEY is required for the judge LLM / embeddings")
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0, google_api_key=key))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=key))
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[AnswerRelevancy(), Faithfulness(), LLMContextPrecisionWithReference()],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(timeout=120, max_workers=4),
        show_progress=True,
    )
    return [dict(s) for s in result.scores]


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def summarize(per_row: list[dict]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for m in METRIC_NAMES:
        vals = [v for r in per_row if (v := _clean(r.get(m))) is not None]
        out[m] = sum(vals) / len(vals) if vals else None
    return out


def build_results(samples, responses, scores) -> dict:
    items = []
    for s, r, sc in zip(samples, responses, scores):
        items.append({
            "id": s.get("id"),
            "category": s["category"],
            "question": s["question"],
            "reference": s["reference"],
            "answer": r.answer,
            "status": r.status,
            "retrieved_contexts": r.retrieved_contexts,
            "citations": [c.model_dump(mode="json") for c in r.citations],
            "scores": {m: _clean(sc.get(m)) for m in METRIC_NAMES},
        })
    by_cat = {
        c: summarize([i["scores"] for i in items if i["category"] == c])
        for c in sorted({i["category"] for i in items})
    }
    return {"judge_model": JUDGE_MODEL, "embedding_model": EMBEDDING_MODEL,
            "summary": summarize([i["scores"] for i in items]), "by_category": by_cat, "items": items}


def print_summary(results: dict) -> None:
    fmt = lambda v: "  n/a" if v is None else f"{v:.3f}"
    print(f"\nJudge: {results['judge_model']}  (n={len(results['items'])})")
    print("Overall:")
    for m, v in results["summary"].items():
        print(f"  {m:40s} {fmt(v)}")
    for c, s in results["by_category"].items():
        print(f"{c}:")
        for m, v in s.items():
            print(f"  {m:40s} {fmt(v)}")


async def main_async(args) -> dict:
    samples = load_dataset(limit=args.limit)
    responses = await collect_responses(samples, args.mock)
    if args.mock and not args.judge:
        scores = [{} for _ in samples]  # no judge calls in pure mock mode
    else:
        scores = await asyncio.to_thread(score, build_rows(samples, responses))
    results = build_results(samples, responses, scores)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print_summary(results)
    print(f"\nWrote {args.output}")
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="use canned ChatResponses instead of the agent")
    p.add_argument("--judge", action="store_true", help="with --mock, still call the Gemini judge")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=str(RESULTS_PATH))
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()

"""Section-aware ingestion of manufacturing docs into Weaviate.

Usage:  python -m app.rag.ingest [--docs-dir docs] [--recreate]

Document format: markdown with `[[PAGE n]]` markers, `## ` sections and
`### ` sub-sections. Category = name of the parent directory (a Category value).
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.config import (
    EMBEDDING_MODEL,
    WEAVIATE_COLLECTION,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
)
from app.schemas import Category

MAX_CHARS = 1500  # only sub-split chunks longer than this
PAGE_RE = re.compile(r"^\[\[PAGE\s+(\d+)\]\]\s*$")
DEFAULT_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"


@dataclass
class Chunk:
    category: str
    document_name: str
    page_number: int
    section: str
    sub_section: str
    text: str


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split on paragraph, then sentence boundaries, packing up to max_chars."""
    if len(text) <= max_chars:
        return [text]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= max_chars:
            units.append(para)
        else:
            units.extend(re.split(r"(?<=[.!?])\s+", para))
    out, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > max_chars:
            out.append(cur)
            cur = u
        else:
            cur = f"{cur}\n{u}" if cur else u
    if cur:
        out.append(cur)
    return out


def chunk_document(
    markdown: str, document_name: str, category: str, max_chars: int = MAX_CHARS
) -> list[Chunk]:
    """Chunk by section/sub-section; never merges or splits across boundaries
    except sub-splitting a single over-long sub-section."""
    page = 1
    section = ""
    sub = ""
    start_page = 1
    buf: list[str] = []
    chunks: list[Chunk] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        buf.clear()
        if not text or not section:
            return
        for piece in _split_long(text, max_chars):
            chunks.append(Chunk(category, document_name, start_page, section, sub, piece))

    for line in markdown.splitlines():
        m = PAGE_RE.match(line.strip())
        if m:
            page = int(m.group(1))
            continue
        if line.startswith("### "):
            flush()
            sub, start_page = line[4:].strip(), page
        elif line.startswith("## "):
            flush()
            section, sub, start_page = line[3:].strip(), "", page
        elif line.startswith("# "):
            flush()
            continue
        else:
            buf.append(line)
    flush()
    return chunks


def load_chunks(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for cat in Category:
        for path in sorted((docs_dir / cat.value).glob("*.md")):
            title = next(
                (l[2:].strip() for l in path.read_text().splitlines() if l.startswith("# ")),
                path.stem,
            )
            chunks += chunk_document(path.read_text(), title, cat.value)
    return chunks


def embedding_input(c: Chunk) -> str:
    head = f"{c.document_name} > {c.section}" + (f" > {c.sub_section}" if c.sub_section else "")
    return f"{head}\n{c.text}"


# Native output sizes of OpenAI text-embedding-3 models. The same model AND
# dimension are used for ingestion and for queries (both go through embed()).
MODEL_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
EMBED_BATCH_SIZE = 96
MAX_EMBED_CHARS = 24000  # ~6k tokens; far below the 8191-token model limit
_CACHE: dict = {}


def embedding_dimensions(model: str | None = None) -> int:
    """Vector size for the configured model (override via EMBEDDING_DIMENSIONS)."""
    model = model or EMBEDDING_MODEL
    override = os.getenv("EMBEDDING_DIMENSIONS")
    if override:
        return int(override)
    if model not in MODEL_DIMENSIONS:
        raise ValueError(
            f"EMBEDDING_MODEL={model!r} is not a text-embedding-3 model "
            f"({', '.join(MODEL_DIMENSIONS)}); set EMBEDDING_DIMENSIONS to override."
        )
    return MODEL_DIMENSIONS[model]


def embed(texts: list[str], client=None) -> list[list[float]]:
    """Embed texts in batches with text-embedding-3; returns one vector per input."""
    if not texts:
        return []
    if client is None:
        if "openai" not in _CACHE:
            from openai import OpenAI

            _CACHE["openai"] = OpenAI(max_retries=5)
        client = _CACHE["openai"]
    dims = embedding_dimensions()
    # the API rejects empty strings; truncate absurdly long inputs
    inputs = [(t.strip() or " ")[:MAX_EMBED_CHARS] for t in texts]
    vectors: list[list[float]] = []
    for i in range(0, len(inputs), EMBED_BATCH_SIZE):
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=inputs[i : i + EMBED_BATCH_SIZE],
            dimensions=dims,
            encoding_format="float",
        )
        data = sorted(resp.data, key=lambda d: d.index)
        vectors += [d.embedding for d in data]
    if len(vectors) != len(texts) or any(len(v) != dims for v in vectors):
        raise RuntimeError(f"embedding response mismatch (expected {dims}-dim vectors)")
    return vectors


def get_client():
    import weaviate

    return weaviate.connect_to_local(
        host=WEAVIATE_HOST, port=WEAVIATE_HTTP_PORT, grpc_port=WEAVIATE_GRPC_PORT
    )


def collection_description() -> str:
    return f"embedding_model={EMBEDDING_MODEL};dimensions={embedding_dimensions()}"


def ingest(docs_dir: Path = DEFAULT_DOCS_DIR, recreate: bool = True) -> int:
    from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances

    chunks = load_chunks(docs_dir)
    if not chunks:
        raise RuntimeError(f"no documents found under {docs_dir}")
    vectors = embed([embedding_input(c) for c in chunks])
    client = get_client()
    try:
        if recreate and client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.delete(WEAVIATE_COLLECTION)
        if not client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.create(
                WEAVIATE_COLLECTION,
                description=collection_description(),
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE
                ),
                properties=[
                    Property(name="category", data_type=DataType.TEXT,
                             tokenization=Tokenization.FIELD),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="document_name", data_type=DataType.TEXT,
                             tokenization=Tokenization.FIELD),
                    Property(name="section", data_type=DataType.TEXT,
                             tokenization=Tokenization.FIELD),
                    Property(name="sub_section", data_type=DataType.TEXT,
                             tokenization=Tokenization.FIELD),
                    Property(name="text", data_type=DataType.TEXT),
                ],
            )
        col = client.collections.get(WEAVIATE_COLLECTION)
        existing = col.config.get().description
        if existing != collection_description():
            raise RuntimeError(
                f"collection was built with [{existing}] but config is "
                f"[{collection_description()}]; re-run without --keep to recreate it"
            )
        with col.batch.dynamic() as batch:
            for c, v in zip(chunks, vectors):
                batch.add_object(
                    properties={
                        "category": c.category,
                        "page_number": c.page_number,
                        "document_name": c.document_name,
                        "section": c.section,
                        "sub_section": c.sub_section,
                        "text": c.text,
                    },
                    vector=v,
                )
        failed = col.batch.failed_objects
        if failed:
            raise RuntimeError(f"{len(failed)} objects failed to insert: {failed[0].message}")
        return len(chunks)
    finally:
        client.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    ap.add_argument("--keep", action="store_true", help="do not recreate the collection")
    a = ap.parse_args()
    print(f"Ingested {ingest(a.docs_dir, recreate=not a.keep)} chunks")

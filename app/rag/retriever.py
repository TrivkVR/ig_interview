"""Category-filtered retrieval from Weaviate (query embedded with the same
text-embedding-3 model/dimensions as ingestion, via app.rag.ingest.embed)."""
from __future__ import annotations

import atexit
import threading

from app.config import WEAVIATE_COLLECTION
from app.rag.ingest import collection_description, embed, get_client
from app.schemas import Category, Citation, RetrievedChunk

_lock = threading.Lock()
_client = None
_checked = False


def _get_client():
    """Shared Weaviate client, reconnected if it was closed or went away."""
    global _client, _checked
    with _lock:
        if _client is not None:
            try:
                if _client.is_connected() and _client.is_ready():
                    return _client
            except Exception:
                pass
            close()
        _client = get_client()
        _checked = False
        return _client


def close() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(close)


def retrieve(query: str, category: Category, k: int = 4) -> list[RetrievedChunk]:
    global _checked
    from weaviate.classes.query import Filter, MetadataQuery

    category = Category(category)
    if not query.strip() or k < 1:
        return []
    qvec = embed([query])[0]
    col = _get_client().collections.get(WEAVIATE_COLLECTION)
    if not _checked:
        built = col.config.get().description
        if built != collection_description():
            raise RuntimeError(
                f"index embedding config [{built}] != query config "
                f"[{collection_description()}]; re-run `python -m app.rag.ingest`"
            )
        _checked = True
    res = col.query.near_vector(
        near_vector=qvec,
        limit=k,
        filters=Filter.by_property("category").equal(category.value),
        return_metadata=MetadataQuery(distance=True),
    )
    out: list[RetrievedChunk] = []
    for o in res.objects:
        p = o.properties
        dist = o.metadata.distance
        out.append(
            RetrievedChunk(
                text=p["text"],
                score=None if dist is None else 1.0 - dist,
                citation=Citation(
                    document_name=p["document_name"],
                    page_number=int(p["page_number"]),
                    section=p["section"],
                    sub_section=p.get("sub_section") or None,
                    category=category,
                ),
            )
        )
    return out

from app.rag.ingest import DEFAULT_DOCS_DIR, chunk_document, load_chunks

DOC = """# T

[[PAGE 1]]

## 1. A

### 1.1 One
alpha text.

[[PAGE 2]]

### 1.2 Two
beta text.

## 2. B

### 2.1 Three
""" + ("Sentence number here. " * 200)


def test_boundaries_and_pages():
    cs = chunk_document(DOC, "T", "safety_procedures", max_chars=500)
    one, two = cs[0], cs[1]
    assert (one.section, one.sub_section, one.page_number, one.text) == ("1. A", "1.1 One", 1, "alpha text.")
    assert (two.sub_section, two.page_number, two.text) == ("1.2 Two", 2, "beta text.")
    assert "beta" not in one.text and "alpha" not in two.text


def test_long_subsection_split_only_within_it():
    cs = [c for c in chunk_document(DOC, "T", "x", max_chars=500) if c.sub_section == "2.1 Three"]
    assert len(cs) > 1
    assert all(len(c.text) <= 500 and c.section == "2. B" for c in cs)


def test_real_docs_cover_all_categories():
    cs = load_chunks(DEFAULT_DOCS_DIR)
    assert {c.category for c in cs} == {
        "safety_procedures", "maintenance_manuals", "quality_control_standards"}
    assert all(c.section and c.sub_section and c.page_number >= 1 for c in cs)


class _FakeEmb:
    def __init__(self, dims):
        self.dims, self.calls = dims, []
        self.embeddings = self

    def create(self, model, input, dimensions, **kw):
        from types import SimpleNamespace as N

        self.calls.append((model, len(input), dimensions))
        return N(data=[N(index=i, embedding=[0.1] * self.dims) for i in reversed(range(len(input)))])


def test_embed_batches_and_dimensions(monkeypatch):
    import app.rag.ingest as ing

    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    for model, dims in [("text-embedding-3-small", 1536), ("text-embedding-3-large", 3072)]:
        monkeypatch.setattr(ing, "EMBEDDING_MODEL", model)
        fake = _FakeEmb(dims)
        out = ing.embed(["a"] * 200 + [""], client=fake)
        assert len(out) == 201 and all(len(v) == dims for v in out)
        assert [c[1] for c in fake.calls] == [96, 96, 9]
        assert all(c == (model, c[1], dims) for c in fake.calls)


def test_non_embedding3_model_rejected(monkeypatch):
    import pytest
    import app.rag.ingest as ing

    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    with pytest.raises(ValueError):
        ing.embedding_dimensions("text-embedding-ada-002")

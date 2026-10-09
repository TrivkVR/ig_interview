# RAG workstream

- Docs: `docs/<category>/*.md` (category = folder name = `Category` value). Markdown with `[[PAGE n]]` markers, `##` sections, `###` sub-sections.
- Start Weaviate: `docker compose up -d weaviate` (HTTP 8080, gRPC 50051).
- Ingest: `OPENAI_API_KEY=... uv run python -m app.rag.ingest` (recreates collection `ManufacturingDocs`; `--keep` appends, but fails if the embedding config differs).
- Embeddings: OpenAI text-embedding-3 only. `EMBEDDING_MODEL` (default `text-embedding-3-small`, 1536 dims; `text-embedding-3-large` = 3072 dims) is used for both ingestion and queries through `app.rag.ingest.embed`, with `dimensions` passed explicitly, batches of 96. The model/dimensions are recorded in the collection description; ingest and retrieve refuse to run on a mismatch. Switching models requires re-ingesting.
- Chunking: section-aware; one chunk per sub-section (or section intro), never crossing section boundaries, sub-split by paragraph/sentence only if over 1500 chars. Page = page where the chunk's heading starts. Embedded text is prefixed with `document > section > sub-section`.
- Collection: cosine HNSW, no vectorizer, properties category/page_number/document_name/section/sub_section/text (metadata fields use exact `field` tokenization).
- Retrieve: `retrieve(query, Category.SAFETY, k=4) -> list[RetrievedChunk]`, filtered by category, score = 1 - cosine distance. Reuses one shared Weaviate client (reconnects if closed).
- Tests: `uv run pytest app/rag` (no live services needed).

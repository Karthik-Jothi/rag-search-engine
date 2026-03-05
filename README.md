# RAG Search Engine

A world-class hybrid search engine built on PostgreSQL for Retrieval-Augmented Generation (RAG) systems.

## Architecture

Multi-signal hybrid search combining:
- **Semantic Search** — pgvector HNSW with text-embedding-3-small (1536D)
- **BM25 Keyword Search** — Custom BM25 scorer via plv8 on tsvector/GIN
- **Fuzzy Matching** — pg_trgm trigram similarity for typo tolerance
- **Cross-Encoder Reranking** — Vertex AI via google_ml_integration

Results are fused using Reciprocal Rank Fusion (RRF) with adaptive weights.

## Infrastructure

- **Database**: GCP Cloud SQL for PostgreSQL 14.20
- **Vector Index**: pgvector 0.8.1 (HNSW)
- **Embedding Model**: OpenAI text-embedding-3-small (1536D)
- **Extensions**: pgvector, pg_trgm, plv8, fuzzystrmatch, unaccent, google_ml_integration

## Project Structure

```
sql/
  schema/         — Table definitions, indexes, extensions
  functions/      — PL/pgSQL and plv8 functions
  migrations/     — Versioned schema migrations
docs/
  design/         — Architecture and design documents
  research/       — Search domain research and analysis
tests/
  search_quality/ — Search quality test cases (12 failure modes)
config/           — Configuration files and dictionaries
```

## Development

All development happens on feature branches. The `main` branch is protected — changes require pull requests.

## Phases

| Phase | Description | Quality Target |
|-------|-------------|---------------|
| 1 | Cloud SQL native (tsvector + pgvector + pg_trgm + plv8 BM25 + RRF) | ~80% of Elasticsearch |
| 2 | Tantivy sidecar on Cloud Run for true BM25 | ~90% of Elasticsearch |
| 3 | Self-hosted PostgreSQL with ParadeDB + pgvectorscale + pgml | ~98% of Elasticsearch |

## License

Proprietary

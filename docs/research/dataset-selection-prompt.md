# Prompt: Evaluation Dataset Selection for a Hybrid RAG Search Engine

Use the following prompt with another LLM to get independent dataset recommendations.

---

## Prompt

```
I am building a hybrid search engine on PostgreSQL for RAG (Retrieval-Augmented
Generation). I need help selecting the right evaluation dataset(s) to measure my
retrieval quality at each stage of development.

### My System

- **Search signals combined via Reciprocal Rank Fusion (RRF):**
  1. Semantic search — pgvector HNSW index, OpenAI text-embedding-3-small (1536D)
  2. BM25 keyword search — custom scorer via plv8 on tsvector/GIN indexes
  3. Fuzzy matching — pg_trgm trigram similarity for typo tolerance
  4. Cross-encoder reranking — Vertex AI via google_ml_integration

- **Infrastructure:** GCP Cloud SQL for PostgreSQL 14.20

- **Target domain:** Multi-tenant helpdesk / technical support documents

- **Development phases:**
  - Phase 1: Cloud SQL native (tsvector + pgvector + pg_trgm + plv8 BM25 + RRF) — targeting ~80% of Elasticsearch quality
  - Phase 2: Tantivy sidecar on Cloud Run for true BM25 — targeting ~90%
  - Phase 3: Self-hosted PostgreSQL with ParadeDB + pgvectorscale + pgml — targeting ~98%

- **Current stage:** Pre-implementation. SQL schemas are designed but not yet deployed. No working search endpoint yet.

### Evaluation Scripts I Already Have

I have a Python evaluation script using `pytrec_eval` that computes Recall@{1,3,5}
and NDCG@{1,3,5} against BEIR-format qrels files (TSV: query-id, corpus-id,
relevance-score). It produces per-collection scores and weighted averages. Any
dataset I use needs to work with this format or be convertible to it.

### Dataset I Already Have (and its problem)

I have the IBM MT-RAG benchmark — 4 domains (ClapNQ/Wikipedia, Cloud/technical docs,
FiQA/finance, Govt/government), totaling ~78K documents and ~366K passages.

**The annotation problem:** MT-RAG relevance labels were created by running IBM's
ELSER retriever, then having humans label only the documents ELSER surfaced. This
means 99%+ of each corpus was never reviewed by annotators. If my hybrid search finds
a genuinely relevant document that ELSER never retrieved, it gets scored as a "miss"
(false negative). The labels are coupled to ELSER's recall boundary, not to true
relevance.

This makes MT-RAG unsuitable for absolute quality measurement. I can only use it to
check whether I at least match what ELSER found — it cannot tell me how good my
system actually is.

### What I Need From You

Recommend evaluation datasets that satisfy these requirements:

1. **Annotation quality:** Relevance judgments must NOT be derived from a single
   retriever's output. I need labels where annotators reviewed a broad pool of
   candidates (from multiple diverse systems) or where the corpus is small enough
   that coverage is near-exhaustive. The key question: if my system retrieves a
   relevant document, will the labels credit it?

2. **Format compatibility:** Must be in BEIR format or easily convertible to it
   (corpus JSONL + queries JSONL + qrels TSV). Available via standard tools like
   `ir_datasets`, HuggingFace, or direct download.

3. **Phased evaluation needs:**
   - **Sanity check dataset** (Phase 1 prototype): Small corpus, fast iteration,
     catches wiring bugs and broken indexes. Must have near-zero false negative risk.
   - **Ranking quality dataset** (Phase 1-2 tuning): Large corpus, tests whether
     hybrid search beats individual signals, validates BM25 + semantic + reranking.
     Must have deep, trustworthy relevance judgments.
   - **Domain stress test dataset** (Phase 2-3): Tests handling of technical
     vocabulary, specialized terminology, domain-specific jargon. Must have graded
     relevance (not just binary).
   - **Helpdesk-specific patterns** (all phases): Typo tolerance, error code lookup,
     negation queries ("not working"), very short queries (2-3 words), multi-tenant
     filtering. No public dataset likely covers this — advise on how to construct it.

4. **Scale:** At least one dataset with 1M+ passages to validate performance at
   realistic scale.

5. **Measurable baselines:** Published baseline results from known retrievers (BM25,
   dense retrievers, hybrid systems) so I can contextualize my scores.

### What I Want in Your Response

For each dataset you recommend:
- Name, corpus size, query count, domain
- How annotations were created (pooling method, number of systems pooled, annotation
  scale)
- False negative risk assessment — what fraction of truly relevant documents might
  be missing from the labels?
- Published baselines for BM25 and at least one dense retriever
- Which of my evaluation phases it serves
- How to obtain it and convert to BEIR format
- Any known limitations or gotchas

Also tell me:
- Which datasets to AVOID and why (specifically datasets where annotations are
  coupled to a single retriever, like MS MARCO dev set)
- Whether any recent (2024-2025) datasets address the false negative problem better
  than older benchmarks
- How to construct a custom test set for helpdesk-specific patterns (typo tolerance,
  error codes, negation, short queries)
```

---

## Why This Prompt Is Structured This Way

1. **System context first** — the LLM needs to understand what kind of retriever
   we're building to recommend appropriate datasets. A BM25-only system needs
   different validation than a hybrid system.

2. **The annotation problem is stated explicitly** — most LLMs will default to
   recommending MS MARCO or Natural Questions. By stating the false negative problem
   upfront, we steer toward datasets with better annotation coverage.

3. **Phased needs** — prevents the LLM from recommending a single dataset. We need
   different datasets for different maturity stages.

4. **Format constraint** — avoids recommendations that would require major conversion
   work. BEIR format is our standard.

5. **"What to avoid"** — forces the LLM to explicitly reason about dataset
   limitations rather than just listing popular options.

6. **Baselines requested** — raw scores are meaningless without context. We need to
   know what BM25 scores on the same dataset to interpret our numbers.

## How to Validate the Response

Cross-check the LLM's recommendations against these known facts:

- **MS MARCO dev** has ~57.6% false negative rate (single Bing result + single annotator). If the LLM recommends it without caveats, the response is unreliable.
- **Natural Questions** labels come from a single Google search page shown to one annotator. Same problem.
- **HotpotQA** has exactly 2 documents labeled per query. Massively incomplete.
- **TREC-DL 2019/2020** used HiCAL active learning + 100+ system pooling. This is the gold standard for large-scale annotation quality.
- **SciFact** has 5,183 abstracts with citation-based ground truth. Near-zero false negatives.
- **D-MERIT (EMNLP 2024)** was purpose-built to study the false negative problem in retrieval evaluation.

If the LLM's response aligns with these facts, it's likely trustworthy. If it
contradicts them (e.g., recommends MS MARCO dev as reliable), treat the full
response with skepticism.

# How Real-World RAG Systems Are Evaluated

A comprehensive guide to the frameworks, metrics, datasets, and testing patterns used
in production to verify that RAG retrieval and generation pipelines are working correctly.

---

## 1. Evaluation Frameworks People Actually Use

### RAGAS (Retrieval Augmented Generation Assessment)

The most widely adopted open-source framework for RAG evaluation. Key property:
**reference-free** — it does not require ground truth answers for most metrics.

| Metric | What It Measures | How It Works |
|--------|-----------------|--------------|
| Context Precision | Are the retrieved chunks ranked by relevance? | LLM checks if each chunk is needed to answer the query; computes mean precision@k |
| Context Recall | Does the retrieved context cover all needed information? | LLM decomposes the ground truth into claims, checks which are supported by context |
| Faithfulness | Is the answer grounded in the retrieved context? | LLM extracts claims from the answer, verifies each against the context |
| Answer Relevancy | Does the answer actually address the question? | LLM generates questions the answer could address, compares to original query via embedding similarity |

**We already use RAGAS.** Our `judge_wrapper.py` integrates ragas==0.3.3 with all four
metrics above.

- Open-source Python library
- Works with any LLM as the evaluator
- Can generate synthetic test sets for evaluation at scale
- [RAGAS Paper (2023)](https://arxiv.org/abs/2309.15217)
- [RAGAS Documentation](https://docs.ragas.io/)

### DeepEval

A **test-driven development (TDD) framework** for LLM evaluation. Think of it as
pytest for RAG.

| Feature | Detail |
|---------|--------|
| Test style | Pytest-compatible — write unit tests for RAG outputs |
| Metrics | 14+ built-in (faithfulness, answer relevancy, hallucination, bias, toxicity, etc.) |
| Self-explaining | Each metric tells you *why* the score cannot be higher |
| CI/CD | First-class support — fail builds if quality SLOs drop below thresholds |
| Synthetic data | Built-in generation for scalable test suites |

**When to use:** Best for teams that want regression testing in CI/CD pipelines.
Write test cases like unit tests, run them on every PR.

- [DeepEval Documentation](https://deepeval.com/docs)
- [DeepEval RAG Evaluation Guide](https://deepeval.com/guides/guides-rag-evaluation)

### TruLens

Introduces the **RAG Triad** concept — three dimensions that together cover RAG quality:

1. **Context Relevance** — Is the retrieved context relevant to the query?
2. **Groundedness** — Is the answer supported by the context?
3. **Answer Relevance** — Does the answer address the query?

Tightly integrated with LangChain and LlamaIndex. Uses "feedback functions" to
programmatically evaluate each step of the execution flow.

- [TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)

### Arize Phoenix

Open-source **AI observability platform** built on the OpenTelemetry standard.

| Feature | Detail |
|---------|--------|
| Tracing | Full execution traces with latency, token usage, retrieval results at each step |
| Evaluation | LLM-based evaluation with versioned datasets |
| Integration | Auto-instrumentation for LlamaIndex, LangChain, Haystack, DSPy |
| Lock-in | None — OpenTelemetry standard means portable traces |
| GitHub | 7,800+ stars |

**When to use:** Teams that need production observability alongside evaluation, or
that use multiple frameworks and want a single pane of glass.

- [Arize Phoenix GitHub](https://github.com/Arize-ai/phoenix)
- [Phoenix RAG Evaluation Cookbook](https://arize.com/docs/phoenix/cookbook/evaluation/evaluate-rag)

### LangSmith

LangChain's proprietary observability and evaluation platform.

- Auto-instrumentation with a single environment variable
- Visual debugging: drill into embedding models, vector search results, chunk ranking
- Shows nested execution steps with full precision

**When to use:** Teams already committed to the LangChain ecosystem.

### Langfuse

Open-source, self-hostable alternative to LangSmith. Comparable features with no
vendor lock-in. Good for privacy-sensitive deployments.

- [Langfuse Documentation](https://langfuse.com/)

### Framework Selection Guide

| If you need... | Use |
|----------------|-----|
| Quick reference-free evaluation | RAGAS |
| CI/CD regression testing | DeepEval |
| LangChain/LlamaIndex observability | TruLens or LangSmith |
| Multi-framework observability | Arize Phoenix |
| Self-hosted, privacy-sensitive | Langfuse |
| Already have RAGAS (like us) | Add DeepEval for CI/CD gates |

---

## 2. Standard Metrics

### Retrieval Metrics

These measure whether the search engine returns the right documents in the right order.

| Metric | Formula (simplified) | What It Tells You | We Compute It? |
|--------|---------------------|-------------------|---------------|
| **Recall@K** | (relevant docs in top-K) / (total relevant docs) | Coverage — did we find everything? | Yes (`run_retrieval_eval.py` via pytrec_eval) |
| **NDCG@K** | Normalized discounted cumulative gain | Ranking quality — are the best results first? | Yes (`run_retrieval_eval.py` via pytrec_eval) |
| **Precision@K** | (relevant docs in top-K) / K | Noise — how much junk is in the results? | Available in pytrec_eval but commented out |
| **MAP** | Mean of average precision across queries | Rank-aware aggregate quality | Available in pytrec_eval but commented out |
| **MRR** | 1 / (rank of first relevant result) | How quickly do we find *something* relevant? | Not yet computed |
| **Context Precision** | RAGAS: mean precision@k for each chunk | Rank quality via LLM judgment | Yes (via judge_wrapper.py / RAGAS) |
| **Context Recall** | RAGAS: claims supported / total claims | Coverage via LLM judgment | Yes (via judge_wrapper.py / RAGAS) |

**Our current setup** computes NDCG@{1,3,5} and Recall@{1,3,5} per domain (ClapNQ,
Cloud, FiQA, Govt) with weighted averages. To enable MAP, Precision, and MRR,
uncomment the evaluator line in `run_retrieval_eval.py:37`.

### Generation / Answer Quality Metrics

These measure whether the LLM produces good answers from the retrieved context.

| Metric | Type | What It Measures | We Compute It? |
|--------|------|-----------------|---------------|
| **Faithfulness** | LLM-judged | Is the answer grounded in retrieved context? | Yes (RAGAS + RADBench) |
| **Answer Relevancy** | LLM-judged | Does the answer address the question? | Yes (RAGAS) |
| **Hallucination** | LLM-judged | Are there claims not supported by context? | Partially (faithfulness inverse) |
| **ROUGE-L** | Algorithmic | N-gram overlap with reference answer | Yes (`run_algorithmic.py`) |
| **BERTScore** | Algorithmic | Semantic similarity with reference | Yes (`run_algorithmic.py`) |
| **Token Recall** | Algorithmic | Word-level coverage of reference | Yes (`run_algorithmic.py`) |
| **Extractiveness** | Algorithmic | How much of the answer is copied from context | Yes (`run_algorithmic.py`) |
| **RADBench Aggregate** | Composite | 3-way harmonic mean: recall x rouge x extractiveness | Yes (`run_algorithmic.py`) |

### End-to-End and Online Metrics

These are measured in production, not in offline evaluation.

| Metric | When to Use |
|--------|-------------|
| **Answer Correctness** | Offline — compare against gold standard answers |
| **Latency (p50, p95, p99)** | Always — query response time |
| **Cost per Query** | Always — token efficiency, API costs |
| **Click-through Rate** | Online — users click cited sources |
| **Thumbs Up/Down** | Online — explicit user satisfaction signal |
| **Session Duration** | Online — engagement proxy |
| **Copy-Paste Rate** | Online — answer utility proxy |

---

## 3. Benchmark Datasets for Validation

### What We Already Have

The IBM MT-RAG benchmark provides 4 domains in BEIR format with qrels:

| Domain | Corpus Size | Passages | Queries | Domain Type |
|--------|------------|----------|---------|-------------|
| ClapNQ | 4,293 docs | 183,408 | Multi-turn | Wikipedia |
| Cloud | 57,638 docs | 61,022 | Multi-turn | IBM technical docs |
| FiQA | 7,661 docs | 49,607 | Multi-turn | Financial QA |
| Govt | 8,578 docs | 72,422 | Multi-turn | Government docs |

Baseline results (query rewrite, best per-row):

| Retriever | Recall@5 | nDCG@10 |
|-----------|---------|---------|
| BM25 | 0.25 | 0.25 |
| BGE-base-1.5 | 0.37 | 0.38 |
| ELSER | 0.52 | 0.54 |

**Limitation:** MT-RAG annotations are ELSER-derived — a "miss" may be a labeling gap,
not a real failure. See `evaluation-dataset-annotation-quality.md` for the full analysis.

### Recommended Additional Datasets

Based on our annotation quality research, these datasets have the most trustworthy
relevance judgments:

| Dataset | Why Use It | Corpus | Queries | Annotation Quality |
|---------|-----------|--------|---------|-------------------|
| **TREC-DL 2019/2020** | Gold standard for retrieval evaluation | 8.8M passages | 43+54 | HiCAL active learning + 100+ system pooling |
| **SciFact** | Quick sanity check, tiny corpus | 5,183 abstracts | 300 | Expert-annotated via citation links |
| **TREC-COVID** | Domain stress test, biomedical | 171K docs | 50 | 69K+ manual judgments, >90% top-10 coverage |

Available via [ir_datasets](https://ir-datasets.com/) and convertible to BEIR format.
SciFact and TREC-COVID are already in the standard [BEIR suite](https://github.com/beir-cellar/beir).

### RAG-Specific Benchmarks

These evaluate the full RAG pipeline (retrieval + generation), not just retrieval:

| Benchmark | Size | Focus | Key Feature |
|-----------|------|-------|-------------|
| **RAGBench** | 100K examples | 5 industry domains | TRACe evaluation framework — scores context relevance, utilization, completeness |
| **ARES** | Varies | General RAG | Automated, reference-free — uses LLM confidence + few-shot PPI for statistical estimation |
| **DRAGONBall** | Varies | Finance, law, medical | Domain-specific RAG evaluation across 3 specialized domains |
| **RGB** | Multiple | General | Tests knowledge acquisition from retrieved context |
| **BEIR 2.0** (2025) | 18+ datasets | Zero-shot retrieval | Updated from original BEIR; includes CodeSearchNet-RAG |
| **MTEB** | 58 datasets, 112 languages | Embedding quality | 500+ tasks including retrieval; broader than BEIR |

Sources:
- [RAGBench Paper](https://arxiv.org/html/2407.11005v1)
- [ARES Paper](https://www.semanticscholar.org/paper/ARES:-An-Automated-Evaluation-Framework-for-Systems-Saad-Falcon-Khattab/4df2b1e7d54fe5ad81dc2ed6774b93ef7891b3c8)
- [BEIR 2.0](https://app.ailog.fr/en/blog/news/beir-benchmark-update)
- [MTEB Overview](https://prasun-mishra.medium.com/massive-text-embedding-benchmark-mteb-helps-to-find-optimal-embedding-for-your-rag-llm-use-case-1af15eab40e1)
- [RAG Benchmarks Survey (Evidently AI)](https://www.evidentlyai.com/blog/rag-benchmarks)
- [RAG Evaluation Survey (2024)](https://arxiv.org/html/2405.07437v2)

---

## 4. Real-World Testing Patterns

### Component Isolation

Test each pipeline stage independently. This is the most important pattern because
end-to-end failures are hard to diagnose.

```
Query → [Retrieval] → [Reranking] → [Context Selection] → [Generation] → Answer
          ↓               ↓                ↓                   ↓
       Recall@K         MRR/NDCG      Context Precision   Faithfulness
       NDCG@K           before/after   Context Recall     Answer Relevancy
```

**We already do this.** Our `run_retrieval_eval.py` evaluates retrieval independently,
and `run_generation_eval.py` evaluates generation independently.

### LLM-as-Judge

Use a capable LLM (GPT-4, Claude) to automatically evaluate outputs at scale.

| Property | Value |
|----------|-------|
| Agreement with humans | ~80% (comparable to human-to-human agreement) |
| Cost savings | 500x-5000x cheaper than human review |
| Speed | Seconds per evaluation vs. minutes for humans |
| Best model for judging | GPT-4o or Claude for general use |
| Hallucination detection | FaithJudge (2025): 84% balanced accuracy, 82% F1-macro |

**Key considerations:**
- Judge model should be more capable than the model being evaluated
- Combine with targeted human review for edge cases, safety-critical queries, and
  subtle quality issues (tone, clarity, ambiguity)
- Use structured rubrics to reduce judge variance

**Enterprise RAG evaluation** (2025-2026 best practice): Case-Aware LLM-as-Judge
evaluates 8 operationally grounded metrics per turn — retrieval quality, grounding
fidelity, answer utility, precision integrity, and workflow alignment.

Sources:
- [LLM as a Judge Guide (2026)](https://labelyourdata.com/articles/llm-as-a-judge)
- [FaithJudge: Benchmarking Hallucination Detection](https://arxiv.org/abs/2505.04847)
- [Case-Aware LLM-as-Judge for Enterprise RAG](https://arxiv.org/html/2602.20379v1)
- [Mistral: Evaluating RAG with LLM as Judge](https://mistral.ai/news/llm-as-rag-judge)

### Regression Testing in CI/CD

The DeepEval pattern: write evaluation test cases like unit tests, run them on every PR.

**How it works:**
1. Define test cases with query, expected context, expected answer
2. Set SLO thresholds (e.g., faithfulness >= 0.8, recall@5 >= 0.4)
3. Run evaluation in CI pipeline
4. Fail the build if any threshold is breached

**Example workflow:**
```
PR opened
  → Run retrieval on 50 representative + 30 edge-case queries
  → Compute NDCG@5, Recall@5, MRR
  → Compare against baseline (ELSER numbers: R@5=0.52, nDCG@10=0.54)
  → Fail if any metric drops >5% from baseline
  → Store results in time-series database for trend analysis
```

**Key metrics to gate on:**
- Retrieval: Recall@5 and NDCG@5 (already computed)
- Generation: Faithfulness score (already computed via RAGAS)
- Latency: p95 response time
- Regression in any dimension triggers alerts

Sources:
- [DeepEval CI/CD Guide](https://www.confident-ai.com/blog/how-to-evaluate-rag-applications-in-ci-cd-pipelines-with-deepeval)
- [Production RAG CI/CD Quality Gates](https://dextralabs.com/blog/production-rag-in-2025-evaluation-cicd-observability/)

### A/B Testing

For production systems with real traffic.

**Experimental design:**
1. Start with a SMART hypothesis
   - Example: "Replacing BM25 with hybrid (BM25+pgvector) retriever increases
     click-through rate on sources by 10% without harming faithfulness"
2. Random assignment via consistent hashing by user ID
3. Run for statistically significant duration
4. Measure online metrics: click-through, thumbs up/down, session duration

**What companies actually track:**
- Pinecone: Precision@K, Recall@K, MRR, NDCG, faithfulness, latency, cost per query,
  version tags. Stores structured run artifacts (JSON/parquet).
- Weaviate: Component-level metrics at each pipeline stage.
- Qdrant: Average query latency on 1M vector benchmarks.

Sources:
- [Pinecone RAG Evaluation](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [A/B Testing for RAG](https://apxml.com/courses/large-scale-distributed-rag/chapter-5-orchestration-operationalization-large-scale-rag/ab-testing-experimentation-rag)
- [Qdrant RAG Evaluation Guide](https://qdrant.tech/blog/rag-evaluation-guide/)

---

## 5. How This Applies to Our System

### Pipeline Component → Evaluation Mapping

| Our Component | What to Evaluate | Metrics | Tool | Status |
|--------------|-----------------|---------|------|--------|
| pgvector HNSW (semantic) | Semantic retrieval quality | Recall@K, NDCG@K | pytrec_eval | Have script, need to run on our index |
| BM25 via plv8 (keyword) | Keyword retrieval quality | Recall@K, NDCG@K | pytrec_eval | Have script, compare vs ELSER baseline |
| pg_trgm (fuzzy) | Typo tolerance, partial match | Custom test suite (12 failure modes) | Manual test cases | `tests/search_quality/` exists but empty |
| RRF fusion (hybrid) | Does fusion beat individual signals? | NDCG@K delta vs individual | pytrec_eval | Need to run each signal independently then fused |
| Cross-encoder reranking | Reranking lift | MRR, NDCG@K before vs after reranking | pytrec_eval | Need before/after comparison |
| Full RAG pipeline | End-to-end answer quality | Faithfulness, answer relevancy | RAGAS (already integrated) | Have scripts, need our own generation tasks |

### Recommended Evaluation Strategy

**Phase 1 — Retrieval Validation (offline, no LLM needed)**
- Run our search engine against SciFact and TREC-COVID (BEIR format, trustworthy annotations)
- Compare Recall@5 and NDCG@5 against published baselines
- Test each signal independently: pgvector alone, BM25 alone, pg_trgm alone, then fused
- Measure reranking lift: NDCG before vs after cross-encoder

**Phase 2 — Generation Validation (needs LLM judge)**
- Use RAGAS (already integrated) for faithfulness and answer relevancy
- Run on MT-RAG generation tasks (842 tasks with gold passages available)
- Compare our results against published MT-RAG baselines

**Phase 3 — Regression Testing (CI/CD)**
- Implement DeepEval-style test cases in `tests/search_quality/`
- 50 representative queries + 30 edge cases (typos, negation, error codes, short queries)
- Gate on: Recall@5 >= 0.40, NDCG@5 >= 0.35, faithfulness >= 0.75
- Run on every PR that touches search logic

**Phase 4 — Production Monitoring (online)**
- Instrument with Arize Phoenix or Langfuse for observability
- Track latency, token usage, retrieval relevance per query
- Collect thumbs up/down signals for continuous evaluation
- A/B test major changes (new embedding model, new reranker, fusion weight changes)

### Gap Analysis: What We Have vs. What We Need

| Capability | Status | Gap |
|-----------|--------|-----|
| Retrieval metrics (NDCG, Recall) | Have script (`run_retrieval_eval.py`) | Need to run on our index |
| Generation metrics (RAGAS) | Have script (`judge_wrapper.py`) | Need to run on our generation output |
| Algorithmic metrics (ROUGE, BERTScore) | Have script (`run_algorithmic.py`) | Need to run on our output |
| MRR metric | Not computed | Uncomment in pytrec_eval config |
| Reranking evaluation | No before/after comparison | Need to log pre-rerank and post-rerank rankings |
| Typo/edge case tests | Empty placeholder | Need to implement 12 failure mode test cases |
| CI/CD quality gates | Not implemented | Need DeepEval or custom pytest harness |
| Production observability | Not implemented | Need Phoenix/Langfuse integration |
| A/B testing | Not implemented | Need traffic splitting infrastructure |

---

## References

### Frameworks
- [RAGAS Paper (2023)](https://arxiv.org/abs/2309.15217)
- [RAGAS Documentation](https://docs.ragas.io/)
- [DeepEval Documentation](https://deepeval.com/docs)
- [TruLens RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- [Arize Phoenix GitHub](https://github.com/Arize-ai/phoenix)
- [Langfuse](https://langfuse.com/)

### Metrics and Methodology
- [Evaluation Metrics for RAG (GeeksforGeeks)](https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/)
- [Retrieval Evaluation Metrics (Weaviate)](https://weaviate.io/blog/retrieval-evaluation-metrics)
- [RAG Evaluation Metrics (Patronus AI)](https://www.patronus.ai/llm-testing/rag-evaluation-metrics)
- [NDCG Deep Dive (Towards Data Science)](https://towardsdatascience.com/how-to-evaluate-retrieval-quality-in-rag-pipelines-part-3-dcgk-and-ndcgk/)

### Benchmarks
- [BEIR Paper (NeurIPS 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)
- [BEIR GitHub](https://github.com/beir-cellar/beir)
- [RAGBench Paper](https://arxiv.org/html/2407.11005v1)
- [RAG Evaluation Survey (2024)](https://arxiv.org/html/2405.07437v2)
- [RAG Benchmarks Survey (Evidently AI)](https://www.evidentlyai.com/blog/rag-benchmarks)
- [BenchmarkQED (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/)

### Testing Patterns
- [DeepEval CI/CD Guide](https://www.confident-ai.com/blog/how-to-evaluate-rag-applications-in-ci-cd-pipelines-with-deepeval)
- [Production RAG CI/CD (DextraLabs)](https://dextralabs.com/blog/production-rag-in-2025-evaluation-cicd-observability/)
- [Pinecone RAG Evaluation](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [RAG Evaluation Best Practices (Evidently AI)](https://www.evidentlyai.com/llm-guide/rag-evaluation)
- [Google Cloud RAG Best Practices](https://cloud.google.com/blog/products/ai-machine-learning/optimizing-rag-retrieval)
- [Qdrant RAG Evaluation Guide](https://qdrant.tech/blog/rag-evaluation-guide/)

### LLM-as-Judge
- [FaithJudge (2025)](https://arxiv.org/abs/2505.04847)
- [Case-Aware LLM-as-Judge for Enterprise RAG](https://arxiv.org/html/2602.20379v1)
- [Mistral: LLM as RAG Judge](https://mistral.ai/news/llm-as-rag-judge)
- [LLM as a Judge Guide (2026)](https://labelyourdata.com/articles/llm-as-a-judge)

### Related Internal Docs
- [Evaluation Dataset Annotation Quality Analysis](evaluation-dataset-annotation-quality.md) — Which datasets have trustworthy relevance judgments

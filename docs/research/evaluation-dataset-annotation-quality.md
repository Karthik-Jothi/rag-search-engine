# Evaluation Dataset Annotation Quality Analysis

## Problem Statement

When evaluating a RAG retrieval pipeline, a "miss" (retrieved document not in the
ground truth) can mean two things:

1. **Real failure**: The pipeline retrieved an irrelevant document.
2. **Labeling gap**: The pipeline found a genuinely relevant document that was never
   annotated as relevant.

Datasets with incomplete relevance judgments conflate these two cases, making them
unreliable for detecting pipeline bugs. This document analyzes which datasets have
sufficiently exhaustive annotations to be safe for pipeline validation.

## Key Insight

Truly exhaustive relevance judgments (every query-document pair reviewed by a human)
only exist for very small collections like the Cranfield collection (~1,400 documents).
All large-scale modern benchmarks rely on incomplete judgments via pooling or sparse
annotation, which inherently introduces false negatives.

The critical question is: **how incomplete are the judgments, and does that matter for
our use case?**

---

## Dataset Analysis

### TIER 1: Near-Exhaustive -- Safe for Detecting Pipeline Bugs

#### TREC Deep Learning 2019/2020 (Passage Ranking)

| Property | Detail |
|---|---|
| Corpus | 8.8M passages (MS MARCO corpus) |
| Queries | 43 (2019), 54 (2020) |
| Annotation method | Deep pooling from 100+ diverse systems + HiCAL active learning |
| Relevance scale | 4-level: Not Relevant (0), Related (1), Highly Relevant (2), Perfectly Relevant (3) |
| Avg relevant/query | Dozens to hundreds |
| Domain | Web (general) |

**Why safe**: The HiCAL process actively hunted for relevant passages beyond what any
single retriever found. Pooling came from BM25, dense, hybrid, and neural retrievers.
Leave-one-out simulations confirm stable system rankings (reusability validated in
SIGIR 2021). The main limitation is the small query count (43-54).

#### SciFact

| Property | Detail |
|---|---|
| Corpus | 5,183 abstracts |
| Queries | 300 test claims |
| Annotation method | Expert-annotated via citation links |
| Relevance scale | Binary (SUPPORTS/REFUTES) with sentence-level rationales |
| Avg relevant/query | ~1.1 |
| Domain | Biomedical/scientific |
| In BEIR | Yes |

**Why safe**: Claims were generated FROM citation sentences, so the cited abstract is
ground truth by construction. The tiny corpus (5K docs) means other relevant but
unlabeled abstracts are rare.

#### D-MERIT (EMNLP 2024)

| Property | Detail |
|---|---|
| Corpus | Wikipedia passages |
| Queries | Group-membership queries (e.g., "journals about linguistics") |
| Annotation method | Leverages Wikipedia's structured category/list system + GPT-4 filtering validated against human judgments |
| Relevance scale | Binary |
| Domain | Wikipedia (general knowledge) |
| In BEIR | No (custom format) |

**Why safe**: Purpose-built to collect ALL relevant passages. The authors demonstrated
that partially-annotated datasets give misleading system rankings. This is the gold
standard for studying whether a miss is truly a miss.

---

### TIER 2: Pooling-Based -- Usable with Caution

#### TREC-COVID

| Property | Detail |
|---|---|
| Corpus | 171K documents (CORD-19) |
| Queries | 50 |
| Annotation method | Multi-round pooling across 5 TREC rounds; 69K+ manual judgments |
| Relevance scale | 3-level graded |
| Avg relevant/query | ~493.5 |
| Domain | COVID-19 biomedical |
| In BEIR | Yes |

**Verdict**: Best labeling coverage in top-10 among BEIR datasets (>90%), but BEIR
authors found that filling unjudged "holes" changed model rankings, particularly
helping neural models.

#### NFCorpus

| Property | Detail |
|---|---|
| Corpus | 3,633 medical documents |
| Queries | 323 test queries |
| Annotation method | Automatically extracted from NutritionFacts.org hyperlink structure |
| Relevance scale | 3-level graded |
| Avg relevant/query | ~38.2 |
| Domain | Medical/nutrition |
| In BEIR | Yes |

**Verdict**: Silver-standard labels from website link structure, not human judgment.
Small corpus helps, but false negatives are expected.

---

### TIER 3: Severely Incomplete -- NOT Safe for Bug Detection

| Dataset | Corpus Size | Annotation Method | False Negative Risk | Why Unsafe |
|---------|------------|-------------------|--------------------:|------------|
| MS MARCO dev | 8.84M passages | Single Bing result + single annotator | ~57.6% | Most "non-relevant" results are actually relevant |
| Natural Questions | 2.68M passages | Single Google page shown to annotator | High | Many valid Wikipedia passages unlabeled |
| HotpotQA | 5.23M passages | Exactly 2 docs labeled per query | Severe | Multi-hop design = massive missing annotations |
| FiQA-2018 | 57K documents | Forum-accepted answers only | High | Other relevant financial posts unlabeled |
| FEVER | 5.42M passages | Minimal evidence sets from annotators | ~28% | Only one evidence set typically recorded |
| LoTTE | Varies (StackExchange) | Community upvotes/accepted answers | High | Missing answers never written or never upvoted |
| Quora Duplicate | ~523K questions | Platform duplicate flags + graph expansion | Moderate | Novel duplicates missed |

---

### IBM MT-RAG Benchmark (Our Dataset)

| Property | Detail |
|---|---|
| Corpus | Multi-tenant helpdesk documents |
| Annotation method | ELSER retrieval + human yes/no labeling |
| False negative risk | Very high (99%+ of corpus never seen by annotators) |

**Verdict**: NOT safe for detecting retrieval bugs in isolation. Useful for comparing
against the specific ELSER baseline, but a "miss" may be a genuine relevant document
that ELSER never surfaced for annotation.

---

## Recommendations for Our Pipeline Validation

### Phase 1: Sanity Check (SciFact via BEIR)
- Validates that the pipeline is fundamentally working
- Tiny corpus = fast iteration
- Citation-based ground truth = minimal false negatives

### Phase 2: Deep Validation (TREC-DL 2019+2020)
- Tests ranking quality, BM25, hybrid search, and reranking
- HiCAL-augmented judgments = most trustworthy large-scale benchmark
- Available via ir_datasets, convertible to BEIR format

### Phase 3: Domain Stress Test (TREC-COVID via BEIR)
- Tests domain vocabulary and technical term retrieval
- 493 relevant docs per query = deep evaluation
- Already in BEIR format

### Phase 4: Targeted Tests on Our Corpus (Manual)
- 30+ hand-crafted test cases for helpdesk-specific patterns
- Error codes, typos, negation, short queries, multi-tenant filtering
- No public dataset covers these patterns

---

## References

- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models (NeurIPS 2021)](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)
- [TREC 2019 Deep Learning Track Overview](https://arxiv.org/abs/2003.07820)
- [TREC DL Reusable Test Collections (SIGIR 2021)](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/04/sigir2021-resource-trecdl-craswell.pdf)
- [SciFact: Verifying Scientific Claims (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.609/)
- [D-MERIT: Evaluating Retrieval with Partial Annotations (EMNLP 2024)](https://arxiv.org/abs/2406.16048)
- [TREC-COVID Overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC8264272/)
- [Natural Questions (TACL)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00276/43518/)
- [Mitigating False Negatives in Retriever Training](https://huggingface.co/blog/dragonkue/mitigating-false-negatives-in-retriever-training)
- [LLMs Can Patch Up Missing Relevance Judgments](https://arxiv.org/html/2405.04727v1)
- [Cranfield Collection (Stanford IR Book)](https://nlp.stanford.edu/IR-book/html/htmledition/standard-test-collections-1.html)

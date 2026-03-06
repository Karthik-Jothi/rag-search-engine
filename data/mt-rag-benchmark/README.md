# IBM MTRAG Benchmark Data

Data sourced from [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark).

MTRAG is a multi-turn conversational benchmark for evaluating RAG systems, featuring 110 human-generated conversations across 4 domains (ClapNQ, Cloud, FiQA, Govt).

## Structure

```
corpora/
  passage_level/    - Passage-level corpus per domain (*.jsonl, git-ignored due to size)
  document_level/   - Document-level corpus per domain (*.jsonl, git-ignored due to size)
human/
  conversations/    - 110 multi-turn conversations (conversations.json)
  retrieval_tasks/  - Retrieval queries + qrels per domain (BEIR format)
  generation_tasks/ - 842 generation tasks under 3 retrieval settings
  evaluations/      - LLM-as-judge and human evaluation results
  mtrageval/        - Sample evaluation input data
synthetic/
  conversations/    - Synthetic conversation data
  generation_tasks/ - Synthetic generation tasks
  evaluations/      - Synthetic evaluation results
scripts/
  evaluation/       - Evaluation scripts (retrieval + generation)
```

## Corpus Setup

The large corpus JSONL files are git-ignored. To download them:

```bash
cd data/mt-rag-benchmark/corpora
# Download from the IBM repo
for level in passage_level document_level; do
  cd $level
  for domain in clapnq cloud fiqa govt; do
    curl -LO "https://github.com/IBM/mt-rag-benchmark/raw/main/corpora/${level}/${domain}.jsonl.zip"
    unzip "${domain}.jsonl.zip" && rm "${domain}.jsonl.zip"
  done
  cd ..
done
```

## Domains

| Corpus | Domain | Documents | Passages |
|--------|--------|-----------|----------|
| ClapNQ | Wikipedia | 4,293 | 183,408 |
| Cloud | Technical Docs | 57,638 | 61,022 |
| FiQA | Finance | 7,661 | 49,607 |
| Govt | Government | 8,578 | 72,422 |

## License

See [LICENSE](LICENSE) (CC-BY-4.0 from IBM).

## Citation

```bibtex
@misc{katsis2025mtrag,
  title={MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems},
  author={Yannis Katsis and Sara Rosenthal and Kshitij Fadnis and Chulaka Gunasekara and Young-Suk Lee and Lucian Popa and Vraj Shah and Huaiyu Zhu and Danish Contractor and Marina Danilevsky},
  year={2025},
  eprint={2501.03468},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
}
```

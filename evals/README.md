# Evaluating ScholarAI

Three scripts. Run them from the `ScholarAI` folder with your virtual environment active.

| Step | Script | Needs Groq key? | Answers the question |
|---|---|---|---|
| 1 | `evals.retrieval_eval` | No | Does the retriever find the passage that contains the answer? Which retrieval setting is best? |
| 2 | `evals.generate_dataset` | Yes | (Helper) Draft a question set for my own paper |
| 3 | `evals.run_eval` | Yes | Are the final answers correct, grounded and cited? Is the upgraded pipeline better than naive RAG? Is the agent worth its extra LLM calls? |

## Benchmark suite (the numbers on the Benchmarks page)

```bash
python -m evals.benchmark download                         # fetch the three arXiv PDFs (once)
python -m evals.benchmark run                              # retrieval + speed, no API key
python -m evals.benchmark run --answers --judge --sleep 3  # also answer quality (needs GROQ_API_KEY)
```

- Papers and question sets are listed in `evals/benchmark/suite.json`. Add your own paper by appending an entry with `pdf` and `dataset`.
- Results go to `evals/benchmark/latest.json` (shown at `/benchmarks`) and `evals/results/`.
- **Invalid questions.** Before scoring, each answer key is searched for in the parsed paper. A question whose facts are not there (a typo in the key, or a different version of the PDF) is listed and left out.
- **Confidence intervals.** Hit rate and context recall carry a 95% bootstrap interval. If two settings' intervals overlap heavily, do not claim one is better.
- `--offline` runs with a toy embedder to check the script; those results are not published.

## Quick try (no API key)

```bash
python -m evals.retrieval_eval --paper evals/sample/sample_paper.pdf --dataset evals/datasets/sample_paper.json
```

`evals/sample/sample_paper.pdf` is a **synthetic** three-page paper written only for testing (its method, datasets and numbers are invented). It lets you check that everything runs. Report numbers from real papers, not from this file.

If the embedding model cannot be downloaded, add `--offline` to run with a toy embedder. That only proves the script works; the numbers mean nothing.

## Evaluate your own paper

```bash
# 1. Draft questions (then open the JSON and delete or fix weak ones)
python -m evals.generate_dataset --paper my_paper.pdf --n 15 --unanswerable 4 --out evals/datasets/my_paper.json --sleep 2

# 2. Retrieval ablation (+ --llm to include query expansion and LLM reranking)
python -m evals.retrieval_eval --paper my_paper.pdf --dataset evals/datasets/my_paper.json

# 3. Answer quality (--judge adds faithfulness / relevance / correctness; --sleep avoids rate limits)
python -m evals.run_eval --paper my_paper.pdf --dataset evals/datasets/my_paper.json --judge --sleep 3
```

Each run prints a table and saves `evals/results/<name>-<time>.md` and `.json`.

For a report, use 3 or more papers from different fields and layouts, 15 to 20 questions each, and average the tables.

## Dataset format

```json
{
  "questions": [
    {"question": "What optimizer and learning rate are used?", "expected_facts": ["Adam", "0.001"]},
    {"question": "What accuracy is reported on ImageNet?", "expected_facts": [], "answerable": false}
  ]
}
```

- `expected_facts`: short, exact, **distinctive** strings a correct answer must contain (a number with its unit, a dataset or method name). Avoid single digits and common words, because a retrieved chunk counts as relevant when it contains one of these strings.
- `"answerable": false`: a question the paper does not answer. A good system says so instead of inventing an answer.
- Optional `gold_pages`: used for relevance only when a question has no `expected_facts`.

## Retrieval metrics (`retrieval_eval`)

Computed over answerable questions.

| Metric | Plain meaning | Good value |
|---|---|---|
| `hit` | Share of questions where at least one retrieved chunk contains an expected fact | High |
| `mrr` | How close to the top the first such chunk is (1 = always first) | High |
| `context_recall` | Share of expected facts present anywhere in the retrieved context. This caps answer quality: a fact that was not retrieved cannot be answered faithfully | High |
| `precision` | Share of retrieved chunks that contain an expected fact (signal vs noise) | High |
| `ndcg` | Ranking quality | High |
| `n_chunks` | Average chunks handed to the LLM (prompt size, cost) | Lower at equal recall |

Configurations, each adding one idea: `bm25` → `dense` → `hybrid` → `hybrid+mmr` → `pipeline` (what the app uses) → `pipeline+ce` (cross-encoder) / `pipeline+expand` / `pipeline+llm-rerank` (the last two need `--llm`).

### Score calibration

The same script prints the cosine-similarity distribution of:

- **relevant** chunks (contain an expected fact),
- **distractor** chunks (best-scoring chunk without one),
- the best chunk for **unanswerable** questions.

It then suggests `MIN_RELEVANCE` (just below the 10th percentile of relevant chunks, so real evidence is kept) and `HIGH_CONFIDENCE` (their median). Put the values in `.env` if they are far from the defaults. Do this once with the real embedding model; the defaults are starting points.

## Answer metrics (`run_eval`)

Three pipelines are compared: `naive` (plain hybrid top-k and one LLM call, which is how fast RAG worked before the retrieval upgrade), `simple` (the current fast RAG pipeline), `agent` (tool-using research agent).

| Metric | Plain meaning | Good value |
|---|---|---|
| `fact_recall` | Share of expected facts that appear in the answer | High |
| `context_recall` | Share of expected facts in the passages the answer was built from | High |
| `cited` | Share of answers that cite a page such as `[p. 4]` | High |
| `citation_precision` | Share of cited pages that were really among the passages read | High |
| `halluc_numbers` | Average count of numbers in an answer that occur nowhere in the paper | 0 |
| `abstain_ok` | On unanswerable questions: share where the system said "not covered" | High |
| `false_abstain` | On answerable questions: share where it wrongly said "not covered" | Low |
| `llm_calls`, `seconds` | Cost per question | Low |
| `faithfulness` (`--judge`) | Share of the answer's claims supported by the retrieved passages | High |
| `relevance` (`--judge`) | Does the answer address the question (0 to 1) | High |
| `correctness` (`--judge`) | Agreement with the reference facts (0 to 1) | High |

Reading the results:

- `context_recall` low → retrieval is the problem (look at the retrieval ablation).
- `context_recall` high but `fact_recall` low → the fact was retrieved but the LLM did not use it (prompt or model problem).
- `abstain_ok` low → the system invents answers to questions the paper does not cover.

## Limits of these metrics

- Fact matching is exact substring. "0.001" matches, "1e-3" does not. Use `--judge` when paraphrases matter.
- The judge is an LLM from the same provider as the answerer, so treat judge scores as indicative.
- Generated question sets are drafts. Review them before reporting numbers, especially the unanswerable questions.

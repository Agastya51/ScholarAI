# ScholarAI — Architecture

How the system is put together. For the overview see the [README](../README.md); for the full academic treatment see [BTP_Report.pdf](BTP_Report.pdf). File paths below are relative to the repository root.

---

## 1. Design principles

**Every answer must be checkable.** Retrieved passages keep their page and section, answers cite them as `[p. N]`, and the interface turns each citation into a rendered page of the original PDF with the passage highlighted. Quotes produced by the model are matched against the paper text without an LLM.

**A figure is read together with its text.** A figure, its caption and the sentences that discuss it are joined before any model sees the figure, even when they sit on different pages.

**Retrieval quality is measured, not assumed.** Each retrieval idea (BM25, dense, fusion, MMR, thresholds, query expansion, reranking) can be switched on and off and is scored by the evaluation scripts in `evals/`.

**Agents are bounded.** Tool loops, self-correcting searches and fact-check loops all have iteration caps, and every request has a budget of LLM calls (`MAX_LLM_CALLS_PER_RUN`, `MAX_LLM_CALLS_ANALYZE`), so a loop cannot exhaust the Groq rate limit.

**Graceful degradation.** A `429` from the primary model falls back to a second model; if the vision model is unavailable a figure is explained from its caption and in-text mentions and labelled as such; if query expansion fails, retrieval continues without it.

**Small footprint.** The app runs on a 512 MB instance: ONNX embeddings (FastEmbed), a single gunicorn worker, embeddings cached on disk by content hash, and bounded per-session memory.

---

## 2. Layers

| Layer | Responsibility | Files |
| --- | --- | --- |
| **Interface** | Flask routes, per-browser sessions, upload validation, server-sent events for live agent runs; HTML pages with shared CSS/JS (fonts and libraries bundled) | `src/app.py`, `src/templates/`, `src/static/` |
| **Document understanding** | Layout-aware parsing, sections, tables, captions, references, figure regions, in-text mentions, page rendering with highlights | `src/rag.py`, `src/figures.py`, `src/viewer.py` |
| **Retrieval** | Hybrid index (FAISS + BM25), fusion, MMR, thresholds, query expansion, reranking, confidence and grounding signals | `src/rag.py`, `src/retrieval.py` |
| **Orchestration** | LangGraph workflows and agents, tool registry | `src/workflows.py`, `src/agents/`, `src/tools.py` |
| **Model access** | Groq chat with tool calling, structured (Pydantic) output, vision, typed errors, call budgets, test backends | `src/llm.py` |
| **External services** | Groq; arXiv, Semantic Scholar, OpenAlex, CrossRef | `src/search.py` |
| **Study material** | Claims with verified quotes, glossary, flashcards, quiz, explanations at three levels, starter questions | `src/study.py` |
| **Evaluation** | Metrics, retrieval ablation, answer evaluation, dataset generator, benchmark suite | `evals/` |

---

## 3. From PDF to index (`src/rag.py`)

`parse_document()` works in two passes over the pages.

**Pass A: layout.** For each page:
- Text blocks are put in reading order by `ordered_blocks()`. A page counts as two-column when enough narrow blocks sit on each side of the centre; it is then read band by band (bands are separated by full-width blocks such as the title, abstract or a wide figure), left column before right column. Lines that PyMuPDF split on wide gaps (for example `3.1` and `Training Details`) are joined back.
- Tables are found from their ruling lines, only on pages that contain vector drawings: PyMuPDF's line-based finder for full grids, and, on pages with a table caption, a finder for LaTeX booktabs tables (horizontal rules only; rows are text lines, columns are the gaps that run through every row). PyMuPDF 2.x's neural layout analyser is switched off because it is slow and memory-hungry on a small CPU. Cell text is assembled from the page's words by position, because the library's own extraction can detach a decimal point from its number. A "table" covering most of the page is treated as a mis-detection.
- Captions (`Fig. 3.`, `Figure 3:`, `TABLE II`) are recognised; prose that merely mentions a figure is not.

**Figures** (`src/figures.py`, see section 5) are located next, because the second pass needs to know where they are.

**Pass B: running text.** Blocks are walked in reading order to build segments of `{page, section, text}`. Numbered and well-known headings start a new section; subsections keep their parent (`3 Method › 3.1 Setup`). Captions, table cells, and short text inside figure regions (axis titles, tick labels, legends) are kept out of the prose. Ligatures are expanded and line-end hyphenation is undone, while real hyphens before digits or after numbers are kept (`60-minute`, `RiverCity-200`).

Finally the reference list is split into numbered entries, and the sentences that cite each figure and table are collected.

**Chunking.** `chunk_segments()` packs whole sentences into chunks of about 900 characters with about 150 characters of overlap, never across a section boundary, so each chunk keeps one page/section anchor. Tables become their own chunks (caption + markdown), figure captions become `caption` chunks, and reference-list chunks are tagged `reference`. Each chunk is indexed as `<section>: <text>` so it can be found by the section it belongs to.

**Indexing.** Chunks are embedded with `BAAI/bge-small-en-v1.5` (384 dimensions; queries get the model's retrieval instruction prefix) and stored in a FAISS inner-product index over normalised vectors, so scores are cosine similarities comparable across questions. A BM25 index is built over the same texts. Embeddings are cached on disk under a hash of the model name and chunk texts, so re-uploading the same paper skips embedding.

---

## 4. Retrieval (`src/retrieval.py`)

```
question
  → BM25 ranking + dense ranking  → Reciprocal Rank Fusion
  → [if evidence is weak: one LLM call writes 2 rewrites + a hypothetical answer passage (HyDE); fuse again]
  → [optional reranking: cross-encoder or listwise LLM]
  → adaptive k (read wider when the best passage is weak) → MMR (relevant but not repetitive)
  → relevance threshold (cosine below MIN_RELEVANCE and no keyword match → dropped)
  → confidence label: high / medium / low / none
```

- Reference-list chunks are excluded unless the question is about citations; a question such as "what is reference [12]?" ranks the entry containing `[12]` first.
- After the answer is written, `check_grounding()` lists numbers in the answer that occur in none of the passages and cited pages that were not among them. Both are shown in the interface.
- Every switch (`USE_MMR`, `QUERY_EXPANSION`, `RERANK_MODE`, `MIN_RELEVANCE`, `HIGH_CONFIDENCE`, `DEFAULT_TOP_K`) is set in `src/config.py` or `.env`.

---

## 5. Figure understanding (`src/figures.py`)

1. **Graphics on a page** are collected from embedded bitmaps and from clusters of vector drawings (thin rules, table borders and page backgrounds are ignored). Pages with thousands of vector paths use a fast linear merge instead of PyMuPDF's clustering.
2. **Each caption claims the graphics above it** (or below it, if nothing is above) within its column band, stopping at another caption or a paragraph. Sub-plots are chained into one region, and short text near the region (axis labels, legends) is included.
3. **A caption with no graphics on its page** takes the largest unclaimed figure on the previous or next page (a figure that pushed its caption over a page break).
4. **Uncaptioned bitmaps** are kept as unlabelled images.
5. **The region is rendered** from the page at up to 150 dpi, so vector plots and typed axis labels appear exactly as printed.
6. **Mentions.** Every sentence that refers to the figure (`Fig. 3`, `Figs. 2 and 3`, `Figures 2–4`, `Figure 5(a)`) is collected with its page and section, plus the following sentence.
7. **Explanation.** The vision model receives the image, the caption and the mentions, and answers in a fixed structure: what it is, how to read it, what it shows, why it matters in the paper (with page citations), and whether the picture supports the text. A question about the figure uses the same context.
8. **Reuse.** A successful explanation is added to the search index as a `figure` chunk. A question that names a figure is answered after looking at that figure, and the agent's `analyze_figure` tool takes an optional question.

---

## 6. Workflows and agents

| Feature | Route | Module | Pattern |
| --- | --- | --- | --- |
| Open a paper | `POST /upload` → poll `GET /api/jobs/<id>`, then `POST /api/prepare` | `app.py`, `workflows.py` | Upload saves the file and starts a background job that parses and indexes it (no model calls; one paper at a time; progress reported page by page). Prepare runs a graph: summarise (targeted retrieval over abstract, method, results, conclusions) → extract search keywords. Figures are explained on demand |
| Quick answer | `POST /ask` (`mode=simple`) | `workflows.py` | Retrieval pipeline + one grounded answer call, with confidence and grounding report |
| Thorough answer | `POST /ask` (`mode=agent`) | `agents/qa_agent.py` | ReAct tool loop over `search_paper`, `get_section`, `get_table`, `analyze_figure`, `get_reference`, external literature tools; capped iterations; conversation memory |
| Related work | `POST /compare` | `workflows.py` | Search four databases → LLM relevance grader → reformulate keywords if too few are relevant (max 3 attempts) → comparative study |
| Improve writing | `POST /improve` | `workflows.py` | Evaluator-optimizer: critique → rewrite → critic plus number/anchor checks → revise failing edits (max 2 rounds) → verified or flagged |
| Peer review | `POST /review` | `agents/review_panel.py` | Parallel reviewers with LangGraph `Send`, structured reviews, meta-review decision |
| Run everything | `POST /analyze` (SSE) | `agents/supervisor.py` | LLM supervisor routes specialists toward a goal; `interrupt` checkpoints let the user approve keywords and pick rewrites |
| Compare papers | `POST /multi-compare` | `agents/multi_paper.py` | Parallel profiling of 2–5 papers → comparison matrix + synthesis |
| Study | `POST /api/study/*`, `/api/explain` | `study.py` | Structured generation, then non-LLM checks (quotes searched in the paper, page numbers validated, invented numbers flagged) |

Every response carries a `trace` (agent, kind, detail) and an `llm_calls` count, shown under "How this was produced".

---

## 7. Sessions and resources

- Each browser receives an anonymous, `HttpOnly` cookie and its own `PaperLibrary`. A context variable carries the session into streamed responses and LangGraph worker threads, so the rest of the code calls `library` without knowing about sessions.
- Idle libraries expire after `SESSION_TTL_SECONDS`; beyond `MAX_SESSIONS` libraries that hold papers, the least recently used is dropped. Dropping a library deletes its uploaded PDFs and figure images. Health checks never create sessions.
- Uploads must start with a PDF signature and respect `MAX_UPLOAD_MB` and `MAX_PDF_PAGES`. Files are stored under server-generated names.
- Paper indexes live in the worker's memory, which is why the `Procfile` runs one gunicorn worker.

---

## 8. Interface

- Server-rendered pages (`src/templates/`) share `src/static/app.css` and `src/static/app.js`. Fonts (Newsreader, Instrument Sans), `marked` and `DOMPurify` are bundled, so the page makes no third-party requests.
- Model output is rendered as Markdown and sanitised with DOMPurify. `[p. N]` and `(p. N)` become buttons that open the page viewer (`/papers/<id>/pages/<n>`), which highlights the cited chunk, quote, table or figure on a server-rendered image of the page; the PDF itself is never modified.
- Results are kept in the browser's `sessionStorage` per tab. If the server no longer has the paper (restart or expiry), the pages say so and offer to upload it again.

---

## 9. Evaluation (`evals/`)

| Script | Needs an API key | Measures |
| --- | --- | --- |
| `retrieval_eval` | No | Hit rate, MRR, context recall, precision, nDCG for each retrieval configuration, plus a similarity-score calibration that suggests the two thresholds |
| `run_eval` | Yes | Fact recall, context recall, citation rate and precision, invented numbers, correct and wrong abstention, cost; optional LLM-judged faithfulness, relevance and correctness |
| `generate_dataset` | Yes | Drafts question sets whose answer key must occur literally in the source passage |
| `benchmark` | No (`--answers`: yes) | The suite in `evals/benchmark/suite.json` (Attention Is All You Need, BERT, ResNet): pooled metrics with 95% bootstrap intervals, parsing statistics, indexing time and search latency; questions whose answer key is not found in the PDF are reported and excluded |

Metric definitions in plain language are in [`evals/README.md`](../evals/README.md).

---

## 10. Tests

`python -m unittest discover -s tests` runs the whole suite without network access or an API key: a scripted fake LLM and vision backend stand in for Groq, a hashing encoder stands in for the embedding model, and test PDFs are generated in code (`tests/pdf_factory.py`), including a two-column page whose native text order is wrong on purpose and a paper whose figures are vector drawings discussed on another page.

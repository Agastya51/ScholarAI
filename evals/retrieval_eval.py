"""Retrieval ablation: which retrieval settings actually find the evidence?

Needs NO Groq key for the default configurations (it only runs the retriever, not the LLM).

Usage (from the ScholarAI folder):
    python -m evals.retrieval_eval --paper evals/sample/sample_paper.pdf --dataset evals/datasets/sample_paper.json
    python -m evals.retrieval_eval --paper my.pdf --dataset evals/datasets/my_paper.json --k 4 --llm

Configurations compared (each one adds a single idea to the previous):
    bm25              keyword search only
    dense             embedding search only
    hybrid            BM25 + dense fused with Reciprocal Rank Fusion          (the strategy ScholarAI used before)
    hybrid+mmr        + Maximal Marginal Relevance (less redundant context)
    pipeline          + adaptive k and relevance threshold                     (what the app uses by default)
    pipeline+ce       + cross-encoder reranking            (needs the reranker model; skipped if it cannot load)
    pipeline+expand   + query rewriting and HyDE           (--llm, 1 LLM call per question)
    pipeline+llm-rerank  + listwise LLM reranking          (--llm, 1 LLM call per question)

Metrics (averaged over the answerable questions):
    hit             share of questions where at least one retrieved chunk contains an expected fact
    mrr             1 / rank of the first such chunk (is the evidence at the top?)
    context_recall  share of expected facts present in the retrieved context (ceiling for the answer)
    precision       share of retrieved chunks that contain an expected fact (signal vs noise)
    ndcg            ranking quality with binary relevance
    chunks          average number of chunks handed to the LLM
"""
import argparse
import time
from typing import Callable, Dict, List

from evals.common import load_dataset, load_eval_paper, percentile, safe_stdout, save_report
from evals.metrics import average, chunk_is_relevant, is_answerable, markdown_table, retrieval_metrics
from src.config import DEFAULT_TOP_K, HIGH_CONFIDENCE, MIN_RELEVANCE
from src.rag import Paper
from src.retrieval import cross_encoder_scores, retrieve_context

Retriever = Callable[[Paper, str, int], List[dict]]

def _search(mode: str, mmr: bool) -> Retriever:
    return lambda paper, q, k: paper.store.search(q, k, mode=mode, mmr=mmr)

def _pipeline(**kwargs) -> Retriever:
    return lambda paper, q, k: retrieve_context(paper, q, k, **kwargs).chunks

CONFIGS: Dict[str, Retriever] = {
    "bm25": _search("sparse", False),
    "dense": _search("dense", False),
    "hybrid": _search("hybrid", False),
    "hybrid+mmr": _search("hybrid", True),
    "pipeline": _pipeline(expansion="off", rerank="none"),
    "pipeline+ce": _pipeline(expansion="off", rerank="cross-encoder"),
    "pipeline+expand": _pipeline(expansion="always", rerank="none"),
    "pipeline+llm-rerank": _pipeline(expansion="off", rerank="llm"),
}
NO_LLM_CONFIGS = ["bm25", "dense", "hybrid", "hybrid+mmr", "pipeline", "pipeline+ce"]
LLM_CONFIGS = ["pipeline+expand", "pipeline+llm-rerank"]
COLUMNS = ["hit", "mrr", "context_recall", "precision", "ndcg", "n_chunks"]

def evaluate_retrieval(paper: Paper, items: List[dict], configs: List[str], k: int = DEFAULT_TOP_K,
                       sleep: float = 0.0) -> dict:
    """Run every configuration over the answerable questions. Returns {"summary", "rows"}."""
    answerable = [it for it in items if is_answerable(it)]
    rows, summary = [], {}
    for name in configs:
        per_question = []
        for item in answerable:
            chunks = CONFIGS[name](paper, item["question"], k)
            m = retrieval_metrics(chunks, item)
            per_question.append(m)
            rows.append({"config": name, "question": item["question"], **m,
                         "pages": [c.get("page") for c in chunks]})
            if sleep and name in LLM_CONFIGS:
                time.sleep(sleep)
        summary[name] = average(per_question, COLUMNS)
    return {"summary": summary, "rows": rows, "n_questions": len(answerable), "k": k}

def score_calibration(paper: Paper, items: List[dict]) -> dict:
    """Cosine-similarity distributions used to set MIN_RELEVANCE / HIGH_CONFIDENCE for your embedding model.

    relevant:      best score among chunks that contain an expected fact      (should be ABOVE the threshold)
    distractor:    best score among chunks that do NOT contain one            (overlap with `relevant` = hard case)
    unanswerable:  best score of any chunk for questions the paper cannot answer (should be BELOW high confidence)
    """
    relevant, distractor, unanswerable = [], [], []
    store = paper.store
    for item in items:
        scores = store.dense_scores(item["question"])
        if not len(scores):
            continue
        if not is_answerable(item):
            unanswerable.append(float(scores.max()))
            continue
        flags = [chunk_is_relevant(meta, item) for meta in store.metadata]
        rel = [float(s) for s, ok in zip(scores, flags) if ok]
        other = [float(s) for s, ok, meta in zip(scores, flags, store.metadata)
                 if not ok and meta.get("kind") != "reference"]
        if rel:
            relevant.append(max(rel))
        if other:
            distractor.append(max(other))

    def stats(values):
        return {"n": len(values), "p10": percentile(values, 0.1), "median": percentile(values, 0.5),
                "p90": percentile(values, 0.9)} if values else {"n": 0}

    out = {"relevant": stats(relevant), "distractor": stats(distractor), "unanswerable": stats(unanswerable),
           "current": {"MIN_RELEVANCE": MIN_RELEVANCE, "HIGH_CONFIDENCE": HIGH_CONFIDENCE}}
    if relevant:
        out["suggested"] = {"MIN_RELEVANCE": round(percentile(relevant, 0.1) - 0.02, 2),
                            "HIGH_CONFIDENCE": round(percentile(relevant, 0.5), 2)}
    return out

def render(result: dict, calibration: dict, paper: Paper, dataset_path: str) -> str:
    table = markdown_table(["config"] + COLUMNS,
                           [[name] + [m.get(c, float("nan")) for c in COLUMNS] for name, m in result["summary"].items()])
    lines = [f"# Retrieval evaluation — {paper.title or paper.filename}", "",
             f"- dataset: `{dataset_path}` ({result['n_questions']} answerable questions), k = {result['k']}",
             f"- index: {len(paper.chunks)} chunks over {paper.pages} pages", "", table, "",
             "## Similarity score calibration", ""]
    rows = []
    for key in ("relevant", "distractor", "unanswerable"):
        s = calibration[key]
        rows.append([key, s["n"]] + ([s["p10"], s["median"], s["p90"]] if s["n"] else ["-", "-", "-"]))
    lines.append(markdown_table(["chunk group", "questions", "p10", "median", "p90"], rows))
    lines += ["", f"Current thresholds: MIN_RELEVANCE={MIN_RELEVANCE}, HIGH_CONFIDENCE={HIGH_CONFIDENCE}."]
    if "suggested" in calibration:
        s = calibration["suggested"]
        lines.append(f"Suggested from this data: MIN_RELEVANCE={s['MIN_RELEVANCE']} (just under the 10th percentile "
                     f"of relevant chunks), HIGH_CONFIDENCE={s['HIGH_CONFIDENCE']} (their median). "
                     "Set them in `.env` if they differ a lot from the current values.")
    return "\n".join(lines) + "\n"

def main():
    safe_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", required=True, help="Path to the PDF")
    parser.add_argument("--dataset", required=True, help="Path to the question set (JSON)")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K, help="Chunks requested per question")
    parser.add_argument("--configs", default="", help="Comma-separated subset of: " + ", ".join(CONFIGS))
    parser.add_argument("--llm", action="store_true", help="Also run the configurations that call the LLM (needs GROQ_API_KEY)")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between LLM calls (rate limits)")
    parser.add_argument("--limit", type=int, default=0, help="Only use the first N questions")
    parser.add_argument("--offline", action="store_true",
                        help="Smoke test with a toy embedder (no model download). Results are not meaningful.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.limit or None)
    paper = load_eval_paper(args.paper, args.offline)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()] or NO_LLM_CONFIGS + (LLM_CONFIGS if args.llm else [])
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Choose from: {', '.join(CONFIGS)}")
    if "pipeline+ce" in configs and (args.offline or cross_encoder_scores("warm-up", ["warm-up passage"]) is None):
        print("[skip] pipeline+ce: the cross-encoder model is not available (offline mode or download failed).")
        configs.remove("pipeline+ce")

    result = evaluate_retrieval(paper, dataset["questions"], configs, args.k, args.sleep)
    calibration = score_calibration(paper, dataset["questions"])
    report = render(result, calibration, paper, args.dataset)
    print("\n" + report)
    stem = save_report("retrieval", {"paper": args.paper, "dataset": args.dataset, **result,
                                     "calibration": calibration}, report)
    print(f"Saved {stem}.md and {stem}.json")

if __name__ == "__main__":
    main()

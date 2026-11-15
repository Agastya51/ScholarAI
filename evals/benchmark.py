"""Benchmark ScholarAI on a suite of papers and publish the numbers to the app's Benchmarks page.

    python -m evals.benchmark download                    # fetch the suite's arXiv PDFs (once)
    python -m evals.benchmark run                         # retrieval + speed benchmark, no API key needed
    python -m evals.benchmark run --answers --judge --sleep 3     # also answer quality (needs GROQ_API_KEY)
    python -m evals.benchmark run --offline               # smoke test with a toy embedder (numbers not meaningful)

What it measures, per paper and pooled over all questions of the suite:
  retrieval   hit rate, MRR, context recall, precision for each retrieval configuration (see evals/retrieval_eval.py),
              with a 95% bootstrap confidence interval on hit rate and context recall
  speed       seconds to parse + index each paper, milliseconds per retrieval
  parsing     pages, chunks, figures (with caption / with in-text mentions), tables, references found
  answers     (--answers) fact recall, citation rate, hallucinated numbers, abstention; (--judge) faithfulness etc.

Before scoring, every question is checked against the parsed text of its paper: a question whose expected facts
occur nowhere in the text is reported as INVALID and left out, so a wrong answer key (or a different version of the
PDF) cannot silently lower the scores.

Results are written to evals/benchmark/latest.json (shown at /benchmarks in the app; commit it to show the numbers on
a deployed instance) and to evals/results/benchmark-<time>.md.
"""
import argparse
import json
import os
import platform
import random
import time
from typing import List

from evals.common import load_dataset, load_eval_paper, safe_stdout, save_report
from evals.metrics import average, facts_of, is_answerable, markdown_table, norm
from evals.retrieval_eval import COLUMNS, NO_LLM_CONFIGS, evaluate_retrieval
from src.config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from src.rag import Paper
from src.retrieval import cross_encoder_scores, retrieve_context

SUITE_PATH = os.path.join("evals", "benchmark", "suite.json")
LATEST_PATH = os.path.join("evals", "benchmark", "latest.json")

def load_suite(path: str = SUITE_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["papers"]

# ----------------------------------------------------------------------------- download
def download(suite: List[dict]) -> int:
    import requests
    missing = 0
    for entry in suite:
        if not entry.get("arxiv") or os.path.exists(entry["pdf"]):
            continue
        url = f"https://arxiv.org/pdf/{entry['arxiv']}"
        print(f"Downloading {entry['title']} from {url} ...")
        try:
            res = requests.get(url, timeout=60, headers={"User-Agent": "ScholarAI-benchmark/1.0"})
            res.raise_for_status()
            if not res.content.startswith(b"%PDF-"):
                raise ValueError("the server did not return a PDF")
            os.makedirs(os.path.dirname(entry["pdf"]), exist_ok=True)
            with open(entry["pdf"], "wb") as f:
                f.write(res.content)
            time.sleep(3)      # be polite to arXiv
        except Exception as e:
            missing += 1
            print(f"  failed: {e}\n  Download it in your browser and save it as {entry['pdf']}")
    return missing

# ----------------------------------------------------------------------------- dataset sanity
def split_valid(paper: Paper, items: List[dict]):
    """Questions whose expected facts all occur somewhere in the parsed paper are valid; the rest are reported."""
    text = norm(paper.text + " " + " ".join(str(c["text"]) for c in paper.chunks))
    valid, invalid = [], []
    for item in items:
        missing = [f for f in facts_of(item) if norm(f) not in text]
        (invalid if missing and is_answerable(item) else valid).append({**item, "missing_facts": missing}
                                                                      if missing else item)
    return valid, invalid

# ----------------------------------------------------------------------------- statistics
def bootstrap_ci(values: List[float], resamples: int = 1000, seed: int = 0):
    """95% percentile bootstrap confidence interval of the mean."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 3), round(means[int(0.975 * resamples) - 1], 3)]

def parsing_stats(paper: Paper) -> dict:
    return {"pages": paper.pages, "chunks": len(paper.chunks), "sections": len(paper.sections()),
            "figures": len(paper.figures),
            "figures_with_caption": sum(bool(f.get("caption")) for f in paper.figures),
            "figures_with_mentions": sum(bool(f.get("mentions")) for f in paper.figures),
            "tables": len(paper.tables), "references": len(paper.references)}

def retrieval_latency_ms(paper: Paper, items: List[dict], k: int) -> float:
    questions = [it["question"] for it in items][:20]
    if not questions:
        return 0.0
    paper.store._query_cache.clear()
    start = time.perf_counter()
    for q in questions:
        retrieve_context(paper, q, k, expansion="off", rerank="none")
    return round((time.perf_counter() - start) * 1000 / len(questions), 1)

# ----------------------------------------------------------------------------- run
def run(suite: List[dict], k: int, offline: bool, answers: bool, judge: bool, sleep: float, modes: List[str],
        include_synthetic: bool) -> dict:
    configs = list(NO_LLM_CONFIGS)
    if offline or cross_encoder_scores("warm-up", ["warm-up passage"]) is None:
        configs.remove("pipeline+ce")
    papers_out, pooled_rows, pooled_answers, skipped = [], [], [], []
    for entry in suite:
        if entry.get("synthetic") and not (include_synthetic or offline):
            continue
        if not os.path.exists(entry["pdf"]):
            skipped.append({"id": entry["id"], "reason": f"{entry['pdf']} not found (run: python -m evals.benchmark download)"})
            print(f"[skip] {entry['id']}: PDF not found")
            continue
        print(f"\n=== {entry['title']} ===")
        start = time.perf_counter()
        paper = load_eval_paper(entry["pdf"], offline, cache=False)
        index_seconds = round(time.perf_counter() - start, 2)
        items = load_dataset(entry["dataset"])["questions"]
        valid, invalid = split_valid(paper, items)
        for item in invalid:
            print(f"  [invalid question, excluded] {item['question'][:70]} (facts not in the PDF text: {item['missing_facts']})")
        result = evaluate_retrieval(paper, valid, configs, k)
        for row in result["rows"]:
            pooled_rows.append({**row, "paper": entry["id"]})
        out = {"id": entry["id"], "title": entry["title"], "synthetic": bool(entry.get("synthetic")),
               "questions": len(valid), "answerable": result["n_questions"],
               "invalid_questions": [{"question": i["question"], "missing_facts": i["missing_facts"]} for i in invalid],
               "index_seconds": index_seconds, "retrieval_ms": retrieval_latency_ms(paper, valid, k),
               "parsing": parsing_stats(paper), "retrieval": result["summary"]}
        if answers:
            from evals.run_eval import evaluate_answers
            ans = evaluate_answers(paper, valid, modes, judge, sleep)
            out["answers"] = ans["summary"]
            pooled_answers += [{**r, "paper": entry["id"]} for r in ans["rows"]]
        papers_out.append(out)
        print(markdown_table(["config"] + COLUMNS, [[c] + [m.get(col, 0.0) for col in COLUMNS]
                                                    for c, m in result["summary"].items()]))

    pooled = {}
    for config in configs:
        rows = [r for r in pooled_rows if r["config"] == config]
        if rows:
            pooled[config] = {**average(rows, COLUMNS), "n": len(rows),
                              "hit_ci": bootstrap_ci([r["hit"] for r in rows]),
                              "context_recall_ci": bootstrap_ci([r["context_recall"] for r in rows])}
    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"), "offline": offline, "k": k,
        "embedding_model": "toy hashing embedder (offline smoke test)" if offline else EMBEDDING_MODEL_NAME,
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "papers": papers_out, "skipped": skipped, "pooled_retrieval": pooled,
    }
    if answers and pooled_answers:
        from evals.run_eval import summarise
        result["pooled_answers"] = {m: summarise([r for r in pooled_answers if r["mode"] == m]) for m in modes}
        result["judge"] = judge
    return result

def render(result: dict) -> str:
    lines = ["# ScholarAI benchmark", "",
             f"- generated: {result['generated']} on {result['machine']}",
             f"- embedding model: {result['embedding_model']}, k = {result['k']}",
             f"- papers: {', '.join(p['title'] for p in result['papers']) or 'none'}", ""]
    if result["offline"]:
        lines += ["**Offline smoke test: these numbers come from a toy embedder and say nothing about real quality.**", ""]
    if result["pooled_retrieval"]:
        lines += ["## Retrieval, pooled over all questions", "",
                  markdown_table(["config", "questions"] + COLUMNS + ["hit 95% CI", "context recall 95% CI"],
                                 [[c, m["n"]] + [m.get(col, 0.0) for col in COLUMNS]
                                  + [str(m["hit_ci"]), str(m["context_recall_ci"])]
                                  for c, m in result["pooled_retrieval"].items()]), ""]
    if result.get("pooled_answers"):
        cols = sorted({k for s in result["pooled_answers"].values() for k in s} - {"n", "errors"})
        lines += ["## Answers, pooled", "",
                  markdown_table(["mode", "n"] + cols, [[m, s["n"]] + [s.get(c, "-") for c in cols]
                                                        for m, s in result["pooled_answers"].items()]), ""]
    lines += ["## Per paper", "",
              markdown_table(["paper", "pages", "chunks", "figures (captioned / cited in text)", "tables", "references",
                              "index s", "retrieval ms", "questions", "invalid"],
                             [[p["title"][:40], p["parsing"]["pages"], p["parsing"]["chunks"],
                               f"{p['parsing']['figures']} ({p['parsing']['figures_with_caption']} / {p['parsing']['figures_with_mentions']})",
                               p["parsing"]["tables"], p["parsing"]["references"], p["index_seconds"],
                               p["retrieval_ms"], p["questions"], len(p["invalid_questions"])]
                              for p in result["papers"]]), ""]
    for s in result["skipped"]:
        lines.append(f"- skipped {s['id']}: {s['reason']}")
    return "\n".join(lines) + "\n"

def main():
    safe_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--suite", default=SUITE_PATH)
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--offline", action="store_true", help="Toy embedder smoke test (numbers not meaningful)")
    parser.add_argument("--answers", action="store_true", help="Also generate and score answers (needs GROQ_API_KEY)")
    parser.add_argument("--judge", action="store_true", help="With --answers: add LLM-judged faithfulness etc.")
    parser.add_argument("--modes", default="naive,simple", help="With --answers: naive,simple,agent")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait after each answer (rate limits)")
    parser.add_argument("--include-synthetic", action="store_true", help="Include the invented sample paper")
    parser.add_argument("--no-publish", action="store_true", help="Do not overwrite evals/benchmark/latest.json")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    if args.command == "download":
        raise SystemExit(1 if download(suite) else 0)

    result = run(suite, args.k, args.offline, args.answers, args.judge, args.sleep,
                 [m.strip() for m in args.modes.split(",") if m.strip()], args.include_synthetic)
    if not result["papers"]:
        raise SystemExit("No papers were benchmarked. Run `python -m evals.benchmark download` first.")
    report = render(result)
    print("\n" + report)
    stem = save_report("benchmark", result, report)
    print(f"Saved {stem}.md and {stem}.json")
    if not args.no_publish and not args.offline:
        with open(LATEST_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Published to {LATEST_PATH} (shown at /benchmarks in the app)")
    elif args.offline:
        print("Offline smoke test: not published to the Benchmarks page.")

if __name__ == "__main__":
    main()

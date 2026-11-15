"""End-to-end answer evaluation: naive RAG vs the upgraded pipeline vs the research agent.

Needs GROQ_API_KEY (it generates real answers).

Usage (from the ScholarAI folder):
    python -m evals.run_eval --paper evals/sample/sample_paper.pdf --dataset evals/datasets/sample_paper.json
    python -m evals.run_eval --paper my.pdf --dataset evals/datasets/my_paper.json --modes naive,simple --judge --sleep 3

Metrics per mode (see evals/README.md for plain-language definitions):
  without an LLM judge
    fact_recall         share of expected facts that appear in the answer            (answerable questions)
    context_recall      share of expected facts present in the passages the answer was built from
    cited               share of answers that cite a page, e.g. [p. 4]
    citation_precision  share of cited pages that were really among the passages read
    halluc_numbers      average count of numbers in an answer that occur nowhere in the paper
    abstain_ok          share of UNANSWERABLE questions where the system said "not covered"   (higher is better)
    false_abstain       share of ANSWERABLE questions where it wrongly said "not covered"     (lower is better)
    llm_calls, seconds  cost per question
  with --judge (one extra LLM call per answer)
    faithfulness        share of the answer's claims supported by the retrieved passages (RAGAS-style)
    relevance           does the answer address the question?                (1-5 scaled to 0-1)
    correctness         agreement with the reference facts                   (1-5 scaled to 0-1)
"""
import argparse
import time
from typing import List, Optional

from pydantic import BaseModel, Field

from evals.common import load_dataset, load_eval_paper, safe_stdout, save_report
from evals.metrics import answer_metrics, average, context_recall, facts_of, is_answerable, markdown_table
from evals.pipelines import MODES, run_pipeline
from src.llm import structured, LLMError
from src.rag import Paper, format_chunks

class Claim(BaseModel):
    claim: str
    supported: bool = Field(description="True only if the context passages state or directly imply this claim")

class AnswerJudgement(BaseModel):
    claims: List[Claim] = Field(default_factory=list, description="Every factual claim made in the answer")
    relevance: int = Field(ge=1, le=5, description="5 = directly and completely addresses the question")
    correctness: int = Field(ge=1, le=5, description="5 = agrees with all reference facts; 1 = wrong")

def judge(question: str, answer: str, contexts: List[dict], item: dict) -> Optional[dict]:
    """One LLM call scoring faithfulness (claims vs retrieved context), relevance and correctness."""
    reference = facts_of(item) if is_answerable(item) else "The paper does NOT contain the answer; the correct " \
                                                           "behaviour is to say so."
    try:
        j = structured(f"""You are grading a question-answering system for research papers.

1. Split the ANSWER into its individual factual claims. For each claim decide whether the CONTEXT passages
   support it (stated or directly implied). Statements like "the paper does not cover this" are not claims.
2. relevance (1-5): does the answer address the question that was asked?
3. correctness (1-5): does the answer agree with the REFERENCE? Paraphrases count as correct.

QUESTION: {question}

REFERENCE: {reference}

CONTEXT:
{format_chunks(contexts, max_chars=1200) or '(no passages were retrieved)'}

ANSWER:
{answer}""", AnswerJudgement, max_tokens=900)
    except LLMError as e:
        print(f"    [judge failed: {str(e)[:80]}]")
        return None
    out = {"relevance": (j.relevance - 1) / 4, "correctness": (j.correctness - 1) / 4}
    if j.claims:
        out["faithfulness"] = sum(c.supported for c in j.claims) / len(j.claims)
        out["unsupported_claims"] = [c.claim for c in j.claims if not c.supported]
    return out

def evaluate_answers(paper: Paper, items: List[dict], modes: List[str], use_judge: bool = False,
                     sleep: float = 0.0, verbose: bool = True) -> dict:
    rows = []
    for n, item in enumerate(items, start=1):
        for mode in modes:
            try:
                out = run_pipeline(mode, paper, item["question"])
            except LLMError as e:
                rows.append({"mode": mode, "question": item["question"], "error": str(e)})
                if verbose:
                    print(f"[{n:2}/{len(items)}] {mode:6} ERROR {str(e)[:90]}")
                continue
            contexts = out.pop("contexts")
            pages = [c.get("page") for c in contexts if c.get("page")]
            row = {"mode": mode, "question": item["question"], "answerable": is_answerable(item), **out,
                   **answer_metrics(out["answer"], item, paper.text, pages), "context_pages": pages}
            if is_answerable(item):
                row["context_recall"] = context_recall(contexts, item)
            if use_judge and out["answer"]:
                row.update(judge(item["question"], out["answer"], contexts, item) or {})
            rows.append(row)
            if verbose:
                flag = f"recall={row['fact_recall']:.2f}" if "fact_recall" in row else \
                    f"abstained={'yes' if row['abstained'] else 'NO'}"
                print(f"[{n:2}/{len(items)}] {mode:6} {flag} calls={row['llm_calls']:2} {item['question'][:60]}")
            if sleep:
                time.sleep(sleep)
    return {"rows": rows, "summary": {m: summarise([r for r in rows if r["mode"] == m]) for m in modes}}

SUMMARY_COLUMNS = ["fact_recall", "context_recall", "faithfulness", "relevance", "correctness", "cited",
                   "citation_precision", "halluc_numbers", "abstain_ok", "false_abstain", "llm_calls", "seconds"]

def summarise(rows: List[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    out = average(answerable, ["fact_recall", "context_recall"])
    out.update(average(ok, ["faithfulness", "relevance", "correctness", "cited", "citation_precision",
                            "llm_calls", "seconds"]))
    if ok:
        out["halluc_numbers"] = round(sum(r["hallucinated_numbers"] for r in ok) / len(ok), 3)
    if unanswerable:
        out["abstain_ok"] = round(sum(r["abstained"] for r in unanswerable) / len(unanswerable), 3)
    if answerable:
        out["false_abstain"] = round(sum(r["abstained"] for r in answerable) / len(answerable), 3)
    out["n"] = len(ok)
    out["errors"] = len(rows) - len(ok)
    return out

def render(result: dict, paper: Paper, dataset_path: str, items: List[dict]) -> str:
    cols = [c for c in SUMMARY_COLUMNS if any(c in s for s in result["summary"].values())]
    table = markdown_table(["mode", "n"] + cols,
                           [[m, s["n"]] + [s.get(c, "-") for c in cols] for m, s in result["summary"].items()])
    n_un = sum(not is_answerable(i) for i in items)
    lines = [f"# Answer evaluation — {paper.title or paper.filename}", "",
             f"- dataset: `{dataset_path}` ({len(items) - n_un} answerable + {n_un} unanswerable questions)", "",
             table, "", "## Answers that missed expected facts", ""]
    misses = [r for r in result["rows"] if r.get("answerable") and r.get("fact_recall", 1) < 1]
    for r in misses[:40]:
        lines.append(f"- **{r['mode']}** (recall {r['fact_recall']:.2f}, context recall "
                     f"{r.get('context_recall', 0):.2f}) — {r['question']}")
    if not misses:
        lines.append("None.")
    lines += ["", "*context recall < 1 means retrieval missed the fact; context recall = 1 with fact recall < 1 "
                  "means the fact was retrieved but the LLM did not use it.*"]
    return "\n".join(lines) + "\n"

def main():
    safe_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--modes", default="naive,simple,agent", help="Comma-separated subset of: " + ", ".join(MODES))
    parser.add_argument("--judge", action="store_true", help="Also score answers with an LLM judge (1 extra call each)")
    parser.add_argument("--limit", type=int, default=0, help="Only use the first N questions")
    parser.add_argument("--offline", action="store_true",
                        help="Smoke test with a toy embedder (no model download). Results are not meaningful.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait after each answer (rate limits)")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"Unknown mode(s): {unknown}. Choose from: {', '.join(MODES)}")
    dataset = load_dataset(args.dataset, args.limit or None)
    paper = load_eval_paper(args.paper, args.offline)

    result = evaluate_answers(paper, dataset["questions"], modes, args.judge, args.sleep)
    report = render(result, paper, args.dataset, dataset["questions"])
    print("\n" + report)
    stem = save_report("answers", {"paper": args.paper, "dataset": args.dataset, **result}, report)
    print(f"Saved {stem}.md and {stem}.json")

if __name__ == "__main__":
    main()

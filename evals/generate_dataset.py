"""Draft an evaluation question set for YOUR paper with the LLM, so you do not have to write it by hand.

Needs GROQ_API_KEY.

Usage (from the ScholarAI folder):
    python -m evals.generate_dataset --paper my.pdf --n 15 --unanswerable 4 --out evals/datasets/my_paper.json

How it works:
  1. Picks fact-dense passages spread over the paper's sections.
  2. Asks the LLM for one question per passage plus the short exact facts that answer it.
  3. KEEPS a question only if every fact literally occurs in the passage (so the answer key cannot be hallucinated)
     and the fact is not given away in the question.
  4. Adds questions the paper does NOT answer (to test that the system says "not covered" instead of inventing).

The output is a DRAFT. Read it once and delete or fix weak questions before you report numbers
(especially the unanswerable ones: the script cannot prove the paper never mentions something).
"""
import argparse
import json
import re
import time
from collections import OrderedDict
from typing import List, Optional

from pydantic import BaseModel, Field

from evals.common import load_eval_paper, safe_stdout
from evals.metrics import norm
from src.llm import structured, LLMError
from src.rag import Paper

class GeneratedQA(BaseModel):
    question: str = Field(description="A specific question answerable from the passage alone")
    expected_facts: List[str] = Field(description="1-3 short exact strings copied from the passage that answer it")

class UnanswerableSet(BaseModel):
    questions: List[str]

def pick_chunks(paper: Paper, n: int) -> List[dict]:
    """Fact-dense chunks, taken round-robin across sections so the questions cover the whole paper."""
    by_section: "OrderedDict[str, List[dict]]" = OrderedDict()
    for c in paper.chunks:
        if c.get("kind") not in ("text", "table") or len(str(c["text"])) < 200:
            continue
        if str(c.get("section", "")) == "Front matter":
            continue
        by_section.setdefault(str(c.get("section")), []).append(c)
    for chunks in by_section.values():
        chunks.sort(key=lambda c: -len(re.findall(r"\d+(?:\.\d+)?%?", str(c["text"]))))
    picked: List[dict] = []
    while len(picked) < n and any(by_section.values()):
        for chunks in by_section.values():
            if chunks and len(picked) < n:
                picked.append(chunks.pop(0))
    return picked

def validate(qa: GeneratedQA, passage: str) -> Optional[dict]:
    """Keep the item only if its answer key is literally in the passage and not leaked by the question."""
    question = (qa.question or "").strip()
    facts = [f.strip() for f in qa.expected_facts if f and f.strip()]
    if len(question) < 15 or not question.endswith("?") or not 1 <= len(facts) <= 3:
        return None
    text, q = norm(passage), norm(question)
    for f in facts:
        nf = norm(f)
        if not 2 <= len(nf) <= 80 or nf not in text or nf in q:
            return None
    return {"question": question, "expected_facts": facts}

def generate_answerable(paper: Paper, n: int, sleep: float = 0.0) -> List[dict]:
    items, seen = [], set()
    for chunk in pick_chunks(paper, int(n * 1.6) + 2):   # over-sample: validation rejects some
        if len(items) >= n:
            break
        passage = str(chunk["text"])
        try:
            qa = structured(f"""Write ONE exam-style question about the passage below, taken from the research paper
"{paper.title}". The question must be specific and answerable from this passage alone, and must make sense
without seeing the passage (name the method / dataset / quantity it asks about; never say "the passage").

Then give expected_facts: 1-3 SHORT strings copied EXACTLY from the passage (a number, a name, a few words)
that a correct answer must contain. Do not put the facts in the question.

Passage (section: {chunk.get('section')}, page {chunk.get('page')}):
{passage}""", GeneratedQA, max_tokens=300)
        except LLMError as e:
            print(f"  [skip] LLM error: {str(e)[:80]}")
            continue
        item = validate(qa, passage)
        if item and norm(item["question"]) not in seen:
            seen.add(norm(item["question"]))
            items.append({**item, "gold_pages": [chunk.get("page")], "section": chunk.get("section"),
                          "answerable": True})
            print(f"  + {item['question'][:90]}")
        else:
            print("  - rejected (facts not literally in the passage, or leaked in the question)")
        if sleep:
            time.sleep(sleep)
    return items

def generate_unanswerable(paper: Paper, n: int) -> List[dict]:
    if n <= 0:
        return []
    overview = "\n".join(f"- {s}" for s in paper.sections()[:30])
    abstract = " ".join(str(c["text"]) for c in paper.chunks if "abstract" in str(c.get("section", "")).lower())[:1500]
    try:
        out = structured(f"""The research paper "{paper.title}" has these sections:
{overview}

Abstract: {abstract or '(not available)'}

Write {n} questions that SOUND relevant to this paper's topic but that the paper almost certainly does NOT answer
(e.g. a dataset, baseline, hardware detail or experiment that is plausible for the field but absent here).
Each question must be specific. Do not write questions about the paper's actual stated results.""",
                         UnanswerableSet, max_tokens=500)
    except LLMError as e:
        print(f"  [skip] could not generate unanswerable questions: {str(e)[:80]}")
        return []
    return [{"question": q.strip(), "expected_facts": [], "answerable": False}
            for q in out.questions[:n] if q and q.strip().endswith("?")]

def main():
    safe_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--out", required=True, help="Where to write the dataset JSON")
    parser.add_argument("--n", type=int, default=12, help="Number of answerable questions")
    parser.add_argument("--unanswerable", type=int, default=3, help="Number of unanswerable questions")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between LLM calls (rate limits)")
    args = parser.parse_args()

    paper = load_eval_paper(args.paper)
    print(f"Generating questions for: {paper.title} ({len(paper.chunks)} chunks)")
    items = generate_answerable(paper, args.n, args.sleep) + generate_unanswerable(paper, args.unanswerable)
    dataset = {
        "description": "AUTO-GENERATED DRAFT. Review every question before reporting numbers: delete vague ones, "
                       "and check that each 'answerable: false' question really is not answered by the paper.",
        "paper": paper.title, "reviewed": False, "questions": items,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    n_ans = sum(i["answerable"] for i in items)
    print(f"\nWrote {args.out}: {n_ans} answerable + {len(items) - n_ans} unanswerable questions. Review it, then run:")
    print(f"  python -m evals.retrieval_eval --paper {args.paper} --dataset {args.out}")
    print(f"  python -m evals.run_eval --paper {args.paper} --dataset {args.out} --judge --sleep 3")

if __name__ == "__main__":
    main()

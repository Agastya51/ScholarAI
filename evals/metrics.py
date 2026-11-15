"""Evaluation metrics for ScholarAI's RAG pipeline. Pure functions, no LLM, no network.

Dataset item format (see evals/datasets/*.json):
    {"question": "...", "expected_facts": ["28.4", "WMT 2014"], "gold_pages": [8], "answerable": true}

A retrieved chunk counts as RELEVANT to an item when it literally contains at least one expected fact
(or, when the item has no expected facts, when it comes from one of the gold pages).
"""
import math
import re
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from src.retrieval import unsupported_numbers

# ---------------------------------------------------------------- text helpers
def norm(s: str) -> str:
    """Lower-case, collapse whitespace, and drop thousands separators so '4,000' matches '4000'."""
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s or "")
    return re.sub(r"\s+", " ", s.lower()).strip()

def is_answerable(item: dict) -> bool:
    return bool(item.get("answerable", True))

def facts_of(item: dict) -> List[str]:
    return [f for f in item.get("expected_facts", []) if f and str(f).strip()]

# ---------------------------------------------------------------- relevance
def chunk_is_relevant(chunk: dict, item: dict) -> bool:
    facts = facts_of(item)
    if facts:
        text = norm(str(chunk.get("text", "")))
        return any(norm(f) in text for f in facts)
    pages = item.get("gold_pages") or []
    return bool(pages) and chunk.get("page") in pages

def relevance_flags(chunks: Sequence[dict], item: dict) -> List[bool]:
    return [chunk_is_relevant(c, item) for c in chunks]

# ---------------------------------------------------------------- retrieval metrics (one question)
def hit(flags: Sequence[bool]) -> float:
    """1 if any retrieved chunk is relevant."""
    return float(any(flags))

def reciprocal_rank(flags: Sequence[bool]) -> float:
    """1 / rank of the first relevant chunk (0 if none). Rewards putting the evidence first."""
    for rank, ok in enumerate(flags, start=1):
        if ok:
            return 1.0 / rank
    return 0.0

def precision(flags: Sequence[bool]) -> float:
    """Share of retrieved chunks that are relevant (how much of the context is signal, not noise)."""
    return sum(flags) / len(flags) if flags else 0.0

def ndcg(flags: Sequence[bool]) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    dcg = sum(1.0 / math.log2(i + 2) for i, ok in enumerate(flags) if ok)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(sum(flags)))
    return dcg / ideal if ideal else 0.0

def context_recall(chunks: Sequence[dict], item: dict) -> float:
    """Share of the expected facts that appear somewhere in the retrieved context.
    This is the ceiling for the answer: a fact that was not retrieved cannot be answered faithfully."""
    facts = facts_of(item)
    if not facts:
        return 1.0
    context = norm(" ".join(str(c.get("text", "")) for c in chunks))
    return sum(norm(f) in context for f in facts) / len(facts)

def retrieval_metrics(chunks: Sequence[dict], item: dict) -> Dict[str, float]:
    flags = relevance_flags(chunks, item)
    return {"hit": hit(flags), "mrr": reciprocal_rank(flags), "precision": precision(flags), "ndcg": ndcg(flags),
            "context_recall": context_recall(chunks, item), "n_chunks": float(len(chunks)),
            "context_chars": float(sum(len(str(c.get("text", ""))) for c in chunks))}

# ---------------------------------------------------------------- answer metrics (one question)
_CITE_RE = re.compile(r"\[p\.\s*(\d+)")
_ABSTAIN_RE = re.compile(
    r"does not (appear to )?(cover|mention|state|specify|provide|report|contain|discuss|address|say|include)"
    r"|doesn't (cover|mention|state|specify|provide|report|contain|discuss|address|say|include)"
    r"|not (mentioned|stated|specified|provided|reported|covered|discussed|addressed|available|found)"
    r"|no (information|mention|evidence|details?) (is |are |was |were )?(provided|available|found|given|about|on|of|regarding)"
    r"|(cannot|can't|could not|couldn't|unable to) (find|determine|answer|locate)"
    r"|outside the scope", re.I)

def fact_recall(answer: str, item: dict) -> float:
    """Share of the expected facts that appear in the answer (exact, case-insensitive substring)."""
    facts = facts_of(item)
    if not facts:
        return 1.0
    a = norm(answer)
    return sum(norm(f) in a for f in facts) / len(facts)

def cited(answer: str) -> bool:
    return bool(_CITE_RE.search(answer or ""))

def cited_pages(answer: str) -> List[int]:
    return [int(p) for p in _CITE_RE.findall(answer or "")]

def citation_precision(answer: str, context_pages: Iterable[int]) -> float:
    """Share of cited pages that were actually among the retrieved passages (1.0 if nothing is cited)."""
    pages = cited_pages(answer)
    allowed = set(context_pages)
    return sum(p in allowed for p in pages) / len(pages) if pages else 1.0

def abstained(answer: str) -> bool:
    """True when the answer says the paper does not contain the information."""
    return bool(_ABSTAIN_RE.search(answer or ""))

def hallucinated_numbers(answer: str, paper_text: str) -> List[str]:
    """Numbers stated in the answer that occur nowhere in the paper (page numbers inside citations are ignored)."""
    return unsupported_numbers(_CITE_RE.sub("[", answer or ""), paper_text)

def answer_metrics(answer: str, item: dict, paper_text: str, context_pages: Iterable[int] = ()) -> Dict[str, float]:
    out = {"cited": float(cited(answer)), "citation_precision": citation_precision(answer, context_pages),
           "hallucinated_numbers": float(len(hallucinated_numbers(answer, paper_text))),
           "abstained": float(abstained(answer))}
    if is_answerable(item):
        out["fact_recall"] = fact_recall(answer, item)
    return out

# ---------------------------------------------------------------- aggregation
def average(rows: Sequence[Dict[str, float]], keys: Sequence[str] = ()) -> Dict[str, float]:
    """Mean of each metric over the rows that report it."""
    keys = list(keys) or sorted({k for r in rows for k in r if isinstance(r[k], (int, float)) and not isinstance(r[k], bool)})
    return {k: round(mean(r[k] for r in rows if k in r), 4) for k in keys if any(k in r for r in rows)}

def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)

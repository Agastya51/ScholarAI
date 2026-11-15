"""Retrieval pipeline built on top of the hybrid VectorStore.

    question
      -> hybrid candidates (BM25 + dense, Reciprocal Rank Fusion)
      -> [query expansion: 2 rewrites + a HyDE passage, fused with the original]      (optional, 1 LLM call)
      -> [reranking: cross-encoder or listwise LLM]                                     (optional)
      -> MMR selection (relevant but not redundant), adaptive k
      -> relevance threshold + confidence label ("high" | "medium" | "low" | "none")

Every stage can be switched on/off (src/config.py) and measured (python -m evals.retrieval_eval).
The confidence label and the grounding check are the runtime quality signals shown in the UI.
"""
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_TOP_K, CANDIDATE_POOL, USE_MMR, MIN_RELEVANCE, HIGH_CONFIDENCE, QUERY_EXPANSION,
    RERANK_MODE, RERANK_MODEL, RERANK_TOP_N,
)
from src.llm import structured, LLMError
from src.rag import Chunk, Paper, excluded_kinds_for, tokenize

ADAPTIVE_K_EXTRA = 2          # extra chunks fetched when the evidence looks weak
KEYWORD_COVERAGE_OK = 0.6     # share of the question's content words a chunk must contain to count as a keyword hit

_STOPWORDS = frozenset("""a an and are as at be been by can did do does for from had has have how in is it its of on or
that the their there this to was were what when where which who whom whose why will with would the paper authors
author used use using does about into than then they them these those any some many much most more
show shows shown describe describes explain explains tell give list say says""".split())

# =========================
# RESULT TYPE
# =========================
@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    confidence: str = "none"                 # high | medium | low | none
    top_score: float = 0.0                   # best cosine similarity between the question and a returned chunk
    queries: List[str] = field(default_factory=list)   # the original question + any expansions
    steps: List[str] = field(default_factory=list)     # human-readable trace of what the pipeline did
    expanded: bool = False
    reranker: str = "none"

    def public(self) -> dict:
        return {"confidence": self.confidence, "top_score": round(self.top_score, 3), "queries": self.queries,
                "steps": self.steps, "expanded": self.expanded, "reranker": self.reranker,
                "chunks": len(self.chunks)}

# =========================
# QUERY EXPANSION (multi-query + HyDE)
# =========================
class QueryExpansion(BaseModel):
    rewrites: List[str] = Field(default_factory=list,
                                description="2 alternative search queries using terminology a paper would use")
    hypothetical_answer: str = Field(default="",
                                     description="1-2 sentences that could plausibly appear in the paper and answer the question")

def expand_query(query: str, title: str = "") -> Optional[QueryExpansion]:
    """One LLM call that bridges the vocabulary gap between how a user asks and how a paper is written.

    - rewrites: the same question in the paper's likely terminology (synonyms, expanded acronyms)
    - hypothetical_answer (HyDE): a fake answer passage; its embedding sits closer to the real passage
      than the embedding of a short question does.
    Returns None if the LLM is unavailable, so retrieval degrades to the un-expanded search.
    """
    try:
        exp = structured(f"""You improve search queries for retrieving passages from one research paper{f' titled "{title}"' if title else ''}.

User question: {query}

Return:
- rewrites: exactly 2 alternative search queries. Use the technical vocabulary a paper would use
  (synonyms, expanded acronyms, the name of the section where the answer would be). Do not answer the question.
- hypothetical_answer: 1-2 sentences written like a passage from the paper that would answer the question.
  Invent plausible wording but NO specific numbers.""", QueryExpansion, max_tokens=400)
    except LLMError:
        return None
    exp.rewrites = [r.strip() for r in exp.rewrites if r and r.strip() and r.strip().lower() != query.lower()][:2]
    exp.hypothetical_answer = (exp.hypothetical_answer or "").strip()[:500]
    return exp if (exp.rewrites or exp.hypothetical_answer) else None

# =========================
# RERANKERS
# =========================
RerankFn = Callable[[str, List[str]], Optional[List[float]]]   # (query, passages) -> one score per passage

_cross_encoder = None
_cross_encoder_failed = False
_custom_reranker: Optional[RerankFn] = None

def set_reranker(fn: Optional[RerankFn]):
    """Inject a reranker (tests / experiments). Pass None to restore the configured one."""
    global _custom_reranker
    _custom_reranker = fn

def cross_encoder_scores(query: str, passages: List[str]) -> Optional[List[float]]:
    """Score (query, passage) pairs jointly with a small ONNX cross-encoder. None if it cannot be loaded."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder_failed:
        return None
    try:
        if _cross_encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            _cross_encoder = TextCrossEncoder(model_name=RERANK_MODEL, threads=1)
        return [float(s) for s in _cross_encoder.rerank(query, passages, batch_size=8)]
    except Exception as e:  # missing model / no memory / no network: keep serving without reranking
        _cross_encoder_failed = True
        print(f"[Reranker] cross-encoder unavailable ({str(e)[:100]}); continuing without reranking")
        return None

class Ranking(BaseModel):
    ranking: List[int] = Field(description="Passage numbers from most to least useful; omit irrelevant passages")

def llm_rerank_scores(query: str, passages: List[str]) -> Optional[List[float]]:
    """Listwise reranking with one LLM call. Passages the model omits get score 0."""
    listing = "\n\n".join(f"[{i}] {p[:450]}" for i, p in enumerate(passages))
    try:
        out = structured(f"""Rank the passages by how useful they are for answering the question.
Return passage numbers from most to least useful. Leave out passages that do not help.

Question: {query}

Passages:
{listing}""", Ranking, max_tokens=300)
    except LLMError:
        return None
    scores = [0.0] * len(passages)
    seen = set()
    for pos, i in enumerate(out.ranking):
        if 0 <= i < len(passages) and i not in seen:
            seen.add(i)
            scores[i] = 1.0 / (pos + 1)
    return scores if seen else None

def rerank_scores(mode: str, query: str, passages: List[str]) -> Optional[List[float]]:
    if _custom_reranker is not None:
        return _custom_reranker(query, passages)
    if mode == "cross-encoder":
        return cross_encoder_scores(query, passages)
    if mode == "llm":
        return llm_rerank_scores(query, passages)
    return None

# =========================
# CONFIDENCE
# =========================
def content_terms(query: str) -> List[str]:
    return [t for t in dict.fromkeys(tokenize(query)) if t not in _STOPWORDS and len(t) > 1]

def keyword_coverage(queries: Sequence[str], text: str) -> float:
    """Best share of a query's content words that literally occur in the text."""
    tokens = set(tokenize(text))
    best = 0.0
    for q in queries:
        terms = content_terms(q)
        if terms:
            best = max(best, sum(t in tokens for t in terms) / len(terms))
    return best

def confidence_label(top_score: float, coverage: float) -> str:
    if top_score >= HIGH_CONFIDENCE:
        return "high"
    if top_score >= MIN_RELEVANCE or coverage >= KEYWORD_COVERAGE_OK:
        return "medium"
    return "low"

# =========================
# PIPELINE
# =========================
def _select(paper: Paper, query: str, queries: List[str], hyde: str, k: int, mode: str, mmr: bool,
            rerank: str, threshold: bool, steps: List[str]) -> RetrievalResult:
    store = paper.store
    n = len(store.documents)
    pool = min(n, max(k * 4, CANDIDATE_POOL))
    exclude = excluded_kinds_for(query)

    rankings: List[List[int]] = []
    for q in queries:
        rankings += store.rankings(q, pool, mode, exclude)
    if hyde and mode != "sparse":
        rankings += store.rankings(hyde, pool, "dense", exclude)
    refs = store.reference_ranking(query)     # non-empty only for questions about the bibliography
    pinned = set(refs[:2])
    if refs:
        rankings += [refs, refs]
    fused = store.fuse(rankings)
    ranked = sorted(fused, key=fused.get, reverse=True)[:pool]
    if not ranked:
        return RetrievalResult([], "none", 0.0, queries, steps + ["no candidates"])
    steps.append(f"{mode} search: {len(ranked)} candidates from {len(queries) + bool(hyde)} quer{'y' if len(queries) + bool(hyde) == 1 else 'ies'}")

    # Cosine similarity of each candidate to the best-matching phrasing of the question
    per_query = [store.dense_scores(q) for q in queries]
    dense = {i: max(float(scores[i]) for scores in per_query) for i in ranked}
    coverage = {i: keyword_coverage(queries, str(store.metadata[i].get("embed_text") or store.documents[i]))
                for i in ranked}

    relevance: Dict[int, float] = {i: fused[i] for i in ranked}
    used_reranker = "none"
    if rerank in ("cross-encoder", "llm") or _custom_reranker is not None:
        head = ranked[:RERANK_TOP_N]
        scores = rerank_scores(rerank, query, [store.documents[i] for i in head])
        if scores is not None:
            used_reranker = rerank if _custom_reranker is None else "custom"
            relevance = {i: float(s) for i, s in zip(head, scores)}
            ranked = sorted(head, key=relevance.get, reverse=True)
            steps.append(f"reranked top {len(head)} with {used_reranker}")
        else:
            steps.append(f"reranker '{rerank}' unavailable; kept fusion order")

    # Adaptive k: when the best evidence is weak, read a little wider
    best = max(ranked, key=lambda i: dense[i])
    pre_label = confidence_label(dense[best], max(coverage.values()))
    k_eff = k if pre_label == "high" else min(len(ranked), k + ADAPTIVE_K_EXTRA)
    if k_eff != k:
        steps.append(f"evidence looks {pre_label}: widened k from {k} to {k_eff}")

    selected = store.mmr(ranked, relevance, k_eff) if mmr else ranked[:k_eff]
    if mmr:
        steps.append(f"MMR selected {len(selected)} diverse chunks")

    if threshold:
        kept = [i for i in selected
                if dense[i] >= MIN_RELEVANCE or coverage[i] >= KEYWORD_COVERAGE_OK or i in pinned]
        if len(kept) < len(selected):
            steps.append(f"dropped {len(selected) - len(kept)} chunks below the relevance threshold")
        # Never return nothing: keep the two best so the answerer can still say "not covered"
        selected = kept or selected[:2]

    chunks = store.describe(selected, fused, query)
    for c in chunks:
        i = int(c["chunk_id"])
        c["dense_score"] = round(dense[i], 4)
        c["coverage"] = round(coverage[i], 2)
        if used_reranker != "none":
            c["rerank_score"] = round(relevance.get(i, 0.0), 4)
    top = max((float(c["dense_score"]) for c in chunks), default=0.0)
    label = confidence_label(top, max((float(c["coverage"]) for c in chunks), default=0.0)) if chunks else "none"
    return RetrievalResult(chunks, label, top, queries, steps, expanded=len(queries) > 1 or bool(hyde),
                           reranker=used_reranker)

def retrieve_context(paper: Paper, query: str, k: int = DEFAULT_TOP_K, *, mode: str = "hybrid",
                     mmr: Optional[bool] = None, expansion: Optional[str] = None, rerank: Optional[str] = None,
                     threshold: bool = True) -> RetrievalResult:
    """Run the full retrieval pipeline for one question against one paper.

    expansion: "off" | "auto" (expand only if the first pass has low confidence) | "always"
    rerank:    "none" | "cross-encoder" | "llm"
    """
    query = (query or "").strip()
    if not query or not len(paper.store.documents):
        return RetrievalResult([], "none", 0.0, [query] if query else [], ["empty query or empty index"])
    mmr = USE_MMR if mmr is None else mmr
    expansion = (expansion or QUERY_EXPANSION).lower()
    rerank = (rerank or RERANK_MODE).lower()

    if expansion == "always":
        exp = expand_query(query, paper.title)
        steps = ["query expansion: " + ("; ".join(exp.rewrites) + (" + HyDE passage" if exp.hypothetical_answer else "")
                                        if exp else "unavailable")]
        queries = [query] + (exp.rewrites if exp else [])
        return _select(paper, query, queries, exp.hypothetical_answer if exp else "", k, mode, mmr, rerank,
                       threshold, steps)

    first = _select(paper, query, [query], "", k, mode, mmr, rerank, threshold, [])
    if expansion != "auto" or first.confidence not in ("low", "none"):
        return first

    # Corrective step: weak evidence -> rewrite the question and search again
    exp = expand_query(query, paper.title)
    if exp is None:
        first.steps.append("low confidence, but query expansion was unavailable")
        return first
    steps = first.steps + ["low confidence -> query expansion: " + "; ".join(exp.rewrites)
                           + (" + HyDE passage" if exp.hypothetical_answer else "")]
    second = _select(paper, query, [query] + exp.rewrites, exp.hypothetical_answer, k, mode, mmr, rerank,
                     threshold, steps)
    if second.top_score >= first.top_score or second.confidence != "low":
        return second
    first.steps = steps + ["expansion did not find better evidence; kept the first pass"]
    return first

# =========================
# RUNTIME GROUNDING CHECK (no LLM call)
# =========================
_NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
_CITE_RE = re.compile(r"\[p\.\s*(\d+)")

def significant_numbers(text: str) -> set:
    """Numbers worth fact-checking: decimals, percentages, or 2+ digit values."""
    return {n.rstrip("%") for n in _NUM_RE.findall(text or "") if "." in n or "%" in n or len(n) >= 2}

def unsupported_numbers(text: str, source_text: str) -> List[str]:
    return sorted(significant_numbers(text) - significant_numbers(source_text))

def check_grounding(answer: str, chunks: List[Chunk]) -> dict:
    """Cheap hallucination signals for an answer, computed against the passages it was given.

    - unsupported_numbers: numbers in the answer that appear in none of the passages
    - unsupported_citations: cited pages ([p. N]) that were not among the retrieved pages
    - cited: whether the answer cites at least one page
    """
    body = _CITE_RE.sub("[", answer or "")          # page numbers inside citations are not claims
    context = "\n".join(str(c["text"]) for c in chunks)
    pages = {int(c["page"]) for c in chunks if c.get("page")}
    cited = [int(p) for p in _CITE_RE.findall(answer or "")]
    bad_numbers = unsupported_numbers(body, context)
    bad_pages = sorted({p for p in cited if p not in pages})
    return {"cited": bool(cited), "unsupported_numbers": bad_numbers, "unsupported_citations": bad_pages,
            "grounded": not bad_numbers and not bad_pages}

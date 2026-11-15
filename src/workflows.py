import operator
import re
from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.agents.base import trace_event
from src.config import (
    MAX_FIGURES_ANALYZED, SEARCH_MAX_ATTEMPTS, SEARCH_MIN_RELEVANT, REWRITE_MAX_REVISIONS
)
from src.figures import explain_paper_figure, explain_figure, figure_context, figures_named_in
from src.llm import llm, structured, LLMError, LLMBudgetExceeded
from src.rag import library, gather_context, search_chunks, format_chunks
from src.retrieval import retrieve_context, check_grounding, significant_numbers, unsupported_numbers
from src.search import clean_query, multi_engine_search

# =========================
# LANGGRAPH STATE
# =========================
class PaperState(TypedDict, total=False):
    paper_id: Optional[str]
    summary: str
    vision: List[str]
    topic: str
    # compare loop
    queries_tried: List[str]
    attempt: int
    candidates: List[dict]
    papers: List[dict]
    grade_reason: str
    comparison: str
    # improve loop
    improvements: str
    edits: List[dict]
    revision: int
    # simple (non-agentic) RAG Q&A
    query: str
    answer: str
    sources: List[dict]
    retrieval: dict
    trace: Annotated[list, operator.add]

def _paper(state: PaperState):
    paper = library.get(state.get("paper_id"))
    if paper is None:
        raise ValueError("No paper loaded. Please upload a paper first.")
    return paper

# =========================
# UPLOAD: summarize || vision -> topic
# =========================
SUMMARY_QUERIES = [
    "abstract problem statement motivation",
    "methodology approach proposed method",
    "results experiments evaluation metrics",
    "conclusion limitations future work",
]

def node_summarize(state: PaperState) -> dict:
    """Generate structured summary using RAG across all key paper sections."""
    paper = _paper(state)
    context = gather_context(SUMMARY_QUERIES, k=3, paper_id=paper.id)

    prompt = f"""You are a precise research analyst. Summarize this paper using ONLY facts found in the text below.

STRICT RULES — violations will make your response useless:
- NEVER write generic phrases like "The authors propose...", "This paper presents...", "The study shows..."
- ALWAYS use the EXACT method name, algorithm name, or system name from the paper (e.g. "ResNet-50", "OFDM", "BERT", "Doppler-aware power control")
- ALWAYS include EXACT numbers: accuracy %, dB gains, dataset size, parameter count, benchmark scores
- If the paper uses a specific dataset, name it exactly (e.g. "ImageNet", "KITTI", "MNIST")
- If a baseline is compared, name it exactly
- Do NOT invent or infer — only report what is literally in the text
- Passages are prefixed with [p. N · Section] anchors; cite the page for key numbers, e.g. "(p. 6)"

## TLDR
One sentence. Must include: the specific problem + the exact method name + the top result with a number.

## Problem
What exact technical problem or gap does this paper address? Quote the specific claim or limitation it targets.

## Method
Name the exact proposed system/algorithm/architecture. Describe its key components using terms from the paper.

## Results
List EXACT metrics: e.g. "Achieved 94.3% accuracy on [dataset], outperforming [baseline] by 2.1%". No vague statements.

## Limitations
Quote or paraphrase the actual limitations the authors acknowledge. Do NOT invent limitations.

Paper text (sampled from full document via semantic retrieval):
{context}"""
    summary = llm(prompt)
    paper.summary = summary
    return {"summary": summary, "trace": [trace_event("reader", "output", summary)]}

def figure_markdown(fig: dict, analysis: str) -> str:
    """One figure as markdown: where it is, its caption, where the text discusses it, and the explanation."""
    header = f"### {fig.get('label') or fig['id']} · page {fig['page']}"
    if fig.get("caption"):
        header += f"\n\n> {fig['caption']}"
    pages = sorted({m["page"] for m in fig.get("mentions", [])})
    if pages:
        header += f"\n\n*Discussed in the text on page{'s' if len(pages) > 1 else ''} {', '.join(map(str, pages))}.*"
    return f"{header}\n\n{analysis}"

def node_vision(state: PaperState) -> dict:
    """Explain up to MAX_FIGURES_ANALYZED figures. Each explanation uses the image, the caption, and every
    sentence of the paper that refers to the figure (cached per paper)."""
    paper = _paper(state)
    results, events = [], []
    for fig in paper.figures[:MAX_FIGURES_ANALYZED]:
        try:
            out = explain_paper_figure(paper, fig["id"])
        except LLMBudgetExceeded:
            raise
        except LLMError as e:
            events.append(trace_event("figure_analyst", "error", f"{fig['id']}: {e}"))
            continue
        results.append(figure_markdown(fig, out["text"]))
        events.append(trace_event(
            "figure_analyst", "tool_call",
            f"{fig['id']} (p. {fig['page']}): image + caption + {len(fig.get('mentions', []))} in-text mention(s)"
            + ("" if out["mode"] == "vision" else " [text-only fallback]")))
    events.append(trace_event("figure_analyst", "output", f"Explained {len(results)} of {len(paper.figures)} figures"))
    return {"vision": results, "trace": events}

def node_extract_topic(state: PaperState) -> dict:
    """Extract high-density technical keywords from summary for academic search."""
    prompt = f"""Extract 3-4 high-density technical keywords from this paper summary.
Focus on core methodology and technical domain (e.g., 'Amplitude Modulation', 'Deep Learning').
Return ONLY keywords separated by commas.

Summary:
{state['summary']}"""
    topic = clean_query(llm(prompt).strip())
    return {"topic": topic, "trace": [trace_event("reader", "decision", f"Search keywords: {topic}")]}

# =========================
# COMPARE: search -> grade -> (reformulate -> search)* -> compare
# =========================
class RelevanceGrade(BaseModel):
    relevant: List[int] = Field(description="Indices of candidates genuinely related to the original paper")
    reason: str = Field(description="One sentence on why the rest were rejected or what is missing")

class Reformulation(BaseModel):
    keywords: List[str] = Field(description="3-4 new search keywords")
    strategy: str = Field(description="How these differ from the queries already tried (broader, synonyms, sub-problem...)")

def node_search(state: PaperState) -> dict:
    topic = state.get("topic", "")
    keywords = [k.strip() for k in clean_query(topic).split(",") if k.strip()]
    found = multi_engine_search(keywords, state.get("summary", ""), limit=10)
    attempt = state.get("attempt", 0) + 1
    return {"candidates": found, "attempt": attempt, "queries_tried": state.get("queries_tried", []) + [topic],
            "trace": [trace_event("literature_scout", "tool_call",
                                  f"Attempt {attempt}: searched '{topic}' -> {len(found)} candidates")]}

def node_grade(state: PaperState) -> dict:
    """LLM grader filters candidates for true relevance; relevant papers accumulate across attempts."""
    candidates = state.get("candidates", [])
    kept = list(state.get("papers", []))
    if not candidates:
        return {"grade_reason": "No results returned.",
                "trace": [trace_event("literature_scout", "decision", "No candidates returned by any engine")]}

    listing = "\n".join(f"{i}. {p['title']} ({p.get('year', '')}): {(p.get('summary') or '')[:300]}"
                        for i, p in enumerate(candidates))
    grade = structured(f"""Decide which candidate papers are genuinely related to the original paper
(same problem, method family, or application domain). Be strict: keyword overlap alone is not enough.

Original paper summary:
{state.get('summary') or state.get('topic')}

Candidates:
{listing}""", RelevanceGrade)

    titles = {p["title"].lower() for p in kept}
    for i in grade.relevant:
        if 0 <= i < len(candidates) and candidates[i]["title"].lower() not in titles:
            kept.append(candidates[i])
            titles.add(candidates[i]["title"].lower())
    kept.sort(key=lambda p: p.get("relevance_score", 0), reverse=True)
    return {"papers": kept[:8], "grade_reason": grade.reason,
            "trace": [trace_event("literature_scout", "decision",
                                  f"{len(grade.relevant)}/{len(candidates)} relevant (total kept {len(kept)}). {grade.reason}")]}

def route_after_grade(state: PaperState) -> str:
    if len(state.get("papers", [])) >= SEARCH_MIN_RELEVANT or state.get("attempt", 0) >= SEARCH_MAX_ATTEMPTS:
        return "compare"
    return "reformulate"

def node_reformulate(state: PaperState) -> dict:
    ref = structured(f"""The literature search did not find enough relevant papers. Propose NEW search keywords.

Original paper summary:
{state.get('summary') or '(not available)'}

Queries already tried (do not repeat them): {state.get('queries_tried', [])}
Grader feedback: {state.get('grade_reason', '')}

Prefer broader terms, common synonyms used in the field, or the core sub-problem.""", Reformulation)
    topic = ", ".join(clean_query(k) for k in ref.keywords if k.strip())
    return {"topic": topic,
            "trace": [trace_event("literature_scout", "thought", f"Reformulated to '{topic}' ({ref.strategy})")]}

def node_compare(state: PaperState) -> dict:
    """Compare uploaded paper with recent academic literature."""
    papers = state.get("papers", [])
    if not papers:
        msg = ("I couldn't retrieve related papers after "
               f"{state.get('attempt', 0)} search attempts ({'; '.join(state.get('queries_tried', []))}). "
               "External APIs may be unreachable or the topic too specific. Edit the topic and re-run.")
        return {"comparison": msg, "trace": [trace_event("comparator", "error", msg)]}

    combined = "\n\n".join([
        f"[{p['year']}] {p['title']} ({p.get('venue', 'Academic Source')}): {p['summary']}"
        for p in papers if p.get('has_abstract')
    ])
    papers_without = [p for p in papers if not p.get('has_abstract')]
    skip_notice = ""
    if papers_without:
        skip_names = ", ".join([f'"{p["title"]}"' for p in papers_without])
        skip_notice = f"\n\nNOTE: The following papers lack full public abstracts: {skip_names}. Do NOT infer details for them."

    prompt = f"""You are a research analyst. Compare the original paper with recent research using markdown.

IMPORTANT RULES:
- ONLY analyze papers for which abstract/summary text is provided below.
- Do NOT write "Not Reported" or "No specific data" for any paper. If a paper lacks enough information, exclude it from analysis.
- Every paper you mention MUST have concrete evidence from its abstract.

Original Paper Summary:
{state.get('summary', '')}

Recent Related Research (with abstracts):
{combined}{skip_notice}

## Key Differences
How does the original differ from recent work?

## Improvements in Recent Work
What have newer papers improved upon?

## Missing Ideas
What concepts from recent research are absent in the original?

## Strengths of Original
What does the original do well?"""
    comparison = llm(prompt)
    return {"comparison": comparison, "trace": [trace_event("comparator", "output", comparison)]}

# =========================
# IMPROVE: critique -> rewrite -> verify -> (revise -> verify)* -> finalize
# =========================
class Edit(BaseModel):
    section: str
    original: str = Field(description="First 100-150 characters of the section copied verbatim from the paper")
    rewritten: str = Field(description="Complete improved text for the section")

class EditSet(BaseModel):
    edits: List[Edit]

class Verdict(BaseModel):
    index: int
    verdict: Literal["pass", "revise"]
    faithfulness: int = Field(ge=1, le=5, description="5 = every claim and number is supported by the evidence")
    issues: List[str] = Field(default_factory=list)

class VerdictSet(BaseModel):
    verdicts: List[Verdict]

IMPROVE_QUERIES = [
    "abstract introduction background",
    "methodology approach proposed solution",
    "results experiments analysis discussion",
    "conclusion limitations future scope",
]

def node_critique(state: PaperState) -> dict:
    """Identify weak sections with publication-quality critique."""
    paper = _paper(state)
    context = gather_context(IMPROVE_QUERIES, k=3, paper_id=paper.id)

    prompt = f"""You are a senior research advisor reviewing this paper for publication-level quality.
Your job is to find SPECIFIC, CONCRETE weaknesses — not generic advice.

STRICT RULES:
- Quote the EXACT weak sentence or paragraph from the paper text (use quotation marks) and its page anchor
- Explain WHY that specific sentence is weak (too vague, missing numbers, missing justification, etc.)
- Suggest the EXACT type of content that should replace it (e.g. "add ablation study comparing X vs Y", "cite [specific metric] from the results")
- NEVER write generic advice like "Improve clarity", "Add more details", "Strengthen the argument"
- If a section is actually strong, say so — do not fabricate weaknesses
- Reference the comparative analysis to point out what competing papers do that this paper does not

Paper text (full document via semantic retrieval, passages prefixed with [page · section]):
{context}

Comparative analysis with recent related work:
{(state.get('comparison') or 'Not available.')[:1500]}

For each weak section, provide:
1. The exact quoted text that is weak
2. The specific reason it is weak
3. A concrete, actionable fix

Use markdown headings: ## Abstract, ## Introduction, ## Methodology, ## Results, ## Conclusion"""
    improvements = llm(prompt)
    return {"improvements": improvements, "revision": 0, "trace": [trace_event("reviewer", "output", improvements)]}

REWRITE_RULES = """STRICT RULES for rewrites:
- Use the EXACT same technical terminology, variable names, and method names as in the original paper
- Include REAL numbers from the paper (accuracy, loss values, dataset sizes, etc.) — do NOT invent statistics
- Do NOT add claims the paper does not support
- Strengthen weak sentences by adding specificity: replace vague claims with exact measurement references
- The rewritten version must be directly usable in the paper — no placeholders like "[add result here]"
- Match the academic writing style of the original
- "original" must be copied CHARACTER FOR CHARACTER from the paper text (100-150 chars, without the [p. N] anchor)"""

def node_rewrite(state: PaperState) -> dict:
    """Generate structured section rewrites."""
    paper = _paper(state)
    context = gather_context(["abstract introduction", "methodology proposed method",
                              "results experiments metrics", "conclusion limitations"], k=3, paper_id=paper.id)
    result = structured(f"""You are a research paper editor rewriting specific weak sections.
The rewritten text must sound like it was written by the original authors — technical, precise, grounded in the paper's own data.

{REWRITE_RULES}

Full paper text (via semantic retrieval):
{context}

Improvement analysis:
{state.get('improvements', '')[:1500]}

For each section that needs improvement (and whose text you can find verbatim above), return an edit.""", EditSet)
    edits = [{**e.model_dump(), "verification": {"status": "pending", "issues": [], "attempts": 0}}
             for e in result.edits]
    return {"edits": edits, "trace": [trace_event("editor", "output", f"Drafted {len(edits)} section rewrites")]}

_significant_numbers = significant_numbers  # kept for backwards compatibility

def anchor_found(original: str, source_text: str) -> bool:
    norm = lambda s: re.sub(r"\s+", " ", s or "").strip().lower()
    return bool(original) and norm(original)[:80] in norm(source_text)

def node_verify(state: PaperState) -> dict:
    """Evaluator: deterministic number/anchor checks + LLM critic for factual drift and unsupported claims."""
    paper = _paper(state)
    edits = [dict(e) for e in state.get("edits", [])]
    to_check = [i for i, e in enumerate(edits) if e["verification"]["status"] in ("pending", "needs_revision")]
    if not to_check:
        return {"edits": edits}

    blocks = []
    for i in to_check:
        e = edits[i]
        evidence = format_chunks(search_chunks(f"{e['original']} {e['rewritten'][:200]}", 4, paper.id))
        blocks.append(f"### Edit {i} — {e['section']}\nREWRITTEN:\n{e['rewritten']}\n\nEVIDENCE FROM PAPER:\n{evidence}")
    verdicts = structured("""You are a strict fact-checker for paper rewrites. For each edit, compare the REWRITTEN
text against the EVIDENCE FROM PAPER. Mark "revise" if it adds numbers, claims, baselines, or results that the
evidence does not support, changes the paper's terminology, or contains placeholders. Otherwise "pass".
List each concrete problem in issues.

""" + "\n\n".join(blocks), VerdictSet)
    by_index = {v.index: v for v in verdicts.verdicts}

    events = []
    for i in to_check:
        e = edits[i]
        v = by_index.get(i)
        issues = list(v.issues) if v else ["Critic returned no verdict"]
        bad_numbers = unsupported_numbers(e["rewritten"], paper.text)
        if bad_numbers:
            issues.append(f"Numbers not found anywhere in the paper: {', '.join(bad_numbers[:8])}")
        if not anchor_found(e["original"], paper.text):
            issues.append("'original' snippet is not verbatim paper text")
        passed = bool(v) and v.verdict == "pass" and not bad_numbers
        e["verification"] = {"status": "verified" if passed else "needs_revision", "issues": issues,
                             "faithfulness": v.faithfulness if v else None,
                             "attempts": e["verification"].get("attempts", 0) + 1}
        events.append(trace_event("fact_checker", "decision",
                                  f"Edit {i} ({e['section']}): {'PASS' if passed else 'REVISE'}"
                                  + (f" — {'; '.join(issues)}" if issues else "")))
    return {"edits": edits, "trace": events}

def route_after_verify(state: PaperState) -> str:
    pending = any(e["verification"]["status"] == "needs_revision" for e in state.get("edits", []))
    return "revise" if pending and state.get("revision", 0) < REWRITE_MAX_REVISIONS else "finalize"

def node_revise(state: PaperState) -> dict:
    """Optimizer: rewrite only the failing edits using the critic's feedback."""
    paper = _paper(state)
    edits = [dict(e) for e in state.get("edits", [])]
    failing = [i for i, e in enumerate(edits) if e["verification"]["status"] == "needs_revision"]
    blocks = []
    for i in failing:
        e = edits[i]
        evidence = format_chunks(search_chunks(f"{e['section']} {e['original']}", 4, paper.id))
        blocks.append(f"### {e['section']}\nORIGINAL ANCHOR: {e['original']}\nPREVIOUS REWRITE:\n{e['rewritten']}\n"
                      f"FACT-CHECK ISSUES: {e['verification']['issues']}\nEVIDENCE:\n{evidence}")
    result = structured(f"""Revise these section rewrites so every issue is fixed. Remove anything not supported
by the evidence rather than inventing support. Return the edits in the same order.

{REWRITE_RULES}

""" + "\n\n".join(blocks), EditSet)
    for i, new in zip(failing, result.edits):
        attempts = edits[i]["verification"].get("attempts", 0)
        edits[i] = {**new.model_dump(), "verification": {"status": "needs_revision", "issues": [], "attempts": attempts}}
    revision = state.get("revision", 0) + 1
    return {"edits": edits, "revision": revision,
            "trace": [trace_event("editor", "thought", f"Revision round {revision}: rewrote {len(failing)} edits")]}

def node_finalize_edits(state: PaperState) -> dict:
    edits = [dict(e) for e in state.get("edits", [])]
    for e in edits:
        if e["verification"]["status"] != "verified":
            e["verification"] = {**e["verification"], "status": "flagged"}
    n_ok = sum(e["verification"]["status"] == "verified" for e in edits)
    return {"edits": edits,
            "trace": [trace_event("fact_checker", "output", f"{n_ok}/{len(edits)} rewrites verified; rest flagged")]}

# =========================
# SIMPLE RAG Q&A (retrieval pipeline + one grounded answer call)
# =========================
QA_PROMPT = """You are a research assistant answering a question about one paper.

Rules:
- Use ONLY the context passages below. Do not use outside knowledge.
- Cite the page anchor for every claim, e.g. [p. 3].
- Copy numbers exactly as they appear in the context.
- If the context does not contain the answer, reply exactly: "The paper does not appear to cover this."
  Then, in one sentence, say what the closest related information in the context is (if any).
- Be concise. Use markdown.

Context:
{context}

Question: {query}

Answer:"""

def node_qa_simple(state: PaperState) -> dict:
    """Non-agentic Q&A: hybrid retrieval pipeline (MMR, threshold, corrective query expansion) + one answer call.

    Returns a `retrieval` report with a confidence label and a grounding check (runtime quality signals).
    """
    paper = library.get(state.get("paper_id"))
    result = retrieve_context(paper, state["query"]) if paper else None
    if not result or not result.chunks:
        return {"answer": "No document loaded or no matching context found. Please upload a paper first.",
                "sources": [], "retrieval": result.public() if result else {"confidence": "none"}}
    chunks = result.chunks
    events_pre = []
    # A question that names a figure ("What does Fig. 3 show?") is answered from the figure itself:
    # the image, its caption, and the sentences that cite it, wherever they are in the paper.
    for fig in figures_named_in(paper, state["query"])[:2]:
        try:
            seen = explain_figure(fig, question=state["query"])
            text = f"{fig.get('caption') or fig.get('label')}\n{figure_context(fig)}\nAnswer from the figure ({seen['mode']}): {seen['text']}"
            events_pre.append(trace_event("figure_analyst", "tool_call", f"looked at {fig['id']} ({seen['mode']})"))
        except LLMBudgetExceeded:
            raise
        except LLMError as e:
            text = figure_context(fig)
            events_pre.append(trace_event("figure_analyst", "error", f"{fig['id']}: {e}"))
        if text:
            chunks = [{"text": text, "page": fig["page"], "section": fig.get("section") or "", "kind": "figure",
                       "figure_id": fig["id"], "dense_score": None}] + chunks
    answer = llm(QA_PROMPT.format(context=format_chunks(chunks), query=state["query"]))
    report = {**result.public(), "grounding": check_grounding(answer, chunks)}
    events = events_pre + [trace_event("retriever", "tool_call", step) for step in result.steps]
    events.append(trace_event("retriever", "decision",
                              f"confidence {result.confidence} (top similarity {result.top_score:.2f})"))
    events.append(trace_event("qa_simple", "output", "single grounded answer"))
    return {"answer": answer, "retrieval": report,
            "sources": [({"type": "figure", "paper_id": paper.id, "page": c.get("page"), "figure_id": c["figure_id"]}
                         if c.get("kind") == "figure" and c.get("chunk_id") is None else
                         {"type": "paper", "paper_id": paper.id, "page": c.get("page"), "section": c.get("section"),
                          "kind": c.get("kind"), "score": c.get("dense_score"), "chunk_id": c.get("chunk_id"),
                          "figure_id": c.get("figure_id"), "snippet": str(c["text"])[:220]}) for c in chunks],
            "trace": events}

# =========================
# GRAPH PIPELINE BUILDERS
# =========================
def build_upload_graph():
    g = StateGraph(PaperState)
    g.add_node("summarize", node_summarize)
    g.add_node("vision", node_vision)
    g.add_node("extract_topic", node_extract_topic)
    g.add_edge(START, "summarize")
    g.add_edge(START, "vision")
    g.add_edge(["summarize", "vision"], "extract_topic")
    g.add_edge("extract_topic", END)
    return g.compile()

def build_prepare_graph():
    """Summary and search keywords only. Used right after an upload; figures are explained on demand, so a
    single request never has to wait for every figure on a slow instance."""
    g = StateGraph(PaperState)
    g.add_node("summarize", node_summarize)
    g.add_node("extract_topic", node_extract_topic)
    g.add_edge(START, "summarize")
    g.add_edge("summarize", "extract_topic")
    g.add_edge("extract_topic", END)
    return g.compile()

def build_compare_graph(include_compare: bool = True):
    """Self-correcting literature search; with include_compare=False it stops after the search loop."""
    g = StateGraph(PaperState)
    g.add_node("search", node_search)
    g.add_node("grade", node_grade)
    g.add_node("reformulate", node_reformulate)
    g.add_edge(START, "search")
    g.add_edge("search", "grade")
    g.add_edge("reformulate", "search")
    if include_compare:
        g.add_node("compare", node_compare)
        g.add_edge("compare", END)
    g.add_conditional_edges("grade", route_after_grade,
                            {"compare": "compare" if include_compare else END, "reformulate": "reformulate"})
    return g.compile()

def build_improve_graph():
    g = StateGraph(PaperState)
    g.add_node("critique", node_critique)
    g.add_node("rewrite", node_rewrite)
    g.add_node("verify", node_verify)
    g.add_node("revise", node_revise)
    g.add_node("finalize", node_finalize_edits)
    g.add_edge(START, "critique")
    g.add_edge("critique", "rewrite")
    g.add_edge("rewrite", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"revise": "revise", "finalize": "finalize"})
    g.add_edge("revise", "verify")
    g.add_edge("finalize", END)
    return g.compile()

def build_qa_simple_graph():
    g = StateGraph(PaperState)
    g.add_node("qa", node_qa_simple)
    g.add_edge(START, "qa")
    g.add_edge("qa", END)
    return g.compile()

upload_graph    = build_upload_graph()
prepare_graph   = build_prepare_graph()
compare_graph   = build_compare_graph()
search_graph    = build_compare_graph(include_compare=False)
improve_graph   = build_improve_graph()
qa_simple_graph = build_qa_simple_graph()

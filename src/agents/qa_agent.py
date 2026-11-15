"""Tool-augmented Q&A agent: decides when to search the paper, read tables/figures, or query external literature."""
import re
from typing import List, Optional

from src.agents.base import build_tool_agent, run_tool_agent, dedupe_sources
from src.config import QA_AGENT_MAX_ITERATIONS, HIGH_CONFIDENCE, MIN_RELEVANCE
from src.rag import library
from src.tools import PAPER_TOOLS, LITERATURE_TOOLS

QA_SYSTEM = """You are ScholarAI's research assistant. The user is asking about the uploaded paper "{title}"{paper_hint}.

How to work:
- Ground every answer in evidence gathered with tools. Usually start with search_paper using specific terms;
  run several focused searches rather than one vague one. Use get_section to read a whole section.
- For exact numbers check tables (list_tables / get_table); for diagrams use list_figures / analyze_figure.
- Questions that link two parts of the paper (e.g. a method and a result) need one search per part.
- If search_paper says its confidence is LOW, rephrase with the paper's own terminology and search again.
- For "what is reference [12]" style questions use get_reference.
- For any question about a figure, call analyze_figure with the figure id and the user's question. It reads the
  image together with the caption and the sentences that cite the figure, even when they are on other pages.
- Use search_literature or lookup_paper ONLY when the question is about work outside this paper
  (external baselines, prior work, state of the art, cited papers).
- Stop calling tools as soon as you have enough evidence.

How to answer:
- Concise markdown. Cite every claim from the paper with its anchor, e.g. [p. 4] or [p. 4 · 3.1 Setup];
  cite external papers by title.
- If the evidence does not answer the question, say so plainly. Never invent numbers or results."""

qa_agent_graph = build_tool_agent("qa_agent", PAPER_TOOLS + LITERATURE_TOOLS + ["list_papers"], QA_AGENT_MAX_ITERATIONS)

def history_to_messages(history: Optional[List[dict]], max_turns: int = 6) -> List[dict]:
    """Convert UI history [{q, a}] into chat turns (most recent `max_turns`)."""
    messages = []
    for h in (history or [])[-max_turns:]:
        if h.get("q") and h.get("a"):
            messages.append({"role": "user", "content": h["q"]})
            messages.append({"role": "assistant", "content": h["a"][:1500]})
    return messages

def answer_question(query: str, history: Optional[List[dict]] = None, paper_id: Optional[str] = None) -> dict:
    paper = library.get(paper_id)
    if paper is None:
        return {"answer": "No document loaded. Please upload a paper first.", "sources": [], "trace": [], "iterations": 0}
    hint = f" (paper_id={paper.id}; other papers in the library can be listed with list_papers)" if len(library.papers) > 1 else ""
    system = QA_SYSTEM.format(title=paper.title or paper.filename, paper_hint=hint)
    result = run_tool_agent(qa_agent_graph, system, query, history_to_messages(history))
    raw_sources = result["sources"]
    result["retrieval"] = evidence_report(result["answer"], raw_sources)
    # Every passage the agent read (before de-duplication for display) — used by the evaluation harness
    result["evidence_chunk_ids"] = sorted({int(s["chunk_id"]) for s in raw_sources
                                           if s.get("type") == "paper" and s.get("paper_id") == paper.id
                                           and s.get("chunk_id") is not None})
    result["sources"] = dedupe_sources(raw_sources)
    return result

def evidence_report(answer: str, sources: List[dict]) -> dict:
    """Runtime quality signal for an agent answer: how strong was the best paper evidence it gathered,
    and do the pages it cites exist among the passages it actually read?"""
    paper_sources = [s for s in sources if s.get("type") in ("paper", "table", "figure")]
    scores = [float(s["score"]) for s in paper_sources if s.get("score") is not None]
    top = max(scores, default=0.0)
    if not paper_sources:
        confidence = "none"
    elif top >= HIGH_CONFIDENCE:
        confidence = "high"
    elif top >= MIN_RELEVANCE or any(s.get("type") in ("table", "figure") or s.get("score") is None
                                     for s in paper_sources):
        confidence = "medium"   # includes direct reads (get_section / get_table / analyze_figure), which have no score
    else:
        confidence = "low"
    pages = {int(s["page"]) for s in paper_sources if s.get("page")}
    cited = [int(p) for p in re.findall(r"\[p\.\s*(\d+)", answer or "")]
    return {"confidence": confidence, "top_score": round(top, 3), "chunks": len(paper_sources),
            "grounding": {"cited": bool(cited), "unsupported_citations": sorted({p for p in cited if p not in pages}),
                          "unsupported_numbers": [], "grounded": all(p in pages for p in cited)}}

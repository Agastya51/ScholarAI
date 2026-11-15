"""Peer-review panel: specialist reviewers run in parallel (LangGraph Send fan-out), a meta-reviewer aggregates."""
import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.agents.base import trace_event
from src.config import REVIEWERS
from src.llm import structured
from src.rag import gather_context

REVIEWER_BRIEFS = {
    "methodology": {
        "role": "Methodology reviewer",
        "focus": "soundness of the proposed method, assumptions, experimental design, baselines, ablations, "
                 "statistical rigor, and whether conclusions follow from the results",
        "queries": ["proposed method architecture algorithm", "assumptions problem formulation",
                    "experimental setup baselines", "ablation study results analysis"],
    },
    "novelty": {
        "role": "Novelty and significance reviewer",
        "focus": "originality relative to the related work provided, significance of the contribution, "
                 "and whether claims of novelty are justified",
        "queries": ["contributions novelty", "related work prior approaches", "comparison with state of the art"],
    },
    "reproducibility": {
        "role": "Reproducibility reviewer",
        "focus": "availability of datasets, code, hyperparameters, training details, hardware, random seeds, "
                 "and enough implementation detail to re-run the experiments",
        "queries": ["dataset description preprocessing", "hyperparameters training details implementation",
                    "hardware runtime code availability", "evaluation protocol metrics"],
    },
    "clarity": {
        "role": "Clarity and presentation reviewer",
        "focus": "structure, clarity of the abstract and introduction, notation consistency, figure and table "
                 "quality, and precision of the writing",
        "queries": ["abstract introduction", "notation definitions", "conclusion summary", "figure table caption"],
    },
}

class Weakness(BaseModel):
    issue: str
    evidence: str = Field(description="Short verbatim quote from the paper supporting the issue, or '' if absent")
    page: Optional[int] = None
    severity: Literal["minor", "major", "critical"] = "minor"

class Review(BaseModel):
    score: int = Field(ge=1, le=10, description="1 = strong reject, 10 = award quality")
    confidence: int = Field(ge=1, le=5)
    summary: str
    strengths: List[str]
    weaknesses: List[Weakness]
    questions: List[str] = Field(default_factory=list, description="Questions for the authors")

class MetaReview(BaseModel):
    decision: Literal["accept", "minor_revision", "major_revision", "reject"]
    overall_score: float = Field(ge=1, le=10)
    summary: str
    consensus: List[str] = Field(description="Points most reviewers agree on")
    disagreements: List[str] = Field(default_factory=list)
    priority_fixes: List[str] = Field(description="Ordered list of the most important changes")

class ReviewState(TypedDict, total=False):
    paper_id: Optional[str]
    summary: str
    comparison: str
    reviewers: List[str]
    reviews: Annotated[list, operator.add]
    meta_review: dict
    review_markdown: str
    trace: Annotated[list, operator.add]

def fan_out_reviewers(state: dict) -> List[Send]:
    """One Send per reviewer so they run in parallel."""
    payload = {"paper_id": state.get("paper_id"), "summary": state.get("summary", ""),
               "comparison": state.get("comparison", "")}
    return [Send("reviewer", {**payload, "reviewer": r})
            for r in (state.get("reviewers") or REVIEWERS) if r in REVIEWER_BRIEFS]

def node_reviewer(state: dict) -> dict:
    name = state["reviewer"]
    brief = REVIEWER_BRIEFS[name]
    context = gather_context(brief["queries"], k=3, paper_id=state.get("paper_id"))
    related = f"\n\nRelated-work comparison:\n{state['comparison'][:2000]}" if name == "novelty" and state.get("comparison") else ""
    review = structured(f"""You are the {brief['role']} on a conference program committee.
Judge ONLY: {brief['focus']}.

Rules:
- Every weakness must cite evidence: a short verbatim quote and its page (from the [p. N · Section] anchors).
  If something is missing from the paper entirely, say so with evidence ''.
- Do not invent weaknesses; credit real strengths. Be specific, never generic.

Paper summary:
{state.get('summary') or '(not available)'}

Paper passages:
{context}{related}""", Review)
    data = {"reviewer": name, **review.model_dump()}
    n_major = sum(w.severity != "minor" for w in review.weaknesses)
    return {"reviews": [data],
            "trace": [trace_event(f"reviewer:{name}", "output",
                                  f"score {review.score}/10, {len(review.weaknesses)} weaknesses ({n_major} major+)")]}

def render_review_markdown(reviews: List[dict], meta: dict) -> str:
    label = meta.get("decision", "").replace("_", " ").title()
    out = [f"## Decision: {label} (overall {meta.get('overall_score', 0):.1f}/10)", "", meta.get("summary", ""), ""]
    if meta.get("priority_fixes"):
        out += ["### Priority fixes"] + [f"{i}. {f}" for i, f in enumerate(meta["priority_fixes"], 1)] + [""]
    if meta.get("consensus"):
        out += ["### Reviewer consensus"] + [f"- {c}" for c in meta["consensus"]] + [""]
    if meta.get("disagreements"):
        out += ["### Disagreements"] + [f"- {d}" for d in meta["disagreements"]] + [""]
    for r in reviews:
        out += [f"### {REVIEWER_BRIEFS.get(r['reviewer'], {}).get('role', r['reviewer'])} — "
                f"{r['score']}/10 (confidence {r['confidence']}/5)", r["summary"], "", "**Strengths**"]
        out += [f"- {s}" for s in r["strengths"]] + ["", "**Weaknesses**"]
        for w in r["weaknesses"]:
            cite = f" (p. {w['page']})" if w.get("page") else ""
            quote = f' — "{w["evidence"]}"' if w.get("evidence") else ""
            out.append(f"- [{w['severity']}] {w['issue']}{quote}{cite}")
        if r.get("questions"):
            out += ["", "**Questions for authors**"] + [f"- {q}" for q in r["questions"]]
        out.append("")
    return "\n".join(out)

def node_meta_review(state: dict) -> dict:
    reviews = sorted(state.get("reviews", []), key=lambda r: r["reviewer"])
    if not reviews:
        return {"meta_review": {}, "review_markdown": "No reviews were produced.",
                "trace": [trace_event("meta_reviewer", "error", "No reviews to aggregate")]}
    digest = "\n\n".join(
        f"[{r['reviewer']}] score {r['score']}/10, confidence {r['confidence']}/5\nSummary: {r['summary']}\n"
        f"Weaknesses: " + "; ".join(f"({w['severity']}) {w['issue']}" for w in r["weaknesses"])
        for r in reviews)
    meta = structured(f"""You are the area chair. Aggregate these independent reviews into a meta-review.
Weigh reviews by confidence, surface consensus and disagreements, and order the priority fixes by impact.

{digest}""", MetaReview).model_dump()
    return {"meta_review": meta, "review_markdown": render_review_markdown(reviews, meta),
            "trace": [trace_event("meta_reviewer", "decision",
                                  f"{meta['decision']} ({meta['overall_score']:.1f}/10)")]}

def build_review_graph():
    g = StateGraph(ReviewState)
    g.add_node("reviewer", node_reviewer)
    g.add_node("meta_review", node_meta_review)
    g.add_conditional_edges(START, fan_out_reviewers, ["reviewer"])
    g.add_edge("reviewer", "meta_review")
    g.add_edge("meta_review", END)
    return g.compile()

review_graph = build_review_graph()

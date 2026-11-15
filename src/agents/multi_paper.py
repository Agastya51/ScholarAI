"""Multi-paper synthesis: profile each paper in parallel (Send fan-out), then build a comparison matrix + narrative."""
import operator
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.agents.base import trace_event
from src.llm import llm, structured
from src.rag import library, gather_context

PROFILE_QUERIES = [
    "abstract problem statement contributions",
    "proposed method architecture algorithm",
    "datasets experimental setup baselines",
    "results metrics accuracy performance",
    "limitations future work conclusion",
]

class PaperProfile(BaseModel):
    title: str
    problem: str
    method: str
    datasets: List[str] = Field(default_factory=list)
    key_results: List[str] = Field(default_factory=list, description="Results with exact numbers")
    baselines: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)

class MultiPaperState(TypedDict, total=False):
    paper_ids: List[str]
    focus: str
    profiles: Annotated[list, operator.add]
    matrix_markdown: str
    synthesis: str
    trace: Annotated[list, operator.add]

def fan_out_profiles(state: MultiPaperState) -> List[Send]:
    return [Send("profile", {"paper_id": pid}) for pid in state.get("paper_ids", [])]

def node_profile(state: dict) -> dict:
    paper = library.get(state["paper_id"])
    if paper is None:
        return {"trace": [trace_event("profiler", "error", f"Paper {state['paper_id']} not in library")]}
    if paper.profile is None:
        context = gather_context(PROFILE_QUERIES, k=2, paper_id=paper.id)
        profile = structured(f"""Extract a factual profile of this research paper. Use only the passages below;
use exact names and numbers; leave a list empty rather than guessing.

Detected title: {paper.title}

Passages:
{context}""", PaperProfile)
        paper.profile = profile.model_dump()
        cached = ""
    else:
        cached = " (cached)"
    return {"profiles": [{"paper_id": paper.id, **paper.profile}],
            "trace": [trace_event("profiler", "output", f"Profiled '{paper.profile['title'][:60]}'{cached}")]}

def _cell(value) -> str:
    if isinstance(value, list):
        value = "; ".join(value) if value else "—"
    return (value or "—").replace("|", "\\|").replace("\n", " ")

def build_matrix(profiles: List[dict]) -> str:
    rows = [("Problem", "problem"), ("Method", "method"), ("Datasets", "datasets"), ("Key results", "key_results"),
            ("Baselines", "baselines"), ("Strengths", "strengths"), ("Limitations", "limitations")]
    headers = [_cell(p["title"][:45] + ("…" if len(p["title"]) > 45 else "")) for p in profiles]
    out = ["| | " + " | ".join(headers) + " |", "|---|" + "---|" * len(profiles)]
    for label, key in rows:
        out.append(f"| **{label}** | " + " | ".join(_cell(p.get(key)) for p in profiles) + " |")
    return "\n".join(out)

def node_synthesize(state: MultiPaperState) -> dict:
    order = {pid: i for i, pid in enumerate(state.get("paper_ids", []))}
    profiles = sorted(state.get("profiles", []), key=lambda p: order.get(p["paper_id"], 99))
    if len(profiles) < 2:
        msg = "Need at least two profiled papers to compare."
        return {"matrix_markdown": build_matrix(profiles) if profiles else "", "synthesis": msg,
                "trace": [trace_event("synthesizer", "error", msg)]}
    matrix = build_matrix(profiles)
    focus = state.get("focus") or "overall approach, results, and trade-offs"
    synthesis = llm(f"""You are writing the synthesis section of a literature review comparing {len(profiles)} papers.
Focus: {focus}. Ground every statement in the comparison matrix; name papers by short title; keep exact numbers.
Never compare numbers measured on different datasets or metrics as if they were comparable — say so instead.

{matrix}

Write markdown with these sections:
## Common Ground
## Key Differences
## Results Comparison
## When to Use Which Approach
## Open Research Gaps""")
    return {"matrix_markdown": matrix, "synthesis": synthesis,
            "trace": [trace_event("synthesizer", "output", f"Synthesized {len(profiles)} papers")]}

def build_multi_paper_graph():
    g = StateGraph(MultiPaperState)
    g.add_node("profile", node_profile)
    g.add_node("synthesize", node_synthesize)
    g.add_conditional_edges(START, fan_out_profiles, ["profile"])
    g.add_edge("profile", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()

multi_paper_graph = build_multi_paper_graph()

def compare_papers(paper_ids: Optional[List[str]] = None, focus: str = "") -> dict:
    ids = [pid for pid in (paper_ids or list(library.papers)) if pid in library.papers][:5]
    if len(ids) < 2:
        raise ValueError("Upload at least two papers to the library to compare them.")
    result = multi_paper_graph.invoke({"paper_ids": ids, "focus": focus, "profiles": [], "trace": []})
    return {"profiles": result.get("profiles", []), "matrix": result.get("matrix_markdown", ""),
            "synthesis": result.get("synthesis", ""), "trace": result.get("trace", [])}

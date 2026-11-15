"""Supervisor multi-agent graph: an LLM supervisor routes between specialist agents toward the user's goal,
with human-in-the-loop checkpoints (keyword approval, rewrite acceptance) via LangGraph interrupts."""
import operator
import uuid
from typing import Annotated, Iterator, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from src.agents.base import trace_event
from src.agents.review_panel import fan_out_reviewers, node_reviewer, node_meta_review
from src.config import SUPERVISOR_MAX_STEPS, MAX_LLM_CALLS_ANALYZE
from src.llm import structured, track_usage, LLMError
from src.workflows import (
    node_summarize, node_extract_topic, node_vision, node_compare, search_graph, improve_graph
)

# =========================
# STATE & WORKERS
# =========================
class AnalysisState(TypedDict, total=False):
    paper_id: Optional[str]
    goal: str
    hitl: bool
    summary: str
    vision: List[str]
    topic: str
    papers: List[dict]
    queries_tried: List[str]
    comparison: str
    improvements: str
    edits: List[dict]
    reviews: Annotated[list, operator.add]
    meta_review: dict
    review_markdown: str
    completed: List[str]
    next: str
    steps: int
    trace: Annotated[list, operator.add]

WORKERS = {
    "reader": {"desc": "Reads the paper: structured summary + search keywords.", "requires": []},
    "figure_analyst": {"desc": "Vision analysis of the paper's figures.", "requires": []},
    "literature_scout": {"desc": "Finds related work across 4 scholarly engines, self-correcting its queries.",
                         "requires": ["reader"]},
    "comparator": {"desc": "Compares the paper against the related work found.", "requires": ["literature_scout"]},
    "review_panel": {"desc": "Parallel peer review (methodology, novelty, reproducibility, clarity) + meta-review. "
                             "Novelty review is stronger after comparator.", "requires": ["reader"]},
    "editor": {"desc": "Critiques weak sections and drafts fact-checked rewrites. Better after comparator.",
               "requires": ["reader"]},
}
DEFAULT_ORDER = ["reader", "figure_analyst", "literature_scout", "comparator", "review_panel", "editor"]

class RouteDecision(BaseModel):
    next: Literal["reader", "figure_analyst", "literature_scout", "comparator", "review_panel", "editor", "FINISH"]
    reason: str

def _done(state: AnalysisState, name: str) -> List[str]:
    return list(state.get("completed", [])) + [name]

def available_workers(state: AnalysisState) -> List[str]:
    done = set(state.get("completed", []))
    return [w for w in DEFAULT_ORDER if w not in done and all(r in done for r in WORKERS[w]["requires"])]

# =========================
# SUPERVISOR
# =========================
def node_supervisor(state: AnalysisState) -> dict:
    steps = state.get("steps", 0) + 1
    available = available_workers(state)
    completed = state.get("completed", [])
    if not available or steps > SUPERVISOR_MAX_STEPS:
        why = "all useful workers finished" if not available else "step limit reached"
        return {"next": "FINISH", "steps": steps, "trace": [trace_event("supervisor", "decision", f"FINISH — {why}")]}

    status = {
        "summary": bool(state.get("summary")), "topic": state.get("topic", ""),
        "related_papers": len(state.get("papers", [])), "comparison": bool(state.get("comparison")),
        "reviews": len(state.get("reviews", [])), "edits": len(state.get("edits", [])),
    }
    options = "\n".join(f"- {w}: {WORKERS[w]['desc']}" for w in available)
    try:
        decision = structured(f"""You are the supervisor of a team of research-analysis agents.
Pick the single next worker that best advances the user's goal, or FINISH when the goal is fully met.
Do not run workers the goal does not need.

User goal: {state.get('goal') or 'Full analysis of the paper'}
Completed workers: {completed or 'none'}
Current results: {status}

Available workers:
{options}""", RouteDecision)
        choice, reason = decision.next, decision.reason
    except LLMError as e:
        choice, reason = available[0], f"router failed ({e}); falling back to default order"

    if choice == "FINISH" and not completed:
        choice, reason = "reader", "nothing has run yet; the reader must go first"
    if choice != "FINISH" and choice not in available:
        reason = f"'{choice}' not available (prerequisites or already done); using {available[0]}"
        choice = available[0]
    return {"next": choice, "steps": steps, "trace": [trace_event("supervisor", "decision", f"{choice} — {reason}")]}

def route_from_supervisor(state: AnalysisState):
    nxt = state.get("next", "FINISH")
    if nxt == "FINISH":
        return END
    if nxt == "review_panel":
        return fan_out_reviewers(state)
    return nxt

# =========================
# WORKER NODES
# =========================
def node_reader(state: AnalysisState) -> dict:
    out = node_summarize(state)
    topic = node_extract_topic({**state, **out})
    return {"summary": out["summary"], "topic": topic["topic"], "completed": _done(state, "reader"),
            "trace": out["trace"] + topic["trace"]}

def node_figure_analyst(state: AnalysisState) -> dict:
    return {**node_vision(state), "completed": _done(state, "figure_analyst")}

def node_literature_scout(state: AnalysisState) -> dict:
    topic = state.get("topic", "")
    events = []
    if state.get("hitl"):
        # Pauses the graph; resumes with {"topic": "..."} or {"skip": true}. Must stay first in the node.
        answer = interrupt({"type": "approve_keywords", "topic": topic,
                            "message": "Approve or edit the literature-search keywords."}) or {}
        if answer.get("skip"):
            return {"papers": [], "completed": _done(state, "literature_scout"),
                    "trace": [trace_event("human", "human", "Skipped literature search")]}
        if answer.get("topic") and answer["topic"] != topic:
            events.append(trace_event("human", "human", f"Edited keywords: '{topic}' -> '{answer['topic']}'"))
            topic = answer["topic"]
        else:
            events.append(trace_event("human", "human", "Approved keywords"))
    result = search_graph.invoke({"paper_id": state.get("paper_id"), "summary": state.get("summary", ""),
                                  "topic": topic, "trace": []})
    return {"topic": topic, "papers": result.get("papers", []), "queries_tried": result.get("queries_tried", []),
            "completed": _done(state, "literature_scout"), "trace": events + result.get("trace", [])}

def node_comparator(state: AnalysisState) -> dict:
    return {**node_compare(state), "completed": _done(state, "comparator")}

def node_panel_meta(state: AnalysisState) -> dict:
    return {**node_meta_review(state), "completed": _done(state, "review_panel")}

def node_editor(state: AnalysisState) -> dict:
    result = improve_graph.invoke({"paper_id": state.get("paper_id"), "summary": state.get("summary", ""),
                                   "comparison": state.get("comparison", ""), "trace": []})
    return {"improvements": result.get("improvements", ""), "edits": result.get("edits", []),
            "completed": _done(state, "editor"), "trace": result.get("trace", [])}

def route_after_editor(state: AnalysisState) -> str:
    return "edit_approval" if state.get("hitl") and state.get("edits") else "supervisor"

def node_edit_approval(state: AnalysisState) -> dict:
    """Human checkpoint: resume with {"accepted": [edit indices]}."""
    edits = state.get("edits", [])
    answer = interrupt({"type": "review_edits", "edits": edits,
                        "message": "Select which rewrites to keep."}) or {}
    accepted = answer.get("accepted")
    if accepted is None:
        kept = edits
    else:
        kept = [e for i, e in enumerate(edits) if i in set(accepted)]
    return {"edits": kept, "trace": [trace_event("human", "human", f"Kept {len(kept)}/{len(edits)} rewrites")]}

# =========================
# GRAPH
# =========================
def build_analysis_graph(checkpointer=None):
    g = StateGraph(AnalysisState)
    g.add_node("supervisor", node_supervisor)
    g.add_node("reader", node_reader)
    g.add_node("figure_analyst", node_figure_analyst)
    g.add_node("literature_scout", node_literature_scout)
    g.add_node("comparator", node_comparator)
    g.add_node("reviewer", node_reviewer)
    g.add_node("meta_review", node_panel_meta)
    g.add_node("editor", node_editor)
    g.add_node("edit_approval", node_edit_approval)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            ["reader", "figure_analyst", "literature_scout", "comparator", "reviewer", "editor", END])
    for worker in ["reader", "figure_analyst", "literature_scout", "comparator", "meta_review", "edit_approval"]:
        g.add_edge(worker, "supervisor")
    g.add_edge("reviewer", "meta_review")
    g.add_conditional_edges("editor", route_after_editor, ["edit_approval", "supervisor"])
    return g.compile(checkpointer=checkpointer or MemorySaver())

analysis_graph = build_analysis_graph()

RESULT_KEYS = ["summary", "vision", "topic", "papers", "queries_tried", "comparison", "improvements", "edits",
               "reviews", "meta_review", "review_markdown", "completed"]

def public_result(values: dict) -> dict:
    return {k: values.get(k) for k in RESULT_KEYS if values.get(k) is not None}

def stream_analysis(thread_id: Optional[str] = None, paper_id: Optional[str] = None, goal: str = "",
                    hitl: bool = False, resume: Optional[dict] = None, graph=None) -> Iterator[dict]:
    """Run (or resume) an analysis, yielding events: start | step | interrupt | done | error."""
    graph = graph or analysis_graph
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
    inp = Command(resume=resume) if resume is not None else {
        "paper_id": paper_id, "goal": goal, "hitl": hitl, "completed": [], "steps": 0, "reviews": [], "trace": []}
    yield {"type": "start", "thread_id": thread_id}
    try:
        with track_usage(MAX_LLM_CALLS_ANALYZE) as usage:
            for chunk in graph.stream(inp, config, stream_mode="updates"):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        yield {"type": "interrupt", "thread_id": thread_id, "payload": update[0].value,
                               "llm_calls": usage["calls"]}
                    else:
                        yield {"type": "step", "node": node, "trace": (update or {}).get("trace", []),
                               "llm_calls": usage["calls"]}
            snapshot = graph.get_state(config)
            if not snapshot.next:
                result = public_result(snapshot.values)
                try:  # a finished run is never resumed: free its checkpoints (bounded memory on small instances)
                    graph.checkpointer.delete_thread(thread_id)
                except Exception:
                    pass
                yield {"type": "done", "thread_id": thread_id, "result": result, "llm_calls": usage["calls"]}
    except Exception as e:
        yield {"type": "error", "thread_id": thread_id, "error": str(e)}

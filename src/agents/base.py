"""Shared agent building blocks: trace events and a reusable tool-calling (ReAct) LangGraph agent."""
import json
import operator
import time
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from src.llm import chat
from src.tools import get_specs, run_tool

TRACE_PREVIEW_CHARS = 600

def trace_event(agent: str, kind: str, detail: str, **data) -> dict:
    """One step in an agent run. kind: thought | tool_call | tool_result | decision | output | error | human."""
    return {"agent": agent, "kind": kind, "detail": detail[:TRACE_PREVIEW_CHARS], "t": round(time.time(), 3), **data}

# =========================
# TOOL-CALLING AGENT GRAPH
# =========================
class ToolAgentState(TypedDict, total=False):
    messages: List[dict]
    iterations: int
    answer: str
    sources: Annotated[list, operator.add]
    trace: Annotated[list, operator.add]

def build_tool_agent(name: str, tool_names: List[str], max_iterations: int):
    """Build a LangGraph ReAct loop: agent -> (tools -> agent)* -> END.

    After `max_iterations` tool rounds the agent is called without tools and told to answer.
    """
    specs = get_specs(tool_names)

    def node_agent(state: ToolAgentState) -> dict:
        iterations = state.get("iterations", 0)
        messages = list(state["messages"])
        budget_left = iterations < max_iterations
        if not budget_left:
            messages.append({"role": "user", "content":
                             "Tool budget exhausted. Answer now using only the evidence gathered so far."})
        res = chat(messages, tools=specs if budget_left else None)

        if res["tool_calls"] and budget_left:
            messages.append({"role": "assistant", "content": res["content"] or "", "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in res["tool_calls"]]})
            events = []
            if res["content"]:
                events.append(trace_event(name, "thought", res["content"]))
            events += [trace_event(name, "tool_call", f"{tc['name']}({json.dumps(tc['arguments'])})",
                                   tool=tc["name"], args=tc["arguments"]) for tc in res["tool_calls"]]
            return {"messages": messages, "iterations": iterations + 1, "trace": events}

        messages.append({"role": "assistant", "content": res["content"]})
        return {"messages": messages, "answer": res["content"],
                "trace": [trace_event(name, "output", res["content"], iterations=iterations)]}

    def node_tools(state: ToolAgentState) -> dict:
        messages = list(state["messages"])
        calls = messages[-1].get("tool_calls", [])
        events, sources = [], []
        for tc in calls:
            fn = tc["function"]
            result = run_tool(fn["name"], json.loads(fn["arguments"] or "{}"))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result.content})
            sources += result.sources
            events.append(trace_event(name, "error" if result.error else "tool_result", result.content,
                                      tool=fn["name"], n_sources=len(result.sources)))
        return {"messages": messages, "sources": sources, "trace": events}

    def route(state: ToolAgentState) -> str:
        last = state["messages"][-1]
        return "tools" if last.get("role") == "assistant" and last.get("tool_calls") else END

    g = StateGraph(ToolAgentState)
    g.add_node("agent", node_agent)
    g.add_node("tools", node_tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()

def run_tool_agent(graph, system: str, user: str, history: Optional[List[dict]] = None) -> dict:
    """Invoke a tool agent graph. history: prior [{"role", "content"}] turns."""
    messages = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user}]
    # Each tool round is two graph steps; leave generous headroom over the iteration cap.
    result = graph.invoke({"messages": messages, "iterations": 0, "sources": [], "trace": []},
                          {"recursion_limit": 50})
    return {"answer": result.get("answer", ""), "sources": result.get("sources", []),
            "trace": result.get("trace", []), "iterations": result.get("iterations", 0)}

def dedupe_sources(sources: List[dict]) -> List[dict]:
    seen, out = set(), []
    for s in sources:
        key = (s.get("type"), s.get("paper_id"), s.get("page"), s.get("section"), s.get("title"),
               s.get("figure_id"), s.get("table_id"))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

"""The answering pipelines compared by evals/run_eval.py.

    naive   plain hybrid top-k + one LLM call           (how ScholarAI's "simple RAG" worked before the upgrade)
    simple  retrieval pipeline (MMR, adaptive k, threshold, corrective query expansion) + one grounded LLM call
    agent   tool-calling research agent (iterative retrieval, tables, figures, references, external literature)
"""
import time
from typing import List

from src.agents.qa_agent import answer_question
from src.llm import llm, track_usage
from src.rag import Paper, format_chunks
from src.workflows import qa_simple_graph

MODES = ["naive", "simple", "agent"]
NAIVE_K = 4

NAIVE_PROMPT = """You are a research assistant. Answer based on the paper context using markdown.
Cite page anchors like [p. 3] for every claim.

Context:
{context}

Question: {query}

Answer:"""

def _contexts_from_sources(paper: Paper, sources: List[dict], chunk_ids: List[int] = ()) -> List[dict]:
    """Recover the full passages behind an answer's sources (sources only carry a short snippet)."""
    out, seen = [], set()
    for i in chunk_ids:
        if 0 <= i < len(paper.chunks) and i not in seen:
            seen.add(i)
            out.append(paper.chunks[i])
    for s in sources:
        if s.get("type") == "paper" and s.get("chunk_id") is not None:
            i = int(s["chunk_id"])
            if 0 <= i < len(paper.chunks) and i not in seen:
                seen.add(i)
                out.append(paper.chunks[i])
        elif s.get("type") == "paper" and s.get("snippet"):
            match = next((j for j, c in enumerate(paper.chunks) if str(c["text"]).startswith(s["snippet"])), None)
            if match is not None and match not in seen:
                seen.add(match)
                out.append(paper.chunks[match])
        elif s.get("type") == "table":
            table = next((t for t in paper.tables if t["id"] == s.get("table_id")), None)
            if table:
                out.append({"text": table["markdown"], "page": table["page"], "kind": "table"})
        elif s.get("type") == "figure":
            analysis = paper.figure_analyses.get(s.get("figure_id", ""), "")
            if analysis:
                out.append({"text": analysis, "page": s.get("page"), "kind": "figure"})
    return out

def run_pipeline(mode: str, paper: Paper, question: str) -> dict:
    """Answer one question. Returns {answer, contexts, llm_calls, seconds, confidence}."""
    start = time.time()
    with track_usage() as usage:
        if mode == "naive":
            contexts = paper.store.search(question, NAIVE_K, mode="hybrid", mmr=False)
            answer = llm(NAIVE_PROMPT.format(context=format_chunks(contexts), query=question)) if contexts else ""
            confidence = ""
        elif mode == "simple":
            result = qa_simple_graph.invoke({"paper_id": paper.id, "query": question, "trace": []})
            answer = result["answer"]
            contexts = _contexts_from_sources(paper, result.get("sources", []))
            confidence = (result.get("retrieval") or {}).get("confidence", "")
        elif mode == "agent":
            result = answer_question(question, paper_id=paper.id)
            answer = result["answer"]
            contexts = _contexts_from_sources(paper, result.get("sources", []), result.get("evidence_chunk_ids", []))
            confidence = (result.get("retrieval") or {}).get("confidence", "")
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}.")
    return {"answer": answer, "contexts": contexts, "llm_calls": usage["calls"],
            "seconds": round(time.time() - start, 1), "confidence": confidence}

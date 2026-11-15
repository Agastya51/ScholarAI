"""Tool registry: existing capabilities wrapped as LLM-callable tools (Groq / OpenAI function format)."""
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.figures import explain_figure, explain_paper_figure, figure_context
from src.llm import LLMBudgetExceeded
from src.rag import library, format_chunks, Paper
from src.retrieval import retrieve_context
from src.search import multi_engine_search, lookup_paper as s2_lookup

ToolOutput = Union[str, Tuple[str, List[dict]]]

@dataclass
class ToolResult:
    content: str
    sources: List[dict] = field(default_factory=list)
    error: bool = False

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., ToolOutput]

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

REGISTRY: Dict[str, Tool] = {}

_PAPER_ID = {"type": "string", "description": "Optional paper id (from list_papers). Defaults to the active paper."}

def tool(name: str, description: str, properties: Optional[dict] = None, required: Optional[List[str]] = None):
    """Decorator that registers a function as an agent tool."""
    def wrap(fn):
        REGISTRY[name] = Tool(name, description, {
            "type": "object", "properties": properties or {}, "required": required or []}, fn)
        return fn
    return wrap

def get_specs(names: List[str]) -> List[dict]:
    return [REGISTRY[n].spec() for n in names]

def run_tool(name: str, args: Dict[str, Any]) -> ToolResult:
    """Execute a tool. Errors are returned to the model as text so it can recover."""
    t = REGISTRY.get(name)
    if t is None:
        return ToolResult(f"Unknown tool '{name}'. Available: {', '.join(REGISTRY)}", error=True)
    try:
        out = t.fn(**(args or {}))
    except LLMBudgetExceeded:
        raise
    except TypeError as e:
        return ToolResult(f"Invalid arguments for {name}: {e}", error=True)
    except Exception as e:
        return ToolResult(f"Tool {name} failed: {e}", error=True)
    if isinstance(out, tuple):
        return ToolResult(out[0], out[1])
    return ToolResult(out)

def _paper(paper_id: Optional[str] = None) -> Paper:
    paper = library.get(paper_id)
    if paper is None:
        raise ValueError("No paper is loaded. Ask the user to upload a paper first.")
    return paper

def _chunk_sources(paper: Paper, chunks: List[dict]) -> List[dict]:
    return [{"type": "paper", "paper_id": paper.id, "page": c.get("page"), "section": c.get("section"),
             "kind": c.get("kind"), "score": c.get("dense_score"), "chunk_id": c.get("chunk_id"),
             "snippet": str(c["text"])[:220]} for c in chunks]

# =========================
# PAPER TOOLS
# =========================
@tool("search_paper",
      "Hybrid (keyword + semantic) search over the uploaded paper. Returns passages with [page · section] anchors "
      "to cite. Use specific terms: dataset names, metric names, method names.",
      {"query": {"type": "string", "description": "What to look for in the paper"},
       "k": {"type": "integer", "description": "Number of passages (1-8)", "default": 5},
       "paper_id": _PAPER_ID},
      ["query"])
def search_paper(query: str, k: int = 5, paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    # No LLM query expansion here: the agent reformulates its own queries when told the evidence is weak.
    result = retrieve_context(paper, query, max(1, min(int(k), 8)), expansion="off")
    if not result.chunks:
        return "No matching passages found. Try different terms or get_section."
    text = format_chunks(result.chunks)
    if result.confidence == "low":
        text += ("\n\n(Retrieval confidence is LOW: these passages may not answer the question. "
                 "Try more specific or different terms, or read a section with get_section.)")
    return text, _chunk_sources(paper, result.chunks)

@tool("list_sections", "List the section headings detected in the paper, in reading order.",
      {"paper_id": _PAPER_ID})
def list_sections(paper_id: Optional[str] = None) -> ToolOutput:
    return "\n".join(f"- {s}" for s in _paper(paper_id).sections()) or "No sections detected."

@tool("get_section", "Read the passages of one section by (partial) heading name, e.g. 'Results' or 'Conclusion'.",
      {"name": {"type": "string"}, "paper_id": _PAPER_ID}, ["name"])
def get_section(name: str, paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    chunks = paper.section_chunks(name)[:6]
    if not chunks:
        return f"No section matching '{name}'. Sections: {', '.join(paper.sections()[:25])}"
    return format_chunks(chunks), _chunk_sources(paper, chunks)

@tool("list_figures", "List the paper's figures: id, page, caption, and the pages where the text discusses each one.",
      {"paper_id": _PAPER_ID})
def list_figures(paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    if not paper.figures:
        return "No figures were found in this paper."
    lines = []
    for f in paper.figures:
        pages = sorted({m["page"] for m in f.get("mentions", [])})
        lines.append(f"- {f['id']} (page {f['page']}){' [explained]' if f['id'] in paper.figure_analyses else ''}"
                     + (f": {f['caption'][:200]}" if f.get("caption") else ": no caption found")
                     + (f" | discussed on p. {', '.join(map(str, pages))}" if pages else ""))
    return "\n".join(lines)

@tool("analyze_figure",
      "Look at one figure with the vision model. It sees the image together with the caption and every sentence "
      "of the paper that refers to the figure. Pass `question` to ask something specific about the figure; "
      "without it you get a general explanation.",
      {"figure_id": {"type": "string", "description": "Figure id from list_figures, e.g. 'fig2'"},
       "question": {"type": "string", "description": "Optional specific question about the figure"},
       "paper_id": _PAPER_ID}, ["figure_id"])
def analyze_figure(figure_id: str, question: str = "", paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    fig = paper.figure(figure_id)
    if fig is None:
        return f"No figure '{figure_id}'. Available: {', '.join(f['id'] for f in paper.figures) or 'none'}"
    if question:
        text = explain_figure(fig, question=question)["text"]
    else:
        text = explain_paper_figure(paper, figure_id)["text"]
    context = figure_context(fig)
    return (f"{fig.get('label') or figure_id} (page {fig['page']})\n{context}\n\n{text}",
            [{"type": "figure", "paper_id": paper.id, "page": fig["page"], "figure_id": figure_id}]
            + [{"type": "paper", "paper_id": paper.id, "page": m["page"], "section": m["section"], "kind": "text",
                "snippet": m["text"][:220]} for m in fig.get("mentions", [])])

@tool("list_tables", "List data tables extracted from the paper (id, page, first row).", {"paper_id": _PAPER_ID})
def list_tables(paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    if not paper.tables:
        return "No tables were detected in this paper."
    return "\n".join(f"- {t['id']} (page {t['page']}): {(t.get('caption') or t['markdown'].splitlines()[0])[:140]}"
                     for t in paper.tables)

@tool("get_table", "Return one extracted table as markdown, by id, e.g. 'table1'.",
      {"table_id": {"type": "string"}, "paper_id": _PAPER_ID}, ["table_id"])
def get_table(table_id: str, paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    t = next((t for t in paper.tables if t["id"] == table_id), None)
    if t is None:
        return f"No table '{table_id}'."
    caption = f"{t['caption']}\n" if t.get("caption") else ""
    said = "".join(f"\n- [p. {m['page']}] {m['text']}" for m in t.get("mentions", []))
    said = f"\n\nWhat the text says about this table:{said}" if said else ""
    return (f"{table_id} (page {t['page']}):\n{caption}{t['markdown']}{said}",
            [{"type": "table", "paper_id": paper.id, "page": t["page"], "table_id": table_id}])

@tool("get_reference",
      "Look up entries of the paper's reference list. Pass a number (e.g. 12 for citation [12]) or a keyword "
      "(author, title word). Returns the bibliographic entries, which can then be passed to lookup_paper.",
      {"query": {"type": "string", "description": "Reference number or a keyword such as an author name"},
       "paper_id": _PAPER_ID}, ["query"])
def get_reference(query: str, paper_id: Optional[str] = None) -> ToolOutput:
    paper = _paper(paper_id)
    if not paper.references:
        return "The reference list could not be parsed for this paper. Try search_paper with 'references <keyword>'."
    q = str(query).strip().strip("[]")
    if q.isdigit():
        hits = [r for r in paper.references if r["n"] == int(q)]
    else:
        hits = [r for r in paper.references if q.lower() in r["text"].lower()][:8]
    if not hits:
        return f"No reference matching '{query}'. The paper has {len(paper.references)} parsed references."
    return "\n".join(f"[{r['n']}] {r['text']}" for r in hits)

@tool("list_papers", "List all papers in the library (for questions spanning multiple uploaded papers).")
def list_papers() -> ToolOutput:
    return json.dumps(library.list(), indent=1) if library.papers else "The library is empty."

# =========================
# EXTERNAL LITERATURE TOOLS
# =========================
@tool("search_literature",
      "Search ArXiv, Semantic Scholar, OpenAlex and CrossRef for related published work. Use when a question "
      "concerns external baselines, prior work, or the state of the art beyond this paper.",
      {"query": {"type": "string", "description": "Comma-separated technical keywords"}}, ["query"])
def search_literature(query: str) -> ToolOutput:
    paper = library.active()
    papers = multi_engine_search([k.strip() for k in query.split(",")], summary=paper.summary if paper else "", limit=5)
    if not papers:
        return "No related literature found. Try broader or alternative keywords."
    lines = [f"- [{p.get('year') or 'n.d.'}] {p['title']} ({p.get('venue', '')}) {p.get('link', '')}\n  "
             f"{(p.get('summary') or '')[:400]}" for p in papers]
    return "\n".join(lines), [{"type": "literature", "title": p["title"], "link": p.get("link")} for p in papers]

@tool("lookup_paper", "Fetch metadata (authors, year, venue, citations, abstract) for one external paper by DOI or title.",
      {"title_or_doi": {"type": "string"}}, ["title_or_doi"])
def lookup_paper(title_or_doi: str) -> ToolOutput:
    info = s2_lookup(title_or_doi)
    if not info:
        return f"No record found for '{title_or_doi}'."
    return json.dumps(info, indent=1), [{"type": "literature", "title": info["title"], "link": info.get("link")}]

PAPER_TOOLS = ["search_paper", "list_sections", "get_section", "list_figures", "analyze_figure",
               "list_tables", "get_table", "get_reference"]
LITERATURE_TOOLS = ["search_literature", "lookup_paper"]

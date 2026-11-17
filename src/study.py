"""Study material generated from one paper, with the parts that can be checked, checked.

  insights   claims & evidence (every quote is searched for in the paper and marked verified or not)
             + glossary of the paper's key terms
  practice   flashcards + multiple-choice quiz, each tied to a page
  explain    the paper explained at three levels (school student / undergraduate / expert)
  questions  starter questions specific to this paper

Everything is cached on the Paper object, so re-opening a tab costs no LLM call.
"""
import re
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from src.llm import llm, structured
from src.rag import Paper, gather_context
from src.retrieval import unsupported_numbers

STUDY_QUERIES = [
    "abstract problem statement motivation contribution",
    "proposed method approach architecture how it works",
    "experimental setup datasets baselines hyperparameters",
    "main results numbers comparison ablation",
    "limitations future work conclusion",
]

STUDY_CONTEXT_CHARS = 11000   # ~3,100 tokens: with the answer and reasoning room it stays under Groq's 8K/min

def study_context(paper: Paper) -> str:
    text = gather_context(STUDY_QUERIES, k=3, paper_id=paper.id)
    if len(text) <= STUDY_CONTEXT_CHARS:
        return text
    cut = text.rfind("\n\n", 0, STUDY_CONTEXT_CHARS)        # end on a whole passage
    return text[:cut if cut > STUDY_CONTEXT_CHARS // 2 else STUDY_CONTEXT_CHARS]

# =========================
# QUOTE VERIFICATION (no LLM)
# =========================
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")

def _tokens(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())

def _norm(text: str) -> str:
    return " ".join(_tokens(text))

def locate_quote(paper: Paper, quote: str) -> Tuple[str, Optional[int], Optional[int]]:
    """Find a quote in the paper. Returns (status, page, chunk_id):

    "exact"      the quote occurs word for word in a passage
    "close"      at least 85% of its words occur together in one passage (light paraphrase or truncation)
    "not_found"  no passage supports it; treat the claim as unverified
    """
    needle = _norm(quote)
    words = needle.split()
    if len(words) < 4:
        return "not_found", None, None
    best, best_id = 0.0, None
    for i, chunk in enumerate(paper.chunks):
        if chunk.get("kind") == "figure":     # model-written figure analyses are not the paper's own words
            continue
        hay = _norm(str(chunk["text"]))
        if needle in hay:
            return "exact", chunk.get("page"), i
        have = set(hay.split())
        share = sum(w in have for w in words) / len(words)
        if share > best:
            best, best_id = share, i
    if best >= 0.85 and best_id is not None:
        return "close", paper.chunks[best_id].get("page"), best_id
    return "not_found", None, None

def _valid_page(paper: Paper, page) -> Optional[int]:
    try:
        page = int(page)
    except (TypeError, ValueError):
        return None
    return page if 1 <= page <= max(paper.pages, 1) else None

# =========================
# INSIGHTS: claims & evidence + glossary
# =========================
class Claim(BaseModel):
    claim: str = Field(description="One specific claim the paper makes, in plain words")
    evidence: str = Field(description="A sentence copied VERBATIM from the context that supports the claim (max 40 words)")
    page: int = Field(description="Page number of the evidence, from its [p. N] anchor")
    strength: Literal["strong", "moderate", "weak"] = Field(
        description="strong = measured result with numbers/baselines; moderate = some evidence; weak = asserted or anecdotal")
    why: str = Field(description="One sentence justifying the strength rating")

class Term(BaseModel):
    term: str
    definition: str = Field(description="What the term means in THIS paper, one or two plain sentences")
    page: int

class Insights(BaseModel):
    claims: List[Claim]
    terms: List[Term]

def build_insights(paper: Paper) -> dict:
    if "insights" in paper.study:
        return paper.study["insights"]
    out = structured(f"""You are helping a student read the research paper "{paper.title}" critically.

From the context below produce:
1. claims: the 5 to 7 most important claims of the paper (main contribution, key results, comparisons, limitations
   the authors admit). For each: the claim in plain words; `evidence` = ONE sentence copied word for word from the
   context that supports it; the page from that passage's [p. N] anchor; a strength rating and why.
   Rate honestly: a claim backed only by the authors' assertion is "weak".
2. terms: 6 to 8 technical terms a newcomer must understand to follow this paper, each defined as the paper uses it.

Rules: use ONLY the context. Copy evidence exactly (it will be checked against the paper automatically).
Never invent numbers.

Context:
{study_context(paper)}""", Insights, max_tokens=2400)

    claims = []
    for c in out.claims[:8]:
        status, page, chunk_id = locate_quote(paper, c.evidence)
        claims.append({"claim": c.claim.strip(), "evidence": c.evidence.strip(), "strength": c.strength,
                       "why": c.why.strip(), "verified": status,
                       # A quote that is not in the paper has no page: never show the model's guess as a reference
                       "page": page if status != "not_found" else None, "chunk_id": chunk_id,
                       "unsupported_numbers": unsupported_numbers(c.claim, paper.text)})
    terms = [{"term": t.term.strip(), "definition": t.definition.strip(), "page": _valid_page(paper, t.page)}
             for t in out.terms[:10] if t.term.strip() and t.definition.strip()]
    result = {"claims": claims, "terms": terms,
              "verified": sum(c["verified"] == "exact" for c in claims), "total": len(claims)}
    paper.study["insights"] = result
    return result

# =========================
# PRACTICE: flashcards + quiz
# =========================
class Flashcard(BaseModel):
    front: str = Field(description="A question or prompt")
    back: str = Field(description="The answer, one or two sentences, with exact values where relevant")
    page: int

class QuizItem(BaseModel):
    question: str
    options: List[str] = Field(description="Exactly 4 answer options")
    answer: int = Field(ge=0, le=3, description="Index (0-3) of the correct option")
    explanation: str = Field(description="Why that option is correct, citing what the paper says")
    page: int

class Practice(BaseModel):
    flashcards: List[Flashcard]
    quiz: List[QuizItem]

def build_practice(paper: Paper) -> dict:
    if "practice" in paper.study:
        return paper.study["practice"]
    out = structured(f"""You are writing revision material for the research paper "{paper.title}".

From the context below produce:
1. flashcards: 8 cards covering the problem, the method's key ideas, the setup, the main numbers and the limitations.
2. quiz: 6 multiple-choice questions with exactly 4 options each. Wrong options must be plausible
   (for a number question, use nearby numbers). Vary which position holds the correct answer.

Rules: use ONLY the context; every answer must be checkable on the page you cite ([p. N] anchors).
Do not ask about things the context does not state.

Context:
{study_context(paper)}""", Practice, max_tokens=2400)

    cards = [{"front": c.front.strip(), "back": c.back.strip(), "page": _valid_page(paper, c.page),
              "unsupported_numbers": unsupported_numbers(c.back, paper.text)}
             for c in out.flashcards[:12] if c.front.strip() and c.back.strip()]
    quiz = []
    for q in out.quiz[:10]:
        options = [o.strip() for o in q.options if o and o.strip()]
        if len(options) != 4 or len(set(options)) != 4 or not q.question.strip():
            continue      # a malformed item is dropped rather than shown
        quiz.append({"question": q.question.strip(), "options": options, "answer": q.answer,
                     "explanation": q.explanation.strip(), "page": _valid_page(paper, q.page)})
    result = {"flashcards": cards, "quiz": quiz}
    paper.study["practice"] = result
    return result

# =========================
# EXPLAIN AT A LEVEL
# =========================
LEVELS = {
    "simple": "a curious 15-year-old. Use everyday words and one analogy. No formulas, no undefined jargon. "
              "About 150 words.",
    "student": "an undergraduate engineering student who knows basic maths and programming but not this field. "
               "Define each technical term the first time it appears. About 220 words.",
    "expert": "a researcher in the same field. Be dense and precise: what is new compared with prior work, the "
              "exact setup and numbers, and the weakest points of the evidence. About 220 words.",
}

def explain_at_level(paper: Paper, level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"Unknown level '{level}'. Choose from: {', '.join(LEVELS)}")
    key = f"explain:{level}"
    if key not in paper.study:
        paper.study[key] = llm(f"""Explain the research paper "{paper.title}" to {LEVELS[level]}

Structure: what problem it tackles and why anyone cares; the core idea; what they found; what is still unsolved.
Use ONLY the material below. Keep every number exactly as written. Write flowing markdown paragraphs, no headings.

Summary of the paper:
{paper.summary or '(no summary yet)'}

Passages from the paper:
{study_context(paper)[:6000]}""", max_tokens=700)
    return str(paper.study[key])

# =========================
# STARTER QUESTIONS
# =========================
class Questions(BaseModel):
    questions: List[str] = Field(description="5 short questions a reader would want answered about this paper")

def suggest_questions(paper: Paper) -> List[str]:
    if "questions" not in paper.study:
        figure_hint = ""
        if paper.figures:
            labels = ", ".join(f.get("label") or f["id"] for f in paper.figures[:4])
            figure_hint = f"\nThe paper has these figures: {labels}. Make one question about a specific figure."
        out = structured(f"""Write 5 short, specific questions a reader would want to ask about this paper.
Mix them: one about the core idea, one about an exact result, one about the experimental setup, one about a
limitation or weakness, and one that compares the method with its baselines.{figure_hint}
Name the paper's own method, datasets and baselines in the questions. Each question under 14 words.

Paper: {paper.title}
Summary:
{paper.summary or study_context(paper)[:3000]}""", Questions, max_tokens=300)
        paper.study["questions"] = [q.strip() for q in out.questions if q and q.strip().endswith("?")][:5]
    return list(paper.study["questions"])

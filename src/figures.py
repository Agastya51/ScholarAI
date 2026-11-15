"""Figure understanding.

A figure in a paper is three things that are often far apart in the PDF:
  1. the picture itself (a bitmap, or vector drawings that are not an "image" at all),
  2. its caption (sometimes on the next page),
  3. the sentences elsewhere in the text that explain it ("As Fig. 3 shows, ...").

This module finds all three and joins them, so the vision model is never asked to explain a picture cold:
  - locate_figures(): for every "Fig. N" caption, find the graphics that belong to it and render that page
    region to an image (works for vector plots too, and keeps axis labels that are text, not pixels)
  - collect_mentions(): every sentence in the paper that refers to "Fig. N" / "Table N", with page + section
  - figure_context(): caption + mentions as text for the model
  - explain_figure() / ask_figure(): vision call with that context; falls back to a text-only explanation
    built from the caption and mentions when the image cannot be analysed
"""
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

Box = Tuple[float, float, float, float]

FIGURE_DPI = 150
MAX_FIGURE_PIXELS = 1600        # longest side of a rendered figure
MAX_FIGURES = 12                # per paper
ATTACH_GAP = 60.0               # max vertical distance (pt) between a caption / figure part and the next part
MAX_MENTIONS = 6
MENTION_CHARS = 420

# =========================
# GEOMETRY HELPERS
# =========================
def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def _x_overlap(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))

def _y_gap(a: Box, b: Box) -> float:
    """Vertical distance between two boxes (0 when their vertical extents overlap)."""
    return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))

def _overlap_share(inner: Box, outer: Box) -> float:
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (w * h) / max(1e-6, _area(inner))

# =========================
# GRAPHICS ON A PAGE
# =========================
MAX_PATHS_FOR_EXACT_CLUSTERING = 2500
_PATH_KINDS = ("fill-path", "stroke-path")

def path_rects(page) -> List["fitz.Rect"]:
    """Bounding boxes of the page's vector paths.

    Read from the page's bbox log rather than get_drawings(): a plot exported from MATLAB or matplotlib can
    hold tens of thousands of paths, and get_drawings() builds a dictionary of points for every one of them
    (measured on such a page: 33 MB and 1 s, against 2 MB and 0.3 s for the bbox log)."""
    try:
        return [fitz.Rect(b) for kind, b in page.get_bboxlog() if kind in _PATH_KINDS]
    except Exception:
        return [d["rect"] for d in page.get_drawings()]

def _coarse_clusters(drawings: Sequence[dict], tol: float = 5.0) -> List[Box]:
    """Merge the bounding boxes of vector paths that touch or nearly touch. Linear in the number of paths
    times the number of resulting clusters, so a scatter plot with 50,000 points stays fast."""
    boxes: List[Box] = []
    for d in drawings:
        r = d.get("rect") if isinstance(d, dict) else d
        if r is None:
            continue
        b = (r.x0, r.y0, r.x1, r.y1)
        for i, o in enumerate(boxes):
            if b[0] <= o[2] + tol and b[2] >= o[0] - tol and b[1] <= o[3] + tol and b[3] >= o[1] - tol:
                boxes[i] = _union(o, b)
                break
        else:
            boxes.append(b)
    merged = True
    while merged:                      # boxes that grew into each other
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, o = boxes[i], boxes[j]
                if a[0] <= o[2] + tol and a[2] >= o[0] - tol and a[1] <= o[3] + tol and a[3] >= o[1] - tol:
                    boxes[i] = _union(a, o)
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes

def page_graphics(page, table_boxes: Sequence[Box] = ()) -> List[Box]:
    """Boxes of everything on the page that is drawn rather than typed: bitmaps and clusters of vector paths.

    Thin rules, table borders, tiny glyph-like marks and page-sized backgrounds are ignored.
    """
    rect = page.rect
    page_area = max(1.0, rect.width * rect.height)
    boxes: List[Box] = []
    try:
        for info in page.get_image_info():
            b = tuple(fitz.Rect(info["bbox"]) & rect)
            if b[2] - b[0] >= 40 and b[3] - b[1] >= 40 and _area(b) < 0.9 * page_area:
                boxes.append(b)
    except Exception:
        pass
    try:
        rects = path_rects(page)
        # PyMuPDF's clustering is slow on plots made of thousands of paths; use the coarse merge there
        clusters = _coarse_clusters(rects) if len(rects) > MAX_PATHS_FOR_EXACT_CLUSTERING \
            else page.cluster_drawings(drawings=[{"rect": r} for r in rects])
        for r in clusters:
            b = tuple(fitz.Rect(r) & rect)
            if b[2] - b[0] < 40 or b[3] - b[1] < 30 or _area(b) >= 0.9 * page_area:
                continue
            if any(_overlap_share(b, tb) >= 0.6 for tb in table_boxes):
                continue    # ruled table borders
            boxes.append(b)
    except Exception:
        pass
    # Drop boxes nested inside another (a bitmap with a vector frame around it)
    keep = [b for b in boxes if not any(o is not b and _overlap_share(b, o) >= 0.95 and _area(o) > _area(b)
                                        for o in boxes)]
    return sorted(set(keep), key=lambda b: (b[1], b[0]))

def _band(caption: Box, rect, two_column: bool = True) -> Box:
    """Horizontal extent a caption's figure can occupy: the whole page on a single-column page or for a
    caption that crosses the page centre, otherwise the caption's half of the page (its column)."""
    width = rect.width
    mid = rect.x0 + width / 2
    if not two_column or (caption[0] < mid - 0.05 * width and caption[2] > mid + 0.05 * width):
        return (rect.x0, 0.0, rect.x1, 0.0)
    if (caption[0] + caption[2]) / 2 < mid:
        return (rect.x0, 0.0, mid + 0.02 * width, 0.0)
    return (mid - 0.02 * width, 0.0, rect.x1, 0.0)

def attach_graphics(caption: Box, graphics: Sequence[Box], claimed: set, blockers: Sequence[Box], rect,
                    below: bool = False, two_column: bool = True) -> List[int]:
    """Indices of the graphics that form the figure sitting directly above (or below) a caption.

    Starts at the caption and keeps adding graphics that are within ATTACH_GAP of what has been collected,
    stay inside the caption's column band, and are not separated from it by another caption or a paragraph.
    """
    band = _band(caption, rect, two_column)
    region: Box = (caption[0], caption[3], caption[2], caption[3]) if below else \
                  (caption[0], caption[1], caption[2], caption[1])
    picked: List[int] = []
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(graphics):
            if i in claimed or i in picked:
                continue
            if _x_overlap(g, band) < 0.3 * (g[2] - g[0]):
                continue
            on_side = g[1] >= caption[3] - 8 if below else g[3] <= caption[1] + 8
            if not on_side or _y_gap(g, region) > ATTACH_GAP:
                continue
            lo, hi = (region[3], g[1]) if below else (g[3], region[1])
            if any(lo <= (b[1] + b[3]) / 2 <= hi and _x_overlap(b, g) > 0 for b in blockers):
                continue
            picked.append(i)
            region = _union(region, g)
            changed = True
    return picked

def _grow(seed: int, graphics: Sequence[Box], claimed: set) -> List[int]:
    """A seed graphic plus the unclaimed graphics chained to it (sub-plots of one figure)."""
    group, region = [seed], graphics[seed]
    changed = True
    while changed:
        changed = False
        for i, g in enumerate(graphics):
            if i in claimed or i in group:
                continue
            x_gap = max(0.0, max(g[0], region[0]) - min(g[2], region[2]))
            if _y_gap(g, region) <= 45 and x_gap <= 45:
                group.append(i)
                region = _union(region, g)
                changed = True
    return group

def _absorb_labels(region: Box, text_blocks: Sequence[dict], limit: Optional[float], below: bool, rect) -> Box:
    """Extend a figure region over short text that belongs to it (axis titles, tick labels, legends)."""
    for _ in range(2):
        for b in text_blocks:
            box = b["bbox"]
            if len(b["text"].strip()) > 120:
                continue
            if limit is not None and (box[1] < limit - 2 if below else box[3] > limit + 2):
                continue   # never cross the caption
            near = (box[0] < region[2] + 14 and box[2] > region[0] - 14 and
                    box[1] < region[3] + 14 and box[3] > region[1] - 14)
            if near:
                region = _union(region, box)
    pad = 4.0
    out = (max(rect.x0, region[0] - pad), max(rect.y0, region[1] - pad),
           min(rect.x1, region[2] + pad), min(rect.y1, region[3] + pad))
    if limit is not None:
        out = (out[0], max(out[1], limit), out[2], out[3]) if below else (out[0], out[1], out[2], min(out[3], limit))
    return out

# =========================
# RENDERING
# =========================
def render_region(page, box: Box, out_stem: str) -> Optional[dict]:
    """Render a page region to an image file. Returns {path, width, height} or None when the region is too small."""
    clip = fitz.Rect(box) & page.rect
    if clip.width < 50 or clip.height < 35:
        return None
    dpi = min(FIGURE_DPI, int(MAX_FIGURE_PIXELS * 72 / max(clip.width, clip.height)))
    pix = page.get_pixmap(clip=clip, dpi=max(dpi, 60), alpha=False)
    data, ext = pix.tobytes("png"), "png"
    if len(data) > 1_200_000:          # photos: keep the upload to the vision model small
        data, ext = pix.tobytes("jpeg", jpg_quality=85), "jpg"
    path = f"{out_stem}.{ext}"
    with open(path, "wb") as f:
        f.write(data)
    return {"path": path, "width": pix.width, "height": pix.height}

# =========================
# LOCATE FIGURES
# =========================
def locate_figures(doc, captions: Sequence[dict], text_blocks: Dict[int, List[dict]],
                   table_boxes: Dict[int, List[Box]], figure_dir: str, max_figures: int = MAX_FIGURES,
                   two_column: Optional[Dict[int, bool]] = None) -> List[dict]:
    """Find, render and describe every figure of a document.

    captions:    caption dicts with keys kind, number, label, text, page, section, bbox
    text_blocks: page number -> [{"bbox", "text"}] (all text blocks, captions included)
    table_boxes: page number -> table bounding boxes
    two_column:  page number -> whether the page is laid out in two columns
    Returns figure dicts: {id, number, label, page, box, caption, caption_page, path, width, height}.
    """
    os.makedirs(figure_dir, exist_ok=True)
    n_pages = doc.page_count
    graphics_cache: Dict[int, List[Box]] = {}
    claimed: Dict[int, set] = {}

    def graphics(page_no: int) -> List[Box]:
        if page_no not in graphics_cache:
            graphics_cache[page_no] = page_graphics(doc[page_no - 1], table_boxes.get(page_no, ()))
            claimed[page_no] = set()
        return graphics_cache[page_no]

    def blockers(page_no: int, own: Box) -> List[Box]:
        caps = [tuple(c["bbox"]) for c in captions if c["page"] == page_no and tuple(c["bbox"]) != tuple(own)]
        prose = [tuple(b["bbox"]) for b in text_blocks.get(page_no, []) if len(b["text"].strip()) >= 150
                 and tuple(b["bbox"]) != tuple(own)]
        return caps + prose

    fig_caps = [c for c in captions if c["kind"] == "figure"]
    found: List[dict] = []    # {caption, page, box, y}

    # Pass 1: graphics on the caption's own page (above it; below only if nothing is above)
    pending = []
    for cap in fig_caps:
        page_no, cbox = cap["page"], tuple(cap["bbox"])
        gs = graphics(page_no)
        rect = doc[page_no - 1].rect
        block = blockers(page_no, cbox)
        below = False
        cols = (two_column or {}).get(page_no, True)
        picked = attach_graphics(cbox, gs, claimed[page_no], block, rect, two_column=cols)
        if not picked:
            picked = attach_graphics(cbox, gs, claimed[page_no], block, rect, below=True, two_column=cols)
            below = bool(picked)
        if not picked:
            pending.append(cap)
            continue
        claimed[page_no].update(picked)
        region = gs[picked[0]]
        for i in picked[1:]:
            region = _union(region, gs[i])
        limit = cbox[3] if below else cbox[1]
        region = _absorb_labels(region, [b for b in text_blocks.get(page_no, []) if tuple(b["bbox"]) != cbox],
                                limit, below, rect)
        found.append({"caption": cap, "page": page_no, "box": region})

    # Pass 2: the caption was pushed to another page; take the largest unclaimed figure on the previous page,
    # then the next one
    for cap in pending:
        for page_no in (cap["page"] - 1, cap["page"] + 1):
            if not 1 <= page_no <= n_pages:
                continue
            gs = graphics(page_no)
            free = [i for i in range(len(gs)) if i not in claimed[page_no] and _area(gs[i]) >= 80 * 60]
            if not free:
                continue
            group = _grow(max(free, key=lambda i: _area(gs[i])), gs, claimed[page_no])
            claimed[page_no].update(group)
            region = gs[group[0]]
            for i in group[1:]:
                region = _union(region, gs[i])
            region = _absorb_labels(region, text_blocks.get(page_no, []), None, False, doc[page_no - 1].rect)
            found.append({"caption": cap, "page": page_no, "box": region})
            break

    # Pass 3: sizeable bitmaps that no caption claimed (still worth showing and explaining)
    for page_no in sorted(graphics_cache):
        page = doc[page_no - 1]
        bitmaps = []
        try:
            bitmaps = [tuple(fitz.Rect(info["bbox"]) & page.rect) for info in page.get_image_info()]
        except Exception:
            pass
        for i, g in enumerate(graphics_cache[page_no]):
            if i in claimed[page_no] or g not in bitmaps or g[2] - g[0] < 100 or g[3] - g[1] < 100:
                continue
            claimed[page_no].add(i)
            found.append({"caption": None, "page": page_no, "box": g})

    found.sort(key=lambda f: (f["page"], f["box"][1], f["box"][0]))
    figures: List[dict] = []
    used_ids: set = set()
    for item in found:
        if len(figures) >= max_figures:
            break
        cap = item["caption"]
        number = str(cap["number"]) if cap else ""
        fig_id = f"fig{number}" if number.isdigit() and f"fig{number}" not in used_ids else ""
        if not fig_id:
            k = 1
            while f"img{k}" in used_ids:
                k += 1
            fig_id = f"img{k}"
        rendered = render_region(doc[item["page"] - 1], item["box"], os.path.join(figure_dir, fig_id))
        if rendered is None:
            continue
        used_ids.add(fig_id)
        figures.append({"id": fig_id, "number": number, "label": cap["label"] if cap else "",
                        "page": item["page"], "caption": cap["text"] if cap else "",
                        "caption_page": cap["page"] if cap else None, "section": "", "box": tuple(item["box"]),
                        "mentions": [], **rendered})
    return figures

# =========================
# MENTIONS ("As shown in Fig. 3, ...")
# =========================
_ABBREV = re.compile(r"\b(Figs?|Eqs?|Secs?|Tabs?|Refs?|al|e\.g|i\.e|vs|cf|approx|resp|No|Nos)\.", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_SUB = r"(?:\s*\([a-h]\)|[a-h](?![a-z]))?"                      # sub-figure letter: 3(a), 3a
_SEP = r"\s*(?:,|and|&|\u2013|-|to)\s*"
_ARABIC = r"\d{1,3}" + _SUB
_ROMAN = r"[IVXL]{1,6}\b"
_REF_PATTERNS = {
    # Figures are numbered with digits; matching is case-insensitive ("fig. 3", "FIGURE 3")
    "figure": re.compile(r"\bfig(?:ure)?s?\s*\.?\s*(" + _ARABIC + r"(?:" + _SEP + _ARABIC + r")*)", re.I),
    # Tables may use Roman numerals (IEEE style), which are only trusted in upper case after a capitalised "Table"
    "table": re.compile(r"\bT(?:ables?|ABLES?)\s*\.?\s*((?:" + _ARABIC + "|" + _ROMAN + r")(?:" + _SEP
                        + r"(?:" + _ARABIC + "|" + _ROMAN + r"))*)"),
}
_NUM_OR_RANGE = re.compile(r"(\d{1,3}|[IVXL]{1,6})(?:[a-h]|\s*\([a-h]\))?(?:\s*(?:\u2013|-|to)\s*(\d{1,3}))?")

def referenced_numbers(text: str, kind: str = "figure") -> List[str]:
    """Figure/table numbers a piece of text refers to: 'Figs. 2 and 3' -> ['2', '3'], 'Figures 2-4' -> ['2','3','4']."""
    out: List[str] = []
    for m in _REF_PATTERNS[kind].finditer(text or ""):
        for first, last in _NUM_OR_RANGE.findall(m.group(1)):
            numbers = [first.upper()]
            if last and first.isdigit() and 0 < int(last) - int(first) <= 12:
                numbers = [str(n) for n in range(int(first), int(last) + 1)]
            elif last:
                numbers.append(last)
            for n in numbers:
                if n not in out:
                    out.append(n)
    return out

def split_sentences(text: str) -> List[str]:
    flat = re.sub(r"\s+", " ", text or "").strip()
    protected = _ABBREV.sub(lambda m: m.group(1) + "§", flat)
    return [s.replace("§", ".").strip() for s in _SENTENCE_END.split(protected) if s.strip()]

def collect_mentions(segments: Sequence[dict], kind: str = "figure") -> Dict[str, List[dict]]:
    """Every place the running text refers to a figure/table: number -> [{page, section, text}].

    A mention is the sentence that names the figure plus the sentence after it (which usually carries the
    actual observation: 'Fig. 3 shows the loss. It plateaus after 40 epochs.').
    """
    mentions: Dict[str, List[dict]] = {}
    for seg in segments:
        sentences = split_sentences(seg["text"])
        for i, sentence in enumerate(sentences):
            numbers = referenced_numbers(sentence, kind)
            if not numbers:
                continue
            text = sentence
            if i + 1 < len(sentences) and len(text) + len(sentences[i + 1]) <= MENTION_CHARS:
                text += " " + sentences[i + 1]
            text = text[:MENTION_CHARS]
            for n in numbers:
                bucket = mentions.setdefault(n, [])
                if len(bucket) < MAX_MENTIONS and all(text != m["text"] for m in bucket):
                    bucket.append({"page": seg["page"], "section": seg["section"], "text": text})
    return mentions

def attach_mentions(items: Sequence[dict], segments: Sequence[dict], kind: str = "figure"):
    """Set item['mentions'] for figures or tables that carry a 'number' (or a 'label' like 'Table II')."""
    by_number = collect_mentions(segments, kind)
    for item in items:
        number = str(item.get("number") or "").upper() or (str(item.get("label", "")).split(" ")[-1].upper())
        item["mentions"] = by_number.get(number, []) if number else []

# =========================
# CONTEXT FOR THE MODEL
# =========================
def figure_context(fig: dict, max_chars: int = 2200) -> str:
    """Caption + in-text mentions of a figure, as grounded text for the model."""
    parts = []
    if fig.get("caption"):
        where = f" (caption printed on page {fig['caption_page']})" if fig.get("caption_page") not in (None, fig.get("page")) else ""
        parts.append(f"Caption{where}: {fig['caption']}")
    if fig.get("mentions"):
        parts.append("What the paper's text says about this figure:")
        parts += [f"- [p. {m['page']} · {m['section']}] {m['text']}" for m in fig["mentions"]]
    return "\n".join(parts)[:max_chars]

def public_figure(fig: dict, paper_id: str, analysis: str = "") -> dict:
    """What the browser needs to show a figure (no server file paths)."""
    return {"id": fig["id"], "label": fig.get("label") or "", "page": fig["page"], "caption": fig.get("caption", ""),
            "caption_page": fig.get("caption_page"), "section": fig.get("section", ""),
            "mentions": fig.get("mentions", []), "width": fig.get("width"), "height": fig.get("height"),
            "image_url": f"/papers/{paper_id}/figures/{fig['id']}", "analysis": analysis}

# =========================
# EXPLAINING A FIGURE
# =========================
EXPLAIN_PROMPT = """You are explaining one figure from a research paper to a student who has not read the paper.

{label}
{context}

Look at the image and write a short explanation in markdown with exactly these parts:

**What it is** - one sentence: the kind of figure and what is drawn or plotted (axes with units, legend entries, blocks).
**How to read it** - one or two sentences a beginner needs (what each axis, colour or arrow means).
**What it shows** - 2 to 4 bullet points with concrete observations: trends, comparisons, values you can read off.
**Why it matters in the paper** - connect it to what the paper's text says about it, citing pages like (p. 5).
**Check** - one line: does the picture support what the text claims about it? Say what you could not read or verify.

Rules:
- Use the caption and the quoted passages to name things correctly. Do not contradict them.
- Read values from the image only when they are legible; otherwise say "approximately" or "not readable".
- Never invent numbers, labels, or conclusions that are in neither the image nor the quoted text.
- If the text does not discuss the figure beyond its caption, say so in "Why it matters"."""

QUESTION_PROMPT = """Answer a question about one figure from a research paper.

{label}
{context}

Question: {question}

Use the image together with the caption and the quoted passages. Cite pages like (p. 5) when you rely on the text.
If the figure and the passages cannot answer the question, say that plainly. Do not invent values. Answer in markdown, briefly."""

TEXT_ONLY_PROMPT = """The image of a figure from a research paper could not be analysed, so explain the figure
from the paper's own words only.

{label}
{context}

{task}

Rules: use ONLY the caption and passages above. Cite pages like (p. 5). Do not describe visual details
(colours, shapes, exact values) that the text does not state. Answer in markdown, briefly."""

def _label(fig: dict) -> str:
    where = f"page {fig['page']}"
    return f"Figure: {fig.get('label') or fig['id']} ({where})"

def explain_figure(fig: dict, question: str = "") -> dict:
    """Explain a figure (or answer a question about it) from the image + caption + in-text mentions.

    Returns {"text", "mode"}: mode is "vision" when the image was read, "text-only" when the vision model was
    unavailable and the explanation was written from the caption and mentions alone.
    Raises LLMError only when neither route works.
    """
    from src.llm import LLMBudgetExceeded, LLMError, llm, vision

    context = figure_context(fig) or "The paper gives no caption and never refers to this figure in its text."
    prompt = (QUESTION_PROMPT.format(label=_label(fig), context=context, question=question) if question
              else EXPLAIN_PROMPT.format(label=_label(fig), context=context))
    try:
        return {"text": vision(fig["path"], prompt), "mode": "vision"}
    except LLMBudgetExceeded:
        raise
    except LLMError as vision_error:
        if not (fig.get("caption") or fig.get("mentions")):
            raise
        task = f"Question: {question}" if question else \
            "Explain what the figure shows and why it matters in the paper, in 3 to 5 sentences."
        text = llm(TEXT_ONLY_PROMPT.format(label=_label(fig), context=context, task=task), max_tokens=500)
        note = f"*Written from the caption and the paper's text only; the image could not be analysed ({vision_error}).*"
        return {"text": f"{note}\n\n{text}", "mode": "text-only"}

def explain_paper_figure(paper, figure_id: str, force: bool = False) -> dict:
    """Cached explanation of one of a paper's figures. A successful image-based explanation is also added to
    the paper's search index, so later questions can retrieve what the figure shows."""
    fig = paper.figure(figure_id)
    if fig is None:
        raise ValueError(f"No figure '{figure_id}'. Available: {', '.join(f['id'] for f in paper.figures) or 'none'}")
    cached_mode = paper.figure_modes.get(figure_id)
    if figure_id in paper.figure_analyses and not (force and cached_mode != "vision"):
        return {"text": paper.figure_analyses[figure_id], "mode": cached_mode or "vision", "cached": True}
    result = explain_figure(fig)
    paper.figure_analyses[figure_id] = result["text"]
    paper.figure_modes[figure_id] = result["mode"]
    if result["mode"] == "vision":
        title = fig.get("caption") or fig.get("label") or figure_id
        paper.add_chunk({"text": f"{title}\nWhat the figure shows (vision analysis): {result['text']}",
                         "page": fig["page"], "section": fig.get("section") or "", "kind": "figure",
                         "figure_id": figure_id})
    return {**result, "cached": False}

def figures_named_in(paper, text: str) -> List[dict]:
    """Figures of the paper that a question refers to by number ('What does Fig. 3 show?')."""
    numbers = referenced_numbers(text, "figure")
    return [f for f in paper.figures if f.get("number") and f["number"] in numbers]

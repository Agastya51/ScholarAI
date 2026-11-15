"""Source viewer: render a page of the original PDF with the cited passage highlighted.

Lets a reader check any claim against the paper itself instead of trusting a page number.
"""
import re
from typing import List, Optional, Sequence, Tuple

try:
    import pymupdf as fitz
except ImportError:
    import fitz

VIEW_DPI = 110
MAX_SEARCHES = 40
WINDOW_WORDS = 8

def highlight_phrases(text: str, window: int = WINDOW_WORDS) -> List[str]:
    """Split a passage into short runs of words that can be found on the page one by one.

    Whole sentences rarely match a PDF's text layer exactly (line breaks, hyphenation, ligatures), short runs
    almost always do, and together they cover the passage.
    """
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    phrases = [" ".join(words[i:i + window]) for i in range(0, len(words), window)]
    return [p for p in phrases if len(p) >= 12]

def render_page(pdf_path: str, page_no: int, phrases: Sequence[str] = (),
                boxes: Sequence[Tuple[float, float, float, float]] = (), dpi: int = VIEW_DPI) -> Tuple[bytes, int]:
    """PNG of one page (1-based) with phrases highlighted and boxes outlined.
    Returns (png bytes, number of highlighted phrases). The PDF on disk is never modified."""
    doc = fitz.open(pdf_path)
    try:
        if not 1 <= page_no <= doc.page_count:
            raise ValueError(f"The paper has {doc.page_count} pages; there is no page {page_no}.")
        page = doc[page_no - 1]
        found = 0
        for phrase in list(phrases)[:MAX_SEARCHES]:
            try:
                quads = page.search_for(phrase, quads=True)
            except Exception:
                quads = []
            if quads:
                page.add_highlight_annot(quads)
                found += 1
        for box in boxes:
            page.draw_rect(fitz.Rect(box) + (-3, -3, 3, 3), color=(0.95, 0.72, 0.05), width=2.2)
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png"), found
    finally:
        doc.close()

def page_view(paper, page_no: int, chunk_id: Optional[int] = None, figure_id: Optional[str] = None,
              quote: str = "") -> Tuple[bytes, int]:
    """Render a page of `paper` highlighting a chunk, a figure, a table, or a free-text quote."""
    phrases: List[str] = []
    boxes = []
    if quote:
        phrases = highlight_phrases(quote)
    if chunk_id is not None and 0 <= chunk_id < len(paper.chunks):
        chunk = paper.chunks[chunk_id]
        figure_id = figure_id or chunk.get("figure_id")
        table = next((t for t in paper.tables if t["id"] == chunk.get("table_id")), None)
        if table is not None and table.get("box") and table["page"] == page_no:
            boxes.append(table["box"])
        elif chunk.get("kind") in ("text", "reference", "caption", None) and chunk.get("page") == page_no:
            phrases += highlight_phrases(str(chunk["text"]))
    if figure_id:
        fig = paper.figure(figure_id)
        if fig is not None and fig["page"] == page_no:
            boxes.append(fig["box"])
    return render_page(paper.path, page_no, phrases, boxes)

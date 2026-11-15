"""Builds a small synthetic two-column 'paper' PDF for parsing tests (no network, no fixtures on disk)."""
import numpy as np

try:
    import pymupdf as fitz
except ImportError:
    import fitz

INTRO = ("Traffic forecasting on road networks is a long-standing problem in intelligent transportation. "
         "Recurrent models such as DCRNN capture temporal structure but scale poorly to large graphs. "
         "This work studies whether attention over road segments can replace recurrence entirely. "
         "LEFTCOLUMNMARKER closes the introduction paragraph of the left column.")
METHOD = ("FlowNet-X stacks diffusion convolution layers with a gated temporal attention block. "
          "The learning rate is 0.001 and the batch size is 64. RIGHTCOLUMNMARKER opens the right column. "
          "Training uses the Adam optimizer for 100 epochs on a single GPU.")
RESULTS = ("Compared with DCRNN, FlowNet-X reduces MAE by 12.5% on METR-LA and by 8.1% on PEMS-BAY. "
           "The gains are largest at the 60 minute horizon where recurrent baselines accumulate error.")
CONCLUSION = ("Limitations include the quadratic cost of attention on large road networks. "
              "Future work will study sparse attention to reduce memory consumption.")

def _noise_png(seed: int, w: int = 220, h: int = 160) -> bytes:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8).tobytes()
    return fitz.Pixmap(fitz.csRGB, w, h, data, False).tobytes("png")

def _text(page, rect, text, size=9):
    left_over = page.insert_textbox(rect, text, fontsize=size, fontname="helv")
    assert left_over >= 0, f"text did not fit in {rect}"

def _table(page, x0, y0, rows, col_w=75, row_h=18):
    """Draw a ruled table so PyMuPDF's line-based table finder detects it."""
    n_cols = len(rows[0])
    for r in range(len(rows) + 1):
        page.draw_line((x0, y0 + r * row_h), (x0 + n_cols * col_w, y0 + r * row_h))
    for c in range(n_cols + 1):
        page.draw_line((x0 + c * col_w, y0), (x0 + c * col_w, y0 + len(rows) * row_h))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            page.insert_text((x0 + c * col_w + 4, y0 + r * row_h + 12), cell, fontsize=8, fontname="helv")

def build_two_column_pdf(path: str) -> str:
    doc = fitz.open()

    # ---- page 1: full-width title + abstract, then two columns ----
    p1 = doc.new_page(width=595, height=842)
    _text(p1, fitz.Rect(50, 50, 550, 80), "FlowNet-X: Attention-Based Traffic Forecasting on Road Graphs", 15)
    _text(p1, fitz.Rect(50, 95, 550, 175),
          "Abstract-We propose FlowNet-X, a graph neural network for traffic forecasting. FlowNet-X achieves "
          "94.3% accuracy on the METR-LA dataset while training three times faster than recurrent baselines. "
          "The model replaces recurrence with gated temporal attention over road segments.")
    # The right column is written to the file BEFORE the left one, so the PDF's native text order is wrong
    # and only a layout-aware reader recovers Introduction -> Method.
    _text(p1, fitz.Rect(310, 186, 550, 206), "II. METHOD", 10)
    _text(p1, fitz.Rect(310, 208, 550, 300), METHOD)
    _text(p1, fitz.Rect(50, 186, 290, 206), "I. INTRODUCTION", 10)
    _text(p1, fitz.Rect(50, 208, 290, 330), INTRO)
    p1.insert_image(fitz.Rect(330, 320, 530, 460), stream=_noise_png(1))
    _text(p1, fitz.Rect(310, 466, 550, 500),
          "Fig. 1. Architecture of FlowNet-X with diffusion convolution and gated temporal attention.", 8)

    # ---- page 2: results with a ruled table; a figure at the bottom whose caption is on page 3 ----
    p2 = doc.new_page(width=595, height=842)
    _text(p2, fitz.Rect(50, 46, 550, 66), "III. RESULTS", 10)
    _text(p2, fitz.Rect(50, 70, 550, 120), RESULTS)
    _text(p2, fitz.Rect(50, 130, 550, 160), "TABLE I\nMAE ON TWO BENCHMARKS", 8)
    _table(p2, 60, 165, [["Model", "METR-LA", "PEMS-BAY"], ["DCRNN", "3.60", "2.07"], ["FlowNet-X", "3.15", "1.90"]])
    p2.insert_image(fitz.Rect(150, 600, 450, 800), stream=_noise_png(2))

    # ---- page 3: the orphan caption, conclusion, references ----
    p3 = doc.new_page(width=595, height=842)
    _text(p3, fitz.Rect(50, 50, 550, 80),
          "Figure 2: Forecast error versus prediction horizon for FlowNet-X and DCRNN on METR-LA.", 8)
    _text(p3, fitz.Rect(50, 91, 550, 111), "IV. CONCLUSION", 10)
    _text(p3, fitz.Rect(50, 114, 550, 160), CONCLUSION)
    _text(p3, fitz.Rect(50, 171, 550, 191), "REFERENCES", 10)
    _text(p3, fitz.Rect(50, 194, 550, 300),
          "[1] Y. Li, R. Yu, C. Shahabi, and Y. Liu, Diffusion convolutional recurrent neural network: "
          "Data-driven traffic forecasting, in Proc. ICLR, 2018.\n"
          "[2] A. Vaswani et al., Attention is all you need, in Proc. NeurIPS, 2017.\n"
          "[3] B. Yu, H. Yin, and Z. Zhu, Spatio-temporal graph convolutional networks, in Proc. IJCAI, 2018.", 8)
    doc.save(path)
    doc.close()
    return path

def _axes(page, x0, y0, w, h, points, label_x, label_y):
    """A small vector line chart: axes, a polyline, tick labels and axis titles typed as text."""
    page.draw_line((x0, y0), (x0, y0 + h))
    page.draw_line((x0, y0 + h), (x0 + w, y0 + h))
    pts = [(x0 + px * w, y0 + h - py * h) for px, py in points]
    page.draw_polyline(pts, color=(0.1, 0.3, 0.7), width=1.5)
    for px, py in pts:
        page.draw_circle((px, py), 2.5, color=(0.1, 0.3, 0.7), fill=(0.1, 0.3, 0.7))
    for i in range(5):
        page.insert_text((x0 + i * w / 4 - 4, y0 + h + 11), str(i * 25), fontsize=7, fontname="helv")
    page.insert_text((x0 + w / 2 - 20, y0 + h + 24), label_x, fontsize=8, fontname="helv")
    page.insert_text((x0 - 2, y0 - 6), label_y, fontsize=8, fontname="helv")

def build_vector_figure_pdf(path: str) -> str:
    """A paper whose figures are vector drawings (no embedded bitmaps) and whose discussion of each figure is
    on a different page from the figure itself."""
    doc = fitz.open()
    p1 = doc.new_page(width=595, height=842)
    _text(p1, fitz.Rect(50, 50, 550, 80), "Loss Curves Without Bitmaps", 15)
    _text(p1, fitz.Rect(50, 96, 550, 116), "1 Introduction", 10)
    _text(p1, fitz.Rect(50, 120, 550, 220),
          "Training dynamics are central to this study. As shown in Fig. 1, the validation loss of the gated "
          "model falls below 0.20 after 50 epochs. The ungated model never reaches that level. "
          "Figs. 1 and 2 together show that the gate matters most early in training. "
          "Unrelated sentence that mentions nothing of interest.")
    p2 = doc.new_page(width=595, height=842)
    _text(p2, fitz.Rect(50, 46, 550, 66), "2 Results", 10)
    _text(p2, fitz.Rect(50, 70, 550, 110),
          "The curves in Figure 2(a) correspond to RiverCity and those in Figure 2(b) to HarborBay. "
          "Both panels use the same axes.")
    _axes(p2, 180, 150, 220, 140, [(0, .9), (.25, .6), (.5, .35), (.75, .22), (1, .18)], "Epoch", "Validation loss")
    _text(p2, fitz.Rect(50, 325, 550, 350), "Fig. 1. Validation loss of the gated model over 100 epochs.", 8)
    # Figure 2: two panels side by side
    _axes(p2, 80, 420, 180, 120, [(0, .8), (.5, .5), (1, .3)], "Epoch", "Loss (a)")
    _axes(p2, 330, 420, 180, 120, [(0, .7), (.5, .6), (1, .5)], "Epoch", "Loss (b)")
    _text(p2, fitz.Rect(50, 575, 550, 600),
          "Figure 2: Loss on (a) RiverCity and (b) HarborBay for the ungated model.", 8)
    _text(p2, fitz.Rect(50, 620, 550, 700),
          "A long closing paragraph follows the figures so that the page has ordinary prose below them. "
          "It restates that the experiments were repeated five times and averaged, and that all models share "
          "the same optimizer settings described earlier in the paper.")
    doc.save(path)
    doc.close()
    return path

"""Layout-aware PDF parsing: reading order, headings, captions, tables, references."""
import os
import shutil
import tempfile
import unittest

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.rag import (
    build_chunks, clean_text, detect_caption, heading_level, load_paper, ordered_blocks, parse_document,
    parse_references,
)
from tests.fakes import hash_encoder
from tests.pdf_factory import build_two_column_pdf

SAMPLE_PAPER = os.path.join(os.path.dirname(__file__), "..", "evals", "sample", "sample_paper.pdf")

class TestTextCleaning(unittest.TestCase):
    def test_ligatures_and_hyphenation(self):
        self.assertEqual(clean_text("eﬃcient traﬃc ﬂow"), "efficient traffic flow")
        self.assertEqual(clean_text("quadrat-\nically"), "quadratically")
        self.assertEqual(clean_text("the 60-\nminute horizon"), "the 60-minute horizon")   # real hyphen is kept
        self.assertEqual(clean_text("RiverCity-\n200 and PEMS-\nBAY"), "RiverCity-200 and PEMS-\nBAY")
        self.assertEqual(clean_text("A\x12B"), "AB")

    def test_caption_detection(self):
        for text, label in [("Fig. 3. Training loss over epochs.", "Figure 3"), ("Figure 12: Results", "Figure 12"),
                            ("TABLE II\nCOMPARISON OF MODELS", "Table II"), ("Table 1: MAE at 60 minutes", "Table 1")]:
            cap = detect_caption(text)
            self.assertIsNotNone(cap, text)
            self.assertEqual(cap["label"], label)
        for prose in ["Figure 3 shows that the loss decreases quickly.", "Tables are hard to parse.",
                      "Figures 2 and 3 compare the baselines."]:
            self.assertIsNone(detect_caption(prose), prose)

    def test_heading_levels(self):
        self.assertEqual(heading_level("3 Method"), 1)
        self.assertEqual(heading_level("IV. EXPERIMENTS"), 1)
        self.assertEqual(heading_level("Abstract"), 1)
        self.assertEqual(heading_level("3.1 Training Details"), 2)
        self.assertEqual(heading_level("A. Datasets"), 2)

    def test_reference_parsing(self):
        refs = parse_references("[1] A. Author, A first paper title, in Proc. X, 2019.\n"
                                "[2] B. Author and C. Author, A second paper title that is long, 2020.")
        self.assertEqual([r["n"] for r in refs], [1, 2])
        self.assertIn("second paper", refs[1]["text"])
        numbered = parse_references("1. A. Author, A first paper title, in Proc. X, 2019.\n"
                                    "2. B. Author, A second paper title that is long, 2020.")
        self.assertEqual(len(numbered), 2)

class TestTwoColumnPdf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pdf = build_two_column_pdf(os.path.join(cls.tmp, "paper.pdf"))
        cls.parsed = parse_document(cls.pdf, figure_dir=os.path.join(cls.tmp, "figs"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_native_order_is_wrong_but_parser_reads_left_column_first(self):
        native = fitz.open(self.pdf)[0].get_text()
        self.assertLess(native.index("RIGHTCOLUMNMARKER"), native.index("LEFTCOLUMNMARKER"))  # the trap
        text = " ".join(s["text"] for s in self.parsed["segments"])
        self.assertLess(text.index("LEFTCOLUMNMARKER"), text.index("RIGHTCOLUMNMARKER"))
        sections = [s["section"] for s in self.parsed["segments"]]
        self.assertLess(sections.index("I. INTRODUCTION"), sections.index("II. METHOD"))

    def test_full_width_blocks_come_before_columns(self):
        blocks = ordered_blocks(fitz.open(self.pdf)[0])
        self.assertIn("FlowNet-X: Attention-Based", blocks[0]["text"])
        self.assertIn("Abstract", blocks[1]["text"])

    def test_figure_caption_on_same_page(self):
        fig1 = self.parsed["figures"][0]
        self.assertEqual((fig1["page"], fig1["label"]), (1, "Figure 1"))
        self.assertIn("Architecture of FlowNet-X", fig1["caption"])

    def test_figure_caption_on_next_page_is_linked(self):
        fig2 = self.parsed["figures"][1]
        self.assertEqual(fig2["page"], 2)
        self.assertEqual(fig2["caption_page"], 3)
        self.assertIn("Forecast error versus prediction horizon", fig2["caption"])

    def test_captions_do_not_interrupt_prose_and_are_indexed_separately(self):
        prose = " ".join(s["text"] for s in self.parsed["segments"])
        self.assertNotIn("Architecture of FlowNet-X with diffusion", prose)
        captions = [c for c in build_chunks(self.parsed) if c["kind"] == "caption"]
        self.assertEqual(len(captions), 2)
        self.assertEqual(captions[0]["figure_id"], "fig1")

    def test_table_keeps_decimal_points_and_caption(self):
        table = self.parsed["tables"][0]
        self.assertIn("|DCRNN|3.60|2.07|", table["markdown"])
        self.assertIn("|Model|METR-LA|PEMS-BAY|", table["markdown"])
        self.assertEqual(table["label"], "Table I")
        self.assertEqual(table["section"], "III. RESULTS")
        chunk = next(c for c in build_chunks(self.parsed) if c["kind"] == "table")
        self.assertTrue(chunk["text"].startswith("Table I: MAE ON TWO BENCHMARKS"))
        prose = " ".join(s["text"] for s in self.parsed["segments"])
        self.assertNotIn("3.60", prose)   # cell text lives in the table chunk only

    def test_references_are_parsed_and_kept_out_of_normal_search(self):
        self.assertEqual([r["n"] for r in self.parsed["references"]], [1, 2, 3])
        paper = load_paper(self.pdf, encoder=hash_encoder)
        self.assertEqual(len(paper.references), 3)
        kinds = {c["kind"] for c in paper.search("attention is all you need", k=4)}
        self.assertNotIn("reference", kinds)
        cited = paper.search("which reference is about attention is all you need", k=3)
        self.assertIn("reference", {c["kind"] for c in cited})
        from src.retrieval import retrieve_context
        numbered = retrieve_context(paper, "What is reference [2]?", k=2, expansion="off")
        self.assertEqual(numbered.chunks[0]["kind"], "reference")
        self.assertIn("[2] A. Vaswani", numbered.chunks[0]["text"])
        shutil.rmtree(os.path.join("uploads", "figures", paper.id), ignore_errors=True)

    def test_page_limit(self):
        from unittest.mock import patch
        with patch("src.rag.MAX_PDF_PAGES", 2):
            with self.assertRaises(ValueError):
                parse_document(self.pdf, figure_dir=os.path.join(self.tmp, "figs2"))

@unittest.skipUnless(os.path.exists(SAMPLE_PAPER), "sample paper not present")
class TestLatexSamplePaper(unittest.TestCase):
    """A real pdfLaTeX two-column paper: split heading lines, booktabs tables, floats."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.parsed = parse_document(SAMPLE_PAPER, figure_dir=cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_numbered_sections_and_subsections(self):
        sections = list(dict.fromkeys(s["section"] for s in self.parsed["segments"]))
        self.assertIn("1 Introduction", sections)
        self.assertIn("3 Method › 3.3 Gated Temporal Attention", sections)
        self.assertIn("4 Experiments › 4.3 Implementation Details", sections)
        self.assertLess(sections.index("1 Introduction"), sections.index("2 Related Work"))

    def test_tables_figures_references(self):
        self.assertEqual(len(self.parsed["tables"]), 2)
        self.assertIn("|RecurNet [1]|3.62|2.11|", self.parsed["tables"][0]["markdown"])
        self.assertEqual(self.parsed["tables"][0]["section"], "4 Experiments › 4.4 Main Results")
        self.assertEqual([f["label"] for f in self.parsed["figures"]], ["Figure 1", "Figure 2"])
        self.assertEqual(len(self.parsed["references"]), 7)

    def test_real_hyphen_survives_line_break(self):
        text = " ".join(s["text"] for s in self.parsed["segments"])
        self.assertIn("60-minute horizon", text)

if __name__ == "__main__":
    unittest.main()

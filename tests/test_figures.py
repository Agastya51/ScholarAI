"""Figure understanding: vector figures, captions, in-text mentions on other pages, grounded explanation."""
import os
import shutil
import tempfile
import unittest

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.app import app
from src.figures import (
    collect_mentions, explain_figure, explain_paper_figure, figure_context, figures_named_in, referenced_numbers,
    split_sentences,
)
from src.llm import LLMError, set_backend, set_vision_backend
from src.rag import library, load_paper
from src.tools import run_tool
from src.workflows import qa_simple_graph
from tests.fakes import FakeLLM, hash_encoder
from tests.pdf_factory import build_vector_figure_pdf

class TestMentionParsing(unittest.TestCase):
    def test_referenced_numbers(self):
        self.assertEqual(referenced_numbers("As shown in Fig. 3, the loss falls."), ["3"])
        self.assertEqual(referenced_numbers("Figs. 2 and 3 compare the models."), ["2", "3"])
        self.assertEqual(referenced_numbers("Figures 2-4 show the ablation."), ["2", "3", "4"])
        self.assertEqual(referenced_numbers("Figure illustrates nothing; the table lists values.", "figure"), [])
        self.assertEqual(referenced_numbers("the table lists values in Tables 1 and 2", "table"), ["1", "2"])
        self.assertEqual(referenced_numbers("See Figure 5(a) and Fig.6b."), ["5", "6"])
        self.assertEqual(referenced_numbers("Table II lists the results.", "table"), ["II"])
        self.assertEqual(referenced_numbers("We configure the model as described."), [])

    def test_sentences_are_not_split_at_abbreviations(self):
        sentences = split_sentences("As shown in Fig. 3, the loss falls. It plateaus later (cf. Sec. 4). Done.")
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("As shown in Fig. 3, the loss falls."))

    def test_mention_carries_the_following_sentence(self):
        segments = [{"page": 4, "section": "Results", "text": "Fig. 2 plots accuracy. It rises until epoch 40. Other text."}]
        mention = collect_mentions(segments)["2"][0]
        self.assertEqual(mention["page"], 4)
        self.assertIn("It rises until epoch 40.", mention["text"])

class FigureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pdf = build_vector_figure_pdf(os.path.join(cls.tmp, "vector.pdf"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        library.reset()
        self.paper = library.add(load_paper(self.pdf, encoder=hash_encoder), make_active=True)
        self.prompts = []

    def tearDown(self):
        set_backend(None)
        set_vision_backend(None)
        library.reset()

    def see(self, answer="**What it is** - a zebrafish line chart."):
        def backend(path, prompt, max_tokens):
            self.prompts.append((path, prompt))
            return answer
        set_vision_backend(backend)

class TestFigureLocation(FigureTestCase):
    def test_vector_figures_are_found_without_any_bitmap(self):
        doc = fitz.open(self.pdf)
        self.assertEqual(sum(len(page.get_images()) for page in doc), 0)   # nothing a bitmap extractor could find
        self.assertEqual([f["id"] for f in self.paper.figures], ["fig1", "fig2"])
        for fig in self.paper.figures:
            self.assertTrue(os.path.exists(fig["path"]))
            self.assertGreater(fig["width"], 300)

    def test_dense_vector_plots_use_the_fast_clustering_and_find_the_same_figures(self):
        from unittest.mock import patch
        with patch("src.figures.MAX_PATHS_FOR_EXACT_CLUSTERING", 0):
            dense = load_paper(self.pdf, encoder=hash_encoder)
        self.assertEqual([f["id"] for f in dense.figures], ["fig1", "fig2"])
        for a, b in zip(dense.figures, self.paper.figures):
            self.assertLess(abs(a["width"] - b["width"]), 12)
        shutil.rmtree(os.path.join("uploads", "figures", dense.id), ignore_errors=True)

    def test_both_panels_of_a_two_panel_figure_are_in_one_image(self):
        fig1, fig2 = self.paper.figures
        self.assertGreater(fig2["width"], 1.6 * fig1["width"])

    def test_mentions_come_from_other_pages(self):
        fig1 = self.paper.figure("fig1")
        self.assertEqual(fig1["page"], 2)
        self.assertEqual({m["page"] for m in fig1["mentions"]}, {1})
        self.assertIn("falls below 0.20 after 50 epochs", fig1["mentions"][0]["text"])
        self.assertEqual(len(fig1["mentions"]), 2)                       # "Fig. 1" and "Figs. 1 and 2"
        self.assertTrue(any("Figure 2(a)" in m["text"] for m in self.paper.figure("fig2")["mentions"]))

    def test_axis_labels_stay_out_of_the_prose(self):
        prose = " ".join(s["text"] for s in self.paper.segments)
        self.assertNotIn("Validation loss\n", prose)
        self.assertNotIn("0 25 50 75 100", prose)
        self.assertIn("0 25 50 75 100", self.paper.text)                 # still part of the full text

    def test_context_joins_caption_and_mentions(self):
        context = figure_context(self.paper.figure("fig1"))
        self.assertIn("Caption: Figure 1: Validation loss of the gated model", context)
        self.assertIn("[p. 1 · 1 Introduction] As shown in Fig. 1", context)

class TestFigureExplanation(FigureTestCase):
    def test_vision_model_receives_image_caption_and_mentions(self):
        self.see()
        out = explain_figure(self.paper.figure("fig1"))
        path, prompt = self.prompts[0]
        self.assertEqual(out["mode"], "vision")
        self.assertTrue(path.endswith("fig1.png"))
        self.assertIn("Validation loss of the gated model", prompt)
        self.assertIn("falls below 0.20 after 50 epochs", prompt)        # said on page 1, figure is on page 2
        self.assertIn("(p. 5)", prompt)                                    # asks for page citations

    def test_question_about_a_figure(self):
        self.see("About 0.18 at epoch 100 (p. 1).")
        out = explain_figure(self.paper.figure("fig1"), question="What is the final loss?")
        self.assertIn("Question: What is the final loss?", self.prompts[0][1])
        self.assertEqual(out["text"], "About 0.18 at epoch 100 (p. 1).")

    def test_text_only_fallback_when_the_image_cannot_be_read(self):
        def broken(path, prompt, max_tokens):
            raise LLMError("rate limited")
        set_vision_backend(broken)
        fake = FakeLLM(lambda messages, tools, json_mode: "The loss falls below 0.20 (p. 1).")
        set_backend(fake)
        out = explain_paper_figure(self.paper, "fig1")
        self.assertEqual(out["mode"], "text-only")
        self.assertIn("image could not be analysed", out["text"])
        self.assertIn("falls below 0.20 after 50 epochs", fake.calls[0]["messages"][-1]["content"])
        # A text-only result is replaced as soon as the image can be read again
        self.see("**What it is** - a line chart.")
        retry = explain_paper_figure(self.paper, "fig1", force=True)
        self.assertEqual(retry["mode"], "vision")
        self.assertEqual(self.paper.figure_modes["fig1"], "vision")

    def test_failure_is_raised_not_cached_when_there_is_no_text_to_fall_back_on(self):
        def broken(path, prompt, max_tokens):
            raise LLMError("vision down")
        set_vision_backend(broken)
        fig = {**self.paper.figure("fig1"), "caption": "", "mentions": []}
        with self.assertRaises(LLMError):
            explain_figure(fig)
        self.assertNotIn("fig1", self.paper.figure_analyses)

    def test_explanation_is_cached_and_becomes_searchable(self):
        self.see("**What it shows** - a zebrafish-shaped curve.")
        before = len(self.paper.chunks)
        explain_paper_figure(self.paper, "fig1")
        explain_paper_figure(self.paper, "fig1")
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(len(self.paper.chunks), before + 1)
        hit = self.paper.search("zebrafish curve", k=1)[0]
        self.assertEqual((hit["kind"], hit["figure_id"], hit["page"]), ("figure", "fig1", 2))

    def test_unknown_figure(self):
        with self.assertRaises(ValueError):
            explain_paper_figure(self.paper, "fig9")

class TestFigureQuestions(FigureTestCase):
    def test_figures_named_in_a_question(self):
        self.assertEqual([f["id"] for f in figures_named_in(self.paper, "What does Fig. 2 show?")], ["fig2"])
        self.assertEqual(figures_named_in(self.paper, "What optimizer is used?"), [])

    def test_fast_rag_looks_at_the_figure_a_question_names(self):
        self.see("The curve ends near 0.18.")
        fake = FakeLLM(lambda messages, tools, json_mode: "It ends near 0.18 [p. 2].")
        set_backend(fake)
        result = qa_simple_graph.invoke({"paper_id": self.paper.id, "query": "What does Figure 1 show?", "trace": []})
        self.assertIn("Question: What does Figure 1 show?", self.prompts[0][1])
        answer_prompt = fake.calls[-1]["messages"][-1]["content"]
        self.assertIn("figure analysis", answer_prompt)
        self.assertIn("The curve ends near 0.18.", answer_prompt)
        self.assertEqual(result["sources"][0], {"type": "figure", "paper_id": self.paper.id, "page": 2, "figure_id": "fig1"})

    def test_agent_tool_passes_the_question_and_returns_mention_sources(self):
        self.see("Panel (a) is RiverCity.")
        result = run_tool("analyze_figure", {"figure_id": "fig2", "question": "Which panel is RiverCity?"})
        self.assertFalse(result.error)
        self.assertIn("Panel (a) is RiverCity.", result.content)
        self.assertIn("What the paper's text says about this figure", result.content)
        self.assertEqual(result.sources[0]["type"], "figure")
        self.assertTrue(any(s["type"] == "paper" and s["page"] == 2 for s in result.sources))
        listing = run_tool("list_figures", {}).content
        self.assertIn("discussed on p. 1", listing)

class TestFigureRoutes(FigureTestCase):
    def setUp(self):
        super().setUp()
        app.config["SESSION_ISOLATION"] = False
        self.client = app.test_client()

    def test_listing_image_explain_and_ask(self):
        self.see("**What it is** - a line chart.")
        figures = self.client.get("/api/figures").get_json()["figures"]
        self.assertEqual([f["id"] for f in figures], ["fig1", "fig2"])
        self.assertNotIn("path", figures[0])                               # no server paths leak to the browser
        image = self.client.get(figures[0]["image_url"])
        self.assertEqual(image.status_code, 200)
        self.assertTrue(image.data.startswith(b"\x89PNG"))
        image.close()
        explained = self.client.post("/api/figures/fig1/explain", json={}).get_json()
        self.assertEqual(explained["figure"]["mode"], "vision")
        self.assertIn("line chart", explained["figure"]["analysis"])
        asked = self.client.post("/api/figures/fig1/ask", json={"question": "Final value?"}).get_json()
        self.assertEqual(asked["mode"], "vision")
        self.assertEqual(self.client.post("/api/figures/fig1/ask", json={}).status_code, 400)
        self.assertEqual(self.client.get(f"/papers/{self.paper.id}/figures/nope").status_code, 404)
        self.assertEqual(self.client.get("/papers/unknown/figures/fig1").status_code, 404)

if __name__ == "__main__":
    unittest.main()

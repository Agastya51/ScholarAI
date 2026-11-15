"""Study material (verified claims, flashcards, quiz), the source viewer, and the benchmark helpers."""
import os
import shutil
import tempfile
import unittest

from evals.benchmark import bootstrap_ci, parsing_stats, split_valid
from src import study
from src.app import app
from src.llm import set_backend
from src.rag import library, load_paper
from src.viewer import highlight_phrases, page_view, render_page
from tests.fakes import FakeLLM, hash_encoder, make_paper, schema_name
from tests.pdf_factory import build_two_column_pdf

QUOTE = "FlowNet-X reduces MAE by 12.5% on METR-LA and by 8.1% on PEMS-BAY."

class TestQuoteVerification(unittest.TestCase):
    def setUp(self):
        library.reset()
        self.paper = make_paper()

    def tearDown(self):
        set_backend(None)
        library.reset()

    def test_locate_quote(self):
        self.assertEqual(study.locate_quote(self.paper, QUOTE)[:2], ("exact", 5))
        self.assertEqual(study.locate_quote(self.paper, "flownet-x   reduces mae by 12.5% on metr-la")[0], "exact")
        close = study.locate_quote(self.paper, "FlowNet-X reduces MAE by 12.5% on METR-LA and 8.1% on PEMS-BAY overall")
        self.assertEqual(close[:2], ("close", 5))
        self.assertEqual(study.locate_quote(self.paper, "The model wins every benchmark by a wide margin")[0], "not_found")
        self.assertEqual(study.locate_quote(self.paper, "too short")[0], "not_found")

    def _insights_backend(self):
        def responder(messages, tools, json_mode):
            self.assertEqual(schema_name(messages), "Insights")
            return {"claims": [
                {"claim": "FlowNet-X cuts MAE by 12.5% on METR-LA.", "evidence": QUOTE, "page": 1,
                 "strength": "strong", "why": "Measured against a named baseline."},
                {"claim": "FlowNet-X is 40% cheaper to run.", "evidence": "FlowNet-X is forty percent cheaper to operate.",
                 "page": 3, "strength": "weak", "why": "Asserted only."}],
                "terms": [{"term": "MAE", "definition": "Mean absolute error of the forecast.", "page": 5},
                          {"term": "Ghost", "definition": "x", "page": 99}]}
        return FakeLLM(responder)

    def test_claims_are_checked_against_the_paper(self):
        fake = self._insights_backend()
        set_backend(fake)
        out = study.build_insights(self.paper)
        real, invented = out["claims"]
        self.assertEqual((real["verified"], real["page"]), ("exact", 5))      # the model said page 1; the quote is on 5
        self.assertIsNotNone(real["chunk_id"])
        self.assertEqual(real["unsupported_numbers"], [])
        self.assertEqual(invented["verified"], "not_found")
        self.assertIsNone(invented["page"])                                    # no page reference for a quote that is not there
        self.assertEqual(invented["unsupported_numbers"], ["40"])
        self.assertEqual((out["verified"], out["total"]), (1, 2))
        self.assertIsNone(out["terms"][1]["page"])                             # page 99 does not exist
        study.build_insights(self.paper)
        self.assertEqual(len(fake.calls), 1)                                   # cached

    def test_practice_drops_malformed_quiz_items(self):
        def responder(messages, tools, json_mode):
            return {"flashcards": [{"front": "Learning rate?", "back": "0.001", "page": 2},
                                   {"front": "", "back": "empty front", "page": 2}],
                    "quiz": [{"question": "Batch size?", "options": ["16", "32", "64", "128"], "answer": 2,
                              "explanation": "The batch size is 64.", "page": 2},
                             {"question": "Only three options?", "options": ["a", "b", "c"], "answer": 0,
                              "explanation": "", "page": 2},
                             {"question": "Duplicate options?", "options": ["a", "a", "b", "c"], "answer": 0,
                              "explanation": "", "page": 2}]}
        set_backend(FakeLLM(responder))
        out = study.build_practice(self.paper)
        self.assertEqual(len(out["flashcards"]), 1)
        self.assertEqual([q["question"] for q in out["quiz"]], ["Batch size?"])

    def test_explain_levels_and_questions(self):
        with self.assertRaises(ValueError):
            study.explain_at_level(self.paper, "wizard")
        fake = FakeLLM(lambda m, t, j: {"questions": ["What is FlowNet-X?", "not a question", "How big is the batch?"]}
                       if j else "Plain explanation.")
        set_backend(fake)
        self.assertEqual(study.explain_at_level(self.paper, "simple"), "Plain explanation.")
        self.assertIn("15-year-old", fake.calls[0]["messages"][-1]["content"])
        self.assertEqual(study.suggest_questions(self.paper), ["What is FlowNet-X?", "How big is the batch?"])

    def test_study_routes(self):
        app.config["SESSION_ISOLATION"] = False
        set_backend(self._insights_backend())
        client = app.test_client()
        data = client.post("/api/study/insights", json={}).get_json()
        self.assertEqual(data["verified"], 1)
        self.assertEqual(data["llm_calls"], 1)
        library.reset()
        self.assertEqual(client.post("/api/study/insights", json={}).status_code, 400)
        self.assertIsNone(client.get("/api/paper").get_json()["paper"])

class TestSourceViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.pdf = build_two_column_pdf(os.path.join(cls.tmp, "paper.pdf"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        library.reset()
        self.paper = library.add(load_paper(self.pdf, encoder=hash_encoder), make_active=True)

    def tearDown(self):
        library.reset()

    def test_phrases(self):
        phrases = highlight_phrases("one two three four five six seven eight nine ten eleven twelve thirteen", window=8)
        self.assertEqual(phrases[0], "one two three four five six seven eight")
        self.assertEqual(len(phrases), 2)
        self.assertEqual(highlight_phrases("tiny"), [])

    def test_cited_passage_is_highlighted_on_its_page(self):
        chunk_id, chunk = next((i, c) for i, c in enumerate(self.paper.chunks) if "learning rate is 0.001" in c["text"])
        png, found = page_view(self.paper, chunk["page"], chunk_id=chunk_id)
        self.assertTrue(png.startswith(b"\x89PNG"))
        self.assertGreaterEqual(found, 2)
        plain, none = page_view(self.paper, chunk["page"])
        self.assertEqual(none, 0)
        self.assertNotEqual(png, plain)                      # the highlight changed the pixels

    def test_quote_figure_and_table_views(self):
        _, found = page_view(self.paper, 3, quote="Limitations include the quadratic cost of attention on large road networks")
        self.assertGreaterEqual(found, 1)
        fig_png, _ = page_view(self.paper, 1, figure_id="fig1")
        self.assertNotEqual(fig_png, page_view(self.paper, 1)[0])
        table_chunk = next(i for i, c in enumerate(self.paper.chunks) if c["kind"] == "table")
        self.assertNotEqual(page_view(self.paper, 2, chunk_id=table_chunk)[0], page_view(self.paper, 2)[0])

    def test_original_file_is_untouched_and_bad_pages_are_rejected(self):
        before = open(self.pdf, "rb").read()
        render_page(self.pdf, 1, ["Traffic forecasting on road networks"])
        self.assertEqual(open(self.pdf, "rb").read(), before)
        with self.assertRaises(ValueError):
            render_page(self.pdf, 9)

    def test_page_route(self):
        app.config["SESSION_ISOLATION"] = False
        client = app.test_client()
        chunk_id = next(i for i, c in enumerate(self.paper.chunks) if "learning rate is 0.001" in c["text"])
        res = client.get(f"/papers/{self.paper.id}/pages/1?chunk={chunk_id}")
        self.assertEqual(res.mimetype, "image/png")
        self.assertGreaterEqual(int(res.headers["X-Highlights"]), 2)
        self.assertEqual(client.get(f"/papers/{self.paper.id}/pages/99").status_code, 400)
        self.assertEqual(client.get("/papers/nope/pages/1").status_code, 404)

class TestBenchmarkHelpers(unittest.TestCase):
    def setUp(self):
        library.reset()
        self.paper = make_paper()

    def tearDown(self):
        library.reset()

    def test_questions_with_facts_missing_from_the_paper_are_set_aside(self):
        items = [{"question": "LR?", "expected_facts": ["0.001"]},
                 {"question": "Wrong key?", "expected_facts": ["0.001", "99.99"]},
                 {"question": "Unanswerable?", "expected_facts": [], "answerable": False}]
        valid, invalid = split_valid(self.paper, items)
        self.assertEqual([i["question"] for i in valid], ["LR?", "Unanswerable?"])
        self.assertEqual(invalid[0]["missing_facts"], ["99.99"])

    def test_bootstrap_interval(self):
        ci = bootstrap_ci([1.0] * 18 + [0.0] * 2)
        self.assertEqual(ci, bootstrap_ci([1.0] * 18 + [0.0] * 2))      # seeded: reproducible
        self.assertTrue(0.7 <= ci[0] < 0.9 < ci[1] <= 1.0)
        self.assertEqual(bootstrap_ci([1.0, 1.0]), [1.0, 1.0])
        self.assertIsNone(bootstrap_ci([]))

    def test_parsing_stats(self):
        stats = parsing_stats(self.paper)
        self.assertEqual(stats["chunks"], len(self.paper.chunks))
        self.assertEqual(stats["figures"], 0)

if __name__ == "__main__":
    unittest.main()

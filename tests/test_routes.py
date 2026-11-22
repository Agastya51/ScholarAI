import json
import unittest

from src.app import app
from src.llm import set_backend
from src.rag import library
from tests.fakes import FakeLLM, make_paper, schema_name, tool_call

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config["SESSION_ISOLATION"] = False   # these tests seed the default library directly
        self.client = app.test_client()
        library.reset()

    def tearDown(self):
        set_backend(None)
        library.reset()

    def test_home_page(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"ScholarAI", res.data)
        self.assertIn(b"Drop a PDF here", res.data)

    def test_feature_pages(self):
        for path, marker in [("/qa", b"Ask the paper"), ("/compare", b"Related work"),
                             ("/improve", b"Improve the writing"), ("/download", b"Export your work"),
                             ("/review", b"Peer review"), ("/analyze", b"Run everything"),
                             ("/library", b"Compare several papers"), ("/figures", b"Figures, explained in context"),
                             ("/study", b"Study this paper"), ("/benchmarks", b"Benchmarks")]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, path)
            self.assertIn(marker, res.data, path)

    def test_ask_empty_query(self):
        res = self.client.post("/ask", json={"query": ""})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

    def test_upload_missing_file(self):
        res = self.client.post("/upload")
        self.assertEqual(res.status_code, 400)

    def test_endpoints_require_a_paper(self):
        for path in ["/improve", "/review", "/analyze"]:
            res = self.client.post(path, json={})
            self.assertEqual(res.status_code, 400, path)
        self.assertEqual(self.client.post("/multi-compare", json={}).status_code, 400)

    def test_ask_agent_returns_sources_trace_and_call_count(self):
        make_paper()

        def responder(messages, tools, json_mode):
            if not any(m["role"] == "tool" for m in messages):
                return tool_call("search_paper", query="PEMS-BAY")
            return "MAE drops 8.1% on PEMS-BAY [p. 5]."

        set_backend(FakeLLM(responder))
        data = self.client.post("/ask", json={"query": "PEMS-BAY results?", "history": []}).get_json()
        self.assertEqual(data["mode"], "agent")
        self.assertEqual(data["llm_calls"], 2)
        self.assertTrue(any(s["page"] == 5 for s in data["sources"]))
        self.assertTrue(data["trace"])

    def test_ask_simple_mode(self):
        make_paper()
        set_backend(FakeLLM(lambda m, t, j: "Simple answer"))
        data = self.client.post("/ask", json={"query": "results", "mode": "simple"}).get_json()
        self.assertEqual((data["mode"], data["answer"], data["llm_calls"]), ("simple", "Simple answer", 1))

    def test_llm_failure_maps_to_502(self):
        make_paper()
        from src.llm import LLMError

        def boom(*a, **k):
            raise LLMError("upstream down")

        set_backend(boom)
        res = self.client.post("/ask", json={"query": "x", "mode": "simple"})
        self.assertEqual(res.status_code, 502)
        self.assertIn("upstream down", res.get_json()["error"])

    def test_analyze_streams_server_sent_events(self):
        make_paper()
        route = iter(["reader", "FINISH"])

        def responder(messages, tools, json_mode):
            if schema_name(messages) == "RouteDecision":
                return {"next": next(route), "reason": "test"}
            return "text"

        set_backend(FakeLLM(responder))
        res = self.client.post("/analyze", json={"goal": "summary only"})
        self.assertEqual(res.mimetype, "text/event-stream")
        events = [json.loads(line[6:]) for line in res.get_data(as_text=True).split("\n\n") if line.startswith("data: ")]
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["result"]["completed"], ["reader"])

    def test_papers_library_listing_and_delete(self):
        make_paper("a")
        make_paper("b", make_active=False)
        data = self.client.get("/papers").get_json()
        self.assertEqual([p["id"] for p in data["papers"]], ["a", "b"])
        self.assertEqual(data["active_id"], "a")
        data = self.client.delete("/papers/a").get_json()
        self.assertEqual(data["active_id"], "b")

if __name__ == "__main__":
    unittest.main()


class TestEvalEndpoint(unittest.TestCase):
    """/api/eval/question scores one question with the metric code in evals/."""

    def setUp(self):
        app.config["SESSION_ISOLATION"] = False
        self.client = app.test_client()
        library.reset()

    def tearDown(self):
        set_backend(None)
        library.reset()

    def test_needs_a_paper(self):
        res = self.client.post("/api/eval/question", json={"question": "x"})
        self.assertEqual(res.status_code, 400)

    def test_scores_retrieval_answers_and_judge(self):
        make_paper()

        def responder(messages, tools, json_mode):
            if json_mode and any("grading a question-answering system" in str(m["content"]) for m in messages):
                return {"claims": [{"claim": "MAE drops 8.1%", "supported": True}], "relevance": 5, "correctness": 5}
            if json_mode:
                return {"rewrites": [], "hypothetical_answer": ""}
            return "MAE drops 8.1% on PEMS-BAY [p. 5]."

        set_backend(FakeLLM(responder))
        data = self.client.post("/api/eval/question", json={
            "question": "How much does MAE drop on PEMS-BAY?", "expected_facts": ["8.1%"],
            "modes": ["simple", "naive"], "judge": True}).get_json()
        self.assertTrue(data["answerable"])
        self.assertIn("valid", data)
        self.assertEqual(set(data["retrieval"]), {"bm25", "dense", "hybrid", "hybrid+mmr", "pipeline"})
        for m in data["retrieval"].values():
            self.assertTrue({"hit", "mrr", "precision", "ndcg", "context_recall"} <= set(m))
        simple = data["answers"]["simple"]
        self.assertEqual(simple["fact_recall"], 1.0)
        self.assertEqual(simple["cited"], 1.0)
        self.assertEqual(simple["correctness"], 1.0)
        self.assertEqual(simple["faithfulness"], 1.0)
        self.assertIn("naive", data["answers"])
        self.assertNotIn("correctness", data["answers"]["naive"])

    def test_unanswerable_question_scores_abstention_only(self):
        make_paper()
        set_backend(FakeLLM(lambda m, t, j: {"rewrites": [], "hypothetical_answer": ""} if j
                            else "The paper does not mention ImageNet results."))
        data = self.client.post("/api/eval/question", json={
            "question": "What ImageNet accuracy is reported?", "expected_facts": [], "answerable": False}).get_json()
        self.assertFalse(data["answerable"])
        self.assertNotIn("retrieval", data)
        self.assertEqual(data["answers"]["simple"]["abstained"], 1.0)
        self.assertNotIn("fact_recall", data["answers"]["simple"])

"""Agent behaviour tests with a scripted fake LLM: tool loops, self-correction, evaluator-optimizer, fan-out,
supervisor routing and human-in-the-loop interrupts."""
import unittest
from unittest.mock import patch

from src.config import QA_AGENT_MAX_ITERATIONS, REVIEWERS
from src.llm import set_backend
from src.rag import library
from tests.fakes import FakeLLM, make_paper, schema_name, last_user, tool_call

def _candidates(n, prefix="Paper"):
    return [{"title": f"{prefix} {i}", "summary": "graph neural traffic forecasting " * 3, "has_abstract": True,
             "year": "2024", "link": f"https://x/{prefix}{i}", "venue": "IEEE", "relevance_score": 0.9 - i * 0.1}
            for i in range(n)]

REVIEW = {"score": 6, "confidence": 4, "summary": "Solid.", "strengths": ["Clear method"],
          "weaknesses": [{"issue": "No ablation", "evidence": "", "page": None, "severity": "major"}],
          "questions": []}
META = {"decision": "minor_revision", "overall_score": 6.2, "summary": "Good paper.", "consensus": ["Needs ablation"],
        "disagreements": [], "priority_fixes": ["Add ablation"]}

class AgentTestCase(unittest.TestCase):
    def setUp(self):
        library.reset()
        self.paper = make_paper()

    def tearDown(self):
        set_backend(None)
        library.reset()

class TestQAAgent(AgentTestCase):
    def test_agent_calls_tool_then_answers_with_sources(self):
        from src.agents.qa_agent import answer_question

        def responder(messages, tools, json_mode):
            if not any(m["role"] == "tool" for m in messages):
                return tool_call("search_paper", query="learning rate")
            self.assertIn("0.001", messages[-1]["content"])  # tool output was fed back
            return "The learning rate is 0.001 [p. 2]."

        set_backend(FakeLLM(responder))
        result = answer_question("What learning rate is used?")
        self.assertEqual(result["answer"], "The learning rate is 0.001 [p. 2].")
        self.assertEqual(result["iterations"], 1)
        self.assertIn(2, [s["page"] for s in result["sources"]])
        kinds = [e["kind"] for e in result["trace"]]
        self.assertEqual(kinds, ["tool_call", "tool_result", "output"])

    def test_agent_is_forced_to_answer_after_iteration_cap(self):
        from src.agents.qa_agent import answer_question
        counter = iter(range(100))

        def responder(messages, tools, json_mode):
            if tools:
                return tool_call("search_paper", call_id=f"c{next(counter)}", query="results")
            return "Final answer."

        fake = FakeLLM(responder)
        set_backend(fake)
        result = answer_question("Loop forever?")
        self.assertEqual(result["answer"], "Final answer.")
        self.assertEqual(result["iterations"], QA_AGENT_MAX_ITERATIONS)
        self.assertIsNone(fake.calls[-1]["tools"])

    def test_unknown_tool_error_is_returned_to_model(self):
        from src.agents.qa_agent import answer_question

        def responder(messages, tools, json_mode):
            if not any(m["role"] == "tool" for m in messages):
                return tool_call("delete_everything")
            self.assertIn("Unknown tool", messages[-1]["content"])
            return "Recovered."

        set_backend(FakeLLM(responder))
        result = answer_question("x")
        self.assertEqual(result["answer"], "Recovered.")
        self.assertIn("error", [e["kind"] for e in result["trace"]])

    def test_history_becomes_chat_turns(self):
        from src.agents.qa_agent import history_to_messages
        msgs = history_to_messages([{"q": "a?", "a": "A"}, {"q": "b?", "a": "B"}], max_turns=1)
        self.assertEqual(msgs, [{"role": "user", "content": "b?"}, {"role": "assistant", "content": "B"}])

class TestSelfCorrectingSearch(AgentTestCase):
    def test_reformulates_until_enough_relevant_papers(self):
        from src.workflows import compare_graph
        searches = []

        def fake_search(keywords, summary="", limit=8):
            searches.append(keywords)
            return _candidates(3, prefix=f"Round{len(searches)}")

        grades = iter([{"relevant": [], "reason": "All off-topic"}, {"relevant": [0, 1, 2], "reason": "ok"}])

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "RelevanceGrade":
                return next(grades)
            if name == "Reformulation":
                self.assertIn("All off-topic", last_user(messages))
                return {"keywords": ["spatio-temporal forecasting", "road networks"], "strategy": "broader"}
            return "## Key Differences\n..."

        set_backend(FakeLLM(responder))
        with patch("src.workflows.multi_engine_search", fake_search):
            result = compare_graph.invoke({"summary": "FlowNet-X", "topic": "FlowNet-X, METR-LA", "trace": []})
        self.assertEqual(result["attempt"], 2)
        self.assertEqual(result["queries_tried"], ["FlowNet-X, METR-LA", "spatio-temporal forecasting, road networks"])
        self.assertEqual([p["title"] for p in result["papers"]], ["Round2 0", "Round2 1", "Round2 2"])
        self.assertTrue(result["comparison"].startswith("## Key Differences"))

    def test_gives_up_after_max_attempts(self):
        from src.config import SEARCH_MAX_ATTEMPTS
        from src.workflows import compare_graph

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "RelevanceGrade":
                return {"relevant": [], "reason": "none"}
            if name == "Reformulation":
                return {"keywords": ["other"], "strategy": "x"}
            return "unused"

        set_backend(FakeLLM(responder))
        with patch("src.workflows.multi_engine_search", lambda *a, **k: _candidates(2)):
            result = compare_graph.invoke({"summary": "s", "topic": "t", "trace": []})
        self.assertEqual(result["attempt"], SEARCH_MAX_ATTEMPTS)
        self.assertEqual(result["papers"], [])
        self.assertIn("couldn't retrieve", result["comparison"])

class TestEvaluatorOptimizer(AgentTestCase):
    def test_hallucinated_number_triggers_revision_then_verifies(self):
        from src.workflows import improve_graph
        anchor = "FlowNet-X stacks diffusion convolution layers with a gated temporal attention block."

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "EditSet" and "FACT-CHECK ISSUES" not in last_user(messages):
                return {"edits": [{"section": "Method", "original": anchor,
                                   "rewritten": "FlowNet-X improves accuracy by 99.9% using diffusion convolution."}]}
            if name == "EditSet":
                self.assertIn("99.9", last_user(messages))
                return {"edits": [{"section": "Method", "original": anchor,
                                   "rewritten": "FlowNet-X stacks diffusion convolution layers (learning rate 0.001)."}]}
            if name == "VerdictSet":
                return {"verdicts": [{"index": 0, "verdict": "pass", "faithfulness": 4, "issues": []}]}
            return "## Methodology\nVague."

        set_backend(FakeLLM(responder))
        result = improve_graph.invoke({"summary": "s", "comparison": "", "trace": []})
        self.assertEqual(result["revision"], 1)
        edit = result["edits"][0]
        self.assertEqual(edit["verification"]["status"], "verified")
        self.assertEqual(edit["verification"]["attempts"], 2)
        self.assertIn("0.001", edit["rewritten"])

    def test_unfixable_edit_is_flagged(self):
        from src.config import REWRITE_MAX_REVISIONS
        from src.workflows import improve_graph

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "EditSet":
                return {"edits": [{"section": "Results", "original": "made up", "rewritten": "Gains of 77.7%."}]}
            if name == "VerdictSet":
                return {"verdicts": [{"index": 0, "verdict": "revise", "faithfulness": 1, "issues": ["invented"]}]}
            return "critique"

        set_backend(FakeLLM(responder))
        result = improve_graph.invoke({"summary": "s", "comparison": "", "trace": []})
        self.assertEqual(result["revision"], REWRITE_MAX_REVISIONS)
        self.assertEqual(result["edits"][0]["verification"]["status"], "flagged")

    def test_unsupported_numbers_helper(self):
        from src.workflows import unsupported_numbers, anchor_found
        self.assertEqual(unsupported_numbers("94.3% and 12.5% and 3 layers and 55.5", self.paper.text), ["55.5"])
        self.assertTrue(anchor_found("Compared  with DCRNN,\nFlowNet-X reduces MAE", self.paper.text))
        self.assertFalse(anchor_found("Not in the paper at all", self.paper.text))

class TestReviewPanel(AgentTestCase):
    def test_parallel_reviewers_and_meta_review(self):
        from src.agents.review_panel import review_graph

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "Review":
                return REVIEW
            if name == "MetaReview":
                self.assertEqual(last_user(messages).count("score 6/10"), len(REVIEWERS))
                return META
            raise AssertionError(name)

        set_backend(FakeLLM(responder))
        result = review_graph.invoke({"summary": "s", "comparison": "c", "reviews": [], "trace": []})
        self.assertEqual(sorted(r["reviewer"] for r in result["reviews"]), sorted(REVIEWERS))
        self.assertEqual(result["meta_review"]["decision"], "minor_revision")
        self.assertIn("Decision: Minor Revision", result["review_markdown"])

class TestSupervisor(AgentTestCase):
    def _graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        from src.agents.supervisor import build_analysis_graph
        return build_analysis_graph(MemorySaver())

    def test_routes_with_human_keyword_approval_and_finishes(self):
        from src.agents.supervisor import stream_analysis
        route = iter(["comparator",            # invalid (prerequisite missing) -> falls back to reader
                      "literature_scout", "FINISH"])

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "RouteDecision":
                return {"next": next(route), "reason": "test"}
            if name == "RelevanceGrade":
                return {"relevant": [0, 1, 2], "reason": "ok"}
            if "Extract 3-4" in last_user(messages):
                return "graph neural networks, traffic forecasting"
            return "## TLDR\nFlowNet-X summary"

        set_backend(FakeLLM(responder))
        graph = self._graph()
        seen_topics = []

        def fake_search(keywords, summary="", limit=8):
            seen_topics.append(", ".join(keywords))
            return _candidates(3)

        with patch("src.workflows.multi_engine_search", fake_search):
            events = list(stream_analysis(paper_id=self.paper.id, goal="Find related work", hitl=True, graph=graph))
            interrupt = next(e for e in events if e["type"] == "interrupt")
            self.assertEqual(interrupt["payload"]["type"], "approve_keywords")
            self.assertEqual(interrupt["payload"]["topic"], "graph neural networks, traffic forecasting")
            self.assertFalse(any(e["type"] == "done" for e in events))

            resumed = list(stream_analysis(thread_id=interrupt["thread_id"], resume={"topic": "traffic GNN"},
                                           graph=graph))
        done = next(e for e in resumed if e["type"] == "done")
        self.assertEqual(seen_topics, ["traffic GNN"])
        self.assertEqual(done["result"]["completed"], ["reader", "literature_scout"])
        self.assertEqual(len(done["result"]["papers"]), 3)
        decisions = [e["detail"] for evt in events + resumed if evt["type"] == "step"
                     for e in evt["trace"] if e["agent"] == "supervisor"]
        self.assertTrue(decisions[0].startswith("reader"))
        self.assertIn("not available", decisions[0])
        human = [e for evt in resumed if evt["type"] == "step" for e in evt["trace"] if e["kind"] == "human"]
        self.assertIn("Edited keywords", human[0]["detail"])

    def test_review_panel_fan_out_inside_supervisor(self):
        from src.agents.supervisor import stream_analysis
        route = iter(["reader", "review_panel", "FINISH"])

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "RouteDecision":
                return {"next": next(route), "reason": "test"}
            if name == "Review":
                return REVIEW
            if name == "MetaReview":
                return META
            return "text"

        set_backend(FakeLLM(responder))
        events = list(stream_analysis(paper_id=self.paper.id, goal="Peer review", graph=self._graph()))
        done = next(e for e in events if e["type"] == "done")
        self.assertEqual(len(done["result"]["reviews"]), len(REVIEWERS))
        self.assertEqual(done["result"]["completed"], ["reader", "review_panel"])
        self.assertGreater(done["llm_calls"], 0)

    def test_edit_approval_interrupt_filters_rewrites(self):
        from src.agents.supervisor import stream_analysis
        route = iter(["reader", "editor", "FINISH"])

        def responder(messages, tools, json_mode):
            name = schema_name(messages)
            if name == "RouteDecision":
                return {"next": next(route), "reason": "test"}
            if name == "EditSet":
                return {"edits": [{"section": s, "original": "FlowNet-X achieves 94.3% accuracy",
                                   "rewritten": "FlowNet-X achieves 94.3% accuracy on METR-LA."}
                                  for s in ("Abstract", "Results")]}
            if name == "VerdictSet":
                return {"verdicts": [{"index": i, "verdict": "pass", "faithfulness": 5, "issues": []} for i in (0, 1)]}
            return "text"

        set_backend(FakeLLM(responder))
        graph = self._graph()
        events = list(stream_analysis(paper_id=self.paper.id, goal="Improve writing", hitl=True, graph=graph))
        interrupt = next(e for e in events if e["type"] == "interrupt")
        self.assertEqual(interrupt["payload"]["type"], "review_edits")
        self.assertEqual(len(interrupt["payload"]["edits"]), 2)
        resumed = list(stream_analysis(thread_id=interrupt["thread_id"], resume={"accepted": [1]}, graph=graph))
        done = next(e for e in resumed if e["type"] == "done")
        self.assertEqual([e["section"] for e in done["result"]["edits"]], ["Results"])

class TestMultiPaper(AgentTestCase):
    def test_profiles_each_paper_and_builds_matrix(self):
        from src.agents.multi_paper import compare_papers
        make_paper("p2", title="SparseFlow", make_active=False)

        def responder(messages, tools, json_mode):
            if schema_name(messages) == "PaperProfile":
                title = "SparseFlow" if "SparseFlow" in last_user(messages) else "FlowNet-X"
                return {"title": title, "problem": "traffic forecasting", "method": "GNN", "datasets": ["METR-LA"],
                        "key_results": ["MAE -12.5%"], "baselines": ["DCRNN"], "strengths": [], "limitations": []}
            self.assertIn("SparseFlow", last_user(messages))
            return "## Common Ground\nBoth forecast traffic."

        set_backend(FakeLLM(responder))
        result = compare_papers()
        self.assertEqual([p["title"] for p in result["profiles"]], ["FlowNet-X", "SparseFlow"])
        self.assertIn("| **Datasets** | METR-LA | METR-LA |", result["matrix"])
        self.assertTrue(result["synthesis"].startswith("## Common Ground"))
        # Profiles are cached on the paper: a second run makes only the synthesis call
        fake = FakeLLM(responder)
        set_backend(fake)
        compare_papers()
        self.assertEqual(len(fake.calls), 1)

    def test_needs_two_papers(self):
        from src.agents.multi_paper import compare_papers
        with self.assertRaises(ValueError):
            compare_papers()

if __name__ == "__main__":
    unittest.main()

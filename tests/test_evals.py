"""Evaluation harness: metrics, retrieval ablation, answer evaluation, dataset generation (fake LLM, no network)."""
import unittest
from unittest.mock import patch

from evals import metrics as M
from evals.generate_dataset import GeneratedQA, pick_chunks, validate
from evals.retrieval_eval import evaluate_retrieval, score_calibration
from evals.run_eval import evaluate_answers, render
from src import retrieval
from src.llm import set_backend
from src.rag import library
from tests.fakes import FakeLLM, make_paper, schema_name, tool_call

ITEMS = [
    {"question": "What learning rate and batch size are used?", "expected_facts": ["0.001", "64"]},
    {"question": "By how much does FlowNet-X reduce MAE on PEMS-BAY?", "expected_facts": ["8.1%"]},
    {"question": "Which ImageNet pretraining checkpoint initialises the encoder?", "expected_facts": [], "answerable": False},
]

class TestMetrics(unittest.TestCase):
    def test_relevance_by_fact_or_page(self):
        item = {"expected_facts": ["4,000"]}
        self.assertTrue(M.chunk_is_relevant({"text": "warmup of 4000 steps"}, item))       # thousands separator
        self.assertFalse(M.chunk_is_relevant({"text": "warmup of 400 steps"}, item))
        self.assertTrue(M.chunk_is_relevant({"text": "x", "page": 3}, {"gold_pages": [3]}))

    def test_ranking_metrics(self):
        flags = [False, True, False, True]
        self.assertEqual(M.hit(flags), 1.0)
        self.assertEqual(M.reciprocal_rank(flags), 0.5)
        self.assertEqual(M.precision(flags), 0.5)
        self.assertEqual(M.ndcg([True, True]), 1.0)
        self.assertLess(M.ndcg(flags), 1.0)
        self.assertEqual(M.reciprocal_rank([False, False]), 0.0)
        self.assertEqual(M.ndcg([]), 0.0)

    def test_context_and_fact_recall(self):
        item = {"expected_facts": ["Adam", "4000"]}
        self.assertEqual(M.context_recall([{"text": "We use Adam."}, {"text": "nothing"}], item), 0.5)
        self.assertEqual(M.fact_recall("Adam with 4,000 warmup steps", item), 1.0)

    def test_answer_signals(self):
        self.assertTrue(M.abstained("The paper does not appear to cover this."))
        self.assertTrue(M.abstained("This is not mentioned in the provided context."))
        self.assertFalse(M.abstained("The learning rate is 0.001 [p. 2]."))
        self.assertEqual(M.cited_pages("See [p. 2] and [p. 5 · Results]."), [2, 5])
        self.assertEqual(M.citation_precision("See [p. 2] and [p. 9].", [2, 3]), 0.5)
        self.assertEqual(M.hallucinated_numbers("Accuracy is 99.9% [p. 12].", "accuracy 94.3%"), ["99.9"])

    def test_average_and_table(self):
        avg = M.average([{"a": 1.0, "b": 0.0}, {"a": 0.0}])
        self.assertEqual(avg, {"a": 0.5, "b": 0.0})
        self.assertIn("| x | 0.500 |", M.markdown_table(["name", "v"], [["x", 0.5]]))

class EvalTestCase(unittest.TestCase):
    def setUp(self):
        library.reset()
        self.paper = make_paper()
        self.patches = [patch.object(retrieval, "HIGH_CONFIDENCE", 0.3), patch.object(retrieval, "MIN_RELEVANCE", 0.15)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        set_backend(None)
        library.reset()

class TestRetrievalEval(EvalTestCase):
    def test_ablation_table_covers_answerable_questions_only(self):
        result = evaluate_retrieval(self.paper, ITEMS, ["bm25", "dense", "hybrid", "hybrid+mmr", "pipeline"], k=2)
        self.assertEqual(result["n_questions"], 2)
        self.assertEqual(set(result["summary"]), {"bm25", "dense", "hybrid", "hybrid+mmr", "pipeline"})
        self.assertEqual(result["summary"]["bm25"]["hit"], 1.0)
        self.assertEqual(result["summary"]["hybrid"]["context_recall"], 1.0)
        self.assertEqual(len(result["rows"]), 10)

    def test_calibration_separates_relevant_from_unanswerable(self):
        cal = score_calibration(self.paper, ITEMS)
        self.assertEqual(cal["relevant"]["n"], 2)
        self.assertEqual(cal["unanswerable"]["n"], 1)
        self.assertGreater(cal["relevant"]["median"], cal["unanswerable"]["median"])
        self.assertIn("MIN_RELEVANCE", cal["suggested"])

class TestAnswerEval(EvalTestCase):
    def _responder(self, messages, tools, json_mode):
        name = schema_name(messages) if json_mode else ""
        if name == "QueryExpansion":
            return {"rewrites": [], "hypothetical_answer": ""}
        if name == "AnswerJudgement":
            return {"claims": [{"claim": "c1", "supported": True}, {"claim": "c2", "supported": False}],
                    "relevance": 5, "correctness": 3}
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        question = user.rsplit("Question:", 1)[-1] if "Question:" in user else user
        if tools:   # agent mode
            if not any(m["role"] == "tool" for m in messages):
                return tool_call("search_paper", query=question[:60])
        if "ImageNet" in question:
            return "The paper does not appear to cover this."
        if "PEMS-BAY" in question:
            return "MAE drops by 8.1% on PEMS-BAY [p. 5]."
        return "The learning rate is 0.001 and the batch size is 64 [p. 2]."

    def test_all_modes_produce_comparable_rows(self):
        set_backend(FakeLLM(self._responder))
        result = evaluate_answers(self.paper, ITEMS, ["naive", "simple", "agent"], verbose=False)
        self.assertEqual(len(result["rows"]), 9)
        for mode in ("naive", "simple", "agent"):
            s = result["summary"][mode]
            self.assertEqual(s["fact_recall"], 1.0, mode)
            self.assertEqual(s["abstain_ok"], 1.0, mode)
            self.assertEqual(s["false_abstain"], 0.0, mode)
            self.assertEqual(s["halluc_numbers"], 0.0, mode)
            self.assertEqual(s["errors"], 0)
        self.assertEqual(result["summary"]["naive"]["llm_calls"], 1.0)
        self.assertEqual(result["summary"]["agent"]["llm_calls"], 2.0)
        self.assertEqual(result["summary"]["agent"]["context_recall"], 1.0)   # agent evidence is recovered for scoring
        report = render(result, self.paper, "d.json", ITEMS)
        self.assertIn("| naive | 3 |", report)
        self.assertIn("2 answerable + 1 unanswerable", report)

    def test_judge_adds_faithfulness_relevance_correctness(self):
        set_backend(FakeLLM(self._responder))
        result = evaluate_answers(self.paper, ITEMS[:1], ["simple"], use_judge=True, verbose=False)
        row = result["rows"][0]
        self.assertEqual(row["faithfulness"], 0.5)
        self.assertEqual(row["relevance"], 1.0)
        self.assertEqual(row["correctness"], 0.5)
        self.assertEqual(row["unsupported_claims"], ["c2"])

    def test_llm_failure_is_recorded_not_raised(self):
        from src.llm import LLMRateLimitError

        def boom(messages, tools=None, json_mode=False, max_tokens=0):
            raise LLMRateLimitError("rate limited")

        set_backend(boom)
        result = evaluate_answers(self.paper, ITEMS[:1], ["naive"], verbose=False)
        self.assertIn("error", result["rows"][0])
        self.assertEqual(result["summary"]["naive"]["errors"], 1)

class TestDatasetGeneration(EvalTestCase):
    PASSAGE = "FlowNet-X is trained with Adam using a learning rate of 0.001 and a batch size of 64."

    def test_validation_requires_literal_non_leaked_facts(self):
        ok = validate(GeneratedQA(question="Which optimizer is used to train FlowNet-X?", expected_facts=["Adam"]), self.PASSAGE)
        self.assertEqual(ok["expected_facts"], ["Adam"])
        invented = GeneratedQA(question="Which optimizer is used to train FlowNet-X?", expected_facts=["SGD"])
        self.assertIsNone(validate(invented, self.PASSAGE))
        leaked = GeneratedQA(question="Is Adam the optimizer used to train FlowNet-X?", expected_facts=["Adam"])
        self.assertIsNone(validate(leaked, self.PASSAGE))
        self.assertIsNone(validate(GeneratedQA(question="Optimizer", expected_facts=["Adam"]), self.PASSAGE))

    def test_pick_chunks_spreads_over_sections_and_skips_short_ones(self):
        long = {"page": 1, "section": "Method", "text": "The learning rate is 0.001 and batch size 64. " * 6}
        long2 = {"page": 2, "section": "Results", "text": "MAE drops 12.5% and 8.1% versus baselines. " * 6}
        library.reset()
        paper = make_paper(segments=[long, long2, {"page": 3, "section": "Tiny", "text": "Short text only here, nothing else."}])
        picked = pick_chunks(paper, 2)
        self.assertEqual({c["section"] for c in picked}, {"Method", "Results"})

if __name__ == "__main__":
    unittest.main()

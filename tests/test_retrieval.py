"""Retrieval pipeline: cosine scores, MMR, thresholds/confidence, query expansion, reranking, grounding, cache."""
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src import retrieval
from src.llm import set_backend
from src.rag import VectorStore, chunk_segments, embed_with_cache, excluded_kinds_for, library
from src.retrieval import (
    check_grounding, confidence_label, content_terms, keyword_coverage, retrieve_context, set_reranker
)
from tests.fakes import FakeLLM, SAMPLE_SEGMENTS, hash_encoder, make_paper, schema_name

class TestVectorStoreScores(unittest.TestCase):
    def setUp(self):
        self.store = VectorStore(encoder=hash_encoder)
        self.store.add_chunks(chunk_segments(SAMPLE_SEGMENTS))

    def test_scores_are_cosine_similarities(self):
        top = self.store.search("learning rate batch size", k=2)[0]
        self.assertEqual(top["page"], 2)
        self.assertGreater(top["dense_score"], 0.3)
        self.assertLessEqual(top["dense_score"], 1.0001)
        self.assertGreater(top["bm25_score"], 0)
        unrelated = self.store.search("stepsize minibatch", k=1, mode="dense")[0]
        self.assertLess(unrelated["dense_score"], 0.05)

    def test_section_header_is_part_of_the_index(self):
        # "conclusion" appears only in the section name, not in the chunk text
        self.assertNotIn("conclusion", SAMPLE_SEGMENTS[3]["text"].lower())
        self.assertEqual(self.store.search("conclusion", k=1, mode="sparse")[0]["page"], 6)

    def test_mmr_prefers_a_new_passage_over_a_duplicate(self):
        store = VectorStore(encoder=hash_encoder)
        store.add_chunks(["the learning rate is 0.001", "the learning rate is 0.001",
                          "the learning rate schedule halves every 20 epochs", "dropout is 0.3"])
        relevance = {0: 1.0, 1: 0.99, 2: 0.6, 3: 0.1}
        self.assertEqual(store.mmr([0, 1, 2, 3], relevance, k=2, lam=1.0), [0, 1])   # pure relevance
        self.assertEqual(store.mmr([0, 1, 2, 3], relevance, k=2, lam=0.3), [0, 2])   # diversity kicks in

    def test_references_only_searched_when_asked(self):
        self.assertEqual(excluded_kinds_for("what is the learning rate"), {"reference"})
        self.assertEqual(excluded_kinds_for("what does reference [12] say"), set())
        self.assertEqual(excluded_kinds_for("which papers are cited for attention"), set())

class TestConfidence(unittest.TestCase):
    def test_labels(self):
        with patch.object(retrieval, "HIGH_CONFIDENCE", 0.6), patch.object(retrieval, "MIN_RELEVANCE", 0.4):
            self.assertEqual(confidence_label(0.7, 0.0), "high")
            self.assertEqual(confidence_label(0.5, 0.0), "medium")
            self.assertEqual(confidence_label(0.1, 0.8), "medium")   # exact keyword match rescues a weak embedding
            self.assertEqual(confidence_label(0.1, 0.2), "low")

    def test_keyword_coverage_ignores_stopwords(self):
        self.assertEqual(content_terms("What is the learning rate of the model?"), ["learning", "rate", "model"])
        self.assertAlmostEqual(keyword_coverage(["what is the learning rate"], "The learning rate is 0.001"), 1.0)
        self.assertEqual(keyword_coverage(["quantum chromodynamics"], "The learning rate is 0.001"), 0.0)

class TestRetrievalPipeline(unittest.TestCase):
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
        set_reranker(None)
        library.reset()

    def test_confident_question_needs_no_llm_call(self):
        fake = FakeLLM(lambda *a: self.fail("no LLM call expected"))
        set_backend(fake)
        result = retrieve_context(self.paper, "learning rate and batch size", k=2, expansion="auto")
        self.assertEqual(result.confidence, "high")
        self.assertFalse(result.expanded)
        self.assertEqual(result.chunks[0]["page"], 2)
        self.assertEqual(fake.calls, [])

    def test_weak_evidence_triggers_one_corrective_expansion(self):
        def responder(messages, tools, json_mode):
            self.assertEqual(schema_name(messages), "QueryExpansion")
            return {"rewrites": ["learning rate batch size"], "hypothetical_answer": "The learning rate is set and the batch size is fixed."}

        fake = FakeLLM(responder)
        set_backend(fake)
        result = retrieve_context(self.paper, "stepsize minibatch", k=2, expansion="auto")
        self.assertEqual(len(fake.calls), 1)
        self.assertTrue(result.expanded)
        self.assertIn("learning rate batch size", result.queries)
        self.assertEqual(result.chunks[0]["page"], 2)
        self.assertIn(result.confidence, ("high", "medium"))
        self.assertTrue(any("query expansion" in s for s in result.steps))

    def test_expansion_off_never_calls_the_llm_and_reports_low_confidence(self):
        fake = FakeLLM(lambda *a: self.fail("no LLM call expected"))
        set_backend(fake)
        result = retrieve_context(self.paper, "zebra xylophone quasar", k=3, expansion="off")
        self.assertEqual(result.confidence, "low")
        self.assertLessEqual(len(result.chunks), 2)   # nothing passes the threshold: only the fallback pair is kept
        self.assertGreaterEqual(len(result.chunks), 1)

    def test_expansion_failure_degrades_gracefully(self):
        def broken(messages, tools, json_mode):
            return "this is not json"

        set_backend(FakeLLM(broken))
        result = retrieve_context(self.paper, "zebra xylophone quasar", k=2, expansion="auto")
        self.assertEqual(result.confidence, "low")
        self.assertFalse(result.expanded)

    def test_custom_reranker_reorders_candidates(self):
        # A reranker that loves the conclusion chunk
        set_reranker(lambda q, passages: [1.0 if "Limitations" in p else 0.0 for p in passages])
        result = retrieve_context(self.paper, "learning rate and batch size", k=1, expansion="off", mmr=False,
                                  threshold=False)
        self.assertEqual(result.chunks[0]["page"], 6)
        self.assertEqual(result.reranker, "custom")
        self.assertIn("rerank_score", result.chunks[0])

    def test_llm_reranker(self):
        def responder(messages, tools, json_mode):
            self.assertEqual(schema_name(messages), "Ranking")
            user = messages[-1]["content"]
            wanted = next(int(line[1]) for line in user.splitlines() if line.startswith("[") and "DCRNN" in line)
            return {"ranking": [wanted]}

        fake = FakeLLM(responder)
        set_backend(fake)
        result = retrieve_context(self.paper, "learning rate and batch size", k=1, expansion="off", rerank="llm",
                                  mmr=False, threshold=False)
        self.assertEqual(result.chunks[0]["page"], 5)
        self.assertEqual(result.reranker, "llm")
        self.assertEqual(len(fake.calls), 1)

    def test_adaptive_k_widens_when_not_confident(self):
        with patch.object(retrieval, "HIGH_CONFIDENCE", 0.99):
            wide = retrieve_context(self.paper, "FlowNet-X METR-LA", k=1, expansion="off", threshold=False)
        narrow = retrieve_context(self.paper, "FlowNet-X METR-LA", k=1, expansion="off", threshold=False)
        self.assertEqual(len(narrow.chunks), 1)
        self.assertGreater(len(wide.chunks), 1)

    def test_empty_inputs(self):
        self.assertEqual(retrieve_context(self.paper, "   ").confidence, "none")

class TestGrounding(unittest.TestCase):
    CHUNKS = [{"text": "FlowNet-X reduces MAE by 12.5% on METR-LA.", "page": 5}]

    def test_grounded_answer(self):
        report = check_grounding("MAE drops by 12.5% [p. 5].", self.CHUNKS)
        self.assertTrue(report["grounded"])
        self.assertTrue(report["cited"])

    def test_invented_number_and_wrong_page(self):
        report = check_grounding("MAE drops by 19.7% [p. 9].", self.CHUNKS)
        self.assertEqual(report["unsupported_numbers"], ["19.7"])
        self.assertEqual(report["unsupported_citations"], [9])
        self.assertFalse(report["grounded"])

    def test_page_numbers_in_citations_are_not_claims(self):
        self.assertEqual(check_grounding("See [p. 5] and [p. 5 · 5. Results].", self.CHUNKS)["unsupported_numbers"], [])

class TestEmbeddingCache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.calls = 0

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def encoder(self, texts):
        self.calls += 1
        return hash_encoder(texts)

    def test_second_embedding_of_same_content_is_free(self):
        texts = ["alpha beta", "gamma delta"]
        first = embed_with_cache(texts, self.encoder, "model-a", self.dir)
        second = embed_with_cache(texts, self.encoder, "model-a", self.dir)
        self.assertEqual(self.calls, 1)
        np.testing.assert_allclose(first, second)

    def test_cache_is_keyed_by_content_and_model(self):
        embed_with_cache(["alpha beta"], self.encoder, "model-a", self.dir)
        embed_with_cache(["alpha beta changed"], self.encoder, "model-a", self.dir)
        embed_with_cache(["alpha beta"], self.encoder, "model-b", self.dir)
        self.assertEqual(self.calls, 3)
        self.assertEqual(len(os.listdir(self.dir)), 3)

    def test_cache_is_pruned(self):
        with patch("src.rag.EMBED_CACHE_MAX_FILES", 2):
            from src import rag
            for i in range(4):
                embed_with_cache([f"text {i}"], self.encoder, "m", self.dir)
                rag._prune_cache(self.dir, keep=2)
        self.assertEqual(len(os.listdir(self.dir)), 2)

    def test_no_cache_dir_means_no_files(self):
        embed_with_cache(["alpha"], self.encoder, "m", None)
        self.assertEqual(os.listdir(self.dir), [])

class TestProductionEmbeddingPath(unittest.TestCase):
    """load_paper() with the default (FastEmbed) encoder path, using a stand-in model object."""
    class FakeFastEmbed:
        def __init__(self):
            self.seen = []

        def embed(self, texts, batch_size=16):
            self.seen.append(list(texts))
            yield from hash_encoder(list(texts))

    def setUp(self):
        from src import rag
        self.rag = rag
        self.tmp = tempfile.mkdtemp()
        self.model = self.FakeFastEmbed()
        self.saved = (rag._embed_model, rag._is_fastembed)
        rag._embed_model, rag._is_fastembed = self.model, True

    def tearDown(self):
        self.rag._embed_model, self.rag._is_fastembed = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reupload_hits_cache_and_queries_get_the_instruction(self):
        from tests.pdf_factory import build_two_column_pdf
        pdf = build_two_column_pdf(os.path.join(self.tmp, "p.pdf"))
        cache = os.path.join(self.tmp, "cache")
        first = self.rag.load_paper(pdf, cache_dir=cache)
        self.assertFalse(first.cache_hit)
        passages_embedded = len(self.model.seen)
        second = self.rag.load_paper(pdf, cache_dir=cache)
        self.assertTrue(second.cache_hit)
        self.assertEqual(len(self.model.seen), passages_embedded)          # no new embedding work
        self.assertEqual(len(second.chunks), len(first.chunks))

        hits = second.search("learning rate batch size", k=1)
        self.assertIn("0.001", hits[0]["text"])
        query_text = self.model.seen[-1][0]
        self.assertTrue(query_text.startswith("Represent this sentence for searching relevant passages: "))
        self.assertTrue(all(not t.startswith("Represent this sentence") for batch in self.model.seen[:passages_embedded]
                            for t in batch))                               # passages are embedded without it
        for p in (first, second):
            shutil.rmtree(os.path.join("uploads", "figures", p.id), ignore_errors=True)

if __name__ == "__main__":
    unittest.main()

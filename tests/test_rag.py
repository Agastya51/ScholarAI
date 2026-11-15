import unittest
from src.rag import chunk_text, chunk_segments, detect_heading, format_chunks, BM25, VectorStore
from tests.fakes import hash_encoder, SAMPLE_SEGMENTS

class TestChunking(unittest.TestCase):
    def test_chunk_text_hard_splits_without_sentence_boundaries(self):
        chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", size=10, overlap=0)
        self.assertEqual(chunks, ["abcdefghij", "klmnopqrst", "uvwxyz"])

    def test_chunk_text_empty(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text(None), [])

    def test_chunk_text_keeps_sentences_whole_and_overlaps(self):
        sentences = [f"Sentence number {i} talks about topic {i}." for i in range(20)]
        chunks = chunk_text(" ".join(sentences), size=200, overlap=60)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 200)
            self.assertTrue(c.endswith("."), c)
        # The last sentence of a chunk is carried into the next one
        for a, b in zip(chunks, chunks[1:]):
            self.assertIn(a.split(". ")[-1].rstrip("."), b)

    def test_chunk_segments_keep_page_and_section(self):
        chunks = chunk_segments(SAMPLE_SEGMENTS)
        self.assertEqual({c["page"] for c in chunks}, {1, 2, 5, 6})
        self.assertIn("3. Method", {c["section"] for c in chunks})
        self.assertIn("[p. 2 · 3. Method]", format_chunks([c for c in chunks if c["page"] == 2]))

class TestHeadings(unittest.TestCase):
    def test_detects_numbered_and_known_headings(self):
        for line in ["1 Introduction", "3.2 Training Details", "IV. EXPERIMENTS", "Abstract", "References",
                     "Related Work"]:
            self.assertIsNotNone(detect_heading(line), line)

    def test_rejects_body_text(self):
        for line in ["We train for 100 epochs with a batch size of 64.", "2 3 4 5", "",
                     "1 The model achieves 94% accuracy on the test set which is very long text indeed."]:
            self.assertIsNone(detect_heading(line), line)

class TestRetrieval(unittest.TestCase):
    def test_bm25_matches_exact_terms(self):
        bm = BM25()
        bm.add(["learning rate 0.001 batch size 64", "results on METR-LA dataset", "conclusion"])
        scores = bm.scores("METR-LA")
        self.assertEqual(int(scores.argmax()), 1)

    def test_hybrid_search_returns_metadata(self):
        store = VectorStore(encoder=hash_encoder)
        store.add_chunks(chunk_segments(SAMPLE_SEGMENTS))
        top = store.search("learning rate batch size", k=1)[0]
        self.assertEqual(top["page"], 2)
        self.assertIn("0.001", top["text"])
        self.assertEqual(store.search("PEMS-BAY", k=1, mode="sparse")[0]["page"], 5)
        self.assertEqual(store.retrieve("", k=2), [])

    def test_vector_store_with_real_embedder(self):
        try:
            from src.rag import get_embedder
            get_embedder()
        except RuntimeError as e:
            self.skipTest(f"embedding model not available offline: {str(e)[:60]}")
        store = VectorStore()
        docs = [
            "Convolutional Neural Networks are designed for computer vision and image processing.",
            "Transformer architectures rely on self-attention mechanisms for language modeling.",
            "Reinforcement learning optimizes decision-making policies through reward signals."
        ]
        store.add_chunks(docs)
        self.assertEqual(len(store.documents), 3)
        self.assertIn("Convolutional", store.retrieve("computer vision images", k=1)[0])
        self.assertIn("Transformer", store.retrieve("self-attention transformers", k=1)[0])
        store.clear()
        self.assertEqual(store.retrieve("anything"), [])

if __name__ == "__main__":
    unittest.main()

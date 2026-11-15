"""Per-browser session isolation, eviction, file cleanup and upload safety."""
import io

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import os
import time
from types import SimpleNamespace
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src import rag
from src.app import app, SESSION_COOKIE
from src.llm import set_backend, set_vision_backend
from src.rag import (
    DEFAULT_SESSION, Paper, PaperLibrary, SessionManager, dispose_paper, library, load_paper, sessions, use_session,
)
from tests.fakes import FakeLLM, hash_encoder, make_paper
from tests.pdf_factory import build_two_column_pdf

def _offline_load_paper(path, filename="", paper_id=None, progress=None):
    return load_paper(path, filename, paper_id, encoder=hash_encoder, progress=progress)

class TestSessionManager(unittest.TestCase):
    def test_each_session_has_its_own_library(self):
        mgr = SessionManager(max_sessions=5, ttl=3600)
        a, b = mgr.get("a"), mgr.get("b")
        a.add(Paper(id="p1", filename="x.pdf", path="x.pdf"))
        self.assertEqual(len(a.papers), 1)
        self.assertEqual(len(b.papers), 0)
        self.assertIs(mgr.get("a"), a)

    @staticmethod
    def _with_paper(mgr, sid):
        mgr.get(sid).add(Paper(id=f"paper-{sid}", filename="x.pdf", path="x.pdf"))

    def test_least_recently_used_session_with_papers_is_evicted(self):
        mgr = SessionManager(max_sessions=2, ttl=3600)
        for sid in ("a", "b"):
            self._with_paper(mgr, sid)
        mgr.get("a")
        self._with_paper(mgr, "c")
        mgr.get("c")
        self.assertEqual({sid for sid, lib in mgr._libraries.items() if lib.papers}, {"a", "c"})

    def test_empty_sessions_never_push_out_a_session_that_holds_a_paper(self):
        mgr = SessionManager(max_sessions=2, ttl=3600)
        self._with_paper(mgr, "real-user")
        for i in range(200):                      # e.g. crawlers or cookie-less clients
            mgr.get(f"visitor-{i}")
        self.assertIn("real-user", mgr._libraries)
        self.assertEqual(len(mgr.get("real-user").papers), 1)
        self.assertLess(len(mgr._libraries), 80)  # ...and empty sessions do not pile up without bound
        self.assertEqual(mgr.count(), 1)

    def test_idle_sessions_expire(self):
        mgr = SessionManager(max_sessions=10, ttl=60)
        old = mgr.get("old")
        mgr.get("fresh")
        old.touched -= 3600
        mgr.get("fresh")
        self.assertNotIn("old", mgr._libraries)

    def test_default_session_is_never_evicted(self):
        mgr = SessionManager(max_sessions=1, ttl=0)
        mgr.get(DEFAULT_SESSION).add(Paper(id="cli", filename="x.pdf", path="x.pdf"))
        mgr.get(DEFAULT_SESSION).touched -= 10
        self._with_paper(mgr, "a"); self._with_paper(mgr, "b")
        self.assertIn(DEFAULT_SESSION, mgr._libraries)
        self.assertEqual(len(mgr.get(DEFAULT_SESSION).papers), 1)

    def test_library_proxy_follows_the_bound_session(self):
        with use_session("proxy-test-a"):
            library.add(Paper(id="pa", filename="a.pdf", path="a.pdf"))
            self.assertEqual(list(library.papers), ["pa"])
        with use_session("proxy-test-b"):
            self.assertEqual(list(library.papers), [])
        sessions.drop("proxy-test-a"); sessions.drop("proxy-test-b")

class TestSessionInParallelBranches(unittest.TestCase):
    """LangGraph runs fan-out branches on worker threads; they must still see the caller's library."""
    def tearDown(self):
        set_backend(None)
        sessions.drop("fanout-test")

    def test_review_panel_fan_out_finds_the_session_paper(self):
        from src.agents.review_panel import review_graph
        from tests.fakes import schema_name

        from tests.test_agents import REVIEW, META

        seen = []   # (session bound on the worker thread, did the reviewer prompt contain paper text?)

        def responder(messages, tools, json_mode):
            prompt = messages[-1]["content"]
            if schema_name(messages) == "MetaReview":
                return META
            seen.append((rag.current_session_id(), "FlowNet-X" in prompt))
            return REVIEW

        set_backend(FakeLLM(responder))
        with use_session("fanout-test"):
            paper = make_paper(paper_id="fan1")
            result = review_graph.invoke({"paper_id": paper.id, "summary": "s", "comparison": "", "reviews": [],
                                          "trace": []})
        self.assertEqual(len(result["reviews"]), 4)
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(sid == "fanout-test" and has_paper_text for sid, has_paper_text in seen), seen)

class TestFileCleanup(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.uploads = os.path.join(self.root, "uploads")
        os.makedirs(os.path.join(self.uploads, "figures", "p1"))
        self.patches = [patch.object(rag, "UPLOAD_DIR", self.uploads),
                        patch.object(rag, "FIGURE_DIR", os.path.join(self.uploads, "figures"))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.root, ignore_errors=True)

    def _touch(self, path):
        with open(path, "w") as f:
            f.write("x")
        return path

    def test_dispose_removes_uploaded_pdf_and_figures(self):
        pdf = self._touch(os.path.join(self.uploads, "p1.pdf"))
        dispose_paper(Paper(id="p1", filename="a.pdf", path=pdf))
        self.assertFalse(os.path.exists(pdf))
        self.assertFalse(os.path.exists(os.path.join(self.uploads, "figures", "p1")))

    def test_dispose_never_deletes_a_file_outside_the_upload_dir(self):
        mine = self._touch(os.path.join(self.root, "my_thesis.pdf"))   # e.g. a PDF passed to the eval scripts
        dispose_paper(Paper(id="p1", filename="my_thesis.pdf", path=mine))
        self.assertTrue(os.path.exists(mine))

    def test_library_cap_evicts_oldest_inactive_paper_and_its_file(self):
        lib = PaperLibrary(max_papers=2)
        paths = [self._touch(os.path.join(self.uploads, f"p{i}.pdf")) for i in range(3)]
        for i, path in enumerate(paths):
            lib.add(Paper(id=f"p{i}", filename=f"{i}.pdf", path=path), make_active=(i == 0))
        self.assertEqual(list(lib.papers), ["p0", "p2"])     # p0 is active, p2 is the newest; p1 was dropped
        self.assertFalse(os.path.exists(paths[1]))
        self.assertTrue(os.path.exists(paths[0]))

class TestAppSessions(unittest.TestCase):
    def setUp(self):
        app.config["SESSION_ISOLATION"] = True
        app.config["INLINE_JOBS"] = False      # real background threads, polled like the browser does
        self.tmp = tempfile.mkdtemp()
        self.pdf = build_two_column_pdf(os.path.join(self.tmp, "paper.pdf"))

    def tearDown(self):
        app.config["SESSION_ISOLATION"] = False
        set_backend(None)
        set_vision_backend(None)
        shutil.rmtree(self.tmp, ignore_errors=True)
        for sid in [s for s in list(sessions._libraries) if s != DEFAULT_SESSION]:
            sessions.drop(sid)
        library.reset()

    def _upload(self, client, route="/papers", pdf=None):
        """Upload, then follow the background job the way the browser does. Returns the finished job's
        result wrapped like a response (status_code, get_json)."""
        with open(pdf or self.pdf, "rb") as f:
            res = client.post(route, data={"file": (io.BytesIO(f.read()), "paper.pdf")},
                              content_type="multipart/form-data")
        if res.status_code != 202:
            return res
        job_id = res.get_json()["job_id"]
        for _ in range(200):
            job = client.get(f"/api/jobs/{job_id}").get_json()
            if job["status"] != "running":
                break
            time.sleep(0.05)
        self.last_job = job
        result = job.get("result") if job["status"] == "done" else {"error": job.get("error")}
        return SimpleNamespace(status_code=200 if job["status"] == "done" else 400, get_json=lambda: result)

    def test_cookie_is_issued_once_and_is_http_only(self):
        client = app.test_client()
        first = client.get("/papers")
        cookie = first.headers.get("Set-Cookie", "")
        self.assertIn(SESSION_COOKIE, cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertNotIn(SESSION_COOKIE, client.get("/papers").headers.get("Set-Cookie", ""))

    def test_health_probe_creates_no_session(self):
        before = len(sessions._libraries)
        for _ in range(5):
            res = app.test_client().get("/health")
            self.assertEqual(res.status_code, 200)
            self.assertNotIn(SESSION_COOKIE, res.headers.get("Set-Cookie", ""))
        self.assertEqual(len(sessions._libraries), before)

    def test_two_browsers_do_not_see_each_others_papers(self):
        alice, bob = app.test_client(), app.test_client()
        with patch("src.app.load_paper", _offline_load_paper):
            res = self._upload(alice)
        self.assertEqual(res.status_code, 200, res.get_json())
        self.assertEqual(len(alice.get("/papers").get_json()["papers"]), 1)
        self.assertEqual(bob.get("/papers").get_json()["papers"], [])
        # Bob has no paper, so paper-dependent endpoints refuse instead of using Alice's
        self.assertEqual(bob.post("/improve", json={}).status_code, 400)
        self.assertEqual(alice.get("/health").get_json()["active_sessions"], 1)

    def test_default_library_is_not_reachable_from_a_browser_session(self):
        make_paper()   # lands in the default (CLI) library
        self.assertEqual(app.test_client().get("/papers").get_json()["papers"], [])

    def test_full_upload_flow_returns_stats(self):
        fake = FakeLLM(lambda messages, tools, json_mode: "graph neural networks, traffic forecasting"
                       if "Extract 3-4" in messages[-1]["content"] else "## TLDR\nsummary")
        set_backend(fake)
        seen = []
        set_vision_backend(lambda path, prompt, max_tokens: seen.append(prompt) or "**What it is** - a diagram.")
        client = app.test_client()
        with patch("src.app.load_paper", _offline_load_paper):
            data = self._upload(client, "/upload").get_json()
        # Step 1 only parses and indexes: no model calls at all
        self.assertEqual((fake.calls, seen, data["llm_calls"]), ([], [], 0))
        self.assertEqual(data["stats"]["pages"], 3)
        self.assertEqual(data["stats"]["tables"], 1)
        self.assertEqual(data["stats"]["references"], 3)
        self.assertEqual(data["stats"]["figures"], 2)
        self.assertEqual([f["id"] for f in data["figures"]], ["fig1", "fig2"])
        # Step 2 writes the summary and keywords, and caches them
        prep = client.post("/api/prepare", json={"paper_id": data["paper_id"]}).get_json()
        self.assertEqual(prep["topic"], "graph neural networks, traffic forecasting")
        self.assertIn("summary", prep["summary"])
        self.assertEqual(prep["llm_calls"], 2)
        again = client.post("/api/prepare", json={"paper_id": data["paper_id"]}).get_json()
        self.assertEqual(again["llm_calls"], 0)
        # Figures are explained on demand, with their captions (also when the caption is on the next page)
        client.post("/api/figures/fig1/explain", json={})
        client.post("/api/figures/fig2/explain", json={})
        self.assertIn("Architecture of FlowNet-X", seen[0])
        self.assertIn("Forecast error versus prediction horizon", seen[1])

    def test_upload_runs_in_the_background_and_reports_progress(self):
        client = app.test_client()
        stages = []

        def slow_load(path, filename="", paper_id=None, progress=None):
            def spy(message):
                stages.append(message)
                progress(message)
            time.sleep(0.3)                       # the upload request must not wait for this
            return load_paper(path, filename, paper_id, encoder=hash_encoder, progress=spy)

        with patch("src.app.load_paper", slow_load):
            with open(self.pdf, "rb") as f:
                start = time.time()
                res = client.post("/upload", data={"file": (io.BytesIO(f.read()), "paper.pdf")},
                                  content_type="multipart/form-data")
                answered_in = time.time() - start
            self.assertEqual(res.status_code, 202)
            self.assertLess(answered_in, 0.25)
            job_id = res.get_json()["job_id"]
            seen = set()
            for _ in range(200):
                job = client.get(f"/api/jobs/{job_id}").get_json()
                seen.add(job["stage"])
                if job["status"] != "running":
                    break
                time.sleep(0.02)
        self.assertEqual(job["status"], "done", job)
        self.assertEqual(job["result"]["stats"]["pages"], 3)
        self.assertIn("Reading the layout of page 1 of 3", stages)
        self.assertIn("Finding figures and their captions", stages)
        self.assertTrue(any(s.startswith("Building the search index") for s in stages))
        self.assertEqual(len(client.get("/papers").get_json()["papers"]), 1)       # added to this browser's library

    def test_job_status_is_private_to_its_browser_and_lost_jobs_say_so(self):
        alice, bob = app.test_client(), app.test_client()
        with patch("src.app.load_paper", _offline_load_paper):
            with open(self.pdf, "rb") as f:
                job_id = alice.post("/papers", data={"file": (io.BytesIO(f.read()), "p.pdf")},
                                    content_type="multipart/form-data").get_json()["job_id"]
            for _ in range(200):
                if alice.get(f"/api/jobs/{job_id}").get_json()["status"] != "running":
                    break
                time.sleep(0.02)
        self.assertEqual(bob.get(f"/api/jobs/{job_id}").status_code, 404)
        lost = alice.get("/api/jobs/unknown-job")
        self.assertEqual(lost.status_code, 404)
        self.assertIn("upload again", lost.get_json()["error"])

    def test_a_pdf_without_text_fails_cleanly_in_the_job(self):
        blank = os.path.join(self.tmp, "blank.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(blank)
        doc.close()
        pdfs = lambda: {f for f in os.listdir("uploads") if f.endswith(".pdf")} if os.path.isdir("uploads") else set()
        before = pdfs()
        res = self._upload(app.test_client(), "/upload", pdf=blank)
        self.assertEqual(res.status_code, 400)
        self.assertIn("No text could be extracted", res.get_json()["error"])
        self.assertEqual(pdfs(), before)                                         # the file was removed again

    def test_non_pdf_content_is_rejected_and_not_left_on_disk(self):
        client = app.test_client()
        before = set(os.listdir("uploads")) if os.path.isdir("uploads") else set()
        res = client.post("/papers", data={"file": (io.BytesIO(b"<html>not a pdf</html>"), "evil.pdf")},
                          content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        self.assertIn("not a valid PDF", res.get_json()["error"])
        after = set(os.listdir("uploads")) if os.path.isdir("uploads") else set()
        self.assertEqual(before, after)

    def test_removing_a_paper_deletes_its_file(self):
        client = app.test_client()
        with patch("src.app.load_paper", _offline_load_paper):
            paper_id = self._upload(client).get_json()["paper"]["id"]
        path = os.path.join("uploads", f"{paper_id}.pdf")
        self.assertTrue(os.path.exists(path))
        client.delete(f"/papers/{paper_id}")
        self.assertFalse(os.path.exists(path))

    def test_streaming_analysis_runs_in_the_callers_session(self):
        client = app.test_client()
        with patch("src.app.load_paper", _offline_load_paper):
            self._upload(client)

        def fake_stream(paper_id=None, **kwargs):
            # Evaluated lazily while the response body streams, i.e. after the request context is gone
            yield {"type": "probe", "papers_visible": len(library.papers), "paper_id": paper_id}

        with patch("src.app.stream_analysis", fake_stream):
            body = client.post("/analyze", json={"goal": "x"}).get_data(as_text=True)
        self.assertIn('"papers_visible": 1', body)

if __name__ == "__main__":
    unittest.main()

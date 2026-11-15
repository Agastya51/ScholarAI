import json
import os
import re
import shutil
import threading
import time
import uuid
from functools import wraps

from flask import Flask, request, jsonify, send_file, render_template, Response, g

from src.config import (
    UPLOAD_DIR, FIGURE_DIR, MAX_LLM_CALLS_PER_RUN, MAX_UPLOAD_MB, MAX_PDF_PAGES, SESSION_ISOLATION, SESSION_TTL_SECONDS
)
from src import study
from src.figures import explain_figure, explain_paper_figure, public_figure
from src.llm import track_usage, LLMError
from src.rag import get_embedder, load_paper, release_memory, library, sessions, set_session, use_session, current_session_id
from src.viewer import page_view
from src.workflows import prepare_graph, compare_graph, improve_graph, qa_simple_graph
from src.agents.qa_agent import answer_question
from src.agents.review_panel import review_graph
from src.agents.supervisor import stream_analysis
from src.agents.multi_paper import compare_papers
from src.pdf_builder import build_analysis_pdf, build_edited_original_pdf

# Initialize Flask application
app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SESSION_ISOLATION"] = SESSION_ISOLATION

# Warm up embedding model at startup so first request doesn't block on model init
try:
    get_embedder()
except Exception as _init_err:
    print(f"[Warning] Model pre-warm notice: {_init_err}")

# =========================
# SESSIONS: one paper library per browser
# =========================
SESSION_COOKIE = "scholarai_sid"
_SID_RE = re.compile(r"^[0-9a-f]{32}$")

@app.before_request
def bind_session():
    """Give every browser its own paper library (anonymous random id in an HttpOnly cookie)."""
    if not app.config.get("SESSION_ISOLATION") or request.path == "/health":
        return   # health probes carry no cookie and must not create sessions
    sid = request.cookies.get(SESSION_COOKIE, "")
    g.new_sid = not _SID_RE.match(sid)
    if g.new_sid:
        sid = uuid.uuid4().hex
    g.sid = sid
    g.session_token = set_session(sid)

@app.after_request
def persist_session(response):
    if getattr(g, "new_sid", False):
        secure = request.headers.get("X-Forwarded-Proto", request.scheme) == "https"
        response.set_cookie(SESSION_COOKIE, g.sid, max_age=7 * 24 * 3600, httponly=True, samesite="Lax",
                            secure=secure)
    return response

@app.teardown_request
def unbind_session(_exc):
    if g.pop("session_token", None) is not None:
        set_session(None)   # back to the default (no-session) library for whatever runs next on this thread

def cleanup_orphans():
    """Uploads from a previous process have no in-memory index any more: delete them (the embedding cache stays)."""
    try:
        for name in os.listdir(UPLOAD_DIR):
            if re.match(r"^[0-9a-f]{10}\.pdf$", name):
                os.remove(os.path.join(UPLOAD_DIR, name))
        shutil.rmtree(FIGURE_DIR, ignore_errors=True)
    except OSError:
        pass

cleanup_orphans()

@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": f"File is too large. The limit is {MAX_UPLOAD_MB} MB."}), 413

def agent_endpoint(fn):
    """Wrap an API handler: per-request LLM call budget, call count in the response, typed error mapping."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            with track_usage(MAX_LLM_CALLS_PER_RUN) as usage:
                result = fn(*args, **kwargs)
            if isinstance(result, dict):
                result["llm_calls"] = usage["calls"]
                return jsonify(result)
            return result
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except LLMError as e:
            return jsonify({"error": f"LLM error: {e}"}), 502
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return wrapper

def _json() -> dict:
    return request.get_json(silent=True) or {}

def _active_summary(data: dict) -> str:
    paper = library.active()
    return data.get("summary") or (paper.summary if paper else "")

def _save_upload() -> tuple:
    if "file" not in request.files:
        raise ValueError("No file part provided")
    file = request.files["file"]
    if not file or not file.filename:
        raise ValueError("No file selected")
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")
    if file.stream.read(5) != b"%PDF-":
        raise ValueError("This file is not a valid PDF")
    file.stream.seek(0)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    paper_id = uuid.uuid4().hex[:10]
    path = os.path.join(UPLOAD_DIR, f"{paper_id}.pdf")
    file.save(path)
    return paper_id, path, file.filename

def _index_upload(make_active: bool):
    """Save, parse and index an uploaded PDF in this request (used by tests and scripts)."""
    paper_id, path, filename = _save_upload()
    return _index_saved(paper_id, path, filename, make_active)

def _index_saved(paper_id: str, path: str, filename: str, make_active: bool, progress=None):
    """Parse and index a saved PDF into the current session's library; never leave the file behind on failure."""
    def discard():
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        shutil.rmtree(os.path.join(FIGURE_DIR, paper_id), ignore_errors=True)

    try:
        paper = load_paper(path, filename, paper_id, progress=progress)
    except Exception:
        discard()
        raise
    if not paper.chunks:
        discard()
        raise ValueError("No text could be extracted from this PDF (it may be a scanned image).")
    return library.add(paper, make_active=make_active)

# =========================
# BACKGROUND JOBS
# Reading and indexing a paper can take minutes on a small CPU. The upload request only saves the file and
# starts a job; the browser polls the job, so no request ever runs into a proxy or worker timeout.
# =========================
JOB_TTL_SECONDS = 1800
_jobs: dict = {}
_jobs_lock = threading.Lock()
_index_lock = threading.Lock()          # one paper is parsed at a time: bounded memory on a small instance

def _new_job(session_id: str) -> dict:
    now = time.time()
    with _jobs_lock:
        for jid in [j for j, job in _jobs.items() if now - job["updated"] > JOB_TTL_SECONDS]:
            _jobs.pop(jid, None)
        job = {"id": uuid.uuid4().hex, "session": session_id, "status": "running", "stage": "Waiting to start",
               "result": None, "error": None, "started": now, "updated": now}
        _jobs[job["id"]] = job
    return job

def _set_job(job: dict, **fields):
    with _jobs_lock:
        job.update(fields, updated=time.time())

def _run_index_job(job: dict, paper_id: str, path: str, filename: str, make_active: bool, build_result):
    with use_session(job["session"]):
        try:
            if not _index_lock.acquire(blocking=False):
                _set_job(job, stage="Waiting for another paper to finish")
                _index_lock.acquire()
            try:
                _set_job(job, stage="Opening the PDF")
                paper = _index_saved(paper_id, path, filename, make_active,
                                     progress=lambda msg: _set_job(job, stage=msg))
            finally:
                _index_lock.release()
                release_memory()
            _set_job(job, status="done", stage="Done", result=build_result(paper))
        except ValueError as e:
            _set_job(job, status="error", error=str(e))
        except Exception as e:
            print(f"[upload] indexing failed: {type(e).__name__}: {e}")
            _set_job(job, status="error", error=f"Could not read this PDF ({type(e).__name__}: {str(e)[:200]})")

def _start_index_job(make_active: bool, build_result):
    paper_id, path, filename = _save_upload()
    job = _new_job(current_session_id())
    worker = threading.Thread(target=_run_index_job, name=f"index-{paper_id}", daemon=True,
                              args=(job, paper_id, path, filename, make_active, build_result))
    worker.start()
    if app.config.get("INLINE_JOBS"):     # tests: finish before answering
        worker.join()
    return jsonify({"job_id": job["id"], "status": job["status"], "stage": job["stage"]}), 202

@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["session"] != current_session_id():
            return jsonify({"error": "This upload is no longer known to the server. The server may have restarted "
                                     "(on the free plan this usually means it ran out of memory). Please upload again.",
                            "status": "lost"}), 404
        out = {k: job[k] for k in ("status", "stage", "error")}
        out["seconds"] = round(time.time() - job["started"])
        if job["status"] == "done":
            out["result"] = job["result"]
    return jsonify(out)

# =========================
# WEB PAGE ROUTES
# =========================
@app.route("/")
def home():
    return render_template("upload.html", page="upload", max_mb=MAX_UPLOAD_MB, max_pages=MAX_PDF_PAGES)

@app.route("/qa")
def page_qa():
    return render_template("qa.html", page="qa")

@app.route("/compare")
def page_compare():
    return render_template("compare.html", page="compare")

@app.route("/improve")
def page_improve():
    return render_template("improve.html", page="improve")

@app.route("/review")
def page_review():
    return render_template("review.html", page="review")

@app.route("/analyze")
def page_analyze():
    return render_template("analyze.html", page="analyze")

@app.route("/library")
def page_library():
    return render_template("library.html", page="library")

@app.route("/download")
def page_download():
    return render_template("download.html", page="download")

@app.route("/figures")
def page_figures():
    return render_template("figures.html", page="figures")

@app.route("/study")
def page_study():
    return render_template("study.html", page="study")

@app.route("/benchmarks")
def page_benchmarks():
    return render_template("benchmarks.html", page="benchmarks")

# =========================
# API ENDPOINTS
# =========================
def _paper_stats(paper) -> dict:
    return {"pages": paper.pages, "sections": len(paper.sections()), "chunks": len(paper.chunks),
            "figures": len(paper.figures), "tables": len(paper.tables), "captions": len(paper.captions),
            "references": len(paper.references), "embedding_cache_hit": paper.cache_hit}

def _upload_result(paper) -> dict:
    return {"paper_id": paper.id, "title": paper.title, "stats": _paper_stats(paper),
            "figures": _figures_payload(paper), "trace": [], "llm_calls": 0}

@app.route("/upload", methods=["POST"])
def upload():
    """Step 1 of opening a paper: save it and start parsing + indexing in the background (no language-model
    calls). Answers 202 with a job id; poll /api/jobs/<id>. Then /api/prepare writes the summary, and figures
    are explained on demand."""
    try:
        return _start_index_job(True, _upload_result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/prepare", methods=["POST"])
@agent_endpoint
def prepare():
    """Step 2: grounded summary and search keywords for the active paper (cached on the paper)."""
    paper = _require_paper(_json().get("paper_id"))
    if paper.summary and getattr(paper, "topic", ""):
        return {"paper_id": paper.id, "summary": paper.summary, "topic": paper.topic, "trace": []}
    result = prepare_graph.invoke({"paper_id": paper.id, "trace": []})
    paper.topic = result["topic"]
    return {"paper_id": paper.id, "summary": result["summary"], "topic": result["topic"],
            "trace": result.get("trace", [])}

@app.route("/ask", methods=["POST"])
@agent_endpoint
def ask():
    data = _json()
    query = data.get("query", "").strip()
    if not query:
        raise ValueError("No query provided")
    if data.get("mode") == "simple":
        result = qa_simple_graph.invoke({"query": query, "trace": []})
        return {"answer": result["answer"], "sources": result.get("sources", []),
                "retrieval": result.get("retrieval", {}), "trace": result.get("trace", []), "mode": "simple"}
    result = answer_question(query, history=data.get("history"))
    return {**result, "mode": "agent"}

@app.route("/compare", methods=["POST"])
@agent_endpoint
def compare():
    data = _json()
    result = compare_graph.invoke({"summary": _active_summary(data), "topic": data.get("topic", ""), "trace": []})
    return {"papers": result.get("papers", []), "comparison": result["comparison"],
            "queries_tried": result.get("queries_tried", []), "trace": result.get("trace", [])}

@app.route("/improve", methods=["POST"])
@agent_endpoint
def improve():
    data = _json()
    if library.active() is None:
        raise ValueError("No paper loaded. Please upload a paper first.")
    result = improve_graph.invoke({"summary": _active_summary(data), "comparison": data.get("comparison", ""),
                                   "trace": []})
    return {"improvements": result["improvements"], "edits": result.get("edits", []),
            "revisions": result.get("revision", 0), "trace": result.get("trace", [])}

@app.route("/review", methods=["POST"])
@agent_endpoint
def review():
    data = _json()
    if library.active() is None:
        raise ValueError("No paper loaded. Please upload a paper first.")
    state = {"summary": _active_summary(data), "comparison": data.get("comparison", ""), "reviews": [], "trace": []}
    if data.get("reviewers"):
        state["reviewers"] = data["reviewers"]
    result = review_graph.invoke(state)
    return {"reviews": result.get("reviews", []), "meta_review": result.get("meta_review", {}),
            "review_markdown": result.get("review_markdown", ""), "trace": result.get("trace", [])}

def _sse(events) -> Response:
    # The response body is generated after the request context is gone, so re-bind the session inside the stream.
    sid = current_session_id()

    def gen():
        with use_session(sid):
            for evt in events:
                yield f"data: {json.dumps(evt, default=str)}\n\n"
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/analyze", methods=["POST"])
def analyze():
    """Start a supervisor multi-agent run; streams Server-Sent Events."""
    data = _json()
    if library.active() is None:
        return jsonify({"error": "No paper loaded. Please upload a paper first."}), 400
    return _sse(stream_analysis(paper_id=library.active_id, goal=data.get("goal", ""), hitl=bool(data.get("hitl"))))

@app.route("/analyze/resume", methods=["POST"])
def analyze_resume():
    """Resume an interrupted run with the human decision (e.g. {"topic": ...} or {"accepted": [...]})."""
    data = _json()
    if not data.get("thread_id"):
        return jsonify({"error": "thread_id is required"}), 400
    return _sse(stream_analysis(thread_id=data["thread_id"], resume=data.get("resume") or {}))

# =========================
# ACTIVE PAPER, FIGURES, SOURCE VIEWER
# =========================
def _require_paper(paper_id=None):
    paper = library.get(paper_id)
    if paper is None:
        raise ValueError("No paper loaded. Please upload a paper first.")
    return paper

def _figures_payload(paper) -> list:
    return [public_figure(f, paper.id, paper.figure_analyses.get(f["id"], "")) | {"mode": paper.figure_modes.get(f["id"])}
            for f in paper.figures]

@app.route("/api/paper")
def api_paper():
    """What the server currently holds for this browser (lets pages detect a paper lost to a restart)."""
    paper = library.active()
    if paper is None:
        return jsonify({"paper": None})
    return jsonify({"paper": {"id": paper.id, "title": paper.title, "filename": paper.filename, "pages": paper.pages,
                              "sections": paper.sections(), "figures": len(paper.figures),
                              "tables": len(paper.tables), "has_summary": bool(paper.summary)}})

@app.route("/api/figures")
def api_figures():
    paper = library.active()
    return jsonify({"paper_id": paper.id if paper else None, "figures": _figures_payload(paper) if paper else []})

@app.route("/api/figures/<figure_id>/explain", methods=["POST"])
@agent_endpoint
def api_explain_figure(figure_id):
    paper = _require_paper()
    out = explain_paper_figure(paper, figure_id, force=bool(_json().get("retry")))
    fig = paper.figure(figure_id)
    return {"figure": public_figure(fig, paper.id, out["text"]) | {"mode": out["mode"]}}

@app.route("/api/figures/<figure_id>/ask", methods=["POST"])
@agent_endpoint
def api_ask_figure(figure_id):
    paper = _require_paper()
    question = (_json().get("question") or "").strip()
    if not question:
        raise ValueError("Type a question about the figure first.")
    fig = paper.figure(figure_id)
    if fig is None:
        raise ValueError(f"No figure '{figure_id}' in this paper.")
    out = explain_figure(fig, question=question)
    return {"answer": out["text"], "mode": out["mode"]}

@app.route("/papers/<paper_id>/figures/<figure_id>")
def figure_image(paper_id, figure_id):
    paper = library.get(paper_id)
    fig = paper.figure(figure_id) if paper else None
    if fig is None or not os.path.exists(fig["path"]):
        return jsonify({"error": "Figure not found. The paper may need to be uploaded again."}), 404
    return send_file(os.path.abspath(fig["path"]), max_age=3600)

@app.route("/papers/<paper_id>/pages/<int:page_no>")
def page_image(paper_id, page_no):
    """One page of the original PDF as an image, with a cited passage / figure / table highlighted."""
    paper = library.get(paper_id)
    if paper is None or not os.path.exists(paper.path):
        return jsonify({"error": "This paper is no longer on the server. Upload it again."}), 404
    chunk = request.args.get("chunk", type=int)
    try:
        png, found = page_view(paper, page_no, chunk_id=chunk, figure_id=request.args.get("figure"),
                               quote=request.args.get("quote", "")[:600])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    response = Response(png, mimetype="image/png")
    response.headers["Cache-Control"] = "private, max-age=600"
    response.headers["X-Highlights"] = str(found)
    return response

# =========================
# STUDY MATERIAL
# =========================
@app.route("/api/study/insights", methods=["POST"])
@agent_endpoint
def api_insights():
    return dict(study.build_insights(_require_paper()))

@app.route("/api/study/practice", methods=["POST"])
@agent_endpoint
def api_practice():
    return dict(study.build_practice(_require_paper()))

@app.route("/api/explain", methods=["POST"])
@agent_endpoint
def api_explain():
    level = _json().get("level", "student")
    return {"level": level, "text": study.explain_at_level(_require_paper(), level)}

@app.route("/api/suggest-questions", methods=["POST"])
@agent_endpoint
def api_suggest_questions():
    return {"questions": study.suggest_questions(_require_paper())}

@app.route("/api/benchmarks")
def api_benchmarks():
    """Latest published benchmark run (written by `python -m evals.benchmark run`)."""
    path = os.path.join("evals", "benchmark", "latest.json")
    if not os.path.exists(path):
        return jsonify({"result": None})
    try:
        with open(path, encoding="utf-8") as f:
            return jsonify({"result": json.load(f)})
    except (OSError, ValueError):
        return jsonify({"result": None})

@app.route("/health")
def health():
    """Liveness probe for the hosting platform (does not touch the LLM)."""
    return jsonify({"status": "ok", "active_sessions": sessions.count(), "session_ttl_seconds": SESSION_TTL_SECONDS})

# =========================
# PAPER LIBRARY (multi-paper)
# =========================
@app.route("/papers", methods=["GET"])
def list_papers():
    return jsonify({"papers": library.list(), "active_id": library.active_id})

@app.route("/papers", methods=["POST"])
def add_paper():
    """Index an additional paper into the library in the background (poll /api/jobs/<id>)."""
    try:
        return _start_index_job(False, lambda paper: {"paper": paper.info(), "papers": library.list()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/papers/<paper_id>", methods=["DELETE"])
def delete_paper(paper_id):
    library.remove(paper_id)
    return jsonify({"papers": library.list(), "active_id": library.active_id})

@app.route("/multi-compare", methods=["POST"])
@agent_endpoint
def multi_compare():
    data = _json()
    return compare_papers(data.get("paper_ids"), data.get("focus", ""))

# =========================
# EXPORTS
# =========================
def _active_pdf_path() -> str:
    paper = library.active()
    if paper is None or not os.path.exists(paper.path):
        raise ValueError("Original PDF not found on server. Please upload a paper first.")
    return paper.path

@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    """Build and download the complete analysis PDF report with appendix."""
    try:
        data = _json()
        buf = build_analysis_pdf(
            _active_pdf_path(), data.get("summary", ""), data.get("vision", []), data.get("comparison", ""),
            data.get("improvements", ""), data.get("edits", []), data.get("papers", []),
            data.get("topic", "Research Paper"), data.get("qa_text", "")
        )
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name="paper_full_analysis.pdf")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download-original", methods=["POST"])
def download_original():
    """Build and download the original PDF with only in-place rewrites applied."""
    try:
        buf = build_edited_original_pdf(_active_pdf_path(), _json().get("edits", []))
        return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="paper_edited.pdf")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from src.config import FLASK_DEBUG, PORT
    app.run(debug=FLASK_DEBUG, port=PORT)

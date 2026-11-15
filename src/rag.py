"""Document understanding and retrieval core.

- PDF parsing: column-aware reading order, section headings, tables, figure/table captions, reference list
- Chunking: sentence-aware with overlap; every chunk keeps its page + section anchor
- VectorStore: FAISS cosine index + BM25, fused with Reciprocal Rank Fusion, optional MMR
- Embedding cache on disk keyed by content hash
- PaperLibrary: one library per browser session (see SessionManager)

LLM-dependent retrieval steps (query expansion, reranking) live in src/retrieval.py.
"""
import contextvars
import hashlib
import math
import os
import re
import shutil
import threading
import time
import uuid
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz
import faiss
import numpy as np

# PyMuPDF 2.x ships an optional neural layout analyser (ONNX, sometimes multi-process) that find_tables() and
# get_layout() use by default. On a small CPU-limited instance it takes seconds per page and a lot of memory,
# so it is switched off: tables are found from their ruling lines, which is what this parser was built for.
if hasattr(fitz, "_layout_enabled"):
    fitz._layout_enabled = False
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

from src.figures import MAX_FIGURES, attach_mentions, locate_figures
from src.config import (
    EMBEDDING_DIM, EMBEDDING_MODEL_NAME, FALLBACK_EMBEDDING_MODEL_NAME, QUERY_INSTRUCTION,
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_TOP_K, RRF_K, CANDIDATE_POOL, USE_MMR, MMR_LAMBDA,
    UPLOAD_DIR, FIGURE_DIR, EMBED_CACHE_DIR, EMBED_CACHE_MAX_FILES, MAX_PDF_PAGES,
    MAX_SESSIONS, SESSION_TTL_SECONDS, MAX_PAPERS_PER_SESSION,
)

# =========================
# EMBEDDINGS (shared model)
# =========================
_embed_model = None
_is_fastembed = False

# Memory on a 512 MB instance. The transformer's working memory grows with batch x tokens^2, and ONNX Runtime's
# arena keeps the largest block it ever allocated. Measured with a model of this size: 16 passages of 512 tokens
# with the arena on took the process from ~250 MB to ~1.1 GB; 4 passages capped at 256 tokens with the arena off
# stays near 280 MB.
EMBED_BATCH = 4
EMBED_MAX_TOKENS = 256
EMBED_MAX_CHARS = 1200        # passages are ~900 characters; longer ones (tables) are cut for the vector only

def _cap_tokens(model):
    """Truncate inputs to EMBED_MAX_TOKENS (the tokenizer sits at different places across fastembed versions)."""
    for owner in (model, getattr(model, "model", None)):
        tok = getattr(owner, "tokenizer", None)
        if tok is not None and hasattr(tok, "enable_truncation"):
            try:
                tok.enable_truncation(max_length=EMBED_MAX_TOKENS)
            except Exception:
                pass

def get_embedder():
    """Load the embedding model once for the whole process, set up for low memory."""
    global _embed_model, _is_fastembed
    if _embed_model is None:
        try:
            from fastembed import TextEmbedding
            # BAAI/bge-small-en-v1.5: 384-dimensional quantised ONNX model; one thread, no memory arena
            try:
                model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, threads=1, enable_cpu_mem_arena=False)
            except TypeError:
                model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, threads=1)
            _cap_tokens(model)
            _embed_model, _is_fastembed = model, True
        except Exception as e:
            try:
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL_NAME)
                _embed_model.max_seq_length = EMBED_MAX_TOKENS
                _is_fastembed = False
            except Exception:
                raise RuntimeError(f"Could not load embedding model: {e}")
    return _embed_model

def release_memory():
    """Give freed memory back to the operating system after a big job (parsing, indexing)."""
    import gc
    gc.collect()
    try:
        fitz.TOOLS.store_shrink(100)
    except Exception:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def embedding_model_id() -> str:
    """Identifier baked into cache keys so a model change never reuses stale vectors."""
    get_embedder()
    name = EMBEDDING_MODEL_NAME if _is_fastembed else FALLBACK_EMBEDDING_MODEL_NAME
    return f"{name}|max{EMBED_MAX_TOKENS}"      # the token cap changes the vectors, so it is part of the cache key

ProgressFn = Callable[[str], None]

def _noop(_message: str):
    pass

def _embed_batch(model, batch: List[str]) -> list:
    if _is_fastembed:
        return list(model.embed(batch, batch_size=len(batch)))
    return list(model.encode(batch, batch_size=len(batch)))

def encode(texts: List[str], progress: Optional[ProgressFn] = None) -> np.ndarray:
    """Embed passages a few at a time, reporting progress. Passages are batched by length so short ones are not
    padded up to a long neighbour; the result keeps the input order."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), np.float32)
    model = get_embedder()
    progress = progress or _noop
    clipped = [t[:EMBED_MAX_CHARS] for t in texts]
    order = sorted(range(len(clipped)), key=lambda i: len(clipped[i]))
    vectors: List[Optional[np.ndarray]] = [None] * len(clipped)
    for start in range(0, len(order), EMBED_BATCH):
        idx = order[start:start + EMBED_BATCH]
        for i, vec in zip(idx, _embed_batch(model, [clipped[i] for i in idx])):
            vectors[i] = np.asarray(vec, dtype=np.float32)
        progress(f"Building the search index: {min(start + EMBED_BATCH, len(order))} of {len(order)} passages")
    return np.stack(vectors).astype(np.float32)

def encode_query(texts: List[str]) -> np.ndarray:
    """Embed queries. bge models expect a retrieval instruction on the query side only."""
    get_embedder()
    if _is_fastembed and QUERY_INSTRUCTION:
        texts = [QUERY_INSTRUCTION + t for t in texts]
    return encode(texts)

def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

# =========================
# EMBEDDING CACHE (disk, keyed by content hash)
# =========================
def embedding_cache_key(texts: List[str], model_id: str) -> str:
    h = hashlib.sha256(model_id.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()[:32]

def embed_with_cache(texts: List[str], encoder: Callable[[List[str]], np.ndarray], model_id: str,
                     cache_dir: Optional[str] = EMBED_CACHE_DIR) -> np.ndarray:
    """Embed `texts`, reusing vectors from disk when the exact same content was embedded before."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    if not cache_dir:
        return np.asarray(encoder(texts), dtype=np.float32)
    path = os.path.join(cache_dir, embedding_cache_key(texts, model_id) + ".npy")
    if os.path.exists(path):
        try:
            cached = np.load(path)
            if cached.shape[0] == len(texts):
                os.utime(path, None)  # mark as recently used
                return cached.astype(np.float32)
        except Exception:
            pass  # unreadable cache entry: fall through and recompute
    vectors = np.asarray(encoder(texts), dtype=np.float32)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, vectors)
        _prune_cache(cache_dir)
    except OSError:
        pass  # caching is best-effort (read-only or full disk)
    return vectors

def _prune_cache(cache_dir: str, keep: int = EMBED_CACHE_MAX_FILES):
    files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".npy")]
    if len(files) <= keep:
        return
    for stale in sorted(files, key=os.path.getmtime)[:len(files) - keep]:
        try:
            os.remove(stale)
        except OSError:
            pass

# =========================
# SPARSE RETRIEVAL (BM25)
# =========================
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())

class BM25:
    """Minimal Okapi BM25 for exact-term matching (symbols, dataset names, hyperparameters)."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tfs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.df: Counter = Counter()

    def add(self, texts: List[str]):
        for t in texts:
            tokens = tokenize(t)
            tf = Counter(tokens)
            self.doc_tfs.append(tf)
            self.doc_lens.append(len(tokens))
            self.df.update(tf.keys())

    def scores(self, query: str) -> np.ndarray:
        n = len(self.doc_tfs)
        out = np.zeros(n, dtype=np.float32)
        if n == 0:
            return out
        avgdl = (sum(self.doc_lens) / n) or 1.0
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.doc_tfs):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out

# =========================
# VECTOR STORE (hybrid)
# =========================
Chunk = Dict[str, object]  # {"text", "page", "section", "kind", ...}

_REFERENCE_QUERY = re.compile(r"\b(refs?|references?|bibliograph\w*|cit(?:e[sd]?|ations?|ing))\b|\[\d{1,3}\]", re.I)

def excluded_kinds_for(query: str) -> Set[str]:
    """Bibliography entries match almost any keyword, so they are searched only when the question is about them."""
    return set() if _REFERENCE_QUERY.search(query or "") else {"reference"}

class VectorStore:
    """FAISS cosine index + BM25 sparse index fused with Reciprocal Rank Fusion (+ optional MMR)."""
    def __init__(self, dimension: int = EMBEDDING_DIM, encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
                 query_encoder: Optional[Callable[[List[str]], np.ndarray]] = None):
        self.dimension = dimension
        self._encoder = encoder
        self._query_encoder = query_encoder
        self.clear()

    def clear(self):
        # Vectors are L2-normalised, so inner product == cosine similarity (scores are comparable across queries)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.vectors = np.zeros((0, self.dimension), dtype=np.float32)
        self.bm25 = BM25()
        self.documents: List[str] = []
        self.metadata: List[Chunk] = []
        self._query_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    # ---- encoding ----
    def encode(self, texts: List[str]) -> np.ndarray:
        return _normalize((self._encoder or encode)(texts))

    def encode_query(self, query: str) -> np.ndarray:
        """Normalised query vector, memoised (the same query is embedded once per paper)."""
        if query in self._query_cache:
            self._query_cache.move_to_end(query)
            return self._query_cache[query]
        if self._query_encoder is not None:
            fn = self._query_encoder
        elif self._encoder is not None:
            fn = self._encoder
        else:
            fn = encode_query
        vec = _normalize(fn([query]))[0]
        self._query_cache[query] = vec
        if len(self._query_cache) > 128:
            self._query_cache.popitem(last=False)
        return vec

    # ---- indexing ----
    def add_chunks(self, chunks: List[Union[str, Chunk]], embeddings: Optional[np.ndarray] = None):
        """Index chunks. Each chunk may carry `embed_text` (text + section header) used for both indexes."""
        if not chunks:
            return
        metas = [c if isinstance(c, dict) else {"text": c, "page": None, "section": None, "kind": "text"} for c in chunks]
        index_texts = [str(m.get("embed_text") or m["text"]) for m in metas]
        vectors = _normalize(embeddings) if embeddings is not None else self.encode(index_texts)
        self.index.add(vectors)
        self.vectors = np.vstack([self.vectors, vectors])
        self.bm25.add(index_texts)
        self.documents.extend(str(m["text"]) for m in metas)
        self.metadata.extend(metas)

    # ---- scoring primitives (used directly by src/retrieval.py) ----
    def dense_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query to every chunk."""
        if not len(self.documents):
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ self.encode_query(query)

    def rankings(self, query: str, pool: int = CANDIDATE_POOL, mode: str = "hybrid",
                 exclude_kinds: Optional[Set[str]] = None) -> List[List[int]]:
        """Ranked candidate ids per retriever. mode: 'hybrid' | 'dense' | 'sparse'."""
        n = len(self.documents)
        if not n or not query:
            return []
        exclude_kinds = exclude_kinds or set()
        allowed = [i for i in range(n) if self.metadata[i].get("kind") not in exclude_kinds] or list(range(n))
        allowed_set = set(allowed)
        out: List[List[int]] = []
        if mode in ("hybrid", "dense"):
            scores = self.dense_scores(query)
            out.append([int(i) for i in np.argsort(-scores) if int(i) in allowed_set][:pool])
        if mode in ("hybrid", "sparse"):
            scores = self.bm25.scores(query)
            out.append([int(i) for i in np.argsort(-scores) if scores[i] > 0 and int(i) in allowed_set][:pool])
        return out

    def reference_ranking(self, query: str) -> List[int]:
        """For questions about the bibliography ('what is reference [12]?'): reference chunks, the ones that
        contain the cited number first. Empty for every other question."""
        if excluded_kinds_for(query):
            return []
        ids = [i for i, m in enumerate(self.metadata) if m.get("kind") == "reference"]
        if not ids:
            return []
        markers = re.findall(r"\[(\d{1,3})\]", query)
        dense = self.dense_scores(query)
        return sorted(ids, key=lambda i: (not any(f"[{n}]" in self.documents[i] for n in markers), -float(dense[i])))

    @staticmethod
    def fuse(rankings: Iterable[List[int]]) -> Dict[int, float]:
        """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (RRF_K + rank)."""
        fused: Dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        return fused

    def mmr(self, candidates: List[int], relevance: Dict[int, float], k: int, lam: float = MMR_LAMBDA) -> List[int]:
        """Maximal Marginal Relevance: pick chunks that are relevant AND not redundant with those already picked."""
        if len(candidates) <= 1 or k <= 0:
            return candidates[:max(k, 0)]
        vals = np.array([relevance[i] for i in candidates], dtype=np.float32)
        span = float(vals.max() - vals.min())
        rel = {i: (float(relevance[i]) - float(vals.min())) / span if span else 1.0 for i in candidates}
        selected: List[int] = []
        remaining = list(candidates)
        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda i: rel[i])
            else:
                chosen = self.vectors[selected]
                best = max(remaining, key=lambda i: lam * rel[i] - (1 - lam) * float(np.max(chosen @ self.vectors[i])))
            selected.append(best)
            remaining.remove(best)
        return selected

    def describe(self, ids: List[int], fused: Dict[int, float], query: str) -> List[Chunk]:
        """Attach scores to chunk metadata: RRF `score`, cosine `dense_score`, keyword `bm25_score`."""
        if not ids:
            return []
        dense = self.dense_scores(query)
        sparse = self.bm25.scores(query)
        return [{**self.metadata[i], "score": round(float(fused.get(i, 0.0)), 5),
                 "dense_score": round(float(dense[i]), 4), "bm25_score": round(float(sparse[i]), 3),
                 "chunk_id": i} for i in ids]

    # ---- public search ----
    def search(self, query: str, k: int = DEFAULT_TOP_K, mode: str = "hybrid", mmr: Optional[bool] = None,
               exclude_kinds: Optional[Set[str]] = None) -> List[Chunk]:
        """Return top-k chunks with metadata and scores. mode: 'hybrid' | 'dense' | 'sparse'."""
        if not self.documents or not query:
            return []
        if exclude_kinds is None:
            exclude_kinds = excluded_kinds_for(query)
        pool = min(len(self.documents), max(k * 4, CANDIDATE_POOL))
        rankings = self.rankings(query, pool, mode, exclude_kinds)
        refs = self.reference_ranking(query)
        fused = self.fuse(rankings + [refs, refs] if refs else rankings)   # counted twice: outweighs one retriever
        ranked = sorted(fused, key=fused.get, reverse=True)
        use_mmr = USE_MMR if mmr is None else mmr
        top = self.mmr(ranked[:pool], fused, k) if use_mmr else ranked[:k]
        return self.describe(top, fused, query)

    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> List[str]:
        return [str(c["text"]) for c in self.search(query, k)]

# =========================
# PDF PARSING (layout + section + page aware)
# =========================
_NUM_PREFIX = re.compile(r"^((\d+(\.\d+)*\.?)|([IVXLC]+\.)|([A-H]\.))\s+")
_KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "related works", "background", "literature review",
    "method", "methods", "methodology", "approach", "proposed method", "proposed approach",
    "system model", "problem formulation", "experiments", "experimental setup", "experimental results",
    "results", "evaluation", "discussion", "results and discussion", "conclusion", "conclusions",
    "conclusion and future work", "limitations", "future work", "references", "bibliography", "acknowledgment",
    "acknowledgments", "acknowledgements", "appendix",
}
_REFERENCE_HEADING = re.compile(r"\b(references|bibliography)\b", re.I)

def detect_heading(line: str) -> Optional[str]:
    """Heuristically detect a section heading line (numbered or a well-known heading name)."""
    s = line.strip()
    if not s or len(s) > 90:
        return None
    body = _NUM_PREFIX.sub("", s).strip().rstrip(":").strip()
    if body.lower() in _KNOWN_HEADINGS:
        return s
    if _NUM_PREFIX.match(s) and body and body[0].isupper():
        words = body.split()
        if 1 <= len(words) <= 8 and not s.endswith((".", ",", ";")) and not re.search(r"\d{3,}|[=<>%]", body):
            return s
    return None

_SUBSECTION_PREFIX = re.compile(r"^(\d+\.\d+(\.\d+)*\.?|[A-H]\.)\s+")

def heading_level(heading: str) -> int:
    """1 for top-level headings ('3 Method', 'IV. EXPERIMENTS', 'Abstract'), 2 for subsections ('3.1 Setup', 'A. Data')."""
    return 2 if _SUBSECTION_PREFIX.match(heading.strip()) else 1

def is_reference_section(section: Optional[str]) -> bool:
    return bool(section) and bool(_REFERENCE_HEADING.search(str(section)))

_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
                            "­": ""})
_HYPHEN_BREAK = re.compile(r"([a-z])-\n([a-z])")      # "quadrat-\nically" -> "quadratically"
_NUMERIC_HYPHEN_BREAK = re.compile(r"(\d)-\n([a-z])")  # "60-\nminute"      -> "60-minute" (the hyphen is real)
_HYPHEN_BEFORE_DIGIT = re.compile(r"(\w)-\n(\d)")     # "RiverCity-\n200"  -> "RiverCity-200"
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

def clean_text(text: str) -> str:
    """Undo PDF typesetting artefacts that break keyword search: ligatures (ﬁ -> fi), line-end hyphenation,
    and control characters left behind by math fonts."""
    text = _CONTROL_CHARS.sub("", (text or "").translate(_LIGATURES))
    text = _NUMERIC_HYPHEN_BREAK.sub(r"\1-\2", text)
    text = _HYPHEN_BEFORE_DIGIT.sub(r"\1-\2", text)
    return _HYPHEN_BREAK.sub(r"\1\2", text)

def _block_text(block: dict) -> str:
    """Text of one layout block, one visual line per text line.

    PyMuPDF splits a visual line wherever there is a wide gap, so a heading typeset as '3.1   Training Details'
    arrives as two lines ('3.1', 'Training Details'). Fragments on the same baseline are joined back together,
    which is what makes numbered headings (and table rows) recognisable. Rotated text is ignored.
    """
    lines: List[str] = []
    prev = None  # bbox of the previous fragment
    for line in block.get("lines", []):
        direction = line.get("dir", (1.0, 0.0))
        if abs(direction[0] - 1.0) > 0.05 or abs(direction[1]) > 0.05:
            continue
        text = "".join(span.get("text", "") for span in line.get("spans", []))
        if not text.strip():
            continue
        x0, y0, x1, y1 = line["bbox"]
        if prev is not None and lines:
            overlap = min(y1, prev[3]) - max(y0, prev[1])
            same_baseline = overlap > 0.6 * min(y1 - y0, prev[3] - prev[1])
            if same_baseline and x0 >= prev[2] - 1:
                lines[-1] = lines[-1].rstrip() + " " + text.strip()
                prev = (prev[0], min(y0, prev[1]), x1, max(y1, prev[3]))
                continue
        lines.append(text.rstrip())
        prev = (x0, y0, x1, y1)
    return "\n".join(lines) + ("\n" if lines else "")

def ordered_blocks(page, info: Optional[dict] = None) -> List[dict]:
    """Text blocks of a page in reading order.

    Single-column pages keep PyMuPDF's native order. Two-column pages (IEEE / ACM style) are read
    left column then right column, within each horizontal band delimited by full-width blocks
    (title, abstract, wide figures / captions).
    """
    width = page.rect.width
    mid = page.rect.x0 + width / 2
    blocks = []
    for b in page.get_text("dict").get("blocks", []):
        if b.get("type") != 0:
            continue
        text = _block_text(b)
        x0, y0, x1, y1 = b["bbox"]
        if not text.strip():
            continue
        if (x1 - x0) < 25 and (y1 - y0) > 120:
            continue  # rotated margin stamp (e.g. the arXiv identifier)
        blocks.append({"bbox": (x0, y0, x1, y1), "text": text})
    if len(blocks) < 4:
        return blocks

    def spans(b) -> bool:
        x0, _, x1, _ = b["bbox"]
        return (x1 - x0) > 0.6 * width or (x0 < mid - 0.08 * width and x1 > mid + 0.08 * width)

    narrow = [b for b in blocks if not spans(b)]
    left = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
    right = [b for b in narrow if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid]
    chars = lambda bs: sum(len(b["text"]) for b in bs)
    total = chars(blocks) or 1
    two_column = (len(left) >= 2 and len(right) >= 2 and chars(narrow) >= 0.5 * total
                  and chars(left) >= 0.2 * chars(narrow) and chars(right) >= 0.2 * chars(narrow))
    if info is not None:
        info["two_column"] = two_column
    if not two_column:
        return blocks

    spanning = sorted((b for b in blocks if spans(b)), key=lambda b: b["bbox"][1])
    band_of = lambda b: sum(1 for s in spanning if s["bbox"][1] <= b["bbox"][1])
    out: List[dict] = []
    for band in range(len(spanning) + 1):
        if band > 0:
            out.append(spanning[band - 1])
        out += sorted((b for b in left if band_of(b) == band), key=lambda b: b["bbox"][1])
        out += sorted((b for b in right if band_of(b) == band), key=lambda b: b["bbox"][1])
    return out

_CAPTION_RE = re.compile(
    r"^\s*(fig(?:ure|\.)?|table)\s*\.?\s*(\d{1,3}|[IVXL]{1,6})\s*(?:[.:|]|—|–|-\s|\n|$)\s*(.*)",
    re.I | re.S)

def detect_caption(block_text: str) -> Optional[dict]:
    """Recognise 'Fig. 3. ...', 'Figure 3: ...', 'TABLE II <newline> ...' captions (not 'Figure 3 shows ...' prose)."""
    m = _CAPTION_RE.match(block_text or "")
    if not m:
        return None
    kind = "table" if m.group(1).lower().startswith("table") else "figure"
    number = m.group(2).upper()
    body = re.sub(r"\s+", " ", clean_text(m.group(3))).strip()
    label = f"{'Table' if kind == 'table' else 'Figure'} {number}"
    return {"kind": kind, "number": number, "label": label, "text": f"{label}: {body}"[:700] if body else label}

def _guess_title(doc, first_page_text: str) -> str:
    meta_title = (doc.metadata or {}).get("title", "").strip()
    if len(meta_title) > 8:
        return meta_title
    for line in first_page_text.splitlines():
        if len(line.split()) >= 3:
            return line.strip()[:200]
    return "Untitled paper"

def _inside(inner, outer, min_overlap: float = 0.8) -> bool:
    """True when at least `min_overlap` of the inner box's area lies in the outer box."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    w = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    h = max(0.0, min(iy1, oy1) - max(iy0, oy0))
    area = max(1e-6, (ix1 - ix0) * (iy1 - iy0))
    return (w * h) / area >= min_overlap

def _nearest_caption(bbox, captions: List[dict]) -> Optional[dict]:
    """Closest caption to an image/table box on the same page; captions directly below (figures) or above
    (tables) with horizontal overlap win over anything else."""
    if not captions:
        return None
    if bbox is None:
        return captions[0]
    x0, y0, x1, y1 = bbox

    def cost(c):
        cx0, cy0, cx1, cy1 = c["bbox"]
        overlap = min(x1, cx1) - max(x0, cx0) > 0
        gap = cy0 - y1 if cy0 >= y1 - 5 else (y0 - cy1 if cy1 <= y0 + 5 else 0.0)
        return abs(gap) + (0 if overlap else 1000)

    return min(captions, key=cost)

class RuleTable:
    """A table found from horizontal rules only (LaTeX booktabs style), with its cell texts already read."""
    def __init__(self, bbox, rows_text: List[List[str]]):
        self.bbox = tuple(bbox)
        self.rows_text = rows_text
        self.row_count = len(rows_text)
        self.col_count = max((len(r) for r in rows_text), default=0)

def _horizontal_rules(drawings) -> List[tuple]:
    rules = []
    for d in drawings:
        r = d.get("rect")
        if r is not None and r.height <= 2.0 and r.width >= 40:
            rules.append((r.x0, (r.y0 + r.y1) / 2, r.x1))
    return sorted(rules, key=lambda r: r[1])

def _split_columns(rows: List[List[tuple]], min_gap: float = 7.0) -> List[float]:
    """x positions that separate columns: gaps that no word in any row crosses."""
    spans = sorted((w[0], w[2]) for row in rows for w in row)
    if not spans:
        return []
    cuts, reach = [], spans[0][1]
    for x0, x1 in spans[1:]:
        if x0 - reach >= min_gap:
            cuts.append((reach + x0) / 2)
        reach = max(reach, x1)
    return cuts

def booktabs_tables(page, drawings, exclude: List[tuple] = ()) -> List[RuleTable]:
    """Tables delimited by horizontal rules of equal width (top rule, header rule, bottom rule) and no
    vertical lines. Rows are text lines; columns are the gaps that run through every row."""
    rules = _horizontal_rules(drawings)
    groups: List[List[tuple]] = []
    for r in rules:
        g = next((g for g in groups if abs(g[-1][0] - r[0]) <= 3 and abs(g[-1][2] - r[2]) <= 3
                  and r[1] - g[-1][1] <= 260), None)
        if g is None:
            groups.append([r])
        elif r[1] - g[-1][1] > 1.5:            # ignore double-drawn rules
            g.append(r)
    out = []
    words = page.get_text("words")
    for g in groups:
        if len(g) < 2:
            continue
        box = (g[0][0] - 2, g[0][1], g[0][2] + 2, g[-1][1])
        if box[3] - box[1] < 20 or any(_inside(box, ex, 0.5) for ex in exclude):
            continue
        inside = [w for w in words if box[0] <= (w[0] + w[2]) / 2 <= box[2] and box[1] <= (w[1] + w[3]) / 2 <= box[3]]
        lines: List[List[tuple]] = []
        for w in sorted(inside, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
            mid = (w[1] + w[3]) / 2
            if lines and abs((lines[-1][0][1] + lines[-1][0][3]) / 2 - mid) <= 3:
                lines[-1].append(w)
            else:
                lines.append([w])
        if len(lines) < 2:
            continue
        cuts = _split_columns(lines)
        if not cuts:
            continue
        rows_text = []
        for line in lines:
            cells = [""] * (len(cuts) + 1)
            for w in sorted(line, key=lambda w: w[0]):
                col = sum((w[0] + w[2]) / 2 > c for c in cuts)
                cells[col] = (cells[col] + " " + w[4]).strip()
            rows_text.append([re.sub(r"\s+", " ", clean_text(c)).strip() for c in cells])
        if len(rows_text) >= 2 and len(rows_text[0]) >= 2:
            out.append(RuleTable(box, rows_text))
    return out

def find_ruled_tables(page, has_table_caption: bool = True) -> list:
    """Tables drawn with ruling lines (the common case in papers), without PyMuPDF's neural layout stage.

    1. PyMuPDF's line-based finder (full grids).
    2. On pages with a table caption, tables delimited by horizontal rules only (LaTeX booktabs).
    Pages with no vector drawings cannot hold a ruled table and are skipped without calling the finder.
    """
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    if not drawings:
        return []
    tables: list = []
    try:
        try:
            found = page.find_tables(use_layout=False)
        except TypeError:                      # PyMuPDF < 2.0 has no use_layout switch (and no layout stage)
            found = page.find_tables()
        tables = [t for t in found.tables if t.row_count >= 2 and t.col_count >= 2]
    except Exception:
        tables = []
    if has_table_caption:
        try:
            tables += booktabs_tables(page, drawings, exclude=[tuple(t.bbox) for t in tables])
        except Exception:
            pass
    return tables

def table_rows(page, table) -> List[List[str]]:
    """Cell texts of a detected table.

    The page's words are read once and each word is placed in the cell that contains its centre.
    (PyMuPDF's own Table.extract() can detach low glyphs such as '.' from their word, turning '3.60'
    into '360' + '.', and reading every cell with get_textbox() rescans the whole page per cell.)
    """
    if isinstance(table, RuleTable):
        return table.rows_text
    try:
        words = page.get_text("words", clip=fitz.Rect(table.bbox))
    except Exception:
        words = []
    rows = []
    for row in table.rows:
        cells = []
        for c in row.cells:
            if not c:
                cells.append("")
                continue
            x0, y0, x1, y1 = c
            inside = [w for w in words if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1]
            inside.sort(key=lambda w: (w[5], w[6], w[7]))          # block, line, word order
            cells.append(re.sub(r"\s+", " ", clean_text(" ".join(w[4] for w in inside))).strip())
        if any(cells):
            rows.append(cells)
    return rows

def rows_to_markdown(rows: List[List[str]]) -> str:
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    norm = [[c.replace("|", "/") for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["|" + "|".join(norm[0]) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines += ["|" + "|".join(r) + "|" for r in norm[1:]]
    return "\n".join(lines)

_REF_SPLIT = re.compile(r"(?m)^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+(?=\S)")

def parse_references(text: str) -> List[dict]:
    """Split a reference list into numbered entries ('[12] ...' or '12. ...')."""
    matches = list(_REF_SPLIT.finditer(text or ""))
    refs = []
    for m, nxt in zip(matches, matches[1:] + [None]):
        body = re.sub(r"\s+", " ", text[m.end(): nxt.start() if nxt else len(text)]).strip()
        n = int(m.group(1) or m.group(2))
        if len(body) >= 20 and (not refs or n == refs[-1]["n"] + 1 or m.group(1)):
            refs.append({"n": n, "text": body[:600]})
    return refs

def parse_document(file_path: str, figure_dir: str = FIGURE_DIR, max_images: int = MAX_FIGURES,
                   progress: Optional[ProgressFn] = None) -> dict:
    """Extract text segments (page + section), figures, tables, captions and references from a PDF.

    Returns dict with keys: title, text, pages, segments [{page, section, text}],
    figures [{id, label, page, path, caption, caption_page, mentions}] (see src/figures.py),
    tables [{id, page, section, markdown, caption, mentions}],
    captions [{kind, number, label, text, page, section}], references [{n, text}].
    """
    doc = fitz.open(file_path)
    try:   # always release the file handle (an open handle blocks deleting the upload on Windows)
        if doc.page_count > MAX_PDF_PAGES:
            raise ValueError(f"This PDF has {doc.page_count} pages; the limit is {MAX_PDF_PAGES}. "
                             "Upload a shorter paper.")
        os.makedirs(figure_dir, exist_ok=True)
        return _parse_pages(doc, figure_dir, max_images, progress or _noop)
    finally:
        doc.close()

def _parse_pages(doc, figure_dir: str, max_images: int, progress: ProgressFn = _noop) -> dict:
    # ---- Pass A: layout of every page (blocks in reading order, tables, captions) ----
    layout: Dict[int, dict] = {}
    captions: List[dict] = []
    for page_no, page in enumerate(doc, start=1):
        progress(f"Reading the layout of page {page_no} of {doc.page_count}")
        info: dict = {}
        blocks = ordered_blocks(page, info)
        has_table_caption = any((detect_caption(b["text"]) or {}).get("kind") == "table" for b in blocks)
        page_tables = find_ruled_tables(page, has_table_caption)
        # A "table" that covers most of the page is usually a mis-detected layout: keep the prose in that case
        page_area = max(1.0, page.rect.width * page.rect.height)
        table_boxes = [tuple(t.bbox) for t in page_tables]
        table_boxes = [tb if (tb[2] - tb[0]) * (tb[3] - tb[1]) <= 0.7 * page_area else (0, 0, 0, 0)
                       for tb in table_boxes]
        block_captions: Dict[int, dict] = {}
        for i, block in enumerate(blocks):
            caption = detect_caption(block["text"])
            if caption:
                block_captions[i] = {**caption, "page": page_no, "section": "", "bbox": block["bbox"]}
                captions.append(block_captions[i])
        layout[page_no] = {"blocks": blocks, "tables": page_tables, "table_boxes": table_boxes,
                           "captions": block_captions, "two_column": bool(info.get("two_column"))}

    # ---- Figures: join each picture with its caption, even across a page break (src/figures.py) ----
    progress("Finding figures and their captions")
    figures = locate_figures(doc, captions,
                             {n: l["blocks"] for n, l in layout.items()},
                             {n: [tb for tb in l["table_boxes"] if tb != (0, 0, 0, 0)] for n, l in layout.items()},
                             figure_dir, max_images,
                             two_column={n: l["two_column"] for n, l in layout.items()})
    figure_boxes: Dict[int, list] = {}
    for fig in figures:
        figure_boxes.setdefault(fig["page"], []).append(fig["box"])

    # ---- Pass B: running text with section anchors; captions, table cells and figure labels are kept out ----
    segments, tables, full_text = [], [], []
    section = "Front matter"
    top_section = ""
    title = ""
    for page_no, page in enumerate(doc, start=1):
        progress(f"Reading the text of page {page_no} of {doc.page_count}")
        lay = layout[page_no]
        table_boxes = lay["table_boxes"]
        page_text_parts: List[str] = []
        table_sections: Dict[int, str] = {}   # section in force where each table sits in the reading order
        buf: List[str] = []

        def flush():
            if any(line.strip() for line in buf):
                segments.append({"page": page_no, "section": section, "text": "\n".join(buf)})
            buf.clear()

        for i, block in enumerate(lay["blocks"]):
            text = clean_text(block["text"])
            page_text_parts.append(text)
            if i in lay["captions"]:
                # Captions are indexed as their own chunks instead of interrupting the paragraph around them
                lay["captions"][i]["section"] = section
                continue
            box = next((j for j, tb in enumerate(table_boxes) if _inside(block["bbox"], tb)), None)
            if box is not None:
                table_sections.setdefault(box, section)
                continue  # table cells are indexed as a markdown table chunk instead
            if len(text.strip()) <= 120 and any(_inside(block["bbox"], fb, 0.6) for fb in figure_boxes.get(page_no, [])):
                continue  # axis titles, tick labels and legends live in the figure image, not in the prose
            for line in text.splitlines():
                heading = detect_heading(line)
                if heading is None and line.strip().lower().startswith("abstract") and len(line) > 15:
                    flush()  # inline IEEE style "Abstract—..."
                    top_section = section = "Abstract"
                    buf.append(line)
                    continue
                if heading:
                    flush()
                    if heading_level(heading) == 1:
                        top_section = section = heading
                    else:
                        # Keep the parent in the anchor so 'Method' also finds '3.1 Problem Formulation'
                        section = f"{top_section} \u203a {heading}" if top_section else heading
                else:
                    buf.append(line)
            buf.append("")  # block boundary = paragraph break for the chunker
        flush()

        page_text = "\n".join(page_text_parts)
        full_text.append(page_text)
        if page_no == 1:
            title = _guess_title(doc, page_text)

        table_captions = [c for c in lay["captions"].values() if c["kind"] == "table"]
        for j, t in enumerate(lay["tables"]):
            try:
                markdown = rows_to_markdown(table_rows(page, t))
            except Exception:
                continue
            if not markdown:
                continue
            cap = _nearest_caption(tuple(t.bbox), table_captions)
            table_section = table_sections.get(j) or (cap["section"] if cap else section)
            tables.append({"id": f"table{len(tables) + 1}", "page": page_no, "section": table_section,
                           "markdown": markdown, "caption": cap["text"] if cap else "",
                           "label": cap["label"] if cap else "", "box": tuple(t.bbox)})

    # Every sentence that cites a figure or table ("As Fig. 3 shows, ..."), wherever it is in the paper
    by_text = {c["text"]: c for c in captions if c["kind"] == "figure"}
    for fig in figures:
        if fig.get("caption") in by_text:
            fig["section"] = by_text[fig["caption"]]["section"]
    prose = [seg for seg in segments if not is_reference_section(seg["section"])]
    attach_mentions(figures, prose, "figure")
    attach_mentions(tables, prose, "table")

    reference_text = "\n".join(s["text"] for s in segments if is_reference_section(s["section"]))
    for c in captions:
        c.pop("bbox", None)
    return {"title": title, "text": "\n".join(full_text), "pages": len(full_text), "segments": segments,
            "figures": figures, "tables": tables, "captions": captions,
            "references": parse_references(reference_text)}

# =========================
# CHUNKING
# =========================
_UNIT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Sentence-aware chunking: pack whole sentences up to `size` chars, carrying ~`overlap` chars forward."""
    if not text:
        return []
    overlap = max(0, min(overlap, size // 2))
    units: List[str] = []
    for u in _UNIT_SPLIT.split(text):
        u = re.sub(r"\s+", " ", u).strip()
        if not u:
            continue
        # Hard-split units that are longer than a chunk (tables, equations, run-on text)
        step = max(1, size - overlap)
        units.extend([u[i:i + size] for i in range(0, len(u), step)] if len(u) > size else [u])

    chunks: List[str] = []
    current: List[str] = []
    length = 0
    for u in units:
        if current and length + len(u) + 1 > size:
            chunks.append(" ".join(current))
            carry: List[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev) + 1
            current, length = carry, carried
        current.append(u)
        length += len(u) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

MAX_TABLE_CHUNK_CHARS = 3500

def _with_header(chunk: Chunk) -> Chunk:
    """Contextual header: index '<section>: <text>' so a chunk is findable by the section it belongs to."""
    section = str(chunk.get("section") or "")
    if section and section != "Front matter":
        chunk["embed_text"] = f"{section}: {chunk['text']}"
    return chunk

def chunk_segments(segments: List[dict], size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Chunk]:
    """Chunk each (page, section) segment so every chunk keeps its citation anchor."""
    out: List[Chunk] = []
    for seg in segments:
        kind = "reference" if is_reference_section(seg.get("section")) else "text"
        for c in chunk_text(seg["text"], size, overlap):
            if len(c) >= 40:
                out.append(_with_header({"text": c, "page": seg["page"], "section": seg["section"], "kind": kind}))
    return out

def build_chunks(parsed: dict) -> List[Chunk]:
    """All retrievable units of a parsed paper: prose chunks, tables (with caption), figure/table captions."""
    chunks = chunk_segments(parsed["segments"])
    for t in parsed.get("tables", []):
        text = (t["caption"] + "\n" if t.get("caption") else "") + t["markdown"]
        if len(text) > MAX_TABLE_CHUNK_CHARS:   # the full table stays available through the get_table tool
            text = text[:MAX_TABLE_CHUNK_CHARS] + f"\n… (table truncated; full table: {t['id']})"
        chunks.append(_with_header({"text": text, "page": t["page"], "section": t["section"], "kind": "table",
                                    "table_id": t["id"]}))
    figure_by_caption = {f["caption"]: f["id"] for f in parsed.get("figures", []) if f.get("caption")}
    for c in parsed.get("captions", []):
        if c["kind"] != "figure" or len(c["text"]) < 25:
            continue  # table captions already travel with their table chunk
        chunk = {"text": c["text"], "page": c["page"], "section": c["section"], "kind": "caption",
                 "label": c["label"]}
        if c["text"] in figure_by_caption:
            chunk["figure_id"] = figure_by_caption[c["text"]]
        chunks.append(_with_header(chunk))
    return chunks

def format_chunks(chunks: List[Chunk], max_chars: int = 0) -> str:
    """Render chunks with citation anchors, e.g. '[p. 3 · 2. Method] ...'."""
    parts = []
    for c in chunks:
        anchor = f"p. {c.get('page')}" if c.get("page") else "p. ?"
        if c.get("section"):
            anchor += f" · {c['section']}"
        if c.get("kind") == "table":
            anchor += " · table"
        elif c.get("kind") == "caption":
            anchor += " · figure caption"
        elif c.get("kind") == "figure":
            anchor += " · figure analysis"
        text = c["text"] if not max_chars else c["text"][:max_chars]
        parts.append(f"[{anchor}] {text}")
    return "\n\n".join(parts)

# =========================
# PAPER + LIBRARY
# =========================
@dataclass
class Paper:
    id: str
    filename: str
    path: str
    title: str = ""
    text: str = ""
    segments: List[dict] = field(default_factory=list)
    figures: List[dict] = field(default_factory=list)
    tables: List[dict] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    store: VectorStore = field(default_factory=VectorStore)
    figure_analyses: Dict[str, str] = field(default_factory=dict)
    figure_modes: Dict[str, str] = field(default_factory=dict)     # "vision" | "text-only" per analysed figure
    study: Dict[str, object] = field(default_factory=dict)         # cached study material (src/study.py)
    summary: str = ""
    topic: str = ""
    profile: Optional[dict] = None
    captions: List[dict] = field(default_factory=list)
    references: List[dict] = field(default_factory=list)
    pages: int = 0
    cache_hit: bool = False

    def search(self, query: str, k: int = DEFAULT_TOP_K, mode: str = "hybrid", mmr: Optional[bool] = None) -> List[Chunk]:
        return self.store.search(query, k, mode, mmr)

    def sections(self) -> List[str]:
        return list(OrderedDict.fromkeys(s["section"] for s in self.segments))

    def figure(self, figure_id: str) -> Optional[dict]:
        return next((f for f in self.figures if f["id"] == figure_id), None)

    def add_chunk(self, chunk: Chunk) -> int:
        """Add one retrievable chunk after indexing (e.g. a figure analysis). Returns its chunk id.
        A chunk with the same kind + figure_id replaces nothing: the newest one simply ranks alongside."""
        chunk = _with_header(dict(chunk))
        self.store.add_chunks([chunk])
        self.chunks.append(chunk)
        return len(self.chunks) - 1

    def section_chunks(self, name: str) -> List[Chunk]:
        needle = name.lower().strip()
        return [c for c in self.chunks if needle in str(c.get("section", "")).lower()
                and c.get("kind") in ("text", "reference", None)]

    def info(self) -> dict:
        return {"id": self.id, "filename": self.filename, "title": self.title,
                "chunks": len(self.chunks), "figures": len(self.figures), "tables": len(self.tables),
                "references": len(self.references), "has_profile": self.profile is not None}

def dispose_paper(paper: Paper):
    """Delete a paper's files — only those this app created under UPLOAD_DIR (never a user's own PDF)."""
    upload_root = os.path.abspath(UPLOAD_DIR)
    path = os.path.abspath(paper.path) if paper.path else ""
    try:
        ours = bool(path) and os.path.commonpath([upload_root, path]) == upload_root
    except ValueError:      # e.g. different drives on Windows: certainly not inside the upload folder
        ours = False
    if ours and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    shutil.rmtree(os.path.join(FIGURE_DIR, paper.id), ignore_errors=True)

class PaperLibrary:
    """Registry of the papers uploaded in ONE session. The 'active' paper is the one the main pages work on."""
    def __init__(self, max_papers: int = MAX_PAPERS_PER_SESSION):
        self.papers: "OrderedDict[str, Paper]" = OrderedDict()
        self.active_id: Optional[str] = None
        self.max_papers = max_papers
        self.touched = time.time()

    def add(self, paper: Paper, make_active: bool = False) -> Paper:
        self.papers[paper.id] = paper
        if make_active or self.active_id is None:
            self.active_id = paper.id
        # Bound memory + disk: drop the oldest papers that are neither active nor the one just added
        while len(self.papers) > self.max_papers:
            victim = next((pid for pid in self.papers if pid not in (self.active_id, paper.id)), None)
            if victim is None:
                break
            dispose_paper(self.papers.pop(victim))
        return paper

    def get(self, paper_id: Optional[str] = None) -> Optional[Paper]:
        return self.papers.get(paper_id or self.active_id or "")

    def active(self) -> Optional[Paper]:
        return self.get()

    def remove(self, paper_id: str):
        paper = self.papers.pop(paper_id, None)
        if paper is not None:
            dispose_paper(paper)
        if self.active_id == paper_id:
            self.active_id = next(iter(self.papers), None)

    def reset(self):
        for paper in list(self.papers.values()):
            dispose_paper(paper)
        self.papers.clear()
        self.active_id = None

    def list(self) -> List[dict]:
        return [{**p.info(), "active": p.id == self.active_id} for p in self.papers.values()]

# =========================
# SESSIONS (one library per browser)
# =========================
DEFAULT_SESSION = "default"   # used by the CLI, evals and tests (no HTTP session)
MAX_EMPTY_SESSIONS = 64       # paper-less sessions kept around before the oldest are forgotten

class SessionManager:
    """Maps session ids to PaperLibrary objects, evicting idle / least-recently-used sessions and their files."""
    def __init__(self, max_sessions: int = MAX_SESSIONS, ttl: int = SESSION_TTL_SECONDS):
        self.max_sessions, self.ttl = max_sessions, ttl
        self._libraries: "OrderedDict[str, PaperLibrary]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, session_id: str) -> PaperLibrary:
        with self._lock:
            lib = self._libraries.get(session_id)
            if lib is None:
                lib = self._libraries[session_id] = PaperLibrary()
            self._libraries.move_to_end(session_id)
            lib.touched = time.time()
            self._evict(keep=session_id)
            return lib

    def _evict(self, keep: str):
        """Drop expired sessions, then the least recently used sessions that hold papers beyond the cap.

        Sessions without papers (health checks, crawlers, visitors who never upload) cost almost nothing,
        so they never push out a session that holds a paper; only their number is bounded.
        """
        now = time.time()
        evictable = [sid for sid in self._libraries if sid not in (keep, DEFAULT_SESSION)]   # oldest first
        expired = [sid for sid in evictable if now - self._libraries[sid].touched > self.ttl]
        live = [sid for sid in evictable if sid not in expired]
        empty = [sid for sid in live if not self._libraries[sid].papers]
        loaded = [sid for sid in live if self._libraries[sid].papers]
        victims = (expired + empty[:max(0, len(empty) - MAX_EMPTY_SESSIONS)]
                   + loaded[:max(0, len(loaded) - (self.max_sessions - 1))])
        for sid in victims:
            self._libraries.pop(sid).reset()

    def drop(self, session_id: str):
        with self._lock:
            lib = self._libraries.pop(session_id, None)
            if lib is not None:
                lib.reset()

    def count(self) -> int:
        """Number of sessions currently holding at least one paper."""
        return sum(1 for lib in self._libraries.values() if lib.papers)

sessions = SessionManager()
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("scholarai_session", default=DEFAULT_SESSION)

def set_session(session_id: Optional[str]):
    """Bind the current execution context (request / stream / thread) to a session. Returns a reset token."""
    return _session_id.set(session_id or DEFAULT_SESSION)

def current_session_id() -> str:
    return _session_id.get()

@contextmanager
def use_session(session_id: Optional[str]):
    token = set_session(session_id)
    try:
        yield sessions.get(session_id or DEFAULT_SESSION)
    finally:
        _session_id.reset(token)

def current_library() -> PaperLibrary:
    return sessions.get(_session_id.get())

class _LibraryProxy:
    """`library` always resolves to the current session's PaperLibrary, so the rest of the code
    (tools, agents, workflows) stays session-agnostic."""
    def __getattr__(self, name):
        return getattr(current_library(), name)

    def __setattr__(self, name, value):
        setattr(current_library(), name, value)

library = _LibraryProxy()

# =========================
# LOADING + CONVENIENCE SEARCH
# =========================
def load_paper(file_path: str, filename: str = "", paper_id: Optional[str] = None,
               encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
               cache_dir: Optional[str] = EMBED_CACHE_DIR, progress: Optional[ProgressFn] = None) -> Paper:
    """Parse, chunk, and index a PDF into a Paper (not yet registered in the library).

    Embeddings are cached on disk by content hash, so re-uploading the same PDF skips the embedding step.
    """
    paper_id = paper_id or uuid.uuid4().hex[:10]
    progress = progress or _noop
    parsed = parse_document(file_path, figure_dir=os.path.join(FIGURE_DIR, paper_id), progress=progress)
    chunks = build_chunks(parsed)
    store = VectorStore(encoder=encoder)
    index_texts = [str(c.get("embed_text") or c["text"]) for c in chunks]
    cache_hit = False
    if chunks:
        if encoder is None and cache_dir:
            model_id = embedding_model_id()
            cache_hit = os.path.exists(os.path.join(cache_dir, embedding_cache_key(index_texts, model_id) + ".npy"))
            if cache_hit:
                progress("Reusing the search index from an earlier upload of this paper")
            vectors = embed_with_cache(index_texts, lambda texts: encode(texts, progress), model_id, cache_dir)
        else:
            progress(f"Building the search index: {len(index_texts)} passages")
            vectors = encoder(index_texts) if encoder is not None else encode(index_texts, progress)
        store.add_chunks(chunks, embeddings=vectors)
    return Paper(id=paper_id, filename=filename or os.path.basename(file_path), path=file_path,
                 title=parsed["title"], text=parsed["text"], segments=parsed["segments"],
                 figures=parsed["figures"], tables=parsed["tables"], chunks=chunks, store=store,
                 captions=parsed["captions"], references=parsed["references"], pages=parsed["pages"],
                 cache_hit=cache_hit)

def search_chunks(query: str, k: int = DEFAULT_TOP_K, paper_id: Optional[str] = None) -> List[Chunk]:
    """Hybrid search over a paper (defaults to the active paper)."""
    paper = library.get(paper_id)
    return paper.search(query, k) if paper else []

def retrieve(query: str, k: int = DEFAULT_TOP_K) -> List[str]:
    """Retrieve top-k relevant chunk texts from the active paper."""
    return [str(c["text"]) for c in search_chunks(query, k)]

def gather_context(queries: List[str], k: int = 3, paper_id: Optional[str] = None) -> str:
    """Run several targeted queries, de-duplicate, and return anchored context."""
    seen, picked = set(), []
    for q in queries:
        for c in search_chunks(q, k, paper_id):
            if c["text"] not in seen:
                seen.add(c["text"])
                picked.append(c)
    return format_chunks(picked)

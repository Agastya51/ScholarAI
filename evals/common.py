"""Shared helpers for the evaluation scripts."""
import json
import os
import re
import sys
import time
import zlib
from typing import List, Optional

import numpy as np

from src.config import EMBEDDING_DIM
from src.rag import Paper, library, load_paper

RESULTS_DIR = os.path.join("evals", "results")

def safe_stdout():
    """Never crash while printing a report on a console with a legacy code page (e.g. Windows cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

def load_dataset(path: str, limit: Optional[int] = None) -> dict:
    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)
    items = dataset.get("questions", [])
    if not items:
        raise SystemExit(f"{path} has no questions.")
    dataset["questions"] = items[:limit] if limit else items
    return dataset

def hashing_encoder(texts: List[str]) -> np.ndarray:
    """Toy bag-of-words hashing embedder for --offline smoke tests (no model download).
    It has no notion of meaning, so scores produced with it say nothing about real retrieval quality."""
    out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(tok.encode()) % EMBEDDING_DIM] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out

def load_eval_paper(pdf_path: str, offline: bool = False, cache: bool = True) -> Paper:
    """Index the PDF exactly the way the app does and make it the active paper."""
    if not os.path.exists(pdf_path):
        raise SystemExit(f"Paper not found: {pdf_path}")
    if offline:
        print("[offline] Using a toy hashing embedder: this only checks that the script runs. "
              "The numbers below are NOT a measure of real retrieval quality.")
    library.reset()
    kwargs = {} if cache else {"cache_dir": None}     # cache=False: time the real embedding work
    return library.add(load_paper(pdf_path, encoder=hashing_encoder if offline else None, **kwargs), make_active=True)

def save_report(prefix: str, payload: dict, markdown: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = os.path.join(RESULTS_DIR, f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}")
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    with open(stem + ".md", "w", encoding="utf-8") as f:
        f.write(markdown)
    return stem

def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

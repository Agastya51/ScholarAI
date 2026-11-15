"""Test doubles: a scripted fake LLM backend and a deterministic embedding encoder (no network, no model download)."""
import json
import re
import zlib

import numpy as np

from src.config import EMBEDDING_DIM
from src.rag import Paper, VectorStore, chunk_segments, library

def hash_encoder(texts):
    """Bag-of-words hashing embedding: deterministic and good enough for retrieval tests."""
    out = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"[a-z0-9]+", t.lower()):
            out[i, zlib.crc32(tok.encode()) % EMBEDDING_DIM] += 1.0
        norm = np.linalg.norm(out[i])
        if norm:
            out[i] /= norm
    return out

def schema_name(messages) -> str:
    """Name of the Pydantic schema requested by src.llm.structured(), or '' for free-text calls."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "") or ""
    names = re.findall(r'"title": "([A-Za-z]+)"', system)
    return names[-1] if names else ""

def last_user(messages) -> str:
    return next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

def tool_call(name, call_id="call_1", **args):
    return {"content": "", "tool_calls": [{"id": call_id, "name": name, "arguments": args}]}

class FakeLLM:
    """Backend whose responses come from `responder(messages, tools, json_mode)`.

    The responder may return a str, a dict ({"content", "tool_calls"}), or a pydantic-compatible dict
    for structured calls (it is JSON-encoded automatically when json_mode is on).
    """
    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def __call__(self, messages, tools=None, json_mode=False, max_tokens=0):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode,
                           "schema": schema_name(messages) if json_mode else ""})
        out = self.responder(messages, tools, json_mode)
        if json_mode and not (isinstance(out, dict) and "content" in out):
            out = json.dumps(out)
        if isinstance(out, str):
            out = {"content": out, "tool_calls": []}
        return out

SAMPLE_SEGMENTS = [
    {"page": 1, "section": "Abstract", "text": "We propose FlowNet-X, a graph neural network for traffic "
     "forecasting. FlowNet-X achieves 94.3% accuracy on the METR-LA dataset."},
    {"page": 2, "section": "3. Method", "text": "FlowNet-X stacks diffusion convolution layers with a gated "
     "temporal attention block. The learning rate is 0.001 and batch size is 64."},
    {"page": 5, "section": "5. Results", "text": "Compared with DCRNN, FlowNet-X reduces MAE by 12.5% on "
     "METR-LA and by 8.1% on PEMS-BAY."},
    {"page": 6, "section": "6. Conclusion", "text": "Limitations include the quadratic cost of attention on "
     "large road networks. Future work will study sparse attention."},
]

def make_paper(paper_id="p1", title="FlowNet-X: Traffic Forecasting", segments=None, make_active=True) -> Paper:
    segments = segments or SAMPLE_SEGMENTS
    chunks = chunk_segments(segments)
    store = VectorStore(encoder=hash_encoder)
    store.add_chunks(chunks)
    paper = Paper(id=paper_id, filename=f"{paper_id}.pdf", path=f"{paper_id}.pdf", title=title,
                  text="\n".join(s["text"] for s in segments), segments=segments, chunks=chunks, store=store)
    return library.add(paper, make_active=make_active)

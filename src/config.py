import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("true", "1", "t", "yes", "on")

# Server Configuration
FLASK_DEBUG = _flag("FLASK_DEBUG", False)
PORT = int(os.environ.get("PORT", 5000))

# LLM Models
# Primary model with automatic fallback on 429 rate limit
TEXT_MODELS = [
    "openai/gpt-oss-120b",        # primary
    "llama-3.3-70b-versatile",    # fallback
]
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MAX_TOKENS = 2048

# Vector Search / Embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"      # FastEmbed (ONNX), 384-dim
FALLBACK_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # only if fastembed is unavailable
EMBEDDING_DIM = 384
# bge models are trained with this instruction on the *query* side for passage retrieval.
QUERY_INSTRUCTION = os.environ.get(
    "QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: ")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", 4))
RRF_K = 60  # Reciprocal Rank Fusion constant for hybrid (BM25 + FAISS) search

# Retrieval pipeline (see src/retrieval.py). Every switch can be measured with `python -m evals.retrieval_eval`.
CANDIDATE_POOL = 20                  # candidates fetched per query before fusion / reranking / MMR
USE_MMR = _flag("USE_MMR", True)     # Maximal Marginal Relevance: trade relevance against redundancy
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", 0.7))   # 1.0 = pure relevance, 0.0 = pure diversity
# Cosine-similarity thresholds (query vs chunk). Calibrate for your papers with evals.retrieval_eval,
# which prints the score distribution of relevant vs irrelevant chunks.
MIN_RELEVANCE = float(os.environ.get("MIN_RELEVANCE", 0.40))     # below this (and no keyword hit) a chunk is dropped
HIGH_CONFIDENCE = float(os.environ.get("HIGH_CONFIDENCE", 0.60))  # top chunk above this => "high" confidence
# Query expansion (multi-query rewrite + HyDE). "auto" = only when the first pass has low confidence
# (costs one extra LLM call), "always", or "off".
QUERY_EXPANSION = os.environ.get("QUERY_EXPANSION", "auto").lower()
# Reranker for the fused candidates: "none" (default), "cross-encoder" (fastembed ONNX ms-marco MiniLM;
# loads a second model, so check memory before enabling it on a 512 MB instance), or "llm" (one extra
# listwise LLM call per question).
RERANK_MODE = os.environ.get("RERANK_MODE", "none").lower()
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = 12                    # how many fused candidates the reranker sees

# Agent Settings
# Hard caps keep agent loops from burning through Groq free-tier rate limits.
MAX_LLM_CALLS_PER_RUN = int(os.environ.get("MAX_LLM_CALLS_PER_RUN", 40))
MAX_LLM_CALLS_ANALYZE = int(os.environ.get("MAX_LLM_CALLS_ANALYZE", 90))  # full multi-agent /analyze run
QA_AGENT_MAX_ITERATIONS = 5          # tool-calling rounds before the QA agent must answer
SEARCH_MAX_ATTEMPTS = 3              # literature search + reformulation attempts
SEARCH_MIN_RELEVANT = 3              # relevant papers needed to stop reformulating
REWRITE_MAX_REVISIONS = 2            # evaluator-optimizer revision rounds for rewrites
SUPERVISOR_MAX_STEPS = 10            # supervisor routing decisions per analysis run
MAX_FIGURES_ANALYZED = 3
REVIEWERS = ["methodology", "novelty", "reproducibility", "clarity"]

# PDF Generator Layout
PAGE_W = 595
PAGE_H = 842
MARGIN = 55
TW = PAGE_W - 2 * MARGIN

# Prestigious Engineering Publishers for Search Reranking Bonus
ENGINEERING_PUBLISHERS = {
    "ieee", "springer", "elsevier", "wiley", "acm", "nature", "taylor",
    "iet", "inspec", "iospress", "hindawi", "mdpi", "sage", "emerald"
}

# Upload Paths
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
FIGURE_DIR = os.path.join(UPLOAD_DIR, "figures")
EMBED_CACHE_DIR = os.path.join(UPLOAD_DIR, "cache")   # embeddings keyed by content hash
EMBED_CACHE_MAX_FILES = 30

# Upload limits (protect a 512 MB instance)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 25))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 60))

# Sessions: every browser gets its own paper library so users never see or overwrite each other's papers.
SESSION_ISOLATION = _flag("SESSION_ISOLATION", True)
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", 8))                 # least-recently-used sessions are evicted
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 3 * 3600))
MAX_PAPERS_PER_SESSION = int(os.environ.get("MAX_PAPERS_PER_SESSION", 6))

import base64
import contextvars
import json
import re
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src.config import GROQ_API_KEY, TEXT_MODELS, VISION_MODEL, LLM_MAX_TOKENS, MAX_LLM_CALLS_PER_RUN

T = TypeVar("T", bound=BaseModel)

# =========================
# ERRORS
# =========================
class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a response (instead of returning an error string)."""

class LLMRateLimitError(LLMError):
    """All fallback models are rate-limited."""

class LLMBudgetExceeded(LLMError):
    """The per-run LLM call budget was exhausted (protects against runaway agent loops)."""

# =========================
# USAGE TRACKING (per run)
# =========================
# A mutable dict inside a ContextVar so parallel LangGraph branches (which copy the context)
# all increment the same counter.
_usage: contextvars.ContextVar[Optional[Dict[str, int]]] = contextvars.ContextVar("llm_usage", default=None)

@contextmanager
def track_usage(budget: int = MAX_LLM_CALLS_PER_RUN):
    """Count LLM calls for one request/run and enforce a call budget."""
    usage = {"calls": 0, "budget": budget, "tokens": 0}
    token = _usage.set(usage)
    try:
        yield usage
    finally:
        _usage.reset(token)

def _count_call(tokens: int = 0):
    usage = _usage.get()
    if usage is None:
        return
    if usage["calls"] >= usage["budget"]:
        raise LLMBudgetExceeded(
            f"LLM call budget of {usage['budget']} exhausted for this run. "
            "Raise MAX_LLM_CALLS_PER_RUN or simplify the request."
        )
    usage["calls"] += 1
    usage["tokens"] += tokens

# =========================
# BACKENDS
# =========================
# A backend takes (messages, tools, json_mode, max_tokens) and returns
# {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}.
Backend = Callable[[List[dict], Optional[List[dict]], bool, int], Dict[str, Any]]

_client = None

def get_client():
    """Lazy initialize Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Please set it in your .env file.")
        from groq import Groq
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def _is_retryable(err: str) -> bool:
    e = err.lower()
    return "429" in e or "rate_limit" in e or "tool_use_failed" in e or "503" in e or "over capacity" in e

def groq_backend(messages, tools=None, json_mode=False, max_tokens=LLM_MAX_TOKENS):
    """Call Groq with automatic model fallback on rate-limit (429) and tool-call failures."""
    client = get_client()
    last_err = ""
    for model in TEXT_MODELS:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            res = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = str(e)
            if _is_retryable(last_err):
                print(f"[LLM fallback] {model} failed ({last_err[:80]}), trying next model...")
                time.sleep(2)
                continue
            raise LLMError(f"{model}: {e}") from e

        msg = res.choices[0].message
        tool_calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        tokens = getattr(getattr(res, "usage", None), "total_tokens", 0) or 0
        return {"content": msg.content or "", "tool_calls": tool_calls, "tokens": tokens}

    raise LLMRateLimitError(
        f"All models are currently rate-limited or unavailable. Please wait a few minutes. ({last_err[:120]})"
    )

_backend: Backend = groq_backend

def set_backend(backend: Optional[Backend]):
    """Swap the LLM backend (used by tests to inject a fake LLM). Pass None to restore Groq."""
    global _backend
    _backend = backend or groq_backend

# =========================
# PUBLIC API
# =========================
def chat(messages: List[dict], tools: Optional[List[dict]] = None,
         json_mode: bool = False, max_tokens: int = LLM_MAX_TOKENS) -> Dict[str, Any]:
    """Multi-turn chat with optional tool calling. Returns {"content", "tool_calls"}; raises LLMError."""
    _count_call()
    result = _backend(messages, tools, json_mode, max_tokens)
    usage = _usage.get()
    if usage is not None:
        usage["tokens"] += result.get("tokens", 0) or 0
    result.setdefault("tool_calls", [])
    result["content"] = result.get("content") or ""
    return result

def llm(prompt: str, system: Optional[str] = None, max_tokens: int = LLM_MAX_TOKENS) -> str:
    """Single-turn text completion. Raises LLMError on failure."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return chat(messages, max_tokens=max_tokens)["content"]

def extract_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    raise LLMError(f"Model did not return valid JSON: {cleaned[:200]}")

def structured(prompt: str, schema: Type[T], system: Optional[str] = None,
               max_tokens: int = LLM_MAX_TOKENS, retries: int = 1) -> T:
    """Get a response validated against a Pydantic schema (JSON mode + one repair retry)."""
    schema_json = json.dumps(schema.model_json_schema())
    sys_msg = (system + "\n\n" if system else "") + (
        "Respond ONLY with a JSON object that matches this JSON schema:\n" + schema_json
    )
    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
    last_err: Exception = LLMError("no attempt made")
    for _ in range(retries + 1):
        raw = chat(messages, json_mode=True, max_tokens=max_tokens)["content"]
        try:
            return schema.model_validate(extract_json(raw))
        except (LLMError, ValidationError) as e:
            last_err = e
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON was invalid ({str(e)[:300]}). Return corrected JSON only."},
            ]
    raise LLMError(f"Structured output failed for {schema.__name__}: {last_err}")

# =========================
# VISION
# =========================
def encode_image(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}

VisionBackend = Callable[[str, str, int], str]   # (image path, prompt, max_tokens) -> text
_vision_backend: Optional[VisionBackend] = None

def set_vision_backend(backend: Optional[VisionBackend]):
    """Inject a vision backend (tests / offline demos). Pass None to restore Groq."""
    global _vision_backend
    _vision_backend = backend

def vision(img_path: str, prompt: str, max_tokens: int = 900) -> str:
    """One multimodal call: an image plus a text prompt. Raises LLMError if the model cannot be reached,
    so callers can fall back instead of showing (or caching) an error string as if it were an analysis."""
    _count_call()
    if _vision_backend is not None:
        return _vision_backend(img_path, prompt, max_tokens)
    try:
        client = get_client()
        mime = _MIME.get(img_path.rsplit(".", 1)[-1].lower(), "image/png")
        res = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encode_image(img_path)}"}},
            ]}],
        )
        text = (res.choices[0].message.content or "").strip()
    except LLMError:
        raise
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            raise LLMRateLimitError(f"The vision model is rate-limited right now: {err[:160]}") from e
        raise LLMError(f"The vision model could not read the image: {err[:200]}") from e
    if not text:
        raise LLMError("The vision model returned an empty answer.")
    return text

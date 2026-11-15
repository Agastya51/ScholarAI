import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import BaseModel

from src import llm as llm_mod
from src.llm import (
    LLMError, LLMBudgetExceeded, LLMRateLimitError, chat, extract_json, llm, set_backend, structured, track_usage
)
from tests.fakes import FakeLLM

class Answer(BaseModel):
    value: int

def _groq_response(content="ok", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=SimpleNamespace(total_tokens=10))

class TestLLMLayer(unittest.TestCase):
    def tearDown(self):
        set_backend(None)

    def test_extract_json_tolerates_fences_and_prose(self):
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! Here it is: [1, 2] done'), [1, 2])
        with self.assertRaises(LLMError):
            extract_json("no json here")

    def test_structured_repairs_invalid_json_once(self):
        replies = iter(["not json", {"value": 7}])
        fake = FakeLLM(lambda m, t, j: next(replies))
        set_backend(fake)
        self.assertEqual(structured("give value", Answer).value, 7)
        self.assertEqual(len(fake.calls), 2)
        self.assertTrue(all(c["json_mode"] for c in fake.calls))

    def test_budget_is_enforced(self):
        set_backend(FakeLLM(lambda m, t, j: "hi"))
        with track_usage(budget=2) as usage:
            llm("a")
            llm("b")
            with self.assertRaises(LLMBudgetExceeded):
                llm("c")
        self.assertEqual(usage["calls"], 2)

    def test_groq_backend_falls_back_on_rate_limit(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs["model"])
            if len(calls) == 1:
                raise Exception("Error code: 429 rate_limit_exceeded")
            return _groq_response("fallback answer")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm_mod, "get_client", return_value=client), patch.object(llm_mod.time, "sleep"):
            self.assertEqual(llm("q"), "fallback answer")
        self.assertEqual(calls, llm_mod.TEXT_MODELS[:2])

    def test_groq_backend_raises_instead_of_returning_error_text(self):
        def create(**kwargs):
            raise Exception("400 invalid request")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm_mod, "get_client", return_value=client):
            with self.assertRaises(LLMError):
                llm("q")

    def test_all_models_rate_limited(self):
        def create(**kwargs):
            raise Exception("429 rate_limit")

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(llm_mod, "get_client", return_value=client), patch.object(llm_mod.time, "sleep"):
            with self.assertRaises(LLMRateLimitError):
                llm("q")

    def test_tool_calls_are_parsed(self):
        tc = SimpleNamespace(id="c1", function=SimpleNamespace(name="search_paper", arguments='{"query": "MAE"}'))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: _groq_response("", [tc]))))
        with patch.object(llm_mod, "get_client", return_value=client):
            res = chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
        self.assertEqual(res["tool_calls"], [{"id": "c1", "name": "search_paper", "arguments": {"query": "MAE"}}])

if __name__ == "__main__":
    unittest.main()


class TestKeyAndConnectionErrors(unittest.TestCase):
    def test_pasted_key_is_cleaned(self):
        import importlib
        import os
        import src.config as config
        saved = os.environ.get("GROQ_API_KEY")
        try:
            for raw in ["gsk_abc\n", " gsk_abc ", '"gsk_abc"', "'gsk_abc'\r\n"]:
                os.environ["GROQ_API_KEY"] = raw
                importlib.reload(config)
                self.assertEqual(config.GROQ_API_KEY, "gsk_abc", repr(raw))
        finally:
            if saved is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = saved
            importlib.reload(config)

    def test_connection_error_names_the_cause(self):
        from unittest.mock import MagicMock, patch
        from src import llm as llm_module

        class APIConnectionError(Exception):
            pass

        err = APIConnectionError("Connection error.")
        err.__cause__ = ValueError("Illegal header value b'Bearer gsk_abc\\n'")
        client = MagicMock()
        client.chat.completions.create.side_effect = err
        with patch.object(llm_module, "get_client", return_value=client):
            with self.assertRaises(llm_module.LLMError) as ctx:
                llm_module.groq_backend([{"role": "user", "content": "hi"}])
        self.assertIn("Could not connect to Groq", str(ctx.exception))
        self.assertIn("Illegal header value", str(ctx.exception))

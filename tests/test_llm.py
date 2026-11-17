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


class TestReasoningBudget(unittest.TestCase):
    """gpt-oss reasons before answering; that must not eat the answer's budget (json_validate_failed)."""

    def _client(self, outcomes):
        from unittest.mock import MagicMock
        client = MagicMock()
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            msg = MagicMock(content=out, tool_calls=[])
            return MagicMock(choices=[MagicMock(message=msg)], usage=MagicMock(total_tokens=10))
        client.chat.completions.create.side_effect = create
        return client, calls

    def test_reasoning_model_gets_headroom_and_low_effort_for_json(self):
        from unittest.mock import patch
        from src import llm as m
        client, calls = self._client(['{"a": 1}'])
        with patch.object(m, "get_client", return_value=client):
            m.groq_backend([{"role": "user", "content": "x"}], json_mode=True, max_tokens=2400)
        self.assertEqual(calls[0]["model"], "openai/gpt-oss-120b")
        self.assertEqual(calls[0]["max_tokens"], 2400 + m.REASONING_HEADROOM)
        self.assertEqual(calls[0]["extra_body"], {"reasoning_effort": "low"})

    def test_cut_off_json_is_retried_bigger_then_falls_back(self):
        from unittest.mock import patch
        from src import llm as m
        cut = Exception("Error code: 400 - {'error': {'code': 'json_validate_failed', 'failed_generation': "
                        "'max completion tokens reached before generating a valid document'}}")
        client, calls = self._client([cut, cut, '{"ok": true}'])
        with patch.object(m, "get_client", return_value=client), patch.object(m.time, "sleep"):
            out = m.groq_backend([{"role": "user", "content": "x"}], json_mode=True, max_tokens=1000)
        self.assertEqual(out["content"], '{"ok": true}')
        self.assertEqual([c["model"] for c in calls], ["openai/gpt-oss-120b", "openai/gpt-oss-120b",
                                                        "llama-3.3-70b-versatile"])
        self.assertEqual(calls[1]["max_tokens"], 2000 + m.REASONING_HEADROOM)
        self.assertNotIn("extra_body", calls[2])

    def test_other_errors_are_not_retried(self):
        from unittest.mock import patch
        from src import llm as m
        client, calls = self._client([Exception("Error code: 400 - invalid model")])
        with patch.object(m, "get_client", return_value=client):
            with self.assertRaises(m.LLMError):
                m.groq_backend([{"role": "user", "content": "x"}])
        self.assertEqual(len(calls), 1)

    def test_request_fits_under_the_free_plan_token_limit(self):
        from src import llm as m
        prompt = [{"role": "system", "content": "s" * 2000}, {"role": "user", "content": "w" * 12000}]   # ~4,100 tokens
        req = m._request("openai/gpt-oss-120b", prompt, None, True, 2400)
        self.assertLessEqual(m._prompt_tokens(prompt) + req["max_tokens"], m.GROQ_TPM_LIMIT)
        self.assertGreaterEqual(req["max_tokens"], 2400 + 300)        # the answer keeps its budget

    def test_study_context_is_capped_on_a_passage_boundary(self):
        from unittest.mock import patch
        from src import study
        long = "\n\n".join(f"[p. {i}] " + "word " * 300 for i in range(20))
        with patch.object(study, "gather_context", return_value=long):
            ctx = study.study_context(type("P", (), {"id": "x"})())
        self.assertLessEqual(len(ctx), study.STUDY_CONTEXT_CHARS)
        self.assertTrue(ctx.rstrip().endswith("word"))


class TestOpenAIProvider(unittest.TestCase):
    """With OPENAI_API_KEY set, OpenAI answers first; Groq is the backup."""

    def _client(self, outcomes):
        from unittest.mock import MagicMock
        client = MagicMock()
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            content, finish = out if isinstance(out, tuple) else (out, "stop")
            msg = MagicMock(content=content, tool_calls=[])
            return MagicMock(choices=[MagicMock(message=msg, finish_reason=finish)], usage=MagicMock(total_tokens=5))
        client.chat.completions.create.side_effect = create
        return client, calls

    def test_openai_is_used_first_with_reasoning_room(self):
        from unittest.mock import patch
        from src import llm as m
        client, calls = self._client(['{"a": 1}'])
        with patch.object(m, "OPENAI_API_KEY", "sk-test"), patch.object(m, "GROQ_API_KEY", "gsk"), \
                patch.object(m, "LLM_PROVIDER", "auto"), patch.object(m, "get_openai_client", return_value=client), \
                patch.object(m, "groq_backend", side_effect=AssertionError("groq must not be called")):
            out = m.default_backend([{"role": "user", "content": "x"}], json_mode=True, max_tokens=2400)
            self.assertEqual(m.active_provider(), "openai")
        self.assertEqual(out["content"], '{"a": 1}')
        self.assertEqual(calls[0]["model"], "gpt-5-mini")
        self.assertEqual(calls[0]["max_completion_tokens"], 2400 + m.OPENAI_REASONING_HEADROOM)
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("max_tokens", calls[0])

    def test_cut_off_answer_is_retried_with_more_room(self):
        from unittest.mock import patch
        from src import llm as m
        client, calls = self._client([("", "length"), "done"])
        with patch.object(m, "OPENAI_API_KEY", "sk-test"), patch.object(m, "get_openai_client", return_value=client):
            out = m.openai_backend([{"role": "user", "content": "x"}], max_tokens=1000)
        self.assertEqual(out["content"], "done")
        self.assertEqual(calls[1]["max_completion_tokens"], 2000 + m.OPENAI_REASONING_HEADROOM)

    def test_no_credit_falls_back_to_groq(self):
        from unittest.mock import patch
        from src import llm as m
        client, calls = self._client([Exception("Error code: 429 - {'error': {'code': 'insufficient_quota'}}")])
        with patch.object(m, "OPENAI_API_KEY", "sk-test"), patch.object(m, "GROQ_API_KEY", "gsk"), \
                patch.object(m, "LLM_PROVIDER", "auto"), patch.object(m, "get_openai_client", return_value=client), \
                patch.object(m, "groq_backend", return_value={"content": "from groq", "tool_calls": []}):
            out = m.default_backend([{"role": "user", "content": "x"}])
        self.assertEqual(out["content"], "from groq")
        self.assertEqual(len(calls), 1)                 # another OpenAI model would not have helped

    def test_without_openai_key_groq_is_used(self):
        from unittest.mock import patch
        from src import llm as m
        with patch.object(m, "OPENAI_API_KEY", ""), patch.object(m, "GROQ_API_KEY", "gsk"), \
                patch.object(m, "LLM_PROVIDER", "auto"), \
                patch.object(m, "groq_backend", return_value={"content": "groq", "tool_calls": []}):
            self.assertEqual(m.active_provider(), "groq")
            self.assertEqual(m.default_backend([{"role": "user", "content": "x"}])["content"], "groq")

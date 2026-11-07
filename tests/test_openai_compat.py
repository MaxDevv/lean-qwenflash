"""OpenAI reasoning controls must reach the Qwen3.8 template as values it accepts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen38-flash-offload" / "qwen38"))
from openai_compat import normalize_chat_request  # noqa: E402


class ReasoningEffortTests(unittest.TestCase):
    def check(self, body, expected):
        self.assertEqual(normalize_chat_request(body), expected)

    def test_openai_levels_map_to_template_levels(self):
        self.check({"reasoning_effort": "minimal"}, {"reasoning_effort": "low"})
        self.check({"reasoning_effort": "medium"}, {"reasoning_effort": "medium"})
        self.check({"reasoning_effort": "high"}, {"reasoning_effort": "xhigh"})
        self.check({"reasoning_effort": "MAX"}, {"reasoning_effort": "xhigh"})

    def test_off_disables_thinking(self):
        self.check({"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}})
        self.check({"reasoning_effort": "off", "chat_template_kwargs": None},
                   {"chat_template_kwargs": {"enable_thinking": False}})

    def test_unknown_level_is_dropped(self):
        self.check({"reasoning_effort": "turbo"}, {})

    def test_template_kwargs_and_responses_style(self):
        self.check({"chat_template_kwargs": {"reasoning_effort": "high"}},
                   {"chat_template_kwargs": {"reasoning_effort": "xhigh"}})
        self.check({"reasoning": {"effort": "low"}}, {"reasoning": {"effort": "low"}, "reasoning_effort": "low"})

    def test_requests_without_reasoning_are_untouched(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
        self.check(dict(body), body)


if __name__ == "__main__":
    unittest.main()

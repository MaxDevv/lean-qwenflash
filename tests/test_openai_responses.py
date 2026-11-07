"""The Responses API translation: requests onto the chat pipeline, chat deltas back into Responses events."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen38-flash-offload" / "qwen38"))
import openai_responses as R  # noqa: E402

CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'}}


def events(sse):
    out = []
    for block in "".join(sse).strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        data = json.loads(lines["data"])
        assert lines["event"] == data["type"], block
        out.append(data)
    return out


class RequestTests(unittest.TestCase):
    def setUp(self):
        R._STORE.clear()

    def test_string_input_and_instructions(self):
        chat, convo = R.responses_to_chat({"model": "m", "input": "Hi", "instructions": "Be brief.",
                                           "max_output_tokens": 500, "reasoning": {"effort": "low"},
                                           "temperature": 0.5, "stream": True})
        self.assertEqual(chat["messages"], [{"role": "system", "content": "Be brief."},
                                            {"role": "user", "content": "Hi"}])
        self.assertEqual((chat["max_tokens"], chat["reasoning_effort"], chat["temperature"], chat["stream"]),
                         (500, "low", 0.5, True))
        self.assertEqual(convo, [{"role": "user", "content": "Hi"}])

    def test_developer_messages_merge_into_one_leading_system_message(self):
        chat, _ = R.responses_to_chat({"instructions": "A", "input": [
            {"role": "developer", "content": [{"type": "input_text", "text": "B"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Q"}]}]})
        self.assertEqual(chat["messages"], [{"role": "system", "content": "A\n\nB"}, {"role": "user", "content": "Q"}])

    def test_developer_messages_later_on_join_the_system_message(self):
        # the template raises "System message must be at the beginning", e.g. for a developer message that comes
        # after earlier turns or with previous_response_id
        chat, convo = R.responses_to_chat({"instructions": "A", "input": [
            {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "R1"},
            {"role": "developer", "content": "B"}, {"role": "user", "content": "Q2"}]})
        self.assertEqual(chat["messages"], [{"role": "system", "content": "A\n\nB"}, {"role": "user", "content": "Q1"},
                                            {"role": "assistant", "content": "R1"}, {"role": "user", "content": "Q2"}])
        self.assertNotIn("system", [m["role"] for m in convo])

    def test_tool_round_trip_items_become_chat_messages(self):
        chat, _ = R.responses_to_chat({"input": [
            {"role": "user", "content": "Weather in Oslo?"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city": "Oslo"}'},
            {"type": "function_call_output", "call_id": "call_1", "output": "12 C, rain"}],
            "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                      {"type": "web_search"}],
            "tool_choice": {"type": "function", "name": "get_weather"}})
        self.assertEqual(chat["messages"][1:], [
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "call_1", "content": "12 C, rain"}])
        # function tools in chat's nested shape; built-in tools are left out
        self.assertEqual(chat["tools"], [{"type": "function", "function": {"name": "get_weather",
                                                                          "parameters": {"type": "object"}}}])
        self.assertEqual(chat["tool_choice"], {"type": "function", "function": {"name": "get_weather"}})

    def test_images_and_unknown_items_are_refused(self):
        with self.assertRaises(R.RequestError):
            R.responses_to_chat({"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}]})
        with self.assertRaises(R.RequestError):
            R.responses_to_chat({"input": [{"type": "computer_call_output"}]})
        with self.assertRaises(R.RequestError):
            R.responses_to_chat({"input": "Hi", "previous_response_id": "resp_missing"})

    def test_previous_response_id_continues_the_conversation(self):
        body = {"input": "My name is Ada.", "instructions": "Be brief."}
        chat, convo = R.responses_to_chat(body)
        first = R.ResponseBuilder(body, "m", convo).whole(None, "Hi Ada.", [], "stop", 10, 3)
        chat, _ = R.responses_to_chat({"input": "What is my name?", "previous_response_id": first["id"]})
        self.assertEqual(chat["messages"], [{"role": "user", "content": "My name is Ada."},
                                            {"role": "assistant", "content": "Hi Ada."},
                                            {"role": "user", "content": "What is my name?"}])
        # store=false keeps nothing
        nostore = R.ResponseBuilder({"input": "x", "store": False}, "m", []).whole(None, "y", [], "stop", 1, 1)
        self.assertNotIn(nostore["id"], R._STORE)


class OutputTests(unittest.TestCase):
    def test_whole_response(self):
        r = R.ResponseBuilder({"input": "x"}, "m", []).whole("thinking", "answer", [CALL], "tool_calls", 7, 5)
        self.assertEqual([o["type"] for o in r["output"]], ["reasoning", "message", "function_call"])
        self.assertEqual(r["output"][0]["content"], [{"type": "reasoning_text", "text": "thinking"}])
        self.assertEqual(r["output"][1]["content"][0]["text"], "answer")
        self.assertEqual((r["output"][2]["call_id"], r["output"][2]["name"]), ("call_1", "get_weather"))
        self.assertEqual((r["status"], r["usage"]["input_tokens"], r["usage"]["output_tokens"]), ("completed", 7, 5))
        cut = R.ResponseBuilder({"input": "x"}, "m", []).whole("thin", "", [], "length", 7, 5)
        self.assertEqual((cut["status"], cut["incomplete_details"]), ("incomplete", {"reason": "max_output_tokens"}))

    def test_stream_events(self):
        b = R.ResponseBuilder({"input": "x", "stream": True}, "m", [])
        sse = b.start()
        for ev in [("reasoning", "Let me "), ("reasoning", "think."), ("content", "Hel"), ("content", "lo"),
                   ("tool_calls", [CALL]), ("end", "tool_calls", 9)]:
            sse += b.feed(ev, 4)
        ev = events(sse)
        self.assertEqual([e["sequence_number"] for e in ev], list(range(len(ev))))
        self.assertEqual([e["type"] for e in ev], [
            "response.created", "response.in_progress",
            "response.output_item.added", "response.content_part.added",
            "response.reasoning_text.delta", "response.reasoning_text.delta", "response.reasoning_text.done",
            "response.content_part.done", "response.output_item.done",
            "response.output_item.added", "response.content_part.added",
            "response.output_text.delta", "response.output_text.delta", "response.output_text.done",
            "response.content_part.done", "response.output_item.done",
            "response.output_item.added", "response.function_call_arguments.delta",
            "response.function_call_arguments.done", "response.output_item.done",
            "response.completed"])
        done = ev[-1]["response"]
        self.assertEqual([o["type"] for o in done["output"]], ["reasoning", "message", "function_call"])
        self.assertEqual(done["output"][1]["content"][0]["text"], "Hello")
        self.assertEqual(done["usage"]["output_tokens"], 9)
        # output_index points at the item each event belongs to
        self.assertEqual({e["output_index"] for e in ev if e["type"] == "response.output_text.delta"}, {1})
        # usage comes after the output in the final event, so it can be read from the stream's tail
        tail = sse[-1]
        self.assertGreater(tail.rfind('"usage"'), tail.rfind('"output"'))

    def test_stream_cut_off_is_incomplete(self):
        b = R.ResponseBuilder({"input": "x", "stream": True}, "m", [])
        ev = events(b.start() + b.feed(("reasoning", "long thought"), 3) + b.feed(("end", "length", 100), 3))
        self.assertEqual(ev[-1]["type"], "response.incomplete")
        self.assertEqual(ev[-1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})


if __name__ == "__main__":
    unittest.main()

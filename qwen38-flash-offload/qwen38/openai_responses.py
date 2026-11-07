"""OpenAI Responses API (/v1/responses) on top of the chat pipeline.

responses_to_chat() turns a Responses request into a chat request: input items become messages, instructions
the system message, function tools chat tools, max_output_tokens max_tokens, reasoning.effort reasoning_effort.
ResponseBuilder turns the chat pipeline's deltas into Responses server-sent events, or builds a whole Response.

- Text only. Function tools are passed to the model. tool_choice "none" leaves them out; a named function or
  "required" is passed on but not enforced, as in /v1/chat/completions (the chat pipeline has no constrained
  decoding), so the model may answer without a call. Built-in tools (web search, file search, computer use, MCP,
  custom/freeform tools) are not available, so they are left out.
- The model's thinking comes back as a reasoning item with reasoning_text content (the gpt-oss/vLLM convention).
  Reasoning items sent back as input are skipped: the chat template drops earlier thinking anyway.
- previous_response_id works for the last STORE_MAX responses while the server runs (in memory; store=false
  skips it). Clients that send the whole conversation each time (store=false) don't need it.
- text.format (structured outputs) and include are accepted and ignored. usage doesn't count reasoning tokens
  separately (reasoning_tokens is 0; they are in output_tokens).
- The chat template takes one system message, first: instructions and every system or developer message, wherever
  it comes in the conversation, are joined into it in order.
"""
import json
import time
import uuid
from collections import OrderedDict

STORE_MAX = 64
_STORE = OrderedDict()   # response id -> chat messages of that conversation, the reply included


class RequestError(ValueError):
    """An invalid or unsupported Responses request (HTTP 400)."""


def _id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _text(content):
    """Text of a message's content: a string, or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, str):
            out.append(part)
            continue
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text", "summary_text", "reasoning_text"):
            out.append(part.get("text") or "")
        elif kind == "refusal":
            out.append(part.get("refusal") or "")
        elif kind in ("input_image", "input_file", "input_audio", "image_url"):
            raise RequestError(f"{kind} is not supported: this server takes text only")
    return "".join(out)


def _messages(items):
    msgs = []
    for it in items:
        if isinstance(it, str):
            msgs.append({"role": "user", "content": it})
            continue
        kind = it.get("type", "message")
        if kind == "message":
            role = it.get("role", "user")
            msgs.append({"role": "system" if role == "developer" else role, "content": _text(it.get("content"))})
        elif kind == "function_call":
            call = {"id": it.get("call_id") or it.get("id") or _id("call"), "type": "function",
                    "function": {"name": it.get("name"), "arguments": it.get("arguments") or "{}"}}
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1].setdefault("tool_calls", []).append(call)   # same turn as the text before it
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            out = it.get("output")
            msgs.append({"role": "tool", "tool_call_id": it.get("call_id"),
                         "content": out if isinstance(out, str) else _text(out)})
        elif kind in ("reasoning", "item_reference"):
            continue
        else:
            raise RequestError(f"input item type {kind!r} is not supported (text messages, function_call and "
                               f"function_call_output are)")
    return msgs


def _tools(tools):
    out = []
    for t in tools or []:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {k: t[k] for k in ("name", "description", "parameters") if t.get(k) is not None}
        out.append({"type": "function", "function": fn})
    return out


def _tool_choice(choice):
    if choice is None or isinstance(choice, str):
        return choice
    if choice.get("type") == "function" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return "auto"


def responses_to_chat(body):
    """(chat request, conversation to remember) for a /v1/responses request. Raises RequestError."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    inp = body.get("input")
    items = [{"role": "user", "content": inp}] if isinstance(inp, str) else list(inp or [])
    prev = body.get("previous_response_id")
    if not items and not prev:
        raise RequestError("input is required")
    history = []
    if prev:
        if prev not in _STORE:
            raise RequestError(f"previous response {prev!r} not found: this server keeps the last {STORE_MAX} "
                               f"responses in memory while it runs; send the whole conversation as input instead")
        _STORE.move_to_end(prev)
        history = [dict(m) for m in _STORE[prev]]
    convo = history + _messages(items)
    # the chat template wants one system message, first: instructions + leading developer/system messages
    lead = [body["instructions"]] if body.get("instructions") else []
    lead += [m["content"] for m in convo if m["role"] == "system"]
    rest = [m for m in convo if m["role"] != "system"]
    messages = ([{"role": "system", "content": "\n\n".join(lead)}] if lead else []) + rest
    if not rest:
        raise RequestError("input has no user message")
    chat = {"model": body.get("model"), "messages": messages, "stream": bool(body.get("stream"))}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("max_output_tokens", "max_tokens"),
                     ("top_k", "top_k"), ("min_p", "min_p"), ("stop", "stop"),
                     ("reasoning_effort", "reasoning_effort"), ("chat_template_kwargs", "chat_template_kwargs")):
        if body.get(src) is not None:
            chat[dst] = body[src]
    tools = _tools(body.get("tools"))
    if tools:
        chat["tools"] = tools
    choice = _tool_choice(body.get("tool_choice"))
    if choice is not None:
        chat["tool_choice"] = choice
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
        chat["reasoning_effort"] = reasoning["effort"]
    return chat, [m for m in convo if m["role"] != "system"]


class ResponseBuilder:
    """Responses output for one request: SSE events from the chat pipeline's deltas, or a whole Response."""

    def __init__(self, body, model, convo):
        self.body, self.model, self.convo = body, model, convo
        self.id, self.created, self.seq = _id("resp"), int(time.time()), 0
        self.output = []      # finished output items
        self.open = None      # the item being streamed: {"item": ..., "text": [...]}
        self.content, self.calls = "", []

    # ---- the Response object
    def response(self, status, usage=None, incomplete=None):
        b = self.body
        return {
            "id": self.id, "object": "response", "created_at": self.created, "status": status, "error": None,
            "incomplete_details": incomplete, "instructions": b.get("instructions"),
            "max_output_tokens": b.get("max_output_tokens"), "model": self.model, "output": list(self.output),
            "parallel_tool_calls": b.get("parallel_tool_calls", True),
            "previous_response_id": b.get("previous_response_id"),
            "reasoning": {"effort": (b.get("reasoning") or {}).get("effort"), "summary": None},
            "store": b.get("store", True) is not False, "temperature": b.get("temperature"),
            "text": b.get("text") or {"format": {"type": "text"}}, "tool_choice": b.get("tool_choice", "auto"),
            "tools": b.get("tools") or [], "top_p": b.get("top_p"), "truncation": b.get("truncation", "disabled"),
            # usage last, after the output, so a proxy can read it from the tail of a long stream
            "usage": usage, "user": b.get("user"), "metadata": b.get("metadata") or {}}

    @staticmethod
    def usage(prompt_tokens, completion_tokens):
        return {"input_tokens": prompt_tokens, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": completion_tokens, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": prompt_tokens + completion_tokens}

    def _finish(self, reason, usage):
        incomplete = {"reason": "max_output_tokens"} if reason == "length" else None
        self._remember()
        return self.response("incomplete" if incomplete else "completed", usage, incomplete)

    def _remember(self):
        if self.body.get("store", True) is False:
            return
        msg = {"role": "assistant", "content": self.content}
        if self.calls:
            msg["tool_calls"] = self.calls
        _STORE[self.id] = self.convo + [msg]
        while len(_STORE) > STORE_MAX:
            _STORE.popitem(last=False)

    @staticmethod
    def _message_item(text, status="completed"):
        return {"id": _id("msg"), "type": "message", "status": status, "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}

    @staticmethod
    def _reasoning_item(text):
        return {"id": _id("rs"), "type": "reasoning", "summary": [],
                "content": [{"type": "reasoning_text", "text": text}]}

    @staticmethod
    def _call_item(call, status="completed"):
        return {"id": _id("fc"), "type": "function_call", "status": status, "call_id": call["id"],
                "name": call["function"]["name"], "arguments": call["function"]["arguments"]}

    # ---- whole response (stream=false)
    def whole(self, reasoning, content, calls, reason, prompt_tokens, completion_tokens):
        if reasoning:
            self.output.append(self._reasoning_item(reasoning))
        if content:
            self.output.append(self._message_item(content))
        for c in calls or []:
            self.output.append(self._call_item(c))
        self.content, self.calls = content or "", list(calls or [])
        return self._finish(reason, self.usage(prompt_tokens, completion_tokens))

    # ---- streaming
    def _ev(self, kind, **fields):
        data = {"type": kind, "sequence_number": self.seq, **fields}
        self.seq += 1
        return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    def start(self):
        r = self.response("in_progress")
        return [self._ev("response.created", response=r), self._ev("response.in_progress", response=r)]

    def _open(self, kind):
        out, idx = [], len(self.output)
        if kind == "reasoning":
            item = {"id": _id("rs"), "type": "reasoning", "summary": [], "content": []}
            part = {"type": "reasoning_text", "text": ""}
        else:
            item = {"id": _id("msg"), "type": "message", "status": "in_progress", "role": "assistant", "content": []}
            part = {"type": "output_text", "text": "", "annotations": []}
        self.open = {"kind": kind, "item": item, "text": []}
        out.append(self._ev("response.output_item.added", output_index=idx, item=item))
        out.append(self._ev("response.content_part.added", item_id=item["id"], output_index=idx, content_index=0,
                            part=part))
        return out

    def _close(self):
        if not self.open:
            return []
        o, idx = self.open, len(self.output)
        self.open = None
        item, text = o["item"], "".join(o["text"])
        if o["kind"] == "reasoning":
            part = {"type": "reasoning_text", "text": text}
            done = [self._ev("response.reasoning_text.done", item_id=item["id"], output_index=idx, content_index=0,
                             text=text)]
            final = dict(item, content=[part])
        else:
            part = {"type": "output_text", "text": text, "annotations": []}
            done = [self._ev("response.output_text.done", item_id=item["id"], output_index=idx, content_index=0,
                             text=text, logprobs=[])]
            final = dict(item, status="completed", content=[part])
            self.content += text
        done.append(self._ev("response.content_part.done", item_id=item["id"], output_index=idx, content_index=0,
                             part=part))
        done.append(self._ev("response.output_item.done", output_index=idx, item=final))
        self.output.append(final)
        return done

    def feed(self, ev, prompt_tokens):
        """SSE events for one chat-pipeline delta: ("reasoning", text), ("content", text), ("tool_calls", calls),
        or ("end", finish_reason, completion_tokens)."""
        kind, out = ev[0], []
        if kind in ("reasoning", "content"):
            if not ev[1]:
                return out
            if not self.open or self.open["kind"] != kind:
                out += self._close() + self._open(kind)
            o = self.open
            o["text"].append(ev[1])
            delta = dict(item_id=o["item"]["id"], output_index=len(self.output), content_index=0, delta=ev[1])
            if kind == "reasoning":
                out.append(self._ev("response.reasoning_text.delta", **delta))
            else:
                out.append(self._ev("response.output_text.delta", **delta, logprobs=[]))
        elif kind == "tool_calls":
            out += self._close()
            for c in ev[1]:
                idx, item = len(self.output), self._call_item(c, "in_progress")
                args = item["arguments"]
                out.append(self._ev("response.output_item.added", output_index=idx, item=dict(item, arguments="")))
                out.append(self._ev("response.function_call_arguments.delta", item_id=item["id"], output_index=idx,
                                    delta=args))
                out.append(self._ev("response.function_call_arguments.done", item_id=item["id"], output_index=idx,
                                    arguments=args))
                item["status"] = "completed"
                out.append(self._ev("response.output_item.done", output_index=idx, item=item))
                self.output.append(item)
                self.calls.append(c)
        elif kind == "end":
            out += self._close()
            r = self._finish(ev[1], self.usage(prompt_tokens, ev[2]))
            out.append(self._ev("response.incomplete" if r["status"] == "incomplete" else "response.completed",
                                response=r))
        return out

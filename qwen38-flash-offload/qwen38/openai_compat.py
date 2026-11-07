"""OpenAI request compatibility for the Qwen3.8 chat template.

OpenAI clients send OpenAI's reasoning scale (minimal / low / medium / high), but the Qwen3.8 template accepts only
low / medium / xhigh and raises on anything else, which surfaced as HTTP 500 (e.g. Hermes Agent sends "high").
normalize_chat_request() maps the request in place before the template sees it:

  reasoning_effort, chat_template_kwargs.reasoning_effort, reasoning.effort (Responses-style)
    minimal -> low, low -> low, medium -> medium, high / xhigh / max -> xhigh
    none / off / disabled -> thinking off (chat_template_kwargs.enable_thinking = false)
    anything else -> dropped (the model's default effort applies)
"""

EFFORT_MAP = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}
THINKING_OFF = ("none", "off", "disabled")


def normalize_chat_request(body):
    """Normalize reasoning controls in an OpenAI chat request dict. Returns the same dict."""
    if not isinstance(body, dict):
        return body
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort") is not None and body.get("reasoning_effort") is None:
        body["reasoning_effort"] = reasoning["effort"]
    kw = body.get("chat_template_kwargs")
    if not isinstance(kw, dict):
        kw = None
    for holder in (body, kw):
        if holder is None or holder.get("reasoning_effort") is None:
            continue
        value = str(holder["reasoning_effort"]).strip().lower()
        if value in THINKING_OFF:
            del holder["reasoning_effort"]
            if not isinstance(body.get("chat_template_kwargs"), dict):
                body["chat_template_kwargs"] = {}
            body["chat_template_kwargs"]["enable_thinking"] = False
        elif value in EFFORT_MAP:
            holder["reasoning_effort"] = EFFORT_MAP[value]
        else:
            del holder["reasoning_effort"]
    return body

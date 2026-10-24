#!/usr/bin/env python3
"""Streaming delivery per 10-s bin (client-side arrival timestamp of every SSE event), one full greedy answer per case.

Cases: prompt ~512 and ~32768 tokens, through /generate (exact cumulative completion_tokens per event) and through
/v1/chat/completions (deltas; tokens per bin = chars per bin x completion_tokens / total chars, from the final usage).
Unique nonce per case (no radix/prefix hits). No max_tokens: the answer runs to natural EOS.
Delivery must track generation: flat bins at ~the C1 decode rate, no end-of-answer dump (tail_1s_share small).
  python3 stream_bins.py URL OUT.json
"""
import json, secrets, sys, time, urllib.request

URL, OUT = sys.argv[1], sys.argv[2]
BIN = 10.0
Q = ("Ignore the records above. Write a long, detailed, well-structured essay (at least 1500 words) on the history "
     "of the printing press, from woodblock printing to the digital era.")


def req(path, payload):
    return urllib.request.Request(URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})


def tokenize(text):
    with urllib.request.urlopen(req("/tokenize", {"prompt": text}), timeout=300) as r:
        return json.loads(r.read())["tokens"]


def filler(nonce, n_units):
    return "".join(f"Record {i} ({nonce}): sensor {i % 97} logged value {(i * 7919) % 10007} at step {i}. "
                   for i in range(n_units))


def build(target):
    nonce = secrets.token_hex(8)
    head, tail = "[gMASK]<sop><|user|>", f"\n\n{Q}<|assistant|><think></think>"
    fixed = len(tokenize(head + tail))
    f = filler(nonce, 4000)
    per_char = len(tokenize(f)) / len(f)
    n_chars = max(0, int((target - fixed) / per_char))
    while True:
        body = filler(nonce, 4000 * (1 + n_chars // len(f)))[:n_chars]
        ids = tokenize(head + body + tail)
        if len(ids) <= target or n_chars == 0:
            return body, ids
        n_chars -= int((len(ids) - target) / per_char) + 1


def run_generate(ids):
    t0 = time.perf_counter(); pts = []; last = None
    with urllib.request.urlopen(req("/generate", {"input_ids": ids, "sampling_params": {"temperature": 0}, "stream": True}),
                                timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip()); now = time.perf_counter(); last = ev
            n = ev["meta_info"]["completion_tokens"]
            if not pts or n > pts[-1][1]:
                pts.append((now - t0, n))
    return pts, last["meta_info"]["completion_tokens"], last["meta_info"]["prompt_tokens"]


def run_chat(body):
    t0 = time.perf_counter(); pts = []; chars = 0; usage = None
    payload = {"model": "glm-5.3-flash", "temperature": 0, "stream": True,
               "messages": [{"role": "user", "content": body + "\n\n" + Q}], "chat_template_kwargs": {"enable_thinking": False}}
    with urllib.request.urlopen(req("/v1/chat/completions", payload), timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:].strip()); now = time.perf_counter()
            usage = ev.get("usage") or usage
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            k = len(d.get("content") or "") + len(d.get("reasoning_content") or "")
            if k:
                chars += k; pts.append((now - t0, chars))
    N = usage["completion_tokens"]
    return [(t, c * N / chars) for t, c in pts], N, usage["prompt_tokens"]


def bins(pts, N):
    t1, tn = pts[0][0], pts[-1][0]
    out, b = [], 0
    while t1 + b * BIN < tn + 1e-9:
        lo, hi = t1 + b * BIN, min(t1 + (b + 1) * BIN, tn)
        c_lo = max([c for t, c in pts if t <= lo] or [0]); c_hi = max([c for t, c in pts if t <= hi] or [0])
        dur = max(hi - lo, 1e-6)
        out.append({"bin": b, "start_s": round(b * BIN, 1), "dur_s": round(dur, 2), "tokens": round(c_hi - c_lo, 1),
                    "tok_s": round((c_hi - c_lo) / dur, 2)})
        b += 1
    before_tail = max([c for t, c in pts if t <= tn - 1.0] or [0])
    full = [x["tok_s"] for x in out if x["dur_s"] >= BIN - 0.01]
    return out, {"ttft_s": round(t1, 2), "span_s": round(tn - t1, 2), "events": len(pts), "completion_tokens": N,
                 "delivery_tok_s": round((N - pts[0][1]) / max(tn - t1, 1e-6), 2),
                 "full_bin_min": min(full) if full else None, "full_bin_max": max(full) if full else None,
                 "last_bin_tok_s": out[-1]["tok_s"], "tail_1s_share": round(1 - before_tail / N, 4)}


res = {"bin_s": BIN, "cases": []}
for target in (512, 32768):
    body, ids = build(target)
    for ep in ("generate", "chat"):
        if ep == "generate":
            pts, N, P = run_generate(ids)
        else:
            body, _ = build(target)   # fresh nonce: no prefix reuse from the /generate case
            pts, N, P = run_chat(body)
        bb, summ = bins(pts, N)
        case = dict(endpoint=ep, prompt_target=target, prompt_tokens=P, **summ, bins=bb)
        res["cases"].append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "bins"}), flush=True)
        print("  bins tok/s:", " ".join(str(x["tok_s"]) for x in bb), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
print("STREAM_BINS DONE", OUT)

#!/usr/bin/env python3
"""Paired decode-path quality check for the CPU tier. One process, one model load, same cache/expert-cache config:
  pass A (exact): CPU tier OFF, greedy decode with return_logits -> tokens T and logits L_A at every step
  pass B (tier):  CPU tier ON, decode the SAME token sequence: greedy with return_logits; at the first token that differs
                  from T, keep the logits up to and including that step, then restart from prompt + T[:k+1] and continue
                  -> decode-mode logits L_B at every position of T
Reports full-vocabulary KL(L_A || L_B), top-1 agreement and greedy first divergence (results/C052e/decode_kl.json).
Needs QWEN38_CPU_TIER=1. In the image: docker run ... IMAGE decode-kl --kl-out /out/decode_kl.json [--kl-tokens 256]
(the entrypoint applies the same mode defaults and server args as for serving).
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen38"))
out_path = None
if "--kl-out" in sys.argv:
    i = sys.argv.index("--kl-out"); out_path = sys.argv[i + 1]; del sys.argv[i:i + 2]
ntok = 256
if "--kl-tokens" in sys.argv:
    i = sys.argv.index("--kl-tokens"); ntok = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
import torch
import serve as SX
import cpu_tier
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

PROMPTS = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
    "What is 17 multiplied by 23? Show the calculation.",
    "Translate into German: 'The train was late, so we missed the beginning of the concert.'",
    "Summarise the causes of the 2008 financial crisis in five bullet points.",
    "Explain what a hash table is to a 12-year-old.",
]


def run(gen, ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler(), stop_conditions=SX.EOS, return_logits=True)
    gen.enqueue(job)
    toks, logits = [], []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("token_ids") is not None and r["token_ids"].numel():
                toks += r["token_ids"][0].tolist()
                logits.append(r["logits"][0].float().cpu())
    return toks, (torch.cat(logits, 0) if logits else torch.zeros(0))


def main():
    SX.load()
    gen = Generator(model=SX.model, cache=SX.cache, tokenizer=SX.tok, max_chunk_size=SX.args.chunk_size)
    res = []
    kls, agree, firstdiv = [], 0, []
    for q in PROMPTS:
        text = f"[gMASK]<sop><|user|>{q}<|assistant|><think></think>"
        ids = SX.tok.encode(text, encode_special_tokens=True)
        cpu_tier.S["started"] = False
        t0 = time.time(); T, LA = run(gen, ids, ntok); ta = time.time() - t0
        cpu_tier.S["started"] = True
        LB, pos, restarts, fd = [], 0, 0, None
        t0 = time.time()
        while pos < len(T):
            ctx = torch.cat([ids, torch.tensor([T[:pos]], dtype=ids.dtype)], -1) if pos else ids
            tb, lb = run(gen, ctx, len(T) - pos)
            k = 0
            while k < len(tb) and pos + k < len(T) and tb[k] == T[pos + k]:
                k += 1
            take = min(k + 1, len(T) - pos, lb.shape[0])
            LB.append(lb[:take])
            if fd is None and k < len(T) - pos:
                fd = pos + k
            pos += take
            restarts += 1
            if take == 0:
                break
        tb_ = time.time() - t0
        LB = torch.cat(LB, 0)[: LA.shape[0]]
        n = min(LA.shape[0], LB.shape[0])
        la, lb = torch.log_softmax(LA[:n], -1), torch.log_softmax(LB[:n], -1)
        fin = torch.isfinite(la) & torch.isfinite(lb)          # padded vocab rows are -inf in both
        kl = torch.where(fin, la.exp() * (la - lb), torch.zeros_like(la)).sum(-1)
        am_a, am_b = la.argmax(-1), lb.argmax(-1)
        ag = int((am_a == am_b).sum())
        dis = (am_a != am_b).nonzero().flatten()
        # for every top-1 disagreement: exact-model margin between its top-1 and the tier's top-1 (nats)
        margins = [float(la[i, am_a[i]] - la[i, am_b[i]]) for i in dis.tolist()]
        kls += kl.tolist(); agree += ag; firstdiv.append(fd)
        r = {"prompt": q, "tokens": n, "kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "top1_agree": ag / max(n, 1),
             "greedy_first_divergence": fd, "disagree_margins_nats": [round(x, 4) for x in margins], "restarts": restarts, "sec_exact": round(ta, 1), "sec_tier": round(tb_, 1)}
        res.append(r)
        print(json.dumps(r), flush=True)
    tot = {"positions": len(kls), "kl_mean": sum(kls) / max(1, len(kls)), "kl_p99": sorted(kls)[int(0.99 * (len(kls) - 1))] if kls else None,
           "top1_agree": agree / max(1, len(kls)), "first_divergence": firstdiv, "cpu_tier": cpu_tier.summary(), "per_prompt": res}
    print("DECODE_KL", json.dumps({k: v for k, v in tot.items() if k not in ("per_prompt", "cpu_tier")}), flush=True)
    if out_path:
        json.dump(tot, open(out_path, "w"), indent=1)
    cpu_tier._H.tier_stop()


if __name__ == "__main__":
    main()

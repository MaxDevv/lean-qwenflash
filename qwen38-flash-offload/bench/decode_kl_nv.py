#!/usr/bin/env python3
"""N119 paired decode-path quality check for the nv2 CPU tier (C052i/j method). One process, the serving config:
  pass A (exact):   CPU lane OFF (every pick on the exact GPU path: VRAM / admitted RAM / NVMe stall), greedy decode
                    with return_logits -> tokens T and logits L_A at every step
  pass B (variant): decode the SAME token sequence T with a Filter that only allows the reference token (forced decode:
                    no re-prefill restarts), CPU lane per variant -> logits L_B at every position
  control:          pass B with the CPU lane OFF must give KL 0 / top-1 1.0 (validates the method)
Reports full-vocabulary KL(L_A || L_B), top-1 agreement, CPU experts per layer call. Launched like serve.py (same CLI
args and QWEN38_* env); output JSON via --kl-out. No output caps on pass A beyond the forced length --kl-tokens.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qwen38"))
out_path = None
if "--kl-out" in sys.argv:
    i = sys.argv.index("--kl-out"); out_path = sys.argv[i + 1]; del sys.argv[i:i + 2]
ntok = 192
if "--kl-tokens" in sys.argv:
    i = sys.argv.index("--kl-tokens"); ntok = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
import faulthandler
faulthandler.enable(all_threads=True)
import torch
import serve as SX
import nv2
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.generator.filter import Filter


class ForcedFilter(Filter):
    def __init__(self, tok, tokens):
        super().__init__(tok, None, None, True)
        self.tokens = list(tokens); self.i = 0
    def reset(self): self.i = 0
    def accept_token(self, token): self.i += 1
    def is_completed(self): return self.i >= len(self.tokens)
    def use_background_worker(self): return False
    def get_next_logit_mask(self):
        m = torch.full((1, self.vocab_size), float("-inf"), dtype=torch.half)
        m[0, self.tokens[self.i]] = 0.0
        return m


PROMPTS = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
    "What is 17 multiplied by 23? Show the calculation.",
    "Translate into German: 'The train was late, so we missed the beginning of the concert.'",
    "Summarise the causes of the 2008 financial crisis in five bullet points.",
    "Explain what a hash table is to a 12-year-old.",
]
# name -> (cpu lane on, nvme->cpu, clamp[, B70 lane on (N137)])
ALL = {"control_off": (False, None, None), "cpu_default": (True, None, True), "cpu_noclamp": (True, None, False),
       "cpu_nvcpu": (True, 1, True), "b70_only": (False, None, None, True), "cpu_b70": (True, None, True, True)}
VARIANTS = os.environ.get("QWEN38_KL_VARIANTS", "control_off,cpu_default,cpu_noclamp").split(",")


def run(gen, ids, n, filt=None):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler(), stop_conditions=SX.EOS, return_logits=True,
              filters=[filt] if filt is not None else None)
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
    NV = nv2.NV
    gen = Generator(model=SX.model, cache=SX.cache, tokenizer=SX.tok, max_chunk_size=SX.args.chunk_size)
    agg = {v: {"kls": [], "agree": 0, "n": 0, "fd": [], "cpu_experts": 0, "dec_reqs": 0} for v in VARIANTS}
    t0 = time.time()
    for q in PROMPTS:
        text = f"[gMASK]<sop><|user|>{q}<|assistant|><think></think>"
        ids = SX.tok.encode(text, encode_special_tokens=True)
        NV.set_cpu(False); NV.set_b70(False)
        T, LA = run(gen, ids, ntok)
        for name in VARIANTS:
            on, nvc, clamp = ALL[name][:3]
            b70 = len(ALL[name]) > 3 and ALL[name][3]
            NV.set_cpu(on, nvcpu=nvc, clamp=clamp)
            NV.set_b70(b70)
            c0 = NV.counters()
            tb, LB = run(gen, ids, len(T), ForcedFilter(SX.tok, T))
            c1 = NV.counters()
            NV.set_cpu(False); NV.set_b70(False)
            fd = next((i for i, (a, b) in enumerate(zip(tb, T)) if a != b), None)
            n = min(LA.shape[0], LB.shape[0])
            la, lb = torch.log_softmax(LA[:n], -1), torch.log_softmax(LB[:n], -1)
            fin = torch.isfinite(la) & torch.isfinite(lb)
            kl = torch.where(fin, la.exp() * (la - lb), torch.zeros_like(la)).sum(-1)
            ag = int((la.argmax(-1) == lb.argmax(-1)).sum())
            a = agg[name]; a["kls"] += kl.tolist(); a["agree"] += ag; a["n"] += n; a["fd"].append(fd)
            a["cpu_experts"] += c1["cpu_experts"] - c0["cpu_experts"]; a["dec_reqs"] += c1["dec_reqs"] - c0["dec_reqs"]
            a["b70_experts"] = a.get("b70_experts", 0) + c1["lane_b70"] - c0["lane_b70"]
            print(json.dumps({"prompt": q[:40], "variant": name, "n": n, "kl_mean": float(kl.mean()), "top1": ag / max(n, 1),
                              "forced_mismatch": fd}), flush=True)
    tot = {}
    for name, a in agg.items():
        k = sorted(a["kls"])
        tot[name] = {"positions": a["n"], "kl_mean": sum(k) / max(1, len(k)), "kl_p99": k[int(0.99 * (len(k) - 1))] if k else None,
                     "kl_max": k[-1] if k else None, "top1_agree": a["agree"] / max(1, a["n"]), "forced_mismatch": a["fd"],
                     "cpu_experts_per_layer_call": a["cpu_experts"] / max(1, a["dec_reqs"]),
                     "b70_experts_per_layer_call": a.get("b70_experts", 0) / max(1, a["dec_reqs"])}
        print("VARIANT", name, json.dumps(tot[name]), flush=True)
    tot["_meta"] = {"tokens_per_prompt": ntok, "prompts": len(PROMPTS), "seconds": round(time.time() - t0, 1),
                    "cpu_selftest": getattr(NV, "selftest", None), "pol": NV.pol}
    if out_path:
        json.dump(tot, open(out_path, "w"), indent=1)
    print("DECODE_KL_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

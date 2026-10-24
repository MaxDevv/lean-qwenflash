"""N136 profiling hooks (no effect on the computation).
  QWEN38_PROF=START:N   cudaProfilerStart at the START-th decode iteration, cudaProfilerStop N iterations later (pair
                       with `nsys profile --capture-range=cudaProfilerApi`)
  QWEN38_NVTX=1         NVTX ranges: generator phases (iterate, forward enqueue, sample launch, receive_sample, deliver)
                       and module forwards to depth QWEN38_NVTX_DEPTH (default 2: blocks + their attn / mlp / norms)
Installed at the END of load (after k_hcfuse), so the NVTX wrappers wrap the fused block forward."""
import os
import torch

STATE = {"decode_iters": 0, "prof": None, "snaps": []}


def _snap(tag):
    """Raw nv2 attribution counters (cumulative) + lookahead stats, so the capture window's per-token split is a diff."""
    out = {"tag": tag, "decode_iters": STATE["decode_iters"]}
    try:
        import nv2
        e = nv2.ext()
        names = [n for n in nv2.NV.live_fields if n.startswith("a_")]
        out["attr"] = dict(zip(names, [int(v) for v in e.attr()]))
    except Exception as ex:
        out["attr_err"] = repr(ex)
    try:
        import k_lookahead
        out["la"] = k_lookahead.summary()
    except Exception:
        pass
    STATE["snaps"].append(out)
    path = os.environ.get("QWEN38_PROF_OUT", "")
    if path:
        import json
        try:
            json.dump(STATE["snaps"], open(path, "w"))
        except OSError:
            pass


def _rng(name, f):
    def w(*a, **k):
        torch.cuda.nvtx.range_push(name)
        try:
            return f(*a, **k)
        finally:
            torch.cuda.nvtx.range_pop()
    return w


def install(model):
    from exllamav3.generator.generator import Generator
    from exllamav3.generator.job import Job
    from exllamav3.generator.async_generator import AsyncGenerator
    nvtx = os.environ.get("QWEN38_NVTX", "0") == "1"
    prof = os.environ.get("QWEN38_PROF", "")
    start = n = -1
    if prof:
        a, _, b = prof.partition(":")
        start, n = int(a), int(b or 25)
    orig_it = Generator.iterate

    def iterate(self, *a, **k):
        dec = any(j.is_prefill_done() for j in self.active_jobs)
        if dec:
            STATE["decode_iters"] += 1
            c = STATE["decode_iters"]
            if c == start:
                torch.cuda.synchronize()
                _snap("start")
                torch.cuda.cudart().cudaProfilerStart()
                STATE["prof"] = "on"
                print(f" -- n136_prof: cudaProfilerStart at decode iteration {c}", flush=True)
            elif c == start + n:
                torch.cuda.synchronize()
                _snap("stop")
                torch.cuda.cudart().cudaProfilerStop()
                STATE["prof"] = "done"
                print(f" -- n136_prof: cudaProfilerStop at decode iteration {c}", flush=True)
        if nvtx:
            torch.cuda.nvtx.range_push("gen.iterate")
            try:
                return orig_it(self, *a, **k)
            finally:
                torch.cuda.nvtx.range_pop()
        return orig_it(self, *a, **k)
    Generator.iterate = iterate
    if nvtx:
        Job.receive_logits = _rng("job.sample_launch", Job.receive_logits)
        Job.receive_sample = _rng("job.receive_sample", Job.receive_sample)
        AsyncGenerator.deliver_results = _rng("gen.deliver", AsyncGenerator.deliver_results)
        cls = type(model)
        cls.forward = _rng("model.forward_enqueue", cls.forward)
        depth = int(os.environ.get("QWEN38_NVTX_DEPTH", "2"))
        done, cnt = set(), 0

        def walk(ms, d):
            nonlocal cnt
            for m in ms:
                if id(m) in done or not hasattr(m, "forward"):
                    continue
                done.add(id(m))
                key = getattr(m, "key", None) or type(m).__name__
                m.forward = _rng(f"{key}|{type(m).__name__}", m.forward)
                cnt += 1
                subs = getattr(m, "modules", None) or []
                if d > 1 and len(subs) <= 16:
                    walk(subs, d - 1)
        walk(model.modules, depth)
        print(f" -- n136_prof: NVTX on {cnt} module forwards (depth {depth}) + generator phases", flush=True)
    print(f" -- n136_prof: installed (profile window {prof or 'off'})", flush=True)

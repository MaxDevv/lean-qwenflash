"""Overlap the shared expert with the MoE miss path (decode, bsz <= 8).

Stock exllamav3 decode MoE layer (with qwen38 expert_cache + cpu_tier), one stream:
    routing -> ft_split -> ec_step -> ec_copy (PCIe gather of admitted misses, ~0.38 ms/expert, link-bound)
            -> run_bszN (routed coop kernels + shared expert coop merged in kernel B) -> ft_combine (CPU-tier wait)
Here:
  - the shared expert is taken out of the fused kernel (BC_BlockSparseMLP built with sh_exp_bc = None, sh_coop off)
    and launched on a side stream right after routing, on the MoE input, through GatedMLP.forward (its BC graph);
    it runs concurrently with ec_copy / routed coop / the CPU-tier wait, and joins at the stock python add
    (BlockSparseMLP.forward: final_hidden_states += shared_experts.forward(x))
  - ec_copy should run on few blocks (QWEN38_EC_COPY_GRID=16, expert_cache.py): 16 blocks still saturate the link
    (25.0 GB/s, kernels/k102) but leave the SMs free; the stock 656-block grid blocks concurrent kernels
Numerics: shared expert through the GatedMLP graph instead of the coop kernel, and added after ft_combine instead of
inside coop B -> not bitwise vs stock decode (~2e-3 rel on the layer output, same as EXL3_MOE_SHARED_COOP=0);
prefill (the teacher-forced panel) is unchanged (python shared path there already).
Env: QWEN38_K_OVL=1 -> serve.py: k_overlap.install() before load, k_overlap.attach(model) after the cache/tier.
"""
import os
import torch

S = {"streams": {}, "overlapped": 0, "fallback": 0, "bc_stripped": 0}
AFTER = os.environ.get("QWEN38_K_OVL_AFTER", "0") == "1"   # launch the shared expert after the miss path is enqueued


def install():
    """Before model load: build every BC_BlockSparseMLP without the embedded shared expert."""
    from exllamav3.ext import exllamav3_ext as ext
    import exllamav3.modules.block_sparse_mlp as bsm
    real = ext.BC_BlockSparseMLP

    def wrapped(*a, **k):
        a = list(a)
        if len(a) > 38 and a[37] is not None:
            a[37] = None          # sh_exp_bc
            a[38] = None          # sh_gate_bc
            k["sh_coop"] = False
            S["bc_stripped"] += 1
        return real(*a, **k)
    ext.BC_BlockSparseMLP = wrapped
    bsm.ext.BC_BlockSparseMLP = wrapped
    print(" -- k_overlap (K103) installed: shared experts leave the fused decode kernel", flush=True)


def _stream(dev):
    s = S["streams"].get(str(dev))
    if s is None:
        s = S["streams"][str(dev)] = torch.cuda.Stream(dev)
    return s


def attach(model):
    import expert_cache
    from exllamav3.modules.mlp import GatedMLP
    mods = [m for m in expert_cache.find_moe(model.modules) if isinstance(m.shared_experts, GatedMLP)]
    for m in mods:
        m.bc_sh_exp = False               # BC has no shared expert now -> python adds it (stock path)
        se = m.shared_experts
        se._k_pending = None
        orig_fwd = se.forward
        orig_route = m.routing_fn

        def routed(bsz, cfg, z, params, _inner=orig_route, _se=se, _f=orig_fwd, _m=m):
            if bsz > 8 or params.get("autosplit_measure") or params.get("tp_warmup"):
                return _inner(bsz, cfg, z, params)
            main = torch.cuda.current_stream(_m.device)
            side = _stream(_m.device)
            ev = torch.cuda.Event(); ev.record(main)       # MoE input ready
            r = _inner(bsz, cfg, z, params) if AFTER else None   # AFTER: ft_split / ec_step / ec_copy enqueued first
            side.wait_event(ev)
            with torch.cuda.stream(side):
                y = _f(z.view(bsz, 1, -1), params)
                done = torch.cuda.Event(); done.record(side)
            y.record_stream(main)
            _se._k_pending = (y, done)
            return r if AFTER else _inner(bsz, cfg, z, params)

        def sh_forward(x, params, *args, _se=se, _f=orig_fwd, **kw):
            p = _se._k_pending
            if p is not None:
                _se._k_pending = None
                y, done = p
                torch.cuda.current_stream(x.device).wait_event(done)
                S["overlapped"] += 1
                return y.view(*x.shape[:-1], y.shape[-1])
            S["fallback"] += 1
            return _f(x, params, *args, **kw)

        m.routing_fn = routed
        se.forward = sh_forward
    print(f" -- k_overlap (K103): {len(mods)} MoE layers, shared expert on a side stream "
          f"(BCs stripped {S['bc_stripped']}, ec_copy grid {os.environ.get('QWEN38_EC_COPY_GRID', 'stock')})", flush=True)
    return len(mods)


def summary():
    return {k: v for k, v in S.items() if k != "streams"}

"""CPU tier for Qwen3.8-Flash-Next on 1 GPU (kernels/cpu_avx2/ft_core.h, ft_tier_ext.cpp, ft_tier_cu.cu).

Per MoE layer in decode (bsz <= QWEN38_CT_MAXBSZ), right after the router and BEFORE expert_cache's step:
  ft_split (GPU, 1 block): hits / misses from the cache's slotof; misses sorted coldest first by a static routing-frequency
      score; n_cpu from a cost model (GPU: t_hit per resident expert + t_zc per zero-copy miss; CPU: a + b per expert);
      the n_cpu coldest misses become a CPU job published to pinned host memory (x rows, picks, seq flag); exllamav3 gets
      the selection with those picks removed (id -1 / weight 0) -> the cache does not admit them (warm misses still go
      zero-copy + admit), the fused MoE kernel skips them
  CPU worker (C++ thread + 21 pinned pool threads) computes the job with ft_mul1 (I16 for <= 2 tokens/expert, AFFINE
      otherwise) straight from the native pinned home copy (exl3_tiers arenas), writes an fp32 partial
  ft_combine (GPU, in BlockSparseMLP.cpu_split_combine, after the GPU experts): waits on the done flag, adds the partial
Prefill / big batches: tier off (GPU staging path unchanged).

Env: QWEN38_CPU_TIER=1, QWEN38_CT_THREADS (default: number of CPUs in QWEN38_CT_CPUS), QWEN38_CT_CPUS ("auto" = one
     logical CPU per physical core minus the first 2, e.g. "2-23" on a 24-core part; or an explicit list "2-23"), QWEN38_CT_MODE (-1 auto | 0 AFFINE | 1 EXACT | 2 I16),
     QWEN38_CT_MAXBSZ (4), QWEN38_CT_STATS (score JSON; default QWEN38_EC_WARM / QWEN38_ZC_STATS),
     QWEN38_CT_TZC (0.40), QWEN38_CT_THIT (0.023), QWEN38_CT_A (0.11), QWEN38_CT_B (0.104), QWEN38_CT_TOK (0.2),
     QWEN38_CT_MAXN (32), QWEN38_CT_FORCE_N (-1), QWEN38_CT_SWZ (1: CPU-side block-contiguous copy, +114.6 GB RAM),
     QWEN38_CT_VALIDATE (0; 1 = panel through the CPU kernel: all misses of every forward <= QWEN38_CT_MAXBSZ tokens), QWEN38_CT_HANDSHAKE (0), QWEN38_CT_SELFTEST (1)
Hook (serve.py): cpu_tier.wrap(model) before expert_cache.attach(model); cpu_tier.start() after it; /stats: summary().
"""
import json, os, time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = os.path.join(os.path.dirname(_HERE), "kernels", "cpu_avx2")
_H = None
_CU = None
S = {"mods": [], "dev": None, "seq": 0, "started": False, "selftest": None}


def _build():
    global _H, _CU
    if _H is not None:
        return
    from torch.utils.cpp_extension import load
    d = os.environ.get("QWEN38_CT_BUILD", os.path.join(os.path.dirname(_HERE), "build", "ct"))
    os.makedirs(d, exist_ok=True)
    # Portable ISA flags (AVX2 + FMA + F16C is all the kernel uses); the campaign build used -march=znver3.
    isa = os.environ.get("QWEN38_CT_CFLAGS", "-mavx2 -mfma -mf16c -mtune=znver3").split()
    _H = load(name="ft_tier_host", sources=[os.path.join(_KDIR, "ft_tier_ext.cpp")], build_directory=d,
              extra_cflags=["-O3", *isa, "-std=c++17", "-I" + _KDIR], extra_ldflags=["-lpthread"], verbose=False)
    _CU = load(name="ft_tier_cu", sources=[os.path.join(_KDIR, "ft_tier_cu.cu")], build_directory=d,
               extra_cuda_cflags=["-O3"], verbose=False)


def auto_cpus(skip=2):
    """One logical CPU per physical core (first SMT sibling) among the CPUs this process may run on, minus the first
    `skip` cores (left to the GPU-feeding Python thread and the OS). On a 24-core EPYC 7443P with SMT: 2-23 (22)."""
    allowed = sorted(os.sched_getaffinity(0))
    seen, firsts = set(), []
    for c in allowed:
        try:
            core = open(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read().strip()
            pkg = open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read().strip()
        except OSError:
            core, pkg = str(c), "0"
        if (pkg, core) not in seen:
            seen.add((pkg, core)); firsts.append(c)
    return firsts[skip:] if len(firsts) > skip + 1 else firsts[-1:]


def _cpus():
    spec = os.environ.get("QWEN38_CT_CPUS", "auto")
    if spec == "auto":
        return auto_cpus()
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _policy(dev):
    g = lambda k, d: float(os.environ.get(k, d))
    return torch.tensor([g("QWEN38_CT_TZC", 0.40), g("QWEN38_CT_THIT", 0.023), g("QWEN38_CT_A", 0.11), g("QWEN38_CT_B", 0.104),
                         g("QWEN38_CT_TOK", 0.2), g("QWEN38_CT_MAXN", 32), g("QWEN38_CT_FORCE_N", -1), g("QWEN38_CT_HANDSHAKE", 0)],
                        dtype=torch.float32, device=dev)


def wrap(model):
    """Before expert_cache.attach: wrap every MoE router (so the cache sees the CPU picks removed) and the combine hook."""
    _build()
    import expert_cache
    mods = expert_cache.find_moe(model.modules)
    assert mods, "no MoE modules"
    dev = mods[0].device
    assert all(m.device == dev for m in mods), "CPU tier: single GPU only"
    S["dev"] = torch.device(dev)
    H = mods[0].hidden_size
    # QWEN38_CT_VALIDATE=1: every forward up to QWEN38_CT_MAXBSZ (default 4096) tokens goes through the tier with ALL
    # cache misses on the CPU (policy force), so the teacher-forced panel measures the CPU kernel's numerics
    validate = os.environ.get("QWEN38_CT_VALIDATE") == "1"
    maxbsz = int(os.environ.get("QWEN38_CT_MAXBSZ", "4096" if validate else "4"))
    if validate:
        os.environ.setdefault("QWEN38_CT_FORCE_N", "100000"); os.environ.setdefault("QWEN38_CT_TIMEOUT_S", "120")
    S.update(maxbsz=maxbsz, H=H, validate=validate, maxpicks=max(64, 8 * maxbsz))
    S["ctrl"] = torch.zeros(16, dtype=torch.int64).pin_memory()
    S["hx"] = torch.zeros(max(8, maxbsz) * H, dtype=torch.half).pin_memory()
    S["picks"] = torch.zeros(S["maxpicks"] * 3, dtype=torch.int32).pin_memory()
    S["hout"] = torch.zeros(max(8, maxbsz) * H, dtype=torch.float32).pin_memory()
    S["dflag"] = torch.zeros(1, dtype=torch.int64, device=dev)
    S["stats"] = torch.zeros(8, dtype=torch.int64, device=dev)
    sp = os.environ.get("QWEN38_CT_STATS") or os.environ.get("QWEN38_EC_WARM") or os.environ.get("QWEN38_ZC_STATS")
    scores = json.load(open(sp)) if sp else {}
    S["pol"] = _policy(dev)
    S["timeout_ns"] = int(float(os.environ.get("QWEN38_CT_TIMEOUT_S", "2")) * 1e9)
    for m in mods:
        E = m.num_local_experts or m.num_experts
        first = int(m.routing_first or 0)
        sc = scores.get(m.key)
        m._ct_score = torch.tensor(sc[first:first + E] if sc else [0.0] * E, dtype=torch.float32, device=dev)
        m._ct_sel = torch.empty(S["maxpicks"], dtype=torch.long, device=dev)
        m._ct_w = torch.empty(S["maxpicks"], dtype=torch.half, device=dev)
        m._ct_pending = False
        orig = m.routing_fn

        def routed(bsz, cfg, z, params, _orig=orig, _m=m):
            sel, w = _orig(bsz, cfg, z, params)
            _m._ct_pending = False
            if (not S["started"] or bsz > S["maxbsz"] or params.get("autosplit_measure") or params.get("tp_warmup")
                    or getattr(_m, "_ec", None) is None or sel.numel() > S["maxpicks"]):
                return sel, w
            pool, li = _m._ec
            S["seq"] += 1
            n = sel.numel()
            so, wo = _m._ct_sel[:n], _m._ct_w[:n]
            zz = z.view(bsz, -1)
            if zz.dtype != torch.half or not zz.is_contiguous():
                zz = zz.half().contiguous()
            _CU.ft_split(sel.contiguous(), w.contiguous(), zz, pool.E, pool.first, pool.slotof[li], _m._ct_score, S["pol"],
                         S["ctrl"].data_ptr(), S["hx"].data_ptr(), S["picks"].data_ptr(), S["seq"], li, int(bsz > 8), so, wo, S["dflag"], S["stats"])
            _m._ct_pending = True
            return so.view(sel.shape), wo.view(w.shape)
        m.routing_fn = routed

        orig_comb = m.cpu_split_combine

        def comb(fhs, cpu_partial, cpu_pending, shape, _orig=orig_comb, _m=m):
            if _m._ct_pending:
                _m._ct_pending = False
                f = fhs if fhs.is_contiguous() else fhs.contiguous()
                _CU.ft_combine(f, S["dflag"], S["ctrl"].data_ptr(), S["hout"].data_ptr(), S["stats"], S["timeout_ns"])
                fhs = f
            return _orig(fhs, cpu_partial, cpu_pending, shape)
        m.cpu_split_combine = comb
    S["mods"] = mods
    _prelaunch(dev, H, mods[0])
    print(f" -- cpu_tier: wrapped {len(mods)} MoE layers on {dev} (max bsz {maxbsz}, scores {sp})", flush=True)


def _prelaunch(dev, H, m):
    """Launch both kernels once BEFORE expert_cache sizes its arena, so any per-kernel launch resources (local-memory
    reservation, module load) are taken while VRAM is still free (otherwise: launch OOM once the elastic arena
    holds all free VRAM)."""
    E = m.num_local_experts or m.num_experts
    sel = torch.zeros(1, 8, dtype=torch.long, device=dev); w = torch.zeros(1, 8, dtype=torch.half, device=dev)
    z = torch.zeros(1, H, dtype=torch.half, device=dev)
    slot = torch.full((E,), -1, dtype=torch.int32, device=dev)
    pol = torch.tensor([0.4, 0.023, 0.11, 0.104, 0.2, 0, 0, 0], dtype=torch.float32, device=dev)   # force 0, no publish
    st = torch.zeros(8, dtype=torch.int64, device=dev)
    _CU.ft_split(sel, w, z, E, int(m.routing_first or 0), slot, m._ct_score, pol, S["ctrl"].data_ptr(), S["hx"].data_ptr(),
                 S["picks"].data_ptr(), 0, 0, 0, m._ct_sel[:8], m._ct_w[:8], S["dflag"], st)
    out = torch.zeros(1, H, dtype=torch.float32, device=dev)
    _CU.ft_combine(out, S["dflag"], S["ctrl"].data_ptr(), S["hout"].data_ptr(), st, S["timeout_ns"])
    torch.cuda.synchronize(dev)


def start():
    """After expert_cache.attach: register the native pinned home copies with the host kernel, self-test, start."""
    mods = S["mods"]
    m0 = mods[0]
    H, I = m0.hidden_size, m0.intermediate_size
    cpus = _cpus()
    threads = int(os.environ.get("QWEN38_CT_THREADS", "0")) or len(cpus)
    assert 1 <= threads <= len(cpus), f"QWEN38_CT_THREADS={threads} needs at least that many CPUs in QWEN38_CT_CPUS ({cpus})"
    mode = int(os.environ.get("QWEN38_CT_MODE", "-1"))
    swz = int(os.environ.get("QWEN38_CT_SWZ", "1"))
    _H.tier_init(threads, cpus, S["ctrl"], S["hx"], S["picks"], S["hout"], mode, H, I, len(mods), swz)
    import exl3_tiers
    arenas = [(a.data_ptr(), a.data_ptr() + a.numel()) for a in exl3_tiers._ARENAS]
    t0 = time.time()
    for m in mods:
        pool, li = m._ec
        ptrs = pool.home_cpu[li].contiguous()
        lo, hi = int(ptrs.min()), int(ptrs.max())
        assert any(a <= lo and hi < b for a, b in arenas) or all(any(a <= int(p) < b for a, b in arenas) for p in ptrs.flatten().tolist()), \
            f"{m.key}: home pointers outside the pinned arenas"
        sc = lambda lins, attr: torch.stack([getattr(l.inner, attr).view(-1) for l in lins]).half().cpu().contiguous()
        g, u, d = m.multi_gate.linears, m.multi_up.linears, m.multi_down.linears
        _H.tier_add_layer(li, ptrs, sc(g, "suh"), sc(g, "svh"), sc(u, "suh"), sc(u, "svh"), sc(d, "suh"), sc(d, "svh"))
    print(f" -- cpu_tier: registered {len(mods)} layers in {time.time() - t0:.1f} s (CPU block-contiguous copy "
          f"{_H.tier_stats()[5] / 1e9:.1f} GB); threads {threads} on cpus {cpus[:threads]}", flush=True)
    if os.environ.get("QWEN38_CT_SELFTEST", "1") == "1":
        S["selftest"] = selftest()
        print(f" -- cpu_tier selftest: {S['selftest']}", flush=True)
    _H.tier_start(cpus[0])
    S["started"] = True


@torch.inference_mode()
def selftest(n_layers=3, n_exp=4, mtok=(1, 3)):
    """ft_mul1 vs fp64 truth from exllamav3's own dequantised weights (LinearEXL3.get_weight_tensor = GPU decode +
    Hadamard + suh/svh), computed on the CPU: rel RMS of the weighted routed sum per mode.
    """
    out = {}
    mods = S["mods"]
    g = torch.Generator(device="cpu").manual_seed(0)
    cases = []
    for li in list(range(0, len(mods), max(1, len(mods) // n_layers)))[:n_layers]:
        m = mods[li]
        E = m.num_local_experts or m.num_experts
        for mt in mtok:
            x = (torch.randn(mt, m.hidden_size, generator=g) * 0.35).half()
            exps = torch.randperm(E, generator=g)[:n_exp].tolist()
            ref = torch.zeros(mt, m.hidden_size, dtype=torch.float64)
            xd = x.double()
            for e in exps:
                Wg = m.gates[e].inner.get_weight_tensor().cpu().double(); Wu = m.ups[e].inner.get_weight_tensor().cpu().double()
                a = (torch.nn.functional.silu(xd @ Wg) * (xd @ Wu)).half().double()
                del Wg, Wu
                Wd = m.downs[e].inner.get_weight_tensor().cpu().double()
                ref += 0.25 * (a @ Wd)
                del Wd
            cases.append((li, x, exps, ref))
    torch.cuda.empty_cache()
    for mode in (1, 0, 2):
        errs = []
        for li, x, exps, ref in cases:
            mt = x.shape[0]
            sel = torch.tensor([exps] * mt, dtype=torch.int32)
            w = torch.full((mt, len(exps)), 0.25)
            cpu = _H.tier_forward(li, x.float(), sel, w, mode).double()
            errs.append(float((cpu - ref).norm() / ref.norm()))
        out[{1: "EXACT", 0: "AFFINE", 2: "I16"}[mode]] = round(max(errs), 6)
    return out


def summary():
    if not S["mods"]:
        return None
    st = S["stats"].tolist()
    h = _H.tier_stats() if _H else []
    calls = max(1, st[3])
    return {"calls": st[3], "unique_hits": st[0], "unique_misses": st[1], "cpu_experts": st[2],
            "cpu_experts_per_call": round(st[2] / calls, 3), "misses_per_call": round(st[1] / calls, 3),
            "hit_rate_unique": round(st[0] / max(1, st[0] + st[1]), 4),
            "gpu_wait_ms_per_job": round(st[4] / 1e6 / max(1, st[5]), 4), "gpu_waits": st[5], "timeouts": st[6],
            "host_jobs": h[0] if h else None, "host_busy_ms_per_job": round(h[4] / 1e6 / max(1, h[0]), 4) if h else None,
            "host_empty_jobs": h[3] if h else None, "cpu_copy_gb": round(h[5] / 1e9, 1) if h else None, "selftest": S["selftest"],
            "policy": S["pol"].tolist()}


def reset_stats():
    S["stats"].zero_()

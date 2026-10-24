"""Dynamic GPU expert cache for stock exllamav3 1.5.1 (Qwen3.8-Flash-Next; bit-exact by construction).

Layout: every routed expert's trellis lives in pinned host memory (exl3_tiers.py with QWEN38_ZC_VRAM=0, the home copy,
always valid). Each GPU gets one slot arena (all free VRAM after load minus a reserve) shared by all MoE layers on that
GPU. Per layer and forward step, after routing and before the expert kernels:

  ec_step  (1 block, device-side, no host sync): dedup the picks; hits -> CLOCK ref bit + pin; if admission is on
           (picks <= QWEN38_EC_ADMIT_MAX, i.e. decode / small batches), each miss takes a CLOCK victim slot (never one
           pinned by this layer's current picks), the victim's pointer-table rows go back to its home (host) address,
           the miss's rows point at the slot, and a copy job is queued
  ec_copy  (grid-stride zero-copy gather): queued experts host -> slot (the same PCIe bytes the kernel would have read)
  exllamav3's own MoE kernels then run unchanged and read each expert through the live pointer tables
  (multi_gate / multi_up / multi_down .ptrs_trellis and BC_BlockSparseMLP's gu_trellis_ptr), so the arithmetic and the
  bytes are identical to an all-VRAM run: only the address differs. Pointer copies exllamav3 takes at load time
  (host-side *_cpu tables, per-expert Linear BCs) keep the home address, which stays valid.
Prefill (large picks) runs with admission off: hits read VRAM, misses read zero-copy, the decode working set survives.

Prefill staging (QWEN38_EC_STAGE_GB > 0, per buffer; two buffers per GPU): for a forward with >= QWEN38_EC_STAGE_MIN
picks per layer, each layer's NON-cached experts are copied host -> a VRAM staging buffer on a copy stream (copy engine)
one layer ahead, while the previous layer computes; the layer's tables point at the buffer for its MoE and go back to
home afterwards. Buffer overflow -> the rest stays zero-copy. Cached set read once per forward (no admission in prefill).

Env: QWEN38_EC=1 (serve.py), QWEN38_EC_RESERVE_GB (per GPU, default 2.5), QWEN38_EC_ADMIT_MAX (default 64 picks),
     QWEN38_EC_WARM (optional JSON {layer_key: [score x E]}: initial fill by score), QWEN38_EC_MAX_SLOTS (per GPU cap)
"""
import json, os, time
import numpy as np
import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void ec_step_k(const int64_t* __restrict__ sel, int n_picks, int first, int li, int E, int S, int admit,
    int* slotof_all, const int64_t* home_all, const int64_t* tabs, int* owner, int* refb, int* pin, int* ctl,
    int* jobs, int maxj, unsigned long long* stats, const long long* __restrict__ cbase, int spc, long long rec,
    long long off_u, long long off_d)
{
    __shared__ unsigned mask[32];
    __shared__ int misses[1024];
    __shared__ int nmiss, nhit, epoch;
    int t = threadIdx.x;
    if (t < 32) mask[t] = 0;
    if (t == 0) { nmiss = 0; nhit = 0; epoch = ctl[1] + 1; ctl[1] = epoch; ctl[2] = 0; }
    __syncthreads();
    for (int i = t; i < n_picks; i += blockDim.x) {
        long long e = sel[i] - first;
        if (e >= 0 && e < E) atomicOr(&mask[e >> 5], 1u << (e & 31));
    }
    __syncthreads();
    int* slotof = slotof_all + (long long) li * E;
    for (int e = t; e < E; e += blockDim.x) {
        if (mask[e >> 5] & (1u << (e & 31))) {
            int s = slotof[e];
            if (s >= 0) { refb[s] = 1; pin[s] = epoch; atomicAdd(&nhit, 1); }
            else { int k = atomicAdd(&nmiss, 1); misses[k] = e; }
        }
    }
    __syncthreads();
    if (t != 0) return;
    stats[0] += nhit; stats[1] += nmiss;
    if (!admit || S == 0) return;
    int hand = ctl[0], nj = 0;
    int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
    int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
    for (int k = 0; k < nmiss && nj < maxj; k++) {
        int e = misses[k], found = -1;
        for (int step = 0; step < 2 * S; step++) {
            int s = hand; hand = (hand + 1 == S) ? 0 : hand + 1;
            if (pin[s] == epoch || pin[s] < 0) continue;   // pinned by this step, or disabled (elastic)
            if (refb[s]) { refb[s] = 0; continue; }
            found = s; break;
        }
        if (found < 0) { stats[3]++; break; }
        int s = found, old = owner[s];
        if (old >= 0) {
            int ol = old / E, oe = old % E;
            slotof_all[(long long) ol * E + oe] = -1;
            const int64_t* h = home_all + ((long long) ol * E + oe) * 3;
            int64_t* og = (int64_t*) tabs[ol * 4 + 0]; int64_t* ou = (int64_t*) tabs[ol * 4 + 1];
            int64_t* od = (int64_t*) tabs[ol * 4 + 2]; int64_t* ogu = (int64_t*) tabs[ol * 4 + 3];
            og[oe] = h[0]; ou[oe] = h[1]; od[oe] = h[2];
            if (ogu) { ogu[2 * oe] = h[0]; ogu[2 * oe + 1] = h[1]; }
            stats[4]++;
        }
        owner[s] = li * E + e; slotof[e] = s; refb[s] = 1; pin[s] = epoch;
        long long a = cbase[s / spc] + (long long) (s % spc) * rec;
        pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
        if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
        jobs[2 * nj] = e; jobs[2 * nj + 1] = s; nj++;
        stats[2]++;
    }
    ctl[0] = hand; ctl[2] = nj;
}

__global__ void ec_copy_k(const int* __restrict__ ctl, const int* __restrict__ jobs, int li, int E,
    const int64_t* __restrict__ home_all, const long long* __restrict__ cbase, int spc, long long rec, long long vg,
    long long vu, long long vd, long long off_u, long long off_d)
{
    long long nj = ctl[2];
    long long vpj = vg + vu + vd;
    long long total = nj * vpj;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (long long base = (long long) blockIdx.x * blockDim.x + threadIdx.x; base < total; base += 4 * stride) {
        int4 v[4]; int4* dst[4]; int n = 0;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            long long idx = base + u * stride;
            if (idx >= total) break;
            long long j = idx / vpj, r = idx - j * vpj;
            int e = jobs[2 * j], s = jobs[2 * j + 1];
            const int64_t* h = home_all + ((long long) li * E + e) * 3;
            const int4* src; long long doff;
            if (r < vg) { src = (const int4*) h[0] + r; doff = r * 16; }
            else if (r < vg + vu) { src = (const int4*) h[1] + (r - vg); doff = off_u + (r - vg) * 16; }
            else { src = (const int4*) h[2] + (r - vg - vu); doff = off_d + (r - vg - vu) * 16; }
            v[u] = *src;
            dst[u] = (int4*) (cbase[s / spc] + (long long) (s % spc) * rec + doff);
            n++;
        }
        for (int u = 0; u < n; u++) *dst[u] = v[u];
    }
}

void ec_step(at::Tensor sel, int64_t first, int64_t li, int64_t E, int64_t S, int64_t admit, at::Tensor slotof, at::Tensor home,
             at::Tensor tabs, at::Tensor owner, at::Tensor refb, at::Tensor pin, at::Tensor ctl, at::Tensor jobs,
             at::Tensor stats, int64_t arena, int64_t spc, int64_t rec, int64_t off_u, int64_t off_d)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && sel.is_contiguous(), "sel int64 contiguous");
    TORCH_CHECK(E <= 1024, "E <= 1024");
    ec_step_k<<<1, 512, 0, st>>>(sel.data_ptr<int64_t>(), (int) sel.numel(), (int) first, (int) li, (int) E, (int) S, (int) admit,
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), owner.data_ptr<int>(),
        refb.data_ptr<int>(), pin.data_ptr<int>(), ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) (jobs.numel() / 2),
        (unsigned long long*) stats.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, off_u, off_d);
}

__global__ void ec_copy_stage_k(const int* ctl, const int* jobs, const int64_t* src,
    const int64_t* cbase, int spc, long long rec)
{
    const long long words = rec / 16, total = (long long) ctl[2] * words;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x)
    {
        const int j = int(i / words), e = jobs[2*j], s = jobs[2*j+1];
        const int4* from = (const int4*) src[e];
        int4* to = (int4*) (cbase[s/spc] + (long long) (s%spc) * rec);
        to[i%words] = from[i%words];
    }
}
void ec_copy_stage(at::Tensor ctl, at::Tensor jobs, at::Tensor src, int64_t cbase,
                   int64_t spc, int64_t rec, int64_t grid)
{
    c10::cuda::CUDAGuard guard(ctl.device());
    auto stream = at::cuda::getCurrentCUDAStream(ctl.device().index());
    ec_copy_stage_k<<<grid,256,0,stream>>>(ctl.data_ptr<int>(),jobs.data_ptr<int>(),src.data_ptr<int64_t>(),
                                        (const int64_t*) cbase,spc,rec);
}

void ec_copy(at::Tensor ctl, at::Tensor jobs, int64_t li, int64_t E, at::Tensor home, int64_t arena, int64_t spc, int64_t rec,
             int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d, int64_t grid)
{
    c10::cuda::CUDAGuard g(ctl.device());
    auto st = at::cuda::getCurrentCUDAStream(ctl.device().index());
    ec_copy_k<<<(int) grid, 256, 0, st>>>(ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) li, (int) E,
        home.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, sg / 16, su / 16, sd / 16, off_u, off_d);
}


__global__ void ec_verify_k(const int* __restrict__ owner, const int* __restrict__ slotof_all, const int64_t* __restrict__ home_all,
    const int64_t* __restrict__ tabs, const long long* __restrict__ cbase, int spc, long long rec, long long sg, long long su, long long sd,
    long long off_u, long long off_d, int S, int E, int L, unsigned long long* out)
{
    // out[0] = slots whose bytes differ from home, out[1] = table entries inconsistent with slotof, out[2] = owned slots
    int s = blockIdx.x;
    if (s < S) {
        int o = owner[s];
        if (o >= 0) {
            int l = o / E, e = o % E;
            if (threadIdx.x == 0) atomicAdd(&out[2], 1ull);
            const int64_t* h = home_all + ((long long) l * E + e) * 3;
            const int4* src[3] = {(const int4*) h[0], (const int4*) h[1], (const int4*) h[2]};
            long long off[3] = {0, off_u, off_d}; long long n[3] = {sg / 16, su / 16, sd / 16};
            __shared__ int bad;
            if (threadIdx.x == 0) bad = 0;
            __syncthreads();
            for (int f = 0; f < 3; f++) {
                const int4* d = (const int4*) (cbase[s / spc] + (long long) (s % spc) * rec + off[f]);
                for (long long i = threadIdx.x; i < n[f]; i += blockDim.x) {
                    int4 a = src[f][i], b = d[i];
                    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) bad = 1;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && bad) atomicAdd(&out[0], 1ull);
            if (threadIdx.x == 0 && slotof_all[(long long) l * E + e] != s) atomicAdd(&out[1], 1ull);
        }
    }
    // table check: one thread per (layer, expert)
    long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < (long long) L * E) {
        int l = gid / E, e = gid % E;
        int sl = slotof_all[gid];
        const int64_t* pg = (const int64_t*) tabs[l * 4 + 0]; const int64_t* pu = (const int64_t*) tabs[l * 4 + 1];
        const int64_t* pd = (const int64_t*) tabs[l * 4 + 2]; const int64_t* gu = (const int64_t*) tabs[l * 4 + 3];
        const int64_t* h = home_all + gid * 3;
        int64_t eg, eu, ed;
        if (sl >= 0) { long long a = cbase[sl / spc] + (long long) (sl % spc) * rec; eg = a; eu = a + off_u; ed = a + off_d; }
        else { eg = h[0]; eu = h[1]; ed = h[2]; }
        bool ok = pg[e] == eg && pu[e] == eu && pd[e] == ed;
        if (gu) ok = ok && gu[2 * e] == eg && gu[2 * e + 1] == eu;
        if (!ok) atomicAdd(&out[1], 1ull);
    }
}

at::Tensor ec_verify(at::Tensor owner, at::Tensor slotof, at::Tensor home, at::Tensor tabs, int64_t arena, int64_t spc, int64_t rec,
                     int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d, int64_t S, int64_t E, int64_t L)
{
    c10::cuda::CUDAGuard g(owner.device());
    auto st = at::cuda::getCurrentCUDAStream(owner.device().index());
    auto out = torch::zeros({4}, owner.options().dtype(at::kLong));
    long long nb = std::max((long long) S, ((long long) L * E + 255) / 256);
    ec_verify_k<<<(int) nb, 256, 0, st>>>(owner.data_ptr<int>(), slotof.data_ptr<int>(), home.data_ptr<int64_t>(),
        tabs.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, sg, su, sd, off_u, off_d, (int) S, (int) E, (int) L,
        (unsigned long long*) out.data_ptr<int64_t>());
    return out;
}

__global__ void ec_evict_k(int s0, int s1, int disable, int E, int* owner, int* slotof_all, const int64_t* home_all,
    const int64_t* tabs, int* refb, int* pin)
{
    int s = s0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= s1) return;
    int o = owner[s];
    if (o >= 0) {
        int l = o / E, e = o % E;
        slotof_all[(long long) l * E + e] = -1;
        const int64_t* h = home_all + ((long long) l * E + e) * 3;
        int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
        int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
        pg[e] = h[0]; pu[e] = h[1]; pd[e] = h[2];
        if (gu) { gu[2 * e] = h[0]; gu[2 * e + 1] = h[1]; }
        owner[s] = -1;
    }
    refb[s] = 0;
    pin[s] = disable ? -1 : 0;
}

void ec_evict(int64_t s0, int64_t s1, int64_t disable, int64_t E, at::Tensor owner, at::Tensor slotof, at::Tensor home,
              at::Tensor tabs, at::Tensor refb, at::Tensor pin)
{
    c10::cuda::CUDAGuard g(owner.device());
    auto st = at::cuda::getCurrentCUDAStream(owner.device().index());
    int n = (int) (s1 - s0);
    if (n <= 0) return;
    ec_evict_k<<<(n + 255) / 256, 256, 0, st>>>((int) s0, (int) s1, (int) disable, (int) E, owner.data_ptr<int>(),
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), refb.data_ptr<int>(), pin.data_ptr<int>());
}

void memcpy_runs(at::Tensor runs, int64_t stream, int64_t device)
{
    // runs: CPU int64 [n, 3] = (dst, src, bytes); cudaMemcpyDefault (src may be a UVA host alias)
    c10::cuda::CUDAGuard g((c10::DeviceIndex) device);
    auto r = runs.accessor<int64_t, 2>();
    for (int64_t i = 0; i < runs.size(0); i++)
        cudaMemcpyAsync((void*) r[i][0], (const void*) r[i][1], (size_t) r[i][2], cudaMemcpyDefault, (cudaStream_t) stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ec_step", &ec_step);
    m.def("ec_copy", &ec_copy);
    m.def("ec_copy_stage", &ec_copy_stage);
    m.def("memcpy_runs", &memcpy_runs);
    m.def("ec_verify", &ec_verify);
    m.def("ec_evict", &ec_evict);
}
"""

_EXT = None
_BC_ARGS = {}      # id(multi_gate.ptrs_trellis) -> gu_trellis_ptr tensor captured from BC_BlockSparseMLP(...)
POOLS = []


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = os.environ.get("QWEN38_EC_BUILD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "ec"))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ec.cu")
        if not os.path.exists(src) or open(src).read() != _SRC:
            open(src, "w").write(_SRC)
        _EXT = load(name="qwen38_ec", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


def install():
    """Before model load: capture gu_trellis_ptr (a local of BlockSparseMLP.load_local) from the BC constructor."""
    from exllamav3.ext import exllamav3_ext as ext
    import exllamav3.modules.block_sparse_mlp as bsm
    real = ext.BC_BlockSparseMLP

    def wrapped(*a, **k):
        obj = real(*a, **k)
        _BC_ARGS[id(a[16])] = (a[16], a[22], a[43])   # gate ptrs (identity key), up ptrs, gu_trellis_ptr
        return obj
    ext.BC_BlockSparseMLP = wrapped
    bsm.ext.BC_BlockSparseMLP = wrapped
    _ext()   # build now (fail fast, before the long load)
    print(" -- expert_cache installed", flush=True)


class Pool:
    def __init__(self, device, mods, reserve, max_slots, admit_max, stage_bytes=0, stage_min=8192):
        self.device = device
        self.nv = None          # nv_tier.NvTier (QWEN38_NV=1): dynamic homes (RAM slot / placeholder), NVMe fills
        self.nv2 = None         # nv2.Nv2 (QWEN38_NV=2): device-side stall, exclusive RAM tier, CPU tier (N119)
        self.stage_min = stage_min
        self.mods = mods
        self.admit_max = admit_max
        m0 = mods[0]
        self.E = m0.num_local_experts or m0.num_experts
        self.first = int(m0.routing_first or 0)
        assert all((m.num_local_experts or m.num_experts) == self.E and int(m.routing_first or 0) == self.first for m in mods)
        sizes = [m0.multi_gate.linears[0].inner.trellis.numel() * m0.multi_gate.linears[0].inner.trellis.element_size(),
                 m0.multi_up.linears[0].inner.trellis.numel() * m0.multi_up.linears[0].inner.trellis.element_size(),
                 m0.multi_down.linears[0].inner.trellis.numel() * m0.multi_down.linears[0].inner.trellis.element_size()]
        al = lambda x: (x + 255) // 256 * 256
        self.sg, self.su, self.sd = sizes
        self.off_u = al(self.sg); self.off_d = self.off_u + al(self.su); self.rec = self.off_d + al(self.sd)
        for m in mods:
            for mm, s in ((m.multi_gate, self.sg), (m.multi_up, self.su), (m.multi_down, self.sd)):
                for l in mm.linears:
                    assert l.inner.trellis.numel() * l.inner.trellis.element_size() == s and s % 16 == 0
        torch.cuda.synchronize(device); torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        budget = os.environ.get("QWEN_VRAM_LIMIT_BYTES")
        if budget:
            free = min(free, int(budget) - (total - free))
        nst = int(stage_bytes // self.rec) if stage_bytes > 0 else 0
        self.stage_n = nst
        self.elastic = float(os.environ.get("QWEN38_EC_ELASTIC_GB", "0")) * 1024 ** 3
        self.stage = None
        if nst and not self.elastic:
            self.stage = [torch.empty((nst, self.rec), dtype=torch.uint8, device=device) for _ in range(2)]
            free -= 2 * nst * self.rec
        S = max(0, int((free - reserve) // self.rec))
        if max_slots is not None:
            S = min(S, max_slots)
        # arena in chunks of spc slots: flex chunks (the last ones) are handed back to torch during prefill (elastic)
        self.spc = int(os.environ.get("QWEN38_EC_CHUNK_SLOTS", "32"))
        nch = (S + self.spc - 1) // self.spc
        nch = S // self.spc
        S = nch * self.spc if S else 0
        self.chunks = []
        for c in range(nch):
            try:
                self.chunks.append(torch.empty((self.spc, self.rec), dtype=torch.uint8, device=device))
            except torch.OutOfMemoryError:
                break
        S = len(self.chunks) * self.spc
        self.S = S
        self.cbase = torch.tensor([t.data_ptr() for t in self.chunks] or [0], dtype=torch.int64, device=device)
        self.flex = []
        if self.elastic and self.chunks:
            nflex = min(len(self.chunks), int(-(-self.elastic // (self.spc * self.rec))))
            self.flex = list(range(len(self.chunks) - nflex, len(self.chunks)))
        self.flex_all = list(self.flex)
        self.in_prefill = False
        self.elastic_stats = {"enter": 0, "exit": 0, "realloc_fail": 0}
        L = len(mods)
        i32 = dict(dtype=torch.int32, device=device)
        self.slotof = torch.full((L, self.E), -1, **i32)
        self.owner = torch.full((max(S, 1),), -1, **i32)
        self.refb = torch.zeros((max(S, 1),), **i32)
        self.pin = torch.zeros((max(S, 1),), **i32)
        self.ctl = torch.zeros((4,), **i32)
        self.jobs = torch.zeros((2 * 1024,), **i32)
        self.stats = torch.zeros((8,), dtype=torch.int64, device=device)
        home, tabs = [], []
        self.keep = []
        for m in mods:
            pg, pu, pd = m.multi_gate.ptrs_trellis, m.multi_up.ptrs_trellis, m.multi_down.ptrs_trellis
            cap = _BC_ARGS.get(id(pg))
            gu = None
            if cap is not None:
                assert cap[0] is pg and cap[1] is pu
                gu = cap[2]
                assert gu.shape == (self.E, 2) and gu.dtype == torch.long and gu.device == pg.device
                assert torch.equal(gu[:, 0], pg) and torch.equal(gu[:, 1], pu), m.key
            else:
                print(f" !! expert_cache: no gu table for {m.key} (fused bszN path would read home)", flush=True)
            for t in (pg, pu, pd):
                assert t.dtype == torch.long and t.is_contiguous() and t.device == torch.device(device)
            home.append(torch.stack([pg, pu, pd], dim=1))
            tabs.append([pg.data_ptr(), pu.data_ptr(), pd.data_ptr(), gu.data_ptr() if gu is not None else 0])
            self.keep.append((pg, pu, pd, gu))
        self.home = torch.stack(home).contiguous()                    # [L, E, 3] host (UVA) addresses
        self.tabs = torch.tensor(tabs, dtype=torch.int64, device=device)
        # QWEN38_EC_COPY_GRID blocks for ec_copy (16 saturates the link and leaves the SMs to concurrent work)
        self.grid = int(os.environ.get("QWEN38_EC_COPY_GRID", "0")) or torch.cuda.get_device_properties(device).multi_processor_count * 8
        self.home_cpu = self.home.cpu()
        self.size_hist = {}
        if nst:
            self.cstream = torch.cuda.Stream(device)
            self.timing = bool(int(os.environ.get("QWEN38_EC_TIMING", "0")))
            self.ev_ready = [torch.cuda.Event(enable_timing=self.timing) for _ in range(2)]
            self.tev = {}   # li -> [copy_start, copy_end, hook(compute stream)] events of the current forward
            self.staged = {}          # li -> (buffer idx, expert idx tensor on device)
            self.last_li = None
            self.slot_cpu = None
            self.stage_stats = {"forwards": 0, "layers": 0, "experts": 0, "overflow": 0}
            self.pending = None
        for li, m in enumerate(mods):
            m._ec = (self, li)

    # ---- prefill staging -------------------------------------------------------------------------------------
    def _restore(self, li):
        b, idx = self.staged.pop(li)
        pg, pu, pd, gu = self.keep[li]
        h = self.home[li].index_select(0, idx)
        pg.index_copy_(0, idx, h[:, 0].contiguous()); pu.index_copy_(0, idx, h[:, 1].contiguous())
        pd.index_copy_(0, idx, h[:, 2].contiguous())
        if gu is not None:
            gu.index_copy_(0, idx, h[:, :2].contiguous())

    def _issue(self, li, b):
        """Enqueue on the copy stream: layer li's non-cached experts -> staging buffer b (after its previous user)."""
        if li >= len(self.mods):
            return
        miss = (self.slot_cpu[li] < 0).nonzero().flatten()
        n = int(miss.numel())
        take = miss[: self.stage_n]
        self.stage_stats["overflow"] += n - int(take.numel())
        runs = []
        base = self.stage[b].data_ptr()
        hc = self.home_cpu[li]
        for j, e in enumerate(take.tolist()):
            d = base + j * self.rec
            runs.append((d, int(hc[e, 0]), self.sg)); runs.append((d + self.off_u, int(hc[e, 1]), self.su))
            runs.append((d + self.off_d, int(hc[e, 2]), self.sd))
        with torch.cuda.stream(self.cstream):
            self.cstream.wait_stream(torch.cuda.current_stream(self.device))   # previous user of buffer b is done
            if self.timing:
                e0 = torch.cuda.Event(enable_timing=True); e0.record(self.cstream)
                self.tev.setdefault(li, [None, None, None])[0] = e0
            if runs:
                _ext().memcpy_runs(torch.tensor(runs, dtype=torch.int64), self.cstream.cuda_stream, self.device.index)
            self.ev_ready[b].record(self.cstream)
            if self.timing:
                e1 = torch.cuda.Event(enable_timing=True); e1.record(self.cstream)
                self.tev.setdefault(li, [None, None, None])[1] = e1
        self.pending = (li, b, take)
        self.stage_stats["layers"] += 1; self.stage_stats["experts"] += int(take.numel())

    def _stage_hook(self, li, n_picks):
        prefill = n_picks >= self.stage_min
        for l in [k for k in self.staged if k != li]:
            self._restore(l)                          # stream order: after that layer's MoE
        if not prefill:
            self.last_li = None
            if self.in_prefill:
                self._elastic_exit()
            return
        cur = torch.cuda.current_stream(self.device)
        if self.last_li is None or li <= self.last_li:  # new forward pass on this device
            if self.timing:
                self._timing_collect()
            if self.elastic and not self.in_prefill:
                self._elastic_enter()
            self.slot_cpu = self.slotof.cpu()           # stable during prefill (no admission)
            self.stage_stats["forwards"] += 1
            self.pending = None
            self._issue(li, li % 2)
        self.last_li = li
        if self.pending is None or self.pending[0] != li:
            self._issue(li, li % 2)
        _, b, take = self.pending
        if self.timing:
            eh = torch.cuda.Event(enable_timing=True); eh.record(cur)
            self.tev.setdefault(li, [None, None, None])[2] = eh
        cur.wait_event(self.ev_ready[b])
        if take.numel():
            idx = take.to(self.device, non_blocking=True)
            base = self.stage[b].data_ptr()
            a = base + torch.arange(take.numel(), dtype=torch.int64, device=self.device) * self.rec
            pg, pu, pd, gu = self.keep[li]
            pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + self.off_u); pd.index_copy_(0, idx, a + self.off_d)
            if gu is not None:
                gu.index_copy_(0, idx, torch.stack([a, a + self.off_u], 1))
            self.staged[li] = (b, idx)
        self._issue(li + 1, (li + 1) % 2)              # next layer, other buffer, overlaps this layer's compute

    def _stage_hook_nv(self, li, n_picks):
        """NV mode (nv_tier.py): as _stage_hook, but the next layer's non-VRAM experts come from their RAM slot or from
        NVMe through a pinned ring, fetched on a background thread one layer ahead. Every non-VRAM expert of a layer is
        staged (no zero-copy fallback: a non-resident expert's home is a placeholder)."""
        prefill = n_picks >= self.stage_min
        for l in [k for k in self.staged if k != li]:
            self._restore(l)                          # stream order: after that layer's MoE
        if not prefill:
            self.last_li = None
            if self.in_prefill:
                self._elastic_exit()
            return
        cur = torch.cuda.current_stream(self.device)
        if self.last_li is None or li <= self.last_li:  # new forward pass on this device
            if self.elastic and not self.in_prefill:
                self._elastic_enter()
            self.slot_np = self.slotof.cpu().numpy()    # stable during prefill (no admission)
            self.stage_stats["forwards"] += 1
            self.pending = None
        self.last_li = li
        if self.pending is None or self.pending[0] != li:
            self.pending = self.nv.stage_submit(li, li % 2)
        h = self.pending
        self.nv.stage_wait(h)
        _, b, take, _f = h
        cur.wait_event(self.ev_ready[b])
        if len(take):
            idx = torch.from_numpy(take.astype(np.int64)).to(self.device)
            base = self.stage[b].data_ptr()
            a = base + torch.arange(len(take), dtype=torch.int64, device=self.device) * self.rec
            pg, pu, pd, gu = self.keep[li]
            pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + self.off_u); pd.index_copy_(0, idx, a + self.off_d)
            if gu is not None:
                gu.index_copy_(0, idx, torch.stack([a, a + self.off_u], 1))
            self.staged[li] = (b, idx)
        self.stage_stats["layers"] += 1; self.stage_stats["experts"] += len(take)
        self.pending = self.nv.stage_submit(li + 1, (li + 1) % 2)   # next layer, other buffer, overlaps this layer

    def _ec_hits(self, li, sel):
        """Staged forwards: CLOCK hit marking only (no admission), as Pool.step with admit off."""
        _ext().ec_step(sel, self.first, li, self.E, self.S, 0, self.slotof, self.home, self.tabs, self.owner, self.refb,
                       self.pin, self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, self.off_u,
                       self.off_d)

    def _restore_nv2(self, li):
        import nv2
        b, idx = self.staged.pop(li)
        hot=getattr(self,'stage_hot',{}).pop(li,None)
        if hot is None or not len(hot):
            nv2.ext().nv_restore(idx, li, self.E, self.nv2.a["home"], self.tabs)
            return
        # The preceding MoE has consumed this layer's staging buffer. Admit its
        # highest-frequency experts directly, before that buffer is reused.
        nv2.ext().stage_end(); self._st_open=False
        source=self.keep[li][0].clone()
        selected=torch.from_numpy(np.asarray(hot,dtype=np.int64)).to(self.device)+self.first
        e=_ext()
        e.ec_step(selected,self.first,li,self.E,self.S,1,self.slotof,self.home,self.tabs,
                  self.owner,self.refb,self.pin,self.ctl,self.jobs,self.stats,self.cbase.data_ptr(),
                  self.spc,self.rec,self.off_u,self.off_d)
        e.ec_copy_stage(self.ctl,self.jobs,source,self.cbase.data_ptr(),self.spc,self.rec,self.grid)
        # Synchronizes the D2D copy before any duplicate RAM source is released.
        self.slot_np=self.slotof.cpu().numpy()
        keys=np.flatnonzero(self.slot_np.reshape(-1)>=0).astype(np.int64)
        nv2.ext().drain_all()
        nv2.ext().drop_keys(torch.from_numpy(keys))
        nv2.ext().nv_rows_all(len(self.mods),self.E,self.slotof,self.nv2.a['home'],self.tabs)
        self.stage_stats['seed_layers']=self.stage_stats.get('seed_layers',0)+1
        self.stage_stats['seed_candidates']=self.stage_stats.get('seed_candidates',0)+len(hot)

    _pfprof = os.environ.get("QWEN38_NV_PFPROF", "0") == "1"
    _pfev = []

    def _stage_hook_nv2(self, li, n_picks, selected=None):
        """N119 nv2: staged prefill. Per forward the host engine pins the layer's RAM-resident experts and reads the
        NVMe-only ones through a FIFO ring several layers ahead; per layer the copy stream H2Ds them into staging buffer
        li % 2 (GIL-free job on a background thread, one layer ahead of the compute stream)."""
        import nv2
        e = nv2.ext()
        prefill = n_picks >= self.stage_min
        selective = os.environ.get('QWEN_SELECTIVE_PREFILL','0') == '1'
        for l in [k for k in self.staged if k != li]:
            self._restore_nv2(l)                      # stream order: after that layer's MoE
        if not prefill:
            if self.last_li is not None or getattr(self, "_st_open", False):
                if self.pending is not None:
                    self.nv2.stage_wait(self.pending)
                    self.pending = None
                e.stage_end()
                self._st_open = False
            self.last_li = None
            if self.in_prefill:
                self._elastic_exit()
            return
        cur = torch.cuda.current_stream(self.device)
        if self.last_li is None or li <= self.last_li:  # new forward pass on this device
            if self.pending is not None:
                self.nv2.stage_wait(self.pending)
                self.pending = None
            if self._pfprof and self._pfev:
                torch.cuda.synchronize(self.device)
                w = [a.elapsed_time(b) for a, b in self._pfev]
                tot = self._pfev[0][0].elapsed_time(self._pfev[-1][1])
                print(f" -- nv2 prefill forward: {len(w)} layers, GPU stalled on staging {sum(w):.0f} ms of {tot:.0f} ms "
                      f"(max {max(w):.0f} ms/layer)", flush=True)
                self.stage_stats["gpu_stall_ms"] = self.stage_stats.get("gpu_stall_ms", 0) + sum(w)
                self._pfev = []
            if self.elastic and not self.in_prefill:
                t_el = time.perf_counter()
                self._elastic_enter(n_picks // self.mods[0].num_experts_per_tok)
                self.stage_stats["elastic_enter_ms"] = self.stage_stats.get("elastic_enter_ms", 0) + (time.perf_counter() - t_el) * 1e3
            self.slot_np = self.slotof.cpu().numpy()    # stable during prefill (no admission)
            t = time.perf_counter()
            if not selective:
                e.stage_begin(torch.from_numpy(np.ascontiguousarray(self.slot_np)), li)
            self._st_open = not selective
            self.stage_stats["forwards"] += 1
            self.stage_stats["begin_ms"] = self.stage_stats.get("begin_ms", 0) + (time.perf_counter() - t) * 1e3
            self.pending = None
        self.last_li = li
        if self.pending is None or self.pending[0] != li:
            if self.pending is not None:
                self.nv2.stage_wait(self.pending)
            needed=None
            if selective:
                assert selected is not None
                # Transfer an E-sized histogram, not the full [tokens, top_k] panel.
                counts=torch.bincount(selected.reshape(-1).long()-self.first,minlength=self.E).cpu().numpy()
                needed=np.flatnonzero(counts)
                if os.environ.get('QWEN_PREFILL_SEED','0')=='1':
                    if not hasattr(self,'stage_hot'): self.stage_hot={}
                    quota=sum(c is not None for c in self.chunks)*self.spc//len(self.mods)
                    tail=int(os.environ.get('QWEN_PREFILL_SEED_TAIL','32'))
                    picks=selected.reshape(-1,self.mods[0].num_experts_per_tok)[-tail:]
                    recent=torch.bincount(picks.reshape(-1).long()-self.first,minlength=self.E).cpu().numpy()
                    hot=np.argsort(-recent,kind='stable')[:quota]
                    self.stage_hot[li]=hot[recent[hot]>0]
                    # Seeding can evict old cache entries; no stale prefill snapshot.
                    self.slot_np=self.slotof.cpu().numpy()
            self.pending = self.nv2.stage_submit(li, li % 2, needed)
            self._st_open = True
        h = self.pending
        self.nv2.stage_wait(h)
        _, b, take, _f, _ev = h
        if self._pfprof:   # GPU-side stall on the staging copy: events right before / after the cross-stream wait
            e0 = torch.cuda.Event(enable_timing=True); e0.record(cur)
        cur.wait_event(self.ev_ready[b])
        if self._pfprof:
            e1 = torch.cuda.Event(enable_timing=True); e1.record(cur)
            self._pfev.append((e0, e1))
        if len(take):
            idx = torch.from_numpy(take.astype(np.int64)).to(self.device)
            base = self.stage[b].data_ptr()
            a = base + torch.arange(len(take), dtype=torch.int64, device=self.device) * self.rec
            pg, pu, pd, gu = self.keep[li]
            pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + self.off_u); pd.index_copy_(0, idx, a + self.off_d)
            if gu is not None:
                gu.index_copy_(0, idx, torch.stack([a, a + self.off_u], 1))
            self.staged[li] = (b, idx)
        self.stage_stats["layers"] += 1; self.stage_stats["experts"] += len(take)
        self.pending = None if selective else self.nv2.stage_submit(li + 1, (li + 1) % 2)
        # Full staging overlaps the next layer. Selective staging waits for its actual router,
        # avoiding speculative disk reads of unselected experts.

    def _elastic_enter(self, ntok=None):
        """Prefill: evict + disable the flex chunks' slots, hand their memory back to torch (activations of big
        chunks), allocate the staging buffers from it."""
        e = _ext()
        self.flex = list(self.flex_all)
        if ntok is not None and self.nv2 is not None and os.environ.get("QWEN38_EC_ELASTIC_DYN", "1") == "1":
            # N119: short forwards need the staging buffers plus small activations, not the whole 10 GB
            need = 2 * self.stage_n * self.rec + ntok * float(os.environ.get("QWEN38_EC_ACT_MB_PER_TOK", "0.6")) * 1e6 + 0.3e9
            nf = min(len(self.flex_all), int(-(-need // (self.spc * self.rec))))
            self.flex = self.flex_all[len(self.flex_all) - nf:]
        if self.nv2 is not None:
            self.elastic_stats["wb"] = self.elastic_stats.get("wb", 0) + self.nv2.elastic_writeback(self.flex)
        for c in self.flex:
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 1, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        torch.cuda.synchronize(self.device)
        for c in self.flex:
            self.chunks[c] = None
        if os.environ.get('QWEN_ELASTIC_TRIM', '0') == '1':
            # Released arena segments can remain reserved and cannot necessarily serve
            # larger activation allocations. Reclaim them under the total-VRAM cap.
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.stage = [torch.empty((self.stage_n, self.rec), dtype=torch.uint8, device=self.device) for _ in range(2)]
        self.in_prefill = True
        self.elastic_stats["enter"] += 1

    def _elastic_exit(self):
        """Back to decode: free staging, reallocate the flex chunks, re-enable their slots (empty)."""
        torch.cuda.synchronize(self.device)
        self.stage = None
        torch.cuda.empty_cache()
        e = _ext()
        for c in self.flex:
            try:
                t = torch.empty((self.spc, self.rec), dtype=torch.uint8, device=self.device)
            except torch.OutOfMemoryError:
                self.elastic_stats["realloc_fail"] += 1
                continue
            self.chunks[c] = t
            self.cbase[c] = t.data_ptr()
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 0, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        self.in_prefill = False
        self.elastic_stats["exit"] += 1

    def _timing_collect(self):
        """Per-layer copy (staging) vs compute ms of the last prefill forward: copy = copy-stream span of the layer's
        staging; compute = gap between consecutive layer hooks on the compute stream (MoE + attention of a layer)."""
        if not self.tev:
            return
        torch.cuda.synchronize(self.device)
        ls = sorted(self.tev)
        copy = [self.tev[l][0].elapsed_time(self.tev[l][1]) for l in ls if self.tev[l][0] and self.tev[l][1]]
        hooks = [self.tev[l][2] for l in ls if self.tev[l][2] is not None]
        comp = [hooks[i].elapsed_time(hooks[i + 1]) for i in range(len(hooks) - 1)]
        self.last_timing = {"layers": len(ls), "copy_ms_mean": round(sum(copy) / max(1, len(copy)), 2),
                            "compute_ms_mean": round(sum(comp) / max(1, len(comp)), 2),
                            "copy_ms_total": round(sum(copy), 1), "compute_ms_total": round(sum(comp), 1)}
        self.tev = {}

    def step(self, li, sel, admit):
        n = sel.numel()
        b = 1 << max(0, (n - 1).bit_length())
        self.size_hist[b] = self.size_hist.get(b, 0) + 1
        if self.stage_n:
            (self._stage_hook_nv if self.nv is not None else self._stage_hook)(li, sel.numel())
        e = _ext()
        sel = sel.reshape(-1)
        if sel.dtype != torch.long or not sel.is_contiguous():
            sel = sel.long().contiguous()
        e.ec_step(sel, self.first, li, self.E, self.S, int(admit), self.slotof, self.home, self.tabs, self.owner, self.refb, self.pin,
                  self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, self.off_u, self.off_d)
        if admit:
            e.ec_copy(self.ctl, self.jobs, li, self.E, self.home, self.cbase.data_ptr(), self.spc, self.rec, self.sg, self.su,
                      self.sd, self.off_u, self.off_d, self.grid)

    def warm(self, scores):
        """Initial fill: top S/L experts per layer by score (admission path, so tables stay consistent)."""
        k = self.S // max(1, len(self.mods))
        if k == 0:
            return
        with torch.cuda.device(self.device):
            for li, m in enumerate(self.mods):
                sc = scores.get(m.key)
                if sc is None:
                    continue
                if getattr(m, "cpu_split_first", None) is not None:
                    top = list(range(min(k, self.E)))       # static CPU split: local ids are already hot -> cold
                else:
                    loc = range(self.first, self.first + self.E)
                    top = sorted(loc, key=lambda x: -sc[x])[:k]
                if self.nv is not None:   # only experts whose bytes are in the RAM tier (home = RAM slot)
                    top = [x for x in top if self.nv.ram_slot[li * self.E + x - self.first] >= 0]
                if self.nv2 is not None:
                    if not hasattr(self, "_warm_ks"):
                        import nv2
                        self._warm_ks = nv2.ext().state()[1].numpy()
                    top = [x for x in top if self._warm_ks[li * self.E + x - self.first] == 2]
                for c in range(0, len(top), 256):
                    sel = torch.tensor(top[c:c + 256], dtype=torch.long, device=self.device)
                    self.step(li, sel, True)
            torch.cuda.synchronize(self.device)
        self.stats.zero_()

    def verify(self):
        """Full consistency check (reads every owned slot and its home copy over PCIe): bytes and pointer tables."""
        if self.nv2 is not None:
            return dict(self.nv2.verify(64), device=str(self.device))
        if self.nv is not None:
            return dict(self.nv.verify(64), device=str(self.device))
        torch.cuda.synchronize(self.device)
        o = _ext().ec_verify(self.owner, self.slotof, self.home, self.tabs, self.cbase.data_ptr(), self.spc, self.rec, self.sg,
                             self.su, self.sd, self.off_u, self.off_d, self.S, self.E, len(self.mods)).tolist()
        return {"device": str(self.device), "bad_slots": o[0], "bad_tables": o[1], "owned": o[2]}

    def summary(self):
        s = self.stats.tolist()
        return {"device": str(self.device), "layers": len(self.mods), "slots": self.S, "slot_bytes": self.rec,
                "arena_gb": round(self.S * self.rec / 1e9, 2),
                "elastic": dict(self.elastic_stats, flex_chunks=len(self.flex), flex_gb=round(len(self.flex) * self.spc * self.rec / 1e9, 2), in_prefill=self.in_prefill) if self.elastic else None,
                "mem_alloc_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 2),
                "mem_peak_gb": round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2), "hits": s[0], "misses": s[1], "admitted": s[2],
                "no_victim": s[3], "evictions": s[4],
                "hit_rate": round(s[0] / max(1, s[0] + s[1]), 4),
                "stage": dict(self.stage_stats, per_buffer=self.stage_n) if self.stage_n else None,
                "picks_hist": dict(sorted(self.size_hist.items())),
                "stage_timing_last_forward": getattr(self, "last_timing", None)}


def find_moe(modules):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    seen = []

    def walk(mm):
        if isinstance(mm, BlockSparseMLP):
            seen.append(mm)
        for c in getattr(mm, "modules", []) or []:
            walk(c)
    for top in modules:
        walk(top)
    return [m for m in seen if getattr(m, "multi_up", None) is not None and m.device is not None
            and (m.num_local_experts is None or m.num_local_experts > 0)]


def attach(model):
    pools = attach_modules(find_moe(model.modules))
    _hook_prefill(type(model))
    return pools


def _hook_prefill(cls):
    """Elastic mode: hand the flex chunks back BEFORE the first layer of a big prefill chunk runs (dense + KDA layers
    ahead of the first MoE layer already need the activation memory)."""
    if getattr(cls, "_qwen38_ec_prefill_hooked", False):
        return
    for name in ("prefill", "forward"):
        orig = getattr(cls, name)

        def wrapped(self, input_ids, params=None, _orig=orig):
            try:
                n = int(input_ids.shape[-1]) * int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 0
            except Exception:
                n = 0
            for p in POOLS:
                if p.elastic and p.stage_n and not p.in_prefill and n * p.mods[0].num_experts_per_tok >= p.stage_min:
                    p._elastic_enter(n)
            return _orig(self, input_ids, params)
        setattr(cls, name, wrapped)
    cls._qwen38_ec_prefill_hooked = True


def attach_modules(mods):
    by_dev = {}
    for m in mods:
        by_dev.setdefault(str(m.device), []).append(m)
    reserve = float(os.environ.get("QWEN38_EC_RESERVE_GB", "2.5")) * 1024 ** 3
    mx = os.environ.get("QWEN38_EC_MAX_SLOTS")
    admit_max = int(os.environ.get("QWEN38_EC_ADMIT_MAX", "64"))
    stage_bytes = float(os.environ.get("QWEN38_EC_STAGE_GB", "0")) * 1024 ** 3
    stage_min = int(os.environ.get("QWEN38_EC_STAGE_MIN", "8192"))
    warm = os.environ.get("QWEN38_EC_WARM")
    scores = json.load(open(warm)) if warm else None
    for dev, ms in by_dev.items():
        p = Pool(torch.device(dev), ms, reserve, int(mx) if mx else None, admit_max, stage_bytes, stage_min)
        POOLS.append(p)
        if os.environ.get("QWEN38_NV") == "1":   # N116: NVMe tier (RAM warm first: the VRAM warm copies from RAM)
            import nv_tier
            nv_tier.attach_pool(p)
            if scores:
                p.nv.warm(scores)
        if os.environ.get("QWEN38_NV") == "2":   # N119: nv2 (RAM warm first: the VRAM warm copies from RAM)
            import nv2
            nv2.attach_pool(p)
            if scores:
                p.nv2.warm(scores)
        if scores:
            p.warm(scores)
        if p.nv is not None:
            p.nv.post_warm()
        if p.nv2 is not None:
            p.nv2.post_warm()
            p.nv2.cpu_setup()
        for m in ms:
            orig = m.routing_fn

            def routed(bsz, cfg, z, params, _orig=orig, _m=m):
                if _m._ec[0].nv2 is not None and _m._ec[0].nv2.pfside and not params.get("autosplit_measure") \
                        and not params.get("tp_warmup"):
                    _m._ec[0].nv2.predict_early(_m._ec[1], z, bsz)
                sel, w = _orig(bsz, cfg, z, params)
                pool, li = _m._ec
                _m._nv2_cpu = False
                if not params.get("autosplit_measure") and not params.get("tp_warmup"):
                    if pool.nv2 is not None:
                        sel, w, _m._nv2_cpu = pool.nv2.layer(li, sel, w, z, bsz)
                        return sel, w
                    if pool.nv is not None:
                        pool.nv.ensure(li, sel, w)   # every pick VRAM- or RAM-resident before ec_step (stall)
                    pool.step(li, sel, sel.numel() <= pool.admit_max)
                return sel, w
            m.routing_fn = routed
            if p.nv2 is not None:
                orig_comb = m.cpu_split_combine

                def comb(fhs, cpu_partial, cpu_pending, shape, _orig=orig_comb, _m=m):
                    if getattr(_m, "_nv2_cpu", False):
                        _m._nv2_cpu = False
                        fhs = _m._ec[0].nv2.combine(fhs)
                    return _orig(fhs, cpu_partial, cpu_pending, shape)
                m.cpu_split_combine = comb
        print(f" -- expert_cache: {dev}: {len(ms)} MoE layers (experts {p.first}..{p.first + p.E - 1}), {p.S} slots x "
              f"{p.rec / 1e6:.2f} MB = {p.S * p.rec / 1e9:.2f} GB ({p.S / max(1, len(ms)):.1f}/layer), admit <= "
              f"{admit_max} picks, stage 2 x {p.stage_n} experts (>= {stage_min} picks)"
              f"{', warm ' + warm if warm else ''}", flush=True)
    return POOLS


def summary():
    return [p.summary() for p in POOLS]


def verify():
    return [p.verify() for p in POOLS]

"""N116 S1: exact NVMe expert tier for the G067 stack (QWEN38_NV=1; QWEN38_MODE=nvme in the entrypoint).

Every routed expert has one of three homes, resolved per MoE layer before its expert kernels run:
  VRAM slot  expert_cache.py Pool (CLOCK, unchanged)              slotof[key] >= 0
  RAM slot   pinned host arena of 9 MiB slots (anon mmap + cudaHostRegister, charged to the memcg), host LRU
  NVMe       /mnt/nvx store record key * 9,474,048 (first 9,437,184 B = one VRAM slot image), read with O_DIRECT
The checkpoint's routed-expert trellis bytes are never read: the loader hands every expert Linear a view of one 9 MiB
VRAM "bounce" slot (gate | up | down), so exllamav3 builds its pointer tables / per-expert BCs without the bytes.

S1 forms (PLAN.md 1.2 A-E, Stage 1):
  decode / small forwards  ensure(li, sel): host sync, every picked expert that is not VRAM-resident is made RAM-resident
                           (NVMe -> RAM slot, parallel pread pool, stall, never mask), then nv_apply writes the dynamic
                           home + the pointer-table rows of non-VRAM keys; ec_step / ec_copy run unchanged against home
  staged prefill           every non-VRAM expert of the next layer goes to the VRAM staging buffer on the copy stream:
                           RAM-resident ones straight from their slot, NVMe ones through a pinned ring (transient, never
                           enters the RAM working set), on a background thread one layer ahead
  per-expert DQ path       (exllamav3 block_sparse_mlp.py ~1246-1300, experts above the batched-recon row cap) reads the
                           per-expert Linear trellis = the bounce slot: a proxy around BC_BlockSparseMLP copies the
                           expert's live table bytes into the bounce slot first (stream order), so it is exact as well
Env: QWEN38_NV_STORE (store .bin), QWEN38_NV_RAM_GB (RAM tier; default auto = memory.max - current - margin),
     QWEN38_NV_MARGIN_GB (3), QWEN38_NV_THREADS (16), QWEN38_NV_PIECE_KB (2304), QWEN38_NV_RING (32 slots),
     QWEN38_NV_CHECK (1: device check that no pick reads the placeholder), QWEN38_NV_WARM (1: RAM warm start by score)
"""
import collections, concurrent.futures as cf, ctypes, json, mmap, os, re, threading, time
import numpy as np
import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unistd.h>
#include <errno.h>
#include <chrono>
namespace py = pybind11;

// ---------------- O_DIRECT pread pool (persistent threads) ----------------
struct Piece { int fd; long long off; long long dst; long long len; };
static std::vector<std::thread> g_thr;
static std::mutex g_mu, g_call;
static std::condition_variable g_cv, g_done_cv;
static std::vector<Piece> g_q;
static size_t g_next = 0;
static long long g_pending = 0;
static int g_err = 0;

static void worker() {
    for (;;) {
        Piece p;
        {
            std::unique_lock<std::mutex> lk(g_mu);
            g_cv.wait(lk, [] { return g_next < g_q.size(); });
            p = g_q[g_next++];
        }
        long long done = 0; int err = 0;
        while (done < p.len) {
            ssize_t r = pread(p.fd, (void*) (p.dst + done), (size_t) (p.len - done), (off_t) (p.off + done));
            if (r < 0) { if (errno == EINTR) continue; err = errno; break; }
            if (r == 0) { err = -1; break; }
            done += r;
        }
        std::lock_guard<std::mutex> lk(g_mu);
        if (err) g_err = err;
        if (--g_pending == 0) g_done_cv.notify_all();
    }
}

void io_start(int64_t n) {
    std::lock_guard<std::mutex> c(g_call);
    if (!g_thr.empty()) return;
    for (int i = 0; i < n; i++) { g_thr.emplace_back(worker); g_thr.back().detach(); }
}

static int64_t io_read_raw(int64_t fd, const int64_t* o, const int64_t* d, int64_t n, int64_t len, int64_t piece) {
    std::lock_guard<std::mutex> c(g_call);
    TORCH_CHECK(!g_thr.empty(), "io_start first");
    if (n == 0) return 0;
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_q.clear(); g_next = 0; g_err = 0;
        for (int64_t s = 0; s < len; s += piece)          // piece-major: the first pieces of all records go first
            for (int64_t i = 0; i < n; i++)
                g_q.push_back({(int) fd, o[i] + s, d[i] + s, std::min(piece, len - s)});
        g_pending = (long long) g_q.size();
    }
    g_cv.notify_all();
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_done_cv.wait(lk, [] { return g_pending == 0; });
        g_q.clear(); g_next = 0;
    }
    TORCH_CHECK(g_err == 0, "nv io_read failed, errno ", g_err);
    return n * len;
}

int64_t io_read(int64_t fd, at::Tensor offs, at::Tensor dsts, int64_t len, int64_t piece) {
    // offs, dsts: CPU int64 [n]; read len bytes at offs[i] into host address dsts[i], split into `piece` reads
    TORCH_CHECK(offs.dtype() == at::kLong && dsts.dtype() == at::kLong && offs.numel() == dsts.numel());
    return io_read_raw(fd, offs.data_ptr<int64_t>(), dsts.data_ptr<int64_t>(), offs.numel(), len, piece);
}

// One prefill staging job, GIL-free: (1) stream waits ev_wait, (2) RAM-resident experts H2D from their slots,
// (3) NVMe experts in batches of `half` ring slots: wait for that half's previous H2D, pread into the ring, H2D to the
// staging buffer, record the half's event, (4) record ev_done. Returns {ms reading, ms waiting on ring events}.
static cudaEvent_t g_ring_ev[2] = {nullptr, nullptr};
std::vector<double> stage_job(int64_t fd, int64_t rec_bytes, int64_t slot, int64_t piece, at::Tensor ram_runs,
                              at::Tensor nv_keys, at::Tensor nv_dst, at::Tensor ring, int64_t stream, int64_t ev_wait,
                              int64_t ev_done, int64_t device)
{
    c10::cuda::CUDAGuard g((c10::DeviceIndex) device);
    cudaStream_t st = (cudaStream_t) stream;
    if (!g_ring_ev[0]) for (int i = 0; i < 2; i++) cudaEventCreateWithFlags(&g_ring_ev[i], cudaEventDisableTiming);
    cudaStreamWaitEvent(st, (cudaEvent_t) ev_wait, 0);
    auto rr = ram_runs.accessor<int64_t, 2>();
    for (int64_t i = 0; i < ram_runs.size(0); i++)
        cudaMemcpyAsync((void*) rr[i][0], (const void*) rr[i][1], (size_t) rr[i][2], cudaMemcpyHostToDevice, st);
    const int64_t* k = nv_keys.data_ptr<int64_t>(); const int64_t* d = nv_dst.data_ptr<int64_t>();
    const int64_t* ra = ring.data_ptr<int64_t>();
    int64_t m = nv_keys.numel(), R = ring.numel(), half = std::max<int64_t>(1, R / 2);
    double t_read = 0, t_wait = 0; int h = 0;
    std::vector<int64_t> offs(half);
    for (int64_t c = 0; c < m; c += half) {
        int64_t nb = std::min(half, m - c);
        auto t0 = std::chrono::steady_clock::now();
        cudaEventSynchronize(g_ring_ev[h]);
        auto t1 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < nb; i++) offs[i] = k[c + i] * rec_bytes;
        io_read_raw(fd, offs.data(), ra + h * half, nb, slot, piece);
        auto t2 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < nb; i++)
            cudaMemcpyAsync((void*) d[c + i], (const void*) ra[h * half + i], (size_t) slot, cudaMemcpyHostToDevice, st);
        cudaEventRecord(g_ring_ev[h], st);
        t_wait += std::chrono::duration<double, std::milli>(t1 - t0).count();
        t_read += std::chrono::duration<double, std::milli>(t2 - t1).count();
        h ^= 1;
    }
    cudaEventRecord((cudaEvent_t) ev_done, st);
    return {t_read, t_wait};
}

// ---------------- device side ----------------
// upd [n, 2] int64 = (key, base address); home[key] = (a, a + off_u, a + off_d); rows of non-VRAM keys follow home
__global__ void nv_apply_k(const int64_t* __restrict__ upd, int n, int64_t* home, const int* __restrict__ slotof,
    const int64_t* __restrict__ tabs, int E, long long off_u, long long off_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long key = upd[2 * i]; long long a = upd[2 * i + 1];
    int l = (int) (key / E), e = (int) (key % E);
    home[key * 3 + 0] = a; home[key * 3 + 1] = a + off_u; home[key * 3 + 2] = a + off_d;
    if (slotof[key] < 0) {
        int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
        int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
        pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
        if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
    }
}

void nv_apply(at::Tensor upd, at::Tensor home, at::Tensor slotof, at::Tensor tabs, int64_t E, int64_t off_u, int64_t off_d)
{
    c10::cuda::CUDAGuard g(home.device());
    auto st = at::cuda::getCurrentCUDAStream(home.device().index());
    int n = (int) upd.size(0);
    if (n == 0) return;
    nv_apply_k<<<(n + 255) / 256, 256, 0, st>>>(upd.data_ptr<int64_t>(), n, home.data_ptr<int64_t>(), slotof.data_ptr<int>(),
        tabs.data_ptr<int64_t>(), (int) E, off_u, off_d);
}

// out[0] += picks that are not VRAM-resident and whose home is the placeholder; out[1] += picks whose table row is not
// the VRAM slot / home it should be (the rows exllamav3 will read)
__global__ void nv_check_k(const int64_t* __restrict__ sel, int n, int first, int li, int E, const int* __restrict__ slotof,
    const int64_t* __restrict__ home, const int64_t* __restrict__ tabs, long long ph, unsigned long long* out)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        long long e = sel[i] - first;
        if (e < 0 || e >= E) continue;
        long long key = (long long) li * E + e;
        if (slotof[key] >= 0) continue;
        const int64_t* pg = (const int64_t*) tabs[li * 4 + 0];
        if (home[key * 3] == ph) atomicAdd(&out[0], 1ull);
        if (pg[e] != home[key * 3]) atomicAdd(&out[1], 1ull);
    }
}

void nv_check(at::Tensor sel, int64_t first, int64_t li, int64_t E, at::Tensor slotof, at::Tensor home, at::Tensor tabs,
              int64_t ph, at::Tensor out)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    nv_check_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), (int) sel.numel(), (int) first, (int) li, (int) E,
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), ph,
        (unsigned long long*) out.data_ptr<int64_t>());
}

// copy expert e of layer li from its live pointer-table rows (VRAM slot / staging / RAM) into the bounce slot
__global__ void nv_bounce_k(const int64_t* __restrict__ tabs, int li, int e, long long dst, long long ng, long long nu,
    long long nd, long long off_u, long long off_d)
{
    const int4* sg = (const int4*) ((const int64_t*) tabs[li * 4 + 0])[e];
    const int4* su = (const int4*) ((const int64_t*) tabs[li * 4 + 1])[e];
    const int4* sd = (const int4*) ((const int64_t*) tabs[li * 4 + 2])[e];
    long long total = ng + nu + nd;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        if (i < ng) ((int4*) dst)[i] = sg[i];
        else if (i < ng + nu) ((int4*) (dst + off_u))[i - ng] = su[i - ng];
        else ((int4*) (dst + off_d))[i - ng - nu] = sd[i - ng - nu];
    }
}

void nv_bounce(at::Tensor tabs, int64_t li, int64_t e, int64_t dst, int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d)
{
    c10::cuda::CUDAGuard g(tabs.device());
    auto st = at::cuda::getCurrentCUDAStream(tabs.device().index());
    nv_bounce_k<<<164, 256, 0, st>>>(tabs.data_ptr<int64_t>(), (int) li, (int) e, dst, sg / 16, su / 16, sd / 16, off_u, off_d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("io_start", &io_start);
    m.def("io_read", &io_read, py::call_guard<py::gil_scoped_release>());
    m.def("stage_job", &stage_job, py::call_guard<py::gil_scoped_release>());
    m.def("nv_apply", &nv_apply);
    m.def("nv_check", &nv_check);
    m.def("nv_bounce", &nv_bounce);
}
"""

GiB = 1024 ** 3
_EXT = None
NV = None                 # the NvTier (one GPU)
_BOUNCE = {}              # device index -> (tensor, base address)
_LOAD = {"active": None, "marked_linears": 0, "skipped_bytes": 0, "files": set()}
_T0 = time.time()


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = os.environ.get("QWEN38_NV_BUILD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "nv"))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "nv.cu")
        if not os.path.exists(src) or open(src).read() != _SRC:
            open(src, "w").write(_SRC)
        _EXT = load(name="qwen38_nv", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


def cgroup_mem():
    """memcg numbers (GiB) + host MemAvailable (GiB)."""
    out = {}
    try:
        cg = os.environ.get("QWEN_MEMORY_CGROUP", "/sys/fs/cgroup")
        out["current"] = int(open(f"{cg}/memory.current").read()) / GiB
        if os.path.exists(f"{cg}/memory.peak"):
            out["peak"] = int(open(f"{cg}/memory.peak").read()) / GiB
        mx = open(f"{cg}/memory.max").read().strip()
        out["max"] = int(mx) / GiB if mx != "max" else None
        for line in open(f"{cg}/memory.stat"):
            k, v = line.split()
            if k in ("anon", "file", "kernel", "shmem", "unevictable", "pagetables", "slab"):
                out[k] = int(v) / GiB
    except OSError:
        pass
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            out["host_avail"] = int(line.split()[1]) / 1048576
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def _log(msg):
    m = cgroup_mem()
    print(f" -- nv_tier [{time.time() - _T0:7.1f}s] {msg} | memcg cur {m.get('current', -1):.2f} peak {m.get('peak', -1):.2f} "
          f"anon {m.get('anon', -1):.2f} file {m.get('file', -1):.2f} GiB, host avail {m.get('host_avail', -1):.1f}", flush=True)


def _manifest():
    store = os.environ.get("QWEN38_NV_STORE", "/nvx/qwen38_flash_exl3_3.05bpw_experts.bin")
    man = json.load(open(store[:-4] + ".json" if store.endswith(".bin") else store + ".json"))
    return store, man


def _rcs_bytes():
    """exllamav3's host-side recurrent-state cache (-rcs / --recurrent_cache_size GB, default 4.0) fills while serving
    (S1b: memcg OOM after it grew); the RAM tier leaves room for it."""
    import sys
    a = sys.argv
    v = float(os.environ.get("QWEN38_RCS_GB", "4.0"))
    for i, x in enumerate(a):
        if x in ("-rcs", "--recurrent_cache_size") and i + 1 < len(a):
            v = float(a[i + 1])
        elif x.startswith("--recurrent_cache_size="):
            v = float(x.split("=", 1)[1])
    return int(v * 1024 ** 3)


class HostArena:
    """Page-aligned anon memory, cudaHostRegister(Mapped) per chunk (charged to the memcg when registered)."""

    def __init__(self, nslots, slot_bytes, chunk_slots, dev_idx, tag):
        self.slot = slot_bytes
        self.n = nslots
        self.cs = chunk_slots
        self.maps, self.bases = [], []
        from exllamav3.ext import exllamav3_ext as ext
        left = nslots
        while left > 0:
            k = min(left, chunk_slots)
            mm = mmap.mmap(-1, k * slot_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            assert a % 4096 == 0
            with torch.cuda.device(dev_idx):
                r = torch.cuda.cudart().cudaHostRegister(a, k * slot_bytes, 2)
            assert int(r) == 0, f"cudaHostRegister failed ({tag}): {r}"
            t = torch.frombuffer(mm, dtype=torch.uint8, count=4096)
            assert ext.pinned_cuda_view(t, dev_idx).data_ptr() == a, "UVA alias != host address"
            self.maps.append(mm); self.bases.append(a)
            left -= k
        self.addr = np.array([self.bases[s // chunk_slots] + (s % chunk_slots) * slot_bytes for s in range(nslots)],
                             dtype=np.int64)

    def view(self, s):
        mm = self.maps[s // self.cs]
        o = (s % self.cs) * self.slot
        return torch.frombuffer(mm, dtype=torch.uint8, count=self.slot, offset=o)


# --------------------------------------------------------------------------------------------------------------------
# A. loader: expert trellis -> bounce views, nothing read
# --------------------------------------------------------------------------------------------------------------------
def install():
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.linear import Linear
    from exllamav3.loader.safetensors import SafetensorsCollection
    store, man = _manifest()
    include_mtp = os.environ.get("QWEN_INCLUDE_MTP") == "1"
    layer_keys = {L["key"] for L in man["layers"] if include_mtp or L["li"] != man.get("mtp_layer_li")}
    vs = man["vram_slot"]
    fields = {f["name"]: f for f in man["fields"]}
    offs = {"gate_proj": 0, "up_proj": vs["off_u"], "down_proj": vs["off_d"]}
    dtmap = {"torch.int16": torch.int16, "I16": torch.int16}

    def bounce(dev_idx):
        if dev_idx not in _BOUNCE:
            t = torch.zeros(vs["slot_bytes"], dtype=torch.uint8, device=f"cuda:{dev_idx}")
            _BOUNCE[dev_idx] = (t, t.data_ptr())
        return _BOUNCE[dev_idx]

    orig_split = BlockSparseMLP.cpu_maybe_split_load

    def split_then_mark(self, device, **kwargs):
        orig_split(self, device, **kwargs)
        if device is None or torch.device(device).type != "cuda" or self.key not in layer_keys:
            return
        assert self.cpu_split_first is None and not self.cpu_offload, "NV tier: no stock CPU expert split"
        dev_idx = torch.device(device).index or 0
        bounce(dev_idx)
        for mat, group in (("gate_proj", self.gates), ("up_proj", self.ups), ("down_proj", self.downs)):
            for l in group:
                assert f".{mat}" in l.key, l.key
                l._nv_slot = (dev_idx, offs[mat], fields[mat + ".trellis"]["bytes"])
                _LOAD["marked_linears"] += 1

    BlockSparseMLP.cpu_maybe_split_load = split_then_mark
    orig_lin_load = Linear.load

    def lin_load(self, device, **kwargs):
        if getattr(self, "_nv_slot", None) is None:
            return orig_lin_load(self, device, **kwargs)
        _LOAD["active"] = (self.key + ".trellis", self._nv_slot)
        try:
            return orig_lin_load(self, device, **kwargs)
        finally:
            _LOAD["active"] = None

    Linear.load = lin_load
    orig_get = SafetensorsCollection.get_tensor

    def get_tensor(self, key, device=None, *args, **kwargs):
        a = _LOAD["active"]
        if a is None or key != a[0]:
            fn = self.tensor_file_map.get(key) if hasattr(self, "tensor_file_map") else None
            if fn:
                _LOAD["files"].add(os.path.join(self.directory, fn) if not os.path.isabs(fn) else fn)
            return orig_get(self, key, device, *args, **kwargs)
        dev_idx, off, sz = a[1]
        meta = self.find_stc(key).get_tensor_meta(key)[key]
        assert meta["n_bytes"] == sz, (key, meta, sz)
        dt = dtmap[meta["dtype"]]
        t, _ = _BOUNCE[dev_idx]
        _LOAD["skipped_bytes"] += sz
        return t[off: off + sz].view(dt).view(meta["shape"])

    SafetensorsCollection.get_tensor = get_tensor
    _ext()
    _log(f"installed: store {store}, {len(layer_keys)} routed layers, loader skips expert trellis bytes")


# --------------------------------------------------------------------------------------------------------------------
# B-E. the tier
# --------------------------------------------------------------------------------------------------------------------
class NvTier:
    def __init__(self, pool):
        self.pool = pool
        self.dev = pool.device
        self.store, man = _manifest()
        self.rec_bytes = man["record_bytes"]
        self.slot = man["vram_slot"]["slot_bytes"]
        assert (pool.sg, pool.su, pool.sd) == (3145728, 3145728, 3145728) and pool.rec == self.slot
        assert pool.off_u == man["vram_slot"]["off_u"] and pool.off_d == man["vram_slot"]["off_d"]
        assert pool.first == 0 and pool.E == man["experts_per_layer"]
        li_of = {L["key"]: L["li"] for L in man["layers"]}
        self.store_li = [li_of[m.key] for m in pool.mods]
        assert self.store_li == list(range(len(pool.mods))), "pool layer order != store layer order"
        self.L, self.E = len(pool.mods), pool.E
        self.K = self.L * self.E
        e = _ext()
        self.threads = int(os.environ.get("QWEN38_NV_THREADS", "16"))
        self.piece = int(os.environ.get("QWEN38_NV_PIECE_KB", "2304")) * 1024
        assert self.piece % 4096 == 0
        e.io_start(self.threads)
        self.fd = os.open(self.store, os.O_RDONLY | os.O_DIRECT)
        dev_idx = self.dev.index or 0
        # placeholder home: zeros in VRAM (never the bytes of any expert; nv_check counts reads of it)
        self.ph_t = torch.zeros(self.slot, dtype=torch.uint8, device=self.dev)
        self.ph = self.ph_t.data_ptr()
        self.bounce_t, self.bounce = _BOUNCE[dev_idx]
        # RAM tier size
        ring = int(os.environ.get("QWEN38_NV_RING", "32"))
        margin = float(os.environ.get("QWEN38_NV_MARGIN_GB", "3")) * GiB
        m = cgroup_mem()
        want = os.environ.get("QWEN38_NV_RAM_GB", "auto")
        if want == "auto":
            assert m.get("max"), "QWEN38_NV_RAM_GB=auto needs a memory-capped container"
            avail = (m["max"] - m["current"]) * GiB - margin - ring * self.slot - _rcs_bytes()
            nram = max(0, int(avail // self.slot))
        else:
            nram = int(float(want) * GiB // self.slot)
        _log(f"(rcs reserve {_rcs_bytes() / GiB:.1f} GiB, margin {margin / GiB:.1f} GiB) before RAM arena: {nram} RAM slots ({nram * self.slot / GiB:.2f} GiB), ring {ring} slots")
        self.ram = HostArena(nram, self.slot, 512, dev_idx, "ram") if nram else None
        self.nram = nram
        self.ring_n = ring
        self.ring = HostArena(ring, self.slot, ring, dev_idx, "ring") if ring else None
        if getattr(pool, "ev_ready", None) is not None:
            for e_ in pool.ev_ready:
                e_.record(torch.cuda.current_stream(self.dev))   # materialise the CUDA events (stage_job records them)
        _log(f"RAM arena registered: {nram} x {self.slot} B")
        # host state
        self.ram_slot = np.full(self.K, -1, dtype=np.int32)
        self.owner = np.full(max(nram, 1), -1, dtype=np.int64)
        self.free = list(range(nram - 1, -1, -1))
        self.lru = collections.OrderedDict()          # key -> slot, oldest first
        self.h_sel = torch.empty(65536, dtype=torch.int64, pin_memory=True)
        self.h_slot = torch.empty(self.E, dtype=torch.int32, pin_memory=True)
        self.check = os.environ.get("QWEN38_NV_CHECK", "1") == "1"
        self.trace = [] if os.environ.get("QWEN38_NV_TRACE") else None
        self.pf_transient = os.environ.get("QWEN38_NV_PF_TRANSIENT", "1") == "1"
        self.h_w = torch.empty(4096, dtype=torch.float32, pin_memory=True)
        self.chk = torch.zeros(4, dtype=torch.int64, device=self.dev)
        self.bg = cf.ThreadPoolExecutor(1, thread_name_prefix="nv-stage")
        self.st = collections.Counter()
        self.lat = collections.defaultdict(list)      # NVMe batch read latencies (ms) by phase
        # all homes and rows -> placeholder (Pool's tables currently hold the bounce addresses)
        self._apply([(k, self.ph) for k in range(self.K)])
        torch.cuda.synchronize(self.dev)
        pool.nv = self

    # ---- primitives --------------------------------------------------------------------------------------------
    def _apply(self, upd):
        if not upd:
            return
        u = torch.tensor(upd, dtype=torch.int64).to(self.dev)
        p = self.pool
        _ext().nv_apply(u, p.home, p.slotof, p.tabs, self.E, p.off_u, p.off_d)
        hc = p.home_cpu.view(-1, 3)
        for k, a in upd:   # host mirror (diagnostics only; staging uses ram_slot)
            hc[k, 0] = a; hc[k, 1] = a + p.off_u; hc[k, 2] = a + p.off_d

    def _read(self, keys, dsts, phase):
        """O_DIRECT pread of the trellis part of records `keys` into host addresses `dsts` (parallel pool)."""
        t = time.perf_counter()
        n = _ext().io_read(self.fd, torch.tensor([int(k) * self.rec_bytes for k in keys], dtype=torch.int64),
                           torch.tensor([int(d) for d in dsts], dtype=torch.int64), self.slot, self.piece)
        ms = (time.perf_counter() - t) * 1e3
        self.st[phase + "_nvme_reads"] += len(keys); self.st[phase + "_nvme_bytes"] += n
        self.st[phase + "_nvme_ms"] += ms; self.st[phase + "_nvme_batches"] += 1
        lt = self.lat[phase]
        if len(lt) < 200000:
            lt.append(round(ms, 3))
        return ms

    def _alloc(self, n, protect):
        """n RAM slots: free list first, then LRU victims (never a protected key). Returns (slots, evicted keys)."""
        slots, ev = [], []
        while len(slots) < n and self.free:
            slots.append(self.free.pop())
        guard = 0
        while len(slots) < n:
            k, s = self.lru.popitem(last=False)
            if k in protect:
                self.lru[k] = s
                guard += 1
                assert guard <= len(self.lru) + 1, "RAM tier smaller than one layer's working set"
                continue
            self.ram_slot[k] = -1; self.owner[s] = -1
            ev.append(k); slots.append(s)
        return slots, ev

    def _fill(self, keys, protect, phase):
        """Make `keys` (not RAM-resident) RAM-resident. Returns the home updates."""
        slots, ev = self._alloc(len(keys), protect)
        upd = [(k, self.ph) for k in ev]
        self.st[phase + "_ram_evictions"] += len(ev)
        ms = 0.0
        for c in range(0, len(keys), 256):
            ms += self._read(keys[c:c + 256], [self.ram.addr[s] for s in slots[c:c + 256]], phase)
        for k, s in zip(keys, slots):
            self.ram_slot[k] = s; self.owner[s] = k; self.lru[k] = s
            upd.append((k, int(self.ram.addr[s])))
        return upd, ms

    # ---- warm start --------------------------------------------------------------------------------------------
    def warm(self, scores):
        """RAM: per layer the top nram/L experts by routing score (a superset of the VRAM warm set)."""
        if not self.nram:
            return
        k = self.nram // self.L
        keys = []
        for li, m in enumerate(self.pool.mods):
            sc = scores.get(m.key)
            order = sorted(range(self.E), key=lambda x: -sc[x]) if sc is not None else list(range(self.E))
            keys += [li * self.E + e for e in order[:k]]
        t = time.perf_counter()
        upd, ms = self._fill(keys, set(), "warm")
        self._apply(upd)
        torch.cuda.synchronize(self.dev)
        _log(f"RAM warm: {len(keys)} experts ({k}/layer) in {time.perf_counter() - t:.1f} s (NVMe {ms / 1e3:.1f} s, "
             f"{len(keys) * self.slot / max(ms, 1e-3) / 1e6:.1f} GB/s)")

    def post_warm(self):
        """VRAM-resident experts go to the LRU head (evicted first): their RAM copy is the redundant one."""
        so = self.pool.slotof.flatten().cpu().numpy()
        n = 0
        for k in list(self.lru.keys()):
            if so[k] >= 0:
                self.lru.move_to_end(k, last=False); n += 1
        self.st.clear(); self.lat.clear()
        self.chk.zero_()
        _log(f"post-warm: {n} RAM-resident experts are also VRAM-resident (moved to the LRU head)")

    # ---- decode / small forwards -------------------------------------------------------------------------------
    def ensure(self, li, sel, w=None):
        p = self.pool
        n = sel.numel()
        if p.stage_n and n >= p.stage_min:
            self.st["staged_layer_calls"] += 1
            return
        phase = "dec" if n <= p.admit_max else "pf"
        t0 = time.perf_counter()
        flat = sel.reshape(-1)
        if n <= self.h_sel.numel():
            self.h_sel[:n].copy_(flat, non_blocking=True)
            self.h_slot.copy_(p.slotof[li], non_blocking=True)
            tr = self.trace is not None and phase == "dec" and w is not None and n <= self.h_w.numel()
            if tr:
                self.h_w[:n].copy_(w.reshape(-1).float(), non_blocking=True)
            torch.cuda.current_stream(self.dev).synchronize()
            if tr:   # route capture (decode): step, layer, top-k ids + router weights (cheap, host side)
                self.trace.append((self.st["dec_steps"] + (li == 0), li, self.h_sel[:n].numpy().astype(np.int16).copy(),
                                   self.h_w[:n].numpy().astype(np.float32).copy()))
            s_np = self.h_sel[:n].numpy()
            row = self.h_slot.numpy()
        else:
            s_np = flat.cpu().numpy(); row = p.slotof[li].cpu().numpy()
        t1 = time.perf_counter()
        ex = np.unique(s_np) - p.first
        ex = ex[(ex >= 0) & (ex < self.E)]
        miss = ex[row[ex] < 0]
        keys = li * self.E + miss
        rs = self.ram_slot[keys]
        hit = keys[rs >= 0]
        nv = [int(k) for k in keys[rs < 0]]
        for k in hit:
            self.lru.move_to_end(int(k))
        self.st[phase + "_calls"] += 1
        self.st[phase + "_picks_unique"] += len(ex)
        self.st[phase + "_vram_hits"] += len(ex) - len(miss)
        self.st[phase + "_ram_hits"] += len(hit)
        self.st[phase + "_nvme_misses"] += len(nv)
        if li == 0 and phase == "dec":
            self.st["dec_steps"] += 1
        if nv:
            upd, _ = self._fill(nv, set(int(k) for k in hit), phase)
            self._apply(upd)
            if phase == "pf" and self.pf_transient:   # I4: prefill fills are evicted first (decode working set kept)
                for k in nv:
                    self.lru.move_to_end(k, last=False)
        if self.check:
            _ext().nv_check(flat if flat.dtype == torch.long else flat.long(), p.first, li, self.E, p.slotof, p.home,
                            p.tabs, self.ph, self.chk)
        t2 = time.perf_counter()
        self.st[phase + "_sync_ms"] += (t1 - t0) * 1e3
        self.st[phase + "_ensure_ms"] += (t2 - t0) * 1e3

    # ---- staged prefill (called from expert_cache.Pool._stage_hook_nv on the main thread) -----------------------
    def stage_submit(self, li, b):
        p = self.pool
        if li >= self.L:
            return None
        miss = np.nonzero(p.slot_np[li] < 0)[0]
        assert len(miss) <= p.stage_n, f"staging overflow {len(miss)} > {p.stage_n} (no zero-copy fallback in NV mode)"
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.dev))         # previous user of buffer b (layer li-2) is enqueued
        self.st["stage_layers"] += 1
        return (li, b, miss, self.bg.submit(self._stage_job, li, b, miss, ev))

    def _stage_job(self, li, b, miss, ev):
        p = self.pool
        base = p.stage[b].data_ptr()
        runs, nk, nd = [], [], []
        for j, e in enumerate(miss.tolist()):
            k = li * self.E + e
            s = self.ram_slot[k]
            if s >= 0:
                runs.append((base + j * self.slot, int(self.ram.addr[s]), self.slot))
            else:
                nk.append(k); nd.append(base + j * self.slot)
        t = time.perf_counter()
        rd, wt = _ext().stage_job(self.fd, self.rec_bytes, self.slot, self.piece,
                                  torch.tensor(runs or [(0, 0, 0)][:0], dtype=torch.int64).reshape(-1, 3),
                                  torch.tensor(nk, dtype=torch.int64), torch.tensor(nd, dtype=torch.int64),
                                  torch.tensor(self.ring.addr.tolist(), dtype=torch.int64), p.cstream.cuda_stream,
                                  ev.cuda_event, p.ev_ready[b].cuda_event, self.dev.index)
        self.st["stage_from_ram"] += len(runs)
        self.st["stage_from_nvme"] += len(nk)
        self.st["stage_nvme_reads"] += len(nk); self.st["stage_nvme_bytes"] += len(nk) * self.slot
        self.st["stage_nvme_ms"] += rd; self.st["stage_ring_wait_ms"] += wt
        self.st["stage_job_ms"] += (time.perf_counter() - t) * 1e3

    def stage_wait(self, handle):
        t = time.perf_counter()
        handle[3].result()
        self.st["stage_wait_ms"] += (time.perf_counter() - t) * 1e3

    # ---- checks / stats ----------------------------------------------------------------------------------------
    def verify(self, n=64, seed=0):
        """I1: sampled VRAM slots and RAM slots byte-compared against fresh O_DIRECT store reads; table/home check."""
        import random
        p = self.pool
        torch.cuda.synchronize(self.dev)
        rnd = random.Random(seed)
        owner = p.owner.cpu().numpy()
        vs = [s for s in range(p.S) if owner[s] >= 0 and p.chunks[s // p.spc] is not None]
        rk = list(self.lru.keys())
        bad_v = bad_r = 0
        if getattr(self, "_vtmp", None) is None:
            self._vtmp = HostArena(1, self.slot, 1, self.dev.index or 0, "verify")
        tmp = self._vtmp
        for s in rnd.sample(vs, min(n, len(vs))):
            self._read([int(owner[s])], [tmp.addr[0]], "verify")
            v = p.chunks[s // p.spc][s % p.spc][: self.slot]
            bad_v += int(not torch.equal(v.cpu(), tmp.view(0)))
        for k in rnd.sample(rk, min(n, len(rk))):
            self._read([k], [tmp.addr[0]], "verify")
            bad_r += int(not torch.equal(self.ram.view(int(self.ram_slot[k])), tmp.view(0)))
        # tables: VRAM rows == slot, other rows == home; homes == RAM slot or placeholder
        cb = p.cbase.cpu().tolist()
        slotof = p.slotof.flatten().cpu().numpy()
        home = p.home.view(-1, 3).cpu().numpy()
        bad_t = 0
        for li in range(self.L):
            pg = p.keep[li][0].cpu().numpy()
            for e in range(self.E):
                k = li * self.E + e
                s = int(slotof[k])
                want = (cb[s // p.spc] + (s % p.spc) * p.rec) if s >= 0 else int(home[k, 0])
                bad_t += int(pg[e] != want)
        rsl = self.ram_slot
        hres = sum(1 for k in range(self.K) if rsl[k] >= 0 and home[k, 0] != self.ram.addr[rsl[k]])
        hnon = sum(1 for k in range(self.K) if rsl[k] < 0 and home[k, 0] != self.ph)
        self.st["verify_runs"] += 1
        return {"vram_checked": min(n, len(vs)), "vram_bad": bad_v, "ram_checked": min(n, len(rk)), "ram_bad": bad_r,
                "table_bad": int(bad_t), "home_bad_resident": int(hres), "home_bad_nonresident": int(hnon)}

    def summary(self):
        st = dict(self.st)
        c = self.chk.tolist()
        r = lambda x: round(x, 3)
        d = max(1, st.get("dec_steps", 0))
        dm = st.get("dec_vram_hits", 0) + st.get("dec_ram_hits", 0) + st.get("dec_nvme_misses", 0)
        out = {"ram_slots": self.nram, "ram_gib": r(self.nram * self.slot / GiB), "ram_resident": len(self.lru),
               "threads": self.threads, "piece_kb": self.piece // 1024, "ring_slots": self.ring_n,
               "violations_placeholder_reads": c[0], "violations_table_rows": c[1],
               "counters": {k: (r(v) if isinstance(v, float) else v) for k, v in st.items()},
               "decode": {"steps": st.get("dec_steps", 0),
                          "unique_picks_per_step": r(st.get("dec_picks_unique", 0) / d),
                          "vram_hits_per_step": r(st.get("dec_vram_hits", 0) / d),
                          "ram_hits_per_step": r(st.get("dec_ram_hits", 0) / d),
                          "nvme_reads_per_step": r(st.get("dec_nvme_misses", 0) / d),
                          "ram_hit_rate_of_vram_misses": r(st.get("dec_ram_hits", 0) / max(1, st.get("dec_ram_hits", 0) + st.get("dec_nvme_misses", 0))),
                          "vram_hit_rate": r(st.get("dec_vram_hits", 0) / max(1, dm)),
                          "ensure_ms_per_step": r(st.get("dec_ensure_ms", 0) / d),
                          "sync_ms_per_step": r(st.get("dec_sync_ms", 0) / d),
                          "nvme_ms_per_step": r(st.get("dec_nvme_ms", 0) / d)},
               "memcg": cgroup_mem()}
        for ph, lt in self.lat.items():
            if lt:
                s = sorted(lt)
                out.setdefault("nvme_batch_ms", {})[ph] = {"n": len(s), "p50": s[len(s) // 2], "p99": s[int(0.99 * (len(s) - 1))],
                                                          "max": s[-1]}
        return out


class BCProxy:
    """Wraps BC_BlockSparseMLP: the per-expert paths (DQ reconstruct, single-expert graph) read the per-expert Linear
    trellis, which in NV mode is the bounce slot. Copy the expert's live bytes there first (same stream)."""

    def __init__(self, bc, li, nv):
        object.__setattr__(self, "_bc", bc)
        object.__setattr__(self, "_li", li)
        object.__setattr__(self, "_nv", nv)

    def _fill(self, e):
        nv, p = self._nv, self._nv.pool
        _ext().nv_bounce(p.tabs, self._li, int(e), nv.bounce, p.sg, p.su, p.sd, p.off_u, p.off_d)

    def run_single_expert_dq(self, y, e, *a):
        self._nv.st["dq_bounce_calls"] += 1
        self._fill(e)
        return self._bc.run_single_expert_dq(y, e, *a)

    def run_single_expert(self, y, e, *a):
        self._nv.st["graph_bounce_calls"] += 1
        self._fill(e)
        return self._bc.run_single_expert(y, e, *a)

    def __getattr__(self, k):
        return getattr(self._bc, k)


def _drop_checkpoint_cache():
    nd = 0
    for f in sorted(_LOAD["files"]):
        try:
            fd = os.open(f, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); nd += 1
        except OSError:
            pass
    return nd


def attach_pool(pool):
    global NV
    nd = _drop_checkpoint_cache()
    _log(f"model loaded + VRAM pool built; DONTNEED on {nd} checkpoint files")
    NV = NvTier(pool)
    return NV


def finish(model):
    """After expert_cache.attach: BC proxies, drop the checkpoint page cache, report memory."""
    nv = NV
    for li, m in enumerate(nv.pool.mods):
        assert m.bc is not None and m.support_quant_paths, m.key
        m.bc = BCProxy(m.bc, li, nv)
    nd = _drop_checkpoint_cache()
    _log(f"ready: {_LOAD['marked_linears']} expert linears on the bounce slot, {_LOAD['skipped_bytes'] / 1e9:.1f} GB of "
         f"checkpoint expert bytes never read, DONTNEED on {nd} shard files")
    if os.environ.get("QWEN38_NV_VERIFY_START", "1") == "1":
        print(f" -- nv_tier startup verify: {nv.verify(32)}", flush=True)


def save_trace(path=None):
    """Decode route capture -> npz: step [N], layer [N], ids [N, k], weights [N, k] (rows of one decode step per layer;
    bsz>1 steps keep all picks of the step flattened)."""
    nv = NV
    if nv is None or nv.trace is None:
        return {"error": "QWEN38_NV_TRACE off"}
    path = path or os.environ["QWEN38_NV_TRACE"]
    tr = list(nv.trace)
    if not tr:
        return {"rows": 0}
    k = max(len(t[2]) for t in tr)
    ids = np.full((len(tr), k), -1, np.int16); ws = np.zeros((len(tr), k), np.float32)
    for i, t in enumerate(tr):
        ids[i, :len(t[2])] = t[2]; ws[i, :len(t[3])] = t[3]
    np.savez_compressed(path, step=np.array([t[0] for t in tr], np.int32), layer=np.array([t[1] for t in tr], np.int16),
                        ids=ids, weights=ws, model_layer_offset=3)
    return {"rows": len(tr), "path": path}


def summary():
    return NV.summary() if NV is not None else None

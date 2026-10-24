// N119 nv2 host engine (one per GPU process). Owns every residency decision ("only the host marks an expert
// resident, and only after its bytes have landed"):
//   controller thread  polls the device's per-layer requests (mapped ring), drains the device's write-back /
//                      admission logs (exclusive RAM tier), runs plan_layer (moetier semantics: lanes gpu / zerocopy /
//                      cpu / nvme, greedy min-max split, NVMe stall), issues O_DIRECT NVMe reads into fresh RAM slots,
//                      writes the reply (lanes), dispatches the CPU job, keeps the landing ring + free list stocked
//   reader pool        pthreads doing O_DIRECT preads of record pieces (decode reads ahead of prefill read-ahead);
//                      the thread that finishes a record's last piece marks it landed (ram_res = 1, after the bytes)
//   CPU worker         AVX2 MUL1 kernels (ft_core.h) computing RAM-resident experts IN PLACE from their RAM slot
//                      (native layout), slots pinned (refcount) for the job; persistent pool (no per-layer spin-up)
//   stage engine       prefill: per forward, pins the RAM-resident experts, reads the NVMe-only ones ahead (FIFO ring,
//                      bounded queue depth, several layers ahead) and H2Ds them into the VRAM staging buffers
// Eviction never touches a key of the request being served, a pinned slot, an in-flight read, or a landing slot.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <execinfo.h>
#include <signal.h>
#include "ft_core.h"
#include "ft_n135.h"
// N135 CPU-lane forward switch: QWEN38_NV_CPU_KERN=1 -> n135::moe_forward4 (ft_n135.h), default 0 -> ft_core moe_forward
static int g_cpu_kern = -1;
static inline void cpu_kern_init()
{
    if (g_cpu_kern >= 0) return;
    const char* s = getenv("QWEN38_NV_CPU_KERN");
    g_cpu_kern = s && *s ? atoi(s) : 0;
    n135::configure_env(); n135::init_p16();
    fprintf(stderr, "[nv2] CPU lane kernel: %s (rg %d cg %d rd %d cd %d bandmaxm %d pfr %d p16 %d rt %d merge %d)\n", g_cpu_kern == 1 ? "n135 v4" : "ft_core moe_forward",
            n135::K_.rg, n135::K_.cg, n135::K_.rd, n135::K_.cd, n135::K_.bandmaxm, n135::K_.pfr, n135::K_.kern, n135::K_.rt, n135::K_.merge);
}
static inline void cpu_moe_forward(Pool& pool, const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route,
                                   float* out, int mode, PhaseTimes* pt = nullptr)
{
    cpu_kern_init();
    if (g_cpu_kern == 1) n135::moe_forward4(pool, L, x, ntok, route, out, mode);
    else moe_forward(pool, L, x, ntok, route, out, mode, pt);
}
#include "nv2_shared.h"
#include <sys/mman.h>
#include <fcntl.h>

using namespace nv2s;
void dev_bind(pybind11::module& m);

namespace nv2h {

// N137 B70 ring layout (b70/ring.py): HDR int64[512] | KEYS int32[16384] | REQ | X fp16[MAXP][H] | OUT fp16[MAXP][H]
constexpr int64_t B70_MAGIC = 0x4E31333752494E47LL;
constexpr int64_t B70_OFF_KEYS = 4096, B70_OFF_REQ = 69632, B70_OFF_X = 73728, B70_OFF_OUT = 73728 + 64LL * 4096 * 2;
constexpr int64_t B70_SIZE = ((B70_OFF_OUT + 64LL * 4096 * 2 + 4095) / 4096) * 4096;
enum { B70_W_MAGIC = 0, B70_W_STATE = 4, B70_W_LOAD_SEQ = 6, B70_W_LOAD_ACK = 7, B70_W_NKEYS = 8, B70_W_REQ_SEQ = 9,
       B70_W_DONE_SEQ = 10, B70_W_ERR = 11, B70_W_NSLOTS = 18, B70_W_HEARTBEAT = 19 };

static inline long long realtime_ns() { struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec; }
static inline double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static inline void cpu_relax() { _mm_pause(); }
static void pin_self(int cpu) { if (cpu < 0) return; cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
static void pin_self_set(const std::vector<int>& cpus) { if (cpus.empty()) return; cpu_set_t s; CPU_ZERO(&s); for (int c : cpus) CPU_SET(c, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }

// ------------------------------------------------------------------------------------------------------------
// reader pool
// ------------------------------------------------------------------------------------------------------------
typedef void (*DoneFn)(void* ctx, int64_t tag, int err, double ms);
struct Track { std::atomic<int> left{0}; std::atomic<int> err{0}; DoneFn fn = nullptr; void* ctx = nullptr; int64_t tag = 0; double t0 = 0; };
struct Piece { int fd; int64_t off, dst, len; Track* tr; };

struct Reader
{
    std::vector<std::thread> th;
    std::mutex mu; std::condition_variable cv;
    std::deque<Piece> hi, lo;
    bool quit = false;
    std::atomic<long> recs{0}, pieces{0};
    void start(int n, std::vector<int> cpus)
    {
        for (int i = 0; i < n; ++i) th.emplace_back([this, cpus] { pin_self_set(cpus); loop(); });
    }
    void submit(int fd, int64_t off, int64_t dst, int64_t len, int64_t piece, bool high, DoneFn fn, void* ctx, int64_t tag)
    {
        Track* tr = new Track; tr->fn = fn; tr->ctx = ctx; tr->tag = tag; tr->t0 = now_ms();
        const int np = int((len + piece - 1) / piece);
        tr->left = np;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto& q = high ? hi : lo;
            for (int64_t s = 0; s < len; s += piece) q.push_back({ fd, off + s, dst + s, std::min(piece, len - s), tr });
        }
        recs++;
        for (int i = 0; i < np; ++i) cv.notify_one();
    }
    void loop()
    {
        for (;;)
        {
            Piece p;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return quit || !hi.empty() || !lo.empty(); });
                if (quit) return;
                if (!hi.empty()) { p = hi.front(); hi.pop_front(); } else { p = lo.front(); lo.pop_front(); }
            }
            int64_t done = 0; int err = 0;
            while (done < p.len)
            {
                ssize_t r = pread(p.fd, (void*) (p.dst + done), size_t(p.len - done), off_t(p.off + done));
                if (r < 0) { if (errno == EINTR) continue; err = errno; break; }
                if (r == 0) { err = -1; break; }
                done += r;
            }
            pieces++;
            if (err) p.tr->err = err;
            if (p.tr->left.fetch_sub(1) == 1)
            {
                Track* tr = p.tr;
                tr->fn(tr->ctx, tr->tag, tr->err.load(), now_ms() - tr->t0);
                delete tr;
            }
        }
    }
    void stop() { { std::lock_guard<std::mutex> lk(mu); quit = true; } cv.notify_all(); for (auto& t : th) t.join(); th.clear(); }
};

// ------------------------------------------------------------------------------------------------------------
// the engine
// ------------------------------------------------------------------------------------------------------------
enum { KS_NONE = 0, KS_INFLIGHT = 1, KS_RES = 2 };
enum { SL_FREE = -1, SL_LAND = -2, SL_STAGE = -3 };

struct Pol
{
    double g0 = 0.17, thit = 0.023, tzc = 0.38, ca = 0.145, cb = 0.165, ctok = 0.040, push = 0.40;
    double nvlat = 0.12, nvdeep = 0.395, nvone = 0.75;
    int cpu_on = 0, nvcpu_on = 0, maxcpu = 32, nv_noadmit = 0;
};

struct Counters
{
    std::atomic<long long> reqs{0}, dec_reqs{0}, small_reqs{0}, picks{0}, vram{0}, ram{0}, nvme{0}, inflight_hit{0},
        lane_zc{0}, lane_cpu{0}, lane_nvg{0}, lane_nvc{0}, reads{0}, read_bytes{0}, wb_landed{0}, wb_dup{0}, adm_freed{0},
        adm_pinned{0}, evictions{0}, cpu_jobs{0}, cpu_experts{0}, cpu_tokens{0}, stage_ram{0}, stage_nv{0},
        stage_reads{0}, read_err{0}, alloc_fail{0}, land_pushed{0}, prefetch{0}, dec_tok{0}, wb_issued{0}, wb_drop{0}, vring_hits{0}, notice_ns{0}, notice_max_ns{0}, serve_max_ns{0};
    std::atomic<long long> lane_b70{0}, b70_jobs{0}, b70_rows{0}, b70_wait_ns{0}, b70_rtt_ns{0}, b70_err{0}, b70_rtt_max_ns{0};
    std::atomic<long long> reply_ns{0}, plan_ns{0}, ctl_busy_ns{0}, cpu_busy_ns{0}, cpu_wait_land_ns{0}, read_ms_x1000{0},
        stage_wait_ns{0}, stage_job_ns{0}, wb_bulk{0};
};

struct Engine;
static Engine* G = nullptr;

struct StageEntry { int li; int j; int key; int ring; std::atomic<int> ready{0}; };

struct Engine
{
    // geometry
    int L = 0, E = 0, K = 0, H = 4096, I = 2048;
    int64_t rec_bytes = 0, slot_bytes = 0, piece = 0, off_u = 0, off_d = 0;
    int fd = -1;
    int64_t ph = 0;
    // mapped region
    volatile int64_t* hc = nullptr; Req* req = nullptr; Rep* rep = nullptr; LogE* wbl = nullptr; LogE* adml = nullptr;
    int32_t* land = nullptr; volatile int64_t* home = nullptr; volatile int32_t* ram_res = nullptr; uint16_t* hx = nullptr; float* hout = nullptr;
    // RAM tier
    int nram = 0; std::vector<int64_t> ram_addr;
    std::vector<int32_t> slot_of; std::vector<int8_t> kst; std::vector<double> karr;
    std::vector<int32_t> owner, pinc, lprev, lnext; int lru_old = -1, lru_new = -1;
    std::vector<int32_t> freel;
    std::vector<uint8_t> prot; std::vector<int32_t> protl;
    std::vector<float> score;
    std::mutex mu; std::condition_variable land_cv;
    int64_t land_tail = 0;
    struct WbP { int key, s, v; bool issued = false; };
    std::deque<WbP> wb_q; std::mutex wb_mu; std::condition_variable wb_cv; std::atomic<int> wb_inflight{0};
    std::thread wb_th;
    std::vector<int32_t> kvr;   // key -> VRAM ring slot backing it (write-back D2H pending) or -1
    std::vector<int> vfree; std::vector<int64_t> vring_addr; std::vector<cudaEvent_t> vr_ev;
    int land_target = 64, free_lo = 96;
    bool exclusive = true;
    // NVMe channel model (plan arrival estimates)
    double nv_free_at = 0;
    // threads
    Reader rd;
    std::thread ctl_th, cpu_th;
    std::atomic<bool> quit{false};
    int64_t next_seq = 1;
    long long t_reply = 0;
    std::vector<long long> dbg_tn = std::vector<long long>(DBG, 0), dbg_tr = std::vector<long long>(DBG, 0), dbg_seq = std::vector<long long>(DBG, 0);
    int64_t wb_read = 0, adm_read = 0;
    int ctl_cpu = -1, dev_index = 0, wb_cpu = -1;
    Pol pol;
    Counters C;
    // CPU tier
    bool cpu_ready = false;
    Pool pool;
    std::vector<Layer> layers;
    std::vector<torch::Tensor> keep;
    int cpu_mode = -1;
    std::atomic<int64_t> cpu_job_seq{0};
    struct CJob { int64_t seq; int li, ntok, np; int e[MAXP], t[MAXP], lane[MAXP]; float w[MAXP]; } cj;
    // N137 B70 expert tier: static resident key set on the B70 server, shared host ring (b70/ring.py layout)
    std::vector<int8_t> b70k; bool b70_on = false; uint8_t* b70r = nullptr; int64_t b70_seq = 0; int64_t b70_timeout_ns = 2000000000LL;
    std::vector<float> xf, tmpout;
    // stage engine
    std::vector<int64_t> pf_ring_addr;
    std::vector<cudaEvent_t> pf_ev; std::vector<int> pf_ev_pending;
    std::vector<std::vector<int64_t>> st_ram_src, st_ram_j;    // per layer: src addr, staging index
    std::vector<std::vector<int>> st_nv_idx;                    // per layer: indices into st_nv
    std::deque<StageEntry> st_nv;
    std::atomic<bool> stage_cache{false};
    std::vector<int32_t> st_pins;
    std::thread st_th; std::atomic<bool> st_cancel{false}; std::atomic<int> st_inflight{0};
    std::mutex st_mu; std::condition_variable st_cv;
    int st_qd = 24;
    bool st_active = false;
    cudaStream_t wb_stream = nullptr;

    // ---------------- LRU over slots ----------------
    void lru_remove(int s)
    {
        const int p = lprev[s], n = lnext[s];
        if (p >= 0) lnext[p] = n; else if (lru_old == s) lru_old = n;
        if (n >= 0) lprev[n] = p; else if (lru_new == s) lru_new = p;
        lprev[s] = lnext[s] = -1;
    }
    void lru_push_new(int s) { lprev[s] = lru_new; lnext[s] = -1; if (lru_new >= 0) lnext[lru_new] = s; lru_new = s; if (lru_old < 0) lru_old = s; }
    void lru_push_old(int s) { lnext[s] = lru_old; lprev[s] = -1; if (lru_old >= 0) lprev[lru_old] = s; lru_old = s; if (lru_new < 0) lru_new = s; }
    void lru_touch(int s) { if (lru_new == s) return; lru_remove(s); lru_push_new(s); }

    void set_home(int key, int64_t a)
    {
        home[int64_t(key) * 3 + 0] = a; home[int64_t(key) * 3 + 1] = a + off_u; home[int64_t(key) * 3 + 2] = a + off_d;
    }
    // resident -> none (caller holds mu). Order: device-visible flag first, then home, then host state.
    int evict_key(int key)
    {
        const int s = slot_of[key];
        __atomic_store_n(&ram_res[key], 0, __ATOMIC_RELEASE);
        set_home(key, ph);
        kst[key] = KS_NONE; slot_of[key] = -1; owner[s] = SL_FREE;
        lru_remove(s);
        C.evictions++;
        return s;
    }
    // one free slot (caller holds mu): free list, else the oldest evictable resident key; -1 if none
    int take_slot()
    {
        if (!freel.empty()) { int s = freel.back(); freel.pop_back(); return s; }
        int guard = 0;
        for (int s = lru_old; s >= 0 && guard < nram; s = lnext[s], ++guard)
        {
            const int key = owner[s];
            if (key < 0 || kst[key] != KS_RES || prot[key] || pinc[s] > 0) continue;
            return evict_key(key);
        }
        C.alloc_fail++;
        return -1;
    }
    void free_slot(int s) { owner[s] = SL_FREE; freel.push_back(s); }

    // landed (reader thread)
    static void on_land(void* ctx, int64_t tag, int err, double ms)
    {
        Engine* e = static_cast<Engine*>(ctx);
        const int key = int(tag);
        e->C.read_ms_x1000 += (long long) (ms * 1000);
        if (err) { e->C.read_err++; e->hc[HC_ERR] = 16; fprintf(stderr, "nv2: read error %d key %d\n", err, key); }
        {
            std::lock_guard<std::mutex> lk(e->mu);
            if (e->kst[key] == KS_INFLIGHT)
            {
                e->kst[key] = KS_RES;
                const int s = e->slot_of[key];
                e->lru_push_new(s);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                __atomic_store_n(&e->ram_res[key], 1, __ATOMIC_RELEASE);
            }
        }
        e->land_cv.notify_all();
    }
    // start an NVMe read of key into a fresh slot (caller holds mu). Returns false if no slot.
    bool start_read(int key, bool high, double* arrival)
    {
        const int s = take_slot();
        if (s < 0) return false;
        owner[s] = key; slot_of[key] = s; kst[key] = KS_INFLIGHT;
        set_home(key, ram_addr[s]);                 // device reads home only after ram_res == 1
        const double t = now_ms();
        const double start = std::max(t, nv_free_at);
        double svc = pol.nvdeep;
        if (start > nv_free_at || nv_free_at == 0) svc += pol.nvlat;
        nv_free_at = start + svc;
        karr[key] = nv_free_at;
        if (arrival) *arrival = nv_free_at;
        rd.submit(fd, int64_t(key) * rec_bytes, ram_addr[s], slot_bytes, piece, high, &Engine::on_land, this, key);
        C.reads++; C.read_bytes += slot_bytes;
        return true;
    }

    // ---------------- logs (exclusive tier) ----------------
    void drain_logs(int64_t upto)   // entries with seq < upto are complete (their kernels ran before request upto)
    {
        const int64_t wbn = __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);
        const int64_t admn = __atomic_load_n(&hc[HC_ADMN], __ATOMIC_ACQUIRE);
        while (true)
        {
            const bool hw = wb_read < wbn && wbl[wb_read % LG].seq < upto;
            const bool ha = adm_read < admn && adml[adm_read % LG].seq < upto;
            if (!hw && !ha) break;
            bool take_wb = hw && (!ha || wbl[wb_read % LG].seq <= adml[adm_read % LG].seq);
            if (take_wb)
            {
                // VRAM victim parked in VRAM ring slot le.slot by nv_copy (complete: its kernel ran before request upto)
                const LogE le = wbl[wb_read % LG]; wb_read++;
                const int key = le.key, v = le.slot;
                if (kst[key] != KS_NONE) { vfree.push_back(v); C.wb_dup++; continue; }
                const int s = take_slot();
                if (s < 0) { vfree.push_back(v); C.wb_drop++; continue; }
                // resident at once, backed by the VRAM ring slot (valid bytes, device-readable); the wb thread copies it
                // to RAM slot s on a copy engine and then switches home to s. Pinned until then (no eviction).
                owner[s] = key; slot_of[key] = s; kst[key] = KS_RES; kvr[key] = v; pinc[s]++;
                lru_push_new(s);
                set_home(key, vring_addr[v]);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                __atomic_store_n(&ram_res[key], 1, __ATOMIC_RELEASE);
                {
                    std::lock_guard<std::mutex> lw(wb_mu);
                    wb_q.push_back({ key, s, v });
                }
                wb_cv.notify_one();
                C.wb_issued++;
            }
            else
            {
                const LogE le = adml[adm_read % LG]; adm_read++;
                const int key = le.key;
                if (!exclusive || kst[key] != KS_RES) continue;
                const int s = slot_of[key];
                if (pinc[s] > 0 || prot[key]) { C.adm_pinned++; lru_remove(s); lru_push_old(s); continue; }
                free_slot(evict_key(key));
                C.evictions--;   // not a capacity eviction
                C.adm_freed++;
            }
        }
    }
    void stock_landing()
    {
        const int64_t used = __atomic_load_n(&hc[HC_WBN], __ATOMIC_ACQUIRE);   // ring slots popped by the device
        while (land_tail - used < land_target && land_tail - used < LR && !vfree.empty())
        {
            land[land_tail % LR] = vfree.back(); vfree.pop_back();
            land_tail++;
            C.land_pushed++;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&hc[HC_LANDT], land_tail, __ATOMIC_RELEASE);
    }
    // write-back thread: the only thread besides Python that makes CUDA calls on the decode path. Nothing on the
    // device waits for it (vring-backed keys are device-readable), so a driver lock held elsewhere cannot deadlock.
    void wb_loop()
    {
        pin_self(wb_cpu);
        cudaSetDevice(dev_index);
        std::deque<WbP> pend;
        while (!quit.load())
        {
            {
                std::unique_lock<std::mutex> lw(wb_mu);
                if (wb_q.empty() && pend.empty()) wb_cv.wait_for(lw, std::chrono::milliseconds(50));
                while (!wb_q.empty()) { pend.push_back(wb_q.front()); wb_q.pop_front(); }
                wb_inflight.store((int) pend.size());
            }
            for (auto& w : pend)
                if (!w.issued)
                {
                    cudaMemcpyAsync((void*) ram_addr[w.s], (const void*) vring_addr[w.v], size_t(slot_bytes), cudaMemcpyDeviceToHost, wb_stream);
                    cudaEventRecord(vr_ev[w.v], wb_stream);
                    w.issued = true;
                }
            int done = 0;
            while (!pend.empty() && cudaEventQuery(vr_ev[pend.front().v]) == cudaSuccess)
            {
                const WbP w = pend.front(); pend.pop_front(); done++;
                std::lock_guard<std::mutex> lk(mu);
                if (kst[w.key] == KS_RES && slot_of[w.key] == w.s && kvr[w.key] == w.v)
                {
                    set_home(w.key, ram_addr[w.s]);
                    kvr[w.key] = -1;
                    C.wb_landed++;
                }
                pinc[w.s]--;
                vfree.push_back(w.v);
            }
            {
                std::lock_guard<std::mutex> lw(wb_mu);
                wb_inflight.store((int) pend.size());
            }
            if (!pend.empty() && !done) std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    void stock_free()
    {
        int guard = 0;
        while ((int) freel.size() < free_lo && guard++ < 16)
        {
            int s = -1;
            for (int x = lru_old, g = 0; x >= 0 && g < nram; x = lnext[x], ++g)
            {
                const int key = owner[x];
                if (key < 0 || kst[key] != KS_RES || prot[key] || pinc[x] > 0) continue;
                s = evict_key(key); break;
            }
            if (s < 0) break;
            freel.push_back(s);
        }
    }

    // ---------------- plan_layer (moetier semantics) ----------------
    struct PK { int u, key, cnt; double arr; float sc; };
    std::vector<PK> V, Rm, N;
    void plan_and_reply(const Req& r, Rep& p)
    {
        const double t0 = now_ms();
        const bool dec = r.kind == RK_DECODE;
        const bool cpu = dec && pol.cpu_on && cpu_ready && r.np <= MAXP && r.ntok <= MAXB;
        V.clear(); Rm.clear(); N.clear();
        int nnv = 0, nb70 = 0;
        const bool b70 = dec && b70_on && cpu_ready && r.np <= MAXP && r.ntok <= MAXB;   // independent of the CPU lane switch
        for (int u = 0; u < r.nu; ++u)
        {
            const int key = r.key[u];
            if (r.cls[u] == LN_VRAM) { p.lane[u] = LN_VRAM; V.push_back({ u, key, r.cnt[u], 0, 0 }); continue; }
            if (b70 && b70k[key]) { p.lane[u] = LN_B70; nb70++; C.lane_b70++; continue; }
            if (kst[key] == KS_RES)
            {
                lru_touch(slot_of[key]);
                if (kvr[key] >= 0) { p.lane[u] = LN_RAM; C.lane_zc++; C.vring_hits++; if (dec) C.ram++; continue; }   // VRAM-ring backed: GPU lane
                Rm.push_back({ u, key, r.cnt[u], 0, score[key] });
            }
            else
            {
                double a = 0;
                if (kst[key] == KS_INFLIGHT) { a = karr[key]; C.inflight_hit++; }
                else if (!start_read(key, true, &a)) { a = 1e9; hc[HC_ERR] = 19; fprintf(stderr, "nv2: no RAM slot for key %d\n", key); }
                N.push_back({ u, key, r.cnt[u], a - t0, score[key] });
                nnv++;
            }
        }
        if (dec) { C.picks += r.nu; C.vram += V.size(); C.ram += Rm.size(); C.nvme += N.size(); }
        double g = V.empty() ? 0.0 : pol.g0 + pol.thit * V.size();
        double c = (cpu && (!Rm.empty() || !N.empty())) ? pol.ca : 0.0;
        int ncpu = 0;
        std::sort(Rm.begin(), Rm.end(), [](const PK& a, const PK& b) { return a.cnt != b.cnt ? a.cnt > b.cnt : (a.sc != b.sc ? a.sc < b.sc : a.key < b.key); });
        for (auto& k : Rm)
        {
            if (!dec) { p.lane[k.u] = LN_RAM; C.lane_zc++; continue; }
            const double ce = c + pol.cb + pol.ctok * (k.cnt - 1);
            const double ge = g + pol.tzc;
            if (cpu && ncpu < pol.maxcpu && std::max(ce, g) <= std::max(c, ge)) { c = ce; p.lane[k.u] = LN_CPU; ncpu++; C.lane_cpu++; }
            else { g = ge; p.lane[k.u] = LN_RAM; C.lane_zc++; }
        }
        std::sort(N.begin(), N.end(), [](const PK& a, const PK& b) { return a.arr < b.arr; });
        for (auto& k : N)
        {
            const double a = std::max(0.0, k.arr);
            const double ce = std::max(c, a) + pol.cb + pol.ctok * (k.cnt - 1);
            const double ge = std::max(g, a + pol.push) + pol.thit;
            if (cpu && pol.nvcpu_on && ncpu < pol.maxcpu && std::max(ce, g) <= std::max(c, ge)) { c = ce; p.lane[k.u] = LN_NVC; ncpu++; C.lane_nvc++; }
            else if (dec && pol.nv_noadmit) { g = std::max(g, a) + pol.tzc; p.lane[k.u] = LN_NZC; C.lane_nvg++; }   // cold: zero-copy, keep VRAM
            else { g = ge; p.lane[k.u] = LN_NVG; C.lane_nvg++; }
        }
        p.ncpu = ncpu + nb70; p.nnv = nnv;
        // layer-ahead prefetch hints (next layer's predicted picks): read the non-resident ones now (behind demand reads)
        for (int i = 0; i < r.npf && i < MAXP; ++i)
        {
            const int key = r.pf_key[i];
            if (key < 0 || key >= K || kst[key] != KS_NONE || (b70_on && b70k[key])) continue;
            if (start_read(key, true, nullptr)) C.prefetch++;
        }
        C.plan_ns += (long long) ((now_ms() - t0) * 1e6);
    }

    void serve(int64_t s)
    {
        const double t0 = now_ms();
        const Req& r = req[s % RQ];
        Rep& p = rep[s % RQ];
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int k : protl) prot[k] = 0;      // request s-1 is complete on the device (its kernels ran before pub(s))
            protl.clear();
            for (int u = 0; u < r.nu; ++u) { prot[r.key[u]] = 1; protl.push_back(r.key[u]); }
            drain_logs(s);                         // write-back landings may evict: never this request's keys
            plan_and_reply(r, p);
            C.reqs++; (r.kind == RK_DECODE ? C.dec_reqs : C.small_reqs)++; if (r.kind == RK_DECODE) C.dec_tok += r.ntok;
        }
        if (p.ncpu > 0)
        {
            cj.seq = s; cj.li = r.li; cj.ntok = r.ntok; cj.np = 0;
            for (int i = 0; i < r.np; ++i)
            {
                const int e = r.pe[i];
                if (e < 0 || e >= E) continue;
                int u = -1;
                for (int q = 0; q < r.nu; ++q) if (r.key[q] == r.li * E + e) { u = q; break; }
                if (u < 0) continue;
                const int ln = p.lane[u];
                if (ln != LN_CPU && ln != LN_NVC && ln != LN_B70) continue;
                const int topk = r.np / std::max(1, r.ntok);
                cj.e[cj.np] = e; cj.t[cj.np] = i / std::max(1, topk); cj.w[cj.np] = r.pw[i]; cj.lane[cj.np] = ln; cj.np++;
            }
            cpu_job_seq.store(s, std::memory_order_release);
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&p.seq, s, __ATOMIC_RELEASE);
        __atomic_store_n(&hc[HC_ACK], s, __ATOMIC_RELEASE);
        t_reply = realtime_ns();
        C.reply_ns += (long long) ((now_ms() - t0) * 1e6);
        {
            std::lock_guard<std::mutex> lk(mu);
            stock_landing();
            stock_free();
        }
        const long long dt = (long long) ((now_ms() - t0) * 1e6);
        C.ctl_busy_ns += dt; C.serve_max_ns = std::max<long long>(C.serve_max_ns.load(), dt);
    }

    void ctl_loop()
    {
        pin_self(ctl_cpu);
        double last = now_ms();
        long n = 0;
        while (!quit.load(std::memory_order_relaxed))
        {
            const int64_t s = next_seq;
            const Req& r = req[s % RQ];
            if (__atomic_load_n(&r.seq, __ATOMIC_ACQUIRE) != s)
            {
                cpu_relax();
                if ((++n & 1023) == 0 && now_ms() - last > 200.0) std::this_thread::sleep_for(std::chrono::microseconds(20));
                continue;
            }
            dbg_tn[s % DBG] = realtime_ns(); dbg_seq[s % DBG] = s;
            serve(s);
            dbg_tr[s % DBG] = t_reply;
            last = now_ms();
            next_seq = s + 1;
        }
    }

    // ---------------- CPU worker ----------------
    void cpu_loop(int cpu0)
    {
        Pool::pin(cpu0);
        int64_t last = cpu_job_seq.load();
        long idle = 0;
        while (!quit.load(std::memory_order_relaxed))
        {
            const int64_t s = cpu_job_seq.load(std::memory_order_acquire);
            if (s == last) { if (++idle < 400000) _mm_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(20)); continue; }
            idle = 0;
            const double t0 = now_ms();
            run_cpu_job();
            C.cpu_busy_ns += (long long) ((now_ms() - t0) * 1e6);
            last = s;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            __atomic_store_n(&hc[HC_CDONE], s, __ATOMIC_RELEASE);
        }
    }
    void run_cpu_job()
    {
        // RAM-resident picks first (computed while the NVMe->CPU picks are still landing), then the NVMe ones
        const int ntok = cj.ntok, np = cj.np, li = cj.li;
        Layer& Ly = layers[li];
        xf.resize(size_t(std::max(ntok, 1)) * H);
        for (size_t i = 0; i < size_t(ntok) * H; ++i) xf[i] = h2f(hx[i]);
        int nb = 0; int btok[MAXP];
        if (b70r)
        {
            int32_t* rq = (int32_t*) (b70r + B70_OFF_REQ);
            uint16_t* bx = (uint16_t*) (b70r + B70_OFF_X);
            for (int k = 0; k < np; ++k)
                if (cj.lane[k] == LN_B70)
                {
                    rq[4 + nb] = cj.e[k]; ((float*) rq)[4 + MAXP + nb] = cj.w[k]; rq[4 + 2 * MAXP + nb] = cj.t[k];
                    std::memcpy(bx + size_t(nb) * H, hx + size_t(cj.t[k]) * H, size_t(H) * 2);
                    btok[nb++] = cj.t[k];
                }
            if (nb) b70_post(nb, ntok, li);
        }
        std::vector<int> pinned;
        std::vector<std::vector<std::pair<int, float>>> r1(ntok), r2(ntok);
        bool any2 = false;
        int maxm = 0; std::vector<int> cnt(E, 0);
        for (int k = 0; k < np; ++k) { maxm = std::max(maxm, ++cnt[cj.e[k]]); any2 |= cj.lane[k] == LN_NVC; }
        const int mode = cpu_mode >= 0 ? cpu_mode : (maxm <= 2 ? 2 : 0);
        auto bind = [&](int k) -> bool {   // caller holds mu; key resident
            const int e = cj.e[k], key = li * E + e;
            if (kvr[key] >= 0) { hc[HC_ERR] = 33; return false; }
            const int s = slot_of[key];
            if (std::find(pinned.begin(), pinned.end(), s) == pinned.end()) { pinc[s]++; pinned.push_back(s); }
            const int64_t a = ram_addr[s];
            Ly.ex[e].g.tr = reinterpret_cast<const uint8_t*>(a);
            Ly.ex[e].u.tr = reinterpret_cast<const uint8_t*>(a + off_u);
            Ly.ex[e].d.tr = reinterpret_cast<const uint8_t*>(a + off_d);
            return true;
        };
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int k = 0; k < np; ++k)
                if (cj.lane[k] == LN_CPU)
                {
                    if (kst[li * E + cj.e[k]] != KS_RES) { hc[HC_ERR] = 34; continue; }
                    if (bind(k)) r1[cj.t[k]].push_back({ cj.e[k], cj.w[k] });
                }
        }
        bool any1 = false; for (auto& v : r1) any1 |= !v.empty();
        if (g_cpu_kern == 1 && n135::K_.merge && any2)
        {
            // N135: ONE forward over the RAM and the NVMe->CPU picks; an NVC job is bound (waits for its landing) by the
            // first worker that reaches it, after every resident job's units
            std::vector<uint8_t> late(E, 0);
            for (int k = 0; k < np; ++k) if (cj.lane[k] == LN_NVC) { late[cj.e[k]] = 1; r1[cj.t[k]].push_back({ cj.e[k], cj.w[k] }); }
            std::function<bool(int)> bind_late = [&](int ex) -> bool {
                std::unique_lock<std::mutex> lk(mu);
                int k = -1;
                for (int q = 0; q < np; ++q) if (cj.e[q] == ex && cj.lane[q] == LN_NVC) { k = q; break; }
                if (k < 0) return false;
                const int key = li * E + ex;
                if (kst[key] != KS_RES)
                {
                    const double tw = now_ms();
                    land_cv.wait_for(lk, std::chrono::seconds(10), [&] { return kst[key] == KS_RES; });
                    C.cpu_wait_land_ns += (long long) ((now_ms() - tw) * 1e6);
                    if (kst[key] != KS_RES) { hc[HC_ERR] = 32; return false; }
                }
                return bind(k);
            };
            n135::moe_forward4(pool, Ly, xf.data(), ntok, r1, hout, mode, late.data(), &bind_late);
            goto n135_done;
        }
        if (any1 || !nb) cpu_moe_forward(pool, Ly, xf.data(), ntok, r1, hout, mode);
        else std::memset(hout, 0, size_t(ntok) * H * 4);
        if (any2)
        {
            {
                std::unique_lock<std::mutex> lk(mu);
                for (int k = 0; k < np; ++k)
                {
                    if (cj.lane[k] != LN_NVC) continue;
                    const int key = li * E + cj.e[k];
                    if (kst[key] != KS_RES)
                    {
                        const double tw = now_ms();
                        land_cv.wait_for(lk, std::chrono::seconds(10), [&] { return kst[key] == KS_RES; });
                        C.cpu_wait_land_ns += (long long) ((now_ms() - tw) * 1e6);
                        if (kst[key] != KS_RES) { hc[HC_ERR] = 32; continue; }
                    }
                    if (bind(k)) r2[cj.t[k]].push_back({ cj.e[k], cj.w[k] });
                }
            }
            tmpout.resize(size_t(ntok) * H);
            cpu_moe_forward(pool, Ly, xf.data(), ntok, r2, tmpout.data(), mode);
            for (size_t i = 0; i < size_t(ntok) * H; ++i) hout[i] += tmpout[i];
        }
    n135_done:
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int s : pinned) pinc[s]--;
        }
        if (nb) b70_collect(nb, btok);
        C.cpu_jobs++; C.cpu_experts += np - nb; C.cpu_tokens += ntok;
    }

    // ---------------- N137 B70 ring client ----------------
    volatile int64_t* b70h() { return (volatile int64_t*) b70r; }
    void b70_post(int nb, int ntok, int li)
    {
        int32_t* rq = (int32_t*) (b70r + B70_OFF_REQ);
        rq[0] = nb; rq[1] = ntok; rq[2] = li;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        b70_t0 = now_ms();
        __atomic_store_n(&b70h()[B70_W_REQ_SEQ], ++b70_seq, __ATOMIC_RELEASE);
    }
    double b70_t0 = 0;
    bool b70_wait()
    {
        const double tw = now_ms();
        long n = 0;
        while (__atomic_load_n(&b70h()[B70_W_DONE_SEQ], __ATOMIC_ACQUIRE) != b70_seq)
        {
            _mm_pause();
            if ((++n & 4095) == 0 && (now_ms() - tw) * 1e6 > b70_timeout_ns) { hc[HC_ERR] = 40; C.b70_err++; return false; }
        }
        const double t1 = now_ms();
        C.b70_wait_ns += (long long) ((t1 - tw) * 1e6);
        const long long rtt = (long long) ((t1 - b70_t0) * 1e6);
        C.b70_rtt_ns += rtt; C.b70_rtt_max_ns = std::max<long long>(C.b70_rtt_max_ns.load(), rtt);
        if (b70h()[B70_W_ERR]) { hc[HC_ERR] = 41; C.b70_err++; return false; }
        return true;
    }
    void b70_collect(int nb, const int* btok)
    {
        if (!b70_wait()) return;
        const uint16_t* bo = (const uint16_t*) (b70r + B70_OFF_OUT);
        for (int i = 0; i < nb; ++i)
        {
            float* o = hout + size_t(btok[i]) * H;
            const uint16_t* s = bo + size_t(i) * H;
            for (int j = 0; j < H; j += 8)
            {
                __m256 a = _mm256_loadu_ps(o + j);
                __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (s + j)));
                _mm256_storeu_ps(o + j, _mm256_add_ps(a, b));
            }
        }
        C.b70_jobs++; C.b70_rows += nb;
    }

    // ---------------- stage engine (prefill) ----------------
    static void on_stage(void* ctx, int64_t tag, int err, double ms)
    {
        Engine* e = static_cast<Engine*>(ctx);
        if (err) { e->C.read_err++; e->hc[HC_ERR] = 17; }
        e->C.read_ms_x1000 += (long long) (ms * 1000);
        if (!err && e->stage_cache.load())
        {
            // Preserve staged NVMe reads in the bounded RAM tier, rather than
            // throwing their bytes away after the H2D. Current RAM sources are
            // pinned by stage_begin; an INFLIGHT reservation cannot be evicted.
            StageEntry& entry = e->st_nv[size_t(tag)];
            int s = -1;
            {
                std::lock_guard<std::mutex> lk(e->mu);
                if (e->kst[entry.key] == KS_NONE)
                {
                    s = e->take_slot();
                    if (s >= 0)
                    {
                        e->owner[s] = entry.key; e->slot_of[entry.key] = s;
                        e->kst[entry.key] = KS_INFLIGHT;
                        e->set_home(entry.key, e->ram_addr[s]);
                    }
                }
            }
            if (s >= 0)
            {
                std::memcpy((void*) e->ram_addr[s], (const void*) e->pf_ring_addr[entry.ring], size_t(e->slot_bytes));
                on_land(ctx, entry.key, 0, 0.0);
            }
        }
        {
            std::lock_guard<std::mutex> lk(e->st_mu);
            e->st_nv[size_t(tag)].ready.store(1);
            e->st_inflight--;
        }
        e->st_cv.notify_all();
    }
    void st_reader()
    {
        pin_self(wb_cpu);
        const int R = (int) pf_ring_addr.size();
        for (size_t g = 0; g < st_nv.size(); ++g)
        {
            if (st_cancel.load()) break;
            const int r = int(g % R);
            if (g >= size_t(R))
            {
                // ring entry r was used by entry g - R: wait until its H2D was enqueued, then until it completed
                std::unique_lock<std::mutex> lk(st_mu);
                st_cv.wait(lk, [&] { return st_cancel.load() || pf_ev_pending[r] == int(g - R) + 1; });
                if (st_cancel.load()) break;
                lk.unlock();
                cudaEventSynchronize(pf_ev[r]);
            }
            {
                std::unique_lock<std::mutex> lk(st_mu);
                st_cv.wait(lk, [&] { return st_cancel.load() || st_inflight.load() < st_qd; });
                if (st_cancel.load()) break;
                st_inflight++;
            }
            st_nv[g].ring = r;
            rd.submit(fd, int64_t(st_nv[g].key) * rec_bytes, pf_ring_addr[r], slot_bytes, piece, false, &Engine::on_stage, this, int64_t(g));
            C.stage_reads++;
        }
    }
    void stage_end()
    {
        if (!st_active) return;
        st_cancel = true;
        st_cv.notify_all();
        if (st_th.joinable()) st_th.join();
        {   // drain in-flight reads
            std::unique_lock<std::mutex> lk(st_mu);
            st_cv.wait(lk, [&] { return st_inflight.load() == 0; });
        }
        for (auto& ev : pf_ev) cudaEventSynchronize(ev);
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int s : st_pins) pinc[s]--;
        }
        st_pins.clear(); st_nv.clear(); st_ram_src.clear(); st_ram_j.clear(); st_nv_idx.clear();
        std::fill(pf_ev_pending.begin(), pf_ev_pending.end(), 0);
        st_active = false;
    }
    // slot_np: [L, E] int32 VRAM residency snapshot (stable during prefill: no admission)
    void stage_begin(const int32_t* so, int li0, const int64_t* selected = nullptr, int nselected = -1)
    {
        stage_end();
        st_cancel = false;
        st_ram_src.assign(L, {}); st_ram_j.assign(L, {}); st_nv_idx.assign(L, {});
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int li = li0; li < (nselected < 0 ? L : li0 + 1); ++li)
            {
                int j = 0;
                for (int i = 0; i < (nselected < 0 ? E : nselected); ++i)
                {
                    const int e = nselected < 0 ? i : int(selected[i]);
                    TORCH_CHECK(e >= 0 && e < E, "staged expert index out of range");
                    if (so[li * E + e] >= 0) continue;
                    const int key = li * E + e;
                    if (kst[key] == KS_RES)
                    {
                        const int s = slot_of[key];
                        pinc[s]++; st_pins.push_back(s);
                        st_ram_src[li].push_back(kvr[key] >= 0 ? vring_addr[kvr[key]] : ram_addr[s]); st_ram_j[li].push_back(j);
                        C.stage_ram++;
                    }
                    else
                    {
                        st_nv.emplace_back();
                        StageEntry& se = st_nv.back(); se.li = li; se.j = j; se.key = key; se.ring = -1;
                        st_nv_idx[li].push_back(int(st_nv.size()) - 1);
                        C.stage_nv++;
                    }
                    ++j;
                }
            }
        }
        st_active = true;
        st_th = std::thread([this] { st_reader(); });
    }
    // enqueue on `stream`: wait ev_wait, H2D layer li's RAM experts, then its NVMe experts as they land in the ring
    double stage_layer(int li, int64_t base, int64_t stream, int64_t ev_wait, int64_t ev_done)
    {
        const double t0 = now_ms();
        cudaStream_t st = (cudaStream_t) stream;
        cudaStreamWaitEvent(st, (cudaEvent_t) ev_wait, 0);
        double waited = 0;
        if (st_active && li < L)
        {
            for (size_t i = 0; i < st_ram_src[li].size(); ++i)
                cudaMemcpyAsync((void*) (base + st_ram_j[li][i] * slot_bytes), (const void*) st_ram_src[li][i], size_t(slot_bytes), cudaMemcpyDefault, st);
            for (int g : st_nv_idx[li])
            {
                StageEntry& se = st_nv[size_t(g)];
                if (!se.ready.load())
                {
                    const double tw = now_ms();
                    std::unique_lock<std::mutex> lk(st_mu);
                    st_cv.wait_for(lk, std::chrono::seconds(30), [&] { return se.ready.load() == 1; });
                    waited += now_ms() - tw;
                    if (!se.ready.load()) { hc[HC_ERR] = 18; continue; }
                }
                cudaMemcpyAsync((void*) (base + se.j * slot_bytes), (const void*) pf_ring_addr[se.ring], size_t(slot_bytes), cudaMemcpyHostToDevice, st);
                cudaEventRecord(pf_ev[se.ring], st);
                {
                    std::lock_guard<std::mutex> lk(st_mu);
                    pf_ev_pending[se.ring] = g + 1;
                }
                st_cv.notify_all();
            }
        }
        cudaEventRecord((cudaEvent_t) ev_done, st);
        C.stage_wait_ns += (long long) (waited * 1e6);
        C.stage_job_ns += (long long) ((now_ms() - t0) * 1e6);
        return waited;
    }
};

// ------------------------------------------------------------------------------------------------------------
// python API
// ------------------------------------------------------------------------------------------------------------
static struct sigaction g_old_segv;
static void segv_handler(int sig, siginfo_t* si, void* uc)
{
    void* bt[64];
    const int n = backtrace(bt, 64);
    char msg[128];
    const int m = snprintf(msg, sizeof(msg), "\nnv2: signal %d at %p, thread %ld, backtrace:\n", sig, si ? si->si_addr : nullptr, (long) pthread_self());
    if (m > 0) { ssize_t w = write(2, msg, size_t(m)); (void) w; }
    backtrace_symbols_fd(bt, n, 2);
    sigaction(SIGSEGV, &g_old_segv, nullptr);
    raise(sig);
}

void init(int64_t L, int64_t E, int64_t H, int64_t I, int64_t rec_bytes, int64_t slot_bytes, int64_t piece, int64_t off_u,
          int64_t off_d, int64_t fd, int64_t ph, int64_t hc, int64_t req, int64_t rep, int64_t wbl, int64_t adml, int64_t land,
          int64_t home, int64_t ram_res, int64_t hx, int64_t hout, at::Tensor ram_addr, at::Tensor pf_ring, at::Tensor score,
          int64_t readers, std::vector<int64_t> reader_cpus, int64_t ctl_cpu, int64_t exclusive, int64_t land_target, int64_t free_lo,
          at::Tensor vring)
{
    TORCH_CHECK(G == nullptr, "nv2 engine already initialised");
    {
        struct sigaction sa; memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = segv_handler; sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, &g_old_segv);
    }
    G = new Engine();
    Engine& e = *G;
    e.L = int(L); e.E = int(E); e.K = int(L * E); e.H = int(H); e.I = int(I);
    e.rec_bytes = rec_bytes; e.slot_bytes = slot_bytes; e.piece = piece; e.off_u = off_u; e.off_d = off_d; e.fd = int(fd); e.ph = ph;
    e.hc = (volatile int64_t*) hc; e.req = (Req*) req; e.rep = (Rep*) rep; e.wbl = (LogE*) wbl; e.adml = (LogE*) adml;
    e.land = (int32_t*) land; e.home = (volatile int64_t*) home; e.ram_res = (volatile int32_t*) ram_res;
    e.hx = (uint16_t*) hx; e.hout = (float*) hout;
    e.nram = int(ram_addr.numel());
    e.ram_addr.assign(ram_addr.data_ptr<int64_t>(), ram_addr.data_ptr<int64_t>() + e.nram);
    e.pf_ring_addr.assign(pf_ring.data_ptr<int64_t>(), pf_ring.data_ptr<int64_t>() + pf_ring.numel());
    e.pf_ev.resize(e.pf_ring_addr.size()); e.pf_ev_pending.assign(e.pf_ring_addr.size(), 0);
    for (auto& ev : e.pf_ev) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    e.score.assign(score.data_ptr<float>(), score.data_ptr<float>() + e.K);
    e.slot_of.assign(e.K, -1); e.kvr.assign(e.K, -1); e.kst.assign(e.K, KS_NONE); e.karr.assign(e.K, 0); e.prot.assign(e.K, 0);
    e.owner.assign(e.nram, SL_FREE); e.pinc.assign(e.nram, 0); e.lprev.assign(e.nram, -1); e.lnext.assign(e.nram, -1);
    for (int s = e.nram - 1; s >= 0; --s) e.freel.push_back(s);
    for (int k = 0; k < e.K; ++k) { e.ram_res[k] = 0; e.set_home(k, ph); }
    e.exclusive = exclusive != 0; e.land_target = int(land_target); e.free_lo = int(free_lo);
    e.ctl_cpu = int(ctl_cpu);
    std::vector<int> rc(reader_cpus.begin(), reader_cpus.end());
    e.wb_cpu = rc.empty() ? -1 : rc.back();
    e.rd.start(int(readers), rc);
    cudaStreamCreateWithFlags(&e.wb_stream, cudaStreamNonBlocking);
    e.vring_addr.assign(vring.data_ptr<int64_t>(), vring.data_ptr<int64_t>() + vring.numel());
    e.vr_ev.resize(e.vring_addr.size());
    for (auto& ev : e.vr_ev) cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    for (int v = int(e.vring_addr.size()) - 1; v >= 0; --v) e.vfree.push_back(v);
    std::lock_guard<std::mutex> lk(e.mu);
    e.stock_landing();
}

void start()
{
    Engine& e = *G;
    e.next_seq = e.hc[HC_PUB] + 1;
    e.ctl_th = std::thread([&e] { e.ctl_loop(); });
    e.wb_th = std::thread([&e] { e.wb_loop(); });
}

void set_pol(std::vector<double> v)
{
    Engine& e = *G; Pol& p = e.pol;
    double* f[] = { &p.g0, &p.thit, &p.tzc, &p.ca, &p.cb, &p.ctok, &p.push, &p.nvlat, &p.nvdeep, &p.nvone };
    for (size_t i = 0; i < 10 && i < v.size(); ++i) *f[i] = v[i];
    if (v.size() > 10) p.cpu_on = int(v[10]);
    if (v.size() > 11) p.nvcpu_on = int(v[11]);
    if (v.size() > 12) p.maxcpu = int(v[12]);
    if (v.size() > 13) p.nv_noadmit = int(v[13]);
}

// blocking fill (warm start): keys -> resident (oldest-first LRU position order = given order)
int64_t warm_read(at::Tensor keys)
{
    Engine& e = *G;
    const int64_t* k = keys.data_ptr<int64_t>();
    int64_t n = 0;
    for (int64_t i = 0; i < keys.numel(); ++i)
    {
        std::lock_guard<std::mutex> lk(e.mu);
        if (e.kst[k[i]] != KS_NONE) continue;
        if (!e.start_read(int(k[i]), false, nullptr)) break;
        ++n;
    }
    // wait for all to land
    std::unique_lock<std::mutex> lk(e.mu);
    e.land_cv.wait(lk, [&] { for (int64_t i = 0; i < keys.numel(); ++i) if (e.kst[k[i]] == KS_INFLIGHT) return false; return true; });
    return n;
}

// exclusive: drop RAM copies of these keys (VRAM-resident after the VRAM warm)
int64_t drop_keys(at::Tensor keys)
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lk(e.mu);
    int64_t n = 0;
    const int64_t* k = keys.data_ptr<int64_t>();
    for (int64_t i = 0; i < keys.numel(); ++i)
        if (e.kst[k[i]] == KS_RES && e.pinc[e.slot_of[k[i]]] == 0 && e.kvr[k[i]] < 0) { e.free_slot(e.evict_key(int(k[i]))); e.C.evictions--; ++n; }
    return n;
}

// elastic enter (exclusive): VRAM-only experts of the slots being handed back -> fresh RAM slots (D2H), host-marked
int64_t wb_bulk(at::Tensor keys, at::Tensor src)
{
    Engine& e = *G;
    const int64_t* k = keys.data_ptr<int64_t>(); const int64_t* a = src.data_ptr<int64_t>();
    std::vector<std::pair<int, int>> done;
    {
        std::lock_guard<std::mutex> lk(e.mu);
        for (int64_t i = 0; i < keys.numel(); ++i)
        {
            const int key = int(k[i]);
            if (e.kst[key] != KS_NONE) continue;
            const int s = e.take_slot();
            if (s < 0) break;
            e.owner[s] = key;                        // reserved
            cudaMemcpyAsync((void*) e.ram_addr[s], (const void*) a[i], size_t(e.slot_bytes), cudaMemcpyDeviceToHost, e.wb_stream);
            done.push_back({ key, s });
        }
    }
    cudaStreamSynchronize(e.wb_stream);
    std::lock_guard<std::mutex> lk(e.mu);
    for (auto& [key, s] : done)
    {
        e.slot_of[key] = s; e.kst[key] = KS_RES; e.lru_push_new(s); e.set_home(key, e.ram_addr[s]);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        __atomic_store_n(&e.ram_res[key], 1, __ATOMIC_RELEASE);
    }
    e.C.wb_bulk += (long long) done.size();
    return (int64_t) done.size();
}

// sync drain of device logs (call after torch.cuda.synchronize, e.g. before verify / elastic)
void drain_all()
{
    Engine& e = *G;
    {
        std::lock_guard<std::mutex> lk(e.mu);
        e.drain_logs(INT64_MAX);
    }
    e.wb_cv.notify_all();
    for (int i = 0; i < 20000; ++i)
    {
        bool empty;
        { std::lock_guard<std::mutex> lw(e.wb_mu); empty = e.wb_q.empty() && e.wb_inflight.load() == 0; }
        if (empty) break;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
}

// key state snapshot: [K] slot (or -1), [K] state
std::vector<at::Tensor> state()
{
    Engine& e = *G;
    std::lock_guard<std::mutex> lk(e.mu);
    auto so = torch::from_blob(e.slot_of.data(), { e.K }, torch::kInt32).clone();
    auto ks = torch::from_blob(e.kst.data(), { e.K }, torch::kInt8).clone();
    auto pc = torch::from_blob(e.pinc.data(), { e.nram }, torch::kInt32).clone();
    auto ow = torch::from_blob(e.owner.data(), { e.nram }, torch::kInt32).clone();
    return { so, ks, pc, ow };
}

int64_t read_sync(int64_t key, int64_t dst)
{
    Engine& e = *G;
    int64_t done = 0;
    while (done < e.slot_bytes)
    {
        ssize_t r = pread(e.fd, (void*) (dst + done), size_t(e.slot_bytes - done), off_t(key * e.rec_bytes + done));
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; return -1; }
        done += r;
    }
    return done;
}

std::vector<int64_t> counters()
{
    Engine& e = *G; Counters& c = e.C;
    std::vector<int64_t> v = { c.reqs, c.dec_reqs, c.small_reqs, c.picks, c.vram, c.ram, c.nvme, c.inflight_hit, c.lane_zc,
        c.lane_cpu, c.lane_nvg, c.lane_nvc, c.reads, c.read_bytes, c.wb_landed, c.wb_dup, c.adm_freed, c.adm_pinned,
        c.evictions, c.cpu_jobs, c.cpu_experts, c.cpu_tokens, c.stage_ram, c.stage_nv, c.stage_reads, c.read_err,
        c.alloc_fail, c.land_pushed, c.prefetch, c.plan_ns, c.ctl_busy_ns, c.cpu_busy_ns, c.cpu_wait_land_ns,
        c.read_ms_x1000, c.stage_wait_ns, c.stage_job_ns, c.wb_bulk, c.dec_tok, c.notice_ns, c.notice_max_ns, c.serve_max_ns, c.reply_ns, c.wb_issued, c.wb_drop, c.vring_hits,
        c.lane_b70, c.b70_jobs, c.b70_rows, c.b70_wait_ns, c.b70_rtt_ns, c.b70_err, c.b70_rtt_max_ns };
    std::lock_guard<std::mutex> lk(e.mu);
    int res = 0, inf = 0; for (int k = 0; k < e.K; ++k) { res += e.kst[k] == KS_RES; inf += e.kst[k] == KS_INFLIGHT; }
    v.push_back(res); v.push_back(inf); v.push_back((int64_t) e.freel.size()); v.push_back(e.land_tail - e.hc[HC_WBN]);
    return v;
}

std::vector<at::Tensor> dbg_host()
{
    Engine& e = *G;
    return { torch::from_blob(e.dbg_seq.data(), { DBG }, torch::kInt64).clone(), torch::from_blob(e.dbg_tn.data(), { DBG }, torch::kInt64).clone(),
             torch::from_blob(e.dbg_tr.data(), { DBG }, torch::kInt64).clone() };
}

std::vector<int64_t> diag()
{
    Engine& e = *G;
    std::vector<int64_t> v;
    for (int i = 0; i < 8; ++i) v.push_back((int64_t) e.hc[i]);
    v.push_back(e.next_seq); v.push_back((int64_t) e.wb_inflight.load());
    v.push_back((int64_t) e.wb_q.size());
    v.push_back((int64_t) e.vfree.size()); v.push_back(e.land_tail); v.push_back(e.wb_read); v.push_back(e.adm_read);
    const Req& r = e.req[e.next_seq % RQ];
    v.push_back(r.seq);
    return v;
}

void reset_counters() { Counters& c = G->C; c.~Counters(); new (&c) Counters(); }

// ---- CPU tier ----
void cpu_init(int64_t threads, std::vector<int64_t> cpus, int64_t mode, double act_limit)
{
    Engine& e = *G;
    init_perm(); init_tables(); cpu_kern_init();
    std::vector<int> c(cpus.begin(), cpus.end());
    cpu_set_t saved; sched_getaffinity(0, sizeof(saved), &saved);
    e.pool.start(int(threads), c);
    sched_setaffinity(0, sizeof(saved), &saved);
    e.cpu_mode = int(mode);
    g_act_limit = float(act_limit);
    e.layers.resize(e.L);
}
void cpu_add_layer(int64_t li, at::Tensor suh_g, at::Tensor svh_g, at::Tensor suh_u, at::Tensor svh_u, at::Tensor suh_d, at::Tensor svh_d)
{
    Engine& e = *G;
    Layer& Ly = e.layers[li];
    Ly.H = e.H; Ly.I = e.I; Ly.ex.resize(e.E);
    auto u16 = [](const at::Tensor& t, int x) { return reinterpret_cast<const uint16_t*>(t.data_ptr()) + size_t(x) * t.size(1); };
    for (int x = 0; x < e.E; ++x)
    {
        Ly.ex[x].g = { nullptr, u16(suh_g, x), u16(svh_g, x), e.H, e.I };
        Ly.ex[x].u = { nullptr, u16(suh_u, x), u16(svh_u, x), e.H, e.I };
        Ly.ex[x].d = { nullptr, u16(suh_d, x), u16(svh_d, x), e.I, e.H };
    }
    for (auto& t : { suh_g, svh_g, suh_u, svh_u, suh_d, svh_d }) e.keep.push_back(t);
}
void cpu_start(int64_t cpu0)
{
    Engine& e = *G;
    e.cpu_ready = true;
    e.cpu_th = std::thread([&e, cpu0] { e.cpu_loop(int(cpu0)); });
}
void cpu_set(int64_t on, int64_t nvcpu, int64_t mode, double act_limit)
{
    Engine& e = *G;
    e.pol.cpu_on = int(on); e.pol.nvcpu_on = int(nvcpu); e.cpu_mode = int(mode); g_act_limit = float(act_limit);
}
// synchronous forward from RAM slots (worker idle / CPU lane off): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
at::Tensor cpu_forward(int64_t li, at::Tensor x, at::Tensor sel, at::Tensor w, int64_t mode)
{
    Engine& e = *G;
    const int m = int(x.size(0)), k = int(sel.size(1));
    auto out = torch::zeros({ m, e.H }, torch::kFloat);
    std::vector<std::vector<std::pair<int, float>>> route(m);
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
    Layer& Ly = e.layers[li];
    std::vector<int> pinned;
    {
        std::lock_guard<std::mutex> lk(e.mu);
        for (int t = 0; t < m; ++t)
            for (int j = 0; j < k; ++j)
            {
                const int ex = sa[t][j]; const int key = int(li) * e.E + ex;
                TORCH_CHECK(e.kst[key] == KS_RES, "cpu_forward: expert not RAM-resident");
                const int s = e.slot_of[key];
                e.pinc[s]++; pinned.push_back(s);
                const int64_t a = e.ram_addr[s];
                Ly.ex[ex].g.tr = (const uint8_t*) a; Ly.ex[ex].u.tr = (const uint8_t*) (a + e.off_u); Ly.ex[ex].d.tr = (const uint8_t*) (a + e.off_d);
                route[t].push_back({ ex, wa[t][j] });
            }
    }
    int maxm = 0; { std::vector<int> cnt(e.E, 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int md = mode >= 0 ? int(mode) : (maxm <= 2 ? 2 : 0);
    cpu_moe_forward(e.pool, Ly, x.data_ptr<float>(), m, route, out.data_ptr<float>(), md);
    std::lock_guard<std::mutex> lk(e.mu);
    for (int s : pinned) e.pinc[s]--;
    return out;
}

// ---- N137 B70 tier ----
// test-only engine (no CUDA, no readers): geometry only, so b70_attach / b70_forward can be checked without a GPU
void b70_test_engine(int64_t L, int64_t E, int64_t H)
{
    TORCH_CHECK(G == nullptr, "nv2 engine already initialised");
    G = new Engine();
    G->L = int(L); G->E = int(E); G->K = int(L * E); G->H = int(H);
    static int64_t hc_dummy[HC_N] = {0};
    G->hc = (volatile int64_t*) hc_dummy;
}
// attach the ring, LOAD the key set on the server (blocks until it answers), mark the keys B70-resident
int64_t b70_attach(std::string path, at::Tensor keys, double load_timeout_s)
{
    Engine& e = *G;
    const int fd = open(path.c_str(), O_RDWR);
    TORCH_CHECK(fd >= 0, "b70: cannot open ring ", path);
    void* p = mmap(nullptr, B70_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    TORCH_CHECK(p != MAP_FAILED, "b70: mmap failed");
    e.b70r = (uint8_t*) p;
    volatile int64_t* h = e.b70h();
    TORCH_CHECK(h[B70_W_MAGIC] == B70_MAGIC, "b70: bad ring magic");
    const int n = int(keys.numel());
    TORCH_CHECK(n <= 16384, "b70: too many keys");
    int32_t* kk = (int32_t*) (e.b70r + B70_OFF_KEYS);
    auto ka = keys.accessor<int64_t, 1>();
    for (int i = 0; i < n; ++i) kk[i] = int32_t(ka[i]);
    h[B70_W_NKEYS] = n;
    const int64_t ls = h[B70_W_LOAD_SEQ] + 1;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    h[B70_W_LOAD_SEQ] = ls;
    const double t0 = now_ms();
    while (h[B70_W_LOAD_ACK] != ls)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        TORCH_CHECK((now_ms() - t0) / 1e3 < load_timeout_s, "b70: LOAD timed out");
    }
    TORCH_CHECK(h[B70_W_STATE] == 2, "b70: server LOAD failed, err ", (long long) h[B70_W_ERR]);
    e.b70k.assign(e.K, 0);
    for (int i = 0; i < n; ++i) e.b70k[ka[i]] = 1;
    e.b70_seq = h[B70_W_DONE_SEQ];
    return (int64_t) ((now_ms() - t0));
}
void b70_set(int64_t on, double timeout_s) { Engine& e = *G; TORCH_CHECK(e.b70r || !on, "b70: not attached"); e.b70_on = on != 0; e.b70_timeout_ns = int64_t(timeout_s * 1e9); }
// synchronous forward through the ring (decode idle or B70 lane off): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
at::Tensor b70_forward(int64_t li, at::Tensor x, at::Tensor sel, at::Tensor w)
{
    Engine& e = *G;
    TORCH_CHECK(e.b70r, "b70: not attached");
    const int m = int(x.size(0)), k = int(sel.size(1));
    TORCH_CHECK(m * k <= MAXP, "b70_forward: too many picks");
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>(); auto xa = x.accessor<float, 2>();
    int32_t* rq = (int32_t*) (e.b70r + B70_OFF_REQ);
    uint16_t* bx = (uint16_t*) (e.b70r + B70_OFF_X);
    int nb = 0; int btok[MAXP];
    for (int t = 0; t < m; ++t)
        for (int j = 0; j < k; ++j)
        {
            TORCH_CHECK(e.b70k[li * e.E + sa[t][j]], "b70_forward: expert not B70-resident");
            rq[4 + nb] = sa[t][j]; ((float*) rq)[4 + MAXP + nb] = wa[t][j]; rq[4 + 2 * MAXP + nb] = t;
            for (int c = 0; c < e.H; ++c) bx[size_t(nb) * e.H + c] = f2h(xa[t][c]);
            btok[nb++] = t;
        }
    auto out = torch::zeros({ m, e.H }, torch::kFloat);
    float* save = e.hout;
    e.hout = out.data_ptr<float>();
    e.b70_post(nb, m, int(li));
    e.b70_collect(nb, btok);
    e.hout = save;
    return out;
}

// ---- stage engine ----
void stage_begin(at::Tensor so, int64_t li0) { G->stage_begin(so.data_ptr<int32_t>(), int(li0)); }
void stage_begin_selected(at::Tensor so, int64_t li0, at::Tensor selected)
{
    TORCH_CHECK(!so.is_cuda() && !selected.is_cuda() && selected.scalar_type() == at::kLong && selected.is_contiguous(),
                "selected staging indices must be contiguous CPU int64");
    G->stage_begin(so.data_ptr<int32_t>(), int(li0), selected.data_ptr<int64_t>(), int(selected.numel()));
}
double stage_layer(int64_t li, int64_t base, int64_t stream, int64_t ev_wait, int64_t ev_done) { return G->stage_layer(int(li), base, stream, ev_wait, ev_done); }
void stage_end() { G->stage_end(); }
void set_stage_qd(int64_t q) { G->st_qd = int(q); }

void shutdown()
{
    if (!G) return;
    G->stage_end();
    G->quit = true;
    if (G->ctl_th.joinable()) G->ctl_th.join();
    G->wb_cv.notify_all();
    if (G->wb_th.joinable()) G->wb_th.join();
    if (G->cpu_th.joinable()) G->cpu_th.join();
    if (G->cpu_ready) G->pool.stop();
    G->rd.stop();
}

}  // namespace nv2h

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace nv2h;
    namespace py = pybind11;
    m.def("init", &init);
    m.def("start", &start);
    m.def("set_pol", &set_pol);
    m.def("warm_read", &warm_read, py::call_guard<py::gil_scoped_release>());
    m.def("drop_keys", &drop_keys);
    m.def("wb_bulk", &wb_bulk, py::call_guard<py::gil_scoped_release>());
    m.def("drain_all", &drain_all);
    m.def("state", &state);
    m.def("read_sync", &read_sync, py::call_guard<py::gil_scoped_release>());
    m.def("counters", &counters);
    m.def("reset_counters", &reset_counters);
    m.def("dbg_host", &dbg_host);
    m.def("diag", &diag);
    m.def("cpu_init", &cpu_init);
    m.def("cpu_add_layer", &cpu_add_layer);
    m.def("cpu_start", &cpu_start);
    m.def("cpu_set", &cpu_set);
    m.def("cpu_forward", &cpu_forward, py::call_guard<py::gil_scoped_release>());
    m.def("b70_attach", &b70_attach, py::call_guard<py::gil_scoped_release>());
    m.def("b70_set", &b70_set);
    m.def("b70_test_engine", &b70_test_engine);
    m.def("b70_forward", &b70_forward, py::call_guard<py::gil_scoped_release>());
    m.def("stage_begin", &stage_begin, py::call_guard<py::gil_scoped_release>());
    m.def("stage_begin_selected", &stage_begin_selected, py::call_guard<py::gil_scoped_release>());
    m.def("set_stage_cache", [](bool on) { G->stage_cache.store(on); });
    m.def("stage_layer", &stage_layer, py::call_guard<py::gil_scoped_release>());
    m.def("stage_end", &stage_end, py::call_guard<py::gil_scoped_release>());
    m.def("set_stage_qd", &set_stage_qd);
    m.def("shutdown", &shutdown, py::call_guard<py::gil_scoped_release>());
    m.def("sizes", [] { return std::vector<int64_t>{ (int64_t) sizeof(Req), (int64_t) sizeof(Rep), (int64_t) sizeof(LogE), MAXU, MAXP, RQ, LG, LR, MAXB, HC_N, DC_N, ST_N, DBG }; });
    dev_bind(m);
}

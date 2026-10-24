// N135 CPU-tier microbench: Qwen3.8-Flash-Next routed experts (real records from the EXL3 checkpoint, engine RAM-slot
// layout), the nv2 CPU job (ft_core.h moe_forward) vs N135 variants (ft_n135.h), cold experts, threads pinned like the
// engine (cores cpu0..cpu0+T-1, master = calling thread on cpu0). Optional user-space perf counters per worker.
// usage: cpubench DATA.bin [k=v ...]
//   threads=22 cpu0=2 iters=300 ns=1,3,4,5,6 ms=1,2,4 var=base,... check=1 hot=0 swz=0 huge=1 pf=6 perf=0 gap_us=150
//   mode=-1 (auto as the engine: I16 when every expert has <= 2 tokens, else AFFINE) | 0 AFFINE | 1 EXACT | 2 I16
//   readbw=1: pure-load pattern roofline (seq / native block / native block + prefetch) and exit
#include "ft_n135.h"

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#endif
#include <map>

using namespace std;

static map<string, string> A;
static string arg(const string& k, const string& d) { auto it = A.find(k); return it == A.end() ? d : it->second; }
static vector<int> ilist(const string& s) { vector<int> v; stringstream ss(s); string t; while (getline(ss, t, ',')) v.push_back(stoi(t)); return v; }
static vector<string> slist(const string& s) { vector<string> v; stringstream ss(s); string t; while (getline(ss, t, ',')) v.push_back(t); return v; }

// ---- perf counters (user space only; perf_event_paranoid 2 allows own threads)
struct Perf
{
    vector<int> fds;   // per thread: cycles, instructions
#ifdef __linux__
    static int open1(pid_t tid, uint32_t type, uint64_t cfg)
    {
        perf_event_attr a{}; a.size = sizeof(a); a.type = type; a.config = cfg; a.exclude_kernel = 1; a.exclude_hv = 1; a.disabled = 0;
        return int(syscall(SYS_perf_event_open, &a, tid, -1, -1, 0));
    }
    void open(const vector<pid_t>& tids)
    {
        for (pid_t t : tids)
        {
            fds.push_back(open1(t, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES));
            fds.push_back(open1(t, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS));
        }
    }
#else
    void open(const vector<pid_t>&) {}
#endif
    vector<uint64_t> read_all() const
    {
        vector<uint64_t> v(fds.size(), 0);
        for (size_t i = 0; i < fds.size(); ++i) if (fds[i] >= 0) { uint64_t x = 0; if (::read(fds[i], &x, 8) == 8) v[i] = x; }
        return v;
    }
};

static vector<pid_t> g_tids;
#ifdef __linux__
static void tid_fn(void*, int w, int) { g_tids[w] = pid_t(syscall(SYS_gettid)); }
#else
static void tid_fn(void*, int w, int) { g_tids[w] = w; }
#endif

// ---- pure-load pattern roofline
struct RB { const uint8_t* base; size_t nrec; int pattern; int pf; atomic<long> next{0}; long units; atomic<uint64_t> sink{0}; };
static void rb_fn(void* vc, int, int)
{
    RB& R = *static_cast<RB*>(vc);
    __m256i acc = _mm256_setzero_si256();
    while (true)
    {
        const long u = R.next.fetch_add(1, memory_order_relaxed);
        if (u >= R.units) break;
        if (R.pattern == 0)
        {
            // contiguous 196,608 B run (= one swizzled gate/up block unit)
            const uint8_t* p = R.base + size_t(u) * 196608;
            for (size_t o = 0; o < 196608; o += 64) acc = _mm256_xor_si256(acc, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + o)));
        }
        else
        {
            // native block unit like moe_forward phase 1: (record, g|u, blk of 16) -> 256 rows x 768 B at 12,288 B stride
            const long rec = u / 32, rem = u % 32, which = rem / 16, blk = rem % 16;
            const uint8_t* p = R.base + size_t(rec) * 9437184 + size_t(which) * 3145728 + size_t(blk) * 768;
            for (int r = 0; r < 256; ++r, p += 12288)
            {
                if (R.pattern == 2) for (int l = 0; l < 12; ++l) _mm_prefetch(reinterpret_cast<const char*>(p + R.pf * 12288) + l * 64, _MM_HINT_T0);
                if (R.pattern == 3) for (int l = 0; l < 12; ++l) _mm_prefetch(reinterpret_cast<const char*>(p + R.pf * 12288) + l * 64, _MM_HINT_T1);
                for (int o = 0; o < 768; o += 64) acc = _mm256_xor_si256(acc, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + o)));
            }
        }
    }
    alignas(32) uint64_t t[4]; _mm256_store_si256(reinterpret_cast<__m256i*>(t), acc);
    R.sink.fetch_add(t[0] ^ t[1] ^ t[2] ^ t[3]);
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "usage: cpubench DATA.bin [k=v ...]\n"); return 2; }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const string data = argv[1];
    for (int i = 2; i < argc; ++i) { string s = argv[i]; auto p = s.find('='); if (p != string::npos) A[s.substr(0, p)] = s.substr(p + 1); }
    const int T = stoi(arg("threads", "22")), CPU0 = stoi(arg("cpu0", "2")), ITERS = stoi(arg("iters", "300"));
    const int mode = stoi(arg("mode", "-1"));
    const int HOT = stoi(arg("hot", "0"));
    const double GAP = stod(arg("gap_us", "150")) * 1e-6;
    g_act_limit = stof(arg("limit", "10"));
    PF_DIST = stoi(arg("pf", "6"));
    n135::configure(A);
    init_perm(); init_tables();

    // ---- load records (engine RAM-slot layout: g/u/d trellis at 0 / 3 MiB / 6 MiB, scales copied out)
    const size_t REC = 9474048, SLOT = 9437184, OFF_U = 3145728, OFF_D = 6291456;
    int fd = open(data.c_str(), O_RDONLY);
    if (fd < 0) { perror("open data"); return 1; }
    const size_t fsz = size_t(lseek(fd, 0, SEEK_END));
    const int NREC = min<int>(int(fsz / REC), stoi(arg("nrec", "360")));
    const size_t ram_bytes = SLOT * NREC;
    uint8_t* ram = nullptr;
    if (arg("guard", "0") == "1")
    {
        // records end exactly at a PROT_NONE page (like the last RAM slot of a mapping): any read past a record faults
        uint8_t* m = static_cast<uint8_t*>(mmap(nullptr, ram_bytes + 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        mprotect(m + ram_bytes, 4096, PROT_NONE);
        ram = m;
    }
    else if (arg("huge", "1") == "1") ram = static_cast<uint8_t*>(big_alloc(ram_bytes));
    else
    {
        // 4 KiB pages, like an mmap'd region that THP did not back
        ram = static_cast<uint8_t*>(mmap(nullptr, ram_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        madvise(ram, ram_bytes, MADV_NOHUGEPAGE);
    }
    vector<uint16_t> scales(size_t(NREC) * (REC - SLOT) / 2);
    {
        vector<uint8_t> buf(REC);
        for (int r = 0; r < NREC; ++r)
        {
            size_t got = 0; const off_t off = off_t(r) * REC;
            while (got < REC) { ssize_t k = pread(fd, buf.data() + got, REC - got, off + got); if (k <= 0) { perror("pread"); return 1; } got += k; }
            memcpy(ram + size_t(r) * SLOT, buf.data(), SLOT);
            memcpy(scales.data() + size_t(r) * (REC - SLOT) / 2, buf.data() + SLOT, REC - SLOT);
        }
        close(fd);
    }
    Layer L; L.H = 4096; L.I = 2048; L.ex.resize(NREC);
    for (int r = 0; r < NREC; ++r)
    {
        const uint16_t* sc = scales.data() + size_t(r) * (REC - SLOT) / 2;   // g suh 4096, svh 2048, u suh, svh, d suh 2048, svh 4096
        uint8_t* a = ram + size_t(r) * SLOT;
        L.ex[r].g = { a, sc, sc + 4096, 4096, 2048 };
        L.ex[r].u = { a + OFF_U, sc + 6144, sc + 6144 + 4096, 4096, 2048 };
        L.ex[r].d = { a + OFF_D, sc + 12288, sc + 12288 + 2048, 2048, 4096 };
    }
    if (arg("swz", "0") == "1") swizzle_layer(L);
    fprintf(stderr, "loaded %d records (%.2f GB), huge=%s swz=%s\n", NREC, NREC * double(SLOT) / 1e9, arg("huge", "1").c_str(), arg("swz", "0").c_str());

    vector<int> cpus; for (int i = 0; i < T; ++i) cpus.push_back(CPU0 + i);
    if (!arg("cpus", "").empty()) cpus = ilist(arg("cpus", ""));
    Pool pool; pool.start(T, cpus);
    g_tids.assign(T, 0); pool.run(&tid_fn, nullptr);
    n135::Engine2 eng; eng.init(pool); eng.late_us = stod(arg("late_us", "50"));
    Perf perf; if (arg("perf", "0") == "1") perf.open(g_tids);

    if (arg("readbw", "0") == "1")
    {
        printf("pattern,pf,GBps\n");
        for (int pat : { 0, 1, 2, 3 })
            for (int pf : (pat >= 2 ? vector<int>{ 2, 4, 8, 16 } : vector<int>{ 0 }))
            {
                double best = 0;
                for (int rep = 0; rep < 5; ++rep)
                {
                    RB R; R.base = ram; R.nrec = NREC; R.pattern = pat; R.pf = pf;
                    R.units = pat == 0 ? long(NREC) * 9437184 / 196608 / 2 : long(NREC / 2) * 32;   // ~half the data per rep
                    // rotate which half
                    if (rep & 1) R.base = ram + size_t(NREC / 2) * SLOT;
                    const double t0 = now(); pool.run(&rb_fn, &R); const double dt = now() - t0;
                    const double bytes = pat == 0 ? double(R.units) * 196608 : double(R.units) * 256 * 768;
                    best = max(best, bytes / dt / 1e9);
                }
                printf("%s,%d,%.1f\n", pat == 0 ? "seq196k" : pat == 1 ? "native_blk" : pat == 2 ? "native_blk_pfT0" : "native_blk_pfT1", pf, best);
                fflush(stdout);
            }
        pool.stop();
        return 0;
    }

    mt19937_64 rng(1234);
    const vector<string> vars = slist(arg("var", "base"));
    const int maxtok = 8;
    vector<float> x = make_x(maxtok, 4096, 7);
    vector<float> out(size_t(maxtok) * 4096), out2(size_t(maxtok) * 4096);
    // the engine resolves auto (-1) before calling moe_forward: I16 when every expert has <= 2 tokens, else AFFINE
    auto resolve = [&](int md, const vector<vector<pair<int, float>>>& route) {
        if (md >= 0) return md;
        map<int, int> cnt; int mx = 0; for (auto& r : route) for (auto& pr : r) mx = max(mx, ++cnt[pr.first]);
        return mx <= 2 ? 2 : 0;
    };
    auto run_var = [&](const string& v, int m, const vector<vector<pair<int, float>>>& route, float* o) {
        if (v == "base") moe_forward(pool, L, x.data(), m, route, o, resolve(mode, route));
        else eng.forward(v, L, x.data(), m, route, o, mode);
    };

    // ---- numerics: every variant vs base and vs the fp64 GPU-exact-weight reference
    if (arg("check", "1") == "1")
    {
        for (int m : { 1, 2, 3, 4 })
        {
            const int n = 4;
            vector<vector<pair<int, float>>> route(m);
            vector<int> ex; while (int(ex.size()) < n) { int e = int(rng() % NREC); if (find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e); }
            for (int t = 0; t < m; ++t) for (int i = 0; i < n; ++i) route[t].push_back({ ex[i], 0.2f + 0.05f * i });
            vector<double> ref(size_t(m) * 4096, 0.0);
            for (int i = 0; i < n; ++i)
            {
                const Expert& E = L.ex[ex[i]];
                Expert En = E; En.g.swz = En.u.swz = En.d.swz = 0;
                if (E.g.swz) { fprintf(stderr, "check: skip fp64 ref on swizzled data\n"); break; }
                vector<float> W[3] = { dense_weights(En.g), dense_weights(En.u), dense_weights(En.d) };
                vector<float> y(4096);
                for (int t = 0; t < m; ++t) { ref_expert(En, W, x.data() + size_t(t) * 4096, y.data()); for (int c = 0; c < 4096; ++c) ref[size_t(t) * 4096 + c] += double(route[t][i].second) * y[c]; }
            }
            vector<float> reff(ref.begin(), ref.end());
            for (int md : { 1, 0, 2, -1 })
            {
                if (mode >= 0 && md != mode) continue;
                moe_forward(pool, L, x.data(), m, route, out.data(), resolve(md, route));
                const Err eb = compare(out.data(), reff.data(), size_t(m) * 4096);
                if (!arg("dump", "").empty()) { FILE* f = fopen(arg("dump", "").c_str(), "ab"); fwrite(out.data(), 4, size_t(m) * 4096, f); fclose(f); }
                fprintf(stderr, "check m=%d mode=%2d base: rel_rms %.3e vs fp64\n", m, md, eb.rel_rms);
                for (auto& v : vars)
                {
                    if (v == "base") continue;
                    std::fill(out2.begin(), out2.end(), 0.f);
                    eng.forward(v, L, x.data(), m, route, out2.data(), md);
                    const Err e1 = compare(out2.data(), reff.data(), size_t(m) * 4096), e2 = compare(out2.data(), out.data(), size_t(m) * 4096);
                    size_t neq = 0; for (size_t c = 0; c < size_t(m) * 4096; ++c) neq += out2[c] != out[c];
                    fprintf(stderr, "check m=%d mode=%2d %s: rel_rms %.3e vs fp64 | vs base rel %.3e max_abs %.3e, %zu/%d values differ\n", m, md, v.c_str(),
                            e1.rel_rms, e2.rel_rms, e2.max_abs, neq, m * 4096);
                    if (v == "v4fail")
                    {
                        // expected: base over the picks that were not late (every second distinct expert dropped)
                        vector<vector<pair<int, float>>> r2(m); vector<char> late(NREC, 0); int k = 0;
                        for (auto& r : route) for (auto& pr : r) if (!late[pr.first] && (k++ & 1)) late[pr.first] = 2;
                        for (int t = 0; t < m; ++t) for (auto& pr : route[t]) if (late[pr.first] != 2) r2[t].push_back(pr);
                        vector<float> o3(size_t(m) * 4096);
                        moe_forward(pool, L, x.data(), m, r2, o3.data(), resolve(md, route));
                        const Err e3 = compare(out2.data(), o3.data(), size_t(m) * 4096);
                        fprintf(stderr, "check m=%d mode=%2d v4fail vs base(non-late picks): rel %.3e max_abs %.3e\n", m, md, e3.rel_rms, e3.max_abs);
                    }
                }
            }
        }
    }

    // ---- timing: per call, cold experts
    printf("var,m,n,iters,mean_ms,p50_ms,p90_ms,ms_per_expert,GBps,ipc,ginstr_per_expert\n");
    for (int m : ilist(arg("ms", "1,2,4")))
        for (int n : ilist(arg("ns", "1,3,4,5,6")))
            for (auto& v : vars)
            {
                vector<double> ts; int cur = int(rng() % NREC);
                vector<uint64_t> p0, p1;
                for (int it = 0; it < ITERS + 20; ++it)
                {
                    vector<vector<pair<int, float>>> route(m);
                    for (int i = 0; i < n; ++i) { cur = (cur + 37) % (HOT > 0 ? HOT : NREC); for (int t = 0; t < m; ++t) route[t].push_back({ cur, 0.125f }); }
                    const double tg = now(); while (now() - tg < GAP) _mm_pause();
                    if (it == 20 && !perf.fds.empty()) p0 = perf.read_all();
                    const double t0 = now();
                    run_var(v, m, route, out.data());
                    const double dt = now() - t0;
                    if (it >= 20) ts.push_back(dt * 1e3);
                }
                double ipc = 0, gi = 0;
                if (!perf.fds.empty())
                {
                    p1 = perf.read_all(); double cyc = 0, ins = 0;
                    for (size_t i = 0; i + 1 < p1.size(); i += 2) { cyc += double(p1[i] - p0[i]); ins += double(p1[i + 1] - p0[i + 1]); }
                    ipc = cyc > 0 ? ins / cyc : 0; gi = ins / ITERS / n / 1e9;   // includes spin-wait instructions of idle workers
                }
                sort(ts.begin(), ts.end());
                double mean = 0; for (double t : ts) mean += t; mean /= ts.size();
                printf("%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.1f,%.2f,%.4f\n", v.c_str(), m, n, ITERS, mean, ts[ts.size() / 2], ts[ts.size() * 9 / 10], mean / n,
                       n * double(SLOT) / (mean * 1e-3) / 1e9, ipc, gi);
                fflush(stdout);
            }
    pool.stop();
    return 0;
}

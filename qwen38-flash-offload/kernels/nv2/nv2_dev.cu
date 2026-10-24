// N119 nv2 device side. Per MoE layer call (decode / small forwards), all on the compute stream, no host sync:
//   nv_pub   (1 block)  dedup the picks, classify VRAM / not, publish a request (+ x rows for a CPU job) to mapped memory
//   nv_step  (1 block)  wait for the host's reply (plan_layer lanes), mask the CPU-lane picks out of the GPU selection,
//                       then the expert-cache CLOCK step on the GPU lane: VRAM victims get a fresh RAM landing slot
//                       (exclusive RAM tier; logged for the host), admissions are logged (host frees their RAM copy)
//   nv_copy  (grid)     gather admitted experts into their VRAM slot: RAM sources first, NVMe sources after their
//                       per-key landed flag (ram_res) is set by the host; the victim's old bytes go to its landing slot
//                       in the same pass (read old -> write host, then write new); block 0 also points the rows of
//                       non-admitted GPU-lane picks at their (landed) RAM slot and checks every GPU-lane row
//   [exllamav3 MoE kernel]
//   nv_combine          wait for the CPU job's done seq (bounded), add its fp32 partial
// Every device wait is bounded (globaltimer) and raises HC_ERR on timeout; the host aborts on HC_ERR.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "nv2_shared.h"
using namespace nv2s;

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ long long ldv64(const volatile long long* p) { return *p; }

__global__ void nv_pub_k(const int64_t* __restrict__ sel, const __half* __restrict__ w, const __half* __restrict__ z,
    int n, int bsz, int H, int li, int E, int first, const int* __restrict__ slotof, int kind, int send_x,
    volatile long long* hc, Req* req, __half* hx, long long* dc, int* uidx, int* uexp, long long timeout_ns,
    const int64_t* __restrict__ pf, int npf, const int* __restrict__ slotof_next)
{
    __shared__ int cnt[MAXU];
    __shared__ int pos[MAXU];
    __shared__ unsigned char pfm[MAXU];
    __shared__ long long s_seq;
    __shared__ int s_nu;
    const int t = threadIdx.x;
    for (int e = t; e < E; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s] - first;
        if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (t == 0)
    {
        int u = 0;
        for (int e = 0; e < E; ++e) pos[e] = cnt[e] ? u++ : -1;
        s_nu = u;
        const long long sq = dc[DC_SEQ] + 1;
        dc[DC_SEQ] = sq; dc[DC_NU] = u;
        s_seq = sq;
        const unsigned long long t0 = gtime();
        while (ldv64(hc + HC_ACK) < sq - RQ + 1)          // ring entry still unread by the host
            if ((long long) (gtime() - t0) > timeout_ns) { hc[HC_ERR] = 2; break; }
    }
    __syncthreads();
    const long long sq = s_seq;
    const int nu = s_nu;
    Req* r = req + (sq % RQ);
    for (int e = t; e < E; e += blockDim.x)
    {
        const int u = pos[e];
        uidx[e] = u;
        if (u >= 0)
        {
            uexp[u] = e;
            r->key[u] = li * E + e; r->cls[u] = slotof[e] >= 0 ? LN_VRAM : 0; r->cnt[u] = cnt[e];
        }
    }
    if (n <= MAXP)
        for (int s = t; s < n; s += blockDim.x) { r->pe[s] = (int) (sel[s] - first); r->pw[s] = __half2float(w[s]); }
    if (send_x)
    {
        const int nx = bsz * H / 8;
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
    }
    if (pf && npf > 0)
    {
        for (int e = t; e < E; e += blockDim.x) pfm[e] = 0;
        __syncthreads();
        for (int i = t; i < npf; i += blockDim.x)
        {
            const long long e = pf[i] - first;
            if (e >= 0 && e < E && slotof_next[e] < 0) pfm[e] = 1;
        }
        __syncthreads();
        if (t == 0)
        {
            int k = 0;
            for (int e = 0; e < E && k < MAXP; ++e) if (pfm[e]) r->pf_key[k++] = (li + 1) * E + e;
            r->npf = k;
        }
    }
    else if (t == 0) r->npf = 0;
    if (t == 0) { r->li = li; r->nu = nu; r->np = n; r->ntok = bsz; r->kind = kind; const long long tp = (long long) gtime(); r->tpub = tp; dc[DC_TPUB] = tp; }
    __threadfence_system();
    __syncthreads();
    if (t == 0)
    {
        *(volatile long long*) &r->seq = sq;
        __threadfence_system();
        hc[HC_PUB] = sq;
    }
}

// N136: nv_pub_k with the two serial thread-0 scans over E (unique-expert positions, prefetch-hint keys) replaced by
// warp-ballot compactions (same ascending order, same MAXP cap), the ack wait overlapped with the counting, and the
// x-row copy issued before the request words. Same request contents bit for bit; 320 threads = 10 warps >= E / 32.
constexpr int PUBF_T = 512;
__global__ __launch_bounds__(PUBF_T) void nv_pub_fast_k(const int64_t* __restrict__ sel, const __half* __restrict__ w,
    const __half* __restrict__ z, int n, int bsz, int H, int li, int E, int first, const int* __restrict__ slotof, int kind,
    int send_x, volatile long long* hc, Req* req, __half* hx, long long* dc, int* uidx, int* uexp, long long timeout_ns,
    const int64_t* __restrict__ pf, int npf, const int* __restrict__ slotof_next)
{
    __shared__ int cnt[MAXU];
    __shared__ unsigned char pfm[MAXU];
    __shared__ int wsum[PUBF_T / 32], wsum2[PUBF_T / 32];
    __shared__ long long s_seq;
    const int t = threadIdx.x, lane = t & 31, wp = t >> 5;
    const bool do_pf = pf && npf > 0;
    for (int e = t; e < E; e += blockDim.x) { cnt[e] = 0; pfm[e] = 0; }
    if (t == 0)
    {
        const long long sq = dc[DC_SEQ] + 1;
        dc[DC_SEQ] = sq;
        s_seq = sq;
    }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s] - first;
        if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    if (do_pf)
        for (int i = t; i < npf; i += blockDim.x)
        {
            const long long e = pf[i] - first;
            if (e >= 0 && e < E && slotof_next[e] < 0) pfm[e] = 1;
        }
    if (t == 0)
    {
        const long long sq = s_seq;
        const unsigned long long t0 = gtime();
        while (ldv64(hc + HC_ACK) < sq - RQ + 1)          // ring entry still unread by the host
            if ((long long) (gtime() - t0) > timeout_ns) { hc[HC_ERR] = 2; break; }
    }
    __syncthreads();
    const long long sq = s_seq;
    Req* r = req + (sq % RQ);
    if (send_x)
    {
        const int nx = bsz * H / 8;
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
    }
    // ballot compaction, one expert per thread (E <= PUBF_T)
    const int e = t;
    const bool f = e < E && cnt[e] > 0;
    const bool g = do_pf && e < E && pfm[e];
    const unsigned bf = __ballot_sync(0xffffffffu, f), bg = __ballot_sync(0xffffffffu, g);
    const unsigned lt = (1u << lane) - 1u;
    if (lane == 0) { wsum[wp] = __popc(bf); wsum2[wp] = __popc(bg); }
    __syncthreads();
    int base = 0, base2 = 0, tot = 0, tot2 = 0;
    #pragma unroll
    for (int k = 0; k < PUBF_T / 32; ++k)
    {
        if (k < wp) { base += wsum[k]; base2 += wsum2[k]; }
        tot += wsum[k]; tot2 += wsum2[k];
    }
    if (e < E)
    {
        const int u = f ? base + __popc(bf & lt) : -1;
        uidx[e] = u;
        if (f)
        {
            uexp[u] = e;
            r->key[u] = li * E + e; r->cls[u] = slotof[e] >= 0 ? LN_VRAM : 0; r->cnt[u] = cnt[e];
        }
        if (g)
        {
            const int k = base2 + __popc(bg & lt);
            if (k < MAXP) r->pf_key[k] = (li + 1) * E + e;
        }
    }
    if (n <= MAXP)
        for (int s = t; s < n; s += blockDim.x) { r->pe[s] = (int) (sel[s] - first); r->pw[s] = __half2float(w[s]); }
    if (t == 0)
    {
        dc[DC_NU] = tot;
        r->npf = do_pf ? min(tot2, MAXP) : 0;
        r->li = li; r->nu = tot; r->np = n; r->ntok = bsz; r->kind = kind;
        const long long tp = (long long) gtime(); r->tpub = tp; dc[DC_TPUB] = tp;
    }
    __threadfence_system();
    __syncthreads();
    if (t == 0)
    {
        *(volatile long long*) &r->seq = sq;
        __threadfence_system();
        hc[HC_PUB] = sq;
    }
}

__global__ void nv_step_k(const int64_t* __restrict__ sel, const __half* __restrict__ w, int64_t* sel_out, __half* w_out,
    int n, int first, int li, int E, int S, int admit, int* slotof_all, const volatile long long* home_all,
    const volatile int* ram_res, const int64_t* tabs, int* owner, int* refb, int* pin, int* ctl, int* jobs, int* jobx,
    int* jobnv, int maxj, unsigned long long* stats, const long long* __restrict__ cbase, int spc, long long rec,
    long long off_u, long long off_d, volatile long long* hc, const Rep* rep, long long* dc, const int* uidx, int* ulane,
    LogE* wbl, LogE* adml, const volatile int* land, long long* dflag, long long timeout_ns, int wb_on, long long* dbg)
{
    __shared__ unsigned mask[32];
    __shared__ int misses[MAXU];
    __shared__ int nmiss, nhit, epoch, s_ok;
    __shared__ long long s_seq;
    const int t = threadIdx.x;
    if (t == 0)
    {
        s_seq = dc[DC_SEQ];
        s_ok = 1;
        const volatile Rep* rp = rep + (s_seq % RQ);
        const unsigned long long t0 = gtime();
        while (rp->seq != s_seq)
            if ((long long) (gtime() - t0) > timeout_ns) { hc[HC_ERR] = 1; s_ok = 0; stats[ST_TIMEOUT]++; break; }
        const unsigned long long t1 = gtime();
        stats[ST_WAITNS] += t1 - t0; stats[ST_WAITS]++;
        if (dbg) { long long* d = dbg + (s_seq % DBG) * DBGW; d[0] = s_seq; d[1] = dc[DC_TPUB]; d[2] = (long long) t0; d[3] = (long long) t1; }
        __threadfence_system();
    }
    __syncthreads();
    const long long sq = s_seq;
    const volatile Rep* rp = rep + (sq % RQ);
    const int nu = (int) dc[DC_NU];
    for (int u = t; u < nu; u += blockDim.x) ulane[u] = s_ok ? rp->lane[u] : LN_RAM;
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s] - first;
        int ln = LN_VRAM;
        if (e >= 0 && e < E) { const int u = uidx[e]; if (u >= 0) ln = ulane[u]; }
        const bool c = ln == LN_CPU || ln == LN_NVC || ln == LN_B70;
        sel_out[s] = c ? -1 : sel[s];
        w_out[s] = c ? __float2half(0.0f) : w[s];
    }
    if (t < 32) mask[t] = 0;
    if (t == 0)
    {
        dflag[0] = (s_ok && rp->ncpu > 0) ? sq : 0;
        nmiss = 0; nhit = 0; epoch = ctl[1] + 1; ctl[1] = epoch; ctl[2] = 0;
    }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel_out[s] - first;
        if (e >= 0 && e < E)
        {
            const int u = uidx[e];
            const int ln = u >= 0 ? ulane[u] : LN_VRAM;
            if (ln != LN_RZC && ln != LN_NZC) atomicOr(&mask[e >> 5], 1u << (e & 31));   // zero-copy lanes: no admission
        }
    }
    __syncthreads();
    int* slotof = slotof_all + (long long) li * E;
    for (int e = t; e < E; e += blockDim.x)
        if (mask[e >> 5] & (1u << (e & 31)))
        {
            const int s = slotof[e];
            if (s >= 0) { refb[s] = 1; pin[s] = epoch; atomicAdd(&nhit, 1); }
            else { const int k = atomicAdd(&nmiss, 1); misses[k] = e; }
        }
    __syncthreads();
    if (t != 0) return;
    stats[ST_HIT] += nhit; stats[ST_MISS] += nmiss;
    if (!admit || S == 0) { if (dbg) dbg[(sq % DBG) * DBGW + 4] = (long long) gtime(); return; }
    int hand = ctl[0], nj = 0;
    int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
    int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
    long long wbn = dc[DC_WBN], admn = dc[DC_ADMN], lh = dc[DC_LANDH];
    const long long lt = ldv64(hc + HC_LANDT);
    bool full = false;
    for (int pass = 0; pass < 2 && !full; ++pass)          // RAM-sourced jobs first, NVMe-sourced (landed wait) after
        for (int k = 0; k < nmiss && nj < maxj; k++)
        {
            const int e = misses[k];
            const int u = uidx[e];
            const int nvj = (u >= 0 && ulane[u] == LN_NVG) ? 1 : 0;
            if (nvj != pass) continue;
            int found = -1;
            for (int step = 0; step < 2 * S; step++)
            {
                const int s = hand; hand = (hand + 1 == S) ? 0 : hand + 1;
                if (pin[s] == epoch || pin[s] < 0) continue;
                if (refb[s]) { refb[s] = 0; continue; }
                found = s; break;
            }
            if (found < 0) { stats[ST_NOVICT]++; full = true; break; }
            const int s = found, old = owner[s];
            int jx = -1;
            if (old >= 0)
            {
                const int ol = old / E, oe = old % E;
                slotof_all[(long long) ol * E + oe] = -1;
                const volatile long long* h = home_all + (long long) old * 3;
                int64_t* og = (int64_t*) tabs[ol * 4 + 0]; int64_t* ou = (int64_t*) tabs[ol * 4 + 1];
                int64_t* od = (int64_t*) tabs[ol * 4 + 2]; int64_t* ogu = (int64_t*) tabs[ol * 4 + 3];
                const long long h0 = h[0], h1 = h[1], h2 = h[2];
                og[oe] = h0; ou[oe] = h1; od[oe] = h2;
                if (ogu) { ogu[2 * oe] = h0; ogu[2 * oe + 1] = h1; }
                stats[ST_EVICT]++;
                if (wb_on && ram_res[old] == 0)
                {
                    if (lh < lt)
                    {
                        jx = land[lh % LR]; lh++;
                        LogE* le = wbl + (wbn % LG); le->key = old; le->slot = jx; le->seq = sq; wbn++;
                        stats[ST_WB]++;
                    }
                    else stats[ST_WB_NOSLOT]++;
                }
            }
            const int key = li * E + e;
            owner[s] = key; slotof[e] = s; refb[s] = 1; pin[s] = epoch;
            const long long a = cbase[s / spc] + (long long) (s % spc) * rec;
            pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
            if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
            jobs[2 * nj] = e; jobs[2 * nj + 1] = s; jobx[nj] = jx; jobnv[nj] = nvj;
            LogE* le = adml + (admn % LG); le->key = key; le->slot = s; le->seq = sq; admn++;
            nj++;
            stats[ST_ADMIT]++;
        }
    ctl[0] = hand; ctl[2] = nj;
    dc[DC_WBN] = wbn; dc[DC_ADMN] = admn; dc[DC_LANDH] = lh;
    if (dbg) { dbg[(sq % DBG) * DBGW + 4] = (long long) gtime(); dbg[(sq % DBG) * DBGW + 7] = nj; }
    __threadfence_system();
    hc[HC_WBN] = wbn; hc[HC_ADMN] = admn;
}

__global__ void nv_copy_k(const int* __restrict__ ctl, const int* __restrict__ jobs, const int* __restrict__ jobx,
    const int* __restrict__ jobnv, int li, int E, const volatile long long* home_all, const volatile int* ram_res,
    const long long* __restrict__ cbase, int spc, long long rec, long long vg, long long vu, long long vd, long long off_u,
    long long off_d, const long long* __restrict__ ram_addr, const long long* dc, const int* uexp, const int* ulane,
    const int* slotof_all, const int64_t* tabs, long long ph, volatile long long* hc, long long timeout_ns,
    unsigned long long* stats, long long* dbg, long long* dcw)
{
    const int t = threadIdx.x;
    const long long sqd = dc[DC_SEQ];
    if (dbg && blockIdx.x == 0 && t == 0) dbg[(sqd % DBG) * DBGW + 5] = (long long) gtime();
    __shared__ long long s_h[3];
    __shared__ long long s_wb;
    __shared__ int s_job;
    if (blockIdx.x == 0)
    {
        // rows of GPU-lane picks: admitted / VRAM -> their slot (set by nv_step); otherwise -> the landed RAM slot
        const int nu = (int) dc[DC_NU];
        int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
        int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
        for (int u = t; u < nu; u += blockDim.x)
        {
            const int ln = ulane[u];
            if (ln == LN_CPU || ln == LN_NVC || ln == LN_B70) continue;
            const int e = uexp[u];
            const long long key = (long long) li * E + e;
            const int s = slotof_all[key];
            if (s >= 0)
            {
                const long long a = cbase[s / spc] + (long long) (s % spc) * rec;
                if (pg[e] != a || pu[e] != a + off_u || pd[e] != a + off_d) atomicAdd(&stats[ST_BAD], 1ull);
                continue;
            }
            if (ln == LN_NVG || ln == LN_RAM || ln == LN_RZC || ln == LN_NZC)
            {
                const unsigned long long t0 = gtime();
                while (ram_res[key] != 1)
                {
                    if ((long long) (gtime() - t0) > timeout_ns) { hc[HC_ERR] = 3; atomicAdd(&stats[ST_TIMEOUT], 1ull); break; }
                    __nanosleep(500);
                }
            }
            const long long h0 = home_all[key * 3], h1 = home_all[key * 3 + 1], h2 = home_all[key * 3 + 2];
            if (h0 == ph) atomicAdd(&stats[ST_BAD], 1ull);
            pg[e] = h0; pu[e] = h1; pd[e] = h2;
            if (gu) { gu[2 * e] = h0; gu[2 * e + 1] = h1; }
            atomicAdd(&stats[ST_FIX], 1ull);
        }
    }
    const long long nj = ctl[2];
    if (nj == 0) { if (dbg && blockIdx.x == 0 && t == 0) dbg[(sqd % DBG) * DBGW + 6] = (long long) gtime(); return; }
    const long long vpj = vg + vu + vd;
    const int CH = blockDim.x * 4;
    const long long nch = (vpj + CH - 1) / CH;
    const long long total = nj * nch;
    if (t == 0) s_job = -1;
    bool wrote_host = false;
    for (long long c = blockIdx.x; c < total; c += gridDim.x)
    {
        const int j = (int) (c / nch);
        const long long r0 = (c % nch) * CH;
        __syncthreads();
        if (t == 0 && s_job != j)
        {
            const long long key = (long long) li * E + jobs[2 * j];
            if (jobnv[j])
            {
                const unsigned long long t0 = gtime();
                while (ram_res[key] != 1)
                {
                    if ((long long) (gtime() - t0) > timeout_ns) { hc[HC_ERR] = 4; atomicAdd(&stats[ST_TIMEOUT], 1ull); break; }
                    __nanosleep(500);
                }
                if (blockIdx.x == 0) { atomicAdd(&stats[ST_CWAITNS], gtime() - t0); atomicAdd(&stats[ST_CWAITS], 1ull); }
            }
            s_h[0] = home_all[key * 3]; s_h[1] = home_all[key * 3 + 1]; s_h[2] = home_all[key * 3 + 2];
            s_wb = jobx[j] >= 0 ? ram_addr[jobx[j]] : 0;
            s_job = j;
        }
        __syncthreads();
        const int s = jobs[2 * j + 1];
        const long long dbase = cbase[s / spc] + (long long) (s % spc) * rec;
        int4 v[4]; int4* dst[4]; int4* wbd[4];
        int nn = 0;
        #pragma unroll
        for (int q = 0; q < 4; q++)
        {
            const long long r = r0 + q * blockDim.x + t;
            if (r < vpj)
            {
                const int4* src; long long doff;
                if (r < vg) { src = (const int4*) s_h[0] + r; doff = r * 16; }
                else if (r < vg + vu) { src = (const int4*) s_h[1] + (r - vg); doff = off_u + (r - vg) * 16; }
                else { src = (const int4*) s_h[2] + (r - vg - vu); doff = off_d + (r - vg - vu) * 16; }
                v[q] = *src;
                dst[q] = (int4*) (dbase + doff);
                wbd[q] = s_wb ? (int4*) (s_wb + doff) : nullptr;
                nn = q + 1;
            }
        }
        #pragma unroll
        for (int q = 0; q < 4; q++) if (q < nn && wbd[q]) { *wbd[q] = *dst[q]; wrote_host = true; }
        #pragma unroll
        for (int q = 0; q < 4; q++) if (q < nn) *dst[q] = v[q];
    }
    if (wrote_host) __threadfence_system();
    if (dbg)
    {
        __syncthreads();
        if (t == 0)
        {
            __threadfence();
            const unsigned long long k = atomicAdd((unsigned long long*) &dcw[DC_CDONE], 1ull);
            if (k == gridDim.x - 1) { dbg[(sqd % DBG) * DBGW + 6] = (long long) gtime(); dcw[DC_CDONE] = 0; }
        }
    }
}

__global__ void nv_combine_k(void* out, int is_half, int n, const long long* __restrict__ dflag, const volatile long long* done,
    const float* hout, unsigned long long* stats, volatile long long* hc, long long timeout_ns)
{
    __shared__ int go;
    if (threadIdx.x == 0)
    {
        const long long s = dflag[0];
        go = s != 0;
        if (go)
        {
            const unsigned long long t0 = gtime();
            unsigned long long t1 = t0;
            while (*done < s)
            {
                t1 = gtime();
                if ((long long) (t1 - t0) > timeout_ns) { go = 0; hc[HC_ERR] = 5; if (blockIdx.x == 0) atomicAdd(&stats[ST_TIMEOUT], 1ull); break; }
            }
        }
    }
    __syncthreads();
    if (!go) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const float v = __ldcv(hout + i);
        if (is_half) { __half* o = (__half*) out; o[i] = __float2half(__half2float(o[i]) + v); }
        else ((float*) out)[i] += v;
    }
}

// staged-prefill restore: rows of experts idx[] (local ids) of layer li back to home (mapped)
__global__ void nv_restore_k(const int64_t* __restrict__ idx, int n, int li, int E, const volatile long long* home_all,
    const int64_t* tabs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int e = (int) idx[i];
    const long long key = (long long) li * E + e;
    int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
    int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
    const long long h0 = home_all[key * 3], h1 = home_all[key * 3 + 1], h2 = home_all[key * 3 + 2];
    pg[e] = h0; pu[e] = h1; pd[e] = h2;
    if (gu) { gu[2 * e] = h0; gu[2 * e + 1] = h1; }
}

// all rows of non-VRAM experts -> home (startup / after bulk residency changes)
__global__ void nv_rows_all_k(int L, int E, const int* __restrict__ slotof_all, const volatile long long* home_all, const int64_t* tabs)
{
    const long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long) L * E) return;
    const int l = (int) (g / E), e = (int) (g % E);
    if (slotof_all[g] >= 0) return;
    int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
    int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
    const long long h0 = home_all[g * 3], h1 = home_all[g * 3 + 1], h2 = home_all[g * 3 + 2];
    pg[e] = h0; pu[e] = h1; pd[e] = h2;
    if (gu) { gu[2 * e] = h0; gu[2 * e + 1] = h1; }
}

// ---------------------------------------------------------------------------------------------------------------
static inline cudaStream_t cur(const at::Tensor& t) { return at::cuda::getCurrentCUDAStream(t.device().index()); }

void nv_pub(at::Tensor sel, at::Tensor w, int64_t z, int64_t n, int64_t bsz, int64_t H, int64_t li, int64_t E, int64_t first,
            at::Tensor slotof_row, int64_t kind, int64_t send_x, int64_t hc, int64_t req, int64_t hx, at::Tensor dc,
            at::Tensor uidx, at::Tensor uexp, int64_t timeout_ns, int64_t pf, int64_t npf, at::Tensor slotof_next)
{
    c10::cuda::CUDAGuard g(sel.device());
    TORCH_CHECK(sel.dtype() == at::kLong && sel.is_contiguous() && w.dtype() == at::kHalf && w.is_contiguous());
    TORCH_CHECK(E <= MAXU);
    static const int pub_fast = getenv("QWEN38_NV_PUBFAST") && atoi(getenv("QWEN38_NV_PUBFAST")) == 1 && MAXU <= PUBF_T;
    if (pub_fast)
    {
        nv_pub_fast_k<<<1, PUBF_T, 0, cur(sel)>>>(sel.data_ptr<int64_t>(), (const __half*) w.data_ptr(), (const __half*) z,
            (int) n, (int) bsz, (int) H, (int) li, (int) E, (int) first, slotof_row.data_ptr<int>(), (int) kind,
            (int) send_x, (volatile long long*) hc, (Req*) req, (__half*) hx, (long long*) dc.data_ptr<int64_t>(),
            uidx.data_ptr<int>(), uexp.data_ptr<int>(), timeout_ns, (const int64_t*) pf, (int) npf,
            slotof_next.data_ptr<int>());
        return;
    }
    nv_pub_k<<<1, 256, 0, cur(sel)>>>(sel.data_ptr<int64_t>(), (const __half*) w.data_ptr(), (const __half*) z, (int) n,
        (int) bsz, (int) H, (int) li, (int) E, (int) first, slotof_row.data_ptr<int>(), (int) kind, (int) send_x,
        (volatile long long*) hc, (Req*) req, (__half*) hx, (long long*) dc.data_ptr<int64_t>(), uidx.data_ptr<int>(),
        uexp.data_ptr<int>(), timeout_ns, (const int64_t*) pf, (int) npf, slotof_next.data_ptr<int>());
}

void nv_step(at::Tensor sel, at::Tensor w, at::Tensor sel_out, at::Tensor w_out, int64_t n, int64_t first, int64_t li,
             int64_t E, int64_t S, int64_t admit, at::Tensor slotof, int64_t home, int64_t ram_res, at::Tensor tabs,
             at::Tensor owner, at::Tensor refb, at::Tensor pin, at::Tensor ctl, at::Tensor jobs, at::Tensor jobx,
             at::Tensor jobnv, at::Tensor stats, int64_t arena, int64_t spc, int64_t rec, int64_t off_u, int64_t off_d,
             int64_t hc, int64_t rep, at::Tensor dc, at::Tensor uidx, at::Tensor ulane, int64_t wbl, int64_t adml,
             int64_t land, at::Tensor dflag, int64_t timeout_ns, int64_t wb_on, at::Tensor dbg)
{
    c10::cuda::CUDAGuard g(sel.device());
    nv_step_k<<<1, 512, 0, cur(sel)>>>(sel.data_ptr<int64_t>(), (const __half*) w.data_ptr(), sel_out.data_ptr<int64_t>(),
        (__half*) w_out.data_ptr(), (int) n, (int) first, (int) li, (int) E, (int) S, (int) admit, slotof.data_ptr<int>(),
        (const volatile long long*) home, (const volatile int*) ram_res, tabs.data_ptr<int64_t>(), owner.data_ptr<int>(),
        refb.data_ptr<int>(), pin.data_ptr<int>(), ctl.data_ptr<int>(), jobs.data_ptr<int>(), jobx.data_ptr<int>(),
        jobnv.data_ptr<int>(), (int) jobx.numel(), (unsigned long long*) stats.data_ptr<int64_t>(),
        (const long long*) arena, (int) spc, rec, off_u, off_d, (volatile long long*) hc, (const Rep*) rep,
        (long long*) dc.data_ptr<int64_t>(), uidx.data_ptr<int>(), ulane.data_ptr<int>(), (LogE*) wbl, (LogE*) adml,
        (const volatile int*) land, (long long*) dflag.data_ptr<int64_t>(), timeout_ns, (int) wb_on,
        dbg.numel() ? (long long*) dbg.data_ptr<int64_t>() : nullptr);
}

void nv_copy(at::Tensor ctl, at::Tensor jobs, at::Tensor jobx, at::Tensor jobnv, int64_t li, int64_t E, int64_t home,
             int64_t ram_res, int64_t arena, int64_t spc, int64_t rec, int64_t sg, int64_t su, int64_t sd, int64_t off_u,
             int64_t off_d, at::Tensor ram_addr, at::Tensor dc, at::Tensor uexp, at::Tensor ulane, at::Tensor slotof,
             at::Tensor tabs, int64_t ph, int64_t hc, int64_t timeout_ns, at::Tensor stats, int64_t grid, at::Tensor dbg)
{
    c10::cuda::CUDAGuard g(ctl.device());
    nv_copy_k<<<(int) grid, 256, 0, cur(ctl)>>>(ctl.data_ptr<int>(), jobs.data_ptr<int>(), jobx.data_ptr<int>(),
        jobnv.data_ptr<int>(), (int) li, (int) E, (const volatile long long*) home, (const volatile int*) ram_res,
        (const long long*) arena, (int) spc, rec, sg / 16, su / 16, sd / 16, off_u, off_d,
        (const long long*) ram_addr.data_ptr<int64_t>(), (const long long*) dc.data_ptr<int64_t>(), uexp.data_ptr<int>(),
        ulane.data_ptr<int>(), slotof.data_ptr<int>(), tabs.data_ptr<int64_t>(), ph, (volatile long long*) hc, timeout_ns,
        (unsigned long long*) stats.data_ptr<int64_t>(), dbg.numel() ? (long long*) dbg.data_ptr<int64_t>() : nullptr,
        (long long*) dc.data_ptr<int64_t>());
}

void nv_combine(at::Tensor out, at::Tensor dflag, int64_t done, int64_t hout, at::Tensor stats, int64_t hc, int64_t timeout_ns)
{
    c10::cuda::CUDAGuard g(out.device());
    TORCH_CHECK(out.is_contiguous() && (out.dtype() == at::kFloat || out.dtype() == at::kHalf));
    nv_combine_k<<<16, 256, 0, cur(out)>>>(out.data_ptr(), out.dtype() == at::kHalf, (int) out.numel(),
        (const long long*) dflag.data_ptr<int64_t>(), (const volatile long long*) done, (const float*) hout,
        (unsigned long long*) stats.data_ptr<int64_t>(), (volatile long long*) hc, timeout_ns);
}

void nv_restore(at::Tensor idx, int64_t li, int64_t E, int64_t home, at::Tensor tabs)
{
    c10::cuda::CUDAGuard g(idx.device());
    const int n = (int) idx.numel();
    if (!n) return;
    nv_restore_k<<<(n + 255) / 256, 256, 0, cur(idx)>>>(idx.data_ptr<int64_t>(), n, (int) li, (int) E,
        (const volatile long long*) home, tabs.data_ptr<int64_t>());
}

void nv_rows_all(int64_t L, int64_t E, at::Tensor slotof, int64_t home, at::Tensor tabs)
{
    c10::cuda::CUDAGuard g(slotof.device());
    const long long n = L * E;
    nv_rows_all_k<<<(int) ((n + 255) / 256), 256, 0, cur(slotof)>>>((int) L, (int) E, slotof.data_ptr<int>(),
        (const volatile long long*) home, tabs.data_ptr<int64_t>());
}

// device scratch peek on a private non-blocking stream (works while the compute stream is blocked in a wait)
std::vector<int64_t> peek(int64_t dptr, int64_t n)
{
    static cudaStream_t st = nullptr;
    if (!st) cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    std::vector<int64_t> v(n);
    cudaMemcpyAsync(v.data(), (const void*) dptr, n * 8, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    return v;
}

void dev_bind(pybind11::module& m)
{
    namespace py = pybind11;
    m.def("nv_pub", &nv_pub, py::call_guard<py::gil_scoped_release>());
    m.def("nv_step", &nv_step, py::call_guard<py::gil_scoped_release>());
    m.def("nv_copy", &nv_copy, py::call_guard<py::gil_scoped_release>());
    m.def("nv_combine", &nv_combine, py::call_guard<py::gil_scoped_release>());
    m.def("peek", &peek, py::call_guard<py::gil_scoped_release>());
    m.def("nv_restore", &nv_restore);
    m.def("nv_rows_all", &nv_rows_all);
}

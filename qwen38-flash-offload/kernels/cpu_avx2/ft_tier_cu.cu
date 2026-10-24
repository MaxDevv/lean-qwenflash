// FreeToken CPU tier: GPU side. Two kernels per MoE layer in decode, both on the compute stream, no host sync:
//  ft_split  (after the router, before the expert cache step): pick the coldest cache misses for the CPU
//            (greedy on a cost model), publish the job to pinned host memory (x rows, picks, then the seq flag),
//            and hand exllamav3 a selection with those picks removed (id -1, weight 0: the cache step skips
//            them, the fused MoE kernel treats weight 0 as inactive)
//  ft_combine (after the GPU experts): wait for the worker's done flag, add its fp32 partial into the MoE output
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// policy p[]: 0 t_zc (ms per zero-copy miss incl. its GPU compute), 1 t_hit (ms per GPU-resident expert),
// 2 cpu_a (ms fixed per CPU job incl. handoff), 3 cpu_b (ms per CPU expert), 4 cpu_tok (extra fraction per extra
// token of an expert), 5 max_cpu, 6 force_n (-1 = cost model), 7 handshake (1 = publish even empty jobs)
__global__ void ft_split_k(const int64_t* __restrict__ sel, const __half* __restrict__ w, const __half* __restrict__ z,
    int n, int bsz, int topk, int H, int E, int first, const int* __restrict__ slotof, const float* __restrict__ score,
    const float* __restrict__ p, volatile long long* ctrl, __half* hx, int* hpicks, long long seq, int li, int keep_ids,
    int64_t* sel_out, __half* w_out, long long* dflag, long long* stats)
{
    // E <= 1024. cnt[e] = tokens that picked local expert e; cpu[e] = 1 if the CPU takes e
    __shared__ int cnt[1024];
    __shared__ unsigned char cpu[1024];
    __shared__ int npk, publish;
    __shared__ int s_miss[1024];
    const int t = threadIdx.x;
    for (int e = t; e < E; e += blockDim.x) { cnt[e] = 0; cpu[e] = 0; }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s] - first;
        if (__half2float(w[s]) != 0.0f && e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (t == 0)
    {
        // misses sorted coldest first (static routing-frequency score); decode has <= 8 x bsz of them
        // miss list in SHARED memory: a 4 KB per-thread local array made the runtime grow the device-wide
        // local-memory reservation (~0.5 GB) and the next launch hit OOM once the elastic cache had re-taken all
        // free VRAM
        int nm = 0, nh = 0;
        int* miss = s_miss;
        for (int e = 0; e < E; ++e)
        {
            if (!cnt[e]) continue;
            if (slotof[e] >= 0) { ++nh; continue; }
            int b = nm++;
            while (b > 0 && score[miss[b - 1]] > score[e]) { miss[b] = miss[b - 1]; --b; }
            miss[b] = e;
        }
        int best = 0;
        const int force = (int) p[6], maxc = min((int) p[5], nm);
        if (force >= 0) best = min(force, nm);
        else
        {
            float bt = 1e30f, c = 0.0f;
            for (int k = 0; k <= maxc; ++k)
            {
                if (k) c += p[3] * (1.0f + p[4] * (cnt[miss[k - 1]] - 1));
                const float gpu = p[1] * (nh + nm - k) + p[0] * (nm - k);
                const float tt = fmaxf(gpu, k ? p[2] + c : 0.0f);
                if (tt < bt - 1e-6f) { bt = tt; best = k; }
            }
        }
        for (int k = 0; k < best; ++k) cpu[miss[k]] = 1;
        publish = best > 0 || p[7] > 0.5f;
        stats[0] += nh; stats[1] += nm; stats[2] += best; stats[3] += 1;
        npk = 0;
        if (publish)
            for (int s = 0; s < n; ++s)
            {
                const long long e = sel[s] - first;
                if (e < 0 || e >= E || !cpu[e] || __half2float(w[s]) == 0.0f) continue;
                hpicks[npk * 3] = s / topk; hpicks[npk * 3 + 1] = (int) e; hpicks[npk * 3 + 2] = __float_as_int(__half2float(w[s]));
                ++npk;
            }
    }
    __syncthreads();
    // CPU picks leave the GPU selection: id -1 + weight 0 in the fused decode path (skipped entirely, and the
    // cache step ignores them); keep_ids (bsz > MAX_BSZN paths, which index by expert id) keeps the id with
    // weight 0 -> exact zero contribution
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s];
        const bool c = e - first >= 0 && e - first < E && cpu[e - first];
        sel_out[s] = (c && !keep_ids) ? -1 : e;
        w_out[s] = c ? __float2half(0.0f) : w[s];
    }
    if (publish)
    {
        const int nx = bsz * H / 8;   // int4 = 8 halves
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
        __threadfence_system();
        __syncthreads();
        if (t == 0)
        {
            ctrl[2] = li; ctrl[3] = bsz; ctrl[4] = npk;
            __threadfence_system();
            ctrl[0] = seq;
            __threadfence_system();
            dflag[0] = seq;
        }
    }
    else if (t == 0) dflag[0] = 0;
}

__global__ void ft_combine_k(void* out, int is_half, int n, const long long* __restrict__ dflag, volatile long long* ctrl,
    const float* hout, long long* stats, long long timeout_ns)
{
    __shared__ int go;
    if (threadIdx.x == 0)
    {
        const long long s = dflag[0];
        go = s != 0;
        if (go)
        {
            unsigned long long t0, t1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            t1 = t0;
            while (ctrl[1] < s)
            {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
                if ((long long) (t1 - t0) > timeout_ns) { go = 0; if (blockIdx.x == 0) atomicAdd((unsigned long long*) &stats[6], 1ull); break; }
            }
            if (blockIdx.x == 0) { atomicAdd((unsigned long long*) &stats[4], (unsigned long long) (t1 - t0)); atomicAdd((unsigned long long*) &stats[5], 1ull); }
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

void ft_split(torch::Tensor sel, torch::Tensor w, torch::Tensor z, int64_t E, int64_t first, torch::Tensor slotof, torch::Tensor score,
              torch::Tensor pol, int64_t ctrl, int64_t hx, int64_t hpicks, int64_t seq, int64_t li, int64_t keep_ids,
              torch::Tensor sel_out, torch::Tensor w_out, torch::Tensor dflag, torch::Tensor stats)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && w.dtype() == at::kHalf && z.dtype() == at::kHalf && z.is_contiguous(), "dtypes");
    const int bsz = (int) z.size(0), H = (int) z.size(1), n = (int) sel.numel(), topk = n / bsz;
    TORCH_CHECK(E <= 1024 && H % 8 == 0, "ft_split: E <= 1024");
    ft_split_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), (const __half*) w.data_ptr(), (const __half*) z.data_ptr(), n, bsz, topk, H,
        (int) E, (int) first, slotof.data_ptr<int>(), score.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl,
        (__half*) hx, (int*) hpicks, seq, (int) li, (int) keep_ids, sel_out.data_ptr<int64_t>(), (__half*) w_out.data_ptr(),
        (long long*) dflag.data_ptr<int64_t>(), (long long*) stats.data_ptr<int64_t>());
}

void ft_combine(torch::Tensor out, torch::Tensor dflag, int64_t ctrl, int64_t hout, torch::Tensor stats, int64_t timeout_ns)
{
    c10::cuda::CUDAGuard g(out.device());
    auto st = at::cuda::getCurrentCUDAStream(out.device().index());
    TORCH_CHECK(out.is_contiguous() && (out.dtype() == at::kFloat || out.dtype() == at::kHalf), "out fp32/fp16 contiguous");
    const int n = (int) out.numel();
    ft_combine_k<<<16, 256, 0, st>>>(out.data_ptr(), out.dtype() == at::kHalf, n, (const long long*) dflag.data_ptr<int64_t>(),
        (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ft_split", &ft_split);
    m.def("ft_combine", &ft_combine);
}

"""Fused mHC hyper-connection sites for decode (Qwen3.8-Flash-Next, hc_mult 4), bit-exact by construction.

Stock exllamav3 1.5.1 per sublayer site (TransformerBlock.forward, decode R rows):
    hc_mix_partials -> hc_mix_finalize -> rms_norm (attn_norm / mlp_norm) -> [sublayer] -> hc_apply
= 4 launches per site, 90 sites per token (360 launches, ~1.1 ms/token in C050's profile).

Fused (R <= 8, D % 256 == 0):
    k_apply_partials : hc_apply of the PREVIOUS site (pending, deferred) + hc_mix_partials of this site, one launch
                       (a block owns 64 column quads of all 4 streams, so the in-place update has no cross-block race)
    k_finalize_norm  : hc_mix_finalize + the site's RMSNorm, one block per row (the norm needs the whole row)
= 2 launches per site. Every arithmetic step replicates the stock kernels' expressions, reduction trees and chunking
(64 partial chunks of 256 columns, the finalize's 8-warp partial re-reduce, rms_norm's reduce_dyn over the same block
size), compiled with the same nvcc flags (-O3 --use_fast_math), so outputs are bitwise identical (checked against
the stock kernels in the campaign).

The mlp site's hc_apply is deferred into the next block's attn-site launch; any other consumer of the stream stack
(HyperHead, non-fused paths, big-R forwards) flushes it first with the stock ext.hc_apply.

Env: QWEN38_K_HCFUSE=1 (serve.py hook: k_hcfuse.install(model) after model load), QWEN38_K_HCFUSE_MAXR (8).
"""
import os
import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define HC 4
#define MM 24
#define CLAMPF(x) fmaxf(-65504.0f, fminf(x, 65504.0f))

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ---- K_A: [hc_apply(prev site)] + hc_mix_partials(this site) ----------------------------------------------------
// grid (D / 128, R), 128 threads = 4 warps (stream h) x 32 lanes (one column quad each). The stock partials kernel's
// 64-thread chunk = two warps' shfl trees + (0 + red0) + red1: here each warp writes its tree result as a half-chunk
// partial (R, 2 * chunks, 25) and k_finalize_norm forms (0 + half0) + half1 -> the identical chunk value.
template <typename Y_T, bool APPLY>
__global__ __launch_bounds__(128) void k_apply_partials(
    float* __restrict__ x, const Y_T* __restrict__ y, const float* __restrict__ post, const float* __restrict__ comb,
    const half* __restrict__ fn, float* __restrict__ hpart, const int D)
{
    const int r = blockIdx.y;
    const int jb = blockIdx.x;
    const int g = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int D4 = D / 4;
    const int q = jb * 32 + lane;
    const int row_len4 = HC * D4;
    const int nhalf = HC * D / 128;
    float4* x4 = (float4*) (x + (size_t) r * HC * D);
    float4 s;
    if constexpr (APPLY)
    {
        float post_g = __ldg(post + (size_t) r * HC + g);
        float comb_c[HC];
        #pragma unroll
        for (int h = 0; h < HC; ++h) comb_c[h] = __ldg(comb + ((size_t) r * HC + h) * HC + g);   // comb_r[h][g]
        float4 xv[HC];
        #pragma unroll
        for (int h = 0; h < HC; ++h) xv[h] = x4[(size_t) h * D4 + q];
        float4 yv;
        if constexpr (std::is_same_v<Y_T, half>)
        {
            half2 y01 = ((const half2*) (y + (size_t) r * D))[q * 2];
            half2 y23 = ((const half2*) (y + (size_t) r * D))[q * 2 + 1];
            float2 lo = __half22float2(y01);
            float2 hi = __half22float2(y23);
            yv = make_float4(lo.x, lo.y, hi.x, hi.y);
        }
        else
            yv = ((const float4*) (y + (size_t) r * D))[q];
        float4 o;
        o.x = post_g * yv.x;
        o.y = post_g * yv.y;
        o.z = post_g * yv.z;
        o.w = post_g * yv.w;
        #pragma unroll
        for (int h = 0; h < HC; ++h)
        {
            o.x = fmaf(comb_c[h], xv[h].x, o.x);
            o.y = fmaf(comb_c[h], xv[h].y, o.y);
            o.z = fmaf(comb_c[h], xv[h].z, o.z);
            o.w = fmaf(comb_c[h], xv[h].w, o.w);
        }
        __syncthreads();                       // every warp has read all four streams of its quads
        x4[(size_t) g * D4 + q] = o;
        s = o;
    }
    else
        s = x4[(size_t) g * D4 + q];

    const int c = g * D4 + q;
    float acc[MM + 1];
    #pragma unroll
    for (int k = 0; k <= MM; ++k) acc[k] = 0.0f;
    acc[MM] = fmaf(s.x, s.x, acc[MM]);
    acc[MM] = fmaf(s.y, s.y, acc[MM]);
    acc[MM] = fmaf(s.z, s.z, acc[MM]);
    acc[MM] = fmaf(s.w, s.w, acc[MM]);
    #pragma unroll
    for (int j = 0; j < MM; ++j)
    {
        int2 pk = ((const int2*) fn)[(size_t) j * row_len4 + c];
        float2 lo = __half22float2(*(const half2*) &pk.x);
        float2 hi = __half22float2(*(const half2*) &pk.y);
        float4 w = make_float4(lo.x, lo.y, hi.x, hi.y);
        float d = fmaf(s.x, w.x, fmaf(s.y, w.y, fmaf(s.z, w.z, s.w * w.w)));
        acc[j] += d;
    }
    __shared__ float red[HC][MM + 1];
    #pragma unroll
    for (int k = 0; k <= MM; ++k)
    {
        float v = acc[k];
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) red[g][k] = v;
    }
    __syncwarp();
    if (lane <= MM)
        hpart[((size_t) r * nhalf + g * (D / 128) + jb) * (MM + 1) + lane] = red[g][lane];
}

// ---- K_B: hc_mix_finalize (post, comb, collapsed -> half) + RMSNorm(collapsed) -> half ------------------------
// grid (2R): blocks [0, R) collapse + norm row r; blocks [R, 2R) run row r's sinkhorn/post on warp 0 (so the norm's
// block-wide reduction never waits on the ~20-iteration sinkhorn chain). threads = rms_norm's block size for D.
template <typename W_T, int CHUNKS_A>
__global__ __launch_bounds__(1024) void k_finalize_norm(
    const float* __restrict__ streams, const float* __restrict__ partials,
    const float* __restrict__ base, const float* __restrict__ scale, float* __restrict__ post, float* __restrict__ comb,
    const W_T* __restrict__ nw, half* __restrict__ out, const int D, const float rms_eps, const float hc_eps,
    const int sinkhorn_iters, const float n_eps, const float cbias, const float cscale)
{
    const int R = gridDim.x / 2;
    const bool sink_block = blockIdx.x >= R;
    const int r = sink_block ? blockIdx.x - R : blockIdx.x;
    const int row_len = HC * D;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    if (sink_block && warp >= 8) return;
    __shared__ float mix_s[MM + 1];
    __shared__ float pre_s[HC];
    __shared__ float red_s[8][MM + 1];
    if (warp < 8 && lane <= MM)
    {
        const float* p = partials + (size_t) r * 2 * CHUNKS_A * (MM + 1) + lane;
        float pv[CHUNKS_A / 8];
        #pragma unroll
        for (int k = 0; k < CHUNKS_A / 8; ++k)
        {
            float h0 = p[(size_t) (2 * (warp + 8 * k)) * (MM + 1)], h1 = p[(size_t) (2 * (warp + 8 * k) + 1) * (MM + 1)];
            float cv = 0.0f;
            cv += h0;
            cv += h1;
            pv[k] = cv;
        }
        float v = 0.0f;
        #pragma unroll
        for (int k = 0; k < CHUNKS_A / 8; ++k) v += pv[k];
        red_s[warp][lane] = v;
    }
    if (sink_block) asm volatile("bar.sync 1, 256;"); else __syncthreads();
    if (tid <= MM)
    {
        float v = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) v += red_s[w][tid];
        mix_s[tid] = v;
    }
    if (sink_block) asm volatile("bar.sync 1, 256;"); else __syncthreads();
    float rmr = rsqrtf(mix_s[MM] / (float) row_len + rms_eps);

    if (sink_block)
    {
        if (warp != 0) return;
        if (tid < HC)
            post[(size_t) r * HC + tid] = 2.0f * sigmoidf_(fmaf(mix_s[HC + tid] * rmr, scale[1], base[HC + tid]));
        if (tid < HC * HC)
        {
            const unsigned mask = (HC * HC == 32) ? 0xffffffffu : ((1u << (HC * HC)) - 1u);
            float v = fmaf(mix_s[2 * HC + tid] * rmr, scale[2], base[2 * HC + tid]);
            float m = v;
            #pragma unroll
            for (int o = 1; o < HC; o <<= 1) m = fmaxf(m, __shfl_xor_sync(mask, m, o));
            v = __expf(v - m);
            float s = v;
            #pragma unroll
            for (int o = 1; o < HC; o <<= 1) s += __shfl_xor_sync(mask, s, o);
            v = __fdividef(v, s) + hc_eps;
            float cs = v;
            #pragma unroll
            for (int o = HC; o < HC * HC; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
            v = __fdividef(v, cs + hc_eps);
            for (int it = 0; it < sinkhorn_iters - 1; ++it)
            {
                float rs = v;
                #pragma unroll
                for (int o = 1; o < HC; o <<= 1) rs += __shfl_xor_sync(mask, rs, o);
                v = __fdividef(v, rs + hc_eps);
                cs = v;
                #pragma unroll
                for (int o = HC; o < HC * HC; o <<= 1) cs += __shfl_xor_sync(mask, cs, o);
                v = __fdividef(v, cs + hc_eps);
            }
            comb[(size_t) r * HC * HC + tid] = v;
        }
        return;
    }

    if (tid < HC)
        pre_s[tid] = sigmoidf_(fmaf(mix_s[tid] * rmr, scale[0], base[tid])) + hc_eps;
    __syncthreads();

    float pre_r[HC];
    #pragma unroll
    for (int h = 0; h < HC; ++h) pre_r[h] = pre_s[h];
    const int columns = D / 4;
    const float4* s4 = (const float4*) (streams + (size_t) r * row_len);
    float4 x4 = {};
    float sum = 0.0f;
    if (tid < columns)
    {
        float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int h = 0; h < HC; ++h)
        {
            float4 s = s4[(size_t) h * columns + tid];
            o.x = fmaf(pre_r[h], s.x, o.x);
            o.y = fmaf(pre_r[h], s.y, o.y);
            o.z = fmaf(pre_r[h], s.z, o.z);
            o.w = fmaf(pre_r[h], s.w, o.w);
        }
        half2 a = __floats2half2_rn(o.x, o.y);
        half2 b = __floats2half2_rn(o.z, o.w);
        x4.x = CLAMPF(__half2float(__low2half(a)));
        x4.y = CLAMPF(__half2float(__high2half(a)));
        x4.z = CLAMPF(__half2float(__low2half(b)));
        x4.w = CLAMPF(__half2float(__high2half(b)));
        sum = fma(x4.x, x4.x, sum);
        sum = fma(x4.y, x4.y, sum);
        sum = fma(x4.z, x4.z, sum);
        sum = fma(x4.w, x4.w, sum);
    }
    // rms_norm's reduce_dyn
    __shared__ float sums[32];
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    const int num_warps = blockDim.x / 32;
    if (num_warps > 1)
    {
        if (lane == 0) sums[warp] = sum;
        __syncthreads();
        sum = lane < num_warps ? sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    float rmf = rsqrtf(sum / (float) D + n_eps) * cscale;
    if (tid < columns)
    {
        float4 w4;
        if constexpr (std::is_same_v<W_T, __nv_bfloat16>)
        {
            const __nv_bfloat162* wp = (const __nv_bfloat162*) (nw + 4 * tid);
            __nv_bfloat162 w01 = wp[0], w23 = wp[1];
            w4.x = __bfloat162float(__low2bfloat16(w01));
            w4.y = __bfloat162float(__high2bfloat16(w01));
            w4.z = __bfloat162float(__low2bfloat16(w23));
            w4.w = __bfloat162float(__high2bfloat16(w23));
        }
        else
        {
            const half2* wp = (const half2*) (nw + 4 * tid);
            half2 w01 = wp[0], w23 = wp[1];
            w4.x = __half2float(__low2half(w01));
            w4.y = __half2float(__high2half(w01));
            w4.z = __half2float(__low2half(w23));
            w4.w = __half2float(__high2half(w23));
        }
        if (cbias != 0.0f)
        {
            w4.x += cbias; w4.y += cbias; w4.z += cbias; w4.w += cbias;
        }
        x4.x = x4.x * w4.x * rmf;
        x4.y = x4.y * w4.y * rmf;
        x4.z = x4.z * w4.z * rmf;
        x4.w = x4.w * w4.w * rmf;
        half2* o2 = (half2*) (out + (size_t) r * D + 4 * tid);
        o2[0] = __halves2half2(__float2half_rn(x4.x), __float2half_rn(x4.y));
        o2[1] = __halves2half2(__float2half_rn(x4.z), __float2half_rn(x4.w));
    }
}

void apply_partials(at::Tensor x, c10::optional<at::Tensor> y, c10::optional<at::Tensor> post, c10::optional<at::Tensor> comb,
                    at::Tensor fn, at::Tensor partials)
{
    c10::cuda::CUDAGuard g(x.device());
    auto st = at::cuda::getCurrentCUDAStream(x.device().index());
    int R = x.size(0), D = x.size(2);
    TORCH_CHECK(x.size(1) == HC && D % 256 == 0 && x.is_contiguous() && x.dtype() == at::kFloat, "x (R, 4, D%256) fp32");
    TORCH_CHECK(fn.dtype() == at::kHalf && fn.size(0) == MM && fn.size(1) == HC * D, "fn half (24, 4D)");
    TORCH_CHECK(partials.numel() >= (int64_t) R * (HC * D / 128) * (MM + 1), "partials");
    dim3 grid(D / 128, R);
    if (!y)
        k_apply_partials<float, false><<<grid, 128, 0, st>>>((float*) x.data_ptr(), nullptr, nullptr, nullptr,
            (const half*) fn.data_ptr(), (float*) partials.data_ptr(), D);
    else if (y->dtype() == at::kHalf)
        k_apply_partials<half, true><<<grid, 128, 0, st>>>((float*) x.data_ptr(), (const half*) y->data_ptr(),
            (const float*) post->data_ptr(), (const float*) comb->data_ptr(), (const half*) fn.data_ptr(),
            (float*) partials.data_ptr(), D);
    else
    {
        TORCH_CHECK(y->dtype() == at::kFloat, "y half/float");
        k_apply_partials<float, true><<<grid, 128, 0, st>>>((float*) x.data_ptr(), (const float*) y->data_ptr(),
            (const float*) post->data_ptr(), (const float*) comb->data_ptr(), (const half*) fn.data_ptr(),
            (float*) partials.data_ptr(), D);
    }
}

void finalize_norm(at::Tensor x, at::Tensor partials, at::Tensor base, at::Tensor scale, at::Tensor post, at::Tensor comb,
                   at::Tensor nw, at::Tensor out, double rms_eps, double hc_eps, int64_t iters, double n_eps, double cbias,
                   double cscale)
{
    c10::cuda::CUDAGuard g(x.device());
    auto st = at::cuda::getCurrentCUDAStream(x.device().index());
    int R = x.size(0), D = x.size(2);
    int threads = std::min(1024, ((D / 4 + 31) / 32) * 32);
    TORCH_CHECK(threads >= 256 && D / 4 <= 1024, "D range");
    TORCH_CHECK(HC * D / 256 == 64, "K101 kernels are built for 64 partial chunks (D = 4096)");
    #define ARGSB(WT) (const float*) x.data_ptr(), (const float*) partials.data_ptr(), \
        (const float*) base.data_ptr(), (const float*) scale.data_ptr(), (float*) post.data_ptr(), \
        (float*) comb.data_ptr(), (const WT*) nw.data_ptr(), (half*) out.data_ptr(), D, (float) rms_eps, \
        (float) hc_eps, (int) iters, (float) n_eps, (float) cbias, (float) cscale
    if (nw.dtype() == at::kBFloat16)
        k_finalize_norm<__nv_bfloat16, 64><<<2 * R, threads, 0, st>>>(ARGSB(__nv_bfloat16));
    else
    {
        TORCH_CHECK(nw.dtype() == at::kHalf, "norm weight bf16/half");
        k_finalize_norm<half, 64><<<2 * R, threads, 0, st>>>(ARGSB(half));
    }
    #undef ARGSB
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_partials", &apply_partials);
    m.def("finalize_norm", &finalize_norm);
}
"""

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = os.environ.get("QWEN38_K_BUILD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "k101"))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "k101.cu")
        if not os.path.exists(src) or open(src).read() != _SRC:
            open(src, "w").write(_SRC)
        _EXT = load(name="qwen38_k101", sources=[src], build_directory=d,
                    extra_cuda_cflags=["-lineinfo", "-O3", "--use_fast_math"], verbose=False)
    return _EXT


# ---- runtime -----------------------------------------------------------------------------------------------------
_PEND = None          # (x, y, post, comb): mlp-site hc_apply deferred into the next site's launch
_WS = {}              # (device, R, D) -> workspaces
STATS = {"fused_sites": 0, "flushes": 0, "fallback_blocks": 0, "checked_sites": 0, "check_mismatch": 0}
MAXR = int(os.environ.get("QWEN38_K_HCFUSE_MAXR", "8"))
CHECK = int(os.environ.get("QWEN38_K_HCFUSE_CHECK", "0"))   # shadow-check the first N fused sites vs the stock kernels


def _ws(dev, R, D):
    k = (str(dev), R, D)
    w = _WS.get(k)
    if w is None:
        f32 = dict(dtype=torch.float, device=dev)
        w = {"partials": torch.empty((R, 4 * D // 128, 25), **f32),
             "post": [torch.empty((R, 4), **f32) for _ in range(2)],
             "comb": [torch.empty((R, 4, 4), **f32) for _ in range(2)],
             "out": [torch.empty((R, D), dtype=torch.half, device=dev) for _ in range(3)], "i": 0}
        _WS[k] = w
    return w


def flush():
    global _PEND
    if _PEND is not None:
        from exllamav3.ext import exllamav3_ext as xe
        x, y, post, comb = _PEND
        _PEND = None
        R, H, D = x.shape[0] * x.shape[1], x.shape[2], x.shape[3]
        xe.hc_apply(x.view(R, H, D), y.reshape(R, D), post, comb)
        STATS["flushes"] += 1


def _site(hc, norm, x, pend):
    """One fused site: [apply pending] + mix + norm -> (post, comb, normed half (b, s, D))."""
    b, s, H, D = x.shape
    R = b * s
    w = _ws(x.device, R, D)
    i = w["i"]; w["i"] = i + 1
    post, comb, out = w["post"][i % 2], w["comb"][i % 2], w["out"][i % 3]
    if hc.fn_h is None:
        hc.fn_h = hc.fn.half()
    e = ext()
    x3 = x.view(R, H, D)
    chk = STATS["checked_sites"] < CHECK and not torch.cuda.is_current_stream_capturing()
    if chk:
        from exllamav3.ext import exllamav3_ext as xe
        xs = x3.clone()
        if pend is not None:
            xe.hc_apply(xs, pend[1].reshape(R, D), pend[2], pend[3])
        ch = xe.hc_mix_num_chunks(R, H * D)
        ps = torch.empty((R, ch, 25), dtype=torch.float, device=x.device)
        post_s = torch.empty((R, 4), dtype=torch.float, device=x.device)
        comb_s = torch.empty((R, 4, 4), dtype=torch.float, device=x.device)
        coll = torch.empty((R, D), dtype=torch.half, device=x.device)
        xe.hc_mix(xs, hc.fn_h, hc.base, hc.scale, hc.rms_eps, hc.hc_eps, hc.sinkhorn_iters, ps, post_s, comb_s, coll)
        ys = torch.empty_like(coll)
        xe.rms_norm(coll, norm.weight, ys, norm.rms_norm_eps, norm.constant_bias, norm.constant_scale, False, False, 1)
    if pend is not None:
        _, y, pp, pc = pend
        e.apply_partials(x3, y.reshape(R, D), pp, pc, hc.fn_h, w["partials"])
    else:
        e.apply_partials(x3, None, None, None, hc.fn_h, w["partials"])
    e.finalize_norm(x3, w["partials"], hc.base, hc.scale, post, comb, norm.weight, out, hc.rms_eps, hc.hc_eps,
                    hc.sinkhorn_iters, norm.rms_norm_eps, norm.constant_bias, norm.constant_scale)
    STATS["fused_sites"] += 1
    if chk:
        STATS["checked_sites"] += 1
        if not (torch.equal(xs, x3) and torch.equal(post_s, post) and torch.equal(comb_s, comb) and torch.equal(ys, out)):
            STATS["check_mismatch"] += 1
    return post, comb, out.view(b, s, D)


def _norm_ok(n):
    return (n is not None and type(n).__name__ == "RMSNorm" and n.groups == 1 and not n.span_heads and not n.unweighted
            and n.weight is not None and n.weight.dtype in (torch.bfloat16, torch.half) and n.weight.is_contiguous())


def _block_ok(blk):
    return (blk.attn is not None and blk.mlp is not None and blk.attn_hc is not None and blk.mlp_hc is not None
            and blk.attn_hc.hc_mult == 4 and blk.mlp_hc.hc_mult == 4 and _norm_ok(blk.attn_norm) and _norm_ok(blk.mlp_norm)
            and blk.attn_resid_scalar is None and blk.mlp_resid_scalar is None and blk.layer_scalar_f is None
            and blk.attn_hc.hidden_size == 4096)


def install(model):
    from exllamav3.modules.transformer import TransformerBlock, to2
    from exllamav3.modules.hyperconnections import HyperConnection
    _ = ext()
    blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
    ok = {id(m) for m in blocks if isinstance(m.attn_hc, HyperConnection) and _block_ok(m)}
    orig = TransformerBlock.forward

    def fwd(self, x, params, out_dtype=None):
        global _PEND
        if id(self) not in ok:
            flush()
            return orig(self, x, params, out_dtype)
        R = x.shape[0] * x.shape[1] if x.dim() == 4 else 0
        exp = params.get("export_state_layers")
        if (x.dim() != 4 or R > MAXR or x.dtype != torch.float or not x.is_contiguous() or exp
                or "quant_preserve" in params or "capture" in params):
            flush()
            STATS["fallback_blocks"] += 1
            return orig(self, x, params, out_dtype)
        pend = None
        if _PEND is not None:
            if _PEND[0] is x or _PEND[0].data_ptr() == x.data_ptr():
                pend, _PEND = _PEND, None
            else:
                flush()
        post, comb, y = _site(self.attn_hc, self.attn_norm, x, pend)
        y = self.attn.forward(y, params)
        if params.get("prefill"):
            return to2(x, out_dtype, self.out_dtype)
        post2, comb2, y2 = _site(self.mlp_hc, self.mlp_norm, x, (x, y, post, comb))
        y2 = self.mlp.forward(y2, params)
        if out_dtype not in (None, torch.float) or self.out_dtype not in (None, torch.float):
            _PEND = (x, y2, post2, comb2)
            flush()
            return to2(x, out_dtype, self.out_dtype)
        _PEND = (x, y2, post2, comb2)
        return x

    TransformerBlock.forward = fwd
    # any other module that consumes the stream stack flushes the deferred apply first
    for m in model.modules:
        if not isinstance(m, TransformerBlock):
            for name in ("forward", "prepare_for_device"):
                f0 = getattr(m, name)

                def wrapped(*a, _f=f0, **k):
                    flush()
                    return _f(*a, **k)
                setattr(m, name, wrapped)
    print(f" -- k_hcfuse (K101): {len(ok)}/{len(blocks)} blocks fused for R <= {MAXR}", flush=True)
    return len(ok)

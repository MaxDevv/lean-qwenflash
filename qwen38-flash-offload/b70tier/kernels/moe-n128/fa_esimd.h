#pragma once
// EXPERIMENTAL, not wired in: correct (rel ~1e-5 vs XPU FA2) but 38-48 TFLOPS vs FA2 61-75 TFLOPS.
// Flash-attention forward (prefill) for Intel Xe2 in ESIMD: fp16 Q/K/V, head_dim 256, GQA, optional causal
// (bottom-right aligned, as flash-attn varlen). Returns O and the natural-log LSE so partial results over
// key ranges can be merged (exl3xpu/fp8kv_prefill.py).
//
// Thread = 8 query rows x 1 query head, 256-register mode.
//   Qc  [16 d-chunks][8 rows][16]  fp16  (DPAS A operands are free register slices)
//   O   [16 d-chunks][8 rows][16]  fp32  accumulator
// Per 32-key block:
//   S = Q K^T : 2 key tiles x 16 d-chunks DPAS, B = K^T chunk from a transposed 2D dword load (= VNNI)
//   online softmax in base 2 (scale * log2 e folded), rescale O once per block
//   O += P V  : 16 d-chunks x 2 key tiles DPAS, B = V chunk from a VNNI-transformed 2D load
// Work-group = the GQA group (query heads sharing one KV head) for one query tile: K/V hits in L1.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace exl3fa {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

constexpr int D = 256;
constexpr int QR = 8;           // query rows per thread (DPAS repeat count)
constexpr int KT = 16;          // keys per DPAS tile
constexpr int KBT = 2;          // key tiles per softmax block
constexpr int KB = KT * KBT;    // keys per softmax block
constexpr int NC = D / 16;      // d-chunks
constexpr float LOG2E = 1.4426950408889634f;

struct FaKernel {
    const fp16* q; const fp16* k; const fp16* v;   // q [Lq, Hq, D], k/v [Lk, Hk, D]
    fp16* o; float* lse;                           // o [Lq, Hq, D], lse [Lq, Hq]
    int Lq, Lk, Hq, Hk, G;                         // G = Hq / Hk
    float scale_log2;                              // softmax scale * log2(e)
    int causal;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int g = id % G;
        int rest = id / G;
        int kvh = rest % Hk;
        int qt = rest / Hk;
        int q0 = qt * QR;
        if (q0 >= Lq) return;
        int h = kvh * G + g;

        // ---- Q tile, chunk-major, pre-scaled rows beyond Lq are zero
        simd<fp16, NC * QR * 16> Qc = 0;
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            if (q0 + r < Lq) {
                simd<fp16, D> row = block_load<fp16, D>(q + ((size_t)(q0 + r) * Hq + h) * D);
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    Qc.template select<16, 1>((c * QR + r) * 16) = row.template select<16, 1>(c * 16);
            }
        }

        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;

        // causal: query row r sits at absolute key position off + q0 + r
        int off = Lk - Lq;
        int kend = causal ? ((off + q0 + QR) < Lk ? (off + q0 + QR) : Lk) : Lk;

        // 2D surfaces for this KV head: rows = keys, row pitch = Hk * D halfs
        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(k + (size_t)kvh * D);
        const fp16* vbase = v + (size_t)kvh * D;
        unsigned surfW = D * 2 - 1;                       // bytes - 1
        unsigned surfH = (unsigned)Lk - 1;                // rows - 1
        unsigned pitch = (unsigned)Hk * D * 2 - 1;        // bytes - 1

        for (int kb0 = 0; kb0 < kend; kb0 += KB) {
            simd<float, KBT * QR * KT> S;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> acc = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    // K^T chunk: 8 dword columns (16 d) x 16 keys, transposed -> [8 dpair][16 key] = VNNI B
                    simd<uint32_t, 8 * KT> kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        kb32, surfW, surfH, pitch, c * 8, kb0 + t * KT);
                    acc = xmx::dpas<8, 8, float, float, fp16, fp16>(
                        acc, kt.template bit_cast_view<fp16>().read(),
                        simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>(c * QR * 16)));
                }
                S.template select<QR * KT, 1>(t * QR * KT) = acc * scale_log2;
            }
            // masking: keys >= Lk, and causal keys beyond each row's position
            bool need_mask = (kb0 + KB > Lk) || (causal && kb0 + KB > off + q0);
            if (need_mask) {
                simd<int, KT> lane(0, 1);
#pragma unroll
                for (int t = 0; t < KBT; ++t)
#pragma unroll
                    for (int r = 0; r < QR; ++r) {
                        simd<int, KT> kj = lane + (kb0 + t * KT);
                        int lim = causal ? ((off + q0 + r) < (Lk - 1) ? (off + q0 + r) : (Lk - 1)) : Lk - 1;
                        simd<float, KT> sv = S.template select<KT, 1>((t * QR + r) * KT);
                        sv.merge(simd<float, KT>(-INFINITY), kj > lim);
                        S.template select<KT, 1>((t * QR + r) * KT) = sv;
                    }
            }
            // online softmax (base 2)
            simd<float, QR> mb;
#pragma unroll
            for (int r = 0; r < QR; ++r) {
                simd<float, KT> mx = S.template select<KT, 1>(r * KT);
#pragma unroll
                for (int t = 1; t < KBT; ++t) mx = max(mx, simd<float, KT>(S.template select<KT, 1>((t * QR + r) * KT)));
                mb[r] = hmax<float>(mx);
            }
            simd<float, QR> m_new = max(m, mb);
            simd<float, QR> alpha = exp2(m - m_new);
            simd<float, QR * KT> mrep = m_new.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
            simd<float, QR> rs = 0.0f;
            simd<fp16, KBT * QR * KT> P;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> p = exp2(simd<float, QR * KT>(S.template select<QR * KT, 1>(t * QR * KT)) - mrep);
#pragma unroll
                for (int r = 0; r < QR; ++r) rs[r] += reduce<float>(simd<float, KT>(p.template select<KT, 1>(r * KT)), std::plus<>());
                P.template select<QR * KT, 1>(t * QR * KT) = convert<fp16>(p);
            }
            l = l * alpha + rs;
            m = m_new;
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    // V chunk: 16 keys x 16 d, VNNI-transformed -> [8 keypair][16 d][2]
                    simd<fp16, KT * 16> vt = xesimd::lsc_load_2d<fp16, 16, KT, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        vbase, surfW, surfH, pitch, c * 16, kb0 + t * KT);
                    acc = xmx::dpas<8, 8, float, float, fp16, fp16>(
                        acc, vt, simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                }
                oc = acc;
            }
        }

        // ---- normalise + store
        simd<float, QR> inv = 1.0f / l;
        simd<float, QR> lse_r = (m + log2(l)) * (1.0f / LOG2E);
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            if (q0 + r >= Lq) break;
            simd<fp16, D> row;
            float ir = inv[r];
#pragma unroll
            for (int c = 0; c < NC; ++c)
                row.template select<16, 1>(c * 16) = convert<fp16>(simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) * ir);
            block_store<fp16, D>(o + ((size_t)(q0 + r) * Hq + h) * D, row);
            lse[(size_t)(q0 + r) * Hq + h] = lse_r[r];
        }
    }
};

}  // namespace exl3fa

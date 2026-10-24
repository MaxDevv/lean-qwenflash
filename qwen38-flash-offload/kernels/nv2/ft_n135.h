// N135: Qwen3.8-Flash-Next CPU-tier forward v4 (AVX2, EXL3 mul1 K=3 trellis, NATIVE RAM-slot layout, ft_core.h codebook).
//
// What changes against ft_core.h moe_forward (kept as the default; this is opt-in):
//  1. one pool.run per layer call: a dataflow unit queue (prep -> gate/up -> act -> down -> out) with per-job
//     completion counters instead of 4 global barriers, so one job's tail overlaps the next stage of earlier jobs;
//  2. GEMV units are rectangles (R tile rows x C tiles) of ONE matrix: R = K gives ft_core's block unit (no partial
//     sums), R < K gives row bands whose memory is long contiguous runs of the native layout (C * 96 B per row);
//     band partials are lane-reduced per unit and summed in band order by the consumer;
//  3. "p16" I16 kernel: two trellis groups of the same output column are extracted into one register as 16-bit
//     states (srlv / sllv / blendw instead of srlv + and twice), hashed with 16-bit multiplies, and multiplied
//     against int16 activation pairs with ONE vpmaddwd per 16 weights. Integer accumulation per lane is identical to
//     ft_core's I16 kernel (same row sets per lane, exact int32), so with R = K the result is bit-identical;
//  4. vectorised 4096-wide input prep (same butterfly order and fp16 rounding points as prep_block).
// Numerics: mode semantics are ft_core's (0 AFFINE, 1 EXACT, 2 I16, -1 auto); band partials re-associate fp32 sums.
#pragma once
#include "ft_core.h"
#include <map>
#include <functional>

namespace {
namespace n135 {

// ------------------------------------------------------------------------------------------
// knobs (bench: key=value; engine: env QWEN38_NV_CPU_*)
// ------------------------------------------------------------------------------------------
struct Knobs
{
    // defaults (batch B, 2026-10-09): p16 kernel; full-row bands only when every job of the call has 1 token (m = 1),
    // ft_core-shaped block units otherwise (bands keep C * 64 * m accumulators live and lose at m >= 2)
    int kern = 1;        // I16 kernel: 0 = ft_core math (int32 dup activations; rt2 when rt = 1), 1 = p16
    int rg = 32;         // gate/up unit rows (tile rows); 0 = all (ft_core's block unit)
    int cg = 128;        // gate/up unit tiles (16 outputs each); 128 = full row (12 KB contiguous)
    int rd = 16;         // down unit rows; 0 = all
    int cd = 128;        // down unit tiles (half a 24 KB row)
    int pfr = 6;         // software prefetch distance in tile rows (0 = off)
    int pfhint = 0;      // 0 = T0, 1 = T1 (L2), 2 = NTA
    int pfpro = 1;       // prefetch the first pfr rows of a unit at unit start
    int i16max = 2;      // auto mode: I16 when every expert has <= i16max tokens, else AFFINE (ft_core rule: 2)
    int merge = 1;       // engine: 1 = RAM and NVMe->CPU picks of a job in ONE forward (late-bound NVC jobs)
    int bandmaxm = 1;    // use the rg/cg/rd/cd band shapes only when the call's max tokens per job <= bandmaxm
    int rt = 0;          // 1: I16 units keep a tile's accumulators in registers over each 8-row scale block (rt2; p16: m = 1 only)
};
inline Knobs K_;

inline void configure(const std::map<std::string, std::string>& A)
{
    auto g = [&](const char* k, int& v) { auto it = A.find(k); if (it != A.end()) v = std::stoi(it->second); };
    g("kern", K_.kern); g("rg", K_.rg); g("cg", K_.cg); g("rd", K_.rd); g("cd", K_.cd); g("pfr", K_.pfr); g("pfhint", K_.pfhint); g("pfpro", K_.pfpro); g("rt", K_.rt); g("bandmaxm", K_.bandmaxm); g("i16max", K_.i16max); g("merge", K_.merge);
}
inline void configure_env()
{
    auto g = [](const char* k, int& v) { const char* s = getenv(k); if (s && *s) v = atoi(s); };
    g("QWEN38_NV_CPU_KERN_P16", K_.kern); g("QWEN38_NV_CPU_RG", K_.rg); g("QWEN38_NV_CPU_CG", K_.cg); g("QWEN38_NV_CPU_RD", K_.rd);
    g("QWEN38_NV_CPU_CD", K_.cd); g("QWEN38_NV_CPU_PFR", K_.pfr); g("QWEN38_NV_CPU_PFHINT", K_.pfhint); g("QWEN38_NV_CPU_PFPRO", K_.pfpro); g("QWEN38_NV_CPU_RT", K_.rt); g("QWEN38_NV_CPU_BANDMAXM", K_.bandmaxm); g("QWEN38_NV_CPU_MERGE", K_.merge); g("QWEN38_NV_CPU_I16MAX", K_.i16max);
}

// ------------------------------------------------------------------------------------------
// p16 I16 kernel
// ------------------------------------------------------------------------------------------
alignas(32) inline int32_t SHL[8];
// pshufb controls like ft_core CTRL, but with the 16-byte window start clamped to byte 80 of the 96-byte tile: ft_core's
// windows for groups 29-31 start at 84/88/88 and read 4-8 bytes past the tile, which faults when the tile is the last one
// of a mapping (e.g. the down matrix of the last RAM slot). Same bytes selected, so the decode is identical.
alignas(32) inline uint8_t CTRLP[32][32];
constexpr int win_lo(int g) { return g == 0 ? -1 : (4 * ((3 * g - 2) / 4) < 80 ? 4 * ((3 * g - 2) / 4) : 80); }
alignas(32) inline uint8_t CTRLS[32][32];   // shared windows: group g decoded from the window of group (g & ~1)
inline void init_p16()
{
    for (int j = 0; j < 8; ++j) SHL[j] = (3 + 3 * j) % 8;
    for (int pass = 0; pass < 2; ++pass)
    for (int g = 0; g < 32; ++g)
    {
        const int q0 = 3 * g - 2;
        const int gw = pass ? (g & ~1) : g;
        int win[16];
        if (gw == 0) { for (int b = 0; b < 4; ++b) win[b] = 92 + b; for (int b = 4; b < 16; ++b) win[b] = b - 4; }
        else { const int L = win_lo(gw); for (int b = 0; b < 16; ++b) win[b] = L + b; }
        auto wpos = [&](int q) -> int {
            if (q >= 96) return -1;
            const int m = mem_of_stream(q);
            for (int b = 0; b < 16; ++b) if (win[b] == m) return b;
            return -2;
        };
        for (int j = 0; j < 8; ++j)
        {
            const int fb = (3 + 3 * j) / 8;
            const int src[4] = { -1, wpos(q0 + fb + 2), wpos(q0 + fb + 1), wpos(q0 + fb) };
            for (int b = 0; b < 4; ++b)
            {
                if (src[b] == -2) { fprintf(stderr, "n135 table error g=%d j=%d\n", g, j); exit(1); }
                (pass ? CTRLS : CTRLP)[g][(j < 4 ? 0 : 16) + (j % 4) * 4 + b] = src[b] < 0 ? 0x80 : uint8_t(src[b]);
            }
        }
    }
}

struct KP
{
    __m256i shr, shl, c1, c2, one8;
};
inline KP kp()
{
    return { _mm256_load_si256(reinterpret_cast<const __m256i*>(SHV)), _mm256_load_si256(reinterpret_cast<const __m256i*>(SHL)),
             _mm256_set1_epi16(short(MUL1 & 0xffff)), _mm256_set1_epi16(short(MUL1 >> 16)), _mm256_set1_epi8(1) };
}

template <int g>
__attribute__((always_inline)) inline __m256i shuf(const uint8_t* tile)
{
    __m256i win;
    if constexpr (g == 0)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + 80));
        win = _mm256_broadcastsi128_si256(_mm_alignr_epi8(a, b, 12));
    }
    else
    {
        constexpr int L = win_lo(g);
        win = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + L)));
    }
    return _mm256_shuffle_epi8(win, _mm256_load_si256(reinterpret_cast<const __m256i*>(CTRLP[g])));
}

// bytesum(state * MUL1) for the 8 weights of group g0 (low 16-bit halves) and of g0 + 1 (high halves), as int16
template <int g0>
__attribute__((always_inline)) inline __m256i pair_sums(const uint8_t* tile, const KP& K)
{
    const __m256i a = _mm256_srlv_epi32(shuf<g0>(tile), K.shr);          // state(g0) in bits 0..15, garbage above
    const __m256i b = _mm256_sllv_epi32(shuf<g0 + 1>(tile), K.shl);      // state(g0+1) in bits 16..31, garbage below
    const __m256i st = _mm256_blend_epi16(a, b, 0xAA);
    const __m256i lo = _mm256_mullo_epi16(st, K.c1);                                           // bits 0..15 of st * MUL1
    const __m256i hi = _mm256_add_epi16(_mm256_mulhi_epu16(st, K.c1), _mm256_mullo_epi16(st, K.c2));   // bits 16..31
    return _mm256_add_epi16(_mm256_maddubs_epi16(lo, K.one8), _mm256_maddubs_epi16(hi, K.one8));
}

// acc layout per tile: [c 8][m][8 lanes] int32 (same as ft_core); X per token per tile row: [pair 2][16 int16]
template <int M, int C = 0>
__attribute__((always_inline)) inline void tile_accum_p16(const uint8_t* tile, const int16_t* const* X, int32_t* acc, const KP& K)
{
    if constexpr (C < 8)
    {
        const __m256i s0 = pair_sums<4 * C + 0>(tile, K);
        const __m256i s1 = pair_sums<4 * C + 2>(tile, K);
        #pragma GCC unroll 8
        for (int i = 0; i < M; ++i)
        {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + (C * M + i) * 8));
            a = _mm256_add_epi32(a, _mm256_madd_epi16(s0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i]))));
            a = _mm256_add_epi32(a, _mm256_madd_epi16(s1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i] + 16))));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + (C * M + i) * 8), a);
        }
        tile_accum_p16<M, C + 1>(tile, X, acc, K);
    }
}

// quantise one transformed 128-block for p16: int16 pairs [tt 8][pair 2][16]; same scale / rounding as quant_block_i16
inline float quant_block_p16(const float* xt, int16_t* xq_blk, float* sum)
{
    float amax = 0; for (int r = 0; r < 128; ++r) amax = std::max(amax, std::fabs(xt[r]));
    const float sc = amax > 0 ? amax / 16383.0f : 1.0f, rs = 1.0f / sc;
    int xi[128]; long si = 0;
    for (int r = 0; r < 128; ++r) { xi[r] = int(std::nearbyint(xt[r] * rs)); si += xi[r]; }
    for (int tt = 0; tt < 8; ++tt)
        for (int p = 0; p < 2; ++p)
            for (int j = 0; j < 8; ++j)
            {
                // 32-bit lane j of pair p: (row of group 2p, lane j), (row of group 2p + 1, lane j)
                static constexpr int dr[8] = { 0, 1, 8, 9, 0, 1, 8, 9 };
                xq_blk[tt * 32 + p * 16 + 2 * j + 0] = int16_t(xi[tt * 16 + 2 * (2 * p) + dr[j]]);
                xq_blk[tt * 32 + p * 16 + 2 * j + 1] = int16_t(xi[tt * 16 + 2 * (2 * p + 1) + dr[j]]);
            }
    *sum += float(si) * sc;
    return sc;
}

// ------------------------------------------------------------------------------------------
// vectorised prep (same operations as ft_core prep_block: x * suh, had128 butterflies in stage order, * HAD, r16)
// ------------------------------------------------------------------------------------------
__attribute__((always_inline)) inline void had128_v(__m256* r)
{
    for (int k = 0; k < 16; ++k)
    {
        __m256 v = r[k];
        __m256 s = _mm256_permute_ps(v, 0xB1);
        v = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xAA);
        s = _mm256_permute_ps(v, 0x4E);
        v = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xCC);
        s = _mm256_permute2f128_ps(v, v, 1);
        r[k] = _mm256_blend_ps(_mm256_add_ps(v, s), _mm256_sub_ps(s, v), 0xF0);
    }
    for (int w = 1; w < 16; w *= 2)
        for (int b = 0; b < 16; b += 2 * w)
            for (int i = 0; i < w; ++i)
            {
                const __m256 a = r[b + i], c = r[b + w + i];
                r[b + i] = _mm256_add_ps(a, c); r[b + w + i] = _mm256_sub_ps(a, c);
            }
}
inline void prep_block_v(const float* x, const uint16_t* suh, float* xt)
{
    __m256 r[16];
    const __m256 had = _mm256_set1_ps(HAD);
    for (int k = 0; k < 16; ++k)
        r[k] = _mm256_mul_ps(_mm256_loadu_ps(x + 8 * k), _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(suh + 8 * k))));
    had128_v(r);
    for (int k = 0; k < 16; ++k)
        _mm256_storeu_ps(xt + 8 * k, _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(r[k], had), _MM_FROUND_TO_NEAREST_INT)));
}

// ------------------------------------------------------------------------------------------
// rectangle unit: tile rows [r0, r0 + R) x tiles [t0, t0 + C) of `mat` (native layout, or swizzled with C = 8 at
// t0 % 8 == 0), M tokens. Writes the lane-reduced raw dot products y[i * ldy + (t * 16 + c)], c in 0..15.
// KIND: 0 AFFINE / 1 EXACT (fp32 permuted X, ft_core tile_accum), 2 I16 dup (ft_core tile_accum_i16), 3 I16 p16.
// I16 kinds flush the int32 accumulators with the 128-row block scale every 8 rows (R, r0 multiples of 8).
// ------------------------------------------------------------------------------------------
constexpr int MAXCT = 256;

template <int M, int KIND>
void rect(const Mat& mat, int r0, int R, int t0, int C, const void* const* xin, const float* const* qs, float* y, size_t ldy)
{
    // per-thread accumulators on the heap (static TLS of this size would come out of each pool thread's stack)
    static thread_local float* accf = nullptr;
    static thread_local int32_t* acci = nullptr;
    if (!accf)
    {
        accf = static_cast<float*>(std::aligned_alloc(64, size_t(MAXCT) * 64 * 8 * 4));
        acci = static_cast<int32_t*>(std::aligned_alloc(64, size_t(MAXCT) * 64 * 8 * 4));
    }
    const size_t nacc = size_t(C) * 64 * M;
    std::memset(accf, 0, nacc * 4);
    if constexpr (KIND >= 2) std::memset(acci, 0, nacc * 4);
    const Consts K = consts();
    const KP P = kp();
    const int tiles_n = mat.n / 16;
    const size_t row_stride = mat.swz ? size_t(8) * TILE_BYTES : size_t(tiles_n) * TILE_BYTES;
    const uint8_t* base = mat.swz ? mat.tr + size_t(t0 / 8) * (mat.k / 16) * 8 * TILE_BYTES : mat.tr + size_t(t0) * TILE_BYTES;
    const int lines = (C * TILE_BYTES + 63) / 64;
    const int pfr = K_.pfr;
    auto pf = [&](const uint8_t* q) {
        const char* c = reinterpret_cast<const char*>(q);
        if (K_.pfhint == 0) for (int l = 0; l < lines; ++l) _mm_prefetch(c + l * 64, _MM_HINT_T0);
        else if (K_.pfhint == 1) for (int l = 0; l < lines; ++l) _mm_prefetch(c + l * 64, _MM_HINT_T1);
        else for (int l = 0; l < lines; ++l) _mm_prefetch(c + l * 64, _MM_HINT_NTA);
    };
    if (pfr > 0 && K_.pfpro) for (int r = r0; r < std::min(r0 + pfr, r0 + R); ++r) pf(base + size_t(r) * row_stride);
    for (int r = r0; r < r0 + R; ++r)
    {
        const uint8_t* p = base + size_t(r) * row_stride;
        if (pfr > 0 && r + pfr < mat.k / 16) pf(p + size_t(pfr) * row_stride);
        if constexpr (KIND <= 1)
        {
            const float* X[M];
            for (int i = 0; i < M; ++i) X[i] = static_cast<const float*>(xin[i]) + size_t(r) * 32;
            #pragma GCC unroll 1
            for (int t = 0; t < C; ++t) tile_accum<M, KIND == 1>(p + t * TILE_BYTES, X, accf + size_t(t) * 64 * M, K);
        }
        else
        {
            if constexpr (KIND == 2)
            {
                const int32_t* X[M];
                for (int i = 0; i < M; ++i) X[i] = static_cast<const int32_t*>(xin[i]) + size_t(r) * 32;
                #pragma GCC unroll 1
                for (int t = 0; t < C; ++t) tile_accum_i16<M>(p + t * TILE_BYTES, X, acci + size_t(t) * 64 * M, K);
            }
            else
            {
                const int16_t* X[M];
                for (int i = 0; i < M; ++i) X[i] = static_cast<const int16_t*>(xin[i]) + size_t(r) * 32;
                #pragma GCC unroll 1
                for (int t = 0; t < C; ++t) tile_accum_p16<M>(p + t * TILE_BYTES, X, acci + size_t(t) * 64 * M, P);
            }
            if ((r & 7) == 7)
            {
                for (int i = 0; i < M; ++i)
                {
                    const __m256 sc = _mm256_set1_ps(qs[i][r >> 3]);
                    for (int t = 0; t < C; ++t)
                        for (int c = 0; c < 8; ++c)
                        {
                            int32_t* ai = acci + size_t(t) * 64 * M + (c * M + i) * 8; float* af = accf + size_t(t) * 64 * M + (c * M + i) * 8;
                            _mm256_store_ps(af, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_load_si256(reinterpret_cast<const __m256i*>(ai))), sc, _mm256_load_ps(af)));
                            _mm256_store_si256(reinterpret_cast<__m256i*>(ai), _mm256_setzero_si256());
                        }
                }
            }
        }
    }
    for (int i = 0; i < M; ++i)
        for (int t = 0; t < C; ++t)
            for (int c = 0; c < 8; ++c)
            {
                const float* v = accf + size_t(t) * 64 * M + (c * M + i) * 8;
                y[i * ldy + t * 16 + c] = (v[0] + v[1]) + (v[2] + v[3]);
                y[i * ldy + t * 16 + c + 8] = (v[4] + v[5]) + (v[6] + v[7]);
            }
}

template <int gw>
__attribute__((always_inline)) inline __m256i window(const uint8_t* tile)
{
    if constexpr (gw == 0)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + 80));
        return _mm256_broadcastsi128_si256(_mm_alignr_epi8(a, b, 12));
    }
    else return _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tile + win_lo(gw))));
}

// ft_core I16 product-byte pair sums (decode8p math) of group g from an already loaded shared window
template <int g>
__attribute__((always_inline)) inline __m256i dec8p_w(__m256i win, const Consts& K)
{
    const __m256i st = _mm256_and_si256(_mm256_srlv_epi32(
        _mm256_shuffle_epi8(win, _mm256_load_si256(reinterpret_cast<const __m256i*>(CTRLS[g]))), K.sh), K.m16);
    return _mm256_maddubs_epi16(_mm256_mullo_epi32(st, K.mul), K.one8);
}

// rt2 tile (m = 1): ft_core I16 math (int32-dup activations), 8 accumulators in registers, one window per group pair
template <int C = 0>
__attribute__((always_inline)) inline void tile_i16_reg(const uint8_t* tile, const int32_t* X, __m256i* a, const Consts& K)
{
    if constexpr (C < 8)
    {
        const __m256i w0 = window<4 * C>(tile), w1 = window<4 * C + 2>(tile);
        const __m256i p0 = dec8p_w<4 * C + 0>(w0, K), p1 = dec8p_w<4 * C + 1>(w0, K);
        const __m256i p2 = dec8p_w<4 * C + 2>(w1, K), p3 = dec8p_w<4 * C + 3>(w1, K);
        // same summation as tile_accum_i16: acc += madd(p_q, X_q) for q = 0..3 in order
        a[C] = _mm256_add_epi32(a[C], _mm256_madd_epi16(p0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + 0))));
        a[C] = _mm256_add_epi32(a[C], _mm256_madd_epi16(p1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + 8))));
        a[C] = _mm256_add_epi32(a[C], _mm256_madd_epi16(p2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + 16))));
        a[C] = _mm256_add_epi32(a[C], _mm256_madd_epi16(p3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + 24))));
        tile_i16_reg<C + 1>(tile, X, a, K);
    }
}

// p16 pair sums of groups g0 (even) and g0 + 1 from their shared window
template <int g0>
__attribute__((always_inline)) inline __m256i pair_sums_w(__m256i win, const KP& K)
{
    const __m256i a = _mm256_srlv_epi32(_mm256_shuffle_epi8(win, _mm256_load_si256(reinterpret_cast<const __m256i*>(CTRLS[g0]))), K.shr);
    const __m256i b = _mm256_sllv_epi32(_mm256_shuffle_epi8(win, _mm256_load_si256(reinterpret_cast<const __m256i*>(CTRLS[g0 + 1]))), K.shl);
    const __m256i st = _mm256_blend_epi16(a, b, 0xAA);
    const __m256i lo = _mm256_mullo_epi16(st, K.c1);
    const __m256i hi = _mm256_add_epi16(_mm256_mulhi_epu16(st, K.c1), _mm256_mullo_epi16(st, K.c2));
    return _mm256_add_epi16(_mm256_maddubs_epi16(lo, K.one8), _mm256_maddubs_epi16(hi, K.one8));
}

// register-accumulator p16 unit (m = 1): per 8-row group (one I16 scale block), tile-major: the 8 accumulators of a tile
// stay in registers across the 8 rows; the next group is prefetched into L2 alongside. Same integer sums per lane and
// same per-block fp32 flush order as rect<1, 3>, so the result is bit-identical to it for the same unit shape.
template <int C = 0>
__attribute__((always_inline)) inline void tile_p16_reg(const uint8_t* tile, const int16_t* X, __m256i* a, const KP& K)
{
    if constexpr (C < 8)
    {
        const __m256i s0 = pair_sums_w<4 * C + 0>(window<4 * C + 0>(tile), K);
        const __m256i s1 = pair_sums_w<4 * C + 2>(window<4 * C + 2>(tile), K);
        a[C] = _mm256_add_epi32(a[C], _mm256_madd_epi16(s0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X))));
        a[C] = _mm256_add_epi32(a[C], _mm256_madd_epi16(s1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X + 16))));
        tile_p16_reg<C + 1>(tile, X, a, K);
    }
}

// rt2 for M tokens: C-chunk [C0, C0 + NC) of a tile with NC * M register accumulators a[(C - C0) * M + i]
template <int M, int C0, int NC, int C = C0>
__attribute__((always_inline)) inline void tile_i16_reg_m(const uint8_t* tile, const int32_t* const* X, __m256i* a, const Consts& K)
{
    if constexpr (C < C0 + NC)
    {
        const __m256i w0 = window<4 * C>(tile), w1 = window<4 * C + 2>(tile);
        const __m256i p0 = dec8p_w<4 * C + 0>(w0, K), p1 = dec8p_w<4 * C + 1>(w0, K);
        const __m256i p2 = dec8p_w<4 * C + 2>(w1, K), p3 = dec8p_w<4 * C + 3>(w1, K);
        #pragma GCC unroll 8
        for (int i = 0; i < M; ++i)
        {
            __m256i& r = a[(C - C0) * M + i];
            r = _mm256_add_epi32(r, _mm256_madd_epi16(p0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i] + 0))));
            r = _mm256_add_epi32(r, _mm256_madd_epi16(p1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i] + 8))));
            r = _mm256_add_epi32(r, _mm256_madd_epi16(p2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i] + 16))));
            r = _mm256_add_epi32(r, _mm256_madd_epi16(p3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(X[i] + 24))));
        }
        tile_i16_reg_m<M, C0, NC, C + 1>(tile, X, a, K);
    }
}

// rt2 unit (ft_core I16 math, any M <= 4): per 8-row group (one scale block), per tile, per C-chunk of 8 / M columns,
// the chunk's accumulators stay in registers over the 8 rows; flushed to fp32 with the block scale like rect<M, 2>.
template <int M> constexpr int rt2_nc() { return M == 1 ? 8 : M == 2 ? 4 : M <= 4 ? 2 : 1; }   // C per chunk: <= 8 accumulators
template <int M, int CH0>
__attribute__((always_inline)) inline void rt2_chunk(const uint8_t* p, size_t row_stride, const int32_t* const* xq, int g8, float* af, const __m256* sc, const Consts& K)
{
    constexpr int NC = rt2_nc<M>();
    __m256i a[NC * M];
    #pragma GCC unroll 8
    for (int k = 0; k < NC * M; ++k) a[k] = _mm256_setzero_si256();
    const int32_t* X[M];
    for (int rr = 0; rr < 8; ++rr)
    {
        for (int i = 0; i < M; ++i) X[i] = xq[i] + size_t(g8 + rr) * 32;
        tile_i16_reg_m<M, CH0, NC>(p + size_t(rr) * row_stride, X, a, K);
    }
    #pragma GCC unroll 8
    for (int c = 0; c < NC; ++c)
        for (int i = 0; i < M; ++i)
        {
            float* f = af + ((CH0 + c) * M + i) * 8;
            _mm256_store_ps(f, _mm256_fmadd_ps(_mm256_cvtepi32_ps(a[c * M + i]), sc[i], _mm256_load_ps(f)));
        }
}

template <int M, int CH0>
__attribute__((always_inline)) inline void rt2_chunks(const uint8_t* p, size_t row_stride, const int32_t* const* xq, int g8, float* af, const __m256* sc, const Consts& K)
{
    if constexpr (CH0 < 8)
    {
        rt2_chunk<M, CH0>(p, row_stride, xq, g8, af, sc, K);
        rt2_chunks<M, CH0 + rt2_nc<M>()>(p, row_stride, xq, g8, af, sc, K);
    }
}

template <int M>
void rect_rt2(const Mat& mat, int r0, int R, int t0, int C, const void* const* xin, const float* const* qs, float* y, size_t ldy)
{
    static thread_local float* accf = nullptr;
    if (!accf) accf = static_cast<float*>(std::aligned_alloc(64, size_t(MAXCT) * 64 * 8 * 4));
    std::memset(accf, 0, size_t(C) * 64 * M * 4);
    const Consts K = consts();
    const int tiles_n = mat.n / 16, tk = mat.k / 16;
    const size_t row_stride = size_t(tiles_n) * TILE_BYTES;
    const uint8_t* base = mat.tr + size_t(t0) * TILE_BYTES;
    const int32_t* xq[M]; for (int i = 0; i < M; ++i) xq[i] = static_cast<const int32_t*>(xin[i]);
    const int lines = (C * TILE_BYTES + 63) / 64;
    if (K_.pfpro) for (int r = r0; r < std::min(r0 + 8, r0 + R); ++r) for (int l = 0; l < lines; ++l) _mm_prefetch(reinterpret_cast<const char*>(base + size_t(r) * row_stride) + l * 64, _MM_HINT_T0);
    for (int g8 = r0; g8 < r0 + R; g8 += 8)
    {
        __m256 sc[M]; for (int i = 0; i < M; ++i) sc[i] = _mm256_set1_ps(qs[i][g8 >> 3]);
        const bool pfn = g8 + 8 < tk;
        for (int t = 0; t < C; ++t)
        {
            const uint8_t* p = base + size_t(g8) * row_stride + size_t(t) * TILE_BYTES;
            float* af = accf + size_t(t) * 64 * M;
            rt2_chunks<M, 0>(p, row_stride, xq, g8, af, sc, K);
            if (pfn && (t & 1) == 0)
                for (int rr = 0; rr < 8; ++rr)
                {
                    const char* q = reinterpret_cast<const char*>(p + size_t(8 + rr) * row_stride);
                    _mm_prefetch(q, _MM_HINT_T1); _mm_prefetch(q + 64, _MM_HINT_T1); _mm_prefetch(q + 128, _MM_HINT_T1);
                }
        }
    }
    for (int i = 0; i < M; ++i)
        for (int t = 0; t < C; ++t)
            for (int c = 0; c < 8; ++c)
            {
                const float* v = accf + size_t(t) * 64 * M + (c * M + i) * 8;
                y[i * ldy + t * 16 + c] = (v[0] + v[1]) + (v[2] + v[3]);
                y[i * ldy + t * 16 + c + 8] = (v[4] + v[5]) + (v[6] + v[7]);
            }
}

template <int KK>
void rect_rt1(const Mat& mat, int r0, int R, int t0, int C, const void* const* xin, const float* const* qs, float* y, size_t ldy)
{
    static thread_local float* accf = nullptr;
    if (!accf) accf = static_cast<float*>(std::aligned_alloc(64, size_t(MAXCT) * 64 * 4));
    std::memset(accf, 0, size_t(C) * 64 * 4);
    const KP P = kp();
    const Consts K = consts();
    const int tiles_n = mat.n / 16, tk = mat.k / 16;
    const size_t row_stride = size_t(tiles_n) * TILE_BYTES;
    const uint8_t* base = mat.tr + size_t(t0) * TILE_BYTES;
    const int lines = (C * TILE_BYTES + 63) / 64;
    if (K_.pfpro) for (int r = r0; r < std::min(r0 + 8, r0 + R); ++r) for (int l = 0; l < lines; ++l) _mm_prefetch(reinterpret_cast<const char*>(base + size_t(r) * row_stride) + l * 64, _MM_HINT_T0);
    for (int g8 = r0; g8 < r0 + R; g8 += 8)
    {
        const __m256 sc = _mm256_set1_ps(qs[0][g8 >> 3]);
        const bool pfn = g8 + 8 < tk;
        for (int t = 0; t < C; ++t)
        {
            __m256i a[8];
            #pragma GCC unroll 8
            for (int c = 0; c < 8; ++c) a[c] = _mm256_setzero_si256();
            const uint8_t* p = base + size_t(g8) * row_stride + size_t(t) * TILE_BYTES;
            if constexpr (KK == 3)
                for (int rr = 0; rr < 8; ++rr) tile_p16_reg(p + size_t(rr) * row_stride, static_cast<const int16_t*>(xin[0]) + size_t(g8 + rr) * 32, a, P);
            else
                for (int rr = 0; rr < 8; ++rr) tile_i16_reg(p + size_t(rr) * row_stride, static_cast<const int32_t*>(xin[0]) + size_t(g8 + rr) * 32, a, K);
            if (pfn && (t & 1) == 0)
                for (int rr = 0; rr < 8; ++rr)
                {
                    const char* q = reinterpret_cast<const char*>(p + size_t(8 + rr) * row_stride);
                    _mm_prefetch(q, _MM_HINT_T1); _mm_prefetch(q + 64, _MM_HINT_T1); _mm_prefetch(q + 128, _MM_HINT_T1);
                }
            float* af = accf + size_t(t) * 64;
            #pragma GCC unroll 8
            for (int c = 0; c < 8; ++c)
                _mm256_store_ps(af + c * 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(a[c]), sc, _mm256_load_ps(af + c * 8)));
        }
    }
    for (int t = 0; t < C; ++t)
        for (int c = 0; c < 8; ++c)
        {
            const float* v = accf + size_t(t) * 64 + c * 8;
            y[t * 16 + c] = (v[0] + v[1]) + (v[2] + v[3]);
            y[t * 16 + c + 8] = (v[4] + v[5]) + (v[6] + v[7]);
        }
    (void) ldy;
}

inline void rect_m(int kind, int m, const Mat& mat, int r0, int R, int t0, int C, const void* const* x, const float* const* qs, float* y, size_t ldy)
{
#define N135_CASE(MM) case MM: \
    if (kind == 3 && MM == 1 && K_.rt && !mat.swz) rect_rt1<3>(mat, r0, R, t0, C, x, qs, y, ldy); \
    else if (kind == 2 && K_.rt && !mat.swz) rect_rt2<MM>(mat, r0, R, t0, C, x, qs, y, ldy); \
    else if (kind == 3) rect<MM, 3>(mat, r0, R, t0, C, x, qs, y, ldy); \
    else if (kind == 2) rect<MM, 2>(mat, r0, R, t0, C, x, qs, y, ldy); \
    else if (kind == 1) rect<MM, 1>(mat, r0, R, t0, C, x, qs, y, ldy); \
    else rect<MM, 0>(mat, r0, R, t0, C, x, qs, y, ldy); break;
    switch (m) { N135_CASE(1) N135_CASE(2) N135_CASE(3) N135_CASE(4) N135_CASE(5) N135_CASE(6) N135_CASE(7) default: N135_CASE(8) }
#undef N135_CASE
}

// ------------------------------------------------------------------------------------------
// dataflow forward
// ------------------------------------------------------------------------------------------
enum : uint8_t { U_PREP, U_GU, U_ACT, U_DOWN, U_OUT };
struct Unit { uint8_t kind, which; uint16_t j; uint16_t a, b; };   // a = band, b = column group (GEMV); a = blk (ACT/OUT)

struct alignas(64) Ctr { std::atomic<int> v{0}; char pad[60]; };

struct Fwd4
{
    const Layer* L = nullptr;
    const float* x = nullptr; float* out = nullptr; int ntok = 0, mode = 2, kind = 3;
    std::vector<Job> jobs; std::vector<int> joff; int ns = 0;
    int rg = 0, cg = 8, rd = 0, cd = 8, nbg = 1, ncg = 1, nbd = 1, ncd = 1;
    // per slot: activations (layout by kind), per-128-block scales, sums; band partials (lane-reduced raw dots)
    std::vector<float> xg, xu, xd, qg, qu, qd, sg, su, sd, pg, pu, pd;
    std::vector<Unit> units;
    std::vector<Ctr> c_prep, c_gu, c_act, c_down;
    std::vector<int> n_gu, n_down;
    std::atomic<int> next{0};
    // late-bound jobs (engine: NVMe->CPU picks): bound by the first unit that needs them, once their landing arrives
    const std::function<bool(int)>* bind = nullptr;
    std::vector<uint8_t> jlate;
    std::vector<Ctr> c_bind;            // 0 unbound, 1 binding, 2 bound, 3 failed (skipped)
};

// true when job j's expert pointers are valid (late jobs: bind on first use; a failed bind skips the job)
inline bool job_ready(Fwd4& F, int j)
{
    if (!F.jlate[j]) return true;
    std::atomic<int>& st = F.c_bind[j].v;
    int v = st.load(std::memory_order_acquire);
    if (v < 2)
    {
        int z = 0;
        if (v == 0 && st.compare_exchange_strong(z, 1, std::memory_order_acq_rel))
        {
            const bool ok = (*F.bind)(F.jobs[j].e);
            st.store(ok ? 2 : 3, std::memory_order_release);
        }
        else
            while (st.load(std::memory_order_acquire) < 2) _mm_pause();
    }
    return st.load(std::memory_order_acquire) == 2;
}

inline void spin_until(const std::atomic<int>& c, int target) { while (c.load(std::memory_order_acquire) < target) _mm_pause(); }

inline void fwd4_fn(void* vc, int, int)
{
    Fwd4& F = *static_cast<Fwd4*>(vc);
    const Layer& L = *F.L;
    const int H = L.H, I = L.I, BH = H / 128, BI = I / 128;
    const size_t XH = size_t(H) * 2, XI = size_t(I) * 2;   // floats per slot (largest layout: 32 x 4 B per tile row)
    const int nu = int(F.units.size());
    const bool i16 = F.mode == 2;
    while (true)
    {
        const int ui = F.next.fetch_add(1, std::memory_order_relaxed);
        if (ui >= nu) break;
        const Unit U = F.units[ui];
        if (U.kind != U_OUT && !job_ready(F, U.j))
        {
            // skipped job: complete its counters so later stages and the out units do not wait on it
            (U.kind == U_PREP ? F.c_prep : U.kind == U_GU ? F.c_gu : U.kind == U_ACT ? F.c_act : F.c_down)[U.j].v.fetch_add(1, std::memory_order_release);
            continue;
        }
        const Job& J = F.jobs[U.j];
        const int s0 = F.joff[U.j];
        const Expert& E = L.ex[J.e];
        if (U.kind == U_PREP)
        {
            const int i = U.a, which = U.which, s = s0 + i;
            const Mat& M = which ? E.u : E.g;
            float* xp = (which ? F.xu : F.xg).data() + size_t(s) * XH;
            float* qs = (which ? F.qu : F.qg).data() + size_t(s) * BH;
            float sum = 0; alignas(32) float xt[128];
            const float* xr = F.x + size_t(J.tok[i]) * H;
            for (int b = 0; b < BH; ++b)
            {
                prep_block_v(xr + b * 128, M.suh + b * 128, xt);
                if (i16)
                {
                    if (F.kind == 3) qs[b] = quant_block_p16(xt, reinterpret_cast<int16_t*>(xp) + size_t(b) * 256, &sum);
                    else qs[b] = quant_block_i16(xt, reinterpret_cast<int32_t*>(xp) + size_t(b) * 256, &sum);
                    continue;
                }
                for (int r = 0; r < 128; ++r) sum += xt[r];
                for (int tt = 0; tt < 8; ++tt) permute_rows(xt + tt * 16, xp + (size_t(b) * 8 + tt) * 32);
            }
            (which ? F.su : F.sg)[s] = sum;
            F.c_prep[U.j].v.fetch_add(1, std::memory_order_release);
        }
        else if (U.kind == U_GU)
        {
            spin_until(F.c_prep[U.j].v, 2 * J.m);
            const Mat& M = U.which ? E.u : E.g;
            const void* xp[MAXM]; const float* qs[MAXM];
            for (int i = 0; i < J.m; ++i)
            {
                xp[i] = (U.which ? F.xu : F.xg).data() + size_t(s0 + i) * XH;
                qs[i] = (U.which ? F.qu : F.qg).data() + size_t(s0 + i) * BH;
            }
            const int tk = M.k / 16, R = F.rg, C = F.cg;
            // partials [band][slot][I]
            float* y = (U.which ? F.pu : F.pg).data() + (size_t(U.a) * F.ns + s0) * I + size_t(U.b) * C * 16;
            rect_m(F.kind, J.m, M, U.a * R, std::min(R, tk - U.a * R), U.b * C, C, xp, qs, y, I);
            F.c_gu[U.j].v.fetch_add(1, std::memory_order_release);
        }
        else if (U.kind == U_ACT)
        {
            spin_until(F.c_gu[U.j].v, F.n_gu[U.j]);
            const int blk = U.a;
            for (int i = 0; i < J.m; ++i)
            {
                const int s = s0 + i;
                alignas(32) float g[128], up[128], a[128], xt[128];
                for (int c = 0; c < 128; ++c) { g[c] = F.pg[size_t(s) * I + blk * 128 + c]; up[c] = F.pu[size_t(s) * I + blk * 128 + c]; }
                for (int bd = 1; bd < F.nbg; ++bd)
                    for (int c = 0; c < 128; ++c) { g[c] += F.pg[(size_t(bd) * F.ns + s) * I + blk * 128 + c]; up[c] += F.pu[(size_t(bd) * F.ns + s) * I + blk * 128 + c]; }
                if (F.mode != 1)
                {
                    const float sgv = F.sg[s], suv = F.su[s];
                    for (int c = 0; c < 128; ++c) { g[c] = KINV * g[c] + CAFF * sgv; up[c] = KINV * up[c] + CAFF * suv; }
                }
                out_block(g, E.g.svh + blk * 128);
                out_block(up, E.u.svh + blk * 128);
                for (int r = 0; r < 128; ++r) a[r] = r16(act_gu(g[r], up[r]));
                prep_block(a, E.d.suh + blk * 128, xt);
                float* xp = F.xd.data() + size_t(s) * XI;
                float sum = 0;
                if (i16)
                {
                    if (F.kind == 3) F.qd[size_t(s) * BI + blk] = quant_block_p16(xt, reinterpret_cast<int16_t*>(xp) + size_t(blk) * 256, &sum);
                    else F.qd[size_t(s) * BI + blk] = quant_block_i16(xt, reinterpret_cast<int32_t*>(xp) + size_t(blk) * 256, &sum);
                }
                else
                {
                    for (int r = 0; r < 128; ++r) sum += xt[r];
                    for (int tt = 0; tt < 8; ++tt) permute_rows(xt + tt * 16, xp + (size_t(blk) * 8 + tt) * 32);
                }
                F.sd[size_t(s) * BI + blk] = sum;
            }
            F.c_act[U.j].v.fetch_add(1, std::memory_order_release);
        }
        else if (U.kind == U_DOWN)
        {
            spin_until(F.c_act[U.j].v, BI);
            const void* xp[MAXM]; const float* qs[MAXM];
            for (int i = 0; i < J.m; ++i) { xp[i] = F.xd.data() + size_t(s0 + i) * XI; qs[i] = F.qd.data() + size_t(s0 + i) * BI; }
            const int tk = E.d.k / 16, R = F.rd, C = F.cd;
            float* y = F.pd.data() + (size_t(U.a) * F.ns + s0) * H + size_t(U.b) * C * 16;
            rect_m(F.kind, J.m, E.d, U.a * R, std::min(R, tk - U.a * R), U.b * C, C, xp, qs, y, H);
            F.c_down[U.j].v.fetch_add(1, std::memory_order_release);
        }
        else
        {
            // out block: reduce down bands, fixup, output transform, weighted accumulate in job order (race-free)
            const int blk = U.a;
            for (int t = 0; t < F.ntok; ++t) std::memset(F.out + size_t(t) * H + blk * 128, 0, 512);
            for (int j = 0; j < int(F.jobs.size()); ++j)
            {
                spin_until(F.c_down[j].v, F.n_down[j]);
                if (F.jlate[j] && F.c_bind[j].v.load(std::memory_order_acquire) != 2) continue;
                const Job& Jj = F.jobs[j]; const int sj = F.joff[j];
                const Expert& Ej = L.ex[Jj.e];
                for (int i = 0; i < Jj.m; ++i)
                {
                    const int s = sj + i;
                    alignas(32) float y[128];
                    for (int c = 0; c < 128; ++c) y[c] = F.pd[size_t(s) * H + blk * 128 + c];
                    for (int bd = 1; bd < F.nbd; ++bd) for (int c = 0; c < 128; ++c) y[c] += F.pd[(size_t(bd) * F.ns + s) * H + blk * 128 + c];
                    if (F.mode != 1)
                    {
                        float sdv = 0; for (int b = 0; b < BI; ++b) sdv += F.sd[size_t(s) * BI + b];
                        for (int c = 0; c < 128; ++c) y[c] = KINV * y[c] + CAFF * sdv;
                    }
                    out_block(y, Ej.d.svh + blk * 128);
                    float* o = F.out + size_t(Jj.tok[i]) * H + blk * 128; const float wt = Jj.w[i];
                    for (int c = 0; c < 128; ++c) o[c] += wt * y[c];
                }
            }
        }
    }
}

template <typename PoolT>
// late_e (optional, indexed by expert id): experts whose bytes may not be resident yet; their jobs run after every other
// job and are bound by bind_late(e) (blocking until landed; false = skip) when a worker first reaches them.
void moe_forward4(PoolT& pool, const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route, float* out, int mode,
                  const uint8_t* late_e = nullptr, const std::function<bool(int)>* bind_late = nullptr)
{
    static Fwd4 F;
    F.L = &L; F.x = x; F.out = out; F.ntok = ntok; F.jobs.clear(); F.joff.clear(); F.jlate.clear(); F.bind = bind_late;
    int nram = 0;
    for (int pass = 0; pass < (late_e ? 2 : 1); ++pass)
    {
        std::vector<int> jid(L.ex.size(), -1);
        for (int t = 0; t < ntok; ++t)
            for (auto& [e, wt] : route[t])
            {
                if (late_e && (late_e[e] != 0) != (pass == 1)) continue;
                if (jid[e] < 0 || F.jobs[jid[e]].m == MAXM) { jid[e] = int(F.jobs.size()); F.jobs.push_back({ e, 0, {}, {} }); F.jlate.push_back(uint8_t(pass)); }
                Job& J = F.jobs[jid[e]]; J.tok[J.m] = t; J.w[J.m] = wt; ++J.m;
            }
        if (pass == 0) nram = int(F.jobs.size());
    }
    if (!late_e) nram = int(F.jobs.size());
    const int H = L.H, I = L.I, BH = H / 128, BI = I / 128;
    const int nj = int(F.jobs.size());
    if (!nj) { std::memset(out, 0, size_t(ntok) * H * 4); return; }
    int maxm = 0, ns = 0;
    for (auto& J : F.jobs) { F.joff.push_back(ns); ns += J.m; maxm = std::max(maxm, J.m); }
    F.ns = ns;
    F.mode = mode >= 0 ? mode : (maxm <= K_.i16max ? 2 : 0);
    F.kind = F.mode == 2 ? (K_.kern ? 3 : 2) : F.mode;
    const int tkg = H / 16, tng = I / 16, tkd = I / 16, tnd = H / 16;
    const bool band = maxm <= K_.bandmaxm;
    F.rg = band && K_.rg > 0 ? std::min(K_.rg, tkg) : tkg; F.cg = band ? std::min(K_.cg, tng) : 8;
    F.rd = band && K_.rd > 0 ? std::min(K_.rd, tkd) : tkd; F.cd = band ? std::min(K_.cd, tnd) : 8;
    if (L.ex[F.jobs[0].e].g.swz) { F.cg = 8; F.cd = 8; }
    F.nbg = (tkg + F.rg - 1) / F.rg; F.ncg = (tng + F.cg - 1) / F.cg;
    F.nbd = (tkd + F.rd - 1) / F.rd; F.ncd = (tnd + F.cd - 1) / F.cd;
    auto grow = [](std::vector<float>& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(F.xg, size_t(ns) * H * 2); grow(F.xu, size_t(ns) * H * 2); grow(F.xd, size_t(ns) * I * 2);
    grow(F.qg, size_t(ns) * BH); grow(F.qu, size_t(ns) * BH); grow(F.qd, size_t(ns) * BI);
    grow(F.sg, ns); grow(F.su, ns); grow(F.sd, size_t(ns) * BI);
    grow(F.pg, size_t(F.nbg) * ns * I); grow(F.pu, size_t(F.nbg) * ns * I); grow(F.pd, size_t(F.nbd) * ns * H);
    if (int(F.c_prep.size()) < nj) { F.c_prep = std::vector<Ctr>(nj + 8); F.c_gu = std::vector<Ctr>(nj + 8); F.c_act = std::vector<Ctr>(nj + 8); F.c_down = std::vector<Ctr>(nj + 8); F.c_bind = std::vector<Ctr>(nj + 8); }
    F.n_gu.assign(nj, 2 * F.nbg * F.ncg); F.n_down.assign(nj, F.nbd * F.ncd);
    for (int j = 0; j < nj; ++j) { F.c_prep[j].v.store(0, std::memory_order_relaxed); F.c_gu[j].v.store(0, std::memory_order_relaxed); F.c_act[j].v.store(0, std::memory_order_relaxed); F.c_down[j].v.store(0, std::memory_order_relaxed); F.c_bind[j].v.store(0, std::memory_order_relaxed); }
    F.units.clear();
    // stage-major per job group: resident jobs [0, nram), then late jobs [nram, nj)
    for (int grp = 0; grp < 2; ++grp)
    {
        const int j0 = grp ? nram : 0, j1 = grp ? nj : nram;
        for (int j = j0; j < j1; ++j) for (int i = 0; i < F.jobs[j].m; ++i) for (int w = 0; w < 2; ++w) F.units.push_back({ U_PREP, uint8_t(w), uint16_t(j), uint16_t(i), 0 });
        // gate/up: per job, band-major, g and u interleaved so the two halves of a 128-block finish together
        for (int j = j0; j < j1; ++j) for (int b = 0; b < F.nbg; ++b) for (int c = 0; c < F.ncg; ++c) for (int w = 0; w < 2; ++w) F.units.push_back({ U_GU, uint8_t(w), uint16_t(j), uint16_t(b), uint16_t(c) });
        for (int j = j0; j < j1; ++j) for (int b = 0; b < BI; ++b) F.units.push_back({ U_ACT, 0, uint16_t(j), uint16_t(b), 0 });
        for (int j = j0; j < j1; ++j) for (int b = 0; b < F.nbd; ++b) for (int c = 0; c < F.ncd; ++c) F.units.push_back({ U_DOWN, 0, uint16_t(j), uint16_t(b), uint16_t(c) });
    }
    for (int b = 0; b < BH; ++b) F.units.push_back({ U_OUT, 0, 0, uint16_t(b), 0 });
    F.next.store(0, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    pool.run(&fwd4_fn, &F);
}

// ------------------------------------------------------------------------------------------
// bench adapter
// ------------------------------------------------------------------------------------------
struct Engine2
{
    Pool* pool = nullptr;
    double late_us = 50;
    void init(Pool& p) { pool = &p; init_p16(); }
    void forward(const std::string& v, const Layer& L, const float* x, int m, const std::vector<std::vector<std::pair<int, float>>>& route, float* out, int mode)
    {
        if (v == "band") { moe_forward_band(*pool, L, x, m, route, out, mode); return; }
        if (v == "v4") { moe_forward4(*pool, L, x, m, route, out, mode); return; }
        if (v == "v4late" || v == "v4fail")
        {
            // every second distinct expert of the call is late-bound; the bind waits late_us (a landing) then succeeds
            // (v4late) or fails (v4fail: those experts are skipped)
            std::vector<uint8_t> late(L.ex.size(), 0); int k = 0;
            for (auto& r : route) for (auto& pr : r) if (!late[pr.first] && (k++ & 1)) late[pr.first] = 2;
            for (auto& q : late) q = q == 2;
            const bool ok = v == "v4late";
            const double us = late_us;
            std::function<bool(int)> fn = [ok, us](int) { const double t0 = now(); while (now() - t0 < us * 1e-6) _mm_pause(); return ok; };
            moe_forward4(*pool, L, x, m, route, out, mode, late.data(), &fn);
            return;
        }
        // v4:kern:rg:cg:rd:cd[:rt[:bandmaxm]]  (per-variant knob override; restored after the call)
        if (v.rfind("v4:", 0) == 0)
        {
            const Knobs save = K_;
            int vals[7] = { K_.kern, K_.rg, K_.cg, K_.rd, K_.cd, K_.rt, 8 };   // explicit shapes apply at every m
            std::stringstream ss(v.substr(3)); std::string t; int k = 0;
            while (std::getline(ss, t, ':') && k < 7) vals[k++] = std::stoi(t);
            K_.kern = vals[0]; K_.rg = vals[1]; K_.cg = vals[2]; K_.rd = vals[3]; K_.cd = vals[4]; K_.rt = vals[5]; K_.bandmaxm = vals[6];
            moe_forward4(*pool, L, x, m, route, out, mode);
            K_ = save;
            return;
        }
        moe_forward(*pool, L, x, m, route, out, mode);
    }
};

}  // namespace n135
}  // namespace

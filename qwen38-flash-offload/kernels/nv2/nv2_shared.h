// N119 nv2: shared layout between the device kernels (nv2_dev.cu) and the host engine (nv2_host.cpp).
// All structs below live in ONE pinned, GPU-mapped host region (anon mmap + cudaHostRegister(Mapped)).
#pragma once
#include <cstdint>

namespace nv2s {

constexpr int MAXU = 512;   // capacity: GLM 288, Qwen3.8 512 experts
constexpr int MAXP = 80;    // decode: <= 8 tokens x top-10
constexpr int RQ = 64;      // request / reply ring entries
constexpr int LG = 8192;    // write-back / admission log ring entries
constexpr int LR = 256;     // landing ring (fresh RAM slots for VRAM victims)
constexpr int MAXB = 8;     // tokens per CPU job

// host control words (mapped int64)
enum { HC_PUB = 0, HC_ACK = 1, HC_ERR = 2, HC_WBN = 3, HC_ADMN = 4, HC_LANDT = 5, HC_CDONE = 6, HC_N = 64 };
// device control words (device int64)
enum { DC_SEQ = 0, DC_LANDH = 1, DC_WBN = 2, DC_ADMN = 3, DC_NU = 4, DC_TPUB = 5, DC_CDONE = 6, DC_N = 16 };
constexpr int DBGW = 8;    // debug ring words per request
constexpr int DBG = 4096;   // latency debug ring (device: tpub, t_step, t_seen; host: t_notice, t_reply)
// lanes (reply)
enum { LN_RAM = 0, LN_NVG = 1, LN_CPU = 2, LN_VRAM = 3, LN_NVC = 4, LN_RZC = 5, LN_NZC = 6, LN_B70 = 7 };
// LN_B70 (N137): computed by the B70 expert server from its own VRAM; masked on the GPU like a CPU pick, the CPU
// worker posts it to the B70 ring and adds the returned rows into hout before HC_CDONE
// LN_RAM / LN_NVG: GPU, admitted to VRAM (copy) | LN_RZC / LN_NZC: GPU zero-copy from the (landed) RAM slot, not admitted
// request kinds
enum { RK_DECODE = 0, RK_SMALL = 1 };

struct Req {                 // device -> host, one per MoE layer call (seq written last)
    int64_t seq;
    int64_t tpub;            // device globaltimer at publication (ns)
    int32_t li, nu, np, ntok, kind, pad;
    int32_t key[MAXU];       // unique picked keys (li * E + e), expert id ascending
    int32_t cls[MAXU];       // LN_VRAM if VRAM-resident at publication, else 0
    int32_t cnt[MAXU];       // tokens that picked it
    int32_t pe[MAXP];        // per pick: local expert id (np <= MAXP only)
    float pw[MAXP];          // per pick: routing weight
    int32_t npf, pad2;       // prefetch hints: predicted picks of the NEXT layer (not VRAM-resident), deduped
    int32_t pf_key[MAXP];
};

struct Rep {                 // host -> device (seq written last)
    int64_t seq;
    int32_t ncpu, nnv;
    int32_t lane[MAXU];
};

struct LogE { int64_t seq; int32_t key, slot; };

// stats (device uint64) indices
enum { ST_HIT = 0, ST_MISS = 1, ST_ADMIT = 2, ST_NOVICT = 3, ST_EVICT = 4, ST_WB = 5, ST_WB_NOSLOT = 6, ST_FIX = 7,
       ST_BAD = 8, ST_WAITNS = 9, ST_WAITS = 10, ST_CWAITNS = 11, ST_CWAITS = 12, ST_TIMEOUT = 13, ST_N = 16 };

}  // namespace nv2s

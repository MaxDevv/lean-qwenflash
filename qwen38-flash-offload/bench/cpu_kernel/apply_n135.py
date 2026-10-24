#!/usr/bin/env python3
"""N135: patch an nv2 engine source dir (the one holding nv2_host.cpp + ft_core.h) to carry the N135 CPU-lane forward
behind a runtime switch, leaving the old kernel the default.

  QWEN38_NV_CPU_KERN=0 (default)  ft_core.h moe_forward, unchanged behaviour
  QWEN38_NV_CPU_KERN=1            n135::moe_forward4 (ft_n135.h): dataflow forward + p16 I16 kernel; tuning env
                                 QWEN38_NV_CPU_RG/CG/RD/CD (unit rows/tiles), QWEN38_NV_CPU_PFR/PFHINT/PFPRO (prefetch),
                                 QWEN38_NV_CPU_KERN_P16 (1 = p16 I16 kernel, 0 = ft_core I16 kernel inside v4)

usage: apply_n135.py ENGINE_SRC_DIR [FT_N135_H]   (idempotent; copies ft_n135.h next to nv2_host.cpp)
"""
import os, re, shutil, sys

d = sys.argv[1]
src_h = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft_n135.h")
p = os.path.join(d, "nv2_host.cpp")
s = open(p).read()
if "N135 CPU-lane forward switch" in s:
    print("already patched:", p)
else:
    hook = '''#include "ft_core.h"
#include "ft_n135.h"
// N135 CPU-lane forward switch: QWEN38_NV_CPU_KERN=1 -> n135::moe_forward4 (ft_n135.h), default 0 -> ft_core moe_forward
static int g_cpu_kern = -1;
static inline void cpu_kern_init()
{
    if (g_cpu_kern >= 0) return;
    const char* s = getenv("QWEN38_NV_CPU_KERN");
    g_cpu_kern = s && *s ? atoi(s) : 0;
    n135::configure_env(); n135::init_p16();
    fprintf(stderr, "[nv2] CPU lane kernel: %s (rg %d cg %d rd %d cd %d bandmaxm %d pfr %d p16 %d rt %d merge %d)\\n", g_cpu_kern == 1 ? "n135 v4" : "ft_core moe_forward",
            n135::K_.rg, n135::K_.cg, n135::K_.rd, n135::K_.cd, n135::K_.bandmaxm, n135::K_.pfr, n135::K_.kern, n135::K_.rt, n135::K_.merge);
}
static inline void cpu_moe_forward(Pool& pool, const Layer& L, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route,
                                   float* out, int mode, PhaseTimes* pt = nullptr)
{
    cpu_kern_init();
    if (g_cpu_kern == 1) n135::moe_forward4(pool, L, x, ntok, route, out, mode);
    else moe_forward(pool, L, x, ntok, route, out, mode, pt);
}'''
    assert s.count('#include "ft_core.h"') == 1, "ft_core.h include not found exactly once"
    s = s.replace('#include "ft_core.h"', hook, 1)
    n0 = len(re.findall(r"\bmoe_forward\((e\.)?pool, Ly,", s))
    assert n0 >= 2, f"expected >= 2 moe_forward call sites, found {n0}"
    s = re.sub(r"\bmoe_forward\(((?:e\.)?pool), Ly,", r"cpu_moe_forward(\1, Ly,", s)
    # merged RAM + NVMe->CPU forward (QWEN38_NV_CPU_MERGE=1, only with QWEN38_NV_CPU_KERN=1)
    # insert the merged block at the start of the line holding the RAM-pick forward (works for both the plain
    # "cpu_moe_forward(pool, Ly, ..., r1, ...)" line and main's guarded "if (any1 || !nb) cpu_moe_forward(...)" line,
    # and never jumps over a declaration: the label sits right before the unpin block)
    key = "cpu_moe_forward(pool, Ly, xf.data(), ntok, r1, hout, mode"
    unpin = "        {\n            std::lock_guard<std::mutex> lk(mu);\n            for (int s : pinned) pinc[s]--;\n        }"
    if s.count(key) == 1 and s.count(unpin) == 1 and "bool any2" in s and "auto bind = [&](int k) -> bool" in s:
        merged = '''        if (g_cpu_kern == 1 && n135::K_.merge && any2)
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
'''
        at = s.rfind("\n", 0, s.index(key)) + 1
        s = s[:at] + merged + s[at:]
        s = s.replace(unpin, "    n135_done:\n" + unpin, 1)
        print("merged NVC path: inserted")
    else:
        print("merged NVC path: anchors not found, skipped (switch still works)")
    # init next to the table init in cpu_init (selftest runs right after)
    s = s.replace("init_perm(); init_tables();", "init_perm(); init_tables(); cpu_kern_init();", 1)
    open(p, "w").write(s)
    print(f"patched {p}: {n0} call sites")
# ft_core.h: clamp the 16-byte trellis window start to byte 80 of the 96-byte tile (groups 29-31 read 4-8 bytes past the
# tile otherwise: a fault when the tile is the last one before an unmapped page). Same bytes selected; bit-identical.
fc = os.path.join(d, "ft_core.h")
c = open(fc).read()
if "N135 window clamp" not in c and "window clamp" not in c and "std::min(4 * (q0 / 4), 80)" not in c:
    a1 = "else { const int L = 4 * (q0 / 4); for (int b = 0; b < 16; ++b) win[b] = L + b; WOFF[g] = L; }"
    a2 = "        constexpr int L = 4 * ((3 * g - 2) / 4);"
    assert c.count(a1) == 1 and c.count(a2) == 2, "ft_core.h window code not as expected"
    c = c.replace(a1, "else { const int L = std::min(4 * (q0 / 4), 80); /* N135 window clamp */ for (int b = 0; b < 16; ++b) win[b] = L + b; WOFF[g] = L; }")
    c = c.replace(a2, "        constexpr int L = 4 * ((3 * g - 2) / 4) < 80 ? 4 * ((3 * g - 2) / 4) : 80;   // N135 window clamp (no read past the tile)")
    open(fc, "w").write(c)
    print("ft_core.h: window clamp applied")
else:
    print("ft_core.h: window clamp already present (main >= 4dafe6f)")
shutil.copy(src_h, os.path.join(d, "ft_n135.h"))
print("copied ft_n135.h ->", d)

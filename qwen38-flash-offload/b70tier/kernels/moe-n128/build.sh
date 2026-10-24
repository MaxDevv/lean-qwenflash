#!/usr/bin/env bash
# Build _moe_n128.so: the exl3xpu grouped-MoE library (pointer-table experts, power-of-two gate/up and down splits) with
# the Qwen3.8-Flash-Next swiglu clamp at 10 (EXL3_ACT_LIMIT, sglang swiglu_clamped semantics) for the B70 expert server.
# Source: exl3_moe.n128.sycl (trellis-serve exl3_moe.sycl + the a9/n124 splits + the clamp; campaign N128). Runs inside
# the sglang-exl3-xpu-flashnext image (oneAPI icpx, torch XPU); no GPU needed. Device code is spir64 (JIT on first
# launch; the server warms every row count at start), the flags the N128/N137 measurements used.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
cd "$(dirname "$0")"
OUT=${MOE_OUT:-_moe_n128.so}
T=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel \
  -DEXL3_ACT_LIMIT=10.0f -D_GLIBCXX_USE_CXX11_ABI=1 -I . -I"$T/include" -I"$T/include/torch/csrc/api/include" \
  -x c++ exl3_moe.n128.sycl -x none -o "$OUT.tmp" \
  -L"$T/lib" -Wl,-rpath,"$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
mv -f "$OUT.tmp" "$OUT"
ls -la "$OUT"

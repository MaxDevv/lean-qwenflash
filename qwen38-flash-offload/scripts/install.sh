#!/bin/bash
# Image-build step (also usable on a host with exllamav3 1.5.1 installed): pre-build the torch extensions (GPU expert
# cache kernels, CPU tier host + GPU split/combine, fast split kernel, fused hyper-connection kernels, NVMe tier
# engines nv_tier (build/nv) and nv2 (build/nv2: host controller + CPU lane -mavx2, device kernels)), so the first
# start compiles none of them. Needs nvcc + ninja (no GPU needed); built for $TORCH_CUDA_ARCH_LIST.
# The Triton kernels compile on first use; their autotune picks come from data/triton_pin_exact.json (no benchmarking).
#   scripts/install.sh [ROOT]     ROOT defaults to the checkout this script lives in
set -euo pipefail
ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
mkdir -p "$ROOT/triton_cache"
cd "$ROOT"
MAX_JOBS=${MAX_JOBS:-4} python3 - <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "qwen38"))
from torch.utils.cpp_extension import load
import expert_cache, cpu_tier, k_hcfuse, k_ftsplit, nv_tier, nv2
expert_cache._ext()
cpu_tier._build()
k_hcfuse.ext()
d = os.environ.get("QWEN38_K110_BUILD", os.path.join(os.getcwd(), "build", "k110f"))
os.makedirs(d, exist_ok=True)
fast = load(name="ft_split_fast", sources=[k_ftsplit._SRC], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
nv1, nv2x = nv_tier._ext(), nv2.ext()
print("extensions built:", expert_cache._EXT.__file__, cpu_tier._H.__file__, cpu_tier._CU.__file__, k_hcfuse._EXT.__file__,
      fast.__file__, nv1.__file__, nv2x.__file__)
PY

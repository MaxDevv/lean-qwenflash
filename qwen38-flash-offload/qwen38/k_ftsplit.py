"""Faster CPU-tier handoff kernel (kernels/k110/ft_split_fast.cu), bit-identical outputs to the stock
ft_split (kernels/cpu_avx2/ft_tier_cu.cu): 3000 random decode cases, 0 mismatches. Stock 24 / 39 / 73 us at bsz
1 / 4 / 8 with a job -> 8.6 / 11.5 / 19 us (the kernel sits on the critical path before the CPU job can start).
Env: QWEN38_K_FTSPLIT=1 (serve.py, after cpu_tier.start): cpu_tier._CU.ft_split -> fast kernel for <= 64 picks.
No change to cpu_tier.py: its module-level _CU handle is swapped for a namespace with the fast ft_split.
"""
import os, types
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "kernels", "k110", "ft_split_fast.cu")
STATS = {"fast": 0, "stock": 0}


def install():
    import cpu_tier
    from torch.utils.cpp_extension import load
    d = os.environ.get("QWEN38_K110_BUILD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "k110f"))
    os.makedirs(d, exist_ok=True)
    fast = load(name="ft_split_fast", sources=[_SRC], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    stock = cpu_tier._CU

    def ft_split(sel, *a):
        if sel.numel() <= 64:
            STATS["fast"] += 1
            return fast.ft_split(sel, *a)
        STATS["stock"] += 1
        return stock.ft_split(sel, *a)
    cpu_tier._CU = types.SimpleNamespace(ft_split=ft_split, ft_combine=stock.ft_combine)
    print(" -- k_ftsplit (K110): CPU-tier ft_split -> fast kernel (<= 64 picks)", flush=True)

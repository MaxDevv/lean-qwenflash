"""N137 step 2 reference: Qwen3.8-Flash-Next routed experts through exllamav3's CUDA path on the RTX 3090.

Real checkpoint experts (store layers li 0 / 20 / 41 = model layers 3 / 23 / 44, 16 experts each), built as
exllamav3 LinearEXL3 (trellis + suh + svh + mul1) and run exactly like BlockSparseMLP's per-expert torch path:
u = up(x), g = gate(x), ext.silu_mul(g, u, a, act_limit 10), y = down(a); routed output = sum_k w_k * y_k (fp32).
Also an fp64 reference from the dequantised weights (get_weight_tensor). Writes /out/cuda_ref.npz.
"""
import json, os, sys, time
import numpy as np
import torch
from safetensors import safe_open
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.ext import exllamav3_ext as ext

M_DIR = os.environ.get("MODEL", "/models")
H, I, E, LIM = 4096, 2048, 288, 10.0
LAYERS = [(0, 3), (20, 23), (41, 44)]          # (store li, model layer)
NE = 16
dev = torch.device("cuda", 0)
idx = json.load(open(f"{M_DIR}/model.safetensors.index.json"))["weight_map"]
rng = np.random.default_rng(137)


def T(name):
    with safe_open(f"{M_DIR}/{idx[name]}", framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def lin(pre, fin, fout):
    d = {k: T(f"{pre}.{k}") for k in ("trellis", "suh", "svh", "mul1")}
    return LinearEXL3(None, fin, fout, suh=d["suh"].to(dev), svh=d["svh"].to(dev), trellis=d["trellis"].to(dev),
                      mul1=d["mul1"].to(dev), key=pre)


out = {}
meta = {"layers": [], "cases": []}
for li, ml in LAYERS:
    exps = sorted(rng.choice(E, NE, replace=False).tolist())
    mods = {}
    for e in exps:
        p = f"model.language_model.layers.{ml}.mlp.experts.{e}"
        mods[e] = (lin(p + ".gate_proj", H, I), lin(p + ".up_proj", H, I), lin(p + ".down_proj", I, H))
    W64 = {}
    for e in exps:
        g, u, d = mods[e]
        W64[e] = tuple(m.get_weight_tensor().cpu().double() for m in (g, u, d))
    meta["layers"].append({"li": li, "model_layer": ml, "experts": exps})
    for M in (1, 2, 4, 8):
        for scale in (0.5, 4.0):
            x = (torch.randn(M, H, generator=torch.Generator().manual_seed(1000 * li + 10 * M + int(scale))) * scale).half()
            ids = np.stack([rng.choice(exps, 8, replace=False) for _ in range(M)]).astype(np.int32)
            w = rng.random((M, 8)).astype(np.float32)
            w = w / w.sum(1, keepdims=True) * 2.5
            xd = x.to(dev)
            y = torch.zeros(M, H, dtype=torch.float32, device=dev)
            y64 = torch.zeros(M, H, dtype=torch.float64)
            for t in range(M):
                xt = xd[t:t + 1].contiguous()
                for k in range(8):
                    e = int(ids[t, k])
                    g, u, d = mods[e]
                    uu = u.forward(xt, {})
                    gg = g.forward(xt, {})
                    a = uu if uu.dtype == torch.half else torch.empty_like(uu, dtype=torch.half)
                    ext.silu_mul(gg, uu, a, LIM)
                    yk = d.forward(a, {})
                    y[t] += float(w[t, k]) * yk[0].float()
                    Wg, Wu, Wd = W64[e]
                    x64 = xt.cpu().double()
                    g64 = (x64 @ Wg).clamp(max=LIM)
                    u64 = (x64 @ Wu).clamp(-LIM, LIM)
                    a64 = torch.nn.functional.silu(g64) * u64
                    y64[t] += float(w[t, k]) * (a64 @ Wd)[0]
            torch.cuda.synchronize()
            tag = f"l{li}_m{M}_s{scale}"
            out[tag + "_x"] = x.numpy()
            out[tag + "_ids"] = ids
            out[tag + "_w"] = w
            out[tag + "_y"] = y.cpu().numpy()
            out[tag + "_y64"] = y64.numpy()
            rel = float((y.cpu().double() - y64).norm() / y64.norm())
            meta["cases"].append({"tag": tag, "li": li, "M": M, "scale": scale, "cuda_vs_fp64_rel_l2": round(rel, 6),
                                  "finite": bool(torch.isfinite(y).all())})
            print(json.dumps(meta["cases"][-1]), flush=True)
    del mods, W64
    torch.cuda.empty_cache()
np.savez(os.environ.get("OUT", "/out/cuda_ref.npz"), **out)
json.dump(meta, open(os.environ.get("OUT", "/out/cuda_ref.npz").replace(".npz", ".json"), "w"), indent=1)
print("CUDA REF DONE", torch.cuda.max_memory_allocated() / 2**30, "GiB peak")

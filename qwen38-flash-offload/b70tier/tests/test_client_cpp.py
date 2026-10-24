"""N137 step 2: the engine's C++ ring client (nv2_host.cpp b70_attach / b70_forward, the code the CPU worker runs in
decode) against a live b70srv.py, on the cuda_ref cases. CPU-only container (engine image, no GPU): test engine only."""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/opt/qwen38/qwen38")
import nv2
e = nv2.ext()
E, H = 288, 4096
ref = np.load("/out/cuda_ref.npz")
meta = json.load(open("/out/cuda_ref.json"))
e.b70_test_engine(43, E, H)
keys = [L["li"] * E + x for L in meta["layers"] for x in L["experts"]]
ms = e.b70_attach(os.environ.get("QWEN38_B70_RING", "/run/local-ai/shared/b70.ring"), torch.tensor(keys, dtype=torch.int64), 900.0)
print("attach + LOAD", ms, "ms", flush=True)
rows = []
for c in meta["cases"]:
    tag = c["tag"]
    x, ids, w, y, y64 = (ref[tag + s] for s in ("_x", "_ids", "_w", "_y", "_y64"))
    t = time.perf_counter()
    yb = e.b70_forward(c["li"], torch.from_numpy(x.astype(np.float32)), torch.from_numpy(ids), torch.from_numpy(w)).double().numpy()
    dt = (time.perf_counter() - t) * 1e6
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    rows.append({"tag": tag, "b70cpp_vs_cuda_rel_l2": round(rel(yb, y.astype(np.float64)), 6), "b70cpp_vs_fp64_rel_l2": round(rel(yb, y64), 6),
                 "cuda_vs_fp64_rel_l2": c["cuda_vs_fp64_rel_l2"], "finite": bool(np.isfinite(yb).all()), "us": round(dt, 1)})
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open("/o/test_client_cpp.json", "w"), indent=1)
print("CPP CLIENT DONE worst b70_vs_cuda", max(r["b70cpp_vs_cuda_rel_l2"] for r in rows))

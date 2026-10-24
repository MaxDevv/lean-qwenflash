"""N137 step 2 client tests against a running b70srv.py, through the shared ring only (what the 3090 engine will do).

  numerics  LOAD the 48 cuda_ref experts, replay every cuda_ref case (one ring row per pick, rows added per token in
            fp32 on the host), compare with the exllamav3 CUDA output and the fp64 reference.
  latency   LOAD N keys (warm-score ranks after the 3090's VRAM set), then time post -> DONE round trips for np rows.
Env: QWEN38_B70_RING, REF (/out/cuda_ref.npz), MODE (numerics | latency | both), NKEYS (3000), WARM (stats_own_dec.json),
     VRAM_SKIP (1376), CALLS (3000), OUT (/out/test_ring.json)
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ring as RG

E, H = RG.E, RG.H
r = RG.Ring(os.environ.get("QWEN38_B70_RING", RG.DEFAULT_PATH))
res = {}
seq = [int(r.hdr[RG.W_REQ_SEQ])]
lseq = [int(r.hdr[RG.W_LOAD_SEQ])]


def wait_until(cond, timeout, what):
    t = time.perf_counter()
    while not cond():
        if time.perf_counter() - t > timeout:
            raise TimeoutError(what)
    return time.perf_counter() - t


def load(keys):
    keys = np.asarray(keys, np.int32)
    r.keys[:len(keys)] = keys
    r.hdr[RG.W_NKEYS] = len(keys)
    lseq[0] += 1
    r.hdr[RG.W_LOAD_SEQ] = lseq[0]
    dt = wait_until(lambda: int(r.hdr[RG.W_LOAD_ACK]) == lseq[0], 900, "load")
    assert int(r.hdr[RG.W_STATE]) == RG.ST_READY, ("load state", int(r.hdr[RG.W_STATE]), int(r.hdr[RG.W_ERR]))
    return dt


def call(li, ids, w, xrows):
    n = len(ids)
    r.req[0], r.req[1], r.req[2] = n, n, li
    r.ids[:n] = ids
    r.w[:n] = w
    r.x[:n] = xrows
    seq[0] += 1
    t = time.perf_counter_ns()
    r.hdr[RG.W_REQ_SEQ] = seq[0]
    s = seq[0]
    while int(r.hdr[RG.W_DONE_SEQ]) != s:
        pass
    dt = time.perf_counter_ns() - t
    e = int(r.hdr[RG.W_ERR])
    if e:
        raise RuntimeError(f"server error {e}")
    return r.out[:n].astype(np.float32), dt


mode = os.environ.get("MODE", "both")
if mode in ("numerics", "both"):
    ref = np.load(os.environ.get("REF", "/out/cuda_ref.npz"))
    meta = json.load(open(os.environ.get("REF", "/out/cuda_ref.npz").replace(".npz", ".json")))
    keys = [L["li"] * E + e for L in meta["layers"] for e in L["experts"]]
    res["numerics_load_s"] = round(load(keys), 2)
    rows = []
    for c in meta["cases"]:
        tag, li, M = c["tag"], c["li"], c["M"]
        x, ids, w, y, y64 = (ref[tag + s] for s in ("_x", "_ids", "_w", "_y", "_y64"))
        tok = np.repeat(np.arange(M), ids.shape[1])
        o, _ = call(li, ids.reshape(-1), w.reshape(-1), x[tok])
        yb = np.zeros((M, H), np.float64)
        np.add.at(yb, tok, o.astype(np.float64))
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        cos = float((yb * y).sum() / np.linalg.norm(yb) / np.linalg.norm(y))
        rows.append({"tag": tag, "M": M, "scale": c["scale"], "b70_vs_cuda_rel_l2": round(rel(yb, y.astype(np.float64)), 6),
                     "b70_vs_fp64_rel_l2": round(rel(yb, y64), 6), "cuda_vs_fp64_rel_l2": c["cuda_vs_fp64_rel_l2"],
                     "cos": round(cos, 7), "max_abs": round(float(np.abs(yb - y).max()), 5),
                     "finite": bool(np.isfinite(yb).all())})
        print(json.dumps(rows[-1]), flush=True)
    res["numerics"] = rows
    res["numerics_worst_b70_vs_cuda"] = max(x["b70_vs_cuda_rel_l2"] for x in rows)

if mode in ("latency", "both"):
    nk = int(os.environ.get("NKEYS", "3000"))
    warm = json.load(open(os.environ.get("WARM", "/w/stats_own_dec.json")))
    mf = json.load(open(os.environ.get("STORE_JSON", "/g/qwen38_flash_exl3_3.05bpw_experts.json")))
    li_of = {L["key"].replace("model.language_model.", ""): L["li"] for L in mf["layers"]}
    sc = np.full(mf["n_layers"] * E, -1.0)
    for k, v in warm.items():
        kk = k.replace("model.language_model.", "")
        if kk in li_of and isinstance(v, list):
            li = li_of[kk]
            sc[li * E: li * E + E] = np.asarray(v[:E], np.float64)
    order = np.argsort(-sc, kind="stable")
    order = order[sc[order] >= 0]
    skip = int(os.environ.get("VRAM_SKIP", "1376"))
    keys = order[skip:skip + nk]
    res["latency_load_s"] = round(load(keys), 2)
    res["latency_nkeys"] = int(len(keys))
    by_li = {}
    for k in keys.tolist():
        by_li.setdefault(k // E, []).append(k % E)
    lis = sorted(l for l in by_li if len(by_li[l]) >= 16)
    rng = np.random.default_rng(0)
    xr = (rng.standard_normal((RG.MAXP, H)) * 0.5).astype(np.float16)
    calls = int(os.environ.get("CALLS", "3000"))
    lat = {}
    for n in (1, 2, 3, 4, 8, 16, 32):
        ts = []
        for c in range(calls if n <= 4 else calls // 3):
            li = lis[c % len(lis)]
            ids = rng.choice(by_li[li], n, replace=n > len(by_li[li]))
            _, dt = call(li, ids.astype(np.int32), np.full(n, 0.3, np.float32), xr[:n])
            ts.append(dt / 1e3)
        ts = np.asarray(ts[50:])
        lat[n] = {"us_p50": round(float(np.percentile(ts, 50)), 1), "us_p90": round(float(np.percentile(ts, 90)), 1),
                  "us_p99": round(float(np.percentile(ts, 99)), 1), "n": int(len(ts))}
        print(json.dumps({"rows": n, **lat[n]}), flush=True)
    res["latency_rtt"] = lat
json.dump(res, open(os.environ.get("OUT", "/out/test_ring.json"), "w"), indent=1)
print("TEST DONE", json.dumps({k: v for k, v in res.items() if k not in ("numerics",)}))

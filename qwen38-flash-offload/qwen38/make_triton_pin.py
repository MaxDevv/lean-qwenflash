#!/usr/bin/env python3
"""Build a v2 triton_pin JSON (qwen38/triton_pin.py) from a Triton cache dir: every *.autotune.json -> (kernel, key, pick).
Pick = what Triton itself selects from the file (min over configs_timings by timing list). Default per kernel (for keys
not in the cache) = the pick of the largest recorded key (most like long prefill chunks).
  python3 qwen38/make_triton_pin.py <TRITON_CACHE_DIR> data/triton_pin.json"""
import glob, json, os, sys

src, out = sys.argv[1], sys.argv[2]
pins = {}
for f in sorted(glob.glob(os.path.join(src, "*", "*.autotune.json"))):
    name = os.path.basename(f)[:-len(".autotune.json")]
    d = json.load(open(f))
    best = min(d["configs_timings"], key=lambda ct: ct[1])[0]
    cfg = {"kwargs": best["kwargs"], "num_warps": best["num_warps"], "num_stages": best["num_stages"]}
    pins.setdefault(name, {"keys": []})["keys"].append([d["key"], cfg])


def size(key):
    return sum(k for k in key if isinstance(k, int) and not isinstance(k, bool))


for name, p in pins.items():
    p["keys"].sort(key=lambda kc: size(kc[0]))
    p["default"] = p["keys"][-1][1]
json.dump(pins, open(out, "w"), indent=1)
for name, p in sorted(pins.items()):
    print(name, [(k[:3], c["kwargs"], c["num_warps"], c["num_stages"]) for k, c in p["keys"]])

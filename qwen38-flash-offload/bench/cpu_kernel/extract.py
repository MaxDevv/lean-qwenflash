# N135: extract real Qwen3.8-Flash-Next routed experts from the EXL3 checkpoint into engine RAM-slot records
# (gate/up/down trellis native layout, then fp16 suh/svh of g,u,d = the nv2 store record format).
# usage: extract.py OUT.bin [layers=0,10,20,30,41] [per=72]
import json, os, struct, sys
M = os.path.expanduser("~/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw")
out = sys.argv[1]
layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,10,20,30,41").split(",")]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 72
idx = json.load(open(M + "/model.safetensors.index.json"))["weight_map"]
hdr = {}
def tensor(name):
    f = idx[name]
    if f not in hdr:
        fh = open(M + "/" + f, "rb"); n = struct.unpack("<Q", fh.read(8))[0]; hdr[f] = (fh, 8 + n, json.loads(fh.read(n)))
    fh, base, h = hdr[f]; m = h[name]; a, b = m["data_offsets"]
    fh.seek(base + a); return fh.read(b - a), m["dtype"], m["shape"]
REC = 9474048
keys = []
with open(out, "wb") as o:
    for li in layers:
        for e in range(per):
            p = f"model.language_model.layers.{li + 3}.mlp.experts.{e}."
            tr = b""; sc = b""
            for pj in ("gate_proj", "up_proj", "down_proj"):
                t, dt, sh = tensor(p + pj + ".trellis"); assert dt == "I16" and len(t) == 3145728, (dt, sh); tr += t
            for pj in ("gate_proj", "up_proj", "down_proj"):
                for s in ("suh", "svh"):
                    t, dt, sh = tensor(p + pj + "." + s); assert dt == "F16", dt; sc += t
            rec = tr + sc; assert len(rec) == REC, len(rec); o.write(rec); keys.append(li * 288 + e)
json.dump({"record_bytes": REC, "keys": keys, "layers": layers, "per": per, "src": M}, open(out + ".json", "w"))
print("wrote", len(keys), "records")

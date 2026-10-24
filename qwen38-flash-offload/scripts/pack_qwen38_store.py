#!/usr/bin/env python3
"""Pack Qwen3.8-Flash-Next EXL3 routed experts into one O_DIRECT-friendly store on NVMe (QWEN38_MODE=nvme / nvme-exact).
(Campaign N116; the measured store sits on a 4-drive NVMe RAID0 at /mnt/nvx/qwen38.)

Python stdlib only, CPU only. Reads the checkpoint read-only (safetensors headers + pread), writes only under --out-dir.

Store layout (little-endian, every record 4096-aligned, layer-major):
  record index r = li * E + e     li = 0..L-1 over the MoE layers in model order (model layers 3..44), then the MTP
                                  layer (model layer 45, mtp.safetensors) as li = 42 when --mtp is on
  record bytes  [gate.trellis][up.trellis][down.trellis] [gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh] [pad]
  The trellis part (first `trellis_bytes` = 9,437,184 B for 3.05bpw) is byte-identical to one VRAM slot of
  expert_cache.py (slot = gate | up @ off_u | down @ off_d, each 256-aligned; here all three are 3,145,728 B so
  off_u = 3,145,728, off_d = 6,291,456 and the slot has no padding), so an NVMe -> RAM slot -> VRAM slot fill is one
  contiguous copy and the pointer tables become (a, a + off_u, a + off_d). A reader that keeps suh/svh resident reads
  only the trellis part (also 4096-aligned).
  The .mul1 codebook markers (one I32 scalar per matrix) are checked to be identical and recorded in the manifest.

Outputs in --out-dir: <name>.bin (records), <name>.json (layout + per-record sha256 of the source bytes).

Modes:
  pack            write the store (O_DIRECT, preallocated), hash every record from the SOURCE buffer before the write
  verify          O_DIRECT re-read of every record, sha256 vs the manifest (independent of the page cache)
  cmp             byte-exact compare of --sample N random records (+ first/last of every layer) against fresh source reads
"""
import argparse, concurrent.futures as cf, hashlib, json, mmap, os, random, re, struct, sys, time

ALIGN = 4096
MATS = ("gate_proj", "up_proj", "down_proj")


def al(x, a=ALIGN):
    return (x + a - 1) // a * a


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


class Src:
    """Tensor locator over the checkpoint shards (read-only)."""

    def __init__(self, model_dir, use_mtp):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        files = sorted(set(idx.values()))
        if use_mtp and os.path.exists(os.path.join(model_dir, "mtp.safetensors")):
            files.append("mtp.safetensors")
        self.loc = {}
        self.fds = {}
        for fn in files:
            base, h = st_header(os.path.join(model_dir, fn))
            for k, v in h.items():
                if ".mlp.experts." not in k or (not use_mtp and k.startswith("mtp.")):
                    continue
                a, b = v["data_offsets"]
                self.loc[k] = (fn, base + a, b - a, v["dtype"], tuple(v["shape"]))
            self.fds[fn] = os.open(os.path.join(model_dir, fn), os.O_RDONLY)
        layers = {}
        for k in self.loc:
            m = re.match(r"(.*\.layers\.(\d+)\.mlp)\.experts\.(\d+)\.", k)
            assert m is not None, f"Unrecognized expert key: {k}"
            order = (int(k.startswith("mtp.")), int(m.group(2)))
            layers.setdefault(order, [m.group(1), set()])[1].add(int(m.group(3)))
        self.layers = [(l[1], layers[l][0], len(layers[l][1])) for l in sorted(layers)]

    def read(self, key, mv):
        fn, off, n, _, _ = self.loc[key]
        got = 0
        while got < n:
            r = os.preadv(self.fds[fn], [mv[got:n]], off + got)
            assert r > 0, (key, got, n)
            got += r
        return n


def layout(src):
    """Field list (key suffix, offset, bytes) for one record, from layer-0 expert-0 shapes (asserted for all)."""
    _, pfx, _ = src.layers[0]
    fields, off = [], 0
    for m in MATS:
        k = f"{pfx}.experts.0.{m}.trellis"
        n = src.loc[k][2]
        fields.append((f"{m}.trellis", off, n, src.loc[k][3], src.loc[k][4])); off += n
    trellis = off
    assert trellis % ALIGN == 0, trellis
    for m in MATS:
        for s in ("suh", "svh"):
            k = f"{pfx}.experts.0.{m}.{s}"
            n = src.loc[k][2]
            fields.append((f"{m}.{s}", off, n, src.loc[k][3], src.loc[k][4])); off += n
    return fields, trellis, al(off)


def check_uniform(src, fields, n_layers):
    mul1 = set()
    for li in range(n_layers):
        _, pfx, E = src.layers[li]
        for e in range(E):
            for name, _, n, dt, shp in fields:
                v = src.loc[f"{pfx}.experts.{e}.{name}"]
                assert v[2] == n and v[3] == dt and v[4] == shp, (pfx, e, name, v, n, dt, shp)
            for m in MATS:
                k = f"{pfx}.experts.{e}.{m}.mul1"
                if k in src.loc:
                    fn, off, n, dt, _ = src.loc[k]
                    mul1.add(os.pread(src.fds[fn], n, off).hex())
    return sorted(mul1)


def fill_record(src, pfx, e, fields, buf):
    mv = memoryview(buf)
    for name, off, n, _, _ in fields:
        src.read(f"{pfx}.experts.{e}.{name}", mv[off:off + n])
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pack", "verify", "cmp"])
    ap.add_argument("--model", default=os.environ.get("QWEN38_MODEL_DIR", "/models"), help="EXL3 3.05bpw checkpoint directory")
    ap.add_argument("--out-dir", default=os.environ.get("QWEN38_NV_STORE_DIR", "/nvx"), help="store directory (NVMe, O_DIRECT capable)")
    ap.add_argument("--name", default="qwen38_flash_exl3_3.05bpw_experts")
    ap.add_argument("--mtp", type=int, default=1, help="append the MTP layer's 288 experts as the last layer")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=116)
    a = ap.parse_args()
    out_bin = os.path.join(a.out_dir, a.name + ".bin")
    out_meta = os.path.join(a.out_dir, a.name + ".json")
    assert os.path.isdir(a.out_dir), f"--out-dir {a.out_dir} does not exist"
    assert os.path.realpath(a.out_dir) != os.path.realpath(a.model), "--out-dir must not be the checkpoint directory"
    t0 = time.time()
    src = Src(a.model, bool(a.mtp))

    if a.mode == "pack":
        fields, trellis, rec = layout(src)
        L = len(src.layers)
        E = src.layers[0][2]
        assert all(x[2] == E for x in src.layers)
        mul1 = check_uniform(src, fields, L)
        total = L * E * rec
        os.makedirs(a.out_dir, exist_ok=True)
        st = os.statvfs(a.out_dir)
        free = st.f_bavail * st.f_frsize
        print(f"layers {L} ({src.layers[0][0]}..{src.layers[-1][0]}), E {E}, rec {rec} B (trellis {trellis}), total "
              f"{total / 1e9:.2f} GB, free {free / 1e12:.2f} TB, mul1 markers {mul1}", flush=True)
        assert total < free * 0.9, "not enough free space"
        fd = os.open(out_bin, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
        os.posix_fallocate(fd, 0, total)
        hashes = [None] * (L * E)
        import threading
        tl = threading.local()

        def job(r):
            li, e = divmod(r, E)
            if not hasattr(tl, "buf"):
                tl.buf = mmap.mmap(-1, rec)          # page-aligned (O_DIRECT source)
            buf = tl.buf
            mv = fill_record(src, src.layers[li][1], e, fields, buf)
            if rec > fields[-1][1] + fields[-1][2]:
                pad0 = fields[-1][1] + fields[-1][2]
                mv[pad0:rec] = bytes(rec - pad0)
            h = hashlib.sha256(mv).hexdigest()
            n = os.pwrite(fd, mv, r * rec)
            assert n == rec, (r, n)
            hashes[r] = h
            return r

        done = 0
        with cf.ThreadPoolExecutor(a.threads) as ex:
            for li in range(L):
                tl0 = time.time()
                list(ex.map(job, range(li * E, (li + 1) * E)))
                done += E
                dt = time.time() - tl0
                print(f"  layer {li:2d} (model {src.layers[li][0]}) {E} records {E * rec / 1e9:.2f} GB in {dt:.1f} s "
                      f"({E * rec / 1e9 / max(dt, 1e-6):.2f} GB/s), total {done * rec / 1e9:.1f} GB, {time.time() - t0:.0f} s",
                      flush=True)
        os.fsync(fd); os.close(fd)
        meta = dict(format="qwen38-expert-store-v1", model_dir=a.model, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    record_bytes=rec, trellis_bytes=trellis, align=ALIGN, experts_per_layer=E, n_layers=L,
                    order="record = li * experts_per_layer + e (layer-major)",
                    layers=[dict(li=i, model_layer=l, key=p) for i, (l, p, _) in enumerate(src.layers)],
                    mtp_layer_li=(L - 1 if a.mtp and (src.layers[-1][0] == 45 or src.layers[-1][1].startswith("mtp.")) else None),
                    fields=[dict(name=n, offset=o, bytes=b, dtype=d, shape=list(s)) for n, o, b, d, s in fields],
                    vram_slot=dict(off_u=fields[1][1], off_d=fields[2][1], slot_bytes=trellis),
                    mul1_markers_hex=mul1, sha256=hashes, total_bytes=total)
        tmp = out_meta + ".tmp"
        json.dump(meta, open(tmp, "w"))
        os.replace(tmp, out_meta)
        print(f"PACKED {L * E} records, {total / 1e9:.2f} GB in {time.time() - t0:.0f} s -> {out_bin}", flush=True)
        return

    meta = json.load(open(out_meta))
    rec, E, L = meta["record_bytes"], meta["experts_per_layer"], meta["n_layers"]
    assert os.path.getsize(out_bin) == meta["total_bytes"] == rec * E * L
    fd = os.open(out_bin, os.O_RDONLY | os.O_DIRECT)
    import threading
    tl = threading.local()

    def rd(r):
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, rec)
        n = os.preadv(fd, [tl.buf], r * rec)
        assert n == rec, (r, n)
        return tl.buf

    if a.mode == "verify":
        bad = []

        def job(r):
            h = hashlib.sha256(rd(r)).hexdigest()
            if h != meta["sha256"][r]:
                bad.append(r)

        with cf.ThreadPoolExecutor(a.threads) as ex:
            for li in range(L):
                list(ex.map(job, range(li * E, (li + 1) * E)))
        dt = time.time() - t0
        print(f"VERIFY {'OK' if not bad else 'FAIL'}: {L * E} records, {L * E * rec / 1e9:.2f} GB O_DIRECT re-read + sha256 "
              f"in {dt:.0f} s ({L * E * rec / 1e9 / dt:.2f} GB/s); bad {len(bad)} {bad[:20]}", flush=True)
        sys.exit(1 if bad else 0)

    # cmp: byte-exact against fresh source reads (different code path from the packer's buffer reuse)
    fields = [(f["name"], f["offset"], f["bytes"]) for f in meta["fields"]]
    rng = random.Random(a.seed)
    pick = set(rng.sample(range(L * E), min(a.sample, L * E)))
    for li in range(L):
        pick.update((li * E, li * E + E - 1))
    pick = sorted(pick)
    bad = []
    for r in pick:
        li, e = divmod(r, E)
        pfx = meta["layers"][li]["key"]
        got = rd(r)
        for name, off, n in fields:
            fn, soff, sn, _, _ = src.loc[f"{pfx}.experts.{e}.{name}"]
            assert sn == n
            want = os.pread(src.fds[fn], n, soff)
            if got[off:off + n] != want:
                bad.append((r, name))
        tail = fields[-1][1] + fields[-1][2]
        if any(got[tail:rec]):
            bad.append((r, "pad"))
    print(f"CMP {'OK' if not bad else 'FAIL'}: {len(pick)} records byte-compared against the checkpoint (every field + "
          f"zero pad), {time.time() - t0:.0f} s; mismatches {len(bad)} {bad[:20]}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

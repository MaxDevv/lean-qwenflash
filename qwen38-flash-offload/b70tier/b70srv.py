"""N137 B70 expert server: a persistent process on one Arc Pro B70 holding a static set of Qwen3.8-Flash-Next experts.

It spins on the shared ring (ring.py). For a LOAD it reads the keys' raw records from the NVMe store (O_DIRECT),
copies each into VRAM staging, permutes it into exl3xpu blob order (glmtier.make_perm) and points the layer's
pointer table at the slot. For a request it H2Ds the pick rows, runs exl3xpu moe_forward (_moe_n128.so, swiglu clamp
10) with one expert per row, D2Hs the weighted rows and publishes DONE_SEQ.

It creates the ring file at startup, so it starts first; the 3090 engine (QWEN38_B70=1) attaches to it, sends the key
set to load (LOAD), then one request per decode MoE layer call that has B70 picks.
Env: QWEN38_B70_RING (/run/local-ai/shared/b70.ring), QWEN38_NV_STORE (/nvx/qwen38_flash_exl3_3.05bpw_experts.bin, the
     same packed store the engine reads), QWEN38_B70_SPIN_CPU (pin the polling thread; unset = no pin), QWEN38_B70_COPY
     (sysptr | usm | torch), QWEN38_B70_LOG (json stats path), QWEN38_B70_PROF, QWEN38_B70_READERS (8), QWEN38_B70_MOE_LIB
     (default b70tier/kernels/moe-n128/_moe_n128.so; overrides the base image's EXL3_MOE_LIB), GU/DN splits (8/4)
"""
import concurrent.futures as cf, ctypes, json, mmap, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.environ.get("EXL3XPU_PATH", "/opt/trellis-serve/xpu"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the base image sets EXL3_MOE_LIB to its stock _moe_sgl.so (no swiglu clamp): always point it at the GLM build
os.environ["EXL3_MOE_LIB"] = os.environ.get("QWEN38_B70_MOE_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             "kernels", "moe-n128", "_moe_n128.so"))
from exl3xpu import moe_offload
from exl3xpu.moe_offload import s64
from qwenperm import make_perm
import ring as RG

H, I, K, E = 4096, 2048, 3, 288
libc = ctypes.CDLL(None, use_errno=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), "b70srv:", *a, flush=True)


class Server:
    def __init__(self):
        self.r = RG.Ring(os.environ.get("QWEN38_B70_RING", RG.DEFAULT_PATH))
        self.dev = torch.device("xpu", 0)
        self.X = moe_offload.ops()
        self.X.moe_set_splits(int(os.environ.get("GU", "8")), int(os.environ.get("DN", "4")))
        self.X.moe_set_prefill_min_m(1 << 30)
        self.blob = int(self.X.blob_bytes(H, I, K))
        store = os.environ.get("QWEN38_NV_STORE", "/nvx/qwen38_flash_exl3_3.05bpw_experts.bin")
        meta = json.load(open(store.rsplit(".", 1)[0] + ".json"))
        self.rec = int(meta["record_bytes"])
        assert self.rec == self.blob, (self.rec, self.blob)
        self.L = int(meta["n_layers"])
        self.fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
        self.perm = make_perm(H, I, K).to(self.dev)
        self.copy_mode = os.environ.get("QWEN38_B70_COPY", "sysptr")
        self.slots = None
        self.slot_of = np.full(self.L * E, -1, np.int64)
        self.ptrs = torch.zeros(self.L * E, dtype=torch.int64, device=self.dev)
        # device request buffer: ids int32[MAXP] | w f32[MAXP] | x fp16[MAXP, H] (same order as the ring's REQ/X)
        # one device copy of the ring's REQ page + X rows (single H2D per request)
        self.d_req = torch.zeros(4096 + RG.MAXP * H * 2, dtype=torch.uint8, device=self.dev)
        self.d_ids = self.d_req[16:16 + RG.MAXP * 4].view(torch.int32)
        self.d_w = self.d_req[16 + RG.MAXP * 4:16 + RG.MAXP * 8].view(torch.float32)
        self.d_x = self.d_req[4096:].view(torch.float16).view(RG.MAXP, H)
        self.req_addr = self.r.base + RG.OFF_REQ
        # N137 COPY=usm: bounce through pinned USM host buffers (CPU memcpy ring <-> USM, USM <-> device); sysptr
        # copies from the shared tmpfs ring cost 45-78 us per submit (measured), USM submits ~7 us
        if self.copy_mode == "usm":
            self.h_req = torch.empty(4096 + RG.MAXP * H * 2, dtype=torch.uint8, pin_memory=True)
            self.h_out = torch.empty(RG.MAXP * H * 2, dtype=torch.uint8, pin_memory=True)
            self.h_req_np, self.h_out_np = self.h_req.numpy(), self.h_out.numpy()
            self.ring_req_np = self.r.buf[RG.OFF_REQ:RG.OFF_REQ + 4096 + RG.MAXP * H * 2]
            self.ring_out_np = self.r.buf[RG.OFF_OUT:RG.OFF_OUT + RG.MAXP * H * 2]
        self.ids_addr = self.r.base + RG.OFF_REQ + 16
        self.w_addr = self.ids_addr + RG.MAXP * 4
        self.x_addr = self.r.base + RG.OFF_X
        self.out_addr = self.r.base + RG.OFF_OUT
        self.t = dict(calls=0, seen_to_done_ns=[], kern_ns=[], np=[])
        self.logp = os.environ.get("QWEN38_B70_LOG", "")
        self.prof = [] if os.environ.get("QWEN38_B70_PROF") else None
        p = torch.xpu.get_device_properties(self.dev)
        log(f"device {p.name} total {p.total_memory / 2**30:.2f} GiB, blob {self.blob}, store layers {self.L}, "
            f"copy {self.copy_mode}")

    # ---------------------------------------------------------------- LOAD
    def load(self, keys):
        n = len(keys)
        t0 = time.perf_counter()
        self.slots = None
        torch.xpu.empty_cache()
        self.slots = torch.empty((n, self.blob), dtype=torch.uint8, device=self.dev)
        R = 16
        stage = torch.empty((R, self.blob), dtype=torch.uint8, device=self.dev)
        sbase = stage.data_ptr()
        nbuf = 2 * R
        host = mmap.mmap(-1, nbuf * self.rec, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        hbase = ctypes.addressof(ctypes.c_char.from_buffer(host))
        libc.madvise(ctypes.c_void_p(hbase), ctypes.c_size_t(nbuf * self.rec), 14)

        def rd(key, b):
            mv = memoryview((ctypes.c_char * self.rec).from_address(hbase + b * self.rec)).cast("B")
            got = 0
            while got < self.rec:
                k = os.preadv(self.fd, [mv[got:]], key * self.rec + got)
                if k <= 0:
                    raise IOError(f"short read key {key}")
                got += k
            return b

        pool = cf.ThreadPoolExecutor(int(os.environ.get("QWEN38_B70_READERS", "8")))
        self.slot_of[:] = -1
        base = s64(self.slots.data_ptr())
        for i in range(0, n, R):
            chunk = keys[i:i + R]
            half = (i // R) % 2
            futs = [pool.submit(rd, int(k), half * R + j) for j, k in enumerate(chunk)]
            for f in futs:
                f.result()
            torch.xpu.synchronize()        # previous chunk's copies from the other half are done (ordering only)
            for j in range(len(chunk)):
                self.X.memcpy_async(s64(sbase + j * self.rec), s64(hbase + (half * R + j) * self.rec), self.rec)
            m = len(chunk)
            d = torch.arange(i, i + m, dtype=torch.int64, device=self.dev)
            self.slots.view(torch.int32).index_copy_(0, d, stage[:m].view(torch.int32).index_select(1, self.perm))
            for j, k in enumerate(chunk):
                self.slot_of[int(k)] = i + j
        torch.xpu.synchronize()
        pool.shutdown()
        # pointer table: every key points at slot 0 unless resident (the server never runs a non-resident id)
        pt = np.full(self.L * E, base, np.int64)
        res = self.slot_of >= 0
        pt[res] = base + self.slot_of[res] * self.blob
        self.ptrs.copy_(torch.from_numpy(pt))
        torch.xpu.synchronize()
        # spot check: slot bytes == permuted record (first, middle, last)
        bad = 0
        for j in sorted({0, n // 2, n - 1}):
            raw = torch.frombuffer(bytearray(os.pread(os.open(os.environ.get("QWEN38_NV_STORE", "/nvx/qwen38_flash_exl3_3.05bpw_experts.bin"), os.O_RDONLY), self.rec, int(keys[j]) * self.rec)), dtype=torch.uint8)
            want = raw.view(torch.int32)[self.perm.cpu()].contiguous()
            got = self.slots[j].view(torch.int32).cpu()
            bad += int(not torch.equal(want, got))
        host.close()
        dt = time.perf_counter() - t0
        # warm every row count once (first moe_forward per shape JIT-compiles, measured up to 1.5 s on the 3090 path)
        tw = time.perf_counter()
        li0 = int(keys[0]) // E
        e0 = int(keys[0]) % E
        for m in range(1, RG.MAXP + 1):
            self.X.moe_forward(self.d_x[:m], torch.full((m, 1), e0, dtype=torch.int32, device=self.dev),
                               torch.full((m, 1), 0.1, dtype=torch.float32, device=self.dev),
                               self.ptrs[li0 * E:(li0 + 1) * E], I, K, E)
        torch.xpu.synchronize()
        log(f"warm-up of 1..{RG.MAXP} rows in {time.perf_counter() - tw:.1f} s")
        log(f"LOAD {n} experts ({n * self.blob / 1e9:.2f} GB) in {dt:.1f} s ({n * self.rec / dt / 1e9:.2f} GB/s), "
            f"spot check bad {bad}, mem allocated {torch.xpu.memory_allocated(self.dev) / 2**30:.2f} GiB")
        return bad

    # ---------------------------------------------------------------- one request
    def serve(self, s):
        r = self.r
        t0 = time.perf_counter_ns()
        n, ntok, li = int(r.req[0]), int(r.req[1]), int(r.req[2])
        if not (0 < n <= RG.MAXP) or not (0 <= li < self.L):
            r.hdr[RG.W_ERR] = 10
            return
        ids = r.ids[:n]
        P = self.prof
        if P is not None: ta = time.perf_counter_ns()
        if (ids < 0).any() or (ids >= E).any() or (self.slot_of[li * E + ids.astype(np.int64)] < 0).any():
            r.hdr[RG.W_ERR] = 11           # non-resident id: refuse (the kernel would read a wrong expert)
            return
        if self.copy_mode == "usm":
            nb = 4096 + n * H * 2
            self.h_req_np[:nb] = self.ring_req_np[:nb]
            self.X.memcpy_async(s64(self.d_req.data_ptr()), s64(self.h_req.data_ptr()), nb)
            if P is not None: tb = time.perf_counter_ns()
        elif self.copy_mode == "sysptr":
            self.X.memcpy_async(s64(self.d_req.data_ptr()), s64(self.req_addr), 4096 + n * H * 2)
            if P is not None: tb = time.perf_counter_ns()
        else:
            self.d_ids[:n].copy_(torch.from_numpy(ids.copy()))
            self.d_w[:n].copy_(torch.from_numpy(r.w[:n].copy()))
            self.d_x[:n].copy_(torch.from_numpy(r.x[:n]))
        out = self.X.moe_forward(self.d_x[:n], self.d_ids[:n].view(n, 1), self.d_w[:n].view(n, 1),
                                 self.ptrs[li * E:(li + 1) * E], I, K, E)
        if P is not None: tc = time.perf_counter_ns()
        if self.copy_mode == "usm":
            self.X.memcpy_async(s64(self.h_out.data_ptr()), s64(out.data_ptr()), n * H * 2)
            if P is not None: td = time.perf_counter_ns()
            torch.xpu.synchronize()
            self.ring_out_np[:n * H * 2] = self.h_out_np[:n * H * 2]
            if P is not None: P.append((ta - t0, tb - ta, tc - tb, td - tc, time.perf_counter_ns() - td))
        elif self.copy_mode == "sysptr":
            self.X.memcpy_async(s64(self.out_addr), s64(out.data_ptr()), n * H * 2)
            if P is not None: td = time.perf_counter_ns()
            torch.xpu.synchronize()
            if P is not None: P.append((ta - t0, tb - ta, tc - tb, td - tc, time.perf_counter_ns() - td))
        else:
            o = out.cpu()
            r.out[:n] = o.numpy()
        t1 = time.perf_counter_ns()
        self.t["calls"] += 1
        if len(self.t["np"]) < 200000:
            self.t["seen_to_done_ns"].append(t1 - t0)
            self.t["np"].append(n)

    def dump(self):
        if not self.logp or not self.t["np"]:
            return
        a, nn = np.asarray(self.t["seen_to_done_ns"]) / 1e3, np.asarray(self.t["np"])
        out = {"calls": self.t["calls"], "by_np": {}}
        for v in sorted(set(nn.tolist())):
            x = a[nn == v]
            out["by_np"][int(v)] = {"n": int(len(x)), "us_p50": round(float(np.percentile(x, 50)), 1),
                                    "us_p90": round(float(np.percentile(x, 90)), 1)}
        if self.prof:
            q = np.asarray(self.prof[100:]) / 1e3
            out["prof_us_p50"] = dict(zip(["parse", "checks+h2d_submit", "moe_forward_host", "d2h_submit", "sync_wait"],
                                          [round(float(v), 1) for v in np.percentile(q, 50, axis=0)]))
        json.dump(out, open(self.logp, "w"), indent=1)

    def loop(self):
        r = self.r
        r.hdr[RG.W_PID] = os.getpid()
        r.hdr[RG.W_STATE] = RG.ST_NONE
        last_req, last_load = int(r.hdr[RG.W_REQ_SEQ]), int(r.hdr[RG.W_LOAD_SEQ])
        r.hdr[RG.W_DONE_SEQ] = last_req
        hb, idle = time.time(), 0
        log("ready for LOAD / requests")
        while True:
            s = int(r.hdr[RG.W_REQ_SEQ])
            if s != last_req:
                if r.hdr[RG.W_STATE] == RG.ST_READY:
                    self.serve(s)
                else:
                    r.hdr[RG.W_ERR] = 12
                last_req = s
                r.hdr[RG.W_DONE_SEQ] = s
                idle = 0
                continue
            ls = int(r.hdr[RG.W_LOAD_SEQ])
            if ls != last_load:
                r.hdr[RG.W_STATE] = RG.ST_LOADING
                n = int(r.hdr[RG.W_NKEYS])
                try:
                    bad = self.load(np.array(r.keys[:n], dtype=np.int64))
                    r.hdr[RG.W_NSLOTS] = n
                    r.hdr[RG.W_STATE] = RG.ST_READY if bad == 0 else RG.ST_ERROR
                    if bad:
                        r.hdr[RG.W_ERR] = 20
                except Exception as ex:      # noqa: BLE001
                    log("LOAD failed:", repr(ex))
                    r.hdr[RG.W_ERR] = 21
                    r.hdr[RG.W_STATE] = RG.ST_ERROR
                last_load = ls
                r.hdr[RG.W_LOAD_ACK] = ls
                continue
            if r.hdr[RG.W_QUIT]:
                break
            idle += 1
            if idle > 2000000:
                time.sleep(0.0002)
            if (idle & 0xFFFF) == 0 and time.time() - hb > 1.0:
                hb = time.time()
                r.hdr[RG.W_HEARTBEAT] = int(hb)
                if self.t["calls"]:
                    self.dump()
        self.dump()
        log("quit", self.t["calls"], "calls")


if __name__ == "__main__":
    cpu = os.environ.get("QWEN38_B70_SPIN_CPU")
    if cpu:
        os.sched_setaffinity(0, {int(cpu)})
    rp = os.environ.get("QWEN38_B70_RING", RG.DEFAULT_PATH)
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    RG.Ring(rp, create=True)       # fresh ring (zeroed header): the engine attaches after this
    Server().loop()

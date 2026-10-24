"""N119 nv2: S2 (exact, device-side stall) and S3 (+ AVX2 CPU tier on RAM-resident experts) NVMe expert tier for the
G067 stack (QWEN38_NV=2; QWEN38_MODE=nvme2 / nvme3 in the entrypoint). Builds on nv_tier.py's loader (expert trellis
-> one VRAM bounce slot, nothing read) and expert_cache.py's VRAM CLOCK pool.

Per MoE layer call with <= QWEN38_EC_STAGE_MIN picks (decode, small forwards), no host sync:
  nv_pub   device publishes the layer's unique picks (+ the MoE input rows for a CPU job) to a mapped request ring
  host     controller thread (nv2_host.cpp) drains the device's write-back / admission logs, runs plan_layer (moetier
           semantics: lanes gpu | zerocopy (= GPU-admitted RAM expert) | cpu | nvme->gpu | nvme->cpu, greedy min-max,
           NVMe misses stall = exact), starts O_DIRECT reads into fresh RAM slots, replies with the lanes, hands the
           CPU job to the persistent CPU worker
  nv_step  device waits for the reply (bounded), masks CPU-lane picks (id -1, weight 0), CLOCK admission of GPU-lane
           misses; VRAM victims get a fresh RAM landing slot (exclusive RAM tier), admissions are logged
  nv_copy  gathers admitted experts: RAM sources first, NVMe sources once their landed flag is up (per key, set by
           the host after the read completes); victim bytes go to the landing slot in the same pass
  MoE      exllamav3's kernels, unchanged, on the live pointer tables
  combine  waits for the CPU job (bounded), adds its fp32 partial
Prefill (>= stage_min picks): staging buffers filled on the copy stream from RAM slots (pinned for the forward) and,
for NVMe-only experts, from a pinned FIFO ring that a reader fills several layers ahead (QWEN38_NV_PF_RING slots).
Env: QWEN38_NV_RAM_GB (auto), QWEN38_NV_MARGIN_GB (3), QWEN38_NV_THREADS (16 readers), QWEN38_NV_PIECE_KB (2304),
     QWEN38_NV_PF_RING (192), QWEN38_NV_PF_QD (24), QWEN38_NV_EXCL (1), QWEN38_NV_ELASTIC_WB (1), QWEN38_NV_LAND (64),
     QWEN38_NV_FREE_LO (96), QWEN38_NV_CPU (0: S2; 1: S3 CPU tier), QWEN38_NV_NVCPU (0), QWEN38_NV_CPU_CPUS (2-23),
     QWEN38_NV_CPU_MODE (-1 auto), QWEN38_NV_CLAMP (1: swiglu clamp on the CPU share, B004 fix), QWEN38_NV_POL (cost model
     "g0,thit,tzc,ca,cb,ctok,push,nvlat,nvdeep,nvone"), QWEN38_NV_MAXCPU (32), QWEN38_NV_CTL_CPU (25),
     QWEN38_NV_READER_CPUS (26-39), QWEN38_NV_TIMEOUT_S (20), QWEN38_NV_COPY_GRID (48)
N137 B70 expert tier (needs the CPU tier, QWEN38_NV_CPU=1): QWEN38_B70 (0), QWEN38_B70_RING (/run/local-ai/shared/b70.ring,
     created by b70tier/b70srv.py on the B70), QWEN38_B70_N (3000 experts), QWEN38_B70_ORDER (score: warm-score order after the VRAM warm
     set | rr: the RAM tier's round-robin rank), QWEN38_B70_TIMEOUT_S (2). Decode picks of B70-resident experts are masked
     on the 3090 and computed on the B70; their RAM copies are dropped (exclusive) and RAM refills with colder experts.
"""
import collections, concurrent.futures as cf, ctypes, json, mmap, os, signal, threading, time
import numpy as np
import torch
import nv_tier as NT

GiB = 1024 ** 3
_EXT = None
NV = None


def _cpus(spec):
    out = []
    for part in spec.split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        here = os.path.join(root, "kernels", "nv2")   # nv2_host.cpp, nv2_dev.cu, nv2_shared.h, ft_core.h
        d = os.environ.get("QWEN38_NV2_BUILD", os.path.join(root, "build", "nv2"))
        os.makedirs(d, exist_ok=True)
        _EXT = load(name="qwen38_nv2", sources=[os.path.join(here, "nv2_host.cpp"), os.path.join(here, "nv2_dev.cu")],
                    build_directory=d, extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-mtune=znver3", "-std=c++17",
                                                     "-I" + here],
                    extra_cuda_cflags=["-O3", "-I" + here], extra_ldflags=["-lpthread"], verbose=False)
    return _EXT


def _mapped(nbytes, dev_idx, tag):
    """One page-aligned anon region, cudaHostRegister(Mapped); UVA alias == host address (checked)."""
    from exllamav3.ext import exllamav3_ext as xe
    nbytes = (nbytes + 4095) // 4096 * 4096
    mm = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    with torch.cuda.device(dev_idx):
        r = torch.cuda.cudart().cudaHostRegister(a, nbytes, 2)
    assert int(r) == 0, f"cudaHostRegister failed ({tag}): {r}"
    t = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes)
    assert xe.pinned_cuda_view(t[:4096], dev_idx).data_ptr() == a, "UVA alias != host address"
    t.zero_()
    return mm, a, t


class Nv2:
    def __init__(self, pool):
        e = ext()
        sz = e.sizes()
        self.SREQ, self.SREP, self.SLOG, self.MAXU, self.MAXP, self.RQ, self.LG, self.LR, self.MAXB, HCN, DCN, STN, DBG = sz
        self.pool = pool
        self.dev = pool.device
        dev_idx = self.dev.index or 0
        self.store, man = NT._manifest()
        self.rec_bytes = man["record_bytes"]
        self.slot = man["vram_slot"]["slot_bytes"]
        assert (pool.sg, pool.su, pool.sd) == tuple(man["fields"][i]["bytes"] for i in range(3)) and pool.rec == self.slot
        assert pool.off_u == man["vram_slot"]["off_u"] and pool.off_d == man["vram_slot"]["off_d"]
        assert pool.first == 0 and pool.E == man["experts_per_layer"]
        li_of = {L["key"]: L["li"] for L in man["layers"]}
        assert [li_of[m.key] for m in pool.mods] == list(range(len(pool.mods))), "pool layer order != store layer order"
        self.L, self.E = len(pool.mods), pool.E
        self.K = self.L * self.E
        # A fresh model need not ship routing statistics. Keep cold-start bookkeeping valid
        # without reading arbitrary experts just to manufacture a warm cache.
        self._rank = [li * self.E + ex for ex in range(self.E) for li in range(self.L)]
        m0 = pool.mods[0]
        self.H, self.I = m0.hidden_size, m0.intermediate_size
        self.act_limit = float(getattr(m0, "act_limit", 0.0) or 0.0)
        g = lambda k, d: os.environ.get(k, d)
        self.cpu_on = g("QWEN38_NV_CPU", "0") == "1"
        self.b70 = g("QWEN38_B70", "0") == "1"
        assert not self.b70 or self.cpu_on, "QWEN38_B70=1 needs the CPU tier (QWEN38_NV_CPU=1): the CPU worker runs the B70 lane"
        self.b70_ring = g("QWEN38_B70_RING", "/run/local-ai/shared/b70.ring")
        self.b70_timeout = float(g("QWEN38_B70_TIMEOUT_S", "2"))
        self.b70_keys = []
        self.timeout_ns = int(float(g("QWEN38_NV_TIMEOUT_S", "5")) * 1e9)
        self.grid = int(g("QWEN38_NV_COPY_GRID", "48"))
        self.wb_on = g("QWEN38_NV_EXCL", "1") == "1"
        self.prefetch = g("QWEN38_NV_PREFETCH", "0") == "1"
        # N136: the layer-ahead routing guess runs on a side stream concurrently with this layer's router (both read the
        # same MoE input; 36-block GEMVs on an 82-SM card), instead of after it on the compute stream (QWEN38_NV_PFSIDE=1;
        # single-row decode only, see predict_early)
        self.pfside = self.prefetch and g("QWEN38_NV_PFSIDE", "0") == "1"
        self._pfe = None
        if self.pfside:
            self.side = torch.cuda.Stream(self.dev)
            self._pfev = [torch.cuda.Event() for _ in range(8)]
            self._pfev_i = 0
        self._pfb = {}
        self._pfb = {}
        self.elastic_wb = self.wb_on and g("QWEN38_NV_ELASTIC_WB", "1") == "1"
        self.fd = os.open(self.store, os.O_RDONLY | os.O_DIRECT)
        self.ph_t = torch.zeros(self.slot, dtype=torch.uint8, device=self.dev)
        self.ph = self.ph_t.data_ptr()
        self.bounce_t, self.bounce = NT._BOUNCE[dev_idx]
        self.st = collections.Counter()
        # ---- CPU tier scale copies first (charged to the cap before the RAM tier is sized)
        self.scales = None
        if self.cpu_on:
            self.scales = self._copy_cpu_scales()
            NT._log(f"CPU tier scale copies: {sum(t.numel() * 2 for s in self.scales for t in s) / GiB:.2f} GiB")
        # ---- mapped control region
        H = self.H
        lay = [("hc", HCN * 8), ("req", self.RQ * self.SREQ), ("rep", self.RQ * self.SREP), ("wbl", self.LG * self.SLOG),
               ("adml", self.LG * self.SLOG), ("land", self.LR * 4), ("home", self.K * 24), ("res", self.K * 4),
               ("hx", self.MAXB * H * 2), ("hout", self.MAXB * H * 4)]
        off, self.off = 0, {}
        for k, n in lay:
            self.off[k] = off
            off += (n + 4095) // 4096 * 4096
        self.cmm, self.cbase_addr, self.ct = _mapped(off, dev_idx, "control")
        A = lambda k: self.cbase_addr + self.off[k]
        self.a = {k: A(k) for k, _ in lay}
        self.hc = self.ct[self.off["hc"]: self.off["hc"] + HCN * 8].view(torch.int64)
        self.home = self.ct[self.off["home"]: self.off["home"] + self.K * 24].view(torch.int64)
        self.res = self.ct[self.off["res"]: self.off["res"] + self.K * 4].view(torch.int32)
        # ---- RAM tier + prefill ring sizes
        ring_pf = int(g("QWEN38_NV_PF_RING", "192"))
        margin = float(g("QWEN38_NV_MARGIN_GB", "3")) * GiB
        mem = NT.cgroup_mem()
        want = g("QWEN38_NV_RAM_GB", "auto")
        if want == "auto":
            assert mem.get("max"), "QWEN38_NV_RAM_GB=auto needs a memory-capped container"
            avail = (mem["max"] - mem["current"]) * GiB - margin - ring_pf * self.slot - NT._rcs_bytes() - 0.3 * GiB
            nram = max(0, int(avail // self.slot))
        else:
            nram = int(float(want) * GiB // self.slot)
        NT._log(f"nv2: rcs reserve {NT._rcs_bytes() / GiB:.1f} GiB, margin {margin / GiB:.1f} GiB, prefill ring {ring_pf} "
                f"slots ({ring_pf * self.slot / GiB:.2f} GiB) -> RAM tier {nram} slots ({nram * self.slot / GiB:.2f} GiB)")
        self.ram = NT.HostArena(nram, self.slot, 512, dev_idx, "ram")
        self.nram = nram
        self.pf = NT.HostArena(ring_pf, self.slot, ring_pf, dev_idx, "pfring")
        NT._log(f"nv2: RAM arena + prefill ring registered")
        # VRAM victim ring (write-back source: nv_copy parks the victim's bytes here D2D, the host drains it D2H on a
        # copy engine; SM stores to host memory run at ~4 GB/s vs 25 GB/s for the DMA engine)
        nvr = int(g("QWEN38_NV_VRING", "32")) if self.wb_on else 0
        self.vring = torch.empty((max(nvr, 1), self.slot), dtype=torch.uint8, device=self.dev) if nvr else None
        self.vring_addr = [self.vring[i].data_ptr() for i in range(nvr)] if nvr else []
        self.ram_addr_dev = torch.tensor(self.vring_addr or [0], dtype=torch.int64, device=self.dev)
        # ---- device scratch
        D = lambda n, dt=torch.int32: torch.zeros(n, dtype=dt, device=self.dev)
        self.dc, self.uidx, self.uexp, self.ulane = D(DCN, torch.int64), D(self.E), D(self.MAXU), D(self.MAXU)
        self.jobx, self.jobnv = D(pool.jobs.numel() // 2), D(pool.jobs.numel() // 2)
        self.dflag, self.dstats = D(1, torch.int64), D(STN, torch.int64)
        self.dbg = D(DBG * 8, torch.int64)
        self.sel_out = torch.empty(16384, dtype=torch.int64, device=self.dev)
        self.w_out = torch.empty(16384, dtype=torch.half, device=self.dev)
        # ---- scores (tie-break: colder first to the CPU)
        sp = g("QWEN38_EC_WARM", "")
        self.scores = json.load(open(sp)) if sp else {}
        sc = np.zeros(self.K, np.float32)
        for li, m in enumerate(pool.mods):
            s = self.scores.get(m.key)
            if s is not None:
                sc[li * self.E:(li + 1) * self.E] = np.asarray(s[:self.E], np.float32)
        # ---- host engine
        rcpus = _cpus(g("QWEN38_NV_READER_CPUS", "26-39"))
        e.init(self.L, self.E, H, self.I, self.rec_bytes, self.slot, int(g("QWEN38_NV_PIECE_KB", "2304")) * 1024,
               pool.off_u, pool.off_d, self.fd, self.ph, self.a["hc"], self.a["req"], self.a["rep"], self.a["wbl"],
               self.a["adml"], self.a["land"], self.a["home"], self.a["res"], self.a["hx"], self.a["hout"],
               torch.tensor(self.ram.addr.tolist(), dtype=torch.int64), torch.tensor(self.pf.addr.tolist(), dtype=torch.int64),
               torch.from_numpy(sc), int(g("QWEN38_NV_THREADS", "16")), rcpus, int(g("QWEN38_NV_CTL_CPU", "25")),
               int(self.wb_on), int(g("QWEN38_NV_LAND", "64")), int(g("QWEN38_NV_FREE_LO", "96")),
               torch.tensor(self.vring_addr, dtype=torch.int64))
        e.set_stage_qd(int(g("QWEN38_NV_PF_QD", "24")))
        if g('QWEN_STAGE_RAM_ADMIT','0')=='1': e.set_stage_cache(True)
        pol = [float(x) for x in g("QWEN38_NV_POL", "0.17,0.023,0.38,0.145,0.165,0.040,0.40,0.12,0.395,0.75").split(",")]
        self.pol = pol + [1.0 if self.cpu_on else 0.0, 1.0 if g("QWEN38_NV_NVCPU", "1" if self.cpu_on else "0") == "1" else 0.0,
                          float(g("QWEN38_NV_MAXCPU", "32")), 1.0 if g("QWEN38_NV_NOADMIT", "1" if self.cpu_on else "0") == "1" else 0.0]
        e.set_pol(self.pol)
        # ---- the pool reads homes from the mapped table from now on (ec_step / ec_copy / ec_evict take its address)
        pool.home = self.home.view(self.L, self.E, 3)
        pool.home_cpu = pool.home
        pool.nv2 = self
        e.nv_rows_all(self.L, self.E, pool.slotof, self.a["home"], pool.tabs)
        torch.cuda.synchronize(self.dev)
        if getattr(pool, "ev_ready", None) is not None:
            for ev in pool.ev_ready:
                ev.record(torch.cuda.current_stream(self.dev))
        self.bg = cf.ThreadPoolExecutor(1, thread_name_prefix="nv2-stage")
        e.start()
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()
        NT._log(f"nv2 engine up: {self.L} layers x {self.E}, readers {g('QWEN38_NV_THREADS', '16')} on {rcpus[:3]}.., "
                f"exclusive {self.wb_on}, CPU tier {self.cpu_on}")

    # ------------------------------------------------------------------------------------------------------------
    def _watchdog(self):
        last, quiet = None, 0
        while True:
            time.sleep(1.0)
            pub = int(self.hc[0])
            quiet = quiet + 1 if pub == last else 0
            last = pub
            if quiet == 10 and int(self.hc[0]) != int(self.hc[1]):
                print(f" !! nv2 watchdog: no progress 10 s; host diag {ext().diag()} dc {ext().peek(self.dc.data_ptr(), 8)}", flush=True)
            err = int(self.hc[2])
            if err:
                print(f" !! nv2 ERROR flag {err} (1 reply timeout, 2 ring full, 3 fix wait, 4 copy wait, 5 CPU wait, 16-19 "
                      f"read/alloc); host diag {ext().diag()} dc {ext().peek(self.dc.data_ptr(), 8)} counters {self.counters()} "
                      f"dev {ext().peek(self.dstats.data_ptr(), 16)}", flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(30)
                os._exit(70)

    def counters(self):
        names = ["reqs", "dec_reqs", "small_reqs", "picks", "vram", "ram", "nvme", "inflight_hit", "lane_zc", "lane_cpu",
                 "lane_nvg", "lane_nvc", "reads", "read_bytes", "wb_landed", "wb_dup", "adm_freed", "adm_pinned",
                 "evictions", "cpu_jobs", "cpu_experts", "cpu_tokens", "stage_ram", "stage_nv", "stage_reads", "read_err",
                 "alloc_fail", "land_pushed", "prefetch", "plan_ns", "ctl_busy_ns", "cpu_busy_ns", "cpu_wait_land_ns",
                 "read_ms_x1000", "stage_wait_ns", "stage_job_ns", "wb_bulk", "dec_tok", "notice_ns", "notice_max_ns", "serve_max_ns", "reply_ns", "wb_issued", "wb_drop", "vring_hits", "lane_b70", "b70_jobs", "b70_rows", "b70_wait_ns", "b70_rtt_ns", "b70_err", "b70_rtt_max_ns",
                 "resident", "inflight", "free", "landing"]
        return dict(zip(names, ext().counters()))

    # ---- warm start: RAM by score (incl. the VRAM warm set), VRAM warm, exclusive: drop VRAM copies, refill RAM
    def warm(self, scores):
        p = self.pool
        order = []
        for li, m in enumerate(p.mods):
            sc = scores.get(m.key)
            o = sorted(range(self.E), key=lambda x: -sc[x]) if sc is not None else list(range(self.E))
            order.append([li * self.E + x for x in o])
        rank = []    # round-robin by rank across layers (top-1 of every layer first)
        for r in range(self.E):
            rank += [order[li][r] for li in range(self.L)]
        t = time.perf_counter()
        n1 = ext().warm_read(torch.tensor(rank[:self.nram], dtype=torch.int64))
        ext().nv_rows_all(self.L, self.E, p.slotof, self.a["home"], p.tabs)
        torch.cuda.synchronize(self.dev)
        NT._log(f"nv2 RAM warm: {n1} experts in {time.perf_counter() - t:.1f} s")
        self._rank = rank

    def post_warm(self):
        p = self.pool
        torch.cuda.synchronize(self.dev)
        so = p.slotof.flatten().cpu().numpy()
        vk = np.nonzero(so >= 0)[0]
        self.b70_keys = self._b70_pick(so) if self.b70 else []
        b70set = set(self.b70_keys)
        nd = 0
        if self.wb_on:
            drop = np.concatenate([vk.astype(np.int64), np.asarray(self.b70_keys, np.int64)])
            nd = ext().drop_keys(torch.from_numpy(drop))
            rest = [k for k in self._rank if so[k] < 0 and k not in b70set]
            st = ext().state()
            ks = st[1].numpy()
            rest = [k for k in rest if ks[k] == 0][: nd]
            if rest:
                ext().warm_read(torch.tensor(rest, dtype=torch.int64))
        ext().nv_rows_all(self.L, self.E, p.slotof, self.a["home"], p.tabs)
        torch.cuda.synchronize(self.dev)
        if self.b70:
            ms = ext().b70_attach(self.b70_ring, torch.tensor(self.b70_keys, dtype=torch.int64), 900.0)
            ext().b70_set(1, self.b70_timeout)
            NT._log(f"nv2 B70 tier: {len(self.b70_keys)} experts loaded on the B70 server in {ms / 1e3:.1f} s ({self.b70_ring})")
        ext().reset_counters()
        self.dstats.zero_()
        c = self.counters()
        NT._log(f"nv2 post-warm: {len(vk)} VRAM-resident, {nd} RAM copies of them dropped (exclusive), RAM resident "
                f"{c['resident']}, free {c['free']}")

    def _b70_pick(self, so):
        """B70 key set: the next experts after the 3090's VRAM warm set, by warm score (or the RAM tier's rank order)."""
        n = int(os.environ.get("QWEN38_B70_N", "3000"))
        if os.environ.get("QWEN38_B70_ORDER", "score") == "rr":
            order = self._rank
        else:
            sc = np.full(self.K, -1.0)
            for li, m in enumerate(self.pool.mods):
                s = self.scores.get(m.key)
                if s is not None:
                    sc[li * self.E:(li + 1) * self.E] = np.asarray(s[:self.E], np.float64)
            order = [int(k) for k in np.argsort(-sc, kind="stable") if sc[k] >= 0]
        return [k for k in order if so[k] < 0][:n]

    def set_b70(self, on):
        """Runtime switch (decode-KL harness): on=False -> B70-resident picks take the normal RAM/NVMe lanes (exact)."""
        if self.b70:
            ext().b70_set(int(bool(on)), self.b70_timeout)

    # ---- per MoE layer call (decode / small forwards) -----------------------------------------------------------
    def layer(self, li, sel, w, z, bsz):
        p = self.pool
        e = ext()
        n = sel.numel()
        if p.stage_n and n >= p.stage_min:
            p._stage_hook_nv2(li, n, sel)
            s = sel.reshape(-1)
            if s.dtype != torch.long or not s.is_contiguous():
                s = s.long().contiguous()
            p._ec_hits(li, s)
            self.st["staged_calls"] += 1
            return sel, w, False
        if p.stage_n:
            p._stage_hook_nv2(li, n)
        dec = n <= p.admit_max
        s = sel.reshape(-1)
        if s.dtype != torch.long or not s.is_contiguous():
            s = s.long().contiguous()
        ww = w.reshape(-1)
        if ww.dtype != torch.half or not ww.is_contiguous():
            ww = ww.half().contiguous()
        send_x = self.cpu_on and dec and bsz <= self.MAXB and n <= self.MAXP
        zp = 0
        if send_x:
            zz = z.view(bsz, -1)
            if zz.dtype != torch.half or not zz.is_contiguous():
                zz = zz.half().contiguous()
            self._zkeep = zz
            zp = zz.data_ptr()
        pf, npf, son = 0, 0, p.slotof[li]
        pre, self._pfe = self._pfe, None
        if self.prefetch and dec and li + 1 < self.L and bsz <= 8:
            if pre is not None and pre[0] == li and pre[1] == bsz:
                torch.cuda.current_stream(self.dev).wait_event(pre[3])
                pf = pre[2]
                self.st["pf_side"] += 1
            else:
                pf = self._predict(li + 1, z, bsz)
            npf = bsz * self.pool.mods[li + 1].num_experts_per_tok
            son = p.slotof[li + 1]
        e.nv_pub(s, ww, zp, n, bsz, self.H, li, self.E, p.first, p.slotof[li], 0 if dec else 1, int(send_x),
                 self.a["hc"], self.a["req"], self.a["hx"], self.dc, self.uidx, self.uexp, self.timeout_ns, pf, npf, son)
        so, wo = self.sel_out[:n], self.w_out[:n]
        e.nv_step(s, ww, so, wo, n, p.first, li, self.E, p.S, int(dec), p.slotof, self.a["home"], self.a["res"], p.tabs,
                  p.owner, p.refb, p.pin, p.ctl, p.jobs, self.jobx, self.jobnv, self.dstats, p.cbase.data_ptr(), p.spc,
                  p.rec, p.off_u, p.off_d, self.a["hc"], self.a["rep"], self.dc, self.uidx, self.ulane, self.a["wbl"],
                  self.a["adml"], self.a["land"], self.dflag, self.timeout_ns, int(self.wb_on), self.dbg)
        e.nv_copy(p.ctl, p.jobs, self.jobx, self.jobnv, li, self.E, self.a["home"], self.a["res"], p.cbase.data_ptr(), p.spc,
                  p.rec, p.sg, p.su, p.sd, p.off_u, p.off_d, self.ram_addr_dev, self.dc, self.uexp, self.ulane, p.slotof,
                  p.tabs, self.ph, self.a["hc"], self.timeout_ns, self.dstats, self.grid, self.dbg)
        self.st["dec_calls" if dec else "small_calls"] += 1
        return so.view(sel.shape), wo.view(w.shape), send_x

    def predict_early(self, li, z, bsz):
        """N136: launch layer li+1's routing guess on the side stream before layer li's router runs (decode only)."""
        # bsz 1 only: for more rows exllamav3's routing projection is routing_gemm_det, whose decode-class calls share
        # per-device static workspaces, so a concurrent guess on the side stream would corrupt the layer's own routing
        # (N136 AB1: C4 streams looped). The single-row GEMV path has no shared scratch.
        if not self.pfside or li + 1 >= self.L or bsz != 1:
            return
        main = torch.cuda.current_stream(self.dev)
        e0 = self._pfev[self._pfev_i]; e1 = self._pfev[self._pfev_i + 1]
        self._pfev_i = (self._pfev_i + 2) % len(self._pfev)
        e0.record(main)
        with torch.cuda.stream(self.side):
            self.side.wait_event(e0)
            ptr = self._predict(li + 1, z, bsz)
            e1.record(self.side)
        z.record_stream(self.side)
        if self._pfz is not z:
            self._pfz.record_stream(self.side)
        self._pfe = (li, bsz, ptr, e1)

    def _predict(self, lj, z, bsz):
        """Layer-ahead routing guess: layer lj's router (sigmoid + selection bias, top-8) on layer lj-1's MoE input,
        one fused exllamav3 routing kernel into private buffers. Residency hint only (never changes the computation)."""
        from exllamav3.modules import block_sparse_mlp_routing as R
        cfg = self.pool.mods[lj].routing_cfg
        b = self._pfb.get(bsz)
        if b is None:
            b = self._pfb[bsz] = (torch.empty((bsz, self.E), dtype=torch.half, device=self.dev),
                                  torch.empty((bsz, cfg.num_experts_per_tok), dtype=torch.long, device=self.dev),
                                  torch.empty((bsz, cfg.num_experts_per_tok), dtype=torch.half, device=self.dev))
        zz = z.view(bsz, -1)
        if zz.dtype != torch.half or not zz.is_contiguous():
            zz = zz.half().contiguous()
        R._gate_t(cfg)
        if self.pool.mods[lj].router_type == "std":
            R.ext.routing_std(zz, cfg.gate_tensor, b[0], b[1], b[2], cfg.per_expert_scale,
                              cfg.gate_tensor_t, cfg.router_bias, cfg.gate_i8, cfg.gate_sb)
        else:
            R.ext.routing_ds3_nogroup(zz, cfg.gate_tensor, b[0], R._esb_h(cfg), b[1], b[2], cfg.routed_scaling_factor,
                                      cfg.gate_tensor_t, R.ROUTING_ACT_SIGMOID, cfg.gate_i8, cfg.gate_sb)
        self._pfz = zz
        return b[1].data_ptr()

    def combine(self, fhs):
        f = fhs if fhs.is_contiguous() else fhs.contiguous()
        ext().nv_combine(f, self.dflag, self.a["hc"] + 6 * 8, self.a["hout"], self.dstats, self.a["hc"], self.timeout_ns)
        return f

    # ---- CPU tier -------------------------------------------------------------------------------------------------
    def _copy_cpu_scales(self):
        sc = lambda lins, attr: torch.stack([getattr(l.inner, attr).view(-1) for l in lins]).half().cpu().contiguous()
        return [(sc(m.multi_gate.linears, "suh"), sc(m.multi_gate.linears, "svh"),
                 sc(m.multi_up.linears, "suh"), sc(m.multi_up.linears, "svh"),
                 sc(m.multi_down.linears, "suh"), sc(m.multi_down.linears, "svh")) for m in self.pool.mods]

    def enable_cpu(self):
        """Initialize the CPU lane between requests, without reloading the model."""
        torch.cuda.synchronize(self.dev)
        ext().drain_all()
        if self.scales is None:
            self.scales=self._copy_cpu_scales()
            self.cpu_on=True
            self.cpu_setup()
            # The validation warm-read may have duplicated existing GPU experts.
            keys=np.flatnonzero(self.pool.slotof.flatten().cpu().numpy()>=0).astype(np.int64)
            ext().drop_keys(torch.from_numpy(keys))
            ext().nv_rows_all(self.L,self.E,self.pool.slotof,self.a['home'],self.pool.tabs)
        self.set_cpu(True, nvcpu=True)

    def cpu_setup(self):
        if not self.cpu_on:
            return
        e = ext()
        cpus = _cpus(os.environ.get("QWEN38_NV_CPU_CPUS", "2-23"))
        threads = int(os.environ.get("QWEN38_NV_CPU_THREADS", "0")) or len(cpus)
        mode = int(os.environ.get("QWEN38_NV_CPU_MODE", "-1"))
        lim = self.act_limit if os.environ.get("QWEN38_NV_CLAMP", "1") == "1" else 0.0
        e.cpu_init(threads, cpus, mode, lim)
        for li, s in enumerate(self.scales):
            e.cpu_add_layer(li, *s)
        self.selftest = self._selftest(lim)
        NT._log(f"nv2 CPU tier: {threads} threads on {cpus[0]}-{cpus[-1]}, mode {mode}, swiglu clamp {lim}, selftest "
                f"{self.selftest}")
        e.cpu_start(cpus[0])
        self.cpu_mode, self.cpu_lim = mode, lim

    @torch.inference_mode()
    def _selftest(self, lim, n_layers=3, n_exp=4):
        """CPU kernel on RAM slots vs an fp64 reference from exllamav3's own dequantised weights (the expert's bytes are
        copied into the bounce slot so LinearEXL3.get_weight_tensor decodes them; same swiglu clamp)."""
        p = self.pool
        st = ext().state()
        ks = st[1].numpy()
        g = torch.Generator(device="cpu").manual_seed(0)
        out = {}
        for li in list(range(0, self.L, max(1, self.L // n_layers)))[:n_layers]:
            m = p.mods[li]
            res = [x for x in range(self.E) if ks[li * self.E + x] == 2][:n_exp]
            if len(res) < n_exp:
                cold = [x for x in range(self.E) if ks[li * self.E + x] == 0][:n_exp - len(res)]
                ext().warm_read(torch.tensor([li * self.E + x for x in cold], dtype=torch.int64))
                res += cold
                assert len(res) == n_exp, "Not enough RAM experts for CPU validation"
            for mt in (1, 3):
                x = (torch.randn(mt, self.H, generator=g) * 0.35).half()
                ref = torch.zeros(mt, self.H, dtype=torch.float64)
                xd = x.double()
                for ex in res:
                    sl = int(st[0][li * self.E + ex])
                    self.bounce_t[: self.slot].copy_(self.ram.view(sl).to(self.dev))
                    torch.cuda.synchronize(self.dev)
                    Wg = m.gates[ex].inner.get_weight_tensor().cpu().double(); Wu = m.ups[ex].inner.get_weight_tensor().cpu().double()
                    a1, a2 = torch.nn.functional.silu(xd @ Wg), xd @ Wu
                    if lim:
                        a1, a2 = a1.clamp(max=lim), a2.clamp(-lim, lim)
                    a = (a1 * a2).half().double()
                    Wd = m.downs[ex].inner.get_weight_tensor().cpu().double()
                    ref += 0.25 * (a @ Wd)
                sel = torch.tensor([res] * mt, dtype=torch.int32)
                w = torch.full((mt, len(res)), 0.25)
                for mode in (1, 0, 2):
                    cpu = ext().cpu_forward(li, x.float(), sel, w, mode).double()
                    k = {1: "EXACT", 0: "AFFINE", 2: "I16"}[mode]
                    out[k] = max(out.get(k, 0.0), round(float((cpu - ref).norm() / ref.norm()), 6))
        self.bounce_t.zero_()
        torch.cuda.synchronize(self.dev)
        return out

    def set_cpu(self, on, nvcpu=None, mode=None, clamp=None):
        """Runtime switch (decode-KL harness): on=False -> every pick on the GPU path (exact)."""
        nvc = self.pol[11] if nvcpu is None else float(nvcpu)
        md = self.cpu_mode if mode is None else mode
        lim = self.cpu_lim if clamp is None else (self.act_limit if clamp else 0.0)
        self.pol[10],self.pol[11]=float(bool(on) and self.cpu_on),nvc
        self.cpu_mode,self.cpu_lim=md,lim
        ext().cpu_set(int(self.pol[10]), int(nvc), md, lim)

    # ---- elastic write-back (exclusive tier) ----------------------------------------------------------------------
    def elastic_writeback(self, chunks):
        p = self.pool
        if not self.elastic_wb or not chunks:
            return 0
        torch.cuda.synchronize(self.dev)
        ext().drain_all()
        own = p.owner.cpu().numpy()
        cb = p.cbase.cpu().tolist()
        keys, src = [], []
        for c in chunks:
            for s in range(c * p.spc, (c + 1) * p.spc):
                if own[s] >= 0:
                    keys.append(int(own[s])); src.append(cb[c] + (s % p.spc) * p.rec)
        t = time.perf_counter()
        n = ext().wb_bulk(torch.tensor(keys, dtype=torch.int64), torch.tensor(src, dtype=torch.int64)) if keys else 0
        self.st["elastic_wb"] += n; self.st["elastic_wb_ms"] += (time.perf_counter() - t) * 1e3
        return n

    # ---- staged prefill (called from Pool._stage_hook_nv2 on the main thread) --------------------------------------
    def stage_submit(self, li, b, selected=None):
        p = self.pool
        if li >= self.L:
            return None
        if selected is None:
            miss = np.nonzero(p.slot_np[li] < 0)[0]
        else:
            selected=np.asarray(selected,dtype=np.int64)
            miss=selected[p.slot_np[li,selected]<0]
            ext().stage_begin_selected(torch.from_numpy(np.ascontiguousarray(p.slot_np)),li,
                                       torch.from_numpy(np.ascontiguousarray(miss)))
        assert len(miss) <= p.stage_n, f"staging overflow {len(miss)} > {p.stage_n}"
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.dev))
        fut = self.bg.submit(ext().stage_layer, li, p.stage[b].data_ptr(), p.cstream.cuda_stream, ev.cuda_event,
                             p.ev_ready[b].cuda_event)
        return (li, b, miss, fut, ev)

    def stage_wait(self, h):
        t = time.perf_counter()
        h[3].result()
        self.st["stage_wait_ms"] += (time.perf_counter() - t) * 1e3

    # ---- checks / stats -----------------------------------------------------------------------------------------
    def verify(self, n=64, seed=0):
        """I1: sampled VRAM slots and RAM slots byte-compared against fresh O_DIRECT store reads; pointer tables, homes,
        landed flags consistent with the host's residency."""
        import random
        p = self.pool
        torch.cuda.synchronize(self.dev)
        ext().drain_all()
        rnd = random.Random(seed)
        owner = p.owner.cpu().numpy()
        vs = [s for s in range(p.S) if owner[s] >= 0 and p.chunks[s // p.spc] is not None]
        so_k, ks, pc, ow = [t.numpy() for t in ext().state()]
        rk = [k for k in range(self.K) if ks[k] == 2]
        if getattr(self, "_vtmp", None) is None:
            self._vtmp = NT.HostArena(1, self.slot, 1, self.dev.index or 0, "verify")
        tmp = self._vtmp
        bad_v = bad_r = 0
        for s in rnd.sample(vs, min(n, len(vs))):
            ext().read_sync(int(owner[s]), int(tmp.addr[0]))
            bad_v += int(not torch.equal(p.chunks[s // p.spc][s % p.spc][: self.slot].cpu(), tmp.view(0)))
        for k in rnd.sample(rk, min(n, len(rk))):
            ext().read_sync(k, int(tmp.addr[0]))
            bad_r += int(not torch.equal(self.ram.view(int(so_k[k])), tmp.view(0)))
        cb = p.cbase.cpu().tolist()
        slotof = p.slotof.flatten().cpu().numpy()
        home = self.home.view(-1, 3).numpy()
        res = self.res.numpy()
        bad_t = stale = 0       # rows of non-VRAM experts are don't-care (set at use: nv_copy fix / staging)
        for li in range(self.L):
            pg = p.keep[li][0].cpu().numpy()
            for x in range(self.E):
                k = li * self.E + x
                s = int(slotof[k])
                if s >= 0:
                    bad_t += int(pg[x] != cb[s // p.spc] + (s % p.spc) * p.rec)
                else:
                    stale += int(pg[x] != int(home[k, 0]))
        addr = self.ram.addr
        hres = sum(1 for k in rk if home[k, 0] != addr[so_k[k]] or res[k] != 1)
        hnon = sum(1 for k in range(self.K) if ks[k] == 0 and (home[k, 0] != self.ph or res[k] != 0))
        both = int(sum(1 for k in rk if slotof[k] >= 0))
        return {"vram_checked": min(n, len(vs)), "vram_bad": bad_v, "ram_checked": min(n, len(rk)), "ram_bad": bad_r,
                "table_bad": int(bad_t), "rows_stale_nonvram": int(stale), "home_bad_resident": int(hres), "home_bad_nonresident": int(hnon),
                "ram_resident": len(rk), "vram_and_ram": both, "pinned": int((pc > 0).sum()),
                "dev_bad_rows": int(self.dstats[8].item())}

    def latency(self):
        """Per-layer-call timeline from the debug rings. Device (globaltimer): publish, step start / reply seen / step end,
        copy start / end, admitted jobs; host (CLOCK_REALTIME): notice, reply. Host-device offset removed with the
        minimum of (host notice - device publish), so the host-side gaps are relative to the fastest observed."""
        d = ext().peek(self.dbg.data_ptr(), self.dbg.numel())
        d = np.asarray(d, np.int64).reshape(-1, 8)
        hs, hn, hr = [t.numpy() for t in ext().dbg_host()]
        ok = (d[:, 0] > 0) & (hs == d[:, 0]) & (d[:, 6] > 0) & (d[:, 4] > 0)
        if ok.sum() < 8:
            return {"n": int(ok.sum())}
        o = np.argsort(d[ok, 0])
        d, hn, hr = d[ok][o], hn[ok][o], hr[ok][o]
        off = np.min(hn - d[:, 1])
        q = lambda x: {"p50": round(float(np.percentile(x, 50)) / 1e3, 1), "p90": round(float(np.percentile(x, 90)) / 1e3, 1),
                       "p99": round(float(np.percentile(x, 99)) / 1e3, 1), "mean": round(float(x.mean()) / 1e3, 1)}
        consec = np.diff(d[:, 0]) == 1
        gap = (d[1:, 1] - d[:-1, 6])[consec]                       # copy end -> next publish (MoE + attention + glue)
        nj = d[:, 7].astype(np.float64)
        cms = (d[:, 6] - d[:, 5]).astype(np.float64)
        rate = (nj[nj > 0] * self.slot) / (cms[nj > 0] + 1)      # bytes per ns = GB/s
        return {"n": int(len(d)), "us_pub_to_notice_rel": q(hn - d[:, 1] - off), "us_host_serve": q(hr - hn),
                "us_step_wait": q(d[:, 3] - d[:, 2]), "us_pub_to_seen": q(d[:, 3] - d[:, 1]),
                "us_step_after_reply": q(d[:, 4] - d[:, 3]), "us_copy": q(d[:, 6] - d[:, 5]),
                "copy_gbps_p50": round(float(np.median(rate)), 2) if len(rate) else None,
                "jobs_per_call": round(float(nj.mean()), 2), "us_gap_copy_to_next_pub": q(gap) if len(gap) else None,
                "us_layer_total": q(np.diff(d[:, 1])[consec]) if consec.any() else None}

    def summary(self):
        c = self.counters()
        d = self.dstats.tolist()
        dn = ["hit", "miss", "admit", "novict", "evict", "wb", "wb_noslot", "fix", "bad_rows", "reply_wait_ns", "reply_waits",
              "nv_wait_ns", "nv_waits", "timeouts"]
        dev = dict(zip(dn, d))
        steps = max(1, c["dec_reqs"] // self.L)
        toks = max(1, c["dec_tok"] // self.L)
        r = lambda x: round(x, 3)
        out = {"ram_slots": self.nram, "ram_gib": r(self.nram * self.slot / GiB), "pf_ring": self.pf.n, "exclusive": self.wb_on,
               "cpu_tier": self.cpu_on, "pol": self.pol, "host": c, "dev": dev, "py": dict(self.st),
               "decode_per_step": {"steps_est": steps, "tokens_est": toks,
                                   "picks": r(c["picks"] / steps) if c["small_reqs"] == 0 else None,
                                   "vram_hits": r(c["vram"] / steps), "ram_hits": r(c["ram"] / steps),
                                   "nvme": r(c["nvme"] / steps), "cpu_experts": r(c["cpu_experts"] / steps),
                                   "zc_admits": r(c["lane_zc"] / steps), "nvme_gpu": r(c["lane_nvg"] / steps),
                                   "nvme_cpu": r(c["lane_nvc"] / steps),
                                   "reply_wait_ms": r(d[9] / 1e6 / steps), "nvme_wait_ms": r(d[11] / 1e6 / steps),
                                   "cpu_busy_ms": r(c["cpu_busy_ns"] / 1e6 / steps),
                                   "b70_experts": r(c["lane_b70"] / steps), "b70_wait_ms": r(c["b70_wait_ns"] / 1e6 / steps)},
               "b70": {"on": self.b70, "keys": len(self.b70_keys), "jobs": c["b70_jobs"], "rows": c["b70_rows"],
                       "rtt_us_mean": r(c["b70_rtt_ns"] / 1e3 / max(1, c["b70_jobs"])), "rtt_us_max": r(c["b70_rtt_max_ns"] / 1e3),
                       "wait_us_mean": r(c["b70_wait_ns"] / 1e3 / max(1, c["b70_jobs"])), "err": c["b70_err"]},
               "memcg": NT.cgroup_mem()}
        ps = out["decode_per_step"]
        out["decode_per_token"] = {k: (r(v * steps / toks) if isinstance(v, float) else v) for k, v in ps.items()
                                   if k not in ("steps_est", "tokens_est", "picks")}
        if getattr(self, "selftest", None) is not None:
            out["cpu_selftest"] = self.selftest
        try:
            out["latency"] = self.latency()
        except Exception as ex:
            out["latency"] = str(ex)
        return out


def attach_pool(pool):
    global NV
    nd = NT._drop_checkpoint_cache()
    NT._log(f"model loaded + VRAM pool built; DONTNEED on {nd} checkpoint files")
    NV = Nv2(pool)
    return NV


class _BCNv:
    """nv_tier.BCProxy needs .pool, .bounce, .st"""
    def __init__(self, nv):
        self.pool, self.bounce, self.st = nv.pool, nv.bounce, nv.st


def finish(model):
    nv = NV
    shim = _BCNv(nv)
    for li, m in enumerate(nv.pool.mods):
        assert m.bc is not None and m.support_quant_paths, m.key
        m.bc = NT.BCProxy(m.bc, li, shim)
    nd = NT._drop_checkpoint_cache()
    NT._log(f"nv2 ready: {NT._LOAD['marked_linears']} expert linears on the bounce slot, "
            f"{NT._LOAD['skipped_bytes'] / 1e9:.1f} GB of checkpoint expert bytes never read, DONTNEED on {nd} files")
    if os.environ.get("QWEN38_NV_VERIFY_START", "1") == "1":
        print(f" -- nv2 startup verify: {nv.verify(32)}", flush=True)


def summary():
    return NV.summary() if NV is not None else None

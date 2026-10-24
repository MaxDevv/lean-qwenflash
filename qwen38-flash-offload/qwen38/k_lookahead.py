"""N136 decode lookahead (QWEN38_LA=1): launch the next decode step's forward before the host reads the sampled token.

Stock exllamav3 1.5.1 decode step (Generator.iterate_gen) per token:
    forward(k) enqueued -> sampler enqueued -> torch.cuda.synchronize -> receive_sample (detokenize, stop checks,
    page hashing) -> results to the asyncio consumers -> next iterate(): block table, embedding on the CPU, forward(k+1)
The GPU idles from the end of sampling(k) until the host has done all of that and enqueued forward(k+1)'s first
kernels: ~2.1 ms per token on the 3090 (HOM-272 L1_pfon, step-boundary bubble).

With lookahead, for a single active decode job (C1):
    sampler(k) enqueued -> token copied to a device ids buffer -> forward(k+1) enqueued with that DEVICE token
    (embedding gathered on the GPU from the host-mapped embedding table) -> wait for token k only -> receive_sample(k)
so the host bookkeeping of token k runs while the GPU already computes step k+1. The computation is the same:
same kernels, same inputs, same order on the one stream (the embedding row is a copy, CPU or GPU), so outputs are
bitwise identical (checked with greedy equality in exact mode).

The lookahead is only taken when token k cannot change what forward(k+1) needs:
  - one active job, one sequence, no pending job, no draft model / n-gram, no filters, banned strings, forced tokens,
    indexed embeddings, probs / top-k / logits returns, checkpoint holds, mrope
  - the next position is not a page boundary (page hashing / recurrent-state stash happen there) and has an allocated
    page; the job does not requeue (max_rq_tokens) at token k
If token k ends the job (EOS, stop string, max_new_tokens, cancel), the speculative forward(k+1) is dropped: its KV row
lies beyond the page's recorded kv_position and its recurrent state is freed with the job (about one wasted step,
~60 ms of GPU time, per finished request).
Staging buffers (block table, cache_seqlens) are double-buffered by step parity: the buffer refilled for step k+1 was
last read by step k-1's H2D copies, which completed before sampling(k-1) did.

Env: QWEN38_LA=1 (install), QWEN38_LA_CHECK=1 (assert the launched position matches after receive_sample).
Runtime switch: set_enabled(bool) (serve.py /la?on=0|1) for same-server A/B.
"""
import os
import time
import torch
import torch.nn.functional as F

PAGE = 256
ENABLED = True
CHECK = os.environ.get("QWEN38_LA_CHECK", "1") == "1"
# N136: block-table width = pages actually needed (instead of the stock 16-page padding) while the padded width is 16
# pages. exllamav3's BC MLA derives the flash-decoding split count from the width (split_len stays 4 * block_n = 128
# while the count is under the cap), so this drops the empty splits (written as m = -inf, o = 0 and added with weight
# 0 by the combine) from the split and combine kernels; above 16 pages the stock width (and split length) is kept.
BTTRIM = os.environ.get("QWEN38_LA_BTTRIM", "0") == "1"
STATS = {"steps": 0, "la_launched": 0, "la_used": 0, "la_dropped": 0, "skip_boundary": 0, "skip_rq": 0,
         "skip_ineligible": 0, "fallback": 0, "dev_embed": 0, "wait_ms": 0.0}

_EMB = {}       # embedding module id -> (mapped cpu tensor, cuda view)
_ST = None


def set_enabled(on):
    global ENABLED
    ENABLED = bool(on)
    return ENABLED


# ---- device-side embedding from a host-mapped table ---------------------------------------------------------------
def prep_model(model):
    """Move every Embedding weight that lives in host RAM into a page-aligned, cudaHostRegister(Mapped) buffer (same
    bytes, the original storage is released, so host RAM use is unchanged) and patch the module so a CUDA input_ids
    tensor is embedded on the GPU by a zero-copy gather. CPU input_ids keep the stock path. Call right after
    model_init.init (before the NVMe tier sizes its RAM arena from the cgroup headroom)."""
    from exllamav3.modules import Embedding
    from exllamav3.util.tensor import to2
    import mmap, ctypes
    from exllamav3.ext import exllamav3_ext as xe
    dev_idx = torch.cuda.current_device()
    n = 0
    for m in model.modules:
        if not isinstance(m, Embedding) or m.embedding is None:
            continue
        w = m.embedding.weight.data
        if w.device.type != "cpu" or m.normalize or m.multiplier != 1.0:
            continue
        nbytes = w.numel() * w.element_size()
        nb = (nbytes + 4095) // 4096 * 4096
        mm = mmap.mmap(-1, nb, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        r = torch.cuda.cudart().cudaHostRegister(a, nb, 2)
        assert int(r) == 0, f"cudaHostRegister(embedding) failed: {r}"
        host = torch.frombuffer(mm, dtype=w.dtype, count=w.numel()).view(w.shape)
        host.copy_(w)
        assert torch.equal(host, w)
        m.embedding.weight = torch.nn.Parameter(host, requires_grad=False)
        del w
        cview = xe.pinned_cuda_view(host, dev_idx)
        _EMB[id(m)] = (mm, host, cview)
        orig_fwd, orig_prep = m.forward, m.prepare_for_device

        def prep(x, params, _o=orig_prep):
            if isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.long:
                return x
            return _o(x, params)

        def fwd(x, params, out_dtype=None, _o=orig_fwd, _m=m, _cv=cview):
            if isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.long and not params.get("indexed_embeddings"):
                if "input_ids" not in params:
                    params["input_ids"] = x
                y = F.embedding(x, _cv)
                STATS["dev_embed"] += 1
                return to2(y, out_dtype or _m.out_dtype or y.dtype, _m.out_dtype)
            return _o(x, params, out_dtype) if out_dtype is not None else _o(x, params)

        m.prepare_for_device = prep
        m.forward = fwd
        n += 1
    print(f" -- k_lookahead: {n} embedding table(s) host-mapped for device-side lookup", flush=True)
    return n


# ---- lookahead state ------------------------------------------------------------------------------------------------
class _State:
    def __init__(self, dev):
        self.dev = dev
        self.ids = [torch.zeros((1, 1), dtype=torch.long, device=dev) for _ in range(2)]
        self.tok_pin = [torch.zeros((1, 1), dtype=torch.long, pin_memory=True) for _ in range(2)]
        self.ev = [torch.cuda.Event() for _ in range(2)]
        self.bi = {}          # (parity, width) -> pinned block index
        self.sl = [torch.zeros((1,), dtype=torch.int32, pin_memory=True) for _ in range(2)]
        self.par = 0
        self.pending = None   # dict(job, logits, pos, par)

    def block_index(self, par, width):
        k = (par, width)
        b = self.bi.get(k)
        if b is None:
            b = self.bi[k] = torch.zeros((1, width), dtype=torch.int32, pin_memory=True)
        return b


def _job_ok(gen, job):
    return (len(job.sequences) == 1 and job.is_prefill_done() and job.time_first_token is not None
            and not job.filters and job.banned_strings_utf32_offsets is None and job.forced_ids is None
            and not job.embeddings and not job.return_probs and not job.return_top_tokens and not job.return_logits
            and job.checkpoint is None and job.new_tokens >= 0 and job.recurrent_state is not None)


def _gen_ok(gen):
    return (gen.draft_model is None and not gen.ngram_match_min and len(gen.active_jobs) == 1 and not gen.pending_jobs
            and "mrope" not in gen.model.caps and gen.visualizer is None)


def _launch(gen, st, job, ids, par, pos, seq_len_after):
    """Enqueue one decode forward for the job: input ids (1, 1) (device or pinned host), cache position pos."""
    seq = job.sequences[0]
    need = (seq_len_after + gen.num_draft_tokens + PAGE - 1) // PAGE
    mp = (need + 15) // 16 * 16
    if BTTRIM and mp == 16:
        mp = need
    bi = st.block_index(par, mp)
    bi.zero_()
    sbi = seq.block_index_tensor[:, :mp]
    bi[:, :sbi.shape[-1]].copy_(sbi)
    sl = st.sl[par]
    sl[0] = pos
    params = {
        "attn_mode": "flash_attn",
        "block_table": bi,
        "cache": gen.cache,
        "cache_seqlens": sl,
        "recurrent_states": [job.recurrent_state],
        "indexed_embeddings": [],
        "positions": None,
        "recurrent_history": False,
        "pinned_staging": True,
    }
    return gen.model.forward(input_ids=ids, params=params)


def _can_lookahead(gen, job):
    """Decided before token k is known: forward(k+1) would process token k at position kv_position + 1."""
    seq = job.sequences[0]
    nxt = seq.kv_position + 1
    if nxt % PAGE == 0:
        STATS["skip_boundary"] += 1
        return False
    if nxt // PAGE >= len(seq.allocated_pages):
        STATS["skip_boundary"] += 1
        return False
    if job.new_tokens + 1 > job.max_rq_tokens - gen.num_draft_tokens - 1 or job.new_tokens + 1 >= job.max_new_tokens:
        STATS["skip_rq"] += 1
        return False
    return True


def iterate_gen_la(gen, results, draft_tokens=None, _orig=None):
    st = _ST
    p = st.pending
    if draft_tokens is not None or not ENABLED:
        if p is not None:
            st.pending = None
            STATS["la_dropped"] += 1
        return _orig(gen, results, draft_tokens)

    if p is not None:
        job = p["job"]
        if job not in gen.active_jobs or job.is_finished or not _job_ok(gen, job):
            st.pending = None
            STATS["la_dropped"] += 1
            p = None
    if p is None:
        if not _gen_ok(gen) or not _job_ok(gen, gen.active_jobs[0]):
            STATS["fallback"] += 1
            return _orig(gen, results, draft_tokens)
        # first eligible step: stock-equivalent launch with host ids
        job = gen.active_jobs[0]
        seq = job.sequences[0]
        ids = job.get_input_ids_list(None, 0, add_to_cache=True)[0]
        par = st.par
        pin = st.tok_pin[par]
        pin.copy_(ids)
        logits = _launch(gen, st, job, pin, par, seq.kv_position, len(seq.sequence_ids))
        p = {"job": job, "logits": logits, "par": par, "pos": seq.kv_position}
    st.pending = None
    job = p["job"]
    seq = job.sequences[0]
    STATS["steps"] += 1

    # sampling of token k (GPU), token to the device ids buffer + pinned copy, completion event
    job.prepare_logit_mask()
    job.prepare_sampling_past_ids()
    logits = p["logits"]
    token_logits = logits[0:1, :, :]
    sampled = job.receive_logits(token_logits)
    par = p["par"] ^ 1
    st.par = par
    ids = st.ids[par]
    ids.copy_(sampled[0].view(1, 1))
    pin = st.tok_pin[par]
    pin.copy_(ids, non_blocking=True)
    ev = st.ev[par]
    ev.record()

    # lookahead: forward(k+1) on the device token, before the host sees it
    la = None
    if gen.active_jobs == [job] and not gen.pending_jobs and _can_lookahead(gen, job):
        pos1 = seq.kv_position + 1
        la = {"job": job, "logits": _launch(gen, st, job, ids, par, pos1, len(seq.sequence_ids) + 1),
              "par": par, "pos": pos1}
        STATS["la_launched"] += 1

    t0 = time.perf_counter()
    ev.synchronize()
    STATS["wait_ms"] += (time.perf_counter() - t0) * 1e3
    eos, sampled_token, rq = job.receive_sample(token_logits, pin, sampled[1], sampled[2], sampled[3], results)
    completed, requeuing = [], []
    if job.checkpoint_rewound:
        job.checkpoint_rewound = False
    if len(job.sequences) == 1 and rq:
        requeuing.append(job)
    elif eos:
        completed.append(job)

    if la is not None:
        if completed or requeuing or job.checkpoint is not None or job not in gen.active_jobs:
            STATS["la_dropped"] += 1
        else:
            if CHECK:
                assert seq.kv_position == la["pos"], f"lookahead position {la['pos']} != kv_position {seq.kv_position}"
            job.get_input_ids_list(None, 0, add_to_cache=True)   # record token k in the page (stock does it next step)
            st.pending = la
            STATS["la_used"] += 1

    num_jobs = gen.num_remaining_jobs()
    for j in completed + requeuing:
        if j in requeuing and gen.recurrent_cache is not None:
            j.maybe_stash_recurrent(gen.recurrent_cache, PAGE)
        j.deallocate_pages()
        gen.active_jobs.remove(j)
    for j in requeuing:
        gen.pending_jobs.insert(0, j.prepare_for_requeue())
    if num_jobs and not gen.num_remaining_jobs():
        gen.on_queue_drained()


def install(model):
    """Patch Generator.iterate_gen (call after load; prep_model must already have run)."""
    global _ST
    from exllamav3.generator.generator import Generator
    if getattr(Generator, "_n136_la", False):
        return
    _ST = _State(torch.device("cuda", torch.cuda.current_device()))
    orig = Generator.iterate_gen

    def it(self, results, draft_tokens=None):
        return iterate_gen_la(self, results, draft_tokens, _orig=orig)
    Generator.iterate_gen = it
    Generator._n136_la = True
    print(f" -- k_lookahead: decode lookahead installed (enabled {ENABLED}, check {CHECK})", flush=True)


def summary():
    s = dict(STATS)
    s["enabled"] = ENABLED
    s["wait_ms_per_step"] = round(s["wait_ms"] / max(1, s["steps"]), 3)
    return s

"""Zero-copy expert tier for stock exllamav3 1.5.1 (monkeypatch; no source or kernel edits).

Three tiers per MoE layer, by a per-expert ranking (e.g. routing frequency):
  rank [0, V)          VRAM-resident (normal exllamav3 load)
  rank [V, first)      zero-copy: the expert's .trellis lives in pinned host memory; exllamav3's own MoE kernels read it
                       through their per-expert pointer tables over PCIe (ext.pinned_cuda_view = UVA alias, the same
                       mechanism exllamav3 uses for pinned vision towers). suh/svh (tiny) stay in VRAM.
  rank [first, 288)    CPU worker (stock -mcs tail; optional), placement from EXL3_MOE_CPU_SPLIT_STATS
Autosplit accounting uses torch.cuda.memory_allocated, which excludes pinned host memory, so zero-copy experts do not
consume the per-device VRAM budget.

Env:
  QWEN38_ZC_VRAM   int V: experts per layer kept in VRAM (the rest of the GPU slice becomes zero-copy);
                  or a JSON file {layer_key: V}
  QWEN38_ZC_STATS  JSON {layer_key: [score x E]} ranking (default: EXL3_MOE_CPU_SPLIT_STATS); required
One pinned arena per layer (exact size; torch's per-tensor pinned allocations round up to powers of two).
"""
import json, os, re
import torch

_ACTIVE = {"key": None}
_ARENAS = []          # keep pinned stores alive
_STATS = {"zc_experts": 0, "zc_bytes": 0, "layers": 0}


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def pinned_exact(nbytes, dev_idx):
    """Exact-size pinned host buffer (torch's pinned allocator rounds each block up to a power of two: a 2.72 GB layer
    arena would cost 4 GB). Plain host tensor + cudaHostRegister under the GPU's context (pinned_cuda_view checks the
    pointer's device)."""
    t = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    with torch.cuda.device(dev_idx):
        r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 2)   # cudaHostRegisterMapped
    assert int(r) == 0, f"cudaHostRegister failed: {r}"
    assert t.is_pinned()
    return t


def install():
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.linear import Linear
    from exllamav3.loader.safetensors import SafetensorsCollection

    vr = os.environ["QWEN38_ZC_VRAM"]
    per_layer_v = _load_json(vr) if not vr.lstrip("-").isdigit() else None
    v_default = int(vr) if per_layer_v is None else None
    stats_path = os.environ.get("QWEN38_ZC_STATS") or os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
    assert stats_path, "QWEN38_ZC_STATS (or EXL3_MOE_CPU_SPLIT_STATS) required"
    stats = _load_json(stats_path)

    orig_split = BlockSparseMLP.cpu_maybe_split_load

    def split_then_mark(self, device, **kwargs):
        orig_split(self, device, **kwargs)
        if device is None or torch.device(device).type != "cuda" or self.key not in stats:
            return
        score = stats[self.key]
        order = sorted(range(len(score)), key=lambda e: -score[e])
        rank = {e: r for r, e in enumerate(order)}
        V = per_layer_v.get(self.key, 0) if per_layer_v is not None else v_default
        lins = []
        for group in ([self.gates] if self.gated else []) + [self.ups, self.downs]:
            for l in group:
                e = int(re.search(r"experts\.(\d+)\.", l.key).group(1))
                if rank[e] >= V:
                    lins.append(l)
        if not lins:
            return
        stc = self.config.stc
        sizes = [stc.get_tensor_size(l.key + ".trellis") for l in lins]
        total = sum(sizes)
        dev_idx = torch.device(device).index or 0
        arena = pinned_exact(total, dev_idx)
        _ARENAS.append(arena)
        off = 0
        for l, sz in zip(lins, sizes):
            l._zc_slot = (arena, off, sz, dev_idx)
            off += sz
        n_exp = len({re.search(r"experts\.(\d+)\.", l.key).group(1) for l in lins})
        _STATS["zc_experts"] += n_exp; _STATS["zc_bytes"] += total; _STATS["layers"] += 1
        print(f" -- zero-copy tier: {self.key} {n_exp} experts ({total / 1e9:.2f} GB pinned) V={V}", flush=True)

    BlockSparseMLP.cpu_maybe_split_load = split_then_mark

    orig_lin_load = Linear.load

    def lin_load(self, device, **kwargs):
        if getattr(self, "_zc_slot", None) is None:
            return orig_lin_load(self, device, **kwargs)
        _ACTIVE["key"] = self.key + ".trellis"
        _ACTIVE["lin"] = self
        try:
            return orig_lin_load(self, device, **kwargs)
        finally:
            _ACTIVE["key"] = None; _ACTIVE["lin"] = None

    Linear.load = lin_load

    orig_get = SafetensorsCollection.get_tensor

    def get_tensor(self, key, device=None, *args, **kwargs):
        if key != _ACTIVE["key"]:
            return orig_get(self, key, device, *args, **kwargs)
        arena, off, sz, dev_idx = _ACTIVE["lin"]._zc_slot
        kwargs.pop("no_defer", None)
        t = orig_get(self, key, "cpu", *args, no_defer=True, **kwargs)
        assert t.numel() * t.element_size() == sz, (key, t.shape, sz)
        p = arena[off: off + sz].view(t.dtype).view(t.shape)
        p.copy_(t)
        return ext.pinned_cuda_view(p, dev_idx)

    SafetensorsCollection.get_tensor = get_tensor
    print(f" -- exl3_tiers installed: V={vr} stats={stats_path}", flush=True)


def summary():
    return dict(_STATS)

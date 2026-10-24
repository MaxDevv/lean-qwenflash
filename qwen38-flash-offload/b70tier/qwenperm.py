"""Word permutation from the NVMe store's raw expert record (exllamav3 order: gate/up/down trellis, then suh/svh vectors)
to the exl3xpu expert blob (planar4(gate|up), planar4(down), g.suh u.suh g.svh u.svh d.suh d.svh). Both are 9,474,048 B
for Qwen3.8-Flash-Next (H 4096, I 2048, K 3); blob word j = record word perm[j]. Checked byte for byte against
exl3xpu.moe_offload.pack_expert on real checkpoint experts (campaign N128).
"""
import torch


def make_perm(H: int, I: int, K: int) -> torch.Tensor:
    """int64 [blob/4]: blob word j = raw record word perm[j]."""
    W = 8 * K                                    # int32 words per 16x16 tile (16K int16)
    ng = (H // 16) * (I // 16) * W               # words per trellis matrix (gate, up, down are the same size)
    r = torch.arange(3 * ng + (3 * H + 3 * I) // 2, dtype=torch.int64)
    gate = r[0:ng].view(H // 16, I // 16, W)
    up = r[ng:2 * ng].view(H // 16, I // 16, W)
    down = r[2 * ng:3 * ng].view(I // 16, H // 16, W)

    def p4(w):
        rows, n, _ = w.shape
        return w.reshape(rows, n // 4, 4, 8, K).permute(0, 1, 4, 2, 3).reshape(-1)

    o = 3 * ng
    sec = {}
    for name, n in (("g_suh", H), ("g_svh", I), ("u_suh", H), ("u_svh", I), ("d_suh", I), ("d_svh", H)):
        sec[name] = r[o:o + n // 2]
        o += n // 2
    assert o == r.numel()
    perm = torch.cat([p4(torch.cat([gate, up], dim=1)), p4(down), sec["g_suh"], sec["u_suh"], sec["g_svh"],
                      sec["u_svh"], sec["d_suh"], sec["d_svh"]])
    assert perm.numel() == r.numel() and torch.equal(torch.sort(perm).values, r)
    return perm

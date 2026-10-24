# B70 expert tier (optional second GPU)

With `QWEN38_B70=1`, the 55 GiB `nvme` mode uses one Intel Arc Pro B70 (32 GB) in the same machine. It holds 3,000 more
experts in GPU memory. The tier is off by default and no other mode changes.

## How it works

- **Which experts.** At warm-up the engine takes the next 3,000 experts by routing score after the 3090's VRAM set. It
  sends their keys to the **B70 expert server** (`b70tier/b70srv.py`, a separate container on the B70). The server
  reads them from the same NVMe store (`/nvx`) into its VRAM: 28.4 GB in about 5 s.
- **No duplicates.** These experts are dropped from the RAM tier, and RAM refills with colder experts. VRAM, B70 and
  RAM never hold the same expert.
- **Per decode layer.** The host controller gives each B70-resident pick its own lane. The GPU masks those picks like
  CPU-lane picks.
- **The CPU worker drives the B70.** It first posts the B70 picks and their input rows to a shared ring, computes its
  own RAM experts, and then adds the B70's weighted rows into the partial that the 3090 adds back. The 3090, the CPU
  and the B70 run at the same time.
- **The ring.** It is one file in a host tmpfs directory that both containers mount
  (`/run/local-ai/shared/b70.ring`). There is no network and no peer-to-peer transfer. The B70 server runs exl3xpu's
  grouped MoE kernel (`b70tier/kernels/moe-n128`, swiglu clamp 10) over a pointer table of its resident experts.
- **Cost.** About 0.17 ms per layer call plus 0.021 ms per expert. The CPU worker waits 0.4 ms per token for it,
  because the CPU lane is still the longer one.
- **Prefill** keeps the staged path; B70 experts are read from NVMe like any other non-RAM expert there.

## Measured (omarchy, 2026-10-09, same session, campaign N137)

1x RTX 3090 + 1x Arc Pro B70, 55 GiB cap, NVMe RAID0. Decode is the aggregate over streams. Each cell reads
"tier off / tier on" (two runs with the tier on).

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 657 / 648-658 | 17.48 / 26.40-26.61 | 1 | 131k fp16 | 1 / 2 |
| 8k | - | 17.84 / 30.34-30.63 | 2 | 131k fp16 | 1 / 2 |
| 8k | - | 19.11 / 33.00-33.54 | 4 | 131k fp16 | 1 / 2 |
| 32k | 966 / 968-974 | 16.28 / 24.60-24.92 | 1 | 131k fp16 | 1 / 2 |
| 32k | - | 17.35 / 29.09 | 2 | 131k fp16 | 1 / 2 |

With a 120 GiB cap (every expert in RAM), 8k decode goes from 22.09 / 23.32 / 24.64 to 28.51 / 34.55 / 39.99 at C1 / C2 / C4.

Quality:
- Panel: top-1 1.0, KL 0. Prefill is exact.
- Paired decode-KL against the exact GPU path (6 prompts x 192 tokens):

| config | KL mean | top-1 |
|---|---|---|
| shipped `nvme` | 0.0070 | 0.980 |
| with the B70 tier | 0.0034 | 0.988 |
| B70 lane alone | 0.0071 | 0.982 |

- B70 expert outputs vs an fp64 reference: rel L2 0.0013-0.0015 on 48 real experts (the AVX2 CPU tier gets 0.0010).

## Run it (Docker)

The B70 server starts first, then the engine:

```bash
mkdir -p /run/local-ai/shared
docker run -d --name qwen38-b70 --device /dev/dri/renderD131 -e ZE_AFFINITY_MASK=0 --memory 16g --memory-swap 16g \
  --ulimit memlock=-1 -v /mnt/nvme/qwen38:/nvx:ro -v /run/local-ai/shared:/run/local-ai/shared "$B70_IMG"
docker run -d --name qwen38 --gpus '"device=0"' --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1 \
  -e QWEN38_MODE=nvme -e QWEN38_EC_MAX_SLOTS=1376 -e QWEN38_B70=1 -p 127.0.0.1:30000:30000 \
  -v /data/qwen38:/models:ro -v /mnt/nvme/qwen38:/nvx:ro -v /run/local-ai/shared:/run/local-ai/shared "$IMG"
```

- **Render node.** Pick the B70's node by PCI path: `readlink -f /dev/dri/by-path/pci-<bdf>-render`.
- **Wait for the ring.** The engine waits up to `QWEN38_B70_WAIT_S` (180 s) for the ring and refuses to start without it.
- **Server image.** `$B70_IMG` is the B70 expert-server image, built from `local-ai-images/qwen38-b70-expert-server`.

| env | default | what it does |
|---|---|---|
| `QWEN38_B70` | 0 | 1 = use the B70 expert tier (needs `QWEN38_MODE=nvme`) |
| `QWEN38_B70_N` | 3000 | experts on the B70 (3,300 fit in 32 GB) |
| `QWEN38_B70_ORDER` | `score` | `score`: warm-score order after the VRAM set; `rr`: the RAM tier's round-robin rank |
| `QWEN38_B70_RING` | `/run/local-ai/shared/b70.ring` | the ring file, same path in both containers |
| `QWEN38_B70_TIMEOUT_S` | 2 | a B70 reply slower than this stops the server with an error (no silent fallback) |
| `QWEN38_B70_WAIT_S` | 180 | entrypoint wait for the server's ring |
| `QWEN38_B70_SPIN_CPU` (server) | unset | pin the server's polling thread to one CPU; keep it off the bench client's and the CPU lane's cores |

## Tests

These are in `b70tier/tests`:
- `cuda_ref.py`: exllamav3 CUDA reference on real experts (3090).
- `test_ring.py`: numerics and latency through the ring, against a running server.
- `test_client_cpp.py`: the engine's C++ ring client against a running server, no GPU.

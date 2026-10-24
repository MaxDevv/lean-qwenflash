# glm-flash-lite

Run **GLM-5.3-Flash** (125 GB, EXL3 3.05 bpw) on **one 24 GB RTX 3090** with an OpenAI-compatible API, using RAM and
NVMe for the experts that do not fit on the GPU.

| RAM for the server | mode | decode, 1 user | prefill 8k / 32k | output vs stock exllamav3 |
|---|---|---|---|---|
| 218 GiB free | `fast` | 28.2 tok/s | 710 / 951 tok/s | near-exact (decode KL 0.005) |
| **55 GiB cap** + NVMe | **`nvme`** | **19.5 tok/s** | 664 / 969 tok/s | near-exact (decode KL 0.008) |
| 16 GiB cap + NVMe | `nvme` | 14.7 tok/s (lab; 14.38 sweep) | 650 / not run | near-exact (same CPU lane) |
| 55 GiB cap + NVMe | `nvme-exact` | 8.3 tok/s | 564 / 806 tok/s | bit-exact |
| 55 GiB cap + NVMe + 1x Arc Pro B70 | `nvme` + `QWEN38_B70=1` | 26.5 tok/s | 658 / 968 tok/s | near-exact (decode KL 0.003) |

Measured on 1x RTX 3090 (PCIe 4.0 x16), AMD EPYC 7443P (24 cores), 8-channel DDR4, and 4x Samsung 9100 PRO in RAID0.
Decode runs each answer to its natural end. Full tables, dates and raw files: [docs/results.md](docs/results.md).

Image: ghcr.io/sybil-solutions/qwen38-flash-offload (name kept for existing installs)

## Run it

**Omarchy Local AI.** Run the line below. In Local AI, pick **GLM-5.3-Flash, EXL3 3.05 bpw (55 GB RAM, experts on
NVMe)** and press **Start**. The plugin downloads the weights, builds the store and starts the server.

```bash
omarchy plugin add https://github.com/sybil-solutions/omarchy-local-ai --enable
```

**Docker.** Needs the NVIDIA container toolkit, the `hf` CLI, 125 GB for the weights, and 117 GB for the expert store
on a fast local NVMe filesystem (xfs or ext4).

```bash
IMG=ghcr.io/sybil-solutions/qwen38-flash-offload@sha256:aa74200f16c86588b101be2315aed2f643d148559179eef52a02599016b56e69   # v4.6-nvme
hf download turboderp/GLM-5.3-Flash-exl3 --revision 332ab457b709b7ba30dd9a448be5de03b80a7ac9 --local-dir /data/qwen38
docker run --rm -v /data/qwen38:/models:ro -v /mnt/nvme/qwen38:/nvx "$IMG" pack-store     # once: 117 GB expert store
docker run -d --name qwen38 --gpus '"device=0"' --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1 \
  -e QWEN38_MODE=nvme -e QWEN38_EC_MAX_SLOTS=1376 -p 127.0.0.1:30000:30000 -v /data/qwen38:/models:ro -v /mnt/nvme/qwen38:/nvx:ro "$IMG"
```

The server is ready in about 80 s at `http://127.0.0.1:30000/v1`, model `glm-5.3-flash`. For 16 GB of RAM use
`--memory 16g --memory-swap 16g`. For bit-exact output use `QWEN38_MODE=nvme-exact`. The all-RAM `fast` mode uses an
older image; see [docs/results.md](docs/results.md).

## Supported recipes

Every [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) recipe whose launch image is
`ghcr.io/sybil-solutions/qwen38-flash-offload` or `ghcr.io/0xsero/qwen38-flash-offload`, single-GPU only. Only one
hardware card has such recipes: the RTX 3090 24 GB. The GLM-5.3-Flash recipes on other cards (DGX Spark GB10,
RTX PRO 6000 Blackwell, CMP 170HX, Apple M5 Ultra) use other engines and images (vLLM, SGLang, oMLX). No recipe runs
this image on an Arc Pro B70 alone.

Status: **accepted** = the registry lab passed all six gates (load, chat, reasoning, tools, context, speed) on that
digest; **reported** = the lab ran it and a gate failed, or only the owner's campaign measured it; **candidate** = not
in the plugin catalog. Lab rows: prefill is the context gate (one 93,437-token prompt, tokens / request time), decode is
the speed gate (one stream, first 30 s, natural completion, gate 15 tok/s). Sweep rows: `bench/sweep.py`, see
[docs/results.md](docs/results.md). "agg" is the aggregate over all streams.

### RTX 3090 24 GB

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count | recipe | RAM / storage | status | image digest | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 93,437 | 1,015 tok/s | 20.0 tok/s | 1 | 131,072 | 1 | [55 GB + NVMe][r55] | 55 GiB cap; 242.6 GB NVMe | accepted | `aa74200f` (v4.6-nvme) | [lab run 2026-10-09][l55] |
| 8,192 | 664 tok/s | 19.47 tok/s | 1 | 131,072 | 1 | [55 GB + NVMe][r55] | 55 GiB cap; 242.6 GB NVMe | accepted | `aa74200f` (v4.6-nvme) | [sweep N129][res] |
| 8,192 | not run | 19.09 tok/s agg | 2 | 131,072 | 1 | [55 GB + NVMe][r55] | 55 GiB cap; 242.6 GB NVMe | accepted | `aa74200f` (v4.6-nvme) | [sweep N129][res] |
| 8,192 | not run | 19.75 tok/s agg | 4 | 131,072 | 1 | [55 GB + NVMe][r55] | 55 GiB cap; 242.6 GB NVMe | accepted | `aa74200f` (v4.6-nvme) | [sweep N129][res] |
| 32,768 | 969 tok/s | 18.30 tok/s | 1 | 131,072 | 1 | [55 GB + NVMe][r55] | 55 GiB cap; 242.6 GB NVMe | accepted | `aa74200f` (v4.6-nvme) | [sweep N129][res] |
| 32,768 | not run | 18.44 tok/s agg | 2 | 131,072 | 1 | [55 GB + NVMe][r55] | 55 GiB cap; 242.6 GB NVMe | accepted | `aa74200f` (v4.6-nvme) | [sweep N129][res] |
| 93,437 | 1,010 tok/s | 14.7 tok/s | 1 | 131,072 | 1 | [16 GB + NVMe][r16] | 16 GiB cap; 242.6 GB NVMe | reported (speed gate 15) | `aa74200f` (v4.6-nvme) | [lab run 2026-10-09][l16] |
| 8,192 | 650 tok/s | 14.38 tok/s | 1 | 131,072 | 1 | [16 GB + NVMe][r16] | 16 GiB cap; 242.6 GB NVMe | reported | `aa74200f` (v4.6-nvme) | [sweep N129][res] |
| 8,192 | not run | not run | 2 | 131,072 | 1 | [16 GB + NVMe][r16] | 16 GiB cap; 242.6 GB NVMe | reported | `aa74200f` (v4.6-nvme) | not run |
| 8,192 | not run | not run | 4 | 131,072 | 1 | [16 GB + NVMe][r16] | 16 GiB cap; 242.6 GB NVMe | reported | `aa74200f` (v4.6-nvme) | not run |
| 32,768 | not run | not run | 1 | 131,072 | 1 | [16 GB + NVMe][r16] | 16 GiB cap; 242.6 GB NVMe | reported | `aa74200f` (v4.6-nvme) | not run |
| 32,768 | not run | not run | 2 | 131,072 | 1 | [16 GB + NVMe][r16] | 16 GiB cap; 242.6 GB NVMe | reported | `aa74200f` (v4.6-nvme) | not run |
| 93,437 | 989 tok/s | 28.6 tok/s | 1 | 131,072 | 1 | [all experts in RAM][rram] | ~238 GB free RAM; 125.3 GB NVMe | accepted | `c5a58a40` (v4.4-nvme, `fast`) | [lab run 2026-10-09][lram] |
| 8,192 | 710 tok/s | 28.15 tok/s | 1 | 131,072 | 1 | [all experts in RAM][rram] | ~238 GB free RAM; 125.3 GB NVMe | accepted | `bb633b0b` (previous image) | [run G067][res] |
| 8,192 | not run | 31.57 tok/s agg | 2 | 131,072 | 1 | [all experts in RAM][rram] | ~238 GB free RAM; 125.3 GB NVMe | accepted | `bb633b0b` (previous image) | [run G067][res] |
| 8,192 | not run | 33.98 tok/s agg | 4 | 131,072 | 1 | [all experts in RAM][rram] | ~238 GB free RAM; 125.3 GB NVMe | accepted | `bb633b0b` (previous image) | [run G067][res] |
| 32,768 | 951 tok/s | 26.89 tok/s | 1 | 131,072 | 1 | [all experts in RAM][rram] | ~238 GB free RAM; 125.3 GB NVMe | accepted | `bb633b0b` (previous image) | [run G067][res] |
| 32,768 | not run | not run | 2 | 131,072 | 1 | [all experts in RAM][rram] | ~238 GB free RAM; 125.3 GB NVMe | accepted | `bb633b0b` (previous image) | not run |

- RAM / storage: the NVMe recipes run in a container capped at 55 or 16 GiB (the registry asks for 59.1 / 17.2 GB
  free) and need 242.6 GB on a local NVMe filesystem with O_DIRECT (xfs, ext4): 125.3 GB of weights plus the 117.3 GB
  expert store from `pack-store`. All three need an x86-64 CPU with AVX2+FMA+F16C; the NVMe recipes ask for 40 or more
  logical CPUs. The all-RAM recipe needs ~238 GB of free RAM and the 125.3 GB of weights on local NVMe.
- The `fast` sweep rows ran on the previous image `bb633b0b`. The full sweep was not re-run on `c5a58a40`, the digest
  the recipe pins; only the lab run was.
- Image digests: v4.6-nvme `sha256:aa74200f16c86588b101be2315aed2f643d148559179eef52a02599016b56e69` (55 GB recipe
  accepted in local-ai-registry PR [#199](https://github.com/sybil-solutions/local-ai-registry/pull/199));
  v4.4-nvme `sha256:c5a58a40bd93e57227d3986b81e9648d422c465a3ff00e14f17dad65dadf2310`;
  previous all-RAM image `ghcr.io/0xsero/qwen38-flash-offload@sha256:bb633b0bcb85573ad40b6c408af5e1036ae062c593e42479f4e4d2ab521e551d`.

**Hardware cost** (prices dated 2026-10-08, from the earlier campaign): the cheapest build is the 16 GB RTX 3090
build at ~$5.0k, about $340 per tok/s at its 14.7 tok/s lab decode (reported, below the 15 tok/s gate). The 55 GB build
is ~$5.4k, about $270 per tok/s at its accepted 20.0 tok/s.

**Experimental, not supported** (multi-GPU or companion): [glm-5.3-flash.exllamav3-nvme-b70][rb70], the 55 GB NVMe
recipe on an RTX 3090 plus an Intel Arc Pro B70 running the `qwen38-b70-expert-server` companion
(`sha256:8e95f5f9f5e2373d5a80c556cdbc24ae5be8e63dd9f1520bdeb0bc0d02bc5fc5`). It is a candidate with a reported proof
from the owner's campaign N137 (26.5 tok/s C1 decode, 968 tok/s 32k prefill), not lab-run, and the plugin cannot start
companions yet. See [docs/b70-tier.md](docs/b70-tier.md).

[r55]: https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3-nvme.128k.json
[r16]: https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3-nvme16g.128k.json
[rram]: https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3.128k.json
[rb70]: https://github.com/sybil-solutions/local-ai-registry/blob/main/registry/recipes/nvidia/rtx-3090-24gb/glm-5.3-flash.exllamav3-nvme-b70.128k.json
[l55]: https://github.com/sybil-solutions/local-ai-registry/blob/main/lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb.128k.20261009T123730.json
[l16]: https://github.com/sybil-solutions/local-ai-registry/blob/main/lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-16g-128k-rtx-3090-24gb.128k.20261009T124547.json
[lram]: https://github.com/sybil-solutions/local-ai-registry/blob/main/lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-offload-128k-rtx-3090-24gb.128k.20261009T064723.json
[res]: docs/results.md

## How it works

The model has 12,096 routed experts (42 MoE layers x 288), 9.4 MB each, and each token uses 8 per layer. Each expert
is kept in one of three tiers. Two lanes compute them at the same time:

```text
   GPU lane                                       CPU lane
  +-------------------------------+              +---------------------------+
  | VRAM expert cache, ~1,300     |              | 22 AVX2 threads compute   |
  | hottest experts (12.4 GB)     |              | RAM experts in place      |
  +---------------^---------------+              +-------------^-------------+
                  | PCIe copy when admitted                    | reads
  +---------------+--------------------------------------------+-------------+
  | RAM tier, pinned, sized from --memory: ~4,800 experts at 55 GiB          |
  | holds only experts that are NOT in VRAM                                  |
  +---------------^----------------------------------------------------------+
                  | O_DIRECT reads, deep queue, layer-ahead prefetch
  +---------------+----------------------------------------------------------+
  | NVMe store: every expert, 117 GB, one 4K-aligned record each             |
  +--------------------------------------------------------------------------+
```

- **Expert cache in VRAM.** The VRAM left after load holds the hottest experts. exllamav3's own kernels read them
  through repointed tables. On the all-RAM path, cache plus direct reads took decode from 7.5-8.4 tok/s (stock
  exllamav3) to 12.6.
- **RAM tier with no duplicates.** An expert evicted from VRAM moves into RAM, so VRAM and RAM never hold the same
  expert. 55 GiB of RAM then covers ~6,100 different experts instead of ~4,800. Per token at C1: ~120 picks hit
  VRAM, ~175 hit RAM and ~23 come from NVMe.
- **NVMe record store.** Each expert is one 4K-aligned record, read with `O_DIRECT` (no page cache) by 16 threads
  across a 4-drive RAID0. Capping reads at 8 GB/s cost 8 % of C1 decode, 33 % of C4 and 27 % of prefill.
- **CPU tier.** Cold experts already in RAM are computed by the CPU where they sit, instead of being copied over PCIe.
  This runs in parallel with the GPU, which adds the CPU's partial result back. Adding it, together with prefetch,
  took NVMe decode from 8.3 to 15.3 tok/s, at a decode KL of 0.005.
- **Host engine without stalls.** One controller thread plans each layer, issues the reads and answers the GPU
  through mapped memory. The GPU waits on per-expert "landed" flags, with no host sync per layer. Together with the
  no-duplicates RAM tier, this took exact decode from 7.3 to 8.3 tok/s.
- **Layer-ahead prefetch.** The next layer's router runs on the current layer's input, and its NVMe experts are read
  one layer early. NVMe misses fall from 40 to 22 per token, and it is 13 % faster in the same session. An earlier
  "prefetch off is faster" result was drift between sessions.
- **Victim ring.** Evicted experts are parked in a 24-slot VRAM ring. A copy engine drains them to RAM off the
  critical path (~7 GB/s of write-backs at C1).

**Where the time goes now** (v4.6, 55 GiB, C1, 55.8 ms per token): copying admitted experts to VRAM 31 %, non-MoE GPU
work 20 % (KDA/DSA attention GEMVs at 50-57 % of VRAM bandwidth), the MoE kernel 18 %, waiting for the CPU lane 18 %,
NVMe waits 10 %, host 2 % (the decode lookahead removed the step-boundary gaps). No resource is saturated: the GPU does
real work 70 % of the time, the CPU lane 60 %, PCIe and NVMe run at under 40 % of their ceilings, because every layer
waits on its slowest lane. Better scheduling alone would reach about 20 tok/s; every lane at its hardware floor about
41. Profile: moetier `docs/latency-qwen38-v46.md`. Details: [docs/how-it-works.md](docs/how-it-works.md).

## Tune it for your machine

Point your coding agent (Claude Code, Codex, others) at
[`skills/qwen38-offload-setup/SKILL.md`](skills/qwen38-offload-setup/SKILL.md). It checks your GPU, RAM, CPU and NVMe,
picks the mode, builds the store, measures, and tunes one setting at a time.

## More

- [docs/results.md](docs/results.md): every measured table, per mode, with dates and raw files
- [docs/how-it-works.md](docs/how-it-works.md): the mechanisms, how each tier is sized, and every setting
- [docs/b70-tier.md](docs/b70-tier.md): optional Intel Arc Pro B70 as a second expert tier (`QWEN38_B70=1`)
- [docs/troubleshooting.md](docs/troubleshooting.md): error messages and fixes
- [docs/reference.md](docs/reference.md): the full campaign record (image smokes, quality method, reproduction pins)

Credits: [exllamav3](https://github.com/turboderp-org/exllamav3) and the
[EXL3 checkpoint](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) by turboderp; FreeToken (FlashML) for the
host-memory expert tier idea; [SGLang](https://github.com/sgl-project/sglang) as the base image; GLM-5.3-Flash by
Z.ai. MIT license (this repository); the weights carry their own license.

# Lean-QwenFlash

Direct-from-SSD expert offloading for **Qwen3.8-Flash-Next** (125B MoE, 512 experts).

Most setups for this model require 48–64 GB of system RAM just to hold the expert weights in memory. This runtime streams uncached experts directly from NVMe storage over PCIe into a bounded GPU cache via `O_DIRECT`, while streaming the 51B N-gram embedding table from disk. All expert computation runs on the GPU.

Built on [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and adapted from the [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) engine.

---

## Measured Performance

Benchmarked using `turboderp/Qwen3.8-Flash-Next-exl3` (`3.05bpw_h5_ng5`) with 4-bit KV cache:

| Hardware | Context | System RAM Used | VRAM Used | Prefill | Decode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 4090** | 8k | ~8 GB | <16 GB | 1,080 tok/s | 28–33 tok/s (prose)<br>19–22 tok/s (code) |
| **RTX 4090** | 100k | ~8 GB | 15.6 GB | 1,536 tok/s | 20.8 tok/s |
| **RTX 4090** | 262k | ~8 GB | 15.6 GB | 1,627 tok/s | 20.0 tok/s |
| **RTX 3080 Laptop (16GB)** | 100k (8 GB cache) | 19.0 GB | 15.5 GB | 647 tok/s | 17.6 tok/s |
| **RTX 3080 Laptop (16GB)** | 100k (16 GB cache) | 23.9 GB | 15.5 GB | 693 tok/s | 20.9 tok/s |

---

## How It Works

Qwen3.8-Flash-Next activates only 10 of its 512 experts per layer per token (~2% of weights).

* **GPU (16 GB VRAM):** Dense attention mixers, DeltaNet state, routers, shared experts, output head, 4-bit KV cache, and an adaptive GPU expert cache.
* **Host RAM:** Configurable cache (4–16 GB) holding the most recently activated experts.
* **NVMe Storage:** The full ~47 GB packed expert store and 32 GB N-gram table sit on SSD. Cache misses are read directly via asynchronous ring buffers and staged ahead of the compute stream.
* **GPU-Only Math:** All expert computation stays on the GPU. Misses stream over PCIe rather than falling back to CPU execution, avoiding host DRAM bandwidth limits.
* **Prefix Caching:** Recurrent state checkpoints are saved alongside the KV cache so multi-turn agent requests start decoding immediately (~0.8s–1.4s TTFT on cached follow-ups).

---

## Quickstart

### 1. Requirements
* Linux (x86_64)
* NVIDIA GPU with at least 16 GB VRAM
* Fast NVMe SSD (PCIe 3.0 or 4.0, ~3+ GB/s read throughput)
* CUDA 12.4+ and Python 3.10–3.12

### 2. Install

```bash
git clone https://github.com/MaxDevv/lean-qwenflash.git
cd qwen38-flash-offload

python3 -m venv .venv
source .venv/bin/activate
pip install torch fastapi uvicorn safetensors jinja2
pip install -e exllamav3
```

Compile the native offload extensions:
```bash
python compile_local.py
```

### 3. Download & Build Store

Download the 3.05bpw checkpoint and N-gram table (~85 GB total):
```bash
python download_http.py --model-dir ./model
```

Pack the routed experts into a direct-I/O binary store (~47 GB):
```bash
python qwen38-flash-offload/scripts/pack_qwen38_store.py \
  --model ./model \
  --out ./store/qwen.bin
```

---

## Serving

Start the OpenAI-compatible server on port `30000`:

```bash
python qwen_server.py \
  --model ./model \
  --store ./store/qwen.bin \
  --ram-gb 8 \
  --cache 262144 \
  --chunk 24576 \
  --port 30000
```

### Agent Configuration
Point tools like Pi, OpenCode, or Claude Code to:
* **Base URL:** `http://127.0.0.1:30000/v1`
* **Model:** `qwen3.8-flash-next`
* **Context Window:** `262144`

Supports streaming, reasoning effort control (`off`, `low`, `medium`, `xhigh`), and OpenAI function calling.

### CLI Test
To run a one-shot generation from the terminal:

```bash
python infer.py \
  --model ./model \
  --store ./store/qwen.bin \
  --ram-gb 4 \
  --prompt "Write an implementation of merge sort in C."
```

---

## To Do

If I get some free time—or someone wants to contribute or put an agent on it:

- **Lower VRAM:** Adapt and benchmark on 12 GB, and potentially 8 GB GPUs. I expect adapting the existing cache and staging budgets to be relatively straightforward; neither configuration has been tested yet.
- **Higher VRAM:** Benchmark 24 GB and 32 GB GPUs to see how prefill and decode scale as more experts fit in VRAM.

---

## Lineage

* [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp (quantization format & inference runtime).
* [glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) by sybil-solutions (NVMe staging & memory tiering architecture).
* [Qwen3.8-Flash-Next](https://github.com/QwenLM) by Alibaba Cloud Qwen team.

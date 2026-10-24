---
name: qwen38-offload-setup
description: Set up, measure and tune Qwen3.8-Flash-Next (EXL3 3.05 bpw) with qwen38-flash-offload on the user's own Linux machine with one 24 GB NVIDIA GPU, using RAM and NVMe for the experts that do not fit in VRAM. Use when asked to "run Qwen3.8-Flash-Next on my 3090/4090", "set up qwen38-flash-offload" (formerly qwen38-flash-offload), "pick the RAM/NVMe mode", "build the expert store", "why is it slow on my box", or "tune it for my hardware". It detects the hardware, chooses the mode, packs and checks the store, starts the server, measures the standard table, tunes one setting at a time against a control, and reports back. Not for other models, multi-GPU serving, or rented cloud GPUs.
---

# qwen38-flash-offload: set up Qwen3.8-Flash-Next on the user's machine

You set up [qwen38-flash-offload](https://github.com/sybil-solutions/glm-flash-lite) on the machine in front of
you, measure it, tune it, and report how it compares with the reference host. Work step by step. Show the user what you
found before anything slow or destructive.

To learn how the tiers work, read `docs/how-it-works.md` in the repo. For error messages, read `docs/troubleshooting.md`.

## Ground rules

- **Never touch the user's other GPU work.** Before starting, list running containers and GPU processes
  (`docker ps`, `nvidia-smi`). Do not stop, restart or reconfigure any of them without asking. Pick a GPU that is
  idle, or ask which one to use.
- **Ask first** before anything that needs root or is hard to undo:
  - rebooting;
  - BIOS changes (PCIe bifurcation, PCIe generation);
  - creating a RAID (`mdadm`) or a filesystem (`mkfs`), which erases the drives;
  - changing mounts, `/etc/fstab`, or Docker and NVIDIA runtime configuration.

  Never run `mkfs`, `mdadm --create` or `wipefs` on a device until the user has confirmed that exact device name.
- **Never cap output length.** Never set `max_tokens` or `max_new_tokens` on smoke tests or benchmarks. Let every answer
  finish naturally. `bench/sweep.py` already does this.
- **Bound every command that can hang** (docker, ssh, network, anything reading a socket). Use `timeout 20 ...`, and
  treat no answer as "unavailable".
- **Keep the API on 127.0.0.1.** The server has no authentication.
- **Pin everything.** Use the image digest and model revision below, never `latest` or a branch.
- **Report only what you measured.** If a cell was not run, write "not run".

## Pins

| item | value |
|---|---|
| image (NVMe modes) | `ghcr.io/sybil-solutions/qwen38-flash-offload@sha256:aa74200f16c86588b101be2315aed2f643d148559179eef52a02599016b56e69` (v4.6-nvme, repo `0fbc0c5`) |
| image (all-RAM `fast` / `exact`) | `ghcr.io/0xsero/qwen38-flash-offload@sha256:bb633b0bcb85573ad40b6c408af5e1036ae062c593e42479f4e4d2ab521e551d` |
| weights | `turboderp/Qwen3.8-Flash-Next-exl3`, revision `69e33439ae950f17bcbe95c98f117d80f759ab6d` (branch 3.05bpw), 125.3 GB |
| expert store | built from the weights by the image: 117.3 GB, 12,384 records of 9,474,048 B |
| repo (bench scripts, quality panel) | `git clone https://github.com/sybil-solutions/glm-flash-lite` (Python 3, standard library only) |

## Step 1: detect the hardware

Run these and keep the answers. Each one says what it is for.

```bash
# GPU: name, VRAM, driver, PCIe link, and what is already running on it
timeout 20 nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.current --format=csv
timeout 20 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# CPU: logical CPUs, cores, AVX2 / FMA / F16C
nproc; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA node\(s\)'
grep -o -w -E 'avx2|fma|f16c' /proc/cpuinfo | sort -u
# RAM: total and available, swap/zram, and any cgroup limit on this shell
free -g; swapon --show; cat /sys/fs/cgroup/memory.max 2>/dev/null
# DRAM channels (optional, needs root): populated DIMMs
sudo -n dmidecode -t memory 2>/dev/null | grep -E '^\s+Size: [0-9]' | wc -l
# Storage: NVMe drives, filesystems, RAID, free space
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS,ROTA,TRAN
cat /proc/mdstat; findmnt -t xfs,ext4,btrfs,zfs -o TARGET,SOURCE,FSTYPE,OPTIONS
df -h
for d in /sys/class/nvme/nvme*; do echo "$d $(cat $d/model) link $(cat $d/device/current_link_speed 2>/dev/null) x$(cat $d/device/current_link_width 2>/dev/null)"; done
# Docker and the NVIDIA runtime
timeout 10 docker info --format '{{.ServerVersion}} {{.Runtimes}}' ; nvidia-ctk --version
ls -l /etc/cdi/ /var/run/cdi/ 2>/dev/null; timeout 10 nvidia-ctk cdi list 2>/dev/null
command -v hf || command -v huggingface-cli
```

Then answer these questions:

- **GPU.** Is it 24 GB or more, with a driver that supports CUDA 13.0 (580 or newer)? The image is built for sm_86
  (RTX 3090 / A5000); sm_89 (RTX 4090) runs the same binaries, but not measured. Blackwell (RTX 5090, sm_120) needs
  the image rebuilt with `--build-arg TORCH_CUDA_ARCH_LIST="8.6;12.0"`, also not measured. Below 23,000 MiB the
  entrypoint refuses to start.
- **GPU link.** The reference host runs PCIe 4.0 x16. A Gen3 or x8 link slows the admission copies, which are 28 % of
  a decode token on the reference.
- **CPU.** Does it have AVX2, FMA and F16C (needed by the CPU lane in `nvme` and `fast`)? How many physical cores and
  logical CPUs? The measured layout needs 40 logical CPUs (cpus 2-39).
- **RAM.** How much is available for this server, after what the user needs for everything else? Is swap or zram on?
  It is fine to leave it on, as long as the container sets `--memory-swap` equal to `--memory`.
- **NVMe.** Which drive or array is fastest, and is it local NVMe with xfs or ext4? Is there a RAID0? Does it have
  117.3 GB free for the store? The 125.3 GB of weights can go on any disk; only their non-expert part is read, once at
  load.
- **Docker.** Does Docker answer? Is the NVIDIA container toolkit installed, and is the CDI spec current (see
  Troubleshooting)?

## Step 2: choose the mode

**RAM** below means the memory you will give the container (`--memory`). Leave the user's normal working set free.

| situation | mode and cap | expect on the reference host |
|---|---|---|
| ≥ 240 GB free, AVX2, about 24 cores | `fast`, all-RAM image, no cap | 28.2 tok/s C1, 34.0 C4, prefill 710 / 951 |
| needs bit-exact output, ≥ 120 GB free | `exact`, all-RAM image | 12.6 tok/s, bit-exact |
| ≥ 60 GB free + NVMe | `nvme`, `--memory 55g` | 17.3 tok/s C1 (14.9-18.2 across sessions), C4 up to 19.6, prefill 662 / 965 |
| 17-60 GB free + NVMe | `nvme`, `--memory` = what you can spare, at least 15g (the small-host budget applies below 24g) | 16 GiB: 12.9 tok/s C1, prefill 628 |
| bit-exact output in a small cap, or no AVX2 | `nvme-exact`, `--memory 55g` (or less) | 8.3 tok/s, bit-exact |
| under 15 GiB to spare, under 24 GB VRAM, or no NVMe | not supported | the entrypoint refuses |

Notes:

- With 120-240 GB free, `nvme` with a cap of 55g or more beats `exact` (17 vs 12.6 tok/s) unless outputs must be
  bit-exact. Caps above 58 GiB are not measured; 58 vs 55 GiB showed no difference.
- `nvme` decode is near-exact: paired decode KL 0.0047-0.0051, top-1 agreement 0.983-0.986. Prefill is exact in every
  mode.

Reference host: 1x RTX 3090 (PCIe 4.0 x16), EPYC 7443P (24 cores, 48 threads), 8-channel DDR4, and 4x Samsung 9100 PRO
1 TB in md RAID0 (512k chunk, xfs) on a PCIe 4.0 x16 card (~26 GB/s).

**What to expect on weaker hardware.** None of this is measured. Say so to the user, and measure.

- **One NVMe drive instead of a RAID0.** A Gen4 drive peaks around 7 GB/s. On the reference, capping the array at
  8 GB/s cost 8 % of C1, 33 % of C4 and 27 % of prefill at 55 GiB. Expect at least that.
  - At 16 GiB, decode read ~20 GB/s on the reference, so one drive will be much slower there. Give it as much RAM as
    you can.
  - A Gen3 drive (~3.5 GB/s) is worse again.
- **Fewer cores.** The CPU lane runs one thread per physical core (22 on the reference), and it already finishes last
  in 54-86 % of layers. Fewer cores means a slower lane. Measure `nvme` against `nvme-exact`: with very few cores the
  lane may not pay off.
- **2-channel DDR.** At C1 the reference moves ~48 GB/s through DRAM, 35 % of its ~138 GB/s. A 2-channel desktop has a
  much lower peak, so DRAM may become a limit.
- **GPU on PCIe Gen3 or x8.** Admission copies run at ~14.5 GB/s on the reference's Gen4 x16. On a slower link they
  take longer.
- **More VRAM** (32 GB, 48 GB). The cache sizes itself from free VRAM, so you get more slots and more hits. Drop the
  `QWEN38_EC_MAX_SLOTS` cap, or raise it.

Tell the user which mode you chose and why, and what to expect. Then continue.

## Step 3: prepare the store location (only if needed)

The store needs a local NVMe filesystem that supports `O_DIRECT`: plain xfs or ext4 on the drive or on an md RAID0.

- **Not suitable:** a loop file on btrfs, network filesystems, FUSE, or tmpfs. Some LUKS setups refuse `O_DIRECT` too.
- **Check it** with the image's own probe. It needs no GPU:
  ```bash
  IMG=ghcr.io/sybil-solutions/qwen38-flash-offload@sha256:aa74200f16c86588b101be2315aed2f643d148559179eef52a02599016b56e69
  timeout 900 docker pull "$IMG"
  mkdir -p /path/on/nvme/qwen38 && docker run --rm -v /path/on/nvme/qwen38:/nvx "$IMG" python3 /opt/qwen38/docker/preflight.py odirect /nvx && echo O_DIRECT_OK
  ```
- **Several empty NVMe drives** that the user agrees to dedicate: a RAID0 helps most at small RAM caps. Propose
  `mdadm --create /dev/md/qwen38 --level=0 --chunk=512K --raid-devices=N <devices>`, then `mkfs.xfs`, then a mount.
  This erases the drives, so run it only after the user confirms the exact device names.
- **A quad-M.2 PCIe card** needs the slot bifurcated x4x4x4x4 in the BIOS, and the right PCIe generation. If
  `lsblk` shows fewer drives than are installed, tell the user. Do not reboot into the BIOS yourself.

## Step 4: download the weights (pinned)

```bash
W=/path/to/Qwen3.8-Flash-Next-exl3-3.05bpw     # 125.3 GB, any local disk
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 69e33439ae950f17bcbe95c98f117d80f759ab6d --local-dir "$W"
test -f "$W/config.json" && test -f "$W/quantization_config.json" && du -sh "$W"
```

If the `hf` CLI is missing: `pip install -U huggingface_hub`. The download takes a while. Run it in the background and
check progress with `du -sh`.

## Step 5: pack and verify the store

```bash
S=/path/on/nvme/qwen38
docker run --rm -v "$W":/models:ro -v "$S":/nvx "$IMG" pack-store
docker run --rm -v "$W":/models:ro -v "$S":/nvx:ro "$IMG" verify-store
ls -l "$S"    # qwen38_flash_exl3_3.05bpw_experts.bin (117,326,610,432 B) + .json
```

`pack-store` does three things: it writes every record, re-reads each one with `O_DIRECT` and checks its sha256, and
byte-compares 300 sampled records against the checkpoint. It runs on the CPU only. `verify-store` repeats the sha256
pass; it took 11 s on the reference array.

- Any `bad` count above 0 is a failure. Do not serve from that store.
- The `verify-store` speed is a first read on the drive's speed: 6-11 GB/s on the reference. A much lower number
  predicts slow decode.

## Step 6: run

```bash
docker run -d --name qwen38 --gpus '"device=0"' \
  --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1 \
  -e QWEN38_MODE=nvme -e QWEN38_EC_MAX_SLOTS=1376 \
  -p 127.0.0.1:30000:30000 -v "$W":/models:ro -v "$S":/nvx:ro "$IMG"
timeout 600 sh -c 'until curl -sf localhost:30000/health >/dev/null; do sleep 5; done' && echo READY
docker logs qwen38 2>&1 | grep -E '^\[qwen38\]|nv2: rcs|RAM tier|nv2 CPU tier|startup verify|loaded in'
```

Adjust the command:

- **Device.** `device=N` is the idle GPU from step 1.
- **Memory.** `--memory` and `--memory-swap` take the same value: your cap from step 2.
- **Mode.** `QWEN38_MODE` is `nvme` or `nvme-exact`.
- **Expert cache cap.** Keep `QWEN38_EC_MAX_SLOTS=1376` when a desktop runs on the same GPU. On a headless GPU you can
  leave it out.
- **CPU pinning.** With 40 or more logical CPUs, add nothing, or add `--cpuset-cpus 2-39` to fence the server's
  threads off the rest of the system as the reference did. With fewer CPUs, add nothing; the layout is derived.
- **All-RAM modes.** Use the all-RAM image with `--shm-size 16g --ulimit memlock=-1`, no `--memory`, no `/nvx`, and
  `-e QWEN38_MODE=exact` when needed.

Check the startup lines. Write down these values for the report:

- the RAM tier size (`-> RAM tier N slots`)
- the expert-cache slots (`curl -s localhost:30000/stats`, `expert_cache[0].slots`)
- the CPU layout line (`measured layout` or `derived`)
- the startup verify result: the `nv2 startup verify` line should show `'vram_bad': 0` and `'ram_bad': 0`

## Step 7: smoke test

Send one chat request with no output limit, and let it finish:

```bash
curl -s localhost:30000/v1/chat/completions -H 'content-type: application/json' -d '{"model": "qwen3.8-flash-next",
  "messages": [{"role": "user", "content": "Explain in a few paragraphs how a jet engine works."}]}' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); m=r["choices"][0]; print(m["finish_reason"], r.get("usage")); print((m["message"]["content"] or "")[:600])'
```

Pass when `finish_reason` is `stop` and the text reads as coherent English. The reasoning is in
`message.reasoning_content`. To turn thinking off, add `"chat_template_kwargs": {"enable_thinking": false}`.

The server speaks OpenAI Chat Completions, with streaming and `tools`, plus `/v1/completions` and `/v1/models`. Base
URL: `http://127.0.0.1:30000/v1`. Model: `qwen3.8-flash-next`. Any API key works.

## Step 8: measure (the standard table)

From a clone of the repo, with the server idle and nothing else heavy running:

```bash
python3 bench/sweep.py --url http://127.0.0.1:30000 --card mybox --config "v4.6-nvme nvme 55g" --template glm \
  --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2 --no-early-exit --out sweep_A.json
python3 bench/score_ref_panel.py --url http://127.0.0.1:30000 --panel reference/glm-5.3-flash-exl3-ref-panel.json --out score_A.json
```

The sweep measures:

- prefill at 8k and 32k: 3 fresh prompts each, median;
- decode at C1, C2 and C4: chat answers run to their natural end, 2 rounds;
- C1 decode with a 32k-token context.

It takes 30-60 minutes. The panel should give top-1 1.0000 and KL 0, because prefill is exact in every mode.

Report the result in this table. Fill one row per cell, and write "not run" for anything the sweep does not cover.
`sweep.py` has no C2 at 32k.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8,192 | `prefill["8192"].median` | `decode["C1"].aggregate.median` | 1 | 131,072 | 1 |
| 8,192 | - | `decode["C2"]` aggregate (per stream) | 2 | 131,072 | 1 |
| 8,192 | - | `decode["C4"]` aggregate (per stream) | 4 | 131,072 | 1 |
| 32,768 | `prefill["32768"].median` | `decode["C1@32k"]` | 1 | 131,072 | 1 |
| 32,768 | - | not run | 2 | 131,072 | 1 |

Also save `curl -s localhost:30000/stats`. It holds the hit rates, the picks served per token from each tier, and the
container memory peak.

## Step 9: tune one setting at a time

Rules:

- Change one setting per arm.
- Run a control with the current best settings in the same session, right before or after the arm. On the reference,
  arms run hours apart differed by up to 17 % with no change at all.
- To screen, use `--prefill 8192 --conc 1 4` (about 15 minutes). Then run the full table for the winner.
- Restart the container for every arm, because settings are read at start.
- Keep a change only if it beats the same-session control by more than the run-to-run spread, about ±5 %.

| setting | try | why | watch |
|---|---|---|---|
| `QWEN38_EC_MAX_SLOTS` / `QWEN38_EC_RESERVE_GB` | headless GPU: drop the cap. Larger card: let it grow. Shared GPU: keep 1376 and a 1.5 GB reserve | more VRAM slots, more hits | `expert_cache[0].slots`. A CUDA OOM in `graph.cu` means the reserve is too small |
| `--memory` (RAM tier) | as much as the user can spare | more RAM hits, fewer NVMe reads | `nv2.decode_per_token.nvme`, container peak |
| `QWEN38_NV_CPU_CPUS` / `QWEN38_NV_CPU_THREADS` | one thread per physical core, leaving 2 cores for the main and controller threads. Keep readers on the SMT siblings | CPU-lane speed | `nv2.decode_per_token.cpu_busy_ms`, the CPU layout line |
| `QWEN38_NV_PREFETCH` | 1 (default) vs 0 | NVMe misses vs extra reads | `nvme` per token. On the reference, 1 is 13 % faster at C1 in the same session |
| `QWEN38_NV_THREADS` / `QWEN38_NV_PIECE_KB` | 16 x 2304 vs 32-48 x 1024 | queue depth on your drives | `iostat -x 1` read GB/s; reference at 16 GiB: 48 x 1024 gave 12.91 vs 12.57 tok/s |
| `QWEN38_MODE` | `nvme` vs `nvme-exact` on weak CPUs | whether the CPU lane pays off | C1, and exactness if the user needs it |

About prefetch: `QWEN38_NV_PREFETCH` accepts only `0` and `1` in v4.2-nvme. The campaign code had an adaptive mode `2`,
on only at C1, but v4.2 reads `2` as off. Do not use it unless the user runs a build that has it.

**Quality check after any change.** Run the panel (`score_ref_panel.py`); expect 1.0000 / 0. For changes that touch
the CPU lane, also run the paired decode-KL check. Stop the server first, because it needs the GPU. Use the same flags
as the server:

```bash
docker run --rm --gpus '"device=0"' --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1 \
  -e QWEN38_MODE=nvme -e QWEN38_EC_MAX_SLOTS=1376 -v "$W":/models:ro -v "$S":/nvx:ro -v "$PWD":/out "$IMG" \
  decode-kl --kl-out /out/decode_kl.json
```

The reference gives mean KL 0.0047-0.0051 and top-1 0.983-0.986 with the CPU lane on. The control pass gives 0 / 1.0.

## Troubleshooting

| symptom | fix |
|---|---|
| `docker run` names a missing `/dev/nvidiaN` | stale CDI spec: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` (ask first) |
| `does not support O_DIRECT` | move the store to xfs or ext4 on local NVMe (step 3) |
| OOM kill (exit 137), or `memory.swap.max` warning | set `--memory-swap` equal to `--memory`, and keep `--shm-size 1g`. zram or swap otherwise lets pages escape the cap |
| `container memory limit ... needed` | the cap is under 15 GiB, or the image is older than v4.2-nvme |
| decode far below expectation | 1. run `iostat -x 1` during decode, to check the store drive is the fast one and not throttled; 2. check the CPU layout line; 3. look for other GPU or CPU load; 4. compare in the same session (see `docs/troubleshooting.md`) |
| `nv2 ERROR flag` and exit | check `dmesg` for NVMe or PCIe errors and run `verify-store`. If the kernel log shows link faults (`AER`, `pciehp`, `Link Down`), stop GPU and NVMe load and tell the user. Do not loop restarts |
| first answer is slow after a short prompt | expected: the recurrent-state tail forward re-streams experts (~4 s time to first token on short prompts at the reference) |

## Report to the user

Keep it short:

1. **Hardware found.** GPU and link, CPU cores and threads, RAM, store drive or array and filesystem.
2. **Mode and settings.** Mode, `--memory`, image digest, any `-e` you changed, RAM tier slots, cache slots, CPU layout.
3. **The standard table** from step 8, for the final settings.
4. **Against the reference.** Your C1, C4 and prefill next to the reference row for the same mode, with the likely
   reason for any gap: drive speed, cores, DRAM channels, PCIe, or VRAM.
5. **Tuning arms.** Setting, control, arm, result, and kept or not.
6. **Quality.** The panel result, and the decode KL if you ran it.
7. **How to use it.** Base URL `http://127.0.0.1:30000/v1`, model `qwen3.8-flash-next`.

## Optional: contribute the result to the local-ai-registry

The [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) accepts runs from owners' own machines.
Its lab gates are load, chat, reasoning, tools, context and speed (at least 15 tok/s).

```bash
git clone https://github.com/sybil-solutions/local-ai-registry && cd local-ai-registry
python3 lab/lab.py try turboderp/Qwen3.8-Flash-Next-exl3@69e33439ae950f17bcbe95c98f117d80f759ab6d --model qwen3.8-flash-next \
  --engine exllamav3-qwen3.8-flash-next-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb --card rtx-3090-24gb \
  --on endpoint --endpoint http://127.0.0.1:30000 --gpu "RTX 3090"
```

- Run it with `--dry-run` first to see the recipe and launch it will record.
- The run writes evidence to `lab/runs/`. If it passes, it also writes a recipe under `registry/recipes/`.
- The recipe records the image pinned by the engine profile (`registry/launches/<engine>.json`). Serve with that image
  and those flags, so the proof matches what was tested.
- Use the card id that matches the user's GPU (`registry/cards/nvidia/`).
- Open a pull request with the run file and the recipe only if the user agrees to publish.

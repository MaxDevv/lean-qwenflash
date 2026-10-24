# Resumed long-context optimization

The original sandbox expired during the host shutdown. Replacement: `6a5aad0b-ed73-4e8d-b7e1-9eef1a06911d`, RTX4090 spot, 4 vCPUs, 8 GiB RAM, zero swap. Model revision remains `69e33439ae950f17bcbe95c98f117d80f759ab6d` (3.05bpw_h5_ng5). Restored download took 162.18 seconds; store packing took 49 seconds. All 222 sampled checkpoint records matched, and 512 MTP records were appended and verified. Saved compiled extensions were reused.

## Valid 24k-chunk run

Moby Dick source, Q4 KV cache allocated for 262144 tokens, chunk24576, 4 GiB expert RAM, CPU/MTP/prefetch/seeding/RAM-admission off, selective single-pass prefill. Elastic allocator trim enabled. Default native Torch allocator. Launcher: `sandbox_long_chunk24.sh`.

| Actual prompt tokens | Prefill seconds | Subsequent decode tok/s | Generated tokens |
|---:|---:|---:|---:|
| 99982 | 65.106 | 20.772 | 220 |
| 261614 | 160.824 | 20.033 | 219 |
| 261630, three-code retrieval | 159.102 | 10.874 | 21 |

Retrieval recovered all three codes at 10%, 50%, and 90% depth. This is one controlled retrieval test, not general long-context quality validation. Book text is repeated when needed to reach the requested length.

Previous full-cache chunk16384 runs took 76.31/187.485 seconds at 99982/261614 tokens, with 20.75/20.303 tok/s decode. The new profile improves prefill, not decode. The faster ~29 tok/s README runs allocated only 132096 cache tokens and are not comparable full-cache book controls. These are single-run, warm-tier comparisons, not matched cold-state repeated measurements; chunk sizes can change greedy completions.

Decode-step analysis (first streaming iteration excluded because it includes final prefill):

| Prompt tokens | Median ms | P95 ms | Last-40 mean ms |
|---:|---:|---:|---:|
| 99982 | 42.22 | 76.94 | 52.35 |
| 261614 | 43.76 | 84.89 | 47.95 |

Peak total VRAM: 15633285120 bytes, below the 16000000000-byte guard. Zero sampled expert payload/pointer errors and zero cgroup OOM/OOM-kill events. Sampled RAM reached 8589934592 bytes for prose, and 8590159872 bytes for retrieval (225280 bytes above the nominal 8 GiB cgroup cap); the cgroup limit remained enforced. Do not describe this as a strict sampled <=8 GiB result. N-gram reference checks are stored in the result JSON. Full expert arithmetic equivalence remains unestablished, and quantized cache changes outputs.

## Invalid 32k-chunk experiments

All were stopped by the total-VRAM guard; none establishes valid long-context speed:

- Native expandable segments + GC threshold0.75: 16348413952 bytes; Torch allocated11319396352, reserved15200157696 at exit.
- CUDA asynchronous allocator: failed during loading at16730095616 bytes; allocated7150759700, reserved15703474176 at exit.
- Native max_split_size_mb128 + GC threshold0.5: 16346316800 bytes; allocated11319396352, reserved15200157696 at exit.

Unused allocator reservation is evident at exit, but these experiments did not identify the precise allocation responsible or establish a general fix. Retain 24k or smaller chunks rather than treating 32k as supported under the cap.

## Saved artifacts

- `research/4090-long/results/4090-long-chunk24.json`: completed results, per-iteration timings, counters, memory, retrieval and sampled verification.
- `research/4090-resumed-long-artifacts.tar.gz`: restoration logs, download metadata, failed allocator logs/results, completed 24k results/log.
- `research/4090-resume-checkpoint.tar.gz`: earlier restoration checkpoint.

No benchmark process was running at final inspection. The replacement sandbox is retained with restored model/store for subsequent work; its files disappear at expiry.

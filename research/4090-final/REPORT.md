# RTX 4090 spot: completed full-model experiments

## Scope and resources

Required turboderp Qwen3.8-Flash-Next EXL3 `3.05bpw_h5_ng5`, pinned model revision `69e33439ae950f17bcbe95c98f117d80f759ab6d`. All N-gram weights remain disk-offloaded. Single-stream generation; CPU lane disabled except explicit comparisons.

Sandbox `12465044-075f-47f8-9278-2cd989af122e`: RTX 4090, four CPU-equivalents, kernel-enforced **8 GiB RAM**, zero swap. Total-device VRAM guard: **16,000,000,000 bytes**. RAM measurements include file cache. The historical cgroup peak includes setup; sampled inference peaks are reported separately in JSON. No OOM or OOM-kill events occurred throughout these experiments.

Weights and expert store reside at `/var/lib/docker/qwen-opt`, outside the advertised 10 GiB overlay. Direct-I/O storage measurements are in `../sandbox-4090-disk/REPORT.md`: 6.86 GB/s sequential, 6.55 GB/s four-worker expert pattern, 2.60 GB/s single-worker expert pattern. Trunk packing took 50 seconds; 222 checkpoint comparisons had zero mismatches. Appending and verifying 512 MTP experts took 1.22 seconds.

## Measured results

55 benchmark requests plus one real CLI request, totaling **9,989 generated tokens**. Results persisted after each request and recovered locally.

| Measurement | Result |
|---|---:|
| Initial 3 GiB-cache jet-engine decode | 26.28 tok/s |
| Warm 3 GiB-cache jet-engine decode | approximately 29 tok/s |
| Final 4 GiB-cache jet-engine base repetitions | 28.33 / 31.50 tok/s |
| Final matched copy-grid-24 controls | 32.05 / 32.92 tok/s |
| Actual CLI, 251-token natural completion | 28.32 tok/s |
| Synthetic 1k prefill, actual 1006 tokens | 230.31 / 257.53 tok/s |
| Synthetic 8k prefill, actual 8174 tokens | 1286.13 / 1259.76 tok/s |
| README-derived 1k prefill, final configuration | 230.18 / 233.08 tok/s |
| README-derived 8k prefill, final configuration | 1080.95 / 1083.29 tok/s |
| Maximum total VRAM across all runs | 15,719,268,352 bytes |
| Final-profile total VRAM peak | 15,568,273,408 bytes |
| CLI RAM / total VRAM peaks | 8,295,784,448 / 15,561,981,952 bytes |

These are not cold-state throughput comparisons: each fresh generator prevents prefix reuse, but expert tiers remain warm between requests. Initial arithmetic requests include first-use overhead and must not be averaged into steady-state decode. The prefill documents are synthetic: repeated prose or repeated/truncated model README content, not a natural-document benchmark suite. Different documents activate different expert sets.

## Controlled experiments

- **Router prefetch:** off/on/on/off measured 28.37 / 29.29 / 28.97 / 29.21 tok/s. On approximately doubled expert disk bytes per token, from about 50–56 MB to 100 MB, without a compelling speed gain. Leave off.
- **CPU lane:** counterbalanced off/on/on/off measured 29.12 / 18.39 / 21.71 / 28.86 tok/s. CPU altered one completion. Leave off; it is not an exact-arithmetic acceleration.
- **Copy grid:** 24/64/128/64/24 measured 32.05 / 31.50 / 29.40 / 31.45 / 32.92 tok/s on identical greedy completions. Retain 24.
- **Small-prompt staging threshold:** 80 measured 26.33 versus 512 at 29.07 tok/s. Retain 512.
- **Prompt-tail GPU seeding:** with 4 GiB RAM and the same README-derived prompts, enabled prefill was 207–212 tok/s at 1k and 1056–1064 at 8k; disabled was 230–233 and 1081–1083. No justification to enable it here. This comparison is across separate model loads, not a perfectly isolated repeated crossover.
- **Expert RAM:** 4 GiB completed without OOM. Final warm decode improved compared with the 3 GiB configuration, but separate loads/cache histories mean the improvement cannot be attributed solely to RAM capacity.

## MTP: functional, not a universal speedup

Grid order 0/1/3/3/1/0, with three prompts per window. Acceptance and outputs are reproducible across the two passes. First encounters with speculative paths are substantially slower; use the later measurements for the following comparison.

| Prompt | MTP off | Window 1 | Window 3 | Accepted drafts, window 1 / 3 |
|---|---:|---:|---:|---:|
| Jet engine | 28.06 | 26.70 | 23.74 | 74.3% / 48.3% |
| Python LIS code | 19.36 | 19.74 | 20.83 | 96.3% / 84.5% |
| Battery explanation | 22.01 | 22.49 | 19.80 | 74.6% / 48.7% |

Rates are tok/s, not equal-length wall-time comparisons: speculative verification changed greedy completions, consistent with different floating-point batching paths. Acceptance proves that the draft mechanism operates, not that the guessed upstream MTP implementation is reference-equivalent. Keep MTP opt-in; window 3 was modestly beneficial on this coding example but slower on prose.

## Per-token observations and correctness limits

Final base jet runs read **62.44 / 45.27 MB per generated decode token**, with median non-initial step times **29.33 / 28.30 ms**. Faster disk removes the previous Colab bottleneck but does not remove per-layer read latency and GPU/framework work. The physical GPU and RAM sizes also changed, so the entire inference speedup cannot be attributed only to storage.

N-gram forward matched the independent reference bit-for-bit in every benchmark configuration. Sampled RAM/VRAM expert payload, pointer-table and resident-home checks had zero errors after every verified request. Native stale nonresident rows and temporarily duplicated protected RAM/VRAM records are diagnostic, not payload errors. Arithmetic sanity requests returned 391.

These checks **do not establish complete-model logit equivalence against a fully resident reference**. That remains unverified, and neither CPU nor MTP output differences are hidden as exact-equivalence results.

## Tested inference profile

Run on the existing sandbox:

```bash
cd /var/lib/docker/qwen-opt
bash infer_4090.sh --prompt 'Explain how a jet engine works in three concise paragraphs.' --no-thinking --metrics results/my-request.json
```

The wrapper selects 4 GiB expert RAM, 8192-token chunks, disk embeddings, selective single-pass prefill, four 2048 KiB I/O readers, a 128-slot prefill ring, 0.87890625 GiB GPU staging, 0.95 GiB GPU reserve, and the existing 16 GB total-VRAM guard. CPU, MTP, predictor prefetch and experimental cache seeding/admission remain off. Existing generic CLI defaults are preserved; this is an explicit sandbox profile.

Primary JSON: `results/4090-baseline.json`, `4090-mtp.json`, `4090-tuned.json`, `4090-final-sweep.json`, and `4090-cli.json`. Logs and compiled Python-3.12/Torch-2.11/CUDA-12.8 extensions are recovered in `../4090-final-artifacts.tar.gz`. Those extensions are ABI-specific and are not interchangeable with the old Colab Python-3.13 binaries.

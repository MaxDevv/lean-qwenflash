# 8 GiB versus16 GiB expert-RAM comparison

One matched two-arm run on the RTX3080 Laptop in Balanced mode. Both arms use total service RAM cap23900000000 bytes (swap0), NVML total-VRAM guard16000000000 bytes, Q4 KV cache262144, chunk24576,3520 GPU expert slots, four whole-expert disk readers, CPU/MTP/prefetch/seeding/RAM-admission off. Total RAM, not expert cache alone, includes charged filesystem cache; this experiment explicitly exceeds the earlier16GB total-RAM budget at the user's request.

Both arms warmed with arithmetic and prose, then processed the same99982-token Moby Dick prompt followed by the same jet-engine request.

| Expert RAM cache | Prefill seconds | Decode tok/s | Disk MB/token during decode | Recorded GPU NV-read wait ms/token | Peak sampled RAM bytes | Peak total VRAM bytes |
|---:|---:|---:|---:|---:|---:|---:|
|8 GiB|154.45|17.6241|36.7766|12.9783|19029946368|15459090432|
|16 GiB|144.35|20.8841|9.7562|3.7285|23899996160|15475867648|

Decode gain18.50%; outputs exactly identical,233 generated tokens each. The reduction in measured NV-read waits tracks most of the reduction in total time/token. Sampled expert payload/pointer checks and N-gram reference checks passed. Same GPU cache slots in both arms. Only one long-context run per arm, ordered8then16: a meaningful observed improvement, not an extensive repeated significance study or a diverse-context guarantee. NVIDIA power telemetry stayed AC-connected with battery Full.

Chosen server profile:16GiB expert RAM,24GB process-RAM cap,16GB total-VRAM guard, Q4 cache262144, chunk24576, single-stream selective single-pass prefill. The full262k architectural capacity is configured; local full262k retrieval validation is still separate from this100k comparison. Remote full262k retrieval passed previously.

Artifacts: `results/ram8.json`, `results/ram16.json`, `pipeline.log`, `power.jsonl`, and `status.json`.

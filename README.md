# Lean-QwenFlash

A Qwen3.8-Flash-Next adaptation of the GLM-5.3-Flash offload engine, built on ExLlamaV3, demonstrated on actual consumer hardware: an RTX 4090 and an RTX 3080 Laptop.

The runtime keeps bounded expert caches in RAM and VRAM and reads missing experts from SSD. N-gram embeddings also remain disk-backed. The selected inference profiles run expert computation on the GPU rather than using the CPU expert lane.

## Functional-state checkpoint

This is the existing working implementation before model-specific naming and project cleanup. Inference code, defaults and experimental switches are preserved unchanged. The inherited `glm53` names are still present.

- `qwen_offload.py`: Qwen model integration and offload setup.
- `glm53-flash-offload/`: modified native offload engine and Python cache integration.
- `exllamav3/`: model runtime with the optional single-pass prefill change.
- `infer.py`, `benchmark.py`: inference and measurement entry points.
- `qwen_server.py`: OpenAI-compatible Qwen server integration.
- `memory_guard.py`: memory measurement and guards.
- `download_http.py`, `append_mtp.py`: checkpoint download and store support.
- `tests/`, `test_*.py`: store and server tests.

The two existing upstream Git repositories are retained as pinned submodules for this checkpoint, with their modifications committed separately. This records the current working layout, not a decision about the final package architecture. Their configured remotes still identify the upstream projects; the new local commits must be published to fork remotes before a public recursive clone can retrieve them.

## Demonstrated hardware

| Hardware / profile | System RAM | Total VRAM | Decode |
| --- | --- | --- | --- |
| RTX 4090, short-context profile | Kernel-enforced 8 GiB limit, no swap | Below 16 GB guard | Warm prose 28–33 tok/s; coding/science 19–22 tok/s |
| RTX 3080 Laptop, approximately 100k prompt, 8 GiB expert cache | 19.03 GB peak sampled | 15.46 GB | 17.62 tok/s |
| RTX 3080 Laptop, approximately 100k prompt, 16 GiB expert cache | 23.90 GB peak sampled | 15.48 GB | 20.88 tok/s |

The RTX 4090 also completed near-262k-context generation and retrieval. These are separate measured configurations, not interchangeable performance figures. Detailed settings, workload differences and raw measurements are preserved under:

- `research/4090-final/`
- `research/4090-long/`
- `research/local-expert-ram-compare/`

This checkpoint has not rerun inference: the model tensors and derived expert store were previously deleted to reclaim storage.

## Model used

- Repository: `turboderp/Qwen3.8-Flash-Next-exl3`
- Quantization: `3.05bpw_h5_ng5`
- Model revision: `69e33439ae950f17bcbe95c98f117d80f759ab6d`
- Checkpoint including N-grams: approximately 85.14 GB.
- Checkpoint plus packed expert store: approximately 132 GB.

The checkpoint does not include local virtual environments, compiled binaries, downloaded tensors, machine-specific systemd launchers, remote-machine orchestration, or experiment archives. Those local files are left in place. Portable installation and the model-specific cleanup are subsequent work.

## Upstream projects

- https://github.com/sybil-solutions/glm53-flash-offload
- https://github.com/turboderp-org/exllamav3

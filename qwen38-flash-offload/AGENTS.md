# qwen38-flash-offload: agent notes

To set up, measure or tune Qwen3.8-Flash-Next with this repo on a user's own machine, follow
[`skills/qwen38-offload-setup/SKILL.md`](skills/qwen38-offload-setup/SKILL.md). It is plain Markdown with YAML
frontmatter, usable by Claude Code, Codex and other agents.

- Ask before rebooting, changing the BIOS, creating a RAID or filesystem, or touching other GPU services.
- Never cap output length (`max_tokens`) in tests or benchmarks.
- Speed tables use `bench/sweep.py` with the columns prefill size | prefill speed | decode speed | concurrency |
  kv cache | gpu count.
- Mechanisms and settings: `docs/how-it-works.md`. Measured numbers: `docs/results.md`.

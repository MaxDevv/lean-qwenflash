# Local build of the qwen38-flash-offload image from this checkout:
#   docker build -f docker/Dockerfile -t qwen38-flash-offload .
# The published image (ghcr.io/0xsero/qwen38-flash-offload) is built from the same steps by
# github.com/0xSero/local-ai-images (directory qwen38-flash-offload), which fetches this repo at a pinned commit.
#
# Base = the stack the numbers were measured on: lmsysorg/sglang v0.5.20 (CUDA 13.0, torch, Triton, nvcc) +
# exllamav3 v1.5.1 built from source for TORCH_CUDA_ARCH_LIST (8.6 = RTX 3090; sm_86 binaries also run on sm_89).
# For Blackwell (sm_120) rebuild with --build-arg TORCH_CUDA_ARCH_LIST="8.6;12.0" (not measured).
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ninja-build \
 && rm -rf /var/lib/apt/lists/*

# ExLlamaV3 v1.5.1 (MIT, turboderp), exactly the tag the measurements used.
ARG EXLLAMAV3_COMMIT=958ec933361b24eb8426ec7222e5b0062a679dcd
ARG MAX_JOBS=4
RUN git clone https://github.com/turboderp-org/exllamav3 /opt/exllamav3-src \
 && cd /opt/exllamav3-src && git checkout -q ${EXLLAMAV3_COMMIT} \
 && test "$(git rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=${MAX_JOBS} pip install --no-build-isolation --no-deps -v . 2>&1 | tail -3 \
 && pip install --no-deps marisa-trie \
 && pip show exllamav3 >/dev/null \
 && rm -rf /opt/exllamav3-src

COPY . /opt/qwen38
RUN chmod 0755 /opt/qwen38/docker/entrypoint.sh /opt/qwen38/scripts/install.sh \
 && MAX_JOBS=${MAX_JOBS} /opt/qwen38/scripts/install.sh /opt/qwen38

# Data paths; the tuning defaults per mode (fast = G067 with the CPU tier, exact = G066a without) live in the entrypoint.
ENV QWEN38_MODE=fast \
    QWEN38_ZC_STATS=/opt/qwen38/data/stats_own_dec.json QWEN38_EC_WARM=/opt/qwen38/data/stats_own_dec.json \
    QWEN38_TRITON_PIN=/opt/qwen38/data/triton_pin_exact.json \
    EXLLAMAV3_TUNE_CACHE=/opt/qwen38/data/coop_autotune_1gpu.bin TRITON_CACHE_DIR=/opt/qwen38/triton_cache
EXPOSE 30000
ENTRYPOINT ["/opt/qwen38/docker/entrypoint.sh"]

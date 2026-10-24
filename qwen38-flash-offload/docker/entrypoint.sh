#!/bin/bash
# qwen38-flash-offload container entrypoint: preflight checks, model resolve/download, mode defaults, exec the server.
#   docker run ... IMAGE [extra serve.py args]      extra args are appended (argparse: the last value wins)
#   docker run ... IMAGE bash                        any command that is not an option runs as-is
#   docker run ... IMAGE pack-store | verify-store   build / re-check the NVMe expert store (QWEN38_MODE=nvme*), CPU only
set -euo pipefail
ROOT=${QWEN38_ROOT:-/opt/qwen38}

SCRIPT=qwen38/serve.py
if [ "${1:-}" = "decode-kl" ]; then   # paired decode-KL check (bench/decode_kl.py) with the serving config
    SCRIPT=bench/decode_kl.py; shift
elif [ "${1:-}" = "pack-store" ] || [ "${1:-}" = "verify-store" ]; then
    # NVMe expert store: checkpoint at $QWEN38_MODEL_DIR (/models, read-only is fine), store dir at $QWEN38_NV_STORE_DIR
    # (/nvx, writable for pack-store). pack-store = pack + O_DIRECT sha256 verify of every record + byte compare of a
    # sample against the checkpoint; verify-store = the verify step only (~20 s at 6 GB/s).
    P=(python3 "$ROOT/scripts/pack_qwen38_store.py")
    A=(--model "${QWEN38_MODEL_DIR:-/models}" --out-dir "${QWEN38_NV_STORE_DIR:-/nvx}")
    T=${QWEN38_PACK_THREADS:-8}
    O=${QWEN38_NV_STORE_DIR:-/nvx}
    if [ "$1" = "verify-store" ]; then
        [ -r "$O/qwen38_flash_exl3_3.05bpw_experts.bin" ] || { echo "[qwen38] ERROR: no store at $O/qwen38_flash_exl3_3.05bpw_experts.bin; build it with pack-store" >&2; exit 1; }
        python3 "$ROOT/docker/preflight.py" odirect "$O/qwen38_flash_exl3_3.05bpw_experts.bin" || exit 1
    else
        [ -d "$O" ] && [ -w "$O" ] || { echo "[qwen38] ERROR: $O is not a writable directory; mount the store directory there read-write for pack-store" >&2; exit 1; }
        python3 "$ROOT/docker/preflight.py" odirect "$O" || exit 1
    fi
    if [ "$1" = "verify-store" ]; then exec "${P[@]}" verify --threads "$T" "${A[@]}"; fi
    "${P[@]}" pack --threads "$T" "${A[@]}" && "${P[@]}" verify --threads "$T" "${A[@]}" \
        && exec "${P[@]}" cmp --sample "${QWEN38_PACK_SAMPLE:-300}" "${A[@]}"
    exit 1
elif [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    exec "$@"
fi

log() { echo "[qwen38] $*" >&2; }
die() { echo "[qwen38] ERROR: $*" >&2; exit 1; }

# ---- mode defaults; every value can be overridden with -e ---------------------------------------------------------
#   fast  (default) run G067: GPU expert cache + zero-copy + AVX2 CPU tier for cold decode misses, batched decode,
#                   fused hyper-connection sites, fast tier split, shared expert on a side stream
#   exact           run G066a: cache + zero-copy only, no CPU tier, no side-stream shared expert; bit-exact with
#                   stock exllamav3, ~12.6 tok/s decode
#   nvme            55 GB NVMe mode (campaign N119 S3b2): routed experts read from a packed NVMe store (mounted at
#                   /nvx) into a RAM tier sized from the container memory cap (qwen38/nv2.py), GPU expert cache, AVX2 CPU
#                   lane for cold RAM-resident decode picks; needs --memory 55g --memory-swap 55g; prefill exact,
#                   decode not exact (CPU lane)
#   nvme-exact      the same without the CPU lane (N119 S2a): every pick on exllamav3's GPU kernels, bit-exact
#   nvme1/2/3       campaign names: nvme1 = N116 S1 (nv_tier.py only), nvme2 = nvme-exact, nvme3 = nvme without the
#                   shipped tuning defaults
MODE=${QWEN38_MODE:-fast}
NV_MODE=0
case "$MODE" in
    fast)  D_TIER=1; D_RES=1.5; D_K=1; MODE_ARGS=(-ambs 4) ;;
    exact) D_TIER=0; D_RES=1.0; D_K=0; MODE_ARGS=() ;;
    nvme1) D_TIER=0; D_RES=1.0; D_K=0; MODE_ARGS=(); NV_MODE=1   # N116 S1: exact, nv_tier.py (host-stalled NVMe fills)
           export QWEN38_NV=1 ;;
    nvme-exact|nvme2)   # N119 S2: exact, nv2 (device-side stall, exclusive RAM tier), no CPU lane
           D_TIER=0; D_RES=1.0; D_K=0; MODE_ARGS=(); NV_MODE=2
           export QWEN38_NV=2 QWEN38_NV_CPU=${QWEN38_NV_CPU:-0} QWEN38_K_HCFUSE=${QWEN38_K_HCFUSE:-1}
           [ "${QWEN38_MAIN_CPUS+x}" = x ] || QWEN38_MAIN_DEFAULT=24 ;;
    nvme|nvme3)         # N119 S3: nv2 + AVX2 CPU lane on RAM-resident experts, batched decode
           D_TIER=0; D_RES=1.5; D_K=1; MODE_ARGS=(-ambs 4); NV_MODE=2
           export QWEN38_NV=2 QWEN38_NV_CPU=${QWEN38_NV_CPU:-1} QWEN38_K_FTSPLIT=0
           if [ "$MODE" = "nvme" ]; then   # the shipped defaults (campaign arm S3b2; reserve 1.5 GB as nvme3)
               export QWEN38_K_OVL=${QWEN38_K_OVL:-0} QWEN38_NV_VRING=${QWEN38_NV_VRING:-24} QWEN38_NV_PREFETCH=${QWEN38_NV_PREFETCH:-1}
               [ "${QWEN38_MAIN_CPUS+x}" = x ] || QWEN38_MAIN_DEFAULT=24
               export QWEN38_MAX_RQ_TOKENS=${QWEN38_MAX_RQ_TOKENS:-4096}   # page-allocation round, not an output cap: lets C2/C4 run together
               # HOM-272 levers (same-session A/Bs on the RTX 3090, quality class unchanged; each can be set to 0):
               #   QWEN38_NV_CPU_KERN=1  N135 CPU-lane forward (PR #10): C1 +6.2 % at 55 GB, +8.4 % at 16 GB
               #   QWEN38_LA=1, QWEN38_NV_PUBFAST=1, QWEN38_NV_PFSIDE=1 (batch 1 only), QWEN38_LA_BTTRIM=1  N136 (PR #11):
               #     decode lookahead (bit-exact), fast nv_pub, side-stream prefetch guess, MLA block-table trim: C1 +3 %
               export QWEN38_NV_CPU_KERN=${QWEN38_NV_CPU_KERN:-1} QWEN38_LA=${QWEN38_LA:-1} QWEN38_NV_PUBFAST=${QWEN38_NV_PUBFAST:-1}
               export QWEN38_NV_PFSIDE=${QWEN38_NV_PFSIDE:-1} QWEN38_LA_BTTRIM=${QWEN38_LA_BTTRIM:-1}
           fi ;;
    *) die "QWEN38_MODE must be fast, exact, nvme or nvme-exact (got $MODE)" ;;
esac
if [ "$NV_MODE" != "0" ]; then
    export QWEN38_ZC_VRAM=${QWEN38_ZC_VRAM-}           # the NVMe tier replaces the pinned home copy
    export QWEN38_EC_STAGE_MIN=${QWEN38_EC_STAGE_MIN:-512}
    export QWEN38_NV_STORE=${QWEN38_NV_STORE:-/nvx/qwen38_flash_exl3_3.05bpw_experts.bin}
    [ -n "${QWEN38_MAIN_CPUS:-}" ] || unset QWEN38_MAIN_CPUS
    [ "$SCRIPT" = "bench/decode_kl.py" ] && SCRIPT=bench/decode_kl_nv.py   # paired CPU-lane off/on check for nv2
fi
export QWEN38_K_HCFUSE=${QWEN38_K_HCFUSE:-$D_K}
export QWEN38_K_FTSPLIT=${QWEN38_K_FTSPLIT:-$D_K}
export QWEN38_K_OVL=${QWEN38_K_OVL:-$D_K}
export QWEN38_ZC_VRAM=${QWEN38_ZC_VRAM-0}
export QWEN38_ZC_STATS=${QWEN38_ZC_STATS:-$ROOT/data/stats_own_dec.json}
export QWEN38_EC=${QWEN38_EC-1}
export QWEN38_EC_WARM=${QWEN38_EC_WARM:-$ROOT/data/stats_own_dec.json}
export QWEN38_EC_RESERVE_GB=${QWEN38_EC_RESERVE_GB:-$D_RES}
export QWEN38_EC_STAGE_GB=${QWEN38_EC_STAGE_GB:-2.6}
export QWEN38_EC_ELASTIC_GB=${QWEN38_EC_ELASTIC_GB:-10}
export QWEN38_CPU_TIER=${QWEN38_CPU_TIER:-$D_TIER}
export QWEN38_CT_CPUS=${QWEN38_CT_CPUS:-auto}
export QWEN38_CT_THREADS=${QWEN38_CT_THREADS:-0}
export QWEN38_TRITON_PIN=${QWEN38_TRITON_PIN-$ROOT/data/triton_pin_exact.json}
export EXLLAMAV3_TUNE_CACHE=${EXLLAMAV3_TUNE_CACHE:-$ROOT/data/coop_autotune_1gpu.bin}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$ROOT/triton_cache}
export PYTHONUNBUFFERED=1
for k in QWEN38_ZC_VRAM QWEN38_EC QWEN38_TRITON_PIN; do [ -n "${!k}" ] || unset "$k"; done

MODEL=${QWEN38_MODEL_DIR:-/models}
REPO=${QWEN38_MODEL_REPO:-turboderp/Qwen3.8-Flash-Next-exl3}
REV=${QWEN38_MODEL_REVISION:-69e33439ae950f17bcbe95c98f117d80f759ab6d}   # branch 3.05bpw_h5_ng5
PORT=${PORT:-30000}

# ---- preflight --------------------------------------------------------------------------------------------------
command -v nvidia-smi >/dev/null && timeout 20 nvidia-smi -L >&2 || die "no NVIDIA GPU visible (run with --gpus)"
NG=$(timeout 20 nvidia-smi -L | wc -l)
[ "$NG" -eq 1 ] || log "WARNING: $NG GPUs visible; this build serves on ONE GPU (pass --gpus '\"device=N\"')"
VRAM=$(timeout 20 nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "${VRAM:-0}" -ge 23000 ] || die "GPU has ${VRAM} MiB; the measured configuration needs a 24 GB card"

ML=$(ulimit -l)
[ "$ML" = "unlimited" ] || log "WARNING: locked-memory limit is $ML KiB; run with --ulimit memlock=-1 (pinned host experts)"

if [ "$QWEN38_CPU_TIER" = "1" ] || [ "${QWEN38_NV_CPU:-0}" = "1" ]; then
    for f in avx2 fma f16c; do
        grep -qw "$f" /proc/cpuinfo || die "CPU lacks $f; the CPU tier needs AVX2+FMA+F16C (use -e QWEN38_MODE=exact)"
    done
fi
AVAIL_GB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
NEED_GB=$([ "$QWEN38_CPU_TIER" = "1" ] && [ "${QWEN38_CT_SWZ:-1}" != "0" ] && echo 222 || echo 114)   # second (CPU-layout) expert copy or not
# NVMe modes: the RAM tier is sized from the cap (qwen38/nv2.py: cap - process - margin - prefill ring - recurrent-state
# cache), so the floor is the process itself plus a small tier. Measured (HOM-272 N129, 2026-10-08): a 16 GiB cap with the
# small-host budget below gives a 969-expert (8.5 GiB) tier, 13.2 GiB at ready and a 14.4 GiB peak while serving.
SMALL_ARGS=()
if [ "$NV_MODE" != "0" ]; then
    NEED_GB=${QWEN38_NV_MIN_GB:-15}
    if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ] \
       && [ "$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))" -lt "${QWEN38_NV_SMALL_BELOW_GB:-24}" ]; then
        # small-host budget (caps under 24 GiB); every value can be overridden with -e, -rcs with a later -rcs argument.
        # Most decode picks come from NVMe here, so the reader pool is deeper (48 readers x 1 MiB pieces vs 16 x 2304 KiB).
        export QWEN38_NV_MARGIN_GB=${QWEN38_NV_MARGIN_GB:-1.5} QWEN38_NV_PF_RING=${QWEN38_NV_PF_RING:-128}
        export QWEN38_NV_THREADS=${QWEN38_NV_THREADS:-48} QWEN38_NV_PIECE_KB=${QWEN38_NV_PIECE_KB:-1024}
        SMALL_ARGS=(-rcs "${QWEN38_RCS_GB:-1}")   # exllamav3 host recurrent-state cache: 1 GiB instead of 4
        log "small-host NVMe budget (cap < ${QWEN38_NV_SMALL_BELOW_GB:-24} GiB): -rcs ${QWEN38_RCS_GB:-1}, margin ${QWEN38_NV_MARGIN_GB} GiB, prefill ring ${QWEN38_NV_PF_RING} slots, ${QWEN38_NV_THREADS} readers x ${QWEN38_NV_PIECE_KB} KiB"
    fi
fi
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; then
    LIM_GB=$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))
    [ "$LIM_GB" -ge "$NEED_GB" ] || die "container memory limit ${LIM_GB} GiB < ~${NEED_GB} GiB needed (raise --memory or use -e QWEN38_MODE=exact)"
fi
[ "$AVAIL_GB" -ge "$NEED_GB" ] || log "WARNING: MemAvailable ${AVAIL_GB} GiB < ~${NEED_GB} GiB this configuration pins/allocates; expect OOM"
log "mode $MODE: host $(nproc) CPUs visible, MemAvailable ${AVAIL_GB} GiB, GPU ${VRAM} MiB, CPU tier ${QWEN38_CPU_TIER}"
if [ "$NV_MODE" != "0" ]; then
    MAN="${QWEN38_NV_STORE%.bin}.json"
    [ -r "$QWEN38_NV_STORE" ] && [ -r "$MAN" ] || die "NVMe store not found: $QWEN38_NV_STORE (+ $MAN). Build it once with \
'docker run --rm -v <model dir>:/models:ro -v <nvme dir>:/nvx IMAGE pack-store', then mount that dir at /nvx (read-only)"
    if [ "${QWEN38_NV_RAM_GB:-auto}" = "auto" ]; then
        { [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != "max" ]; } \
            || die "QWEN38_MODE=$MODE sizes its RAM tier from the container memory cap: run with --memory 55g --memory-swap 55g (or set QWEN38_NV_RAM_GB)"
    fi
    if [ -r /sys/fs/cgroup/memory.swap.max ] && [ "$(cat /sys/fs/cgroup/memory.swap.max)" != "0" ]; then
        log "WARNING: memory.swap.max is $(cat /sys/fs/cgroup/memory.swap.max): pass --memory-swap equal to --memory, or swapped pages escape the cap"
    fi
    if [ "$NV_MODE" = "2" ]; then   # nv2 pins its threads: the measured layout when the cpuset has it, else derived
        LAYOUT=$(QWEN38_MAIN_DEFAULT=${QWEN38_MAIN_DEFAULT:-} python3 "$ROOT/docker/preflight.py" cpus) || exit 1
        eval "$LAYOUT"
    fi
    python3 "$ROOT/docker/preflight.py" odirect "$QWEN38_NV_STORE" || exit 1
    log "NVMe store $QWEN38_NV_STORE ($(( $(stat -c %s "$QWEN38_NV_STORE") / 1000000000 )) GB), RAM tier ${QWEN38_NV_RAM_GB:-auto}, CPU lane ${QWEN38_NV_CPU:-0}"
fi
# ---- B70 expert tier (QWEN38_B70=1, default off): a B70 expert server (b70tier/b70srv.py, Intel XPU image) must be up
# and own the ring in the shared tmpfs dir; the engine sends it the expert set at warm-up
if [ "${QWEN38_B70:-0}" = "1" ]; then
    { [ "$NV_MODE" = "2" ] && [ "${QWEN38_NV_CPU:-0}" = "1" ]; } || die "QWEN38_B70=1 needs QWEN38_MODE=nvme (the CPU worker runs the B70 lane)"
    RING=${QWEN38_B70_RING:-/run/local-ai/shared/b70.ring}
    for _ in $(seq 1 "${QWEN38_B70_WAIT_S:-180}"); do [ -s "$RING" ] && break; sleep 1; done
    [ -s "$RING" ] || die "QWEN38_B70=1: no B70 expert server ring at $RING (start the B70 server first with the same dir mounted)"
    log "B70 expert tier: ring $RING, ${QWEN38_B70_N:-3000} experts on the B70"
fi

# ---- model ------------------------------------------------------------------------------------------------------
if [ ! -f "$MODEL/config.json" ]; then
    [ "${QWEN38_MODEL_DOWNLOAD:-1}" = "1" ] || die "no model at $MODEL (mount the 3.05bpw checkpoint there)"
    mkdir -p "$MODEL" && [ -w "$MODEL" ] || die "$MODEL is not writable; mount a host directory with ~126 GB free"
    FREE_GB=$(df -Pk "$MODEL" | awk 'NR==2{printf "%d", $4/1048576}')
    [ "$FREE_GB" -ge 120 ] || die "only ${FREE_GB} GiB free at $MODEL; the checkpoint is 117 GiB (125.3 GB)"
    log "downloading $REPO @ $REV into $MODEL (125.3 GB)"
    python3 - "$REPO" "$REV" "$MODEL" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=8)
EOF
fi
[ -f "$MODEL/quantization_config.json" ] || log "WARNING: $MODEL has no quantization_config.json; expected the EXL3 3.05bpw checkpoint"

ARGS=(-m "$MODEL" -cs 131072 --max-batch-size 8 -chunk_size 8192 "${MODE_ARGS[@]}" "${SMALL_ARGS[@]}" --host 0.0.0.0 --port "$PORT"
      --served-name "${SERVED_NAME:-qwen3.8-flash-next}")
# shellcheck disable=SC2206
[ -n "${QWEN38_ARGS:-}" ] && ARGS+=($QWEN38_ARGS)
log "$SCRIPT ${ARGS[*]} $*"
exec python3 "$ROOT/$SCRIPT" "${ARGS[@]}" "$@"

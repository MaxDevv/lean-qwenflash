#!/bin/bash
# P003-ab: same-conditions A/B of decode C1 + prefill 8k: (A) published v4-nvme image, README nvme command;
# (B) campaign S3b2 setup (bb633b0b + N119 s2 code/extension mounts, s3b2 env without GLM53_NV_PFPROF).
#   p003_ab.sh <outdir>
set -u
R=$1; mkdir -p "$R"
cd ~/freetoken-exl3 || exit 1
NAME=p003-glm53-nvme-ab
PORT=30000; URL=http://127.0.0.1:$PORT
MODEL=/home/sero/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw
S=$HOME/freetoken-exl3/runs/N116-glm53-nvme/s2
REPO=runs/P003-glm53-nvme-image-smoke/repo
NEW=ghcr.io/0xsero/glm53-flash-offload@sha256:82eef823f95b89fcb14b8379d45315d3a632aaa054fb04e3418b62a3ec2ff1ff
OLD=ghcr.io/0xsero/glm53-flash-offload@sha256:bb633b0bcb85573ad40b6c408af5e1036ae062c593e42479f4e4d2ab521e551d
K=runs/N116-glm53-nvme/s1/kcheck.sh
log() { echo "[$(date +%T)] $*" | tee -a "$R/ab.log"; }
BASE=(docker run -d --name $NAME --gpus '"device=0"' --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1
      --cpuset-cpus 2-39 --network host -e PORT=$PORT -e GLM53_EC_MAX_SLOTS=1376 -v $MODEL:/models:ro -v /mnt/nvx/glm53:/nvx:ro)
arm() {
  local tag=$1; shift
  ss -ltn | grep -q ":$PORT " && { log "$tag: port busy"; return 1; }
  echo "$*" > "$R/cmd_$tag.txt"
  local t0; t0=$(date +%s); "$@" > /dev/null || { log "$tag: docker run failed"; return 1; }
  local ok=0
  for i in $(seq 1 120); do sleep 5; curl -sf -m 5 $URL/health >/dev/null && { ok=1; break; }; docker ps --format '{{.Names}}' | grep -qx $NAME || break; done
  log "$tag: ready=$ok after $(( $(date +%s) - t0 )) s"
  if [ $ok = 1 ]; then
    taskset -c 40-47 timeout 3600 python3 $REPO/bench/sweep.py --url $URL --card rtx-3090-24gb/glm-5.3-flash-offload --template glm \
      --config "P003-ab $tag" --prefill 8192 --conc 1 --reps 3 --dec-reps 2 --no-early-exit --out "$R/sweep_$tag.json" > "$R/sweep_$tag.log" 2>&1
    log "$tag: $(grep -E '^prefill|^decode' $R/sweep_$tag.log | tr '\n' ' ')"
    curl -s -m 30 $URL/stats > "$R/stats_$tag.json"
  fi
  docker logs $NAME > "$R/server_$tag.log" 2>&1; docker rm -f $NAME > /dev/null; log "$tag: stopped"
}
$K > "$R/kcheck_before.txt" 2>&1 || { log "kcheck TRIPPED"; exit 9; }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$R/gpu_apps.txt"
arm A_image "${BASE[@]}" -e GLM53_MODE=nvme "$NEW"
arm B_campaign "${BASE[@]}" -e GLM53_MODE=nvme3 -e GLM53_MAIN_CPUS=24 -e TORCH_CUDA_ARCH_LIST=8.6 -e GLM53_K_OVL=0 -e GLM53_NV_VRING=24 \
  -e GLM53_NV_PREFETCH=1 -e GLM53_EC_RESERVE_GB=1.5 -e GLM53_MAX_RQ_TOKENS=4096 \
  -v $S/s3b2_55g/code:/opt/glm53/glm53:ro -v $S/build/nv:/opt/glm53/build/nv -v $S/build/nv2b:/opt/glm53/build/nv2 \
  -v $S/s3b2_55g/code/entrypoint.sh:/opt/glm53/docker/entrypoint.sh:ro "$OLD"
$K > "$R/kcheck_after.txt" 2>&1; log "kcheck after: $(tail -1 $R/kcheck_after.txt)"
log "AB_DONE"

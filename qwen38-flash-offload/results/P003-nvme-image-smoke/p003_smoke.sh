#!/bin/bash
# P003: image smoke of the published glm53-flash-offload v4-nvme digest, GLM53_MODE=nvme under --memory 55g, README
# command (host network). Run under bench/gpu0_run.sh + ram_run.sh + cpu_run.sh from ~/freetoken-exl3.
#   p003_smoke.sh <digest sha256:...> <outdir>
set -u
DIG=$1; R=$2; mkdir -p "$R"
IMG=ghcr.io/0xsero/glm53-flash-offload@$DIG
NAME=p003-glm53-nvme-smoke
PORT=30000
URL=http://127.0.0.1:$PORT
MODEL=/home/sero/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw
REPO=$R/repo
K=$HOME/freetoken-exl3/runs/N116-glm53-nvme/s1/kcheck.sh
log() { echo "[$(date +%T)] $*" | tee -a "$R/smoke.log"; }
cpuc() { taskset -c 40-47 "$@"; }
memcg() { local id; id=$(docker inspect -f '{{.Id}}' $NAME 2>/dev/null) || return; local d=/sys/fs/cgroup/system.slice/docker-$id.scope
  echo "current $(cat $d/memory.current) peak $(cat $d/memory.peak 2>/dev/null) max $(cat $d/memory.max) swap.max $(cat $d/memory.swap.max)"; }
memavail() { awk '/MemAvailable/{print $2" kB"}' /proc/meminfo; }

$K > "$R/kcheck_before.txt" 2>&1 || { log "kcheck TRIPPED before start"; cat "$R/kcheck_before.txt"; exit 9; }
log "kcheck before: $(tail -1 $R/kcheck_before.txt)"
docker ps --format '{{.Names}}' | grep -qx $NAME && { log "container $NAME already exists"; exit 8; }
ss -ltn | grep -q ":$PORT " && { log "port $PORT busy"; exit 8; }

# 1. clean anonymous pull by digest
docker image inspect "$IMG" >/dev/null 2>&1 && { log "image already present locally (not a clean pull); removing it"; docker rmi "$IMG" >/dev/null; }
DC=$(mktemp -d); t0=$(date +%s)
DOCKER_CONFIG=$DC timeout 3600 docker pull "$IMG" > "$R/pull.log" 2>&1 || { log "PULL FAILED"; tail -5 "$R/pull.log"; exit 1; }
rm -rf "$DC"; log "anonymous pull ok in $(( $(date +%s) - t0 )) s"
docker image inspect "$IMG" --format '{{index .RepoDigests 0}} {{json .Config.Labels}}' > "$R/image.txt"
docker run --rm --entrypoint cat "$IMG" /opt/glm53/COMMIT > "$R/commit.txt"; log "image COMMIT $(cat $R/commit.txt)"

# 2. store verify (O_DIRECT sha256 of every record)
timeout 900 docker run --rm --name $NAME-verify --cpuset-cpus 40-47 -v $MODEL:/models:ro -v /mnt/nvx/glm53:/nvx:ro "$IMG" verify-store > "$R/verify_store.log" 2>&1
log "verify-store rc=$?: $(tail -1 $R/verify_store.log)"

# 3. serve: the README command
memavail > "$R/mem_before.txt"; nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader > "$R/vram_before.txt"
CMD=(docker run -d --name $NAME --gpus '"device=0"' --memory 55g --memory-swap 55g --shm-size 1g --ulimit memlock=-1
     --cpuset-cpus 2-39 --network host -e PORT=$PORT -e GLM53_MODE=nvme -e GLM53_EC_MAX_SLOTS=1376
     -v $MODEL:/models:ro -v /mnt/nvx/glm53:/nvx:ro "$IMG")
echo "${CMD[*]}" > "$R/cmd.txt"
t0=$(date +%s); "${CMD[@]}" > "$R/cid.txt" || { log "docker run failed"; exit 1; }
ready=0
for i in $(seq 1 120); do
  sleep 5
  docker ps --format '{{.Names}}' | grep -qx $NAME || { log "container exited during load"; docker logs $NAME > "$R/server.log" 2>&1; exit 1; }
  curl -sf -m 5 $URL/health >/dev/null && { ready=1; break; }
done
RT=$(( $(date +%s) - t0 )); log "ready=$ready after $RT s"; echo "ready=$ready after $RT s" > "$R/ready.txt"
[ $ready = 1 ] || { docker logs $NAME > "$R/server.log" 2>&1; docker rm -f $NAME >/dev/null; exit 1; }
memavail > "$R/mem_ready.txt"; memcg > "$R/memcg_ready.txt"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$R/vram_ready.txt"
curl -s -m 30 $URL/server_info > "$R/server_info.json"; curl -s -m 30 $URL/stats > "$R/stats_ready.json"
curl -s -m 30 $URL/v1/models > "$R/models.json"

# 4. functional: chat, no-thinking chat, tool call, streaming (natural completions, no output caps)
cpuc python3 - "$URL" "$R" <<'PY' 2>&1 | tee -a "$R/smoke.log"
import json, sys, urllib.request
url, R = sys.argv[1], sys.argv[2]
def post(path, body):
    r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=3600).read())
gates = {}
o = post("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "What is the capital of Australia?"}], "temperature": 0})
json.dump(o, open(R + "/chat.json", "w")); m = o["choices"][0]["message"]
gates["chat"] = "Canberra" in (m.get("content") or "") and bool(m.get("reasoning_content"))
o = post("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Name three primary colors."}], "temperature": 0,
                                  "chat_template_kwargs": {"enable_thinking": False}})
json.dump(o, open(R + "/chat_nothink.json", "w")); m = o["choices"][0]["message"]
gates["chat_nothink"] = bool(m.get("content")) and not m.get("reasoning_content")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
o = post("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
                                  "tools": tools, "temperature": 0})
json.dump(o, open(R + "/tool.json", "w")); m = o["choices"][0]["message"]
tc = m.get("tool_calls") or []
gates["tool_call"] = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "paris" in tc[0]["function"]["arguments"].lower()
json.dump(gates, open(R + "/gates.json", "w"), indent=1)
print("GATES", gates)
PY
cpuc timeout 3600 python3 $REPO/bench/stream_rate.py --url $URL > "$R/stream_rate.log" 2>&1; log "stream_rate rc=$?: $(tail -2 $R/stream_rate.log | tr '\n' ' ')"

# 5. panel + P2 sweep (protocol of every README table; completions to their natural end)
cpuc timeout 3600 python3 $REPO/bench/score_ref_panel.py --url $URL --panel $REPO/reference/glm-5.3-flash-exl3-ref-panel.json --out "$R/score.json" > "$R/score.log" 2>&1
log "panel: $(tail -1 $R/score.log)"
curl -s -m 60 "$URL/nv_verify?n=64" > "$R/verify_mid.json"; log "nv_verify: $(cat $R/verify_mid.json)"
$K > "$R/kcheck_mid.txt" 2>&1; log "kcheck mid: $(tail -1 $R/kcheck_mid.txt)"
curl -s -m 30 $URL/stats > "$R/stats_before_sweep.json"
cpuc timeout 10800 python3 $REPO/bench/sweep.py --url $URL --card rtx-3090-24gb/glm-5.3-flash-offload --template glm \
  --config "${DIG:7:8} nvme (README command, --memory 55g)" --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2 \
  --no-early-exit --out "$R/sweep.json" > "$R/sweep.log" 2>&1
log "sweep rc=$?"; grep -E "^prefill|^decode" "$R/sweep.log" | tee -a "$R/smoke.log"
curl -s -m 60 "$URL/nv_verify?n=128" > "$R/verify_end.json"; log "nv_verify end: $(cat $R/verify_end.json)"
curl -s -m 30 $URL/stats > "$R/stats_end.json"; memavail > "$R/mem_end.txt"; memcg > "$R/memcg_end.txt"
docker ps --format '{{.Names}}' | grep -qx $NAME || log "SERVER DOWN at end"
docker logs $NAME > "$R/server.log" 2>&1
docker rm -f $NAME >/dev/null && log "stopped $NAME"
$K > "$R/kcheck_after.txt" 2>&1; log "kcheck after: $(tail -1 $R/kcheck_after.txt)"
log "SMOKE_DONE"

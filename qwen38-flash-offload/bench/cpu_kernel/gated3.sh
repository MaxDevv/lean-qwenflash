#!/bin/bash
# N135: queue on ~/gpu3090.lock like a 3090 arm (blocking), then ~/cpu_bench.lock (holds N130 at its next phase boundary),
# then ~/nvx_bench.lock (waits for the running B70 NVMe phase); run SCRIPT only if no 3090 GLM container is up.
# usage: gated3.sh LOG SCRIPT
LOG=$1; SCR=$2
exec 7>~/gpu3090.lock; flock 7
echo "$(date +%T) got gpu3090.lock" >> "$LOG.wait"
exec 9>~/cpu_bench.lock; flock 9
echo "$(date +%T) got cpu_bench.lock" >> "$LOG.wait"
exec 8>~/nvx_bench.lock; flock 8
echo "$(date +%T) got nvx_bench.lock" >> "$LOG.wait"
# busy = any running container that holds an NVIDIA GPU (--gpus: the 3090 arms; B70 servers use /dev/dri and do not
# count), or any NVIDIA compute process other than the Sunshine stream host
busy() { { for c in $(docker ps -q); do docker inspect "$c" --format '{{.Name}} {{len .HostConfig.DeviceRequests}}'; done | awk '$2 > 0 {print $1}'; \
           nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -v sunshine | sed "s/^/gpu-pid-/"; } | tr "\n" " "; }
for i in $(seq 1 60); do
  up=$(busy)
  [ -z "$up" ] && break
  echo "$(date +%T) waiting: 3090 container still up: $up" >> "$LOG.wait"; sleep 10
done
[ -n "$up" ] && { echo "$(date +%T) abort: container up" >> "$LOG.wait"; exit 75; }
{ echo "# $(date -Is) gated3 run: $SCR"; echo "# other containers: $(docker ps --format "{{.Names}}" | tr "\n" " ")"; echo "# load: $(cat /proc/loadavg)"; } >> "$LOG"
"$SCR" >> "$LOG" 2>&1; rc=$?
echo "# rc=$rc end $(date -Is)" >> "$LOG"; echo "$(date +%T) done rc=$rc" >> "$LOG.wait"
exit $rc

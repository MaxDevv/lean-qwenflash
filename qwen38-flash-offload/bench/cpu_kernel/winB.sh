#!/bin/bash
# N135 batch B (target ~7 min). Variants: base = ft_core moe_forward (engine today); v4 = dataflow + p16, block units;
# BA = v4 p16 full-row bands; R2 = v4 ft_core-math register-acc kernel (rt2) on full-row bands; R2B = rt2 on block units
B=./cpubench_g13; D=../data/experts.bin
BA=v4:1:32:128:16:128:0; R2=v4:0:32:128:16:128:1; R2B=v4:0:0:8:0:8:1; V4B=v4:1:0:8:0:8:0
echo "## B1 matrix perf"; $B $D var=base,$V4B,$BA,$R2,$R2B check=0 perf=1 ms=1,2,4 ns=1,3,4,6 iters=200
echo "## B1b auto with I16 up to m=4 for v4 (i16max=4; base keeps its rule)"; $B $D var=base,$R2 check=0 i16max=4 ms=3,4 ns=1,4,6 iters=200
echo "## B2 hot=4"; $B $D var=base,$BA,$R2,$R2B check=0 hot=4 perf=1 ms=1 ns=4 iters=200
echo "## B3 swz=1"; $B $D var=base,$V4B,$R2B check=0 swz=1 perf=1 ms=1 ns=1,4,6 iters=200
echo "## B4 4k pages"; $B $D var=base,$R2 check=0 huge=0 ms=1 ns=4,6 iters=200
echo "## B5 rt2 shapes"; $B $D var=v4:0:16:128:16:128:1,v4:0:64:128:32:128:1,v4:0:32:64:16:64:1,v4:0:32:128:32:256:1,v4:0:32:128:8:128:1,v4:0:32:128:16:256:1,v4:0:8:128:8:128:1 check=0 ms=1,2 ns=4,6 iters=200
echo "## B6 no prologue prefetch"; $B $D var=$R2 check=0 pfpro=0 ms=1 ns=4 iters=200
echo "## B7 threads"; for t in 16 20; do $B $D var=base,$R2 check=0 threads=$t ms=1 ns=4 iters=200 | grep -v "^var" | sed "s/^/t$t,/"; done
echo "## B7 smt 36"; $B $D var=base,$R2 check=0 threads=36 cpus=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39 ms=1 ns=4,6 iters=200
echo "## B8 merged late jobs (late_us=200), R2 knobs"; $B $D var=base,v4late,v4 check=0 late_us=200 kern=0 rg=32 cg=128 rd=16 cd=128 rt=1 ms=1 ns=4,6 iters=200
echo "## B9 readbw"; $B $D readbw=1

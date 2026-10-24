#!/bin/bash
# N135 batch C (~1-2 min): confirm the measured-best default (v4 = p16, bands only at m = 1) vs base, cold, 22 threads
B=./cpubench_g13; D=../data/experts.bin
echo "## C1 default vs base"; $B $D var=base,v4 check=0 perf=1 ms=1,2,3,4 ns=1,3,4,6 iters=300
echo "## C2 default merged late jobs, base+same wait for reference (late_us=0 and 100)"; for lu in 0 100; do $B $D var=base,v4late,v4 check=0 late_us=$lu ms=1,2 ns=4,6 iters=200 | grep -v "^var" | sed "s/^/late$lu,/"; done
echo "## C3 p16 m=1 shape check"; $B $D var=v4:1:64:128:32:128:0:1,v4:1:32:128:32:256:0:1,v4:1:32:128:16:128:0:1 check=0 ms=1 ns=1,4,6 iters=300

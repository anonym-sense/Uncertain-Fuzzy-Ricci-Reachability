#!/bin/bash
JL="${JULIA:-julia}"
cd "$(dirname "$0")/../julia"
for N in 3 4 6 8 10; do for m in none uniform utr; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
  "$JL" +lts --project=. juliareach.jl chain$N $m 256 > ../experiments/logs/chain${N}__jr_$m.log 2>&1 &
done; done
wait; echo CHAIN2_DONE

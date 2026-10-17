#!/bin/bash
# queue of JuliaReach jobs, at most $1 concurrent
MAXP=${1:-8}
JL="${JULIA:-julia}"
cd "$(dirname "$0")/../julia"
jobs_list=()
for b in van_der_pol planar_quad s_curve s_curve_pocket pendulum acc chain1 chain2 chain3 chain4 chain6 chain8 chain10; do
  bud=512; case $b in van_der_pol|planar_quad) bud=1024;; chain*) bud=256;; esac
  for m in none uniform utr; do jobs_list+=("$b $m $bud"); done
done
for j in "${jobs_list[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
  set -- $j
  [ -f ../experiments/results/$1__jr_$2__curvature__path.json ] && continue
  "$JL" +lts --project=. juliareach.jl $1 $2 $3 > ../experiments/logs/$1__jr_$2.log 2>&1 &
done
wait
echo ALL_JR_DONE

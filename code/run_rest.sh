#!/bin/bash
# waits for exp_within, then runs host-only and cross-corpus experiments
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
while pgrep -f exp_within.py >/dev/null; do sleep 20; done
NPROC=4 python3 exp_within_host.py > ../results/within_host.log 2>&1
NPROC=3 python3 exp_cross.py > ../results/cross.log 2>&1
echo ALLDONE >> ../results/cross.log

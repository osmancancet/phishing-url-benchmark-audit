#!/bin/bash
# Controls added in the third internal review round (exp_r3.py), then the final aggregation (finalize.sh).
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
NPROC=4 python3 exp_r3.py > ../results/r3.log
python3 kaggle_www.py > ../results/kaggle_www.log
bash finalize.sh
echo R3_DONE

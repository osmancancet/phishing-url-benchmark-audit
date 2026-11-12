#!/bin/bash
# Feature-attribution analysis added after the third review (exp_shap.py), then the final aggregation.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
{ time NPROC=4 python3 exp_shap.py > ../results/shap.log; } 2>> ../results/shap.log
bash finalize.sh
echo SHAP_DONE

#!/bin/bash
# Full pipeline from raw files to tables (linear and tree models). The CNN runs separately (run_cnn.sh) because it
# takes longer; analyze/make_tables/make_figures are rerun after it. finalize.sh repeats the aggregation and
# writes code digests at the end. Single-threaded BLAS per process.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
python3 env_info.py > ../results/env.log
python3 test_urlfeat.py
python3 data.py > ../results/data.log
python3 audit.py > ../results/audit.log
NPROC=4 python3 exp_within.py > ../results/within.log
python3 exp_probe.py > ../results/probe.log
NPROC=4 python3 exp_extra.py > ../results/extra.log
NPROC=4 python3 exp_r3.py > ../results/r3.log
python3 kaggle_www.py > ../results/kaggle_www.log
NPROC=4 python3 exp_cross.py > ../results/cross.log
python3 explain.py > ../results/explain.log
NPROC=4 python3 exp_shap.py > ../results/shap.log
python3 analyze.py > ../results/analyze.log
python3 make_tables.py
python3 make_figures.py
echo PIPELINE_DONE

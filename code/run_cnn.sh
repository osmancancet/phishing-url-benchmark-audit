#!/bin/bash
# Character CNN (URLNet character branch), then re-aggregation so that numbers.json and the tables include it.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
start=$(date +%s)
NPROC=4 python3 exp_cnn.py > ../results/cnn.log
python3 -c "import json,sys; json.dump({'cnn_hours': ($(date +%s)-$start)/3600}, open('../results/runtime.json','w'))"
python3 analyze.py > ../results/analyze.log
python3 make_tables.py
python3 make_figures.py
echo CNN_DONE

#!/bin/bash
# Final aggregation from the stored per-run results: audit (descriptive, no training), numbers.json, tables, figures,
# then software versions and SHA-256 digests of every code file as they stand at the end of the run.
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
{
date -u +"start %Y-%m-%dT%H:%M:%SZ"
python3 env_info.py > /dev/null
python3 audit.py > ../results/audit.log
python3 analyze.py > ../results/analyze.log
python3 make_tables.py
python3 make_figures.py
python3 env_info.py
date -u +"end %Y-%m-%dT%H:%M:%SZ"
echo FINALIZE_DONE
} > ../results/finalize.log 2>&1

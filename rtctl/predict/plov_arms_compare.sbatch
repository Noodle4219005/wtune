#!/usr/bin/env bash
#SBATCH --job-name=plovagg
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predict/agg/plovagg-%j.out
# Aggregate predictive-PLOV job JOB (default 1074192) into a per-node-step CSV and cross-arm tables.
set -euo pipefail
JOB=${JOB:-1074192}
O=/home/u5727520/wrf-bench/wscore/predict
python3 /home/u5727520/wrf-stack/wtune/rtctl/predict/plov_arms_compare.py "$O" "$JOB" 2.00004e9 "$O/agg/plov_predict_node_steps_$JOB.csv"
sha256sum "$O/agg/plov_predict_node_steps_$JOB.csv"; wc -l "$O/agg/plov_predict_node_steps_$JOB.csv"

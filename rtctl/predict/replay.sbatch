#!/usr/bin/env bash
#SBATCH --job-name=replay
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predict/replay-%j.out
# Causal offline replay of per-rank micro cost predictors over the unstaged karina wtimer traces.
set -u
cd /home/u5727520/wrf-bench/wscore/predict
python3 /home/u5727520/wrf-stack/wtune/rtctl/predict/replay_micro.py replay_micro_$SLURM_JOB_ID.csv

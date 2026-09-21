#!/usr/bin/env bash
#SBATCH --job-name=planTest
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predict/test_wt_plan-%j.out
set -u
cd /home/u5727520/wrf-stack/wtune/plov/tests
T=${TMPDIR:-/tmp}/planTest.$SLURM_JOB_ID; mkdir -p $T
gcc -O2 -Wall -Wextra -I.. test_wt_plan.c -o $T/test_wt_plan -lm || { echo "compile FAILED"; exit 2; }
$T/test_wt_plan; RC=$?
echo "test_rc=$RC"
rm -rf $T
exit $RC

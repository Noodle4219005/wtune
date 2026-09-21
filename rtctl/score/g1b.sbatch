#!/usr/bin/env bash
#SBATCH --job-name=g1b
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/g1b-%j.out
# Gate 1b: per-thread busy fraction inside micro windows from the saved gate-1 perf data (job 1074101).
set -u
G=/home/u5727520/wrf-bench/wscore/g1_V13_1074101/g1
S=/home/u5727520/wrf-stack/wtune/rtctl/score
T=${TMPDIR:-/tmp}/g1b.$SLURM_JOB_ID; mkdir -p $T
declare -A ROLE=([12]=hot [44]=cold [9]=hot [49]=cold [10]=hot [42]=cold [11]=hot [51]=cold)
for r in 12 44 9 49 10 42 11 51; do
  ( zcat $G/perf.r$r.data.gz > $T/r$r.data && perf script -i $T/r$r.data -F tid,time,period,event,ip,sym,dso 2>/dev/null | python3 $S/g1b_threads.py $r ${ROLE[$r]} > $T/r$r.txt; rm -f $T/r$r.data ) &
done
wait
cat $T/r12.txt $T/r44.txt $T/r9.txt $T/r49.txt $T/r10.txt $T/r42.txt $T/r11.txt $T/r51.txt | tee $G/g1b_threads.txt
rm -rf $T

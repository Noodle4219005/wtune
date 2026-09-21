#!/usr/bin/env bash
#SBATCH --job-name=elasticd-test
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=3 --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/ctltest-%j.out
#
# Gate for elasticd before it touches WRF: three fake ranks (elastic_bench, OMP=4 on 2 cores each).
# rank 0 = hot (32-chunk regions back to back), ranks 1-2 = donors (8-chunk region, 70 ms asleep).
# Pass: discovery finds 3 ranks with 4-thread teams; dynamic mode lends and returns (lends>0,
# collides>0, errs=0, restore err=0); hot region median drops vs no controller; donor median rises
# by less than the hot gain.
set -u
module load intel/2024_01_46 2>/dev/null || true
P=/home/u5727520/wrf-stack/wtune/rtctl
D=/home/u5727520/wrf-bench/wscore/ctltest-$SLURM_JOB_ID; mkdir -p "$D"; cd "$D"
icx -O2 -qopenmp "$P/poc/elastic_bench.c" -o elastic_bench || { echo FATAL bench; exit 2; }
gcc -O2 -Wall -Wextra "$P/elasticd.c" -o elasticd || { echo FATAL elasticd; exit 2; }
printf 'rank,score\n0,30\n1,5\n2,5\n' > prof.csv
cat > task.sh <<'EOF'
#!/bin/bash
if [ "$SLURM_PROCID" = 0 ]; then exec ./elastic_bench 32 60 s0.status 0 0 > r0.bench 2>&1
else exec ./elastic_bench 8 60 s$SLURM_PROCID.status 0 70 > r$SLURM_PROCID.bench 2>&1; fi
EOF
chmod +x task.sh
export OMP_NUM_THREADS=4 OMP_PROC_BIND=false KMP_BLOCKTIME=0
for MODE in none dynamic static dynamic2; do
  echo "=== $MODE ==="
  srun -n 3 -c 2 --cpu-bind=cores ./task.sh &
  SP=$!; CP=
  case $MODE in
    dynamic|static) srun --overlap -n 1 -c 12 --cpu-bind=none ./elasticd --exe elastic_bench --omp 4 --rpn 3 --profile prof.csv --start-delay-ms 800 --single-domain --mode $MODE --max-s 40 > ctl.$MODE.out 2>&1 & CP=$! ;;
    dynamic2) srun --overlap -n 1 -c 12 --cpu-bind=none ./elasticd --exe elastic_bench --omp 4 --rpn 3 --profile prof.csv --start-delay-ms 800 --single-domain --mode dynamic --poll-us 2000 --lend-after-ms 6 --hot-run-ms 6 --max-s 40 > ctl.$MODE.out 2>&1 & CP=$! ;;
  esac
  wait $SP; [ -n "$CP" ] && wait $CP
  for r in 0 1 2; do mv r$r.bench $MODE.r$r.bench; done
  for r in 0 1 2; do
    awk -v r=$r -v m=$MODE '/^region/{if($2>=25) w[++n]=$4} END{asort(w); printf "  %s rank%d regions>=25 n=%d median_wall_ms %.2f\n", m, r, n, w[int((n+1)/2)]}' $MODE.r$r.bench
  done
  [ -f ctl.$MODE.out ] && grep -E 'ELASTICD|PLAN|SLOT|TOTAL|RANKCPU' ctl.$MODE.out
done
echo "=== done ==="

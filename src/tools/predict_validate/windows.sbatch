#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --exclude=head
#SBATCH --ntasks-per-node=72
#SBATCH --signal=B:USR1@90
# K=4 sampling windows for the wtune predict validation: win_0 (cold start at t0) and win_1..3
# (restarts from the ground-truth run at 25/50/75%), staged by window_stage.py. All four run in
# this one job on this one node, so K=1 vs K=4 differ only in the number of samples.
# Per window: median s/step over normal steps (step 1 excluded; steps > 1.8x the median are
# radiation steps since the windows write nothing) and r_rad = median(radiation)/median(normal).
set -u
: "${CASE:?CASE required}"
source "$HOME/wtune-scc/env.sh"
ulimit -s unlimited
B=$HOME/wtune-runs/predict-validate/$CASE
cd "$B" || exit 2
R=$(python3 -c "import json;print(json.load(open('stage.json'))['ranks'])")

reap () { pkill -TERM -u "$USER" -x wrf.exe 2>/dev/null; sleep 2; pkill -KILL -u "$USER" -x wrf.exe 2>/dev/null; true; }
WRF_PID=""
on_signal () { echo "WINDOWS_INTERRUPTED case=$CASE sig=$1"; [ -n "$WRF_PID" ] && kill -TERM "$WRF_PID" 2>/dev/null; sleep 5; reap; exit 3; }
trap 'on_signal USR1' USR1; trap 'on_signal TERM' TERM; trap 'on_signal INT' INT

n_pre=$(pgrep -u "$USER" -x wrf.exe | wc -l); reap
echo "PREFLIGHT host=$(hostname -s) reaped=$n_pre"
export OMP_NUM_THREADS=1; unset OMP_PLACES OMP_PROC_BIND
echo "case,window,host,rc,ok,wall_s,n_steps,s_step_normal_median,n_rad,r_rad" > windows_result.csv
for w in win_0 win_1 win_2 win_3; do
  cd "$B/$w" || continue
  rm -f rsl.* wrf.out
  T0=$(date +%s.%N)
  mpirun -np "$R" --map-by "ppr:${R}:node:PE=1" --bind-to core ./wrf.exe > wrf.out 2>&1 &
  WRF_PID=$!; wait "$WRF_PID"; RC=$?; WRF_PID=""
  T1=$(date +%s.%N); reap
  OK=0; grep -aq "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null && OK=1
  python3 - "$CASE" "$w" "$(hostname -s)" "$RC" "$OK" "$(python3 -c "print('%.1f'%($T1-$T0))")" >> "$B/windows_result.csv" <<'PY'
import re, statistics, sys
case, w, host, rc, ok, wall = sys.argv[1:7]
try:
    t = open("rsl.error.0000", errors="ignore").read()
except OSError:
    t = ""
st = [float(x) for x in re.findall(r"Timing for main: time \S+ on domain\s+1:\s+([\d.]+) elapsed", t)][1:]
if st:
    med = statistics.median(st)
    normal = [s for s in st if s <= 1.8 * med]
    rad = [s for s in st if s > 1.8 * med]
    mn = statistics.median(normal)
    print("%s,%s,%s,%s,%s,%s,%d,%.6f,%d,%.3f" % (case, w, host, rc, ok, wall, len(st), mn, len(rad), (statistics.median(rad) / mn) if rad else 0))
else:
    print("%s,%s,%s,%s,%s,%s,0,,0," % (case, w, host, rc, ok, wall))
PY
  echo "WINDOW case=$CASE $w rc=$RC ok=$OK"
done
cat "$B/windows_result.csv"

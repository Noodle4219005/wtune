#!/usr/bin/env bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --exclude=head
#SBATCH --ntasks-per-node=72
#SBATCH --signal=B:USR1@90
# Ground-truth full run for the wtune predict validation. CASE=<name> selects
# ~/wtune-runs/predict-validate/<CASE>, staged by stage.py (case template namelist, probed ANCHOR
# decomposition, io_form 2, restart_interval = run/4). Launch is exactly campaign/sites/scc.sh
# site_launch for a block, omp=1 config. Wall = mpirun launch -> exit. Restarts are kept for the
# K=4 restart-sampling windows; history output is deleted after measuring.
set -u
: "${CASE:?CASE required}"
source "$HOME/wtune-scc/env.sh"
ulimit -s unlimited
D=$HOME/wtune-runs/predict-validate/$CASE
cd "$D" || exit 2
R=$(python3 -c "import json;print(json.load(open('stage.json'))['ranks'])")

reap () { pkill -TERM -u "$USER" -x wrf.exe 2>/dev/null; sleep 2; pkill -KILL -u "$USER" -x wrf.exe 2>/dev/null; true; }
WRF_PID=""
on_signal () {   # USR1 (T-90 s), TERM, INT: stop the run, reap, record, leave
  echo "TRUTH_INTERRUPTED case=$CASE sig=$1 at=$(date +%s)"
  [ -n "$WRF_PID" ] && kill -TERM "$WRF_PID" 2>/dev/null
  sleep 5; reap; exit 3
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT

n_pre=$(pgrep -u "$USER" -x wrf.exe | wc -l); reap
echo "PREFLIGHT host=$(hostname -s) reaped=$n_pre"
rm -f rsl.* wrfout_* wrfrst_* wrf.out

export OMP_NUM_THREADS=1
unset OMP_PLACES OMP_PROC_BIND
T0=$(date +%s.%N)
mpirun -np "$R" --map-by "ppr:${R}:node:PE=1" --bind-to core ./wrf.exe > wrf.out 2>&1 &
WRF_PID=$!
wait "$WRF_PID"; RC=$?
T1=$(date +%s.%N)
WRF_PID=""
WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")
reap

OK=0; grep -aq "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null && OK=1
NRST=$(ls wrfrst_d01_* 2>/dev/null | wc -l)
echo "TRUTH case=$CASE job=$SLURM_JOB_ID host=$(hostname -s) rc=$RC ok=$OK wall_s=$WALL restarts=$NRST"
echo "$CASE,$SLURM_JOB_ID,$(hostname -s),$RC,$OK,$WALL,$NRST" > truth.csv
rm -f wrfout_* 2>/dev/null

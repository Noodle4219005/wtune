#!/usr/bin/env bash
# Autotuner step 2: node-local phase-fingerprint sampler (v2).
#
# One instance runs per node (launched with `srun --overlap -N.. -n.. --ntasks-per-node=1`).
# It observes an already-running, unmodified wrf.exe from OUTSIDE: attach `perf record`
# (user-mode cycles sampling) to every wrf.exe pid on this node, between two step counts,
# then turn the perf.data into a compressed `perf script` text stream. It never sends a
# signal to a wrf.exe process and never uses --wrap or LD_PRELOAD.
#
# Why v2 (job 1074372): on a fully subscribed bulk-synchronous WRF job every wake-up of this
# script preempts some WRF thread, and one delayed thread stalls every rank at the next halo.
# v1 forked awk + pgrep every 0.2 s and cost +13.85% s/step before perf had even attached;
# v1 also asked `perf script` for a `cpu` field that `perf record` does not store, which made
# perf script print nothing. v2:
#   - polls every POLL_S seconds (default 2) with at most one fork per poll,
#   - runs at nice 19 (perf record inherits it),
#   - checks WRF liveness with the `kill -0` builtin (no fork, no /proc scan),
#   - post-processes only after every wrf.exe on this node has exited,
#   - prints pid,tid,time,ip,sym,dso (no cpu) and records lost-sample statistics.
#
# args: RUNDIR OUTDIR FREQ START_STEP STOP_STEP     env: POLL_S (default 2), START_TIMEOUT_S (default 1200)
set -u

RD=$1
OD=$2
FREQ=$3
START_STEP=$4
STOP_STEP=$5
POLL_S=${POLL_S:-2}
START_TIMEOUT_S=${START_TIMEOUT_S:-1200}

renice -n 19 -p $$ > /dev/null 2>&1
H=$(hostname)
mkdir -p "$OD"
META="$OD/meta.$H.txt"

steps () {
  local n
  n=$(grep -c 'Timing for main' "$RD/rsl.out.0000" 2> /dev/null)
  echo "${n:-0}"
}

# Wait for START_STEP (one grep per poll). Give up if the run never reaches it.
SECONDS=0
while [ "$(steps)" -lt "$START_STEP" ]; do
  if [ "$SECONDS" -gt "$START_TIMEOUT_S" ]; then
    echo "META host=$H step $START_STEP not reached after ${START_TIMEOUT_S}s, bailing" >> "$META"
    exit 0
  fi
  sleep "$POLL_S"
done

PIDS=$(pgrep -u "$USER" -x wrf.exe)
if [ -z "$PIDS" ]; then
  echo "META host=$H no wrf.exe pids at step $START_STEP" >> "$META"
  exit 0
fi

alive () {
  local p
  for p in $PIDS; do
    kill -0 "$p" 2> /dev/null && return 0
  done
  return 1
}

for p in $PIDS; do
  r=$(tr '\0' '\n' < /proc/$p/environ 2> /dev/null | awk -F= '$1=="SLURM_PROCID"{print $2}')
  threads=$(ls /proc/$p/task 2> /dev/null | wc -l)
  home=$(grep Cpus_allowed_list /proc/$p/status 2> /dev/null | cut -f2)
  echo "PID host=$H pid=$p rank=${r:-} threads=${threads:-0} home=${home:-}" >> "$META"
done

PIDLIST=$(echo "$PIDS" | paste -sd, -)
echo "START host=$H step=$(steps) epoch=$(date +%s.%N) poll_s=$POLL_S freq=$FREQ" >> "$META"

DATA=/dev/shm/fp.$H.$$.data
perf record -q -F "$FREQ" -e cycles:u -p "$PIDLIST" -o "$DATA" 2> "$OD/perf.$H.err" &
PERF_PID=$!

while [ "$(steps)" -lt "$STOP_STEP" ]; do
  if ! alive; then
    echo "META host=$H wrf.exe exited before step $STOP_STEP (at step $(steps))" >> "$META"
    break
  fi
  sleep "$POLL_S"
done
stop_step_val=$(steps)
kill -INT "$PERF_PID" 2> /dev/null
wait "$PERF_PID" 2> /dev/null
echo "STOP host=$H step=$stop_step_val epoch=$(date +%s.%N)" >> "$META"

# Post-process only once WRF is gone from this node, so it cannot steal cores from timed steps.
while alive; do
  sleep "$POLL_S"
done
echo "WRFEXIT host=$H epoch=$(date +%s.%N)" >> "$META"

perf script -i "$DATA" -F pid,tid,time,ip,sym,dso 2>> "$OD/perf.$H.err" | gzip > "$OD/perf.$H.txt.gz"
perf report -i "$DATA" --stats 2> /dev/null | grep -iE 'LOST|SAMPLE' | sed "s/^/STATS host=$H /" >> "$META"
rm -f "$DATA"

bytes=$(stat -c%s "$OD/perf.$H.txt.gz" 2> /dev/null || echo 0)
lines=$(gunzip -c "$OD/perf.$H.txt.gz" 2> /dev/null | wc -l)
echo "DONE host=$H bytes=$bytes lines=$lines" >> "$META"

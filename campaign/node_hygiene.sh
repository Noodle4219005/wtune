#!/usr/bin/env bash
# campaign/node_hygiene.sh -- per-node orphan wrf.exe reap + busy-CPU census.
#
# scc's Slurm has ProctrackType=pgid and no TaskPlugin. Open MPI 5's prte launcher puts the
# ranks it spawns into their own process group, so a job that ends on TIMEOUT -- or whose
# per-run watchdog kills mpirun -- leaves those ranks invisible to Slurm's job cleanup. A node
# census (2026-09) found orphaned wrf.exe owned by our user on 8/12 compute nodes, running
# 8-20h, each spinning a full core on CPU 64; every later benchmark on those nodes was
# contaminated (pure-MPI runs ~35% slower, hybrid MPI+OpenMP 20-130x slower, from every OpenMP
# barrier waiting on a scheduler time slice shared with the spinner). Called once per node
# (mpirun --map-by ppr:1:node -- see campaign/sites/scc.sh's site_hygiene) by
# strict_campaign.sbatch/temporal_run.sbatch's preflight, post-run, and EXIT/TERM/INT trap
# hygiene.
#
# Usage: node_hygiene.sh <reap|census|reap+census>
#   reap        pkill -TERM then -KILL any $USER-owned wrf.exe on this node; on the local
#               machine (no compute-node wrf.exe of ours) this normally finds nothing.
#               Prints "REAPED_ORPHANS host=<h> n=<k> pids=<p1,p2,...>" only when it found any.
#   census      ~2s /proc/stat sample (two snapshots). Prints one
#               "NODE_DIRTY host=<h> cpu=<c> busy=<pct>" line per CPU >20% busy; silent if none.
#               Then a foreign-process scan (decision 0055, scc work5 incident 2026-09-27: a root
#               VS Code server at 91% of ONE core slowed a 576-rank wrf.exe 5x, and a 2 s
#               per-CPU sample can miss a bursty hog): one
#               "FOREIGN_LOAD host=<h> pid=<p> user=<u> comm=<c> pcpu=<pct>" line per process NOT
#               owned by the invoking user whose ps lifetime %CPU exceeds FOREIGN_PCPU_MIN
#               (default 20) AND whose elapsed time is >= FOREIGN_MIN_ETIME_S (default 30 s: a
#               freshly spawned process -- scc: root check-new-release at 100% with etimes 0 --
#               has a meaningless lifetime %CPU); kernel threads (kthreadd, pid 2, and its
#               children) are ignored.
#               Silent if none. NODE_DIRTY is unchanged by this scan.
#   reap+census reap, then census.
# Every invocation with a valid mode ends with exactly one "HYGIENE_OK host=<h> mode=<mode>" line,
# so the caller can tell "ran on this node and found nothing" from "never ran".
# Always exits 0 -- a dirty/orphaned node is something to report, not something that should
# abort the caller's mpirun dispatch.
set -uo pipefail

MODE=${1:?usage: node_hygiene.sh <reap|census|reap+census>}
HOST=$(hostname -s)

reap () {
  local pids n
  pids=$(pgrep -u "$USER" -x wrf.exe 2>/dev/null) || return 0
  [ -z "$pids" ] && return 0
  n=$(printf '%s\n' "$pids" | wc -l)
  pkill -TERM -u "$USER" -x wrf.exe 2>/dev/null || true
  sleep 3
  pkill -KILL -u "$USER" -x wrf.exe 2>/dev/null || true
  echo "REAPED_ORPHANS host=$HOST n=$n pids=$(printf '%s' "$pids" | tr '\n' ',')"
}

census () {
  local s0 s1
  snap () { awk '/^cpu[0-9]+ / {id=substr($1,4); t=0; for(i=2;i<=NF;i++) t+=$i; idle=$5+$6; print id, t, idle}' /proc/stat; }
  s0=$(snap)
  sleep 2
  s1=$(snap)
  join <(sort -n <<<"$s0") <(sort -n <<<"$s1") | \
    awk -v h="$HOST" '{dt=$4-$2; di=$5-$3; b=(dt>0)?100*(dt-di)/dt:0; if (b>20) printf "NODE_DIRTY host=%s cpu=%s busy=%.0f\n", h, $1, b}'
}

foreign_load () {
  # ps lifetime pcpu (cputime/elapsed) -- a long-lived hog shows up even if it is idle during the
  # 2 s census sample above. uid, not user name: ps truncates long names.
  local me
  me=$(id -u)
  ps -eo pid=,ppid=,uid=,user=,pcpu=,etimes=,comm= 2>/dev/null | \
    awk -v h="$HOST" -v me="$me" -v thr="${FOREIGN_PCPU_MIN:-20}" \
        -v etmin="${FOREIGN_MIN_ETIME_S:-30}" '
      $1 == 2 || $2 == 2 { next }            # kthreadd and kernel threads
      $3 == me { next }                      # our own job user
      $6 + 0 < etmin + 0 { next }            # too young for a meaningful lifetime pcpu
      $5 + 0 > thr + 0 {
        comm = $7; for (i = 8; i <= NF; i++) comm = comm "_" $i
        printf "FOREIGN_LOAD host=%s pid=%s user=%s comm=%s pcpu=%s\n", h, $1, $4, comm, $5
      }'
}

case "$MODE" in
  reap) reap ;;
  census) census; foreign_load ;;
  reap+census) reap; census; foreign_load ;;
  *) echo "node_hygiene.sh: bad mode '$MODE' (want reap|census|reap+census)" >&2; exit 1 ;;
esac
echo "HYGIENE_OK host=$HOST mode=$MODE"
exit 0

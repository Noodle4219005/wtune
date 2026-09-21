#!/usr/bin/env bash
#SBATCH --job-name=wscore1
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:45:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/s1-%j.out
#
# Score stage 1 (no controller): V (pristine wrf.exe, production cyclic, OMP=2, numtiles=13),
# P (PLOV gold reference), E0 readiness candidates (OMP=4 on 2 cores/rank, numtiles 12/16),
# numtiles-only control (V12), and a zero-controller socket-binding pair (Edom/Vdom).
# karina 901x901x45, 224 ranks x 4 nodes, 40 steps, inputs reused from stprep_1070876.
# Pass criteria are pre-registered in knowledge/handoffs/2026-09-13-handoff-wtune-elastic-core-score.md.
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore
STEPS=${STEPS:-40}
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$OUT"; J=$SLURM_JOB_ID
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$")
MIN=$(( (STEPS * DT + 59) / 60 ))
echo "job $J nodes $SLURM_JOB_NODELIST steps $STEPS dt $DT run_minutes $MIN"
echo "wrf.exe sha256 $(sha256sum $W/main/wrf.exe | cut -c1-16)  wrf_wtimer.exe sha256 $(sha256sum $W/main/wrf_wtimer.exe | cut -c1-16)"
echo "wrap symbols: wrf.exe $(nm $W/main/wrf.exe | grep -cE '__wrap_') wrf_wtimer.exe $(nm $W/main/wrf_wtimer.exe | grep -cE '__wrap_')"
echo "sched_migration_cost_ns=$(cat /proc/sys/kernel/sched_migration_cost_ns)"

# rank -> node/cpus map for both bindings (the controller's donor geometry depends on it)
for B in default sockets; do
  BF=; [ $B != default ] && BF="--cpu-bind=$B"
  srun --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 $BF bash -c 'echo $SLURM_PROCID $SLURMD_NODENAME $(grep Cpus_allowed_list /proc/self/status | cut -f2)' | sort -n > "$OUT/cpumap_${B}_$J.txt"
  echo "cpumap $B: $(head -3 "$OUT/cpumap_${B}_$J.txt" | tr '\n' ';') ... $(sed -n '113p' "$OUT/cpumap_${B}_$J.txt")"
done

arm () {  # TAG EXE OMP NUMTILES BIND(default|sockets) PLOV(0|1)
  local TAG=$1 EXE=$2 OMPN=$3 NT=$4 BIND=$5 PLOV=$6
  local D=$OUT/${TAG}_$J
  rm -rf "$D"; mkdir -p "$D"; cd "$D" || return 1
  ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
  for f in "$W"/run/*; do b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$PREP/namelist.input" .
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=$NT," namelist.input
  ln -sf "$EXE" ./wrf.exe
  unset $(env | grep -oE '^WTIMER_[A-Z_0-9]+' | tr '\n' ' ') 2>/dev/null
  if [ "$PLOV" = 1 ]; then set -a; . /home/u5727520/wrf-stack/wtune/plov/production.env; set +a; export WTIMER_DIR="$D" WTIMER_DEBUG_RANK=0; fi
  export OMP_NUM_THREADS=$OMPN
  local BF=; [ "$BIND" != default ] && BF="--cpu-bind=$BIND"
  local T0 T1 RC
  T0=$(date +%s.%N)
  timeout 900 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 $BF ./wrf.exe > wrf.out 2>&1
  RC=$?; T1=$(date +%s.%N)
  local OK N
  OK=$(grep -c "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null); N=$(grep -c "Timing for main" rsl.out.0000 2>/dev/null)
  grep "Timing for main" rsl.out.0000 | awk '{print $9}' > steps.txt
  python3 - "$TAG" "$EXE" "$OMPN" "$NT" "$BIND" "$PLOV" "$OK" "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
tag, exe, omp, nt, bind, plov, ok, rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
w = v[1:] if len(v) > 1 else v
print("RESULT tag=%s exe=%s omp=%s numtiles=%s bind=%s plov=%s ok=%s rc=%s steps=%d med=%.5f med_skip1=%.5f mean_skip1=%.5f sum=%.3f srun_wall=%.1f"
      % (tag, exe.split("/")[-1], omp, nt, bind, plov, ok, rc, len(v), st.median(v) if v else -1, st.median(w) if w else -1,
         st.mean(w) if w else -1, sum(v), float(t1) - float(t0)))
PY
  grep -iE "segmentation|signal|forrtl|backtrace|oversubscri" wrf.out 2>/dev/null | head -3
  cd "$OUT"
}

echo "=== arms ==="
arm V13a  $W/main/wrf.exe        2 13 default 0
arm E0_12 $W/main/wrf.exe        4 12 default 0
arm E0_16 $W/main/wrf.exe        4 16 default 0
arm Edom  $W/main/wrf.exe        4 12 sockets 0
arm Vdom  $W/main/wrf.exe        2 13 sockets 0
arm V12   $W/main/wrf.exe        2 12 default 0
arm P13   $W/main/wrf_wtimer.exe 2 13 default 1
arm V13b  $W/main/wrf.exe        2 13 default 0

echo "=== bit identity vs V13a (final history frame, netCDF) ==="
REF=$(ls "$OUT/V13a_$J"/wrfout_d01_* 2>/dev/null | sort | tail -1)
for X in E0_12 E0_16 Edom Vdom V12 P13 V13b; do
  F=$(ls "$OUT/${X}_$J"/wrfout_d01_* 2>/dev/null | sort | tail -1)
  ( if [ -n "$REF" ] && [ -n "$F" ] && [ "$(basename "$REF")" = "$(basename "$F")" ]; then
      "$W/external/io_netcdf/diffwrf" "$REF" "$F" > "$OUT/${X}_$J/diffwrf_vs_V13a.txt" 2>&1
      NDIFF=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$OUT/${X}_$J/diffwrf_vs_V13a.txt")
      echo "DIFF $X vs V13a file=$(basename "$F") differing_fields=$NDIFF"
    else echo "DIFF $X vs V13a SKIP ref=$(basename "${REF:-none}") x=$(basename "${F:-none}")"; fi ) &
done
wait
echo "=== done ==="

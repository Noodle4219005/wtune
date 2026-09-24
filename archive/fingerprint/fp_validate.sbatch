#!/usr/bin/env bash
#SBATCH --job-name=fpvalid
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=01:00:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/fingerprint/fpvalid-%j.out
#
# Autotuner step 2 validation: does the external perf-record phase fingerprinter
# (fingerprint.py) agree with wtimer's in-process timers, and does attaching perf
# to a pristine wrf.exe (the F arm) cost anything vs the same pristine binary alone
# (the V arm)? ORDER is palindromic (V F F V G V F F V) so any drift across the job
# is split evenly rather than biasing one arm.
#  V = pristine wrf.exe, no sampler                                -> overhead reference
#  F = pristine wrf.exe + sampler                                   -> overhead measurement
#  G = wrf_wtimer.exe (as ./wrf.exe) + sampler + WTIMER_STAGE_*=0    -> validation ground truth
# Env: STEPS, START, STOP, FREQ, ORDER (all overridable).
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore/fingerprint
S=/home/u5727520/wrf-stack/wtune-autotuner/fingerprint
WTIMER_EXE=/home/u5727520/wrf-stack/wtimer/wrf_wtimer.exe
PIXI="pixi run --manifest-path /home/u5727520/wrf-stack/wtune-pixi-env-20260830/pixi.toml"

STEPS=${STEPS:-72}
START=${START:-5}
STOP=${STOP:-68}
FREQ=${FREQ:-100}
# sampler polling interval; srun propagates the environment to sampler_node.sh (job 1074426 at 250 Hz and 2 s
# polling still cost 1.38%, so the final validation uses 100 Hz and 5 s)
export POLL_S=${POLL_S:-5}
ORDER=${ORDER:-"V F F V G V F F V"}
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$")
MIN=$(( (STEPS * DT + 59) / 60 ))

module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0 OMP_NUM_THREADS=2
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

J=$SLURM_JOB_ID
echo "job $J nodes $SLURM_JOB_NODELIST steps $STEPS start $START stop $STOP freq $FREQ order [$ORDER]"

declare -A RUNIDX
V_MEDIANS=()
F_MEDIANS=()
REF=
G_VALIDATION_LINE=

for A in $ORDER; do
  RUNIDX[$A]=$(( ${RUNIDX[$A]:-0} + 1 ))
  TAG=${A}${RUNIDX[$A]}
  D=$OUT/fp_${TAG}_$J
  rm -rf "$D"; mkdir -p "$D"; cd "$D"

  ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
  for f in "$W"/run/*; do
    b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;
      *) [ -e "$b" ] || ln -sf "$f" "$b" ;; esac
  done
  cp "$PREP/namelist.input" .
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=13," namelist.input

  unset $(env | grep -oE '^WTIMER_[A-Z_0-9]+' | tr '\n' ' ') 2>/dev/null

  case "$A" in
    V) ln -sf "$W/main/wrf.exe" ./wrf.exe ;;
    F) ln -sf "$W/main/wrf.exe" ./wrf.exe ;;
    G)
      ln -sf "$WTIMER_EXE" ./wrf.exe
      export WTIMER_DIR="$D" WTIMER_STAGE_MICRO=0 WTIMER_STAGE_RAD=0 WTIMER_STAGE_CUMULUS=0 \
             WTIMER_STAGE_PBL=0 WTIMER_STAGE_SURFACE=0
      ;;
  esac

  T0=$(date +%s.%N)
  timeout 400 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 ./wrf.exe > wrf.out 2>&1 &
  WRF_BG=$!

  SAMPLER_PID=""
  if [ "$A" = F ] || [ "$A" = G ]; then
    mkdir -p "$D/fp"
    sleep 5   # let the WRF step be created before the overlapping sampler step asks for resources
    srun --overlap -N4 -n4 --ntasks-per-node=1 -c1 --mem=2G bash "$S/sampler_node.sh" "$D" "$D/fp" "$FREQ" "$START" "$STOP" &
    SAMPLER_PID=$!
  fi

  wait "$WRF_BG"; RC=$?
  T1=$(date +%s.%N)
  [ -n "$SAMPLER_PID" ] && wait "$SAMPLER_PID"

  grep "Timing for main" rsl.out.0000 | awk '{print $9}' > steps.txt
  RESULT_LINE=$(python3 - "$TAG" "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
tag, rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
try:
    ok = int(any("SUCCESS COMPLETE WRF" in l for l in open("rsl.error.0000")))
except IOError:
    ok = 0
window = v[5:68] if len(v) >= 68 else v
print("RESULT tag=%s rc=%s ok=%d steps=%d med=%.5f mean=%.5f srun_wall=%.1f"
      % (tag, rc, ok, len(v),
         st.median(window) if window else -1,
         st.mean(window) if window else -1,
         float(t1) - float(t0)))
PY
)
  echo "$RESULT_LINE"
  MED=$(echo "$RESULT_LINE" | grep -oE 'med=[-0-9.]+' | cut -d= -f2)
  case "$A" in
    V) V_MEDIANS+=("$MED") ;;
    F) F_MEDIANS+=("$MED") ;;
  esac

  if [ "$A" = F ] || [ "$A" = G ]; then
    EXTRA=""
    [ "$A" = G ] && EXTRA="--wtimer-dir $D"
    FP_OUT=$($PIXI python "$S/fingerprint.py" --sampler-dir "$D/fp" --run-dir "$D" \
      --omp-threads 2 --out "$D/fingerprint" $EXTRA 2>&1)
    fp_rc=$?
    echo "$FP_OUT"
    echo "  fingerprint rc=$fp_rc"
    if [ "$A" = G ]; then
      G_VALIDATION_LINE=$(echo "$FP_OUT" | grep '^VALIDATION' || true)
    fi
  fi

  rm -f wrfout_d01_*_00:00:00
  F_FILE=$(ls wrfout_d01_* 2>/dev/null | sort | tail -1)
  if [ "$A" = V ]; then
    [ -n "$F_FILE" ] && REF=$D/$F_FILE
    echo "  REF=${REF:-none}"
  elif [ -n "$REF" ] && [ -n "$F_FILE" ]; then
    "$W/external/io_netcdf/diffwrf" "$REF" "$F_FILE" > diffwrf_vs_V.txt 2>&1
    echo "DIFF $TAG vs V differing_fields=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' diffwrf_vs_V.txt)"
    rm -f "$F_FILE"
  else
    echo "DIFF $TAG vs V SKIP"
  fi
  cd "$OUT"
done

[ -n "$REF" ] && rm -f "$REF"

echo "=== summary"
V_JOIN=$(IFS=,; echo "${V_MEDIANS[*]:-}")
F_JOIN=$(IFS=,; echo "${F_MEDIANS[*]:-}")
python3 - "$V_JOIN" "$F_JOIN" <<'PY'
import statistics as st
import sys
v_join, f_join = sys.argv[1:3]
v = [float(x) for x in v_join.split(',') if x]
f = [float(x) for x in f_join.split(',') if x]
if v and f:
    v_mm = st.median(v)
    f_mm = st.median(f)
    overhead_pct = (st.mean(f) / st.mean(v) - 1.0) * 100.0
    print("OVERHEAD v_median_of_medians=%.5f f_median_of_medians=%.5f overhead_pct=%.3f" %
          (v_mm, f_mm, overhead_pct))
    print("OVERHEAD pass=%s" % (overhead_pct <= 1.0))
else:
    print("OVERHEAD pass=False reason=missing_V_or_F_medians")
PY
[ -n "$G_VALIDATION_LINE" ] && echo "$G_VALIDATION_LINE"
echo "=== done"

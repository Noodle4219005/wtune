#!/usr/bin/env bash
#SBATCH --job-name=plovpred
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:50:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predict/plovpred-%j.out
#
# Predictive PLOV short-case comparison. V = pristine wrf.exe (reference for diffwrf and s/step).
# R0..R3 use ONE wrapper binary (PBIN) with the production PLOV environment; only
# WTIMER_STAGE_PREDICT differs (0 reactive adaptive, 1 predicted R, 2 + pre-publish, 3 + pre-assignment),
# so any difference is attributable to the mode. Order is palindromic to cancel drift.
# Env: PBIN (predictive PLOV wrapper binary), STEPS (default 40), ORDER.
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore/predict
S=/home/u5727520/wrf-stack/wtune/rtctl/predict
STEPS=${STEPS:-40}
PBIN=${PBIN:?set PBIN to the predictive PLOV wrapper binary}
ORDER=${ORDER:-"V R0 R1 R2 R3 R3b R2b R1b R0b"}
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0 OMP_NUM_THREADS=2
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
J=$SLURM_JOB_ID
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$"); MIN=$(( (STEPS * DT + 59) / 60 ))
echo "job $J nodes $SLURM_JOB_NODELIST steps $STEPS order [$ORDER] pbin sha256 $(sha256sum "$PBIN" | cut -c1-16) wraps=$(nm "$PBIN" | grep -c __wrap_)"
REF=
for A in $ORDER; do
  MODE=${A#R}; MODE=${MODE%b}
  D=$OUT/pp_${A}_$J; rm -rf "$D"; mkdir -p "$D"; cd "$D"
  ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
  for f in "$W"/run/*; do b=$(basename "$f"); case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$PREP/namelist.input" .
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=13," namelist.input
  unset $(env | grep -oE '^WTIMER_[A-Z_0-9]+' | tr '\n' ' ') 2>/dev/null
  if [ "$A" = V ]; then
    ln -sf "$W/main/wrf.exe" ./wrf.exe
  else
    ln -sf "$PBIN" ./wrf.exe
    set -a; . /home/u5727520/wrf-stack/wtune/plov/production.env; set +a
    export OMP_NUM_THREADS=2 WTIMER_DIR="$D" WTIMER_DEBUG_RANK=0 WTIMER_TRACE=1 WTIMER_STAGE_PREDICT=$MODE
  fi
  T0=$(date +%s.%N)
  timeout 300 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 ./wrf.exe > wrf.out 2>&1
  RC=$?; T1=$(date +%s.%N)
  grep "Timing for main" rsl.out.0000 | awk '{print $9}' > steps.txt
  python3 - "$A" "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
tag, rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
ok = int(any("SUCCESS COMPLETE WRF" in l for l in open("rsl.error.0000")))
print("RESULT tag=%s rc=%s ok=%d steps=%d med=%.5f med_skip1=%.5f mean_skip1=%.5f srun_wall=%.1f"
      % (tag, rc, ok, len(v), st.median(v) if v else -1, st.median(v[1:]) if len(v) > 1 else -1,
         st.mean(v[1:]) if len(v) > 1 else -1, float(t1) - float(t0)))
PY
  if [ "$A" != V ]; then
    grep -h "^# micro_calls" wtimer.*.csv 2>/dev/null | sed 's/[a-z_]*=//g; s/#//' | awk '{r+=$3; o+=$4; i+=$5; w+=$6; k+=$7} END{printf "  footer: refused=%d share_out=%d share_in=%d wait_ms=%.1f work_ms=%.1f\n", r, o, i, w, k}'
    python3 "$S/plov_trace_summary.py" "$D" "$A"
  fi
  rm -f wrfout_d01_*_00:00:00
  F=$(ls wrfout_d01_* 2>/dev/null | sort | tail -1)
  if [ "$A" = V ]; then [ -n "$F" ] && REF=$D/$F; echo "  REF=${REF:-none}"
  elif [ -n "$REF" ] && [ -n "$F" ]; then
    "$W/external/io_netcdf/diffwrf" "$REF" "$F" > diffwrf_vs_V.txt 2>&1
    echo "DIFF $A vs V differing_fields=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' diffwrf_vs_V.txt)"
    rm -f "$F"
  else echo "DIFF $A vs V SKIP"; fi
  cd "$OUT"
done
[ -n "$REF" ] && rm -f "$REF"
echo "=== done"

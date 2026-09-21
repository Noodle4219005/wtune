#!/usr/bin/env bash
#SBATCH --job-name=tileprobe
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:30:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predict/tileprobe-%j.out
#
# Tile-level microphysics cost map: pristine V reference vs the research wrapper binary with
# WTIMER_TILE_PROBE=1 (one driver call per tile, timed; staging off). Same inputs, production OMP=2,
# numtiles=13, cyclic placement, STEPS steps. Bit-identity of TP against V is checked on the final frame.
# Env: TPBIN (probe binary), STEPS (default 132).
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore/predict
STEPS=${STEPS:-132}
TPBIN=${TPBIN:?set TPBIN to the tile-probe wrapper binary}
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0 OMP_NUM_THREADS=2
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
J=$SLURM_JOB_ID
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$"); MIN=$(( (STEPS * DT + 59) / 60 ))
echo "job $J nodes $SLURM_JOB_NODELIST steps $STEPS tpbin $TPBIN sha256 $(sha256sum "$TPBIN" | cut -c1-16) wraps=$(nm "$TPBIN" | grep -c __wrap_)"

arm () {  # TAG EXE PROBE(0|1)
  local TAG=$1 EXE=$2 PROBE=$3 D=$OUT/tp_${1}_$J
  rm -rf "$D"; mkdir -p "$D"; cd "$D" || return 1
  ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
  for f in "$W"/run/*; do b=$(basename "$f"); case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$PREP/namelist.input" .
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=13," namelist.input
  ln -sf "$EXE" ./wrf.exe
  unset $(env | grep -oE '^WTIMER_[A-Z_0-9]+' | tr '\n' ' ') 2>/dev/null
  if [ "$PROBE" = 1 ]; then export WTIMER_TILE_PROBE=1 WTIMER_STAGE_MICRO=0 WTIMER_DIR="$D"; fi
  local T0=$(date +%s.%N)
  timeout 900 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 ./wrf.exe > wrf.out 2>&1
  local RC=$? T1=$(date +%s.%N)
  grep "Timing for main" rsl.out.0000 | awk '{print $9}' > steps.txt
  python3 - "$TAG" "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
tag, rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
ok = int(any("SUCCESS COMPLETE WRF" in l for l in open("rsl.error.0000")))
print("RESULT tag=%s rc=%s ok=%d steps=%d med=%.5f srun_wall=%.1f" % (tag, rc, ok, len(v), st.median(v) if v else -1, float(t1) - float(t0)))
PY
  [ "$PROBE" = 1 ] && echo "  tile files: $(ls tiles.*.csv 2>/dev/null | wc -l)  rows(rank0): $(($(wc -l < tiles.0.csv 2>/dev/null || echo 1) - 1))"
  rm -f wrfout_d01_*_00:00:00
  cd "$OUT"
}

arm V  "$W/main/wrf.exe" 0
arm TP "$TPBIN" 1

FV=$(ls "$OUT/tp_V_$J"/wrfout_d01_* 2>/dev/null | sort | tail -1)
FT=$(ls "$OUT/tp_TP_$J"/wrfout_d01_* 2>/dev/null | sort | tail -1)
if [ -n "$FV" ] && [ -n "$FT" ] && [ "$(basename "$FV")" = "$(basename "$FT")" ]; then
  "$W/external/io_netcdf/diffwrf" "$FV" "$FT" > "$OUT/tp_TP_$J/diffwrf_vs_V.txt" 2>&1
  echo "DIFF TP vs V differing_fields=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$OUT/tp_TP_$J/diffwrf_vs_V.txt") file=$(basename "$FT")"
  rm -f "$FV" "$FT"
else
  echo "DIFF TP vs V SKIP v=$(basename "${FV:-none}") tp=$(basename "${FT:-none}")"
fi
python3 /home/u5727520/wrf-stack/wtune/rtctl/predict/replay_tiles.py "$OUT/tp_TP_$J" "$OUT/tiles_$J"
echo "=== done"

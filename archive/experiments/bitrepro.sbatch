#!/usr/bin/env bash
#SBATCH --job-name=bitrepro
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=112 --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/bitrepro/bitrepro-%j.out
#
# QUESTION
#   WRF is bit-identical across some decomposition changes and not others.  Which single
#   dimension of the decomposition breaks bit-for-bit reproducibility?
#
#   Evidence in hand:
#     job 1081239 karina 24 h  : 224x2 nproc 8x28 tiles 12  vs  448x1 nproc 8x56 tiles 8
#                                -> BIT-IDENTICAL, 200/200 variables
#     job 1081569 sc26p1       : 56x2 8x7 vs 28x4 7x4 -> BIT-IDENTICAL
#     job 1080697 PHYSICS-GRAY : C02 448x1 tiles 8 vs C06 224x2 tiles 12, SAME karina case
#                                -> NONZERO (T rms 0.0013, PSFC 0.058)
#   1081239 and 1080697 compare the SAME decomposition pair on the SAME case and disagree,
#   so this job changes exactly ONE dimension at a time from one fixed base.
#
# METHOD
#   One 20-minute (80-step, dt=15) karina forecast per arm, two history frames (t=0 from
#   history_begin=0, t=20 from history_interval=20 -- the same two-frame trick
#   rk_alternation.sbatch uses).  Only the FINAL frame is converted BP -> netCDF and
#   compared EXACTLY (bitcmp.py, no tolerance) against the base arm's final frame.
#
#   20 minutes rather than the 60 first sketched: a decomposition that changes the
#   arithmetic changes it at step 1, not at step 200, so the run only has to be long
#   enough for a 1-ULP difference to survive into the float32 wrfout.  80 steps covers two
#   RRTMG calls (radt=15) and ~1e4 Thompson microphysics column updates per rank.  The
#   shorter run is what keeps the whole job inside the 25 SU budget.
#
# WHY THESE ARMS
#   Lining the two karina pairings up side by side already narrows it to one key.  The
#   448-rank arm is the SAME configuration in both jobs (8x56, numtiles 8, 1 thread,
#   adios2_numaggregators 448).  The 224-rank arm is the same in ranks, threads, numtiles
#   and aggregators -- and differs ONLY in nproc_x/nproc_y:
#       1080697 twin  nproc 7x32   -> every patch i-extent 129   -> NONZERO
#       1081239 ref   nproc 8x28   -> every patch i-extent 113   -> ZERO
#   So numtiles (8 vs 12), threads (1 vs 2), rank count (448 vs 224) and aggregator count
#   all move IDENTICALLY in both pairings and are therefore already exonerated by those two
#   jobs; the surviving suspect is nproc_x, i.e. re-cutting the domain in the I direction.
#   This job tests that directly instead of assuming it.
#
#   nproc_x * nproc_y = ranks, so "x only at fixed ranks" is not a legal arm.  The x and y
#   axes are separated by holding ONE of them fixed and letting the rank count follow:
#
#   BASE    56 ranks x 2 thr  nproc 8x7   numtiles 12  dist cyclic   (112 cores)
#   REPEAT  identical to BASE                           determinism control
#   NPY4    nproc 8x7 -> 8x4   (32 ranks x 2)  Y-CUT ONLY, i-extent stays 112  predict SAME
#   NPX4    nproc 8x7 -> 4x7   (28 ranks x 2)  X-CUT ONLY, j-extent stays 128  predict DIFFERS
#   SHAPE   nproc 8x7 -> 7x8   (56 ranks x 2)  both axes, ranks fixed -- the 1080697 delta
#   TILE8   numtiles 12 -> 8                            tiles only
#   TILE1   numtiles 12 -> 1                            tiles only, extreme
#   THR1    2 threads -> 1, ranks/shape/tiles unchanged  threads only
#   DISTB   --distribution cyclic -> block:block         rank placement only
#
#   REPEAT is load-bearing.  Without it a "DIFFERS" verdict could be ADIOS2 aggregation or
#   the BP->netCDF converter rather than the solver, and every other row would be unreadable.
#   NPY4 and NPX4 are the pair that actually decides the question: they change the same
#   number of things (one nproc axis, and the rank count that necessarily follows), so if
#   one is bit-identical and the other is not, the axis is the cause and the rank count is not.
#
# CONSTRAINTS (inherited from rk_alternation.sbatch, each already cost a run somewhere)
#   * karina ships wrfinput/wrfbdy as BP; no real.exe.  20 min is far inside wrfbdy.
#   * adios2_numaggregators is pinned to THAT ARM's rank count (configs.py --ranks).
#   * All I/O is ADIOS2 (io_form_*=14); wrfout/wrfrst frames are DIRECTORIES -> rm -rf.
#   * WRF refuses a patch thinner than 10 cells: 901/7=128, 901/32=28, 901/14=64 all pass.
#   * /home/u5727520/wrf-stack/wtune-autotuner is READ ONLY; nothing here writes to it.
set -uo pipefail

HERE=/home/u5727520/wrf-bench/wscore/bitrepro
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
REPO=/home/u5727520/wrf-stack/wtune-autotuner          # READ ONLY
CAMPAIGN="$REPO/campaign"
CONV="$REPO/tools/wrf_adios2_to_netcdf4/convert.py"
RKALT=/home/u5727520/wrf-bench/wscore/rk_alt           # bitcmp.py / segnl.py live here
PIXI_ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
PY="pixi run --manifest-path $PIXI_ENV/pixi.toml python"
TABLE_CACHE=/home/u5727520/wrf-bench/wscore/campaign/thompson_tables

PREP_DIR=/home/u5727520/wrf-bench/stprep_1070876
BP_DIR=/home/u5727520/wrf-bench/karina_48h/bp
DT=15
RUN_MIN=${RUN_MIN:-15}
BASE_START=2026-09-02_00:00:00
NODES=1
CORES_PER_NODE=112

J=${SLURM_JOB_ID:-manual}
OUT=${OUT:-$HERE/run_$J}

# arm | ranks | omp | nx | ny | numtiles | srun distribution
ARM_TABLE=(
  "BASE   56 2 8 7 12 cyclic"
  "REPEAT 56 2 8 7 12 cyclic"
  "NPY4   32 2 8 4 12 cyclic"
  "NPX4   28 2 4 7 12 cyclic"
  "SHAPE  56 2 7 8 12 cyclic"
  "TILE8  56 2 8 7  8 cyclic"
  "TILE1  56 2 8 7  1 cyclic"
  "THR1   56 1 8 7 12 cyclic"
  "DISTB  56 2 8 7 12 block:block"
)
ARMS=${ARMS:-"BASE REPEAT NPY4 NPX4 SHAPE TILE8 TILE1 THR1 DISTB"}

END_STAMP=$(python3 - "$BASE_START" "$RUN_MIN" <<'PY'
import sys, datetime as dt
t = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d_%H:%M:%S") + dt.timedelta(minutes=int(sys.argv[2]))
print(t.strftime("%Y-%m-%d_%H:%M:%S"))
PY
)

# ---------------------------------------------------------------- preflight
for f in "$CONV" "$RKALT/bitcmp.py" "$RKALT/segnl.py" "$HERE/bitrms.py" \
         "$W/main/wrf.exe" "$PREP_DIR/namelist.input" \
         "$BP_DIR/wrfinput_d01.bp" "$BP_DIR/wrfbdy_d01.bp"; do
  [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
[ $(( (RUN_MIN * 60) % DT )) -eq 0 ] || { echo "FATAL: ${RUN_MIN}min is not a whole number of ${DT}s steps" >&2; exit 1; }
for row in "${ARM_TABLE[@]}"; do
  set -- $row
  [ $(( $2 % NODES )) -eq 0 ] || { echo "FATAL: $1 ranks $2 not divisible by $NODES nodes" >&2; exit 1; }
  [ $(( $2 * $3 )) -le $(( NODES * CORES_PER_NODE )) ] || { echo "FATAL: $1 wants $(( $2 * $3 )) cores" >&2; exit 1; }
  [ $(( $4 * $5 )) -eq $2 ] || { echo "FATAL: $1 nproc ${4}x${5} != $2 ranks" >&2; exit 1; }
  [ $(( 901 / $4 )) -ge 10 ] && [ $(( 901 / $5 )) -ge 10 ] || { echo "FATAL: $1 patch thinner than 10 cells" >&2; exit 1; }
done

if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo "PLAN karina 901x901x45 dt=${DT}s, ${RUN_MIN} min = $((RUN_MIN*60/DT)) steps per arm"
  echo "  frames $BASE_START (t=0) and $END_STAMP (t=end); only the final frame is compared"
  printf '  %-7s %4s ranks x %s thr  nproc %2sx%-2s  tiles %-2s  dist %s\n' $(printf '%s\n' "${ARM_TABLE[@]}")
  echo "  arms: $ARMS"
  exit 0
fi

mkdir -p "$OUT"

module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false
export KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

echo "job $J  nodes ${SLURM_JOB_NODELIST:-?}"
echo "karina 901x901x45 dt=${DT}s, ${RUN_MIN} min = $((RUN_MIN*60/DT)) steps, frames t=0 and $END_STAMP"
echo "wrf.exe: $(ls -l --time-style=+%F_%T "$W/main/wrf.exe" | awk '{print $6, $7}')"
mmlsquota -u "$USER" home1 2>/dev/null | tail -1

stage () {
  local d=$1
  rm -rf "$d"; mkdir -p "$d"; cd "$d" || exit 1
  ln -sf "$BP_DIR/wrfinput_d01.bp" wrfinput_d01.bp
  ln -sf "$BP_DIR/wrfbdy_d01.bp"   wrfbdy_d01.bp
  for tbl in freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qsV2.dat; do
    [ -f "$TABLE_CACHE/$tbl" ] && ln -sf "$TABLE_CACHE/$tbl" "$tbl"
  done
  for f in "$W"/run/*; do
    b=$(basename "$f")
    case "$b" in namelist.input|namelist.output|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;
      *) [ -e "$b" ] || ln -sf "$f" "$b" ;; esac
  done
  ln -sf "$W/main/wrf.exe" ./wrf.exe
}

RESULTS="$OUT/arms.csv"
echo "arm,ranks,omp,nx,ny,numtiles,dist,cores,ok,wall_s,solver_s,steps,wrf_numtiles_reported" > "$RESULTS"

run_arm () {
  local arm=$1 ranks=$2 omp=$3 nx=$4 ny=$5 tiles=$6 dist=$7
  local RUNDIR="$OUT/$arm" ok=1 rc wall
  echo
  echo "############ arm $arm : ${nx}x${ny}, ${ranks} ranks x ${omp} thr, ${tiles} tiles, dist=$dist ($((ranks*omp)) cores)"
  stage "$RUNDIR"

  python3 "$CAMPAIGN/configs.py" edit-namelist "$PREP_DIR/namelist.input" ./namelist.input \
    --nproc-x "$nx" --nproc-y "$ny" --numtiles "$tiles" \
    --run-minutes "$RUN_MIN" --max-dom 1 \
    --history-mode interval --history-minutes "$RUN_MIN" --ranks "$ranks" \
    || { echo "  namelist edit FAILED"; ok=0; }
  if [ "$ok" = "1" ]; then
    python3 "$RKALT/segnl.py" ./namelist.input --base-start "$BASE_START" \
      --offset-min 0 --run-min "$RUN_MIN" --rk-ord 3 \
      --restart false --restart-interval-min 0 --max-dom 1 \
      || { echo "  segnl FAILED"; ok=0; }
  fi
  cp -f namelist.input "$OUT/${arm}.namelist.input" 2>/dev/null

  if [ "$ok" = "1" ]; then
    wall=$(date +%s.%N)
    OMP_NUM_THREADS=$omp \
    timeout 600 srun --mpi=pmi2 --distribution="$dist" -n "$ranks" \
      --ntasks-per-node=$(( ranks / NODES )) -c "$omp" ./wrf.exe > "wrf.out" 2>&1
    rc=$?
    wall=$(python3 -c "import sys,time;print('%.2f'%(time.time()-float(sys.argv[1])))" "$wall")
    if [ ! -f rsl.error.0000 ] || ! grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then
      echo "  FAILED rc=$rc wall=${wall}s -- last rsl.error lines:"
      tail -15 rsl.error.0000 2>/dev/null | sed 's/^/      /'
      ok=0
    fi
  fi

  # WRF's OWN record of what it parsed and of how many tiles it actually built.
  # Without this a "tiles only" arm could silently have run the base tiling.
  local wrftiles=""
  cp -f namelist.output "$OUT/${arm}.namelist.output" 2>/dev/null
  cp -f rsl.out.0000    "$OUT/${arm}.rsl.out" 2>/dev/null
  cp -f rsl.error.0000  "$OUT/${arm}.rsl.err" 2>/dev/null
  wrftiles=$(grep -iE "num.?tiles" rsl.out.0000 2>/dev/null | head -2 | tr '\n' ';' | tr -d ',')

  local nstep=0 solver=0
  if [ -f rsl.out.0000 ]; then
    nstep=$(grep -c "Timing for main" rsl.out.0000)
    solver=$(grep "Timing for main" rsl.out.0000 | sed -nE 's/.*: +([0-9.]+) elapsed.*/\1/p' \
             | awk '{s+=$1} END {printf "%.3f", s+0}')
  fi

  local frames final
  frames=$(ls -d wrfout_d01_* 2>/dev/null | sort)
  final="wrfout_d01_${END_STAMP}"
  if [ "$ok" = "1" ]; then
    if [ ! -d "$final" ]; then
      echo "  MISSING FINAL FRAME $final; got:"; echo "$frames" | sed 's/^/      /'; ok=0
    fi
  fi
  echo "  ok=$ok wall=${wall:-?}s steps=$nstep solver=${solver}s  frames: $(echo "$frames" | tr '\n' ' ')"
  echo "  WRF reported tiling: ${wrftiles:-<not printed>}"
  echo "  namelist.output: $(grep -iE '^ *(NPROC_X|NPROC_Y|NUMTILES) *=' "$OUT/${arm}.namelist.output" 2>/dev/null | tr -s ' ' | tr '\n' ' ')"

  echo "$arm,$ranks,$omp,$nx,$ny,$tiles,$dist,$((ranks*omp)),$ok,${wall:-},$solver,$nstep,\"${wrftiles:-}\"" >> "$RESULTS"
  rm -f rsl.out.0* rsl.error.0*
  # keep ONLY the final frame; the t=0 frame is the same initial condition in every arm
  rm -rf ./wrfrst_d0*
  for fr in $frames; do [ "$fr" = "$final" ] || rm -rf "./$fr"; done
  cd "$OUT" || exit 1
}

for row in "${ARM_TABLE[@]}"; do
  set -- $row
  case " $ARMS " in *" $1 "*) run_arm "$@" ;; esac
done

# ---------------------------------------------------------------- BP -> netCDF, in parallel
# 4 exclusive nodes are idle at this point; one converter per node, 4 at a time.
echo
echo "================ converting final frames (BP -> netCDF4) ================"
pids=()
for row in "${ARM_TABLE[@]}"; do
  set -- $row
  case " $ARMS " in *" $1 "*) ;; *) continue ;; esac
  src="$OUT/$1/wrfout_d01_${END_STAMP}"
  [ -d "$src" ] || { echo "  $1: no final frame to convert"; continue; }
  srun -N1 -n1 -c 8 --exact bash -c \
    "cd $PIXI_ENV && pixi run python $CONV --input $src --output $OUT/$1.nc" \
    > "$OUT/$1.convert.log" 2>&1 &
  pids+=("$!:$1")
done
for p in "${pids[@]}"; do
  wait "${p%%:*}" || echo "  CONVERT FAILED for ${p##*:} (see $OUT/${p##*:}.convert.log)"
done
ls -l "$OUT"/*.nc 2>/dev/null | awk '{printf "  %-40s %s\n", $9, $5}'

# ---------------------------------------------------------------- exact comparison
echo
echo "================ EXACT bit-identity vs BASE ================"
REF="$OUT/BASE.nc"
if [ ! -s "$REF" ]; then
  echo "FATAL: BASE.nc missing -- nothing to compare against"; exit 1
fi
for row in "${ARM_TABLE[@]}"; do
  set -- $row
  arm=$1
  [ "$arm" = "BASE" ] && continue
  case " $ARMS " in *" $arm "*) ;; *) continue ;; esac
  cand="$OUT/$arm.nc"
  [ -s "$cand" ] || { echo "$arm: SKIPPED (no netCDF)"; continue; }
  $PY "$RKALT/bitcmp.py" "$REF" "$cand" --label "$arm vs BASE" \
      --json-out "$OUT/bitident_$arm.json" 2>&1 | sed 's/^/  /'
  # RMS only matters when the exact check already said they differ
  if ! grep -q '"result": "PASS"' "$OUT/bitident_$arm.json" 2>/dev/null; then
    $PY "$HERE/bitrms.py" "$REF" "$cand" --label "$arm" --json-out "$OUT/rms_$arm.json" 2>&1 | sed 's/^/     /'
  fi
done

echo
echo "================ SUMMARY ================"
$PY "$HERE/summarize_bitrepro.py" --dir "$OUT" 2>&1 || cat "$RESULTS"

for row in "${ARM_TABLE[@]}"; do set -- $row; rm -rf "${OUT:?}/$1"; done
echo
echo "artifacts: $OUT"
du -sh "$OUT" 2>/dev/null
echo "=== done ==="

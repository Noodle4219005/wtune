#!/usr/bin/env bash
#SBATCH --job-name=dt_trade
#SBATCH --account=ACD110018
#SBATCH --partition=cf448
#SBATCH --nodes=4 --ntasks-per-node=112 --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/rk_alt/dt_trade-%j.out
#
# QUESTION
#   Is RK2 with a SHORTENED time_step a better operating point than RK3 at the nominal
#   time_step?  i.e. can the rk_ord speed/accuracy trade be bought at a favourable ratio
#   instead of being all-or-nothing?
#
#   Every arm is a SINGLE wrf.exe call of the whole logical run.  There is no segmentation
#   and no restart chain here: that axis was closed by job 1081239 (alternating RK3/RK2
#   across restart boundaries drifts 99.9% as far as pure RK2 for half the saving).
#   Consequently there is no A2-style restart control to run.
#
# WHAT IS SCORED AGAINST WHAT
#   Arms are scored against TRUTH (rk_ord=3 at the SHORTEST dt), not against REF.  REF is
#   itself scored against TRUTH, and that number is the floor of what any arm can achieve.
#   Every arm uses the SAME decomposition, the same binary and the same allocation, so the
#   arm-to-arm differences are deterministic time-discretisation differences, not chaos.
#
#   The chaos floor and the fidelity gate's `twin` role are supplied by the ALREADY
#   MEASURED conus12 TWIN of job 1080968 (32x14, rk_ord=3, dt=72, same 12 h, same binary)
#   rather than by re-running it -- see EXT_TWIN below.  Re-running it would have cost
#   ~4.5 SU for a number this tree already has.  The cross-job validity of that reuse is
#   itself checked: this job's REF is byte-compared against job 1080968's arm A, which is
#   the identical configuration.  That check is reported as REPRO, and it is a control:
#   if it FAILS, the borrowed TWIN is not comparable and only the raw RMS columns stand.
#
# THE TWO CONSTRAINTS ON A LEGAL time_step (both enforced below, before anything launches)
#   1. 43200 s (the 12 h run) must be a whole number of steps, or the final history frame
#      does not land on the logical end time and the arms are not comparable.
#   2. interval_seconds (10800 s for conus12) must ALSO be a whole number of steps, so a
#      lateral-boundary update always coincides with a step boundary.
#   Both are divisibility constraints on 10800, whose divisors near 72 are ... 50 54 60 72
#   75 ... -- there is NO legal time_step at all between 60 and 72.
#
# COST MODEL THIS EXPERIMENT TESTS (built from job 1080968's measured conus12 numbers)
#   Total(dt) = R + (43200/dt) * c_rk     with R the dt-invariant radiation cost
#   Measured:  REF 69.5 s / 600 steps, pure RK2 64.9 s / 600 steps  => RK2 is -6.5%, NOT
#   the -33% a naive "2 stages instead of 3" model predicts.  dyn_em/solve_em.F:594-621
#   says why: RK3 does 1 + nss/2 + nss acoustic sub-steps per step and RK2 does nss/2 +
#   nss, so with nss=4 dropping to RK2 removes 1 of 7 acoustic sub-steps, not a third of
#   the work.  And dyn_em/solve_em.F:451-470 gives nss = max(2*(INT(300*dt/dx - .01)+1),4)
#   = 4 for EVERY dt in [40.4, 80.4) s at dx=12 km -- the floor binds at dt=36 -- so
#   shortening dt buys no reduction in acoustic sub-steps per step either.
#
# ENV KNOBS
#   ARMSPEC   "NAME:rk_ord:time_step ..."   (default below)
#   TRUTH_ARM name of the arm used as the fidelity reference   (default TRUTH)
#   REF_ARM   name of the arm used as the cost baseline        (default REF)
#   TOTAL_MIN NODES RANKS OMP NUMTILES NX NY TIMEOUT_S KEEP_NC OUT PLAN_ONLY
set -uo pipefail

HERE=/home/u5727520/wrf-bench/wscore/rk_alt
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
REPO=/home/u5727520/wrf-stack/wtune-autotuner          # READ ONLY -- never written to
CAMPAIGN="$REPO/campaign"
CONV="$REPO/tools/wrf_adios2_to_netcdf4/convert.py"
FIDELITY="$REPO/fidelity_gate.py"
PIXI_ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
PY="pixi run --manifest-path $PIXI_ENV/pixi.toml python"
TABLE_CACHE=/home/u5727520/wrf-bench/wscore/campaign/thompson_tables
CORES_PER_NODE=112

# Borrowed from job 1080968 (conus12, 12 h, same binary, same 448 ranks on 4 nodes):
#   EXT_TWIN   arm TWIN, 32x14 decomposition, rk_ord=3, dt=72  -> the chaos floor
#   EXT_REF    arm A,    16x28 decomposition, rk_ord=3, dt=72  -> this job's REF must
#                                                                 reproduce it bit for bit
EXT_RUN=${EXT_RUN:-/home/u5727520/wrf-bench/wscore/rk_alt/run_1080968}
EXT_TWIN_GLOB="$EXT_RUN/TWIN_f*.nc"
EXT_REF_NC="$EXT_RUN/A_f1.nc"

CASE=${CASE:-conus12}
CASES_JSON="$CAMPAIGN/cases.json"
CASE_VARS=$(python3 - "$CASES_JSON" "$CASE" <<'PY'
import json, sys
try:
    c = json.load(open(sys.argv[1]))["cases"][sys.argv[2]]
except KeyError:
    sys.exit("FATAL: case %r is not in cases.json" % sys.argv[2])
tpl = c.get("namelist_template") or (c["prep_dir"].rstrip("/") + "/namelist.input")
for k, v in (("PREP_DIR", c["prep_dir"]), ("BP_DIR", c["bp_dir"]),
             ("NAMELIST_TEMPLATE", tpl), ("DT", c["dt"]),
             ("MAX_DOM", c.get("max_dom", 1)),
             ("E_WE", c["e_we"]), ("E_SN", c["e_sn"]),
             ("CASE_FAMILY", c.get("case_family", "-"))):
    print("%s='%s'" % (k, v))
PY
) || { echo "$CASE_VARS" >&2; exit 1; }
eval "$CASE_VARS"

case "$CASE" in
  conus12) dNODES=4 dRANKS=448 dOMP=1 dTILES=1 dNX=16 dNY=28 dTOTAL=720 dTIMEOUT=900
           dARMSPEC="REF:3:72 B72:2:72 B60:2:60 B54:2:54 B48:2:48 TRUTH:3:36" ;;
  *) echo "FATAL: no default row for case '$CASE' -- pass NODES/RANKS/OMP/NUMTILES/NX/NY/TOTAL_MIN/ARMSPEC" >&2; exit 1 ;;
esac

TOTAL_MIN=${TOTAL_MIN:-$dTOTAL}
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-$dNODES}}
RANKS=${RANKS:-$dRANKS}; OMP=${OMP:-$dOMP}; NUMTILES=${NUMTILES:-$dTILES}
NX=${NX:-$dNX}; NY=${NY:-$dNY}
ARMSPEC=${ARMSPEC:-$dARMSPEC}
TRUTH_ARM=${TRUTH_ARM:-TRUTH}
REF_ARM=${REF_ARM:-REF}
TIMEOUT_S=${TIMEOUT_S:-$dTIMEOUT}
KEEP_NC=${KEEP_NC:-1}
J=${SLURM_JOB_ID:-manual}
OUT=${OUT:-$HERE/dtrun_$J}

for f in "$CONV" "$FIDELITY" "$HERE/segnl.py" "$HERE/bitcmp.py" "$HERE/dt_summarize.py" \
         "$W/main/wrf.exe" "$NAMELIST_TEMPLATE" "$BP_DIR/wrfbdy_d01.bp"; do
  [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
for ((d = 1; d <= MAX_DOM; d++)); do
  f=$(printf "%s/wrfinput_d%02d.bp" "$BP_DIR" "$d")
  [ -e "$f" ] || { echo "FATAL: missing $f (max_dom=$MAX_DOM)" >&2; exit 1; }
done

# the lateral-boundary update interval, read from the case's OWN template
BDY_INT=$(python3 - "$NAMELIST_TEMPLATE" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()
m = re.search(r"\binterval_seconds\s*=\s*([0-9]+)", txt)
print(m.group(1) if m else 0)
PY
)
TOTAL_S=$(( TOTAL_MIN * 60 ))

# ---------------------------------------------------------------- validate every arm
ARM_NAMES=(); ARM_RK=(); ARM_DT=()
for spec in $ARMSPEC; do
  IFS=: read -r nm rk adt <<< "$spec"
  [ -n "${nm:-}" ] && [ -n "${rk:-}" ] && [ -n "${adt:-}" ] || {
    echo "FATAL: bad ARMSPEC entry '$spec' (want NAME:rk_ord:time_step)" >&2; exit 1; }
  case "$rk" in 2|3) ;; *) echo "FATAL: arm $nm: rk_ord must be 2 or 3, got '$rk'" >&2; exit 1 ;; esac
  # CONSTRAINT 1: the logical run must be a whole number of steps, or the final frame does
  # not land on the logical end time (the history alarm rings on the first step AT OR AFTER
  # it) and the arms are compared at different valid times.
  if [ $(( TOTAL_S % adt )) -ne 0 ]; then
    echo "FATAL: arm $nm: time_step=$adt does not divide the ${TOTAL_S}s run ($TOTAL_S/$adt is not an integer)" >&2
    exit 1
  fi
  # CONSTRAINT 2: a lateral-boundary update must coincide with a step boundary.
  if [ "${BDY_INT:-0}" -gt 0 ] && [ $(( BDY_INT % adt )) -ne 0 ]; then
    echo "FATAL: arm $nm: time_step=$adt does not divide interval_seconds=$BDY_INT," >&2
    echo "       so an LBC update would fall mid-step.  Legal values are divisors of $BDY_INT." >&2
    exit 1
  fi
  ARM_NAMES+=("$nm"); ARM_RK+=("$rk"); ARM_DT+=("$adt")
done
case " ${ARM_NAMES[*]} " in *" $TRUTH_ARM "*) ;; *) echo "FATAL: TRUTH_ARM='$TRUTH_ARM' is not in ARMSPEC" >&2; exit 1 ;; esac
case " ${ARM_NAMES[*]} " in *" $REF_ARM "*) ;;   *) echo "FATAL: REF_ARM='$REF_ARM' is not in ARMSPEC" >&2; exit 1 ;; esac

if [ $(( RANKS % NODES )) -ne 0 ]; then echo "FATAL: RANKS ($RANKS) not divisible by NODES ($NODES)" >&2; exit 1; fi
if [ $(( RANKS * OMP )) -gt $(( NODES * CORES_PER_NODE )) ]; then
  echo "FATAL: ranks*threads = $(( RANKS * OMP )) exceeds $NODES x $CORES_PER_NODE cores" >&2; exit 1; fi
if [ $(( NX * NY )) -ne "$RANKS" ]; then echo "FATAL: nproc_x*nproc_y (${NX}x${NY}) != RANKS ($RANKS)" >&2; exit 1; fi
if [ $(( E_WE / NX )) -lt 10 ] || [ $(( E_SN / NY )) -lt 10 ]; then
  echo "FATAL: ${NX}x${NY} gives a patch thinner than WRF's 10-cell minimum" >&2; exit 1; fi

BASE_START=$(python3 "$HERE/segnl.py" show-start "$NAMELIST_TEMPLATE") || {
  echo "FATAL: could not read start time from $NAMELIST_TEMPLATE" >&2; exit 1; }
END_STAMP=$(python3 - "$BASE_START" "$TOTAL_MIN" <<'PY'
import sys, datetime as dt
t = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d_%H:%M:%S") + dt.timedelta(minutes=int(sys.argv[2]))
print(t.strftime("%Y-%m-%d_%H:%M:%S"))
PY
)

# num_sound_steps WRF will actually pick, straight from dyn_em/solve_em.F:451-464
# (non-adaptive branch).  Reported per arm because it decides the per-step acoustic cost.
nss_of () { python3 -c "
import sys
dtv=float(sys.argv[1]); sp=float(sys.argv[2])
print(max(2*(int(300.0*dtv/sp - 0.01)+1), 4))" "$1" "$2"; }
DX=$(python3 - "$NAMELIST_TEMPLATE" <<'PY'
import re, sys
m = re.search(r"\bdx\s*=\s*([0-9.]+)", open(sys.argv[1]).read())
print(m.group(1) if m else 12000)
PY
)

if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo "PLAN case=$CASE ($CASE_FAMILY, ${E_WE}x${E_SN}, dx=${DX}m, max_dom=$MAX_DOM, template dt=${DT}s)"
  echo "  logical    ${TOTAL_MIN} min, $BASE_START -> $END_STAMP  (${TOTAL_S}s)"
  echo "  lbc        interval_seconds=$BDY_INT"
  echo "  decomp     ${NX}x${NY}, ${RANKS} ranks x ${OMP} threads, ${NUMTILES} tiles, adios2_numaggregators=${RANKS}, $NODES nodes"
  echo "  truth arm  $TRUTH_ARM      cost baseline arm  $REF_ARM"
  echo "  borrowed   twin=$EXT_TWIN_GLOB   repro-check=$EXT_REF_NC"
  printf "  %-7s %-7s %-10s %-8s %-6s %-12s %s\n" arm rk_ord time_step steps nss dts_s "acoustic_CFL@340"
  for i in "${!ARM_NAMES[@]}"; do
    a_dt=${ARM_DT[$i]}; n=$(nss_of "$a_dt" "$DX")
    printf "  %-7s %-7s %-10s %-8s %-6s %-12s %s\n" "${ARM_NAMES[$i]}" "${ARM_RK[$i]}" "$a_dt" \
      "$(( TOTAL_S / a_dt ))" "$n" \
      "$(python3 -c "print('%.3f'%($a_dt/$n))")" \
      "$(python3 -c "print('%.4f'%(340.0*($a_dt/$n)/$DX))")"
  done
  exit 0
fi

mkdir -p "$OUT"
ARMS_CSV="$OUT/arms.csv"
echo "arm,rk_ord,time_step,nss,steps_expected,nx,ny,ranks,omp,tiles,ok,wall_s,solver_s,steps,nc" > "$ARMS_CSV"

module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false
export OMP_NUM_THREADS=$OMP KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

echo "job $J  nodes ${SLURM_JOB_NODELIST:-?}  case $CASE ($CASE_FAMILY, ${E_WE}x${E_SN}, dx=${DX}m)"
echo "logical run: ${TOTAL_MIN} min from $BASE_START to $END_STAMP  (interval_seconds=$BDY_INT)"
echo "decomposition ${NX}x${NY}, $RANKS ranks x $OMP threads on $NODES nodes, $NUMTILES tiles"
echo "arms: $ARMSPEC   (truth=$TRUTH_ARM baseline=$REF_ARM)"
mmlsquota -u "$USER" home1 2>/dev/null | tail -2

stage () {
  local d=$1
  rm -rf "$d"; mkdir -p "$d"; cd "$d" || exit 1
  local dom df
  for ((dom = 1; dom <= MAX_DOM; dom++)); do
    df=$(printf "d%02d" "$dom"); ln -sf "$BP_DIR/wrfinput_${df}.bp" "wrfinput_${df}.bp"
  done
  ln -sf "$BP_DIR/wrfbdy_d01.bp" wrfbdy_d01.bp
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

run_arm () {
  local arm=$1 rk=$2 adt=$3
  local RUNDIR="$OUT/$arm"
  local ok=1 rc wall solver steps nss expsteps
  nss=$(nss_of "$adt" "$DX"); expsteps=$(( TOTAL_S / adt ))

  echo
  echo "############ arm $arm : rk_ord=$rk time_step=${adt}s -> $expsteps steps, num_sound_steps=$nss, one call"
  stage "$RUNDIR"

  # history: TWO frames, t=0 and t=TOTAL.  fidelity_gate.py returns INVALID on a single
  # frame (that frame IS the reference's first valid time and the gate refuses to decide
  # on t=0 alone).  One call, so the restart alarm-inheritance trap does not apply here.
  python3 "$CAMPAIGN/configs.py" edit-namelist "$PREP_DIR/namelist.input" ./namelist.input \
    --nproc-x "$NX" --nproc-y "$NY" --numtiles "$NUMTILES" \
    --run-minutes "$TOTAL_MIN" --max-dom "$MAX_DOM" \
    --history-mode interval --history-minutes "$TOTAL_MIN" --ranks "$RANKS" || ok=0
  if [ "$ok" = "1" ]; then
    python3 "$HERE/segnl.py" ./namelist.input --base-start "$BASE_START" \
      --offset-min 0 --run-min "$TOTAL_MIN" --rk-ord "$rk" --time-step "$adt" \
      --restart false --restart-interval-min 0 --max-dom "$MAX_DOM" || ok=0
  fi
  cp -f namelist.input "$OUT/${arm}.namelist.input" 2>/dev/null

  if [ "$ok" = "1" ]; then
    wall=$(date +%s.%N)
    OMP_NUM_THREADS=$OMP \
    timeout "$TIMEOUT_S" srun --mpi=pmi2 --distribution=cyclic -n "$RANKS" \
      --ntasks-per-node=$(( RANKS / NODES )) -c "$OMP" ./wrf.exe > "wrf.out" 2>&1
    rc=$?
    wall=$(python3 -c "import sys,time;print('%.2f'%(time.time()-float(sys.argv[1])))" "$wall")
    if [ ! -f rsl.error.0000 ] || ! grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then
      echo "  FAILED rc=$rc wall=${wall}s -- last rsl.error lines:"
      tail -15 rsl.error.0000 2>/dev/null | sed 's/^/      /'
      ok=0
    fi
  else
    wall=0
  fi
  cp -f namelist.output "$OUT/${arm}.namelist.output" 2>/dev/null
  cp -f rsl.out.0000    "$OUT/${arm}.rsl.out" 2>/dev/null
  cp -f rsl.error.0000  "$OUT/${arm}.rsl.err" 2>/dev/null

  grep "Timing for main" rsl.out.0000 2>/dev/null \
    | sed -nE 's/.*on domain +([0-9]+): +([0-9.]+) elapsed.*/\1 \2/p' > steps_by_domain.txt
  steps=$(awk '$1 == 1' steps_by_domain.txt | wc -l)
  solver=$(awk '$1 == 1 { s += $2 } END { printf "%.3f", s + 0 }' steps_by_domain.txt)
  cp -f steps_by_domain.txt "$OUT/${arm}.steps_by_domain.txt" 2>/dev/null
  rm -f rsl.out.0* rsl.error.0*

  # THE MECHANISM CHECK: WRF must have taken the step count the requested dt implies.  A
  # silently-ignored time_step would otherwise look like "dt had no effect on accuracy".
  if [ "$ok" = "1" ] && [ "$steps" -ne "$expsteps" ]; then
    echo "  STEP COUNT MISMATCH for $arm: time_step=$adt implies $expsteps d01 steps, WRF took $steps"
    ok=0
  fi

  local frames k nc nc_last=""
  frames=$(ls -d wrfout_d01_* 2>/dev/null | sort)
  if [ "$ok" = "1" ]; then
    if [ "$(echo "$frames" | grep -c .)" -lt 2 ]; then
      echo "  WRONG FRAME COUNT for $arm: expected 2, got:"; echo "$frames" | sed 's/^/      /'; ok=0
    elif [ "$(echo "$frames" | tail -1)" != "wrfout_d01_${END_STAMP}" ]; then
      echo "  WRONG FINAL FRAME TIME for $arm: want wrfout_d01_${END_STAMP}, got $(echo "$frames" | tail -1)"; ok=0
    fi
  fi
  if [ "$ok" = "1" ]; then
    k=0
    for frame in $frames; do
      nc="$OUT/${arm}_f${k}.nc"
      srun -N1 -n1 -c 8 bash -c "cd $PIXI_ENV && pixi run python $CONV --input $RUNDIR/$frame --output $nc" \
        >> "$OUT/${arm}.convert.log" 2>&1
      if [ $? -ne 0 ] || [ ! -s "$nc" ]; then
        echo "  CONVERT FAILED for $arm frame $frame (see $OUT/${arm}.convert.log)"; rm -f "$nc"; ok=0; break
      fi
      echo "  frame $frame -> $(basename "$nc") ($(du -sh "$nc" | cut -f1))"
      nc_last="$nc"; k=$(( k + 1 ))
    done
  fi
  rm -rf ./wrfout_d0* ./wrfrst_d0*

  echo "$arm,$rk,$adt,$nss,$expsteps,$NX,$NY,$RANKS,$OMP,$NUMTILES,$ok,$wall,$solver,$steps,$nc_last" >> "$ARMS_CSV"
  echo "  ARM $arm ok=$ok wall=${wall}s solver=${solver}s steps=$steps (expected $expsteps)"
  cd "$OUT" || exit 1
}

for i in "${!ARM_NAMES[@]}"; do
  run_arm "${ARM_NAMES[$i]}" "${ARM_RK[$i]}" "${ARM_DT[$i]}"
done

# ---------------------------------------------------------------- what WRF actually parsed
echo
echo "================ rk_ord AND time_step ACTUALLY USED, from WRF's own namelist.output ================"
for i in "${!ARM_NAMES[@]}"; do
  arm=${ARM_NAMES[$i]}; f="$OUT/${arm}.namelist.output"; rsl="$OUT/${arm}.rsl.out"
  [ -e "$f" ] || continue
  first=$(grep -m1 "Timing for main" "$rsl" 2>/dev/null | sed -nE 's/.*time ([0-9_:-]+) on domain.*/\1/p')
  last=$(grep "Timing for main" "$rsl" 2>/dev/null | tail -1 | sed -nE 's/.*time ([0-9_:-]+) on domain.*/\1/p')
  printf "%-7s RK_ORD=%-3s TIME_STEP=%-4s TIME_STEP_SOUND=%-3s  simulated %s .. %s\n" "$arm" \
    "$(grep -iE '^ *RK_ORD *=' "$f" | grep -oE '[0-9]+' | head -1)" \
    "$(grep -iE '^ *TIME_STEP *=' "$f" | grep -oE '[0-9]+' | head -1)" \
    "$(grep -iE '^ *TIME_STEP_SOUND *=' "$f" | sed -E 's/.*= *//; s/^[0-9]+\*//; s/[, ].*//')" \
    "${first:-?}" "${last:-?}"
done

# ---------------------------------------------------------------- cross-job repro control
echo
REF_NC=$(ls "$OUT"/${REF_ARM}_f*.nc 2>/dev/null | sort | tail -1)
if [ -s "${REF_NC:-/nonexistent}" ] && [ -s "$EXT_REF_NC" ]; then
  $PY "$HERE/bitcmp.py" "$EXT_REF_NC" "$REF_NC" \
    --label "$REF_ARM (this job) vs job 1080968 arm A -- identical configuration, different allocation" \
    --json-out "$OUT/repro.json" 2>&1 | tail -20
else
  echo "REPRO CONTROL SKIPPED: $REF_ARM final netCDF or $EXT_REF_NC missing"
fi

# ---------------------------------------------------------------- fidelity vs TRUTH
echo
truth_files=($(ls "$OUT"/${TRUTH_ARM}_f*.nc 2>/dev/null))
twin_files=($(ls $EXT_TWIN_GLOB 2>/dev/null))
for i in "${!ARM_NAMES[@]}"; do
  arm=${ARM_NAMES[$i]}
  [ "$arm" = "$TRUTH_ARM" ] && continue
  cand_files=($(ls "$OUT"/${arm}_f*.nc 2>/dev/null))
  if [ ${#truth_files[@]} -eq 0 ] || [ ${#twin_files[@]} -eq 0 ] || [ ${#cand_files[@]} -eq 0 ]; then
    echo "fidelity $arm: SKIPPED (need ${TRUTH_ARM}_f*.nc, $EXT_TWIN_GLOB and ${arm}_f*.nc)"; continue
  fi
  $PY "$FIDELITY" --reference "${truth_files[@]}" --twin "${twin_files[@]}" \
    --candidate "${cand_files[@]}" \
    --family dynamics --out-prefix "$OUT/fidelity_${arm}" > "$OUT/fidelity_${arm}.log" 2>&1
  echo "fidelity $arm (vs $TRUTH_ARM): $(grep -h FIDELITY-GATE "$OUT/fidelity_${arm}.log" | tail -1)"
  [ -s "$OUT/fidelity_${arm}.json" ] || tail -5 "$OUT/fidelity_${arm}.log" | sed 's/^/    /'
done

DT_CASE="$CASE" DT_TRUTH="$TRUTH_ARM" DT_REF="$REF_ARM" $PY "$HERE/dt_summarize.py" --dir "$OUT"

if [ "$KEEP_NC" != "1" ]; then rm -f "$OUT"/*.nc; fi
for i in "${!ARM_NAMES[@]}"; do rm -rf "$OUT/${ARM_NAMES[$i]}"; done
echo
echo "artifacts: $OUT"
du -sh "$OUT" 2>/dev/null
echo "=== done ==="

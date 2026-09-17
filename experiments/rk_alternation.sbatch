#!/usr/bin/env bash
#SBATCH --job-name=rk_alt
#SBATCH --account=ACD110018
#SBATCH --partition=cf448
#SBATCH --nodes=4 --ntasks-per-node=112 --exclusive
#SBATCH --time=00:30:00
# 12 h / 4 x 3 h measured shape: ~16 min expected (the 48 min dry run took 6:56).  An
# oversized --time only delays backfill; override on the command line if a shape needs more.
#SBATCH --output=/home/u5727520/wrf-bench/wscore/rk_alt/rk_alt-%j.out
#
# QUESTION
#   rk_ord selects WRF's Runge-Kutta order and is read exactly once, at initialisation
#   (dyn_em/solve_em.F:449, `rk_order = config_flags%rk_ord`; it reaches config_flags only
#   through share/mediation_wrfmain.f90:1020 and is NEVER read back out of a restart file),
#   so the ONLY place it can change inside one logical forecast is a restart boundary.
#   Does alternating RK3/RK2 across those boundaries land BETWEEN pure RK3 and pure RK2 in
#   both cost and solution drift, or does it behave like one of the endpoints?
#
# ARMS  (one allocation, run back to back, same binary; every arm uses the case's
#        measured-best decomposition except TWIN, which is deliberately different)
#   A     rk_ord=3, one wrf.exe call                            REFERENCE
#   A2    rk_ord=3, N-segment restart chain                     CONTROL, must be bit-identical to A
#   B     rk_ord=2, one call                                    the cheap endpoint
#   C     N segments, rk_ord = 3,2,3,2                          alternating
#   D     N segments, rk_ord = 3,2,2,2                          mostly-RK2
#   TWIN  rk_ord=3, one call, a DIFFERENT ranks x threads/nproc  chaos floor for fidelity_gate.py
#
#   ARMS selects the subset: e.g. ARMS="A A2 B C TWIN" drops D, which on conus12
#   (job 1080968) was indistinguishable from pure RK2.
#
#   A2 is the whole experiment's load-bearing arm.  If A2 is not bit-identical to A then the
#   restart chain itself perturbs the solution and every drift number for C and D is
#   measuring segmentation, not rk_ord.  It is checked EXACTLY (bitcmp.py), not to a tolerance.
#
# CONSTRAINTS RESPECTED (each one already cost a run somewhere in this tree)
#   * Every case here ships wrfinput/wrfbdy already and needs no real.exe at all.  Do not
#     run PAST what wrfbdy covers, or segment 2+ dies with
#     "Cannot find a valid time to start the LBC during this restart".
#   * ADIOS2 aggregates one subfile per node by default, so adios2_numaggregators is pinned
#     to THAT ARM'S rank count (always >= the node count) and the rank->node map and the decomposition are
#     IDENTICAL for every segment of an arm.  Changing either between segments corrupted the
#     heap inside the restart read (job 1064690).
#   * All I/O is ADIOS2 (io_form_* = 14); wrfrst/wrfout frames are DIRECTORIES, so they are
#     removed with rm -rf, never rm -f.
#   * Namelist edits are never anchored to the start of a line (keys share lines) -- see
#     segnl.py, which also reads every key back and fails if a substitution silently missed.
#   * Segment boundaries must fall on a time step.  SEG_MIN*60 % dt == 0 is enforced below
#     using the CASE'S OWN dt from cases.json, rather than discovered as a half-step restart.
#
# ENV KNOBS
#   CASE       cases.json case id                    (default conus12; also karina, matthew, severe2023)
#   TOTAL_MIN  total logical run, minutes            (per-case default)
#   SEG_MIN    segment length, minutes               (per-case default; TOTAL_MIN/SEG_MIN segments)
#   ARMS       space-separated subset                (default "A A2 B C D TWIN")
#   NODES RANKS OMP NUMTILES NX NY                   reference decomposition
#   TRANKS TOMP TTILES TNX TNY                       TWIN decomposition
#   TIMEOUT_S KEEP_NC OUT
#
#   conus12 12 h (job 1080968):  CASE=conus12 TOTAL_MIN=720  SEG_MIN=180   -p cf448 -N4
#   conus12 dry run:             CASE=conus12 TOTAL_MIN=48   SEG_MIN=12    (48*60 and 12*60 both / 72)
#   karina  24 h:                CASE=karina  TOTAL_MIN=1440 SEG_MIN=360   -p cf448 -N4
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

# ---------------------------------------------------------------- the case
# Everything case-specific is READ from campaign/cases.json (never hard-coded, never
# written back): prep_dir, bp_dir, dt, max_dom, e_we/e_sn, and the namelist template --
# which is `namelist_template` when the case declares one (matthew) and prep_dir's own
# namelist.input otherwise (conus12, karina, severe2023).
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

# ---------------------------------------------------------------- per-case defaults
# Each row is that case's MEASURED best decomposition, so a bare `CASE=x sbatch ...` runs
# the configuration the campaign already established rather than a guess.  Every field is
# overridable by an env var of the same name.
# TWIN may differ from the reference in ranks x threads as well as in nproc_x/y:
# fidelity_gate.py defines the twin as "same physics, different MPI ranks x OpenMP threads
# (STRICT decomposition change only)", and the campaign's own pairing does exactly that
# (conus12 ref C02 448x1 vs twin C06 224x2).  TRANKS/TOMP/TTILES default to the reference's
# when a case does not name its own.
case "$CASE" in
  # conus12: ledger 1075298 C01, 0.0613 s/step median.  This row IS job 1080968's configuration.
  conus12)    dNODES=4 dRANKS=448 dOMP=1 dTILES=1  dNX=16 dNY=28 \
              dTRANKS=448 dTOMP=1 dTTILES=1 dTNX=32 dTNY=14   dTOTAL=720  dSEG=180 dTIMEOUT=900 ;;
  # karina 901x901x45 dt=15.  Measured over the campaign ledgers:
  #   ref  224x2 nproc 8x28, 12 tiles : median 0.2876, mean-all 0.3004 s/step (n=9)
  #   TWIN 448x1 nproc 8x56,  8 tiles : median 0.2820, mean-all 0.3160 s/step (n=75)
  # 8x56 at 448x1 is karina's most-measured shape and the aspect-7 optimum for 901^2, so it
  # is a well-characterised chaos floor rather than a decomposition nobody has run.
  karina)     dNODES=4 dRANKS=224 dOMP=2 dTILES=12 dNX=8  dNY=28 \
              dTRANKS=448 dTOMP=1 dTTILES=8 dTNX=8  dTNY=56   dTOTAL=1440 dSEG=360 dTIMEOUT=3600 ;;
  matthew)    dNODES=1 dRANKS=16  dOMP=7 dTILES=7  dNX=2  dNY=8 \
              dTRANKS=16  dTOMP=7 dTTILES=7 dTNX=4  dTNY=4    dTOTAL=2880 dSEG=720 dTIMEOUT=900 ;;
  severe2023) dNODES=1 dRANKS=28  dOMP=4 dTILES=4  dNX=4  dNY=7 \
              dTRANKS=28  dTOMP=4 dTTILES=4 dTNX=7  dTNY=4    dTOTAL=1440 dSEG=360 dTIMEOUT=900 ;;
  *) echo "FATAL: no default decomposition for case '$CASE' -- pass NODES/RANKS/OMP/NUMTILES/NX/NY/TRANKS/TOMP/TTILES/TNX/TNY/TOTAL_MIN/SEG_MIN explicitly" >&2; exit 1 ;;
esac

TOTAL_MIN=${TOTAL_MIN:-$dTOTAL}
SEG_MIN=${SEG_MIN:-$dSEG}
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-$dNODES}}
RANKS=${RANKS:-$dRANKS}
OMP=${OMP:-$dOMP}
NUMTILES=${NUMTILES:-$dTILES}
NX=${NX:-$dNX};   NY=${NY:-$dNY}       # the case's measured-best decomposition
TRANKS=${TRANKS:-$dTRANKS}             # TWIN: a DIFFERENT legal shape, everything else equal
TOMP=${TOMP:-$dTOMP}
TTILES=${TTILES:-$dTTILES}
TNX=${TNX:-$dTNX}; TNY=${TNY:-$dTNY}
ARMS=${ARMS:-"A A2 B C D TWIN"}
# per-SEGMENT wall cap.  A one-call karina arm is ~1730 s of solver alone, so the old
# flat 900 s would have killed arm A outright -- it is a per-case row now.
TIMEOUT_S=${TIMEOUT_S:-$dTIMEOUT}
KEEP_NC=${KEEP_NC:-1}
J=${SLURM_JOB_ID:-manual}
OUT=${OUT:-$HERE/run_$J}

for f in "$CONV" "$FIDELITY" "$HERE/segnl.py" "$HERE/bitcmp.py" "$HERE/summarize.py" \
         "$W/main/wrf.exe" "$NAMELIST_TEMPLATE" "$BP_DIR/wrfbdy_d01.bp"; do
  [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done
for ((d = 1; d <= MAX_DOM; d++)); do
  f=$(printf "%s/wrfinput_d%02d.bp" "$BP_DIR" "$d")
  [ -e "$f" ] || { echo "FATAL: missing $f (max_dom=$MAX_DOM)" >&2; exit 1; }
done

NSEG=$(( TOTAL_MIN / SEG_MIN ))
if [ $(( NSEG * SEG_MIN )) -ne "$TOTAL_MIN" ]; then
  echo "FATAL: TOTAL_MIN=$TOTAL_MIN is not a whole number of SEG_MIN=$SEG_MIN segments" >&2; exit 1
fi
# The case's OWN dt, from cases.json -- a boundary that does not land on a step leaves the
# next segment hunting for a restart at a time no step ever produced.
if [ $(( (SEG_MIN * 60) % DT )) -ne 0 ] || [ $(( (TOTAL_MIN * 60) % DT )) -ne 0 ]; then
  echo "FATAL: $CASE has time_step=${DT}s, so a segment boundary must land on a step;" >&2
  echo "       SEG_MIN=$SEG_MIN (${SEG_MIN}*60=$((SEG_MIN*60))s) or TOTAL_MIN=$TOTAL_MIN does not divide by $DT" >&2
  exit 1
fi
for spec in "$RANKS $OMP $NX $NY ref" "$TRANKS $TOMP $TNX $TNY TWIN"; do
  set -- $spec
  if [ $(( $1 % NODES )) -ne 0 ]; then echo "FATAL: $5 ranks ($1) not divisible by NODES ($NODES)" >&2; exit 1; fi
  if [ $(( $1 * $2 )) -gt $(( NODES * CORES_PER_NODE )) ]; then
    echo "FATAL: $5 ranks*threads = $(( $1 * $2 )) exceeds $NODES nodes x $CORES_PER_NODE cores" >&2; exit 1
  fi
  if [ $(( $3 * $4 )) -ne "$1" ]; then
    echo "FATAL: $5 nproc_x*nproc_y (${3}x${4}) must equal its rank count ($1)" >&2; exit 1
  fi
done
if [ "$NX" = "$TNX" ] && [ "$NY" = "$TNY" ] && [ "$RANKS" = "$TRANKS" ] && [ "$OMP" = "$TOMP" ]; then
  echo "FATAL: TWIN must differ from the reference -- it measures the chaos floor" >&2; exit 1
fi
# WRF refuses any patch thinner than 10 cells on either axis and aborts during domain setup.
# Catch an illegal TWIN here rather than one arm into the job.
for pair in "$NX $NY ref" "$TNX $TNY TWIN"; do
  set -- $pair
  if [ $(( E_WE / $1 )) -lt 10 ] || [ $(( E_SN / $2 )) -lt 10 ]; then
    echo "FATAL: $3 decomposition ${1}x${2} gives a patch thinner than WRF's 10-cell minimum" >&2
    echo "       (e_we=$E_WE/$1 = $((E_WE/$1)), e_sn=$E_SN/$2 = $((E_SN/$2)))" >&2
    exit 1
  fi
done

NEVER_MIN=1000000   # a history_interval that can never elapse (same value configs.py uses)
# t=0 of the logical run is a property of the CASE: read it out of the case's own namelist
# template instead of hard-coding a date per case.
BASE_START=$(python3 "$HERE/segnl.py" show-start "$NAMELIST_TEMPLATE") || {
  echo "FATAL: could not read start time from $NAMELIST_TEMPLATE" >&2; exit 1; }
END_STAMP=$(python3 - "$BASE_START" "$TOTAL_MIN" <<'PY'
import sys, datetime as dt
t = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d_%H:%M:%S") + dt.timedelta(minutes=int(sys.argv[2]))
print(t.strftime("%Y-%m-%d_%H:%M:%S"))
PY
)

# PLAN_ONLY=1 resolves and prints the whole configuration, then exits without touching the
# filesystem or launching anything -- so a shape can be reviewed, and its SU quoted, BEFORE
# it is submitted rather than after.
if [ "${PLAN_ONLY:-0}" = "1" ]; then
  echo "PLAN case=$CASE ($CASE_FAMILY, ${E_WE}x${E_SN}, max_dom=$MAX_DOM, dt=${DT}s)"
  echo "  template   $NAMELIST_TEMPLATE"
  echo "  bp inputs  $BP_DIR"
  echo "  logical    ${TOTAL_MIN} min, $BASE_START -> $END_STAMP  ($((TOTAL_MIN*60/DT)) d01 steps)"
  echo "  segments   $NSEG x ${SEG_MIN} min = $((SEG_MIN*60/DT)) steps each"
  echo "  reference  ${NX}x${NY}  ${RANKS} ranks x ${OMP} threads, ${NUMTILES} tiles, adios2_numaggregators=${RANKS}"
  echo "  TWIN       ${TNX}x${TNY}  ${TRANKS} ranks x ${TOMP} threads, ${TTILES} tiles, adios2_numaggregators=${TRANKS}"
  echo "  nodes      $NODES  (ref $((RANKS*OMP)) cores, TWIN $((TRANKS*TOMP)) cores, of $((NODES*CORES_PER_NODE)))"
  echo "  arms       $ARMS"
  echo "  per-segment timeout ${TIMEOUT_S}s"
  exit 0
fi

mkdir -p "$OUT"
ARMS_CSV="$OUT/arms.csv"
echo "arm,rk_pattern,nsegs,nx,ny,ranks,omp,tiles,ok,wall_s,solver_s,steps,nc" > "$ARMS_CSV"

module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false
export OMP_NUM_THREADS=$OMP KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

echo "job $J  nodes ${SLURM_JOB_NODELIST:-?}  case $CASE ($CASE_FAMILY, ${E_WE}x${E_SN}, max_dom=$MAX_DOM)"
echo "  inputs: $BP_DIR   namelist template: $NAMELIST_TEMPLATE"
echo "logical run: ${TOTAL_MIN} min from $BASE_START, $NSEG x ${SEG_MIN} min (dt=${DT}s, $((TOTAL_MIN*60/DT)) d01 steps total)"
echo "decomposition ${NX}x${NY} (TWIN ${TNX}x${TNY}), $RANKS ranks x $OMP threads on $NODES nodes, $NUMTILES tiles, adios2_numaggregators=$RANKS"
echo "arms: $ARMS"
mmlsquota -u "$USER" home1 2>/dev/null | tail -2

# ---------------------------------------------------------------- rk_ord pattern per arm
rk_pattern () {   # arm -> space separated rk_ord, one per segment
  local arm=$1 i out=""
  case "$arm" in
    A|A2|TWIN) for ((i=0;i<${2};i++)); do out="$out 3"; done ;;
    B)         for ((i=0;i<${2};i++)); do out="$out 2"; done ;;
    C)         for ((i=0;i<${2};i++)); do if [ $((i%2)) -eq 0 ]; then out="$out 3"; else out="$out 2"; fi; done ;;
    D)         for ((i=0;i<${2};i++)); do if [ "$i" -eq 0 ]; then out="$out 3"; else out="$out 2"; fi; done ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
  echo "${out# }"
}

stage () {   # rundir
  local d=$1
  rm -rf "$d"; mkdir -p "$d"; cd "$d" || exit 1
  local dom df
  for ((dom = 1; dom <= MAX_DOM; dom++)); do            # a nest has wrfinput_d02.bp as well
    df=$(printf "d%02d" "$dom")
    ln -sf "$BP_DIR/wrfinput_${df}.bp" "wrfinput_${df}.bp"
  done
  ln -sf "$BP_DIR/wrfbdy_d01.bp"   wrfbdy_d01.bp        # boundaries are d01-only, always
  # Thompson microphysics WRITES these tables into the run dir when they are absent
  # (module_mp_thompson.F) -- ~350 MB and several minutes per run.  Share one copy.
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

# ---------------------------------------------------------------- one arm
run_arm () {
  local arm=$1 nsegs=$2 nx=$3 ny=$4 ranks=$5 omp=$6 tiles=$7
  local rk=($(rk_pattern "$arm" "$nsegs"))
  local RUNDIR="$OUT/${arm}"
  local wall=0 solver=0 steps=0 ok=1 i t0off hmode rstin rstout rc segwall
  # A one-call arm's "segment" is the WHOLE logical run -- using SEG_MIN here would silently
  # make arms A/B/TWIN 3 h long while A2/C/D were 12 h, and every comparison meaningless.
  local seglen=$(( TOTAL_MIN / nsegs ))

  echo
  echo "############ arm $arm : $nsegs x ${seglen} min, rk_ord = ${rk[*]}, ${nx}x${ny} on ${ranks} ranks x ${omp} threads, ${tiles} tiles"
  stage "$RUNDIR"

  for ((i = 0; i < nsegs; i++)); do
    t0off=$(( i * seglen ))
    if [ "$i" -eq 0 ]; then rstin=false; else rstin=true; fi
    # A segmented arm needs a restart at every boundary; the LAST segment's value is
    # irrelevant (see the alarm-inheritance note below) but is kept honest.  A one-call
    # arm writes none: proven, arm A produced zero "Writing restart" lines.
    if [ "$nsegs" -gt 1 ]; then rstout=$seglen; else rstout=0; fi

    # --- history: TWO frames per arm, at t=0 and at t=TOTAL_MIN ----------------------
    # TWO, not one: fidelity_gate.py returns INVALID ("no matched valid time after
    # initialization") when a role has a single frame, because that frame IS the
    # reference's first valid time and the gate deliberately refuses to decide on t=0.
    #
    # And the SAME history_interval in EVERY segment, because of the single hardest fact
    # in this whole script (found the expensive way, dry runs 1080954 and 1080960):
    #
    #   WRF STORES EVERY ALARM'S COUNTDOWN IN THE RESTART FILE AND RESTORES IT ON
    #   RESTART, OVERRIDING THE NAMELIST.
    #
    #   share/output_wrf.f90:511-540 writes WRF_ALARM_ISRINGING_nn and
    #   WRF_ALARM_SECS_TIL_NEXT_RING_nn; share/input_wrf.f90:392-435 reads them back and
    #   does `ringTime = curtime + seconds`, keeping only the RingInterval from the
    #   namelist.  (config_flags%override_restart_timers overrides this for the RESTART
    #   alarm only -- never for history.)
    #
    # So a chain whose first segment says "history off" or "interval = 1000000" bakes THAT
    # countdown into every later segment, and the last segment writes nothing no matter
    # what its own namelist says.  Both earlier designs failed exactly there, silently:
    # arms A/B/TWIN produced their frames and A2/C/D produced none.  The same inheritance
    # is why the last segment wrote a wrfrst despite restart_interval=1000000.
    #
    # The fix is to stop fighting it: give every segment the interval the WHOLE logical
    # run wants (TOTAL_MIN) with history_begin=0.  Segment 0 then rings at t=0 (writing
    # the initial frame) and hands on a countdown that lands exactly on t=TOTAL_MIN in the
    # final segment.  A one-call arm gets both frames from the same setting, unchanged.
    python3 "$CAMPAIGN/configs.py" edit-namelist "$PREP_DIR/namelist.input" ./namelist.input \
      --nproc-x "$nx" --nproc-y "$ny" --numtiles "$tiles" \
      --run-minutes "$seglen" --max-dom "$MAX_DOM" \
      --history-mode interval --history-minutes "$TOTAL_MIN" --ranks "$ranks" || { ok=0; break; }
    # then the four segment-specific keys configs.py deliberately does not own
    python3 "$HERE/segnl.py" ./namelist.input --base-start "$BASE_START" \
      --offset-min "$t0off" --run-min "$seglen" --rk-ord "${rk[$i]}" \
      --restart "$rstin" --restart-interval-min "$rstout" --max-dom "$MAX_DOM" || { ok=0; break; }
    cp namelist.input "$OUT/${arm}_seg$(printf %02d $i).namelist.input"

    segwall=$(date +%s.%N)
    OMP_NUM_THREADS=$omp \
    timeout "$TIMEOUT_S" srun --mpi=pmi2 --distribution=cyclic -n "$ranks" \
      --ntasks-per-node=$(( ranks / NODES )) -c "$omp" ./wrf.exe > "wrf_seg$i.out" 2>&1
    rc=$?
    segwall=$(python3 -c "import sys;print('%.2f'%(__import__('time').time()-float(sys.argv[1])))" "$segwall")

    if [ ! -f rsl.error.0000 ] || ! grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then
      echo "  seg$i FAILED rc=$rc wall=${segwall}s -- last rsl.error lines:"
      tail -15 rsl.error.0000 2>/dev/null | sed 's/^/      /'
      ok=0
    fi
    # WRF's OWN record of what it parsed -- this, not our namelist.input, is the proof that
    # rk_ord differed per segment.
    cp -f namelist.output "$OUT/${arm}_seg$(printf %02d $i).namelist.output" 2>/dev/null
    cp -f rsl.out.0000    "$OUT/${arm}_seg$(printf %02d $i).rsl.out"  2>/dev/null
    cp -f rsl.error.0000  "$OUT/${arm}_seg$(printf %02d $i).rsl.err"  2>/dev/null

    local nstep sseg
    # DOMAIN 1 ONLY.  A nested run prints one "Timing for main ... on domain N" line per
    # domain per step, and the domain-1 line ALREADY INCLUDES the nested integration
    # (smoke job 1074515: d01 lines sum 758.7 s of an 841 s wall, while d02+d03 alone sum
    # 821 s).  Counting every line double-counts the nest and inflates both the step count
    # and the solver time.
    grep "Timing for main" rsl.out.0000 2>/dev/null \
      | sed -nE 's/.*on domain +([0-9]+): +([0-9.]+) elapsed.*/\1 \2/p' > steps_by_domain.txt
    nstep=$(awk '$1 == 1' steps_by_domain.txt | wc -l)
    sseg=$(awk '$1 == 1 { s += $2 } END { printf "%.3f", s + 0 }' steps_by_domain.txt)
    cp -f steps_by_domain.txt "$OUT/${arm}_seg$(printf %02d $i).steps_by_domain.txt" 2>/dev/null
    steps=$(( steps + nstep ))
    wall=$(python3 -c "print('%.2f'%($wall+$segwall))")
    solver=$(python3 -c "print('%.3f'%($solver+${sseg:-0}))")
    echo "  seg$i rk_ord=${rk[$i]} restart=$rstin rst_int=$rstout rc=$rc wall=${segwall}s steps=$nstep solver=${sseg}s  [$(grep -c 'SUCCESS COMPLETE WRF' rsl.error.0000 2>/dev/null) success]"

    rm -f rsl.out.0* rsl.error.0*
    # the restart this segment CONSUMED is no longer needed (ADIOS2 frames are directories)
    if [ "$i" -gt 0 ]; then
      local consumed
      consumed=$(python3 - "$BASE_START" "$t0off" <<'PY'
import sys, datetime as dt
t = dt.datetime.strptime(sys.argv[1], "%Y-%m-%d_%H:%M:%S") + dt.timedelta(minutes=int(sys.argv[2]))
print(t.strftime("%Y-%m-%d_%H:%M:%S"))
PY
)
      # EVERY domain: a nested run writes wrfrst_d01_<t> and wrfrst_d02_<t>, and a karina
      # restart is several GB, so leaving the d02 copies behind fills the filesystem.
      rm -rf ./wrfrst_d0*_"$consumed"
    fi
    [ "$ok" = "1" ] || break
  done

  # the two history frames: t=0 and t=TOTAL_MIN.  Both are converted -- fidelity_gate.py
  # needs the initial frame present in every role AND a matched time strictly after it.
  local frames k nc nc_last=""
  frames=$(ls -d wrfout_d01_* 2>/dev/null | sort)
  if [ "$ok" = "1" ]; then
    if [ "$(echo "$frames" | grep -c .)" -lt 2 ]; then
      echo "  WRONG FRAME COUNT for $arm: expected 2 (t=0 and t=${TOTAL_MIN}min), got:"
      echo "$frames" | sed 's/^/      /'
      ok=0
    fi
    # the final frame MUST be at the logical end time, or the arms are not comparable
    if [ "$(echo "$frames" | tail -1)" != "wrfout_d01_${END_STAMP}" ] && [ "$ok" = "1" ]; then
      echo "  WRONG FINAL FRAME TIME for $arm: want wrfout_d01_${END_STAMP}, got $(echo "$frames" | tail -1)"
      ok=0
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
  rm -rf ./wrfout_d0* ./wrfrst_d0*          # BP frames are directories; every domain

  echo "$arm,\"$(rk_pattern "$arm" "$nsegs")\",$nsegs,$nx,$ny,$ranks,$omp,$tiles,$ok,$wall,$solver,$steps,$nc_last" >> "$ARMS_CSV"
  echo "  ARM $arm ok=$ok wall=${wall}s solver=${solver}s steps=$steps"
  cd "$OUT" || exit 1
}

# ---------------------------------------------------------------- run every arm
for arm in $ARMS; do
  case "$arm" in
    A|B|TWIN) nsegs=1 ;;
    A2|C|D)   nsegs=$NSEG ;;
    *) echo "unknown arm '$arm'" >&2; continue ;;
  esac
  if [ "$arm" = "TWIN" ]; then run_arm "$arm" "$nsegs" "$TNX" "$TNY" "$TRANKS" "$TOMP" "$TTILES"
  else                         run_arm "$arm" "$nsegs" "$NX"  "$NY"  "$RANKS"  "$OMP"  "$NUMTILES"; fi
done

# ---------------------------------------------------------------- per-segment rk_ord proof
echo
echo "================ rk_ord ACTUALLY USED, from WRF's own namelist.output ================"
for f in "$OUT"/*_seg*.namelist.output; do
  [ -e "$f" ] || continue
  tag=$(basename "$f" .namelist.output)
  rsl="$OUT/${tag}.rsl.out"
  # the simulated window as the run itself reported it -- stronger than any namelist echo
  first=$(grep -m1 "Timing for main" "$rsl" 2>/dev/null | sed -nE 's/.*time ([0-9_:-]+) on domain.*/\1/p')
  last=$(grep  "Timing for main" "$rsl" 2>/dev/null | tail -1 | sed -nE 's/.*time ([0-9_:-]+) on domain.*/\1/p')
  printf "%-14s namelist.input rk_ord=%-2s  WRF namelist.output RK_ORD=%-2s  restart=%-3s  simulated %s .. %s\n" \
    "$tag" \
    "$(grep -oE '\brk_ord *= *[0-9]+' "${f%.namelist.output}.namelist.input" | grep -oE '[0-9]+$')" \
    "$(grep -iE '^ *RK_ORD *=' "$f" | grep -oE '[0-9]+')" \
    "$(grep -iE '^ *RESTART *=' "$f" | head -1 | awk '{print $3}' | tr -d ,)" \
    "${first:-?}" "${last:-?}"
done
echo "-------- and the restart chain as the rsl logs saw it --------"
grep -H "RESTART run: opening\|Timing for processing restart" "$OUT"/*_seg*.rsl.out 2>/dev/null \
  | sed "s#$OUT/##" | head -40

# ---------------------------------------------------------------- bit-identity control
echo
# compare the FINAL frame only (f1): the t=0 frame is the same initial condition in every
# arm and proves nothing.
A_LAST=$(ls "$OUT"/A_f*.nc 2>/dev/null | sort | tail -1)
A2_LAST=$(ls "$OUT"/A2_f*.nc 2>/dev/null | sort | tail -1)
if [ -s "${A_LAST:-/nonexistent}" ] && [ -s "${A2_LAST:-/nonexistent}" ]; then
  $PY "$HERE/bitcmp.py" "$A_LAST" "$A2_LAST" \
    --label "A2 (restart chain, rk_ord=3) vs A (one call, rk_ord=3)" \
    --json-out "$OUT/bitident.json" 2>&1 | tail -20
else
  echo "BIT-IDENTITY CONTROL SKIPPED: A or A2 final netCDF missing"
fi

# ---------------------------------------------------------------- fidelity vs A
echo
for arm in A2 B C D TWIN; do
  ref_files=($(ls "$OUT"/A_f*.nc 2>/dev/null))
  twin_files=($(ls "$OUT"/TWIN_f*.nc 2>/dev/null))
  cand_files=($(ls "$OUT"/${arm}_f*.nc 2>/dev/null))
  if [ ${#ref_files[@]} -eq 0 ] || [ ${#twin_files[@]} -eq 0 ] || [ ${#cand_files[@]} -eq 0 ]; then
    echo "fidelity $arm: SKIPPED (need A_f*.nc, TWIN_f*.nc and ${arm}_f*.nc)"; continue
  fi
  # family=dynamics: rk_ord is a dynamics knob, and fidelity_gate's 'dynamics' family is
  # core-fields-only (W, the field of interest, is already a core field).
  # Every role gets BOTH frames: the gate refuses to decide on a single time, because that
  # single time is the reference's own initialisation ("no matched valid time after
  # initialization").
  $PY "$FIDELITY" --reference "${ref_files[@]}" --twin "${twin_files[@]}" \
    --candidate "${cand_files[@]}" \
    --family dynamics --out-prefix "$OUT/fidelity_${arm}" > "$OUT/fidelity_${arm}.log" 2>&1
  echo "fidelity $arm: $(grep -h FIDELITY-GATE "$OUT/fidelity_${arm}.log" | tail -1)"
  [ -s "$OUT/fidelity_${arm}.json" ] || tail -5 "$OUT/fidelity_${arm}.log" | sed 's/^/    /'
done

# ---------------------------------------------------------------- one summary table
RK_ALT_CASE="$CASE" $PY "$HERE/summarize.py" --dir "$OUT"

if [ "$KEEP_NC" != "1" ]; then rm -f "$OUT"/*.nc; fi
rm -rf "$OUT"/A "$OUT"/A2 "$OUT"/B "$OUT"/C "$OUT"/D "$OUT"/TWIN
echo
echo "artifacts: $OUT"
du -sh "$OUT" 2>/dev/null
echo "=== done ==="

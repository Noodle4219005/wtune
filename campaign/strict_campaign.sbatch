#!/usr/bin/env bash
#SBATCH --job-name=strictcamp
#SBATCH --account=ACD110018
#SBATCH --partition=cf448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=02:00:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/campaign/strictcamp-%x-%j.out
#
# STRICT-configuration data campaign runner (predictor phase B). Measures every
# STRICT config campaign/configs.py generates for one WRF case, with replicates in
# palindromic order, and appends one row per run to the predictor run-ledger CSV.
# All-ADIOS2 (2026-09-14 user decision): every run's namelist is edited with ADIOS2
# I/O (io_form 14 everywhere, adios2_numaggregators=ranks -- see configs.py's
# _apply_adios2 and regress_bp.sbatch/job 1072441). MODE picks the history policy:
#   timing      (default) -- no run writes history at all (--history-mode off);
#                             fastest, cleanest wall-clock measurement.
#   correctness -- d01 writes exactly one (final) frame (--history-mode end_d01),
#                  converted to netCDF and diffwrf'd against the first successful
#                  run's netCDF; use with a small CONFIGS subset and REPS=1.
#
# Usage: sbatch --export=ALL,CASE=karina campaign/strict_campaign.sbatch
#        sbatch --export=ALL,CASE=karina,MODE=correctness,CONFIGS=C01+C05,REPS=1 campaign/strict_campaign.sbatch
# Env:   CASE (required), MODE (timing|correctness, default timing), REPS (default
#        2), STEPS (default: cases.json's "steps" for this case), TIMEOUT_S
#        (default 600). Override --time per case at submission -- the lead's job,
#        not this script's.
set -uo pipefail

W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
CAMPAIGN=/home/u5727520/wrf-stack/wtune-autotuner/campaign
OUT=/home/u5727520/wrf-bench/wscore/campaign
# derived from the node pool (icpnq = ct448 SNC 8x14, icpnp = cf448 2 NUMA) so ledgers from
# different pools never share a hardware_id
HARDWARE_ID=f1-spr-$(hostname -s | sed 's/[0-9]*$//')
PIXI_ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
CONV=/home/u5727520/SC26-WRF-p0/scripts/vendor/wrf-adios2-to-netcdf4/convert.py

REPS=${REPS:-2}
TIMEOUT_S=${TIMEOUT_S:-600}
MODE=${MODE:-timing}
case "$MODE" in
  timing) HISTORY_MODE_ARG=off ;;
  correctness) HISTORY_MODE_ARG=end_d01 ;;
  *) echo "MODE must be 'timing' or 'correctness', got '$MODE'" >&2; exit 1 ;;
esac
# config generation needs Python >= 3.7 so configs.py can import wtune_decomp's scorer
# (it uses `from __future__ import annotations`; the system python3 is 3.6 and silently falls back)
PY="pixi run --manifest-path $PIXI_ENV/pixi.toml python"
: "${CASE:?CASE env var is required, e.g. --export=ALL,CASE=karina}"

mkdir -p "$OUT"

module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

J=$SLURM_JOB_ID
LEDGER="$OUT/ledger_${CASE}_${J}.csv"
LEDGER_HEADER="run_id,source_dataset,job_id,rep_index,case_raw,case_base,case_family,hardware_id,binary_kind,compliance_class,e_we,e_sn,e_vert,dx,dy,time_step,mp_physics,cu_physics,bl_pbl_physics,sf_sfclay_physics,sf_surface_physics,ra_lw_physics,ra_sw_physics,radt,cudt,history_interval,io_form_history,n_nodes,n_ranks,omp_threads,nproc_x,nproc_y,numtiles,distribution,kmp_blocktime,s_per_step_median,s_per_step_mean,n_steps_measured,diffwrf_differing_fields,fidelity_label,profile_micro_share,profile_radiation_share,profile_dynamics_share,profile_mpi_share,profile_io_share,profile_rank_imbalance,profile_thread_imbalance,notes"
[ -f "$LEDGER" ] || echo "$LEDGER_HEADER" > "$LEDGER"

echo "job $J case $CASE nodes $SLURM_JOB_NODELIST reps $REPS timeout $TIMEOUT_S"

# ---- case metadata from cases.json --------------------------------------------
eval "$(python3 - "$CASE" <<'PY'
import json, sys
case_id = sys.argv[1]
d = json.load(open("/home/u5727520/wrf-stack/wtune-autotuner/campaign/cases.json"))["cases"]
c = d.get(case_id)
if c is None:
    print('echo "unknown case %s" >&2; exit 1' % case_id)
    sys.exit(0)
if not c.get("enabled", True):
    print('echo "case %s is disabled: %s" >&2; exit 1' % (case_id, c.get("reason", "")))
    sys.exit(0)
phys = c["physics"]
def q(v):
    return "'%s'" % str(v).replace("'", "'\\''")
print("PREP_DIR=%s" % q(c["prep_dir"]))
print("BP_DIR=%s" % q(c["bp_dir"]))
print("RESTART=%s" % (1 if c.get("restart") else 0))
print("DT=%s" % q(c["dt"]))
print("STEPS_DEFAULT=%s" % q(c["steps"]))
print("CASE_FAMILY=%s" % q(c["case_family"]))
print("E_WE=%s" % q(c["e_we"]))
print("E_SN=%s" % q(c["e_sn"]))
print("E_VERT=%s" % q(c["e_vert"]))
print("DX=%s" % q(c["dx"]))
print("DY=%s" % q(c["dy"]))
print("INPUT_FILES=%s" % q(" ".join(c["input_files"])))
print("MAX_DOM=%s" % q(c.get("max_dom", 1)))
print("NAMELIST_TEMPLATE=%s" % q(c.get("namelist_template") or (c["prep_dir"] + "/namelist.input")))
overrides = c.get("namelist_overrides") or {}
print("NAMELIST_OVERRIDES=%s" % q(" ".join("%s=%s" % (k, v) for k, v in overrides.items())))
delete_keys = c.get("namelist_delete_keys") or []
print("NAMELIST_DELETE_KEYS=%s" % q(" ".join(delete_keys)))
print("NEST_HISTORY=%s" % (1 if c.get("nest_history") else 0))
for k in ("mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics",
          "sf_surface_physics", "ra_lw_physics", "ra_sw_physics", "radt", "cudt"):
    print("PHYS_%s=%s" % (k.upper(), q(phys.get(k, ""))))
print("PHYSICS_SUITE=%s" % q(phys.get("physics_suite", "")))
PY
)"

STEPS=${STEPS:-$STEPS_DEFAULT}
MIN=$(( (STEPS * DT + 59) / 60 ))
read -ra INPUT_FILES_ARR <<< "$INPUT_FILES"

echo "case metadata: e_we=$E_WE e_sn=$E_SN e_vert=$E_VERT dt=$DT restart=$RESTART steps=$STEPS run_minutes=$MIN"

# ---- resolved configs for this case (TSV) and campaign order -------------------
declare -A CFG_RANKS CFG_OMP CFG_NX CFG_NY CFG_NUMTILES CFG_DIST CFG_BLOCKTIME
while IFS=$'\t' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy; do
  [ "$cid" = "config_id" ] && continue
  [ -z "$cid" ] && continue
  CFG_RANKS[$cid]=$ranks; CFG_OMP[$cid]=$omp; CFG_NX[$cid]=$nx; CFG_NY[$cid]=$ny
  CFG_NUMTILES[$cid]=$numtiles; CFG_DIST[$cid]=$dist; CFG_BLOCKTIME[$cid]=$blocktime
done < <($PY "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null)

mapfile -t ORDER < <($PY "$CAMPAIGN/configs.py" order "$CASE" --reps "$REPS")
# optional subset for smoke tests: CONFIGS=C02+C05 keeps only those ids, preserving the order
# ('+' separated: sbatch --export splits its argument on commas, so CONFIGS=C02,C05 would arrive as C02)
if [ -n "${CONFIGS:-}" ]; then
  KEEP=",${CONFIGS//+/,},"
  FILTERED=()
  for c in "${ORDER[@]}"; do case "$KEEP" in *",$c,"*) FILTERED+=("$c") ;; esac; done
  ORDER=("${FILTERED[@]}")
fi
echo "order (${#ORDER[@]} runs): ${ORDER[*]}"
if [ ${#ORDER[@]} -eq 0 ]; then
  echo "no configs resolved for case $CASE" >&2
  exit 1
fi
declare -A REPCOUNT
REF_NC=""
REF_CONFIG=""

for cid in "${ORDER[@]}"; do
  REPCOUNT[$cid]=$(( ${REPCOUNT[$cid]:-0} + 1 ))
  rep=${REPCOUNT[$cid]}
  ranks=${CFG_RANKS[$cid]}; omp=${CFG_OMP[$cid]}; nx=${CFG_NX[$cid]}; ny=${CFG_NY[$cid]}
  numtiles=${CFG_NUMTILES[$cid]}; dist=${CFG_DIST[$cid]}; blocktime=${CFG_BLOCKTIME[$cid]}
  ntasks_per_node=$(( ranks / 4 ))

  RUNDIR="$OUT/${CASE}_${cid}_r${rep}_${J}"
  rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"

  for f in "${INPUT_FILES_ARR[@]}"; do
    ln -sf "$PREP_DIR/$f" "$f"
  done
  # all-ADIOS2: wrfinput_d0N.bp (N <= max_dom) and wrfbdy_d01.bp come from bp_dir,
  # never from prep_dir's netCDF originals (input_files no longer lists them).
  for ((d = 1; d <= MAX_DOM; d++)); do
    df=$(printf "d%02d" "$d")
    ln -sf "$BP_DIR/wrfinput_${df}.bp" "wrfinput_${df}.bp"
  done
  ln -sf "$BP_DIR/wrfbdy_d01.bp" "wrfbdy_d01.bp"
  for f in "$W"/run/*; do
    b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;
      *) [ -e "$b" ] || ln -sf "$f" "$b" ;; esac
  done
  cp "$NAMELIST_TEMPLATE" ./namelist.input.orig
  SET_ARGS=()
  if [ -n "${NAMELIST_OVERRIDES:-}" ]; then
    for kv in $NAMELIST_OVERRIDES; do
      SET_ARGS+=(--set "$kv")
    done
  fi
  DELETE_ARGS=()
  if [ -n "${NAMELIST_DELETE_KEYS:-}" ]; then
    for k in $NAMELIST_DELETE_KEYS; do
      DELETE_ARGS+=(--delete-key "$k")
    done
  fi
  NEST_ARGS=()
  [ "$NEST_HISTORY" = "1" ] && NEST_ARGS+=(--nest-history-on)
  python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \
    --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
    --run-minutes "$MIN" --max-dom "$MAX_DOM" \
    --history-mode "$HISTORY_MODE_ARG" --ranks "$ranks" \
    "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}"
  rm -f ./namelist.input.orig
  ln -sf "$W/main/wrf.exe" ./wrf.exe

  export OMP_NUM_THREADS=$omp
  if [ "$blocktime" = "default" ]; then
    unset KMP_BLOCKTIME
  else
    export KMP_BLOCKTIME=$blocktime
  fi

  T0=$(date +%s.%N)
  timeout "$TIMEOUT_S" srun --mpi=pmi2 --distribution="$dist" -n "$ranks" \
    --ntasks-per-node="$ntasks_per_node" -c "$omp" ./wrf.exe > wrf.out 2>&1
  RC=$?
  T1=$(date +%s.%N)
  WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

  OK=0
  if [ -f rsl.error.0000 ] && grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then
    OK=1
  fi

  # Nested runs print one "Timing for main ... on domain N" line per domain step, and the domain-1 line
  # ALREADY INCLUDES the nested integration (smoke job 1074515, isc21: domain-1 lines sum 758.7 s against
  # 841 s total wall including ~90 s of init/IO, while d02+d03 lines alone sum 821 s). So a d01 step is its
  # domain-1 line only; adding nested lines double-counts. steps_by_domain.txt keeps all raw lines.
  grep "Timing for main" rsl.out.0000 2>/dev/null | sed -nE 's/.*on domain +([0-9]+): +([0-9.]+) elapsed.*/\1 \2/p' > steps_by_domain.txt
  awk '$1 == 1 { print $2 }' steps_by_domain.txt > steps.txt

  # step_stats() (campaign/configs.py) computes median (radiation-excluded, as
  # before) and mean_all (ALL d01 steps 2..n, radiation included -- see the
  # ledger's s_per_step_mean column). Only trust steps.txt when the run itself
  # reported success; a failed/truncated run's partial steps.txt is not a real
  # sample.
  if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
    STATS=$($PY "$CAMPAIGN/configs.py" step-stats steps.txt)
  else
    NALL_RAW=$(wc -l < steps.txt 2>/dev/null || echo 0)
    NALL=$(( NALL_RAW > 0 ? NALL_RAW - 1 : 0 ))
    NEXCL0=0; [ -s steps.txt ] && NEXCL0=1
    STATS="med= mean= n=0 excl=$NEXCL0 nall=$NALL"
  fi
  # exact key lookup: a substring grep for 'n=' also matches inside 'mean=' (smoke job 1074466 printed n=0)
  kv () { echo "$STATS" | awk -v k="$1" '{for (i = 1; i <= NF; i++) {split($i, a, "="); if (a[1] == k) print a[2]}}'; }
  MED=$(kv med); MEAN=$(kv mean); NMEAS=$(kv n); NEXCL=$(kv excl); NALL=$(kv nall)

  # ---- correctness mode: convert the (only, thanks to history_mode=end_d01) BP
  # frame to netCDF outside the timed region above, diffwrf against the first
  # successful run's netCDF, then discard the BP frame and any non-reference nc.
  # timing mode never writes a wrfout_d0* frame at all (history_mode=off), so
  # this whole block is a no-op there and DIFF_FIELDS/DIFF_STATUS stay empty/off.
  DIFF_FIELDS=""
  DIFF_STATUS="off"
  CONVERT_RC=""
  if [ "$MODE" = "correctness" ]; then
    BP_FRAME=$(ls -d wrfout_d01_* 2>/dev/null | sort | tail -1)
    if [ "$RC" = "0" ] && [ "$OK" = "1" ] && [ -n "$BP_FRAME" ]; then
      NC_FILE="${BP_FRAME}.nc"
      srun -N1 -n1 -c8 bash -c "cd $PIXI_ENV && pixi run python $CONV --input $RUNDIR/$BP_FRAME --output $RUNDIR/$NC_FILE" \
        > convert.log 2>&1
      CONVERT_RC=$?
      if [ "$CONVERT_RC" != "0" ]; then
        rm -f "$NC_FILE"  # convert.py leaves a partial .nc behind on failure
        DIFF_STATUS="convert_failed"
      elif [ -z "$REF_NC" ]; then
        REF_NC="$RUNDIR/$NC_FILE"; REF_CONFIG="$cid"
        DIFF_STATUS="reference"
      elif [ -e "$REF_NC" ]; then
        "$W/external/io_netcdf/diffwrf" "$REF_NC" "$NC_FILE" > diffwrf_vs_ref.txt 2>&1
        DIFF_FIELDS=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' diffwrf_vs_ref.txt)
        if [ "$DIFF_FIELDS" = "0" ]; then DIFF_STATUS="bit_identical"; else DIFF_STATUS="diff_detected"; fi
        rm -f "$NC_FILE"
      else
        DIFF_STATUS="no_reference"
      fi
    else
      DIFF_STATUS="run_failed"
    fi
    [ -n "${BP_FRAME:-}" ] && rm -rf "$BP_FRAME"
  fi

  echo "RESULT case=$CASE config=$cid rep=$rep rc=$RC ok=$OK med=$MED mean=$MEAN n=$NMEAS excl=$NEXCL wall=$WALL diff_fields=$DIFF_FIELDS"

  # ---- ledger row ----
  # fidelity_label is reserved for PHYSICS-GRAY fidelity; STRICT diffwrf status goes to notes.
  # diffwrf only ever compares wrfout_d01_* frames (see BP_FRAME globs above), so a nested
  # (max_dom>1) case's correctness check covers d01 only -- record both facts in notes.
  NOTES="config=$cid n_excluded_steps=$NEXCL ref_config=$REF_CONFIG mode=$MODE io=adios2"
  if [ "$MODE" = "correctness" ]; then
    NOTES="$NOTES history=end_d01 diff_status=$DIFF_STATUS diff_domain=d01 max_dom=$MAX_DOM convert_rc=$CONVERT_RC"
  else
    NOTES="$NOTES history=off"
  fi
  NOTES="$NOTES mean_includes_radiation=1 n_all_steps=$NALL"
  [ "$RC" != "0" ] && NOTES="$NOTES rc=$RC"
  [ "$OK" != "1" ] && NOTES="$NOTES ok=$OK"
  export RUN_ID="${CASE}_${cid}_r${rep}_${J}" SRC_DATASET="strict_campaign" JOB_ID="$J" \
    REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
    HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="STRICT" \
    E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$DT" \
    MP_PHYSICS="$PHYS_MP_PHYSICS" CU_PHYSICS="$PHYS_CU_PHYSICS" BL_PBL_PHYSICS="$PHYS_BL_PBL_PHYSICS" \
    SF_SFCLAY_PHYSICS="$PHYS_SF_SFCLAY_PHYSICS" SF_SURFACE_PHYSICS="$PHYS_SF_SURFACE_PHYSICS" \
    RA_LW_PHYSICS="$PHYS_RA_LW_PHYSICS" RA_SW_PHYSICS="$PHYS_RA_SW_PHYSICS" \
    RADT="$PHYS_RADT" CUDT="$PHYS_CUDT" \
    HISTORY_INTERVAL="$([ "$MODE" = correctness ] && echo "$MIN" || echo 0)" IO_FORM_HISTORY="14" \
    N_NODES="4" N_RANKS="$ranks" OMP_THREADS="$omp" NPROC_X="$nx" NPROC_Y="$ny" \
    NUMTILES="$numtiles" DISTRIBUTION="$dist" KMP_BLOCKTIME_COL="$blocktime" \
    S_PER_STEP_MEDIAN="$MED" S_PER_STEP_MEAN="$MEAN" N_STEPS_MEASURED="$NMEAS" \
    DIFFWRF_DIFFERING_FIELDS="$DIFF_FIELDS" FIDELITY_LABEL="" NOTES="$NOTES" \
    PHYSICS_SUITE_NOTE="$PHYSICS_SUITE"
  python3 - "$LEDGER" <<'PY'
import csv, os, sys
ledger = sys.argv[1]
e = os.environ.get
notes = e("NOTES", "")
suite = e("PHYSICS_SUITE_NOTE", "")
if suite:
    notes = "physics_suite=%s %s" % (suite, notes)
row = [
    e("RUN_ID"), e("SRC_DATASET"), e("JOB_ID"), e("REP_INDEX"), e("CASE_RAW"), e("CASE_BASE"),
    e("CASE_FAMILY"), e("HARDWARE_ID"), e("BINARY_KIND"), e("COMPLIANCE_CLASS"),
    e("E_WE"), e("E_SN"), e("E_VERT"), e("DX"), e("DY"), e("TIME_STEP"),
    e("MP_PHYSICS"), e("CU_PHYSICS"), e("BL_PBL_PHYSICS"), e("SF_SFCLAY_PHYSICS"),
    e("SF_SURFACE_PHYSICS"), e("RA_LW_PHYSICS"), e("RA_SW_PHYSICS"), e("RADT"), e("CUDT"),
    e("HISTORY_INTERVAL"), e("IO_FORM_HISTORY"), e("N_NODES"), e("N_RANKS"), e("OMP_THREADS"),
    e("NPROC_X"), e("NPROC_Y"), e("NUMTILES"), e("DISTRIBUTION"), e("KMP_BLOCKTIME_COL"),
    e("S_PER_STEP_MEDIAN", ""), e("S_PER_STEP_MEAN", ""), e("N_STEPS_MEASURED", ""),
    e("DIFFWRF_DIFFERING_FIELDS", ""), e("FIDELITY_LABEL"),
    "", "", "", "", "", "", "",  # profile_* columns, empty this campaign
    notes,
]
with open(ledger, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY

  # ---- cleanup: keep rsl.out.0000, rsl.error.0000, namelist.input, steps.txt,
  # diffwrf_vs_ref.txt; delete everything else. Every BP frame is already gone
  # (converted+deleted, or convert_failed and deleted, above) except the one
  # netCDF this iteration just became the correctness reference for -- exclude
  # only that (a plain generic safety net in timing mode, which writes nothing
  # matching wrfout_d0* in the first place).
  find . -maxdepth 1 -name 'rsl.*' ! -name 'rsl.out.0000' ! -name 'rsl.error.0000' -delete
  find . -maxdepth 1 -name 'wrfout_d0*' ! -name "$(basename "${REF_NC:-}")" -exec rm -rf {} +
  # keep the launcher/stdout log of failed runs: smoke job 1074658 (maria1km) timed out with no rsl.* files
  # at all, and deleting wrf.out removed the only evidence of where it stalled
  if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then rm -f wrf.out; fi

  cd "$OUT"
done

[ -n "$REF_NC" ] && [ -e "$REF_NC" ] && rm -f "$REF_NC"

echo "=== summary"
python3 - "$LEDGER" <<'PY'
import csv, re, statistics as st, sys
ledger = sys.argv[1]
by_cfg = {}
with open(ledger) as f:
    for row in csv.DictReader(f):
        med = row.get("s_per_step_median", "")
        if not med:
            continue
        # notes may start with physics_suite=... (conus cases), so search instead of assuming position
        m = re.search(r"(?:^|\s)config=(\S+)", row["notes"])
        cid = m.group(1) if m else ""
        by_cfg.setdefault(cid, []).append(float(med))
best_cfg, best_mean = None, None
for cid, meds in sorted(by_cfg.items()):
    mean = st.mean(meds)
    spread = (max(meds) - min(meds)) if len(meds) > 1 else 0.0
    print("CONFIG_SUMMARY config=%s reps=%d mean_of_medians=%.5f spread=%.5f" % (cid, len(meds), mean, spread))
    if best_mean is None or mean < best_mean:
        best_mean, best_cfg = mean, cid
if best_cfg:
    print("BEST config=%s mean_of_medians=%.5f" % (best_cfg, best_mean))
else:
    print("BEST none (no successful runs)")
PY
echo "=== done"

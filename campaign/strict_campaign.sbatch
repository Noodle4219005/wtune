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
#        sbatch --export=ALL,CASE=karina,CONFIG_FILE=/abs/path/sample.json campaign/strict_campaign.sbatch
# Env:   CASE (required), MODE (timing|correctness, default timing), REPS (default
#        2), STEPS (default: cases.json's "steps" for this case), TIMEOUT_S
#        (default 600). Override --time per case at submission -- the lead's job,
#        not this script's.
#        MIN_FREE_GIB (default 60): disk-quota guard threshold (see check_disk_guard
#        below) -- the job aborts itself with a ledger row instead of running into a
#        mid-write 'Disk quota exceeded' when home1 free space under the soft quota
#        drops below this many GiB.
#        CONFIG_FILE (WTune D1 decomposition-space work, predictor/decomp_space.py
#        to_campaign_configs/sample output): measures that generated config set instead of the
#        fixed C01-C16 STRICT templates in the default (non-MODE=physics) loop below -- same
#        palindromic REPS order, same ledger schema (tile geometry goes into notes, see
#        config_source= below). Unset (the default): every line below this comment behaves
#        exactly as it did before CONFIG_FILE existed. Only honored by the default STRICT
#        (timing/correctness) loop, not MODE=physics.
set -uo pipefail

W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
CAMPAIGN=/home/u5727520/wrf-stack/wtune-autotuner/campaign
OUT=/home/u5727520/wrf-bench/wscore/campaign
# derived from the node pool (icpnq = ct448 SNC 8x14, icpnp = cf448 2 NUMA) so ledgers from
# different pools never share a hardware_id
HARDWARE_ID=f1-spr-$(hostname -s | sed 's/[0-9]*$//')
PIXI_ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
# vendored into this repo 2026-09-15: the old external path
# /home/u5727520/SC26-WRF-p0/scripts/vendor/wrf-adios2-to-netcdf4/convert.py disappeared, and
# physics smoke 1076771-1076776 silently produced no netCDF / no fidelity at all.
CONV=${CONV:-/home/u5727520/wrf-stack/wtune-autotuner/tools/wrf_adios2_to_netcdf4/convert.py}
if [ "${MODE:-timing}" != "timing" ] && [ ! -f "$CONV" ]; then
  echo "FATAL: ADIOS2->netCDF converter not found at $CONV (needed by MODE=${MODE}) -- refusing to run" >&2
  exit 1
fi

REPS=${REPS:-2}
TIMEOUT_S=${TIMEOUT_S:-600}
MODE=${MODE:-timing}
case "$MODE" in
  timing) HISTORY_MODE_ARG=off ;;
  correctness) HISTORY_MODE_ARG=end_d01 ;;
  physics) HISTORY_MODE_ARG="" ;;  # MODE=physics picks its own per-replicate history mode below, not a single fixed value
  *) echo "MODE must be 'timing', 'correctness' or 'physics', got '$MODE'" >&2; exit 1 ;;
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

# ---- disk-quota guard (problem 1, 2026-09-15) -----------------------------------------
# Smoke jobs 1076238-1076340 kept every failed/incomplete MODE=physics run's BP history
# and converted netCDF whenever any expected fidelity JSON was missing, filling home1's
# soft quota (grace expired); later runs then failed mid-write with 'Disk quota exceeded'.
# check_disk_guard is called right before every WRF srun (both loops below) so a job
# stops itself with a clear message and a ledger row instead of running blind into that.
MIN_FREE_GIB=${MIN_FREE_GIB:-60}
check_disk_guard () {
  local ctx="$1" q_text free_gib
  if ! q_text=$(mmlsquota -u "$USER" home1 2>&1); then
    echo "WARNING: mmlsquota unavailable ($q_text) -- skipping disk guard for $ctx" >&2
    return 0
  fi
  free_gib=$(printf '%s\n' "$q_text" | $PY "$CAMPAIGN/configs.py" quota-free-gib)
  if [ -z "$free_gib" ] || [ "$free_gib" = "NA" ]; then
    echo "WARNING: could not parse mmlsquota output -- skipping disk guard for $ctx" >&2
    return 0
  fi
  if awk -v f="$free_gib" -v m="$MIN_FREE_GIB" 'BEGIN { exit !(f < m) }'; then
    echo "ABORT: home1 free space ${free_gib} GiB < MIN_FREE_GIB=${MIN_FREE_GIB} GiB before $ctx -- stopping job (case=$CASE job=$J) to avoid a mid-run Disk quota exceeded failure" >&2
    RUN_ID="${CASE}_${ctx}_${J}" JOB_ID="$J" CASE_RAW="$CASE" CASE_BASE="$CASE" \
      HARDWARE_ID="$HARDWARE_ID" NOTES_ABORT="status=ABORTED reason=low_disk free_gib=${free_gib} min_free_gib=${MIN_FREE_GIB} context=${ctx}" \
      python3 - "$LEDGER" <<'PY'
import csv, os, sys
ledger = sys.argv[1]
e = os.environ.get
row = [e("RUN_ID", ""), "guard", e("JOB_ID", ""), "", e("CASE_RAW", ""), e("CASE_BASE", ""),
       "", e("HARDWARE_ID", ""), "", ""] + [""] * 37 + [e("NOTES_ABORT", "")]
with open(ledger, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
    exit 1
  fi
  return 0
}

# ---- Thompson lookup-table shared cache (problem 2, 2026-09-15) ----------------------
# WRF's Thompson microphysics computes and WRITES freezeH2O.dat / qr_acr_qg_V4.dat (or
# qr_acr_qg_mp38V1.dat for mp_physics=38) / qr_acr_qsV2.dat into the run directory at
# init whenever a same-named file is not already there (module_mp_thompson.F:4120-4787;
# verified this WRF build's run/ ships none of them). A case whose prep_dir also lacks
# them (e.g. lafe2017) got a freshly-computed, run-directory-local copy every single run
# -- ~54 GiB duplicated across campaign run dirs. Share one copy per table name instead.
THOMPSON_TABLE_NAMES=(freezeH2O.dat qr_acr_qg_V4.dat qr_acr_qg_mp38V1.dat qr_acr_qsV2.dat)
TABLE_CACHE="$OUT/thompson_tables"
mkdir -p "$TABLE_CACHE"

link_thompson_tables () {
  # call from within RUNDIR, after input_files/BP symlinks are in place -- never
  # overrides a file the case's own input_files already symlinked in.
  local tbl
  for tbl in "${THOMPSON_TABLE_NAMES[@]}"; do
    if [ ! -e "$tbl" ] && [ -f "$TABLE_CACHE/$tbl" ]; then
      ln -sf "$TABLE_CACHE/$tbl" "$tbl"
    fi
  done
}

harvest_thompson_tables () {
  # call from within RUNDIR, right after the run finishes and before any cleanup that
  # might delete a freshly generated table before it is cached. A cached or case-shipped
  # table is always a symlink here and is never touched; only a real regular file was
  # generated by this run.
  local tbl
  for tbl in "${THOMPSON_TABLE_NAMES[@]}"; do
    if [ -f "$tbl" ] && [ ! -L "$tbl" ]; then
      if [ -f "$TABLE_CACHE/$tbl" ]; then
        rm -f "$tbl"
      else
        mv "$tbl" "$TABLE_CACHE/$tbl"
      fi
    fi
  done
}

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
print("PHYSICS_SIM_MINUTES_DEFAULT=%s" % q(c.get("physics_sim_minutes", 60)))
PY
)"

STEPS=${STEPS:-$STEPS_DEFAULT}
MIN=$(( (STEPS * DT + 59) / 60 ))
read -ra INPUT_FILES_ARR <<< "$INPUT_FILES"

echo "case metadata: e_we=$E_WE e_sn=$E_SN e_vert=$E_VERT dt=$DT restart=$RESTART steps=$STEPS run_minutes=$MIN"

# ---- resolved configs for this case (TSV) and campaign order -------------------
# CONFIG_FILE=path.json (predictor/decomp_space.py to_campaign_configs/sample output, WTune D1
# decomposition-space work): measure that generated config set instead of the fixed C01-C16
# STRICT templates. configs.py's own CASE_ID/order dispatch reads the CONFIG_FILE env var itself
# (see configs.py main()); when it is unset this whole block is byte-identical to before
# CONFIG_FILE existed.
declare -A CFG_RANKS CFG_OMP CFG_NX CFG_NY CFG_NUMTILES CFG_DIST CFG_BLOCKTIME
declare -A CFG_NTILES_X CFG_NTILES_Y CFG_TSZ_X CFG_TSZ_Y CFG_TSTRAT CFG_SRC
if [ -n "${CONFIG_FILE:-}" ]; then
  while IFS=$'\t' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy \
                          ntx nty tsx tsy tstrat src; do
    [ "$cid" = "config_id" ] && continue
    [ -z "$cid" ] && continue
    CFG_RANKS[$cid]=$ranks; CFG_OMP[$cid]=$omp; CFG_NX[$cid]=$nx; CFG_NY[$cid]=$ny
    CFG_NUMTILES[$cid]=$numtiles; CFG_DIST[$cid]=$dist; CFG_BLOCKTIME[$cid]=$blocktime
    CFG_NTILES_X[$cid]=$ntx; CFG_NTILES_Y[$cid]=$nty; CFG_TSZ_X[$cid]=$tsx; CFG_TSZ_Y[$cid]=$tsy
    CFG_TSTRAT[$cid]=$tstrat; CFG_SRC[$cid]=$src
  done < <($PY "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null)
else
  while IFS=$'\t' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy; do
    [ "$cid" = "config_id" ] && continue
    [ -z "$cid" ] && continue
    CFG_RANKS[$cid]=$ranks; CFG_OMP[$cid]=$omp; CFG_NX[$cid]=$nx; CFG_NY[$cid]=$ny
    CFG_NUMTILES[$cid]=$numtiles; CFG_DIST[$cid]=$dist; CFG_BLOCKTIME[$cid]=$blocktime
  done < <($PY "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null)
fi


# =================================================================================
# MODE=physics: PHYSICS-GRAY data campaign (decision 0051) -- timing + fidelity.
# Run set: P00 at DECOMP (reference), P00 at TWIN_DECOMP (twin, same physics/
# different STRICT decomposition -- measures the chaos floor), then each requested
# Pxx at DECOMP (candidate). Palindromic replicate order over that run-set exactly
# like the STRICT loop below, tracked by tag (P00_ref/P00_twin/Pxx) instead of by
# STRICT config id. Replicate 1 of every tag writes interval history (fidelity);
# replicate 2+ writes no history (clean timing only). Because cmd_physics_order
# always lists P00_ref/P00_twin FIRST in the entries handed to build_order(), and
# build_order's very first pass is forward, BOTH reference's and twin's own
# replicate 1 always complete before any candidate's replicate 1 begins for any
# REPS>=1 -- so reference/twin BP->netCDF conversion can happen inline, in one
# pass, with no separate deferred/pending-file bookkeeping needed.
# =================================================================================
if [ "$MODE" = "physics" ]; then
  FIDELITY_GATE="$(dirname "$CAMPAIGN")/fidelity_gate.py"
  if [ ! -f "$FIDELITY_GATE" ]; then
    echo "ERROR: fidelity_gate.py not found at $FIDELITY_GATE -- required for MODE=physics" >&2
    exit 1
  fi

  DECOMP=${DECOMP:-C02}
  TWIN_DECOMP=${TWIN_DECOMP:-C06}
  if [ -z "${CFG_RANKS[$DECOMP]:-}" ]; then
    echo "ERROR: DECOMP=$DECOMP is not a resolved STRICT config for case $CASE" >&2
    exit 1
  fi
  if [ -z "${CFG_RANKS[$TWIN_DECOMP]:-}" ]; then
    echo "ERROR: TWIN_DECOMP=$TWIN_DECOMP is not a resolved STRICT config for case $CASE" >&2
    exit 1
  fi

  SIM_MINUTES=${SIM_MINUTES:-$PHYSICS_SIM_MINUTES_DEFAULT}
  HIST_MINUTES_CALC=$(( (SIM_MINUTES + 3) / 6 ))
  [ "$HIST_MINUTES_CALC" -lt 5 ] && HIST_MINUTES_CALC=5
  HIST_MINUTES=${HIST_MINUTES:-$HIST_MINUTES_CALC}
  REPS=${REPS:-2}

  echo "physics campaign: case=$CASE decomp=$DECOMP twin_decomp=$TWIN_DECOMP sim_minutes=$SIM_MINUTES hist_minutes=$HIST_MINUTES reps=$REPS pconfigs=${PCONFIGS:-(default: all applicable)}"

  PCONFIGS_ARGS=()
  [ -n "${PCONFIGS:-}" ] && PCONFIGS_ARGS=(--pconfigs "$PCONFIGS")
  # 2>/dev/null + a tag-shaped grep filter: defends the run order against any stray
  # noise line (e.g. a pixi cache warning) a $PY invocation might print to stdout,
  # which would otherwise silently splice a bogus entry into PORDER.
  mapfile -t PORDER < <($PY "$CAMPAIGN/configs.py" physics-order "$CASE" "${PCONFIGS_ARGS[@]}" --reps "$REPS" \
    2>/dev/null | grep -E '^(P00_ref|P00_twin|P[0-9]+)$')
  echo "physics order (${#PORDER[@]} runs): ${PORDER[*]}"
  if [ ${#PORDER[@]} -eq 0 ]; then
    echo "no physics templates resolved for case $CASE" >&2
    exit 1
  fi

  FIDOUT="$OUT/fidelity_${CASE}_${J}"
  mkdir -p "$FIDOUT"

  declare -A PREPCOUNT
  declare -A EXPECTED_CANDIDATE
  REF_OK=0; TWIN_OK=0
  REF_NC_FILES=(); TWIN_NC_FILES=()
  GOT_FIDELITY=0

  for tag in "${PORDER[@]}"; do
    PREPCOUNT[$tag]=$(( ${PREPCOUNT[$tag]:-0} + 1 ))
    rep=${PREPCOUNT[$tag]}

    case "$tag" in
      P00_ref)  TEMPLATE_ID=P00; DECOMP_ID=$DECOMP; ROLE=reference ;;
      P00_twin) TEMPLATE_ID=P00; DECOMP_ID=$TWIN_DECOMP; ROLE=twin ;;
      *)        TEMPLATE_ID=$tag; DECOMP_ID=$DECOMP; ROLE=candidate ;;
    esac
    if [ "$TEMPLATE_ID" = "P00" ]; then COMPLIANCE_CLASS=STRICT; else COMPLIANCE_CLASS=PHYSICS-GRAY; fi

    ranks=${CFG_RANKS[$DECOMP_ID]}; omp=${CFG_OMP[$DECOMP_ID]}; nx=${CFG_NX[$DECOMP_ID]}; ny=${CFG_NY[$DECOMP_ID]}
    numtiles=${CFG_NUMTILES[$DECOMP_ID]}; dist=${CFG_DIST[$DECOMP_ID]}; blocktime=${CFG_BLOCKTIME[$DECOMP_ID]}
    ntasks_per_node=$(( ranks / 4 ))

    # 2>/dev/null + tail -n1: physics-describe prints exactly one real line (SKIP\treason
    # or the 13-field TSV); this defends the read against any stray noise line (e.g. a
    # pixi cache warning) landing before it.
    IFS=$'\t' read -r P_FAMILY P_EDITS P_PROBE P_TS P_MP P_CU P_BL P_SFCLAY P_SFSURF P_RLW P_RSW P_RADT P_CUDT \
      < <($PY "$CAMPAIGN/configs.py" physics-describe "$CASE" "$TEMPLATE_ID" 2>/dev/null | tail -n1)

    if [ "$ROLE" = "candidate" ] && [ "$rep" = "1" ] && [ "$P_FAMILY" != "SKIP" ]; then
      EXPECTED_CANDIDATE[$tag]=1
    fi

    if [ "$P_FAMILY" = "SKIP" ]; then
      REASON_TXT=$(echo "$P_EDITS" | tr ' \t' '__')
      echo "SKIP case=$CASE template=$TEMPLATE_ID tag=$tag rep=$rep reason=$P_EDITS"
      NOTES="mode=physics template=$TEMPLATE_ID family=- edits= decomp=$DECOMP_ID sim_minutes=$SIM_MINUTES hist_minutes=$HIST_MINUTES fidelity_nd_max= fidelity_floor_ratio_max= fidelity_deciding= tag=$tag role=$ROLE rep=$rep probe=none status=SKIPPED reason=$REASON_TXT"
      export RUN_ID="${CASE}_physics_${tag}_r${rep}_${J}" SRC_DATASET="physics_campaign" JOB_ID="$J" \
        REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
        HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="$COMPLIANCE_CLASS" \
        E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$DT" \
        MP_PHYSICS="$PHYS_MP_PHYSICS" CU_PHYSICS="$PHYS_CU_PHYSICS" BL_PBL_PHYSICS="$PHYS_BL_PBL_PHYSICS" \
        SF_SFCLAY_PHYSICS="$PHYS_SF_SFCLAY_PHYSICS" SF_SURFACE_PHYSICS="$PHYS_SF_SURFACE_PHYSICS" \
        RA_LW_PHYSICS="$PHYS_RA_LW_PHYSICS" RA_SW_PHYSICS="$PHYS_RA_SW_PHYSICS" \
        RADT="$PHYS_RADT" CUDT="$PHYS_CUDT" \
        HISTORY_INTERVAL="0" IO_FORM_HISTORY="14" \
        N_NODES="4" N_RANKS="$ranks" OMP_THREADS="$omp" NPROC_X="$nx" NPROC_Y="$ny" \
        NUMTILES="$numtiles" DISTRIBUTION="$dist" KMP_BLOCKTIME_COL="$blocktime" \
        S_PER_STEP_MEDIAN="" S_PER_STEP_MEAN="" N_STEPS_MEASURED="" \
        DIFFWRF_DIFFERING_FIELDS="" FIDELITY_LABEL="" NOTES="$NOTES" PHYSICS_SUITE_NOTE="$PHYSICS_SUITE"
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
    "", "", "", "", "", "", "",
    notes,
]
with open(ledger, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
      continue
    fi

    check_disk_guard "physics_${tag}_r${rep}"

    RUNDIR="$OUT/${CASE}_physics_${tag}_r${rep}_${J}"
    rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"

    for f in "${INPUT_FILES_ARR[@]}"; do
      ln -sf "$PREP_DIR/$f" "$f"
    done
    for ((d = 1; d <= MAX_DOM; d++)); do
      df=$(printf "d%02d" "$d")
      ln -sf "$BP_DIR/wrfinput_${df}.bp" "wrfinput_${df}.bp"
    done
    ln -sf "$BP_DIR/wrfbdy_d01.bp" "wrfbdy_d01.bp"
    link_thompson_tables
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
    if [ "$rep" = "1" ]; then
      HIST_ARGS=(--history-mode interval --history-minutes "$HIST_MINUTES")
    else
      HIST_ARGS=(--history-mode off)
    fi
    python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \
      --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
      --run-minutes "$SIM_MINUTES" --max-dom "$MAX_DOM" --ranks "$ranks" \
      "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}" "${HIST_ARGS[@]}" \
      --physics-template "$TEMPLATE_ID" --physics-case "$CASE"
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

    # harvest any freshly-computed Thompson table BEFORE the retention cleanup below
    # (which, on a failed run, purges everything except the 5 diagnostic files) so a
    # table generated by this run is never lost -- runs regardless of RC/OK.
    harvest_thompson_tables

    OK=0
    if [ -f rsl.error.0000 ] && grep -q "SUCCESS COMPLETE WRF" rsl.error.0000; then
      OK=1
    fi

    grep "Timing for main" rsl.out.0000 2>/dev/null | sed -nE 's/.*on domain +([0-9]+): +([0-9.]+) elapsed.*/\1 \2/p' > steps_by_domain.txt
    awk '$1 == 1 { print $2 }' steps_by_domain.txt > steps.txt

    if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
      STATS=$($PY "$CAMPAIGN/configs.py" step-stats steps.txt)
    else
      NALL_RAW=$(wc -l < steps.txt 2>/dev/null || echo 0)
      NALL=$(( NALL_RAW > 0 ? NALL_RAW - 1 : 0 ))
      NEXCL0=0; [ -s steps.txt ] && NEXCL0=1
      STATS="med= mean= n=0 excl=$NEXCL0 nall=$NALL"
    fi
    kv () { echo "$STATS" | awk -v k="$1" '{for (i = 1; i <= NF; i++) {split($i, a, "="); if (a[1] == k) print a[2]}}'; }
    MED=$(kv med); MEAN=$(kv mean); NMEAS=$(kv n); NEXCL=$(kv excl); NALL=$(kv nall)

    echo "RESULT case=$CASE mode=physics tag=$tag template=$TEMPLATE_ID role=$ROLE rep=$rep decomp=$DECOMP_ID rc=$RC ok=$OK med=$MED mean=$MEAN n=$NMEAS wall=$WALL"

    FID_LABEL=""; FID_ND=""; FID_FLOOR=""; FID_DECIDING=""
    STATUS_NOTE=""
    if [ "$rep" = "1" ]; then
      if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
        mapfile -t FRAMES < <(ls -d wrfout_d01_* 2>/dev/null | sort)
        NC_FILES=()
        CONV_OK=1
        if [ ${#FRAMES[@]} -eq 0 ]; then
          CONV_OK=0
        fi
        for bf in "${FRAMES[@]}"; do
          ncf="${bf}.nc"
          srun -N1 -n1 -c8 bash -c "cd $PIXI_ENV && pixi run python $CONV --input $RUNDIR/$bf --output $RUNDIR/$ncf" \
            > "convert_${bf}.log" 2>&1
          if [ $? -ne 0 ] || [ ! -s "$ncf" ]; then
            CONV_OK=0
            rm -f "$ncf"
            break
          fi
          NC_FILES+=("$RUNDIR/$ncf")
          # problem 1: BP history is never needed again once its netCDF is verified
          # (non-empty, checked above) -- delete it immediately rather than waiting for
          # the end-of-tag-loop sweep, so a job aborted partway through the tag loop
          # (disk guard, timeout on a later run, ...) never leaves it behind.
          rm -rf "$RUNDIR/$bf"
        done
        if [ "$CONV_OK" = "1" ]; then
          case "$ROLE" in
            reference)
              REF_NC_FILES=("${NC_FILES[@]}")
              REF_OK=1
              ;;
            twin)
              TWIN_NC_FILES=("${NC_FILES[@]}")
              TWIN_OK=1
              if [ "$REF_OK" = "1" ]; then
                FID_PREFIX="$RUNDIR/fidelity_TWIN_sanity"
                $PY "$FIDELITY_GATE" --reference "${REF_NC_FILES[@]}" --twin "${TWIN_NC_FILES[@]}" \
                  --candidate "${TWIN_NC_FILES[@]}" --family all --out-prefix "$FID_PREFIX" \
                  > "$RUNDIR/fidelity_gate.log" 2>&1
                if [ -f "${FID_PREFIX}.json" ]; then
                  # 2>/dev/null + tail -n1: the python one-liner prints exactly one real
                  # line; defends the read against any stray noise line preceding it.
                  # (the pipe must sit on the heredoc's command line: a line starting with
                  # '| tail' after PYEOF is a runtime syntax error inside $(...) that aborted
                  # the whole physics block in smoke jobs 1076238/1076240)
                  # FID_ND / FID_FLOOR are the MEASURED maxima over fields at the last common
                  # time, not the thresholds.
                  read -r FID_LABEL FID_ND FID_FLOOR FID_DECIDING < <($PY - "${FID_PREFIX}.json" 2>/dev/null <<'PYEOF' | tail -n1
import json, sys
d = json.load(open(sys.argv[1]))
summ = d.get("per_field_summary_at_last_time") or {}
def vmax(key):
    vals = [v.get(key) for v in summ.values() if isinstance(v, dict) and isinstance(v.get(key), (int, float))]
    return ("%.6g" % max(vals)) if vals else "NA"
deciding = str(d.get("deciding") or "NA").replace(" ", "_")
print(d.get("label") or "NA", vmax("nd"), vmax("floor_ratio"), deciding)
PYEOF
)
                  cp "${FID_PREFIX}.json" "${FID_PREFIX}.csv" "$FIDOUT/" 2>/dev/null
                  GOT_FIDELITY=$((GOT_FIDELITY + 1))
                else
                  STATUS_NOTE="status=FAILED reason=fidelity_gate_error_see_fidelity_gate.log"
                fi
              fi
              ;;
            candidate)
              if [ "$REF_OK" = "1" ] && [ "$TWIN_OK" = "1" ]; then
                FID_PREFIX="$RUNDIR/fidelity_${TEMPLATE_ID}"
                $PY "$FIDELITY_GATE" --reference "${REF_NC_FILES[@]}" --twin "${TWIN_NC_FILES[@]}" \
                  --candidate "${NC_FILES[@]}" --family "$P_FAMILY" --out-prefix "$FID_PREFIX" \
                  > "$RUNDIR/fidelity_gate.log" 2>&1
                if [ -f "${FID_PREFIX}.json" ]; then
                  # 2>/dev/null + tail -n1: the python one-liner prints exactly one real
                  # line; defends the read against any stray noise line preceding it.
                  # (the pipe must sit on the heredoc's command line: a line starting with
                  # '| tail' after PYEOF is a runtime syntax error inside $(...) that aborted
                  # the whole physics block in smoke jobs 1076238/1076240)
                  # FID_ND / FID_FLOOR are the MEASURED maxima over fields at the last common
                  # time, not the thresholds.
                  read -r FID_LABEL FID_ND FID_FLOOR FID_DECIDING < <($PY - "${FID_PREFIX}.json" 2>/dev/null <<'PYEOF' | tail -n1
import json, sys
d = json.load(open(sys.argv[1]))
summ = d.get("per_field_summary_at_last_time") or {}
def vmax(key):
    vals = [v.get(key) for v in summ.values() if isinstance(v, dict) and isinstance(v.get(key), (int, float))]
    return ("%.6g" % max(vals)) if vals else "NA"
deciding = str(d.get("deciding") or "NA").replace(" ", "_")
print(d.get("label") or "NA", vmax("nd"), vmax("floor_ratio"), deciding)
PYEOF
)
                  cp "${FID_PREFIX}.json" "${FID_PREFIX}.csv" "$FIDOUT/" 2>/dev/null
                  GOT_FIDELITY=$((GOT_FIDELITY + 1))
                else
                  STATUS_NOTE="status=FAILED reason=fidelity_gate_error_see_fidelity_gate.log"
                fi
              else
                STATUS_NOTE="status=FAILED reason=reference_or_twin_unavailable"
              fi
              # problem 1: a candidate's own netCDF is used exactly once (this
              # comparison, above) -- unlike reference/twin it is never needed again by
              # a later candidate, so delete it now regardless of outcome.
              rm -f "${NC_FILES[@]}"
              ;;
          esac
        else
          STATUS_NOTE="status=FAILED reason=bp_to_netcdf_conversion_failed"
        fi
      else
        STATUS_NOTE="status=FAILED reason=run_failed_rc=${RC}_ok=${OK}"
      fi
    fi

    NOTES="mode=physics template=$TEMPLATE_ID family=$P_FAMILY edits=$P_EDITS decomp=$DECOMP_ID sim_minutes=$SIM_MINUTES hist_minutes=$HIST_MINUTES fidelity_nd_max=$FID_ND fidelity_floor_ratio_max=$FID_FLOOR fidelity_deciding=$FID_DECIDING tag=$tag role=$ROLE rep=$rep n_excluded_steps=$NEXCL mean_includes_radiation=1 n_all_steps=$NALL io=adios2 $P_PROBE"
    [ -n "$STATUS_NOTE" ] && NOTES="$NOTES $STATUS_NOTE"
    [ "$RC" != "0" ] && NOTES="$NOTES rc=$RC"
    [ "$OK" != "1" ] && NOTES="$NOTES ok=$OK"

    export RUN_ID="${CASE}_physics_${tag}_r${rep}_${J}" SRC_DATASET="physics_campaign" JOB_ID="$J" \
      REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
      HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="$COMPLIANCE_CLASS" \
      E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$P_TS" \
      MP_PHYSICS="$P_MP" CU_PHYSICS="$P_CU" BL_PBL_PHYSICS="$P_BL" \
      SF_SFCLAY_PHYSICS="$P_SFCLAY" SF_SURFACE_PHYSICS="$P_SFSURF" \
      RA_LW_PHYSICS="$P_RLW" RA_SW_PHYSICS="$P_RSW" \
      RADT="$P_RADT" CUDT="$P_CUDT" \
      HISTORY_INTERVAL="$([ "$rep" = "1" ] && echo "$HIST_MINUTES" || echo 0)" IO_FORM_HISTORY="14" \
      N_NODES="4" N_RANKS="$ranks" OMP_THREADS="$omp" NPROC_X="$nx" NPROC_Y="$ny" \
      NUMTILES="$numtiles" DISTRIBUTION="$dist" KMP_BLOCKTIME_COL="$blocktime" \
      S_PER_STEP_MEDIAN="$MED" S_PER_STEP_MEAN="$MEAN" N_STEPS_MEASURED="$NMEAS" \
      DIFFWRF_DIFFERING_FIELDS="" FIDELITY_LABEL="$FID_LABEL" NOTES="$NOTES" \
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
    "", "", "", "", "", "", "",
    notes,
]
with open(ledger, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY

    # problem 1 retention rule: a successful run's rsl.* is trimmed to the two kept
    # files and wrf.out is redundant with them, so drop it (its own BP history/netCDF
    # were already handled immediately above -- converted+deleted, or, for a
    # candidate, deleted right after its own fidelity computation; reference/twin's
    # netCDF persists until the tag loop ends, see below). A FAILED run (rc!=0,
    # timeout, or no SUCCESS in rsl.error.0000) keeps only the minimum needed to
    # diagnose it -- smoke jobs 1076238-1076340 kept EVERYTHING (BP history + netCDF
    # too) for a failed run whenever any expected fidelity JSON was missing anywhere
    # in the job, and filled home1's soft quota.
    if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
      find . -maxdepth 1 -name 'rsl.*' ! -name 'rsl.out.0000' ! -name 'rsl.error.0000' -delete
      rm -f wrf.out
    else
      find . -maxdepth 1 -mindepth 1 \
        ! -name 'rsl.error.0000' ! -name 'rsl.out.0000' ! -name 'wrf.out' \
        ! -name 'namelist.input' ! -name 'namelist.output' \
        -exec rm -rf {} +
    fi

    cd "$OUT"
  done

  # problem 1: reference's and twin's own netCDF are consulted by EVERY candidate's
  # fidelity_gate call (see the candidate arm above), so they cannot be deleted until
  # the job's last fidelity computation -- that is here, unconditionally, whether or
  # not every expected fidelity JSON got computed (the old conditional kept everything
  # on any miss/failure, which is exactly what filled home1's soft quota). A defensive
  # sweep also catches any residual wrfout_d0* BP dir (there should be none left: every
  # successful conversion deletes its own BP immediately above, and a failed run's BP
  # is purged by the retention block above too).
  EXPECTED_FIDELITY=$(( 1 + ${#EXPECTED_CANDIDATE[@]} ))
  rm -f "${REF_NC_FILES[@]}" "${TWIN_NC_FILES[@]}"
  for d in "$OUT/${CASE}_physics_"*"_${J}"; do
    [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -name 'wrfout_d0*' -exec rm -rf {} +
  done
  echo "RETENTION: reference/twin netCDF and any residual BP history cleaned up for case=$CASE job=$J -- ref_ok=$REF_OK twin_ok=$TWIN_OK got_fidelity=$GOT_FIDELITY expected_fidelity=$EXPECTED_FIDELITY (fidelity JSON/CSV kept in $FIDOUT; see per-run notes in $LEDGER for anything missing/failed)"

  echo "=== done (MODE=physics)"
  exit 0
fi
# A runtime error inside the physics block (e.g. a syntax error in a command substitution)
# aborts that compound command and resumes HERE; never fall through into the STRICT loop
# (smoke jobs 1076238/1076240 did, and ran C01-C16 with an empty --history-mode until timeout).
if [ "$MODE" = "physics" ]; then
  echo "MODE=physics block aborted before completion -- refusing to fall through to the STRICT loop" >&2
  exit 1
fi

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

  check_disk_guard "${cid}_r${rep}"

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
  link_thompson_tables
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
  # Tile geometry (WTune D1 decomposition-space work): only meaningful, and only ever set, for a
  # CONFIG_FILE-generated config -- a plain STRICT template never has these CFG_* entries
  # populated (the CONFIG_FILE-unset branch above never fills them), so TILE_ARGS stays empty and
  # this edit-namelist call is byte-identical to before CONFIG_FILE existed.
  TILE_ARGS=()
  if [ -n "${CONFIG_FILE:-}" ]; then
    TILE_ARGS=(
      --numtiles-x "${CFG_NTILES_X[$cid]:-0}" --numtiles-y "${CFG_NTILES_Y[$cid]:-0}"
      --tile-sz-x "${CFG_TSZ_X[$cid]:-0}" --tile-sz-y "${CFG_TSZ_Y[$cid]:-0}"
      --tile-strategy "${CFG_TSTRAT[$cid]:-0}"
    )
  fi
  python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \
    --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
    --run-minutes "$MIN" --max-dom "$MAX_DOM" \
    --history-mode "$HISTORY_MODE_ARG" --ranks "$ranks" \
    "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}" "${TILE_ARGS[@]}"
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

  # harvest any freshly-computed Thompson table before the cleanup at the bottom of
  # this loop touches the run dir -- runs regardless of RC/OK.
  harvest_thompson_tables

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
  # WTune D1 decomposition-space work: a CONFIG_FILE-generated run's tile geometry has no ledger
  # column of its own (schema unchanged), so it goes into notes instead. Absent for a plain
  # STRICT-template run (CONFIG_FILE unset).
  if [ -n "${CONFIG_FILE:-}" ]; then
    NOTES="$NOTES config_source=${CFG_SRC[$cid]:-generated} numtiles_x=${CFG_NTILES_X[$cid]:-} numtiles_y=${CFG_NTILES_Y[$cid]:-} tile_sz_x=${CFG_TSZ_X[$cid]:-} tile_sz_y=${CFG_TSZ_Y[$cid]:-} tile_strategy=${CFG_TSTRAT[$cid]:-}"
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

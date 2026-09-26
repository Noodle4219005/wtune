#!/usr/bin/env bash
#SBATCH --job-name=strictcamp
#SBATCH --account=ACD110018
#SBATCH --partition=cf448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=02:00:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/campaign/strictcamp-%x-%j.out
#SBATCH --signal=B:USR1@90
#
# --signal=B:USR1@90 (job 740 postmortem, 2026-09): Slurm delivers SIGUSR1 to the batch shell
# (the B: prefix, not to every rank) 90s before --time expires, while the allocation is still
# fully alive -- the USR1 trap below uses that window to kill the in-flight run and reap every
# node BEFORE Slurm's own SIGTERM/SIGKILL (KillWait=30s) arrives. Neither tools/scc_sweep.py's
# nor tools/temporal_validation.py's build_*_sbatch_command() pass --signal on the sbatch
# command line (verified 2026-09: grep shows only -p/-N/--ntasks-per-node/--exclusive/
# --exclude/--time/--job-name/--output/--export), so this in-script #SBATCH directive is
# never overridden by either caller.
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
#        exactly as it did before CONFIG_FILE existed.
#        MODE=physics honors CONFIG_FILE too (2026-09-17), with a different meaning: it does
#        NOT change the run set (that stays P00_ref + P00_twin + the requested templates) --
#        it is where DECOMP and TWIN_DECOMP are looked up. This is the only way to run
#        PHYSICS-GRAY on a case that resolves no STRICT template at all: sc26p1's d02 has 100
#        cells and WRF needs floor(e/nproc) >= 10 on every active domain, capping BOTH domains
#        at 10x10 = 100 ranks, while C01-C16 are anchored at 448/224/112 -- `configs.py sc26p1`
#        prints 16 SKIP lines and an empty table. Example:
#          sbatch -p cf112 -N1 --export=ALL,CASE=sc26p1,MODE=physics,\
#            CONFIG_FILE=/abs/pair.json,DECOMP=G0001,TWIN_DECOMP=G0002,PCONFIGS=D07,REPS=2 \
#            campaign/strict_campaign.sbatch
#        Both ids must exist in the file and must be DIFFERENT decompositions (the twin is the
#        chaos floor, not a replicate); configs.py decomp-pair enforces both, see below.
set -uo pipefail

SITE=${SITE:-f1}
export SITE
# CAMPAIGN locates this checkout's campaign/ dir, so its own default must vary by site (the repo
# lives at a different absolute path on each cluster) -- everything else site-specific is inside
# campaign/sites/$SITE.sh, sourced below, which f1.sh reproduces byte-for-byte as it stood before
# SITE existed. export: CASES_JSON's default path (below) and the CASES_JSON overlay merge both
# need CAMPAIGN from a plain python3 subprocess, not just from this shell.
if [ "$SITE" = "scc" ]; then
  CAMPAIGN=${CAMPAIGN:-$HOME/wtune/campaign}
else
  CAMPAIGN=${CAMPAIGN:-/home/u5727520/wrf-stack/wtune-autotuner/campaign}
fi
export CAMPAIGN
# campaign/sites/$SITE.sh defines W, OUT, HARDWARE_ID, PIXI_ENV, CONV, CASES_JSON_DEFAULT,
# CORES_PER_NODE and the functions site_env() (module load / MPI+IO env, called once below) and
# site_launch() (the per-run srun/mpirun invocation, called from both loops below).
source "$CAMPAIGN/sites/$SITE.sh"
if [ "${MODE:-timing}" != "timing" ] && [ ! -f "$CONV" ]; then
  echo "FATAL: ADIOS2->netCDF converter not found at $CONV (needed by MODE=${MODE}) -- refusing to run" >&2
  exit 1
fi

REPS=${REPS:-2}
TIMEOUT_S=${TIMEOUT_S:-600}
# nodes in this allocation (srun --ntasks-per-node and the ledger n_nodes column); the #SBATCH
# header's 4-node shape gives 4 exactly as before, and a -N1 cf112 run (small cases) gives 1
ALLOC_NODES=${SLURM_JOB_NUM_NODES:-4}
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
# PYBIN (2026-09-26, job 1218 postmortem): pixi's own startup chatter (e.g. its "cache is on a
# network/parallel filesystem" cold-cache warning) prints to STDOUT, not stderr, on the FIRST
# `pixi run` invocation of a job -- `2>/dev/null` on a $PY call does NOT catch it, and it
# silently corrupts any caller that parses that stdout as data (job 1218: the CONFIG_FILE TSV
# read loop below got a stray line spliced in and misread a tile-geometry column, rc=127).
# Resolving the underlying interpreter's own absolute path ONCE, here, and invoking THAT
# directly (not through `pixi run`) for every call whose stdout is parsed as data means pixi's
# own wrapper -- and whatever it prints before exec'ing the interpreter -- never runs again for
# those calls, so its chatter can never reach a data stream. `tail -1`: this resolution call is
# ITSELF pixi's first invocation and could be polluted the same way; the real sys.executable
# line is always LAST (pixi's own chatter, if any, prints before the child process runs).
PYBIN=$($PY -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1)
if [ -z "$PYBIN" ] || [ ! -x "$PYBIN" ]; then
  echo "FATAL: could not resolve a usable python interpreter via \$PY ('$PY') -- got PYBIN='$PYBIN'" >&2
  exit 1
fi
: "${CASE:?CASE env var is required, e.g. --export=ALL,CASE=karina}"

mkdir -p "$OUT"

site_env

J=$SLURM_JOB_ID
LEDGER="$OUT/ledger_${CASE}_${J}.csv"
LEDGER_HEADER="run_id,source_dataset,job_id,rep_index,case_raw,case_base,case_family,hardware_id,binary_kind,compliance_class,e_we,e_sn,e_vert,dx,dy,time_step,mp_physics,cu_physics,bl_pbl_physics,sf_sfclay_physics,sf_surface_physics,ra_lw_physics,ra_sw_physics,radt,cudt,history_interval,io_form_history,n_nodes,n_ranks,omp_threads,nproc_x,nproc_y,numtiles,distribution,kmp_blocktime,s_per_step_median,s_per_step_mean,n_steps_measured,diffwrf_differing_fields,fidelity_label,profile_micro_share,profile_radiation_share,profile_dynamics_share,profile_mpi_share,profile_io_share,profile_rank_imbalance,profile_thread_imbalance,notes"
[ -f "$LEDGER" ] || echo "$LEDGER_HEADER" > "$LEDGER"

# ---- node hygiene: orphan wrf.exe reap + dirty-node census (scc prte/pgid incident,
# 2026-09) ------------------------------------------------------------------------------
# scc's Slurm has ProctrackType=pgid and no TaskPlugin; Open MPI 5's prte launcher puts the
# ranks it spawns into their own process group, so a TIMEOUT'd job -- or one whose per-run
# watchdog (TIMEOUT_S below) kills mpirun -- leaves those ranks running as invisible orphans.
# A node census found orphaned wrf.exe owned by our user on 8/12 compute nodes, 8-20h old,
# each spinning a full core on CPU 64; every later benchmark on those nodes was contaminated
# (pure-MPI ~35% slower, hybrid MPI+OpenMP 20-130x slower, from OpenMP barriers waiting on a
# scheduler time slice shared with the spinner). f1's site_hygiene (campaign/sites/f1.sh) is a
# no-op: its site_launch uses srun, not mpirun/prte, so it has no equivalent failure mode.
# See campaign/node_hygiene.sh for the per-node reap/census logic dispatched here.
NODE_DIRTY_NOTE="node_dirty=none"

run_node_hygiene () {
  # run_node_hygiene <mode> <timeout_s>   mode: reap | census | reap+census
  # Sets REAP_N (total orphans reaped across every node, this call only) and, when a census
  # was requested and found a dirty node, NODE_DIRTY_NOTE (sticky -- the caller records the
  # preflight's one measurement in every ledger row this job writes, not a per-run
  # re-measurement).
  local mode="$1" tmo="$2" out
  site_hygiene "$mode"
  out=$(timeout "$tmo" "${SITE_HYGIENE_CMD[@]}" 2>&1) || true
  [ -n "$out" ] && echo "$out"
  REAP_N=$(printf '%s\n' "$out" | awk '/^REAPED_ORPHANS /{for(i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="n") s+=kv[2]}} END{print s+0}')
  if printf '%s\n' "$out" | grep -q '^NODE_DIRTY '; then
    NODE_DIRTY_NOTE="node_dirty=$(printf '%s\n' "$out" | awk '/^NODE_DIRTY /{h=c=b="";for(i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="host")h=kv[2]; else if(kv[1]=="cpu")c=kv[2]; else if(kv[1]=="busy")b=kv[2]} printf "%s:cpu%s:%s,",h,c,b}' | sed 's/,$//')"
  fi
}

# job 740 postmortem (2026-09): a plain foreground `timeout ... "${SITE_LAUNCH_CMD[@]}"` blocks
# bash in a way that DEFERS every trap until that foreground child exits -- Slurm's TIMEOUT
# SIGTERM landed, mpirun tried (and failed) to tear down ranks it had already lost track of,
# and SIGKILL arrived 30s later (KillWait) having never let our trap run at all: the job log
# shows zero REAPED_ORPHANS/trap output, and a census of work3 afterwards found all 72 ranks
# still alive, pinned at 100%. Fix: launch the run in the BACKGROUND and `wait` on it
# explicitly -- a trapped signal interrupts a blocked `wait` immediately (verified locally: the
# trap runs before the next script line, even mid-`wait`; see this task's report for the
# transcript), so USR1/TERM/INT now fire in time. `#SBATCH --signal=B:USR1@90` above adds an
# early-warning signal 90s before the hard time limit, while the allocation is still alive, so
# there is time to kill the run and reap every node BEFORE Slurm's own KillWait=30s TERM/KILL.
WRF_PID=""
CUR_RUN_ID=""

stop_run_and_reap () {
  # stop_run_and_reap <kill_wait_s> <reap_tmo_s>: if a run is in flight (WRF_PID alive), TERM
  # it, poll up to <kill_wait_s> for it to exit, KILL if it hasn't, then reap every node. Safe
  # to call with nothing in flight (WRF_PID unset/dead): the kill step is then a silent no-op
  # and only the reap runs, exactly like the old TERM/INT/EXIT-only reap did.
  local kill_wait="$1" reap_tmo="$2" i
  if [ -n "$WRF_PID" ] && kill -0 "$WRF_PID" 2>/dev/null; then
    kill -TERM "$WRF_PID" 2>/dev/null || true
    for ((i = 0; i < kill_wait; i++)); do
      kill -0 "$WRF_PID" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$WRF_PID" 2>/dev/null && { kill -KILL "$WRF_PID" 2>/dev/null || true; }
  fi
  run_node_hygiene reap "$reap_tmo"
}

write_interrupted_row () {
  # write_interrupted_row <sig>: record the in-flight config's ledger row as INTERRUPTED.
  # CUR_RUN_ID is set only for the exact window between launching a run in the background and
  # capturing its RC (both loops below); a signal landing outside a run (editing a namelist,
  # blocked on check_disk_guard, etc.) has nothing in flight and this is a no-op. Same
  # minimal-row shape as check_disk_guard's own abort row above.
  local sig="$1"
  [ -z "$CUR_RUN_ID" ] && return 0
  RUN_ID="$CUR_RUN_ID" JOB_ID="$J" CASE_RAW="$CASE" CASE_BASE="$CASE" \
    HARDWARE_ID="$HARDWARE_ID" \
    NOTES_ABORT="status=INTERRUPTED reason=preempt_sig${sig} $NODE_DIRTY_NOTE" \
    python3 - "$LEDGER" <<'PY'
import csv, os, sys
ledger = sys.argv[1]
e = os.environ.get
row = [e("RUN_ID", ""), "interrupted", e("JOB_ID", ""), "", e("CASE_RAW", ""), e("CASE_BASE", ""),
       "", e("HARDWARE_ID", ""), "", ""] + [""] * 37 + [e("NOTES_ABORT", "")]
with open(ledger, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
  echo "WROTE_INTERRUPTED_ROW run_id=$CUR_RUN_ID sig=$sig" >&2
  CUR_RUN_ID=""
}

# NODE_HYGIENE_TRAP_DONE: USR1's own `exit 0` below re-enters this same function via the EXIT
# trap -- the guard makes that second entry a fast no-op instead of a second kill/reap/write
# (verified locally: EXIT fires after USR1's exit, sees the guard, returns immediately).
NODE_HYGIENE_TRAP_DONE=0
node_hygiene_trap () {
  local sig="$1"
  [ "$NODE_HYGIENE_TRAP_DONE" = "1" ] && return 0
  NODE_HYGIENE_TRAP_DONE=1
  if [ "$sig" = "USR1" ]; then
    # T-90s, allocation still alive: generous budget (well inside the 90s window).
    stop_run_and_reap 10 30 >&2 || true
  else
    # real Slurm TERM/INT, or plain script exit: KillWait=30s total, so a much tighter budget.
    stop_run_and_reap 3 15 >&2 || true
  fi
  write_interrupted_row "$sig" >&2 || true
  [ "$sig" != "EXIT" ] && exit 0
}
trap 'node_hygiene_trap EXIT' EXIT
trap 'node_hygiene_trap TERM' TERM
trap 'node_hygiene_trap INT' INT
trap 'node_hygiene_trap USR1' USR1

# preflight (design point 1): the allocation is held --exclusive and no WRF launch has
# happened yet, so any wrf.exe found here is an orphan from a dead job -- reap is safe.
# One measurement for the whole job; every ledger row below carries this same NODE_DIRTY_NOTE.
run_node_hygiene reap+census 90

# ---- disk-quota guard (problem 1, 2026-09-15) -----------------------------------------
# Smoke jobs 1076238-1076340 kept every failed/incomplete MODE=physics run's BP history
# and converted netCDF whenever any expected fidelity JSON was missing, filling home1's
# soft quota (grace expired); later runs then failed mid-write with 'Disk quota exceeded'.
# check_disk_guard is called right before every WRF srun (both loops below) so a job
# stops itself with a clear message and a ledger row instead of running blind into that.
MIN_FREE_GIB=${MIN_FREE_GIB:-60}
# Inputs: ADIOS2 .bp from bp_dir (f1 default), or -- when a site's case overlay leaves bp_dir
# empty (scc has only the official netCDF files) -- netCDF wrfinput/wrfbdy from prep_dir, with
# WTUNE_INPUT_FORMAT=netcdf telling configs.py to write io_form_input/boundary=2.
link_wrf_inputs() {
  local d df
  for ((d = 1; d <= MAX_DOM; d++)); do
    df=$(printf "d%02d" "$d")
    if [ -n "$BP_DIR" ]; then ln -sf "$BP_DIR/wrfinput_${df}.bp" "wrfinput_${df}.bp"
    else ln -sf "$PREP_DIR/wrfinput_${df}" "wrfinput_${df}"; fi
  done
  if [ -n "$BP_DIR" ]; then ln -sf "$BP_DIR/wrfbdy_d01.bp" "wrfbdy_d01.bp"
  else ln -sf "$PREP_DIR/wrfbdy_d01" "wrfbdy_d01"; fi
}

check_disk_guard () {
  local ctx="$1" q_text free_gib
  if ! q_text=$(mmlsquota -u "$USER" home1 2>&1); then
    echo "WARNING: mmlsquota unavailable ($q_text) -- skipping disk guard for $ctx" >&2
    return 0
  fi
  free_gib=$(printf '%s\n' "$q_text" | "$PYBIN" "$CAMPAIGN/configs.py" quota-free-gib)
  if [ -z "$free_gib" ] || [ "$free_gib" = "NA" ]; then
    echo "WARNING: could not parse mmlsquota output -- skipping disk guard for $ctx" >&2
    return 0
  fi
  if awk -v f="$free_gib" -v m="$MIN_FREE_GIB" 'BEGIN { exit !(f < m) }'; then
    echo "ABORT: home1 free space ${free_gib} GiB < MIN_FREE_GIB=${MIN_FREE_GIB} GiB before $ctx -- stopping job (case=$CASE job=$J) to avoid a mid-run Disk quota exceeded failure" >&2
    RUN_ID="${CASE}_${ctx}_${J}" JOB_ID="$J" CASE_RAW="$CASE" CASE_BASE="$CASE" \
      HARDWARE_ID="$HARDWARE_ID" NOTES_ABORT="status=ABORTED reason=low_disk free_gib=${free_gib} min_free_gib=${MIN_FREE_GIB} context=${ctx} $NODE_DIRTY_NOTE" \
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

# ---- per-run OpenMP/threading env (site-conditional) ----------------------------------
# f1's KMP_BLOCKTIME knob has no equivalent under scc's nvhpc OpenMP runtime (SITE FACTS,
# decision 0050 step 8) -- never export it there, and record the ledger's kmp_blocktime column
# as "n/a" instead of the numeric/"default" value configs.py generated. scc uses OMP_PLACES/
# OMP_PROC_BIND for in-rank thread pinning instead (only meaningful when omp>1). Sets
# OMP_NUM_THREADS and BLOCKTIME_RECORD (the ledger column value) on both sites; call once per
# run, before site_launch, with the same (omp, blocktime) already resolved for this config.
configure_omp_env () {
  local omp="$1" blocktime="$2"
  export OMP_NUM_THREADS="$omp"
  if [ "$SITE" = "scc" ]; then
    unset KMP_BLOCKTIME
    if [ "$omp" -gt 1 ]; then
      export OMP_PLACES=cores OMP_PROC_BIND=close
    else
      unset OMP_PLACES OMP_PROC_BIND
    fi
    BLOCKTIME_RECORD="n/a"
  else
    if [ "$blocktime" = "default" ]; then
      unset KMP_BLOCKTIME
    else
      export KMP_BLOCKTIME=$blocktime
    fi
    BLOCKTIME_RECORD="$blocktime"
  fi
}

echo "job $J case $CASE nodes $SLURM_JOB_NODELIST reps $REPS timeout $TIMEOUT_S"

# ---- case metadata from cases.json --------------------------------------------
eval "$(python3 - "$CASE" <<'PY'
import json, os, sys
case_id = sys.argv[1]
campaign_dir = os.environ.get("CAMPAIGN") or "/home/u5727520/wrf-stack/wtune-autotuner/campaign"
site = os.environ.get("SITE") or "f1"
# CASES_JSON (optional): alternate case file, same schema; configs.py honours the same env var
d = json.load(open(os.environ.get("CASES_JSON") or os.path.join(campaign_dir, "cases.json")))["cases"]
if site != "f1":
    # campaign/cases.<site>.json (decision 0050 step 8): {"cases": {id: {overriding keys}}},
    # e.g. prep_dir/bp_dir -- merged over cases.json for THIS case only, same rule as
    # campaign/configs.py's own load_cases() overlay. A str value may use $VAR/~ (expanded here,
    # since a site's real $HOME is not known at authoring time); unset SITE (f1) never applies
    # this at all, so f1's case metadata is byte-identical to before SITE existed.
    overlay_path = os.path.join(campaign_dir, "cases.%s.json" % site)
    if os.path.exists(overlay_path):
        over = json.load(open(overlay_path))["cases"].get(case_id)
        if over:
            merged = dict(d.get(case_id, {}))
            for k, v in over.items():
                if isinstance(v, str):
                    v = os.path.expanduser(os.path.expandvars(v))
                merged[k] = v
            d = dict(d)
            d[case_id] = merged
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

# empty bp_dir (site overlay, e.g. scc) => netCDF inputs; see link_wrf_inputs
if [ -z "$BP_DIR" ]; then export WTUNE_INPUT_FORMAT=netcdf; else export WTUNE_INPUT_FORMAT=adios2; fi

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
  # Tab is IFS whitespace, so `IFS=$'\t' read` collapses consecutive tabs and an EMPTY field
  # (e.g. patch_x/patch_y of a generated config) shifts every later column left -- job 1225 got
  # tile_sz_y='anchor'. Translate tabs to \037 (not whitespace) so empty fields survive.
  while IFS=$'\037' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy \
                          ntx nty tsx tsy tstrat src; do
    [ "$cid" = "config_id" ] && continue
    [ -z "$cid" ] && continue
    CFG_RANKS[$cid]=$ranks; CFG_OMP[$cid]=$omp; CFG_NX[$cid]=$nx; CFG_NY[$cid]=$ny
    CFG_NUMTILES[$cid]=$numtiles; CFG_DIST[$cid]=$dist; CFG_BLOCKTIME[$cid]=$blocktime
    CFG_NTILES_X[$cid]=$ntx; CFG_NTILES_Y[$cid]=$nty; CFG_TSZ_X[$cid]=$tsx; CFG_TSZ_Y[$cid]=$tsy
    CFG_TSTRAT[$cid]=$tstrat; CFG_SRC[$cid]=$src
  done < <("$PYBIN" "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null | tr '\t' '\037')
else
  while IFS=$'\t' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy; do
    [ "$cid" = "config_id" ] && continue
    [ -z "$cid" ] && continue
    CFG_RANKS[$cid]=$ranks; CFG_OMP[$cid]=$omp; CFG_NX[$cid]=$nx; CFG_NY[$cid]=$ny
    CFG_NUMTILES[$cid]=$numtiles; CFG_DIST[$cid]=$dist; CFG_BLOCKTIME[$cid]=$blocktime
  done < <("$PYBIN" "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null)
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
  FIDELITY_GATE="$(dirname "$CAMPAIGN")/src/fidelity_gate.py"
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
  # Single rule set for the reference/twin pair, whether the ids are STRICT templates or come
  # from CONFIG_FILE (configs.resolve_decomp_pair, unit-tested in campaign/test_configs.py
  # TestResolveDecompPair / TestDecompPairCLI). It re-checks resolvability with a message that
  # names the available ids -- the two guards above cannot, because their CFG_* tables are built
  # with stderr discarded and look identical whether the id was wrong or configs.py itself failed
  # -- and it additionally refuses TWIN_DECOMP == DECOMP, which would measure a ~0 chaos floor
  # and relabel every candidate LARGE-DRIFT. Fail-fast: this runs before the first WRF srun.
  if ! $PY "$CAMPAIGN/configs.py" decomp-pair "$CASE" "$DECOMP" "$TWIN_DECOMP"; then
    echo "ERROR: DECOMP=$DECOMP / TWIN_DECOMP=$TWIN_DECOMP rejected for case $CASE (reason above; CONFIG_FILE=${CONFIG_FILE:-unset})" >&2
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
  mapfile -t PORDER < <("$PYBIN" "$CAMPAIGN/configs.py" physics-order "$CASE" "${PCONFIGS_ARGS[@]}" --reps "$REPS" \
    2>/dev/null | grep -E '^(P00_ref|P00_twin|[PD][0-9]+)$')  # [PD]: D01+ are the &dynamics/
    # diagnostics/cadence templates (2026-09-16); a P-only filter silently dropped every one of them
    # and left an empty candidate list (job 1080256)
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
    ntasks_per_node=$(( ranks / ALLOC_NODES ))
    # ledger kmp_blocktime column: "n/a" on scc (no such knob under nvhpc OpenMP, see
    # configure_omp_env below), else the template/generated value verbatim, unchanged from before
    # SITE existed. Computed here too (not only in configure_omp_env) so a SKIP row below -- which
    # never reaches configure_omp_env, since no run happens -- still records the right value.
    if [ "$SITE" = "scc" ]; then BLOCKTIME_RECORD="n/a"; else BLOCKTIME_RECORD="$blocktime"; fi

    # 2>/dev/null + tail -n1: physics-describe prints exactly one real line (SKIP\treason
    # or the 13-field TSV); this defends the read against any stray noise line (e.g. a
    # pixi cache warning) landing before it.
    IFS=$'\t' read -r P_FAMILY P_EDITS P_PROBE P_TS P_MP P_CU P_BL P_SFCLAY P_SFSURF P_RLW P_RSW P_RADT P_CUDT \
      < <("$PYBIN" "$CAMPAIGN/configs.py" physics-describe "$CASE" "$TEMPLATE_ID" 2>/dev/null | tail -n1)
    # physics-describe writes "-" for empty fields (tab IFS collapses empty fields); map back
    [ "$P_FAMILY" != "SKIP" ] && for _v in P_EDITS P_PROBE P_TS P_MP P_CU P_BL P_SFCLAY P_SFSURF P_RLW P_RSW P_RADT P_CUDT; do
      [ "${!_v}" = "-" ] && printf -v "$_v" '%s' ""
    done

    if [ "$ROLE" = "candidate" ] && [ "$rep" = "1" ] && [ "$P_FAMILY" != "SKIP" ]; then
      EXPECTED_CANDIDATE[$tag]=1
    fi

    if [ "$P_FAMILY" = "SKIP" ]; then
      REASON_TXT=$(echo "$P_EDITS" | tr ' \t' '__')
      echo "SKIP case=$CASE template=$TEMPLATE_ID tag=$tag rep=$rep reason=$P_EDITS"
      NOTES="mode=physics template=$TEMPLATE_ID family=- edits= decomp=$DECOMP_ID sim_minutes=$SIM_MINUTES hist_minutes=$HIST_MINUTES fidelity_nd_max= fidelity_floor_ratio_max= fidelity_deciding= tag=$tag role=$ROLE rep=$rep probe=none status=SKIPPED reason=$REASON_TXT $NODE_DIRTY_NOTE"
      export RUN_ID="${CASE}_physics_${tag}_r${rep}_${J}" SRC_DATASET="physics_campaign" JOB_ID="$J" \
        REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
        HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="$COMPLIANCE_CLASS" \
        E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$DT" \
        MP_PHYSICS="$PHYS_MP_PHYSICS" CU_PHYSICS="$PHYS_CU_PHYSICS" BL_PBL_PHYSICS="$PHYS_BL_PBL_PHYSICS" \
        SF_SFCLAY_PHYSICS="$PHYS_SF_SFCLAY_PHYSICS" SF_SURFACE_PHYSICS="$PHYS_SF_SURFACE_PHYSICS" \
        RA_LW_PHYSICS="$PHYS_RA_LW_PHYSICS" RA_SW_PHYSICS="$PHYS_RA_SW_PHYSICS" \
        RADT="$PHYS_RADT" CUDT="$PHYS_CUDT" \
        HISTORY_INTERVAL="0" IO_FORM_HISTORY="14" \
        N_NODES="$ALLOC_NODES" N_RANKS="$ranks" OMP_THREADS="$omp" NPROC_X="$nx" NPROC_Y="$ny" \
        NUMTILES="$numtiles" DISTRIBUTION="$dist" KMP_BLOCKTIME_COL="$BLOCKTIME_RECORD" \
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
    link_wrf_inputs
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
    # Tile geometry, same rule as the STRICT loop below: only a CONFIG_FILE-generated
    # decomposition ever has these CFG_* entries populated, so with CONFIG_FILE unset TILE_ARGS
    # stays empty and this edit-namelist call is byte-identical to before. Before 2026-09-17 the
    # physics loop omitted them unconditionally, which silently dropped the tile geometry of any
    # generated decomposition it was handed.
    TILE_ARGS=()
    if [ -n "${CONFIG_FILE:-}" ]; then
      TILE_ARGS=(
        --numtiles-x "${CFG_NTILES_X[$DECOMP_ID]:-0}" --numtiles-y "${CFG_NTILES_Y[$DECOMP_ID]:-0}"
        --tile-sz-x "${CFG_TSZ_X[$DECOMP_ID]:-0}" --tile-sz-y "${CFG_TSZ_Y[$DECOMP_ID]:-0}"
        --tile-strategy "${CFG_TSTRAT[$DECOMP_ID]:-0}"
      )
    fi
    python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \
      --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
      --run-minutes "$SIM_MINUTES" --max-dom "$MAX_DOM" --ranks "$ranks" \
      "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}" "${HIST_ARGS[@]}" "${TILE_ARGS[@]}" \
      --physics-template "$TEMPLATE_ID" --physics-case "$CASE"
    rm -f ./namelist.input.orig
    ln -sf "$W/main/wrf.exe" ./wrf.exe

    configure_omp_env "$omp" "$blocktime"

    T0=$(date +%s.%N)
    site_launch "$ranks" "$ntasks_per_node" "$omp" "$dist"
    # background + explicit wait, not a foreground pipeline -- see the node-hygiene block's
    # own comment above (job 740) for why. CUR_RUN_ID marks exactly the window a USR1/TERM/INT
    # trap should record as interrupted if it lands here.
    CUR_RUN_ID="${CASE}_physics_${tag}_r${rep}_${J}"
    timeout "$TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 &
    WRF_PID=$!
    wait "$WRF_PID"
    RC=$?
    # wait returns 128+signum instead of the launched program's own exit status when a trapped
    # signal interrupts it; node_hygiene_trap already wrote this run's ledger row and exited
    # before this line in that case, so RC>128 here can only be a narrow race (signal landing
    # after wait returned but before CUR_RUN_ID is cleared below) -- keep OK=0/FAILED, never
    # treat it as success.
    WRF_PID=""
    CUR_RUN_ID=""
    T1=$(date +%s.%N)
    WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

    # post-run reap (design point 2): mpirun returned (normal exit or TIMEOUT_S watchdog) --
    # reap any wrf.exe it left behind before the next run starts.
    run_node_hygiene reap 60
    [ "${REAP_N:-0}" -gt 0 ] && echo "WARN orphans_after_run cfg=$tag n=$REAP_N"

    # harvest any freshly-computed Thompson table BEFORE the retention cleanup below
    # (which, on a failed run, purges everything except the 5 diagnostic files) so a
    # table generated by this run is never lost -- runs regardless of RC/OK.
    harvest_thompson_tables

    OK=0
    if [ -f rsl.error.0000 ] && grep -qa "SUCCESS COMPLETE WRF" rsl.error.0000; then
      OK=1
    fi

    grep -a "Timing for main" rsl.out.0000 2>/dev/null |   # -a: debug_level>=50 puts a Fortran NUL in mminlu, and GNU grep then prints "Binary file ... matches" and harvests ZERO steps (SC26-P1 job 1081398: rc=0 ok=1 but n=0 while all 324 timing lines were present)
    sed -nE 's/.*on domain +([0-9]+): +([0-9.]+) elapsed.*/\1 \2/p' > steps_by_domain.txt
    awk '$1 == 1 { print $2 }' steps_by_domain.txt > steps.txt

    if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
      STATS=$("$PYBIN" "$CAMPAIGN/configs.py" step-stats steps.txt)
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
        # Convert every frame of THIS run in parallel (2026-09-15 fix): the previous
        # one-frame-at-a-time serial loop measured ~3 min per BP frame on the
        # 1000x1000x100 lafe grid, which alone burned through the whole job's time limit
        # before the run set finished (job 1076893, cancelled mid-P06). Launch one
        # --exclusive srun step per frame in the background -- Slurm packs them onto
        # whatever nodes/cores are free in this job's own allocation, no manual node
        # assignment needed -- then wait for every one before checking results, so one
        # slow frame does not stall the others.
        CONV_PIDS=()
        for bf in "${FRAMES[@]}"; do
          ncf="${bf}.nc"
          srun --exclusive -N1 -n1 -c8 bash -c "cd $PIXI_ENV && pixi run python $CONV --input $RUNDIR/$bf --output $RUNDIR/$ncf" \
            > "convert_${bf}.log" 2>&1 &
          CONV_PIDS+=("$!")
        done
        CONV_RCS=()
        for pid in "${CONV_PIDS[@]}"; do
          wait "$pid"
          CONV_RCS+=("$?")
        done
        for i in "${!FRAMES[@]}"; do
          bf="${FRAMES[$i]}"
          ncf="${bf}.nc"
          rc="${CONV_RCS[$i]:-1}"
          if [ "$rc" != "0" ] || [ ! -s "$ncf" ]; then
            CONV_OK=0
            rm -f "$ncf"
            continue
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
                  # 2>/dev/null + tail -n1: fidelity-tokens prints exactly one real line;
                  # defends the read against any stray noise line preceding it. The extraction
                  # itself is configs.fidelity_notes_tokens() (unit-tested in
                  # campaign/test_configs.py TestFidelityNotesTokens) rather than an inline
                  # heredoc, which used to exist here as two byte-identical copies and once
                  # aborted the whole physics block with a $(...) syntax error (smoke jobs
                  # 1076238/1076240). If the helper fails, `read` gets nothing and FID_* stay
                  # empty -- the same notes shape as a run with no fidelity JSON at all.
                  # FID_ND / FID_FLOOR are the MEASURED maxima over fields at the last common
                  # time, not the thresholds.
                  read -r FID_LABEL FID_ND FID_FLOOR FID_DECIDING \
                    < <("$PYBIN" "$CAMPAIGN/configs.py" fidelity-tokens "${FID_PREFIX}.json" 2>/dev/null | tail -n1)
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
                  # 2>/dev/null + tail -n1: fidelity-tokens prints exactly one real line;
                  # defends the read against any stray noise line preceding it. The extraction
                  # itself is configs.fidelity_notes_tokens() (unit-tested in
                  # campaign/test_configs.py TestFidelityNotesTokens) rather than an inline
                  # heredoc, which used to exist here as two byte-identical copies and once
                  # aborted the whole physics block with a $(...) syntax error (smoke jobs
                  # 1076238/1076240). If the helper fails, `read` gets nothing and FID_* stay
                  # empty -- the same notes shape as a run with no fidelity JSON at all.
                  # FID_ND / FID_FLOOR are the MEASURED maxima over fields at the last common
                  # time, not the thresholds.
                  read -r FID_LABEL FID_ND FID_FLOOR FID_DECIDING \
                    < <("$PYBIN" "$CAMPAIGN/configs.py" fidelity-tokens "${FID_PREFIX}.json" 2>/dev/null | tail -n1)
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
    # Same rule as the STRICT loop below: a CONFIG_FILE-generated decomposition's tile geometry
    # has no ledger column of its own, so it goes into notes. Absent when CONFIG_FILE is unset.
    if [ -n "${CONFIG_FILE:-}" ]; then
      NOTES="$NOTES config_source=${CFG_SRC[$DECOMP_ID]:-generated} numtiles_x=${CFG_NTILES_X[$DECOMP_ID]:-} numtiles_y=${CFG_NTILES_Y[$DECOMP_ID]:-} tile_sz_x=${CFG_TSZ_X[$DECOMP_ID]:-} tile_sz_y=${CFG_TSZ_Y[$DECOMP_ID]:-} tile_strategy=${CFG_TSTRAT[$DECOMP_ID]:-} twin_decomp=$TWIN_DECOMP"
    fi
    [ "$RC" != "0" ] && NOTES="$NOTES rc=$RC"
    [ "$OK" != "1" ] && NOTES="$NOTES ok=$OK"
    NOTES="$NOTES $NODE_DIRTY_NOTE"

    export RUN_ID="${CASE}_physics_${tag}_r${rep}_${J}" SRC_DATASET="physics_campaign" JOB_ID="$J" \
      REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
      HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="$COMPLIANCE_CLASS" \
      E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$P_TS" \
      MP_PHYSICS="$P_MP" CU_PHYSICS="$P_CU" BL_PBL_PHYSICS="$P_BL" \
      SF_SFCLAY_PHYSICS="$P_SFCLAY" SF_SURFACE_PHYSICS="$P_SFSURF" \
      RA_LW_PHYSICS="$P_RLW" RA_SW_PHYSICS="$P_RSW" \
      RADT="$P_RADT" CUDT="$P_CUDT" \
      HISTORY_INTERVAL="$([ "$rep" = "1" ] && echo "$HIST_MINUTES" || echo 0)" IO_FORM_HISTORY="14" \
      N_NODES="$ALLOC_NODES" N_RANKS="$ranks" OMP_THREADS="$omp" NPROC_X="$nx" NPROC_Y="$ny" \
      NUMTILES="$numtiles" DISTRIBUTION="$dist" KMP_BLOCKTIME_COL="$BLOCKTIME_RECORD" \
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
  # an EMPTY associative array expands to an unbound variable under set -u (job 1080256 died here
  # once the filter above had dropped every candidate), so count it through a +x guard
  EXPECTED_N=0
  [ -n "${EXPECTED_CANDIDATE[*]+x}" ] && EXPECTED_N=${#EXPECTED_CANDIDATE[@]}
  EXPECTED_FIDELITY=$(( 1 + EXPECTED_N ))
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

mapfile -t ORDER < <("$PYBIN" "$CAMPAIGN/configs.py" order "$CASE" --reps "$REPS")
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

# ---------------------------------------------------------------------------
# RANKING_START=spinup (2026-09-26, wtune probe multi-point redesign, wired end-to-end): every
# candidate in the STRICT loop below starts from a SHARED WRF restart (t0+<minutes>) instead of
# t0, so a short ranking probe lands in a representative (post-spin-up) part of the run instead
# of the artificially-cheap cold-start ramp -- see wtune_probe_plan.py's module docstring and
# configs.py's edit_namelist restart_from/write_restart_interval/restart_io_form params. Fully
# opt-in via RANKING_START (default "t0"): unset/"t0" makes every line in this block and the
# RESTART_ARGS append below UNREACHABLE, so a run with RANKING_START unset is byte-identical to
# before this block existed (proof: `bash -n` plus a rendered-namelist diff, both run offline --
# see this task's report). Only meaningful for MODE=timing and only for a CONFIG_FILE-driven run
# (needs an ANCHOR row; the fixed STRICT C01-C16 templates have none).
#
# SPINUP_MINUTES (2026-09-26, C3 end-to-end multi-point wiring, decision 0054 "one allocation
# where feasible"): comma-separated list of simulated minutes, one restart TARGET per
# trajectory/restart-sourced probe point in ONE prediction, all measured within this SAME
# allocation. Each candidate's REPLICATE index (rep, 1-indexed, the existing REPS/build_order
# palindrome loop) selects its target via `(rep-1) mod len(targets)` -- REPS=len(targets) gives
# each replicate a DIFFERENT target; REPS=1 (or SPINUP_MINUTES unset) always resolves to index 0,
# reproducing the single-target behaviour byte-for-byte. Defaults to SPINUP_MIN alone (a single-
# element list), so every caller that only ever set SPINUP_MIN (every job before SPINUP_MINUTES
# existed) is unaffected.
RANKING_START=${RANKING_START:-t0}
case "$RANKING_START" in
  t0|spinup) ;;
  *) echo "RANKING_START must be 't0' or 'spinup', got '$RANKING_START'" >&2; exit 1 ;;
esac
SPINUP_MIN=${SPINUP_MIN:-60}
# Restart format for the ranking spin-up -> candidate handoff. 14 = ADIOS2 BP (parallel; 11.5 GB
# conus25 restart took 324 s to write / 120 s to read as serial netCDF, job 1231), with
# adios2_numaggregators = rank count set by configs.py _apply_adios2 (a smaller value corrupts the
# heap on read after a rank-placement change, wiki adios2-numaggregators-below-node-count-...).
# 2 = serial netCDF (the pre-2026-09-27 behaviour).
RANKING_RESTART_IO_FORM=${RANKING_RESTART_IO_FORM:-14}
case "$RANKING_RESTART_IO_FORM" in 2|14) ;; *) echo "RANKING_RESTART_IO_FORM must be 2 or 14" >&2; exit 1 ;; esac
# WRF's ADIOS2 reader opens with adios2_mode_read and inquires Times without BeginStep; with the
# default BP5 engine that is a streaming read and the inquiry fails (job 1273: "adios2 error in
# ext_adios2_open_for_read_begin"). BP4 keeps file-mode reads working. WRF picks up ./adios2.xml
# at init (io_adios2/wrf_io.F90), so the engine is chosen per restart file there -- no code change.
write_adios2_restart_xml() {  # dir restart_time max_dom
  [ "$RANKING_RESTART_IO_FORM" = 14 ] || return 0
  { echo '<?xml version="1.0"?>'; echo '<adios-config>'
    for d in $(seq 1 "$3"); do
      printf '  <io name="wrfrst_d%02d_%s">\n    <engine type="BP4"/>\n  </io>\n' "$d" "$2"
    done
    echo '</adios-config>'; } > "$1/adios2.xml"
}
SPINUP_MINUTES=${SPINUP_MINUTES:-$SPINUP_MIN}
# ':' separated (',' also accepted): a ',' inside a value collides with sbatch --export's own
# KEY=VAL,KEY=VAL syntax, which silently truncated 60,122 to 60 in job 1225.
IFS=':,' read -ra TARGET_MIN_ARR <<< "$SPINUP_MINUTES"
RESTART_SRC_DIR=${RESTART_SRC_DIR:-}
declare -A RESTART_SRC_DIR_BY_MIN RESTART_TIME_BY_MIN RANKING_SPINUP_OK_BY_MIN ANCHOR_NAMELIST_BY_MIN
RANKING_SPINUP_OK=""
RESTART_TIME=""
ANCHOR_NAMELIST=""
RESTART_ARGS=()

physics_signature () {
  # Stable, order-fixed extraction of the physics keys a restart-compat check cares about.
  # STRICT compliance (compliance: strict) never varies physics across CONFIG_FILE-driven
  # candidates in THIS loop today, so this check is not expected to ever trip here -- it exists
  # as a safety net against a future NAMELIST_OVERRIDES/physics_template change that would make
  # the anchor's restart file physically incompatible (a different restart variable set),
  # per-candidate, without crashing the job (see the per-candidate use below).
  local f="$1" k
  for k in mp_physics cu_physics bl_pbl_physics sf_sfclay_physics sf_surface_physics \
           ra_lw_physics ra_sw_physics; do
    grep -oE "^[[:space:]]*${k}[[:space:]]*=[^,!]*" "$f" 2>/dev/null | head -1
  done
}

if [ "$RANKING_START" = "spinup" ] && [ "$MODE" = "timing" ]; then
  if [ -z "${CFG_RANKS[ANCHOR]:-}" ]; then
    echo "ERROR: RANKING_START=spinup requires an ANCHOR config (CONFIG_FILE-driven run only)" >&2
    exit 1
  fi
  A_RANKS=${CFG_RANKS[ANCHOR]}; A_OMP=${CFG_OMP[ANCHOR]}; A_NX=${CFG_NX[ANCHOR]}
  A_NY=${CFG_NY[ANCHOR]}; A_NUMTILES=${CFG_NUMTILES[ANCHOR]}; A_DIST=${CFG_DIST[ANCHOR]}
  A_BLOCKTIME=${CFG_BLOCKTIME[ANCHOR]}
  A_NTASKS_PER_NODE=$(( A_RANKS / ALLOC_NODES ))
  for M in "${TARGET_MIN_ARR[@]}"; do
    SRC_DIR="$OUT/${CASE}_rankspinup_m${M}_${J}"
    if [ -n "$RESTART_SRC_DIR" ] && [ "${#TARGET_MIN_ARR[@]}" -eq 1 ] \
       && ls "$RESTART_SRC_DIR"/wrfrst_d0*_* >/dev/null 2>&1 \
       && [ -f "$RESTART_SRC_DIR/namelist.input" ]; then
      # legacy single-target reuse path (RESTART_SRC_DIR given): only safe/meaningful with
      # exactly one target -- reusing one caller-supplied restart for several DIFFERENT targets
      # would silently mislabel every target after the first, so multi-target reuse is refused
      # and this branch is skipped (falls through to writing a fresh one below) instead.
      SRC_DIR="$RESTART_SRC_DIR"
      echo "RANKING_SPINUP[m=$M]: reusing existing restart at $SRC_DIR"
      RANKING_SPINUP_OK_BY_MIN[$M]=1
      ANCHOR_NAMELIST_BY_MIN[$M]="$SRC_DIR/namelist.input"
    else
      check_disk_guard "rankspinup_m${M}"
      rm -rf "$SRC_DIR"; mkdir -p "$SRC_DIR"; cd "$SRC_DIR"
      for f in "${INPUT_FILES_ARR[@]}"; do ln -sf "$PREP_DIR/$f" "$f"; done
      link_wrf_inputs
      link_thompson_tables
      for f in "$W"/run/*; do
        b=$(basename "$f")
        case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;
          *) [ -e "$b" ] || ln -sf "$f" "$b" ;; esac
      done
      cp "$NAMELIST_TEMPLATE" ./namelist.input.orig
      python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \
        --nproc-x "$A_NX" --nproc-y "$A_NY" --numtiles "$A_NUMTILES" \
        --run-minutes "$M" --max-dom "$MAX_DOM" --history-mode off --ranks "$A_RANKS" \
        --write-restart-interval "$M" --restart-io-form "$RANKING_RESTART_IO_FORM"
      rm -f ./namelist.input.orig
      write_adios2_restart_xml "$SRC_DIR" "$(python3 "$CAMPAIGN/configs.py" restart-time "$CASE" "$M")" "$MAX_DOM"
      ln -sf "$W/main/wrf.exe" ./wrf.exe
      configure_omp_env "$A_OMP" "$A_BLOCKTIME"
      echo "RANKING_SPINUP[m=$M]: launching anchor spin-up in $SRC_DIR"
      site_launch "$A_RANKS" "$A_NTASKS_PER_NODE" "$A_OMP" "$A_DIST"
      timeout "$TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 &
      WRF_PID=$!
      wait "$WRF_PID"
      RANKING_SPINUP_RC=$?
      WRF_PID=""
      run_node_hygiene reap 60
      if [ "$RANKING_SPINUP_RC" -ne 0 ] || ! ls wrfrst_d0*_* >/dev/null 2>&1; then
        echo "WARNING: RANKING_SPINUP[m=$M] failed (rc=$RANKING_SPINUP_RC) or wrote no wrfrst_d0* -- this target falls back to t0 for any candidate that maps to it" >&2
      else
        echo "RANKING_SPINUP[m=$M]: OK, restart written in $SRC_DIR"
        RANKING_SPINUP_OK_BY_MIN[$M]=1
        ANCHOR_NAMELIST_BY_MIN[$M]="$SRC_DIR/namelist.input"
      fi
      cd "$OUT"
    fi
    RESTART_SRC_DIR_BY_MIN[$M]="$SRC_DIR"
    if [ -n "${RANKING_SPINUP_OK_BY_MIN[$M]:-}" ]; then
      RESTART_TIME_BY_MIN[$M]=$(python3 "$CAMPAIGN/configs.py" restart-time "$CASE" "$M")
      echo "RANKING_SPINUP[m=$M]: restart time = ${RESTART_TIME_BY_MIN[$M]}"
    fi
  done
fi

declare -A REPCOUNT
REF_NC=""
REF_CONFIG=""

for cid in "${ORDER[@]}"; do
  REPCOUNT[$cid]=$(( ${REPCOUNT[$cid]:-0} + 1 ))
  rep=${REPCOUNT[$cid]}
  ranks=${CFG_RANKS[$cid]}; omp=${CFG_OMP[$cid]}; nx=${CFG_NX[$cid]}; ny=${CFG_NY[$cid]}
  numtiles=${CFG_NUMTILES[$cid]}; dist=${CFG_DIST[$cid]}; blocktime=${CFG_BLOCKTIME[$cid]}
  ntasks_per_node=$(( ranks / ALLOC_NODES ))

  # RANKING_START=spinup, SPINUP_MINUTES multi-target: this replicate's own target minute, and
  # that target's own restart state (see the setup block above). (rep-1) mod len(targets) means
  # REPS=1 (or SPINUP_MINUTES unset -> a single-element list) always resolves index 0, so this
  # is a no-op lookup returning the SAME single target every rep -- unchanged from before
  # SPINUP_MINUTES existed.
  CUR_TARGET_IDX=$(( (rep - 1) % ${#TARGET_MIN_ARR[@]} ))
  CUR_TARGET_MIN=${TARGET_MIN_ARR[$CUR_TARGET_IDX]}
  RANKING_SPINUP_OK=${RANKING_SPINUP_OK_BY_MIN[$CUR_TARGET_MIN]:-}
  RESTART_SRC_DIR=${RESTART_SRC_DIR_BY_MIN[$CUR_TARGET_MIN]:-}
  RESTART_TIME=${RESTART_TIME_BY_MIN[$CUR_TARGET_MIN]:-}
  ANCHOR_NAMELIST=${ANCHOR_NAMELIST_BY_MIN[$CUR_TARGET_MIN]:-}

  check_disk_guard "${cid}_r${rep}"

  RUNDIR="$OUT/${CASE}_${cid}_r${rep}_${J}"
  rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"

  for f in "${INPUT_FILES_ARR[@]}"; do
    ln -sf "$PREP_DIR/$f" "$f"
  done
  # all-ADIOS2: wrfinput_d0N.bp (N <= max_dom) and wrfbdy_d01.bp come from bp_dir,
  # never from prep_dir's netCDF originals (input_files no longer lists them).
  link_wrf_inputs
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
  # RANKING_START=spinup (see the setup block above): a SECOND, additive edit-namelist pass that
  # re-points this candidate's namelist.input at the shared anchor restart -- never changes the
  # FIRST pass above, so RANKING_START unset/"t0" leaves this candidate's rendered namelist
  # byte-identical to before this feature existed (the block below is unreachable then).
  if [ -n "$RANKING_SPINUP_OK" ]; then
    if [ -n "$(diff <(physics_signature "$ANCHOR_NAMELIST") <(physics_signature ./namelist.input))" ]; then
      echo "RESTART_COMPAT_FALLBACK cid=$cid reason=physics_mismatch -- starting from t0 instead of the shared restart"
    else
      ln -sf "$RESTART_SRC_DIR"/wrfrst_d0*_* .
      write_adios2_restart_xml . "$RESTART_TIME" "$MAX_DOM"
      cp ./namelist.input ./namelist.input.restart_orig
      python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.restart_orig ./namelist.input \
        --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
        --run-minutes "$MIN" --max-dom "$MAX_DOM" \
        --history-mode "$HISTORY_MODE_ARG" --ranks "$ranks" \
        "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}" "${TILE_ARGS[@]}" \
        --restart-from "$RESTART_TIME" --restart-io-form "$RANKING_RESTART_IO_FORM"
      rm -f ./namelist.input.restart_orig
      echo "RANKING_START=spinup: cid=$cid starts from the shared restart at $RESTART_TIME"
    fi
  fi
  ln -sf "$W/main/wrf.exe" ./wrf.exe

  configure_omp_env "$omp" "$blocktime"

  T0=$(date +%s.%N)
  site_launch "$ranks" "$ntasks_per_node" "$omp" "$dist"
  # background + explicit wait -- see the physics loop's own copy above (and the node-hygiene
  # block's job-740 comment) for why.
  CUR_RUN_ID="${CASE}_${cid}_r${rep}_${J}"
  timeout "$TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 &
  WRF_PID=$!
  wait "$WRF_PID"
  RC=$?
  # see the physics loop's own copy above: RC>128 here is a narrow-race safety net only, since
  # node_hygiene_trap already wrote the interrupted row and exited before this line otherwise.
  WRF_PID=""
  CUR_RUN_ID=""
  T1=$(date +%s.%N)
  WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

  # post-run reap (design point 2): see the physics loop's own copy above for why.
  run_node_hygiene reap 60
  [ "${REAP_N:-0}" -gt 0 ] && echo "WARN orphans_after_run cfg=$cid n=$REAP_N"

  # harvest any freshly-computed Thompson table before the cleanup at the bottom of
  # this loop touches the run dir -- runs regardless of RC/OK.
  harvest_thompson_tables

  OK=0
  if [ -f rsl.error.0000 ] && grep -qa "SUCCESS COMPLETE WRF" rsl.error.0000; then
    OK=1
  fi

  # Nested runs print one "Timing for main ... on domain N" line per domain step, and the domain-1 line
  # ALREADY INCLUDES the nested integration (smoke job 1074515, isc21: domain-1 lines sum 758.7 s against
  # 841 s total wall including ~90 s of init/IO, while d02+d03 lines alone sum 821 s). So a d01 step is its
  # domain-1 line only; adding nested lines double-counts. steps_by_domain.txt keeps all raw lines.
  grep -a "Timing for main" rsl.out.0000 2>/dev/null |   # -a: debug_level>=50 puts a Fortran NUL in mminlu, and GNU grep then prints "Binary file ... matches" and harvests ZERO steps (SC26-P1 job 1081398: rc=0 ok=1 but n=0 while all 324 timing lines were present)
    sed -nE 's/.*on domain +([0-9]+): +([0-9.]+) elapsed.*/\1 \2/p' > steps_by_domain.txt
  awk '$1 == 1 { print $2 }' steps_by_domain.txt > steps.txt

  # step_stats() (campaign/configs.py) computes median (radiation-excluded, as
  # before) and mean_all (ALL d01 steps 2..n, radiation included -- see the
  # ledger's s_per_step_mean column). Only trust steps.txt when the run itself
  # reported success; a failed/truncated run's partial steps.txt is not a real
  # sample.
  if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
    STATS=$("$PYBIN" "$CAMPAIGN/configs.py" step-stats steps.txt)
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
  NOTES="$NOTES $NODE_DIRTY_NOTE"
  export RUN_ID="${CASE}_${cid}_r${rep}_${J}" SRC_DATASET="strict_campaign" JOB_ID="$J" \
    REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
    HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="STRICT" \
    E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$DT" \
    MP_PHYSICS="$PHYS_MP_PHYSICS" CU_PHYSICS="$PHYS_CU_PHYSICS" BL_PBL_PHYSICS="$PHYS_BL_PBL_PHYSICS" \
    SF_SFCLAY_PHYSICS="$PHYS_SF_SFCLAY_PHYSICS" SF_SURFACE_PHYSICS="$PHYS_SF_SURFACE_PHYSICS" \
    RA_LW_PHYSICS="$PHYS_RA_LW_PHYSICS" RA_SW_PHYSICS="$PHYS_RA_SW_PHYSICS" \
    RADT="$PHYS_RADT" CUDT="$PHYS_CUDT" \
    HISTORY_INTERVAL="$([ "$MODE" = correctness ] && echo "$MIN" || echo 0)" IO_FORM_HISTORY="14" \
    N_NODES="$ALLOC_NODES" N_RANKS="$ranks" OMP_THREADS="$omp" NPROC_X="$nx" NPROC_Y="$ny" \
    NUMTILES="$numtiles" DISTRIBUTION="$dist" KMP_BLOCKTIME_COL="$BLOCKTIME_RECORD" \
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

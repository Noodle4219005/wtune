#!/usr/bin/env bash
#SBATCH --job-name=strictcamp
# partition, nodes, tasks, time, account and output come from the sbatch command line
# (wtune.tools.scc_sweep.build_sbatch_command), which knows the machine.
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
# All-ADIOS2 (2026-09-14 user decision; 2026-10-03: inputs too -- an empty bp_dir is converted
# into a BP cache, see "ADIOS2 inputs"): every run's namelist is edited with ADIOS2
# I/O (io_form 14 everywhere, adios2_numaggregators=ranks -- see configs.py's
# _apply_adios2 and regress_bp.sbatch/job 1072441). MODE picks the history policy:
#   timing      (default) -- no run writes history at all (--history-mode off);
#                             fastest, cleanest wall-clock measurement.
#
# Usage: sbatch --export=ALL,CASE=karina,CONFIG_FILE=/abs/path/sample.json campaign/strict_campaign.sbatch
# Env:   CASE (required), MODE (timing only), REPS (default
#        2), STEPS (default: cases.json's "steps" for this case), TIMEOUT_S
#        (default 600). Override --time per case at submission -- the lead's job,
#        not this script's.
#        MIN_FREE_GIB (default 60): disk-quota guard threshold (see check_disk_guard
#        below) -- the job aborts itself with a ledger row instead of running into a
#        mid-write 'Disk quota exceeded' when the site's QUOTA_FS free space under the
#        soft quota drops below this many GiB (no QUOTA_FS: no guard).
#        CONFIG_FILE (required; WTune D1 decomposition-space work, predictor/decomp_space.py
#        to_campaign_configs/sample output): the generated config set this job measures --
#        palindromic REPS order, same ledger schema (tile geometry goes into notes, see
#        config_source= below).
#        Two OPTIONAL per-config CONFIG_FILE fields
#        (decision 0059, `wtune tune --pipeline full`; read by `configs.py config-extras`):
#          "physics_namelists": {"<point case id>": "/abs/namelist.input", ...}
#              this config's run for CASE=<point case id> uses that file as its namelist template
#              instead of the case's own (physics edits pre-rendered by wtune.physics_advice.
#              render_candidate_namelist + wtune.run.render_segment_namelist). nproc/numtiles/
#              ranks/adios2/history/restart are still applied per config on top, so ANY decomposition
#              (a twin = same physics file, different nx/ny/omp/ranks) overrides what the file
#              carries. Combine with RANKING_START=spinup: the run resumes from the point's shared
#              restart (physics_signature must match the anchor's -- a candidate that edits
#              mp/cu/pbl/sfclay/sfc/ra_lw/ra_sw is SKIPPED under RANKING_STRICT_RESTART=1).
#              A listed file that does not exist gives a SKIPPED ledger row, never a silent fallback.
#          "window_frame": true | "end" | "mid_end"   (or env WINDOW_FRAME=1|2 for every config)
#              level 1 (true/"end"): d01 writes exactly ONE history frame at the window end
#              (history_begin == history_interval == run minutes, --history-mode end_d01; the
#              window-start frame is skipped). level 2 ("mid_end"): frames at the window midpoint
#              (run_minutes//2) AND end (--history-mode mid_end_d01). A restarted WRF run never
#              writes a frame at its own start time (scc job 2194), and fidelity_gate.run_gate
#              labels INVALID unless a matched time exists after the reference's first valid time,
#              so a one-time frame cannot be gated: gate input = the run's two sorted frames. Every frame is converted to NetCDF AFTER wrf.exe returned,
#              outside the timed region. Only honoured in MODE=timing. Frames are kept as
#              <OUT>/<CASE>_<cid>_r<rep>_<job>/wrfout_d01_<Times>.nc; the ledger row's notes carry
#              window_frame=<abs path of the END frame> window_frame_time=<its Times> and, at
#              level 2, window_frame_mid=<abs path of the mid frame>; on failure
#              window_frame=none:<reason> and no frame is kept.
set -uo pipefail

SITE=${SITE:-scc}
export SITE
# CAMPAIGN is this checkout's campaign/ dir: the submitter exports it; submitted by hand from the
# checkout it is found under $SLURM_SUBMIT_DIR. Everything site-specific is inside
# campaign/sites/$SITE.sh, sourced below. export: CASES_JSON's default path (below) and the
# CASES_JSON overlay merge both need CAMPAIGN from a plain python3 subprocess.
CAMPAIGN=${CAMPAIGN:-${SLURM_SUBMIT_DIR:-$PWD}/campaign}
[ -d "$CAMPAIGN/sites" ] || { echo "CAMPAIGN=$CAMPAIGN has no sites/: export CAMPAIGN=<checkout>/campaign" >&2; exit 2; }
export CAMPAIGN
# campaign/sites/$SITE.sh defines W, OUT, HARDWARE_ID, PIXI_ENV, CONV, CASES_JSON_DEFAULT,
# CORES_PER_NODE and the functions site_env() (module load / MPI+IO env, called once below) and
# site_launch() (the per-run srun/mpirun invocation, called from both loops below).
source "$CAMPAIGN/sites/$SITE.sh"

REPS=${REPS:-2}
TIMEOUT_S=${TIMEOUT_S:-600}
# nodes in this allocation (srun --ntasks-per-node and the ledger n_nodes column); the #SBATCH
# header's 4-node shape gives 4 exactly as before, and a -N1 cf112 run (small cases) gives 1
ALLOC_NODES=${SLURM_JOB_NUM_NODES:-4}
MODE=${MODE:-timing}
case "$MODE" in
  timing) HISTORY_MODE_ARG=off ;;
  *) echo "MODE must be 'timing', got '$MODE' (the old modes are in archive/unused-2026-10/campaign/strict_campaign_full.sbatch)" >&2; exit 1 ;;
esac
: "${CONFIG_FILE:?CONFIG_FILE env var is required (the fixed C01-C16 run set is in archive/unused-2026-10/campaign/strict_campaign_full.sbatch)}"
# config generation needs Python >= 3.7 so configs.py can import wtune.decomp's scorer
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
# the WRF build: sha1 of the wrf.exe contents (coldstart_sample.build_id_expr); ledger notes
# build= and the Thompson table cache key
BUILD_ID=$(sha1sum < "$(readlink -f "$W/main/wrf.exe")" 2>/dev/null | cut -c1-16)
BUILD_ID=${BUILD_ID:-unknown}
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
# FOREIGN_LOAD (decision 0055): campaign/node_hygiene.sh's census also reports any process not
# owned by this job's user above 20% lifetime CPU (scc work5 2026-09-27: a root VS Code server at
# 91% of one core slowed a 576-rank wrf.exe 5x, invisible to the per-CPU NODE_DIRTY sample).
# Recorded in NOTES only for RANKING_START=spinup runs, or whenever it fires (see the ledger-row
# block), so a clean RANKING_START=t0 run's notes are byte-identical to before.
FOREIGN_LOAD=0
FOREIGN_LOAD_NOTE="foreign_load=0"

run_node_hygiene () {
  # run_node_hygiene <mode> <timeout_s>   mode: reap | census | reap+census
  # Sets REAP_N (total orphans reaped across every node, this call only) and, when a census
  # was requested and found a dirty node, NODE_DIRTY_NOTE (sticky -- the caller records the
  # preflight's one measurement in every ledger row this job writes, not a per-run
  # re-measurement).
  local mode="$1" tmo="$2" out hrc=0 nok
  site_hygiene "$mode"
  out=$(timeout "$tmo" "${SITE_HYGIENE_CMD[@]}" 2>&1) || hrc=$?
  [ -n "$out" ] && echo "$out"
  # fail-closed: node_hygiene.sh prints one HYGIENE_OK line per node it completed on. A census
  # whose dispatch failed or that did not report from every node is "unknown", never "none".
  # (a no-op site_hygiene -- f1's `true` -- has nothing to count.)
  if [ "${SITE_HYGIENE_CMD[0]}" != "true" ]; then
    nok=$(printf '%s\n' "$out" | grep -c "^HYGIENE_OK .*mode=$mode\$")
    if [ "$hrc" != "0" ] || [ "$nok" -lt "$ALLOC_NODES" ]; then
      echo "WARN hygiene_incomplete mode=$mode rc=$hrc ok_nodes=$nok/$ALLOC_NODES" >&2
      case "$mode" in *census*) [ "$NODE_DIRTY_NOTE" = "node_dirty=none" ] && NODE_DIRTY_NOTE="node_dirty=unknown" ;; esac
    fi
  fi
  REAP_N=$(printf '%s\n' "$out" | awk '/^REAPED_ORPHANS /{for(i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="n") s+=kv[2]}} END{print s+0}')
  if printf '%s\n' "$out" | grep -q '^NODE_DIRTY '; then
    NODE_DIRTY_NOTE="node_dirty=$(printf '%s\n' "$out" | awk '/^NODE_DIRTY /{h=c=b="";for(i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="host")h=kv[2]; else if(kv[1]=="cpu")c=kv[2]; else if(kv[1]=="busy")b=kv[2]} printf "%s:cpu%s:%s,",h,c,b}' | sed 's/,$//')"
  fi
  if printf '%s\n' "$out" | grep -q '^FOREIGN_LOAD '; then
    FOREIGN_LOAD=1
    FOREIGN_LOAD_NOTE="foreign_load=1 foreign=$(printf '%s\n' "$out" | awk '/^FOREIGN_LOAD /{h=p=u=c=x="";for(i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="host")h=kv[2]; else if(kv[1]=="pid")p=kv[2]; else if(kv[1]=="user")u=kv[2]; else if(kv[1]=="comm")c=kv[2]; else if(kv[1]=="pcpu")x=kv[2]} printf "%s:%s:%s:%s:%s,",h,p,u,c,x}' | sed 's/,$//')"
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
# Inputs: ADIOS2 .bp from BP_DIR -- the case's bp_dir (f1), or the BP cache an empty bp_dir is
# converted into (see "ADIOS2 inputs" below). Only the LEGACY escape WTUNE_INPUT_FORMAT=netcdf
# (BP_DIR emptied there) links prep_dir's netCDF wrfinput/wrfbdy, with io_form_input/boundary=2.
# Per-record inputs (WTUNE_PER_RECORD_BP, from stage-bp-inputs): WRF 4.8's ADIOS2 reader returns
# record 1 for every read of a multi-record BP, so wrfbdy/wrflowinp/wrffdda are held as
# $BP_DIR/<name>/<name>_<date>.bp, one BP per record, and every record file is linked here under
# its own name (configs.py _apply_adios2 sets multi_bdy_files / the dated innames). The netCDF
# original is linked too when prep_dir has it: a run starting off a record time (a shared restart,
# see _apply_adios2) reads that stream from netCDF instead.
link_per_record() {  # name [fresh copy dir]
  local c r
  for c in ${2:+"$2/$1"} "$BP_DIR/$1" "$BP_DIR/$1.bp"; do
    [ -d "$c" ] || continue
    for r in "$c/$1_"*.bp; do [ -d "$r" ] && ln -sfn "$r" "$(basename "$r")"; done
    break
  done
  [ -f "$PREP_DIR/$1" ] && ln -sf "$PREP_DIR/$1" "$1"
  return 0
}
# WTUNE_BP_FRESH_DIR/copy_<i>/<name>/ (configs.py fresh_probe_inputs, made on the head before
# sbatch): run i links its own copy -- NFS open stall, wiki 2026-10-03 adios2 gate.
FRESH_RUN_IDX=0
fresh_copy_for_run() {
  FRESH_COPY=""
  [ -n "${WTUNE_BP_FRESH_DIR:-}" ] && [ -n "${WTUNE_PER_RECORD_BP:-}" ] || return 0
  FRESH_COPY="$WTUNE_BP_FRESH_DIR/copy_$FRESH_RUN_IDX"
  if [ ! -d "$FRESH_COPY" ]; then
    FRESH_COPY=$(ls -d "$WTUNE_BP_FRESH_DIR"/copy_* 2>/dev/null | sort -V | tail -1)
    echo "WARNING: fresh BP copies exhausted at run $FRESH_RUN_IDX -- reusing ${FRESH_COPY:-the cache} (NFS open stalls likely)" >&2
  fi
  FRESH_RUN_IDX=$(( FRESH_RUN_IDX + 1 ))
}
link_wrf_inputs() {
  local d df n bdy_pr=0
  fresh_copy_for_run
  for ((d = 1; d <= MAX_DOM; d++)); do
    df=$(printf "d%02d" "$d")
    if [ "${RESTART:-0}" = 1 ]; then :     # a restart case: wrfrst comes from INPUT_FILES
    elif [ -n "$BP_DIR" ]; then ln -sf "$BP_DIR/wrfinput_${df}.bp" "wrfinput_${df}.bp"
    else ln -sf "$PREP_DIR/wrfinput_${df}" "wrfinput_${df}"; fi
  done
  if [ -n "$BP_DIR" ]; then
    for n in ${WTUNE_PER_RECORD_BP:-}; do
      link_per_record "$n" "$FRESH_COPY"
      [ "$n" = "wrfbdy_d01" ] && bdy_pr=1
    done
    [ "$bdy_pr" = 1 ] || ln -sf "$BP_DIR/wrfbdy_d01.bp" "wrfbdy_d01.bp"
  else ln -sf "$PREP_DIR/wrfbdy_d01" "wrfbdy_d01"; fi
}

check_disk_guard () {
  local ctx="$1" q_text free_gib
  [ -n "${QUOTA_FS:-}" ] || return 0      # the site has no per-user GPFS quota
  # the filesystem this run writes on (f1-arm: a project on work2 under QUOTA_FS=home2)
  local fs; fs=$(findmnt -no SOURCE,FSTYPE -T . 2>/dev/null | awk '$2 == "gpfs" {print $1}')
  fs=${fs:-$QUOTA_FS}
  if ! q_text=$(mmlsquota -u "$USER" "$fs" 2>&1); then
    echo "WARNING: mmlsquota unavailable ($q_text) -- skipping disk guard for $ctx" >&2
    return 0
  fi
  free_gib=$(printf '%s\n' "$q_text" | "$PYBIN" "$CAMPAIGN/configs.py" quota-free-gib --filesystem "$fs")
  if [ -z "$free_gib" ] || [ "$free_gib" = "NA" ]; then
    echo "WARNING: could not parse mmlsquota output -- skipping disk guard for $ctx" >&2
    return 0
  fi
  if awk -v f="$free_gib" -v m="$MIN_FREE_GIB" 'BEGIN { exit !(f < m) }'; then
    echo "ABORT: $fs free space ${free_gib} GiB < MIN_FREE_GIB=${MIN_FREE_GIB} GiB before $ctx -- stopping job (case=$CASE job=$J) to avoid a mid-run Disk quota exceeded failure" >&2
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
TABLE_CACHE="${WTUNE_TABLE_CACHE:-$HOME/wtune-runs/thompson_tables}/$BUILD_ID"
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

# Only rank 0's rsl files are read and kept, so the other ranks write to /dev/null: on a GPFS with
# 16 MiB blocks every open rsl file holds a whole block (f1-arm, isc21 at 8 nodes: 2304 files,
# 36 GiB of quota while running).
# WTUNE_KEEP_ALL_RSL=1 keeps every rank's files, to read a crashed rank's traceback.
rsl_rank0_only () {
  local r n
  [ "${WTUNE_KEEP_ALL_RSL:-0}" = 1 ] && return 0
  for ((r = 1; r < $1; r++)); do
    printf -v n '%04d' "$r"
    ln -sf /dev/null "rsl.out.$n"
    ln -sf /dev/null "rsl.error.$n"
  done
}

# Files the final ./namelist.input names (iofields_filename, sst_update's wrflowinp, wrffdda,
# auxinput*, tslist, ...) were never staged: input_files + wrfinput/wrfbdy only. climate-ea50
# (scc probe jobs 2541-2545): iofields_d01.txt absent -> WRF 4.8 hung after "calling input_input"
# until the 1260 s timeout, 0 steps. Call from within RUNDIR once namelist.input is final;
# never overrides an existing entry; a file found nowhere is reported (stderr), not fabricated.
link_namelist_files () {
  # shellcheck disable=SC2086
  python3 "$CAMPAIGN/configs.py" link-namelist-files ./namelist.input . ${NAMELIST_REF_DIRS:-} || true
}

harvest_thompson_tables () {
  # call from within RUNDIR, right after the run finishes and before any cleanup that
  # might delete a freshly generated table before it is cached. A cached or case-shipped
  # table is always a symlink here and is never touched; only a real regular file was
  # generated by this run.
  # A run with no 'Timing for main' may have died writing a table: nothing is cached from it.
  # cp to a temporary name in the cache, then mv: the cache may be on another filesystem.
  local tbl
  grep -qs 'Timing for main' rsl.error.0000 || return 0
  for tbl in "${THOMPSON_TABLE_NAMES[@]}"; do
    if [ -f "$tbl" ] && [ ! -L "$tbl" ]; then
      if [ -f "$TABLE_CACHE/$tbl" ]; then
        rm -f "$tbl"
      else
        cp -f "$tbl" "$TABLE_CACHE/.$tbl.$J.$$" && mv -f "$TABLE_CACHE/.$tbl.$J.$$" "$TABLE_CACHE/$tbl" \
          && rm -f "$tbl"
      fi
    fi
  done
}

# ---- per-run OpenMP/threading env (the site file's OMP_RUNTIME) ------------------------
# f1's KMP_BLOCKTIME knob has no equivalent under scc's nvhpc OpenMP runtime (SITE FACTS,
# decision 0050 step 8) -- never export it there, and record the ledger's kmp_blocktime column
# as "n/a" instead of the numeric/"default" value configs.py generated. scc uses OMP_PLACES/
# OMP_PROC_BIND for in-rank thread pinning instead (only meaningful when omp>1). Sets
# OMP_NUM_THREADS and BLOCKTIME_RECORD (the ledger column value) on both sites; call once per
# run, before site_launch, with the same (omp, blocktime) already resolved for this config.
configure_omp_env () {
  local omp="$1" blocktime="$2"
  export OMP_NUM_THREADS="$omp"
  if [ "${OMP_RUNTIME:-}" != "intel" ]; then
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
campaign_dir = os.environ["CAMPAIGN"]
site = os.environ["SITE"]
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
print("CASE_RESTART_TIME=%s" % q(c.get("restart_time", "")))      # a restart-only case
print("CASE_RESTART_IO_FORM=%s" % q(c.get("restart_io_form", 2)))
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
# NAMELIST_REF_DIRS: where link_namelist_files looks for files the namelist names (iofields,
# wrflowinp, wrffdda, ...) -- prep_dir, the template's dir, and for a rebuilt case
# (wtune.tune.write_rebuilt_case_entry) the original case's prep_dir.
_ref = [c["prep_dir"], os.path.dirname(c.get("namelist_template") or c["prep_dir"] + "/x")]
_orig = d.get(c.get("rebuilt_from") or "") or {}
if _orig.get("prep_dir"):
    _ref.append(os.path.expanduser(os.path.expandvars(_orig["prep_dir"])))
print("NAMELIST_REF_DIRS=%s" % q(" ".join(dict.fromkeys(x for x in _ref if x))))
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

# ---- ADIOS2 inputs (2026-10-03 user decision: netCDF is no longer used for WRF I/O) ----------
# An empty bp_dir (site overlay, e.g. scc; every rebuilt point, wtune.tune.write_rebuilt_case_entry)
# no longer means netCDF inputs: prep_dir's netCDF wrfinput_d0N/wrfbdy_d01 and the aux inputs the
# case namelist reads (wrflowinp_d0N when sst_update=1, wrffdda_d0N when grid_fdda>0, ...; see
# configs.py adios2_input_streams) are converted ONCE into a BP cache, here on the compute node and
# before any timed region: configs.py stage-bp-inputs -> src/wtune_bp.convert_inputs (BP4, manifest
# cached: reconverted only when a source changed; the adios2 bindings run in envs/adios2's python,
# $WTUNE_ADIOS2_PYTHON overrides) and <cache>/<name>.bp links. Cache:
# ${WTUNE_BP_CACHE:-$HOME/wtune-runs/bp-cache}/<case>/<sha1(prep_dir)[:12]>. Race-safe: the stage
# holds a lockf lock on <cache>/.stage.lock (NFS/GPFS-safe across nodes), so several jobs of one case
# starting together convert once; the others block, then find the manifest current. A non-empty
# bp_dir (f1) is only checked (--check: nothing converted or written). Either way BP_DIR then holds
# wrfinput_d0N.bp/wrfbdy_d01.bp (link_wrf_inputs) and WTUNE_INPUT_FORMAT=adios2.
# WTUNE_AUX_BP_STREAMS (read by configs.py _apply_adios2): the aux streams available as .bp -- only
# those get io_form 14 + "<name>.bp" innames; an aux file missing as BP keeps the case's io_form.
# WTUNE_INPUT_FORMAT=netcdf exported at submission is the LEGACY escape hatch (warned below).
# wrfbdy/wrflowinp/wrffdda are converted one BP per record (WTUNE_PER_RECORD_BP, link_per_record):
# WRF 4.8's ADIOS2 reader never selects a step, so a multi-record BP yields record 1 for every read.
# WRF opens <name>_<clock time>.bp, so a RANKING candidate restarting at a time with no record
# (SPINUP_MINUTES not a multiple of interval_seconds / auxinput4_interval) reads that stream from
# its netCDF file for that run (configs.py _apply_adios2, a NOTE on stderr) -- correct, unlike any
# BP layout there.
WTUNE_AUX_BP_STREAMS=""
WTUNE_PER_RECORD_BP=""
if [ "${WTUNE_INPUT_FORMAT:-}" = "netcdf" ]; then
  echo "WARNING: WTUNE_INPUT_FORMAT=netcdf -- LEGACY netCDF wrfinput/wrfbdy from prep_dir (ADIOS2 is the default for every WRF stream since 2026-10-03)" >&2
  BP_DIR=""
else
  if [ -z "$BP_DIR" ]; then
    BP_STAGE_ARGS=("$PREP_DIR")
  else
    BP_STAGE_ARGS=("$BP_DIR" --cache-dir "$BP_DIR" --check)
  fi
  if ! BP_STAGE_OUT=$("$PYBIN" "$CAMPAIGN/configs.py" stage-bp-inputs "$CASE" "$NAMELIST_TEMPLATE" "${BP_STAGE_ARGS[@]}"); then
    echo "FATAL: ADIOS2 input staging failed for case $CASE (see above) -- refusing to run (legacy escape: --export=...,WTUNE_INPUT_FORMAT=netcdf)" >&2
    exit 1
  fi
  eval "$BP_STAGE_OUT"
  export WTUNE_INPUT_FORMAT=adios2
  # BP_DIR first, so link_namelist_files finds the aux <name>.bp before prep_dir's netCDF
  NAMELIST_REF_DIRS="$BP_DIR $NAMELIST_REF_DIRS"
  echo "ADIOS2 inputs: BP_DIR=$BP_DIR aux streams as BP: ${WTUNE_AUX_BP_STREAMS:-none} per record: ${WTUNE_PER_RECORD_BP:-none}"
  if [ -n "$WTUNE_PER_RECORD_BP" ] && [ -z "${WTUNE_BP_FRESH_DIR:-}" ]; then
    echo "WARNING: no WTUNE_BP_FRESH_DIR: runs share the cache's per-record inodes -- NFS open stalls may distort timings (wiki 2026-10-03 adios2 gate)" >&2
  fi
fi
# WTUNE_PER_RECORD_BP/WTUNE_BP_DIR (configs.py _apply_adios2): per-record streams and where their
# record files live -- a run whose start has no <name>_<start>.bp reads that stream from netCDF.
export WTUNE_PER_RECORD_BP WTUNE_BP_DIR="$BP_DIR"
export WTUNE_AUX_BP_STREAMS

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
fi

# Per-config physics_namelists / window_frame (see the header). Parsed with the same \037
# separator trick as above: an EMPTY physics_namelist field must not shift the columns.
# WINDOW_FRAME=1 (job-wide env) turns the window-end frame on for every config.
declare -A CFG_PHYS_NL CFG_WINDOW_FRAME
WINDOW_FRAME=${WINDOW_FRAME:-0}
if [ -n "${CONFIG_FILE:-}" ]; then
  while IFS=$'\037' read -r cid pnl wf; do
    [ -z "$cid" ] && continue
    CFG_PHYS_NL[$cid]=$pnl; CFG_WINDOW_FRAME[$cid]=$wf
  done < <("$PYBIN" "$CAMPAIGN/configs.py" config-extras "$CASE" 2>/dev/null | tr '\t' '\037')
fi
WINDOW_FRAME_ANY=0
case "$WINDOW_FRAME" in 0|"") ;; *) WINDOW_FRAME_ANY=1 ;; esac
for cid in "${!CFG_WINDOW_FRAME[@]}"; do case "${CFG_WINDOW_FRAME[$cid]}" in 0|"") ;; *) WINDOW_FRAME_ANY=1 ;; esac; done
if [ "$MODE" = "timing" ] && [ "$WINDOW_FRAME_ANY" = "1" ] && [ ! -f "$CONV" ]; then
  echo "FATAL: ADIOS2->netCDF converter not found at $CONV (needed by window_frame) -- refusing to run" >&2
  exit 1
fi

mapfile -t ORDER < <("$PYBIN" "$CAMPAIGN/configs.py" order "$CASE" --reps "$REPS")
echo "order (${#ORDER[@]} runs): ${ORDER[*]}"
if [ ${#ORDER[@]} -eq 0 ]; then
  echo "no configs resolved for case $CASE" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# RANKING_START=spinup (2026-09-26, wtune probe multi-point redesign, wired end-to-end): every
# candidate in the STRICT loop below starts from a SHARED WRF restart (t0+<minutes>) instead of
# t0, so a short ranking probe lands in a representative (post-spin-up) part of the run instead
# of the artificially-cheap cold-start ramp -- see wtune/probe_plan.py's module docstring and
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
  rm -f "$1/adios2.xml"  # a symlink here (input_files / $W/run) must not be written through
  { echo '<?xml version="1.0"?>'; echo '<adios-config>'
    for d in $(seq 1 "$3"); do
      printf '  <io name="wrfrst_d%02d_%s">\n    <engine type="BP4"/>\n  </io>\n' "$d" "$2"
    done
    echo '</adios-config>'; } > "$1/adios2.xml"
}
SPINUP_MINUTES=${SPINUP_MINUTES:-$SPINUP_MIN}
# Decision 0055 (wtune tune --rank-by predicted), all opt-in:
#   SPINUP_TIMEOUT_S        watchdog for the shared anchor spin-up only (default TIMEOUT_S, i.e.
#                           unchanged); candidate runs keep TIMEOUT_S.
#   RANKING_STRICT_RESTART  1: a candidate that cannot start from the shared restart (spin-up
#                           failed / wrote no wrfrst, or physics_signature mismatch) is NOT run
#                           from t0; it gets a status=SKIPPED ledger row with restart_used=0 and
#                           the reason (write_restart_skip_row). Default 0 keeps the old t0
#                           fallback exactly.
# RANKING_ACTIVE marks the jobs whose candidate rows carry restart_target=/restart_used=/
# foreign_load= notes (RANKING_START=spinup in MODE=timing -- the only combination that runs the
# spin-up block); a RANKING_START=t0 job never sets it, so its notes are unchanged.
SPINUP_TIMEOUT_S=${SPINUP_TIMEOUT_S:-$TIMEOUT_S}
RANKING_STRICT_RESTART=${RANKING_STRICT_RESTART:-0}
RANKING_ACTIVE=0
[ "$RANKING_START" = "spinup" ] && [ "$MODE" = "timing" ] && RANKING_ACTIVE=1

write_restart_skip_row () {
  # write_restart_skip_row <run_id> <cid> <rep> <reason> <target_min>: RANKING_STRICT_RESTART=1's
  # replacement for a silent t0 fallback -- one ledger row, empty s_per_step_median (so every
  # aggregator drops it as a measurement) and notes that say exactly why. Same minimal-row shape
  # as write_interrupted_row.
  RUN_ID="$1" JOB_ID="$J" REP_INDEX="$3" CASE_RAW="$CASE" CASE_BASE="$CASE" \
    HARDWARE_ID="$HARDWARE_ID" \
    NOTES_SKIP="status=SKIPPED reason=$4 config=$2 restart_target=$5 restart_used=0 $NODE_DIRTY_NOTE $FOREIGN_LOAD_NOTE" \
    python3 - "$LEDGER" <<'PY'
import csv, os, sys
ledger = sys.argv[1]
e = os.environ.get
row = [e("RUN_ID", ""), "restart_skip", e("JOB_ID", ""), e("REP_INDEX", ""), e("CASE_RAW", ""),
       e("CASE_BASE", ""), "", e("HARDWARE_ID", ""), "", ""] + [""] * 37 + [e("NOTES_SKIP", "")]
with open(ledger, "a", newline="") as f:
    csv.writer(f).writerow(row)
PY
  echo "RANKING_STRICT_RESTART: SKIPPED run_id=$1 reason=$4 (no t0 fallback)" >&2
}
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

# ---------------------------------------------------------------------------
# Point-job run list (decisions 0055/0059), all opt-in; defaults leave ORDER untouched.
#   ANCHOR_END   unset/1: the orchestrator's trailing ANCHOREND run is kept (drift gate on, as
#                before). 0: ANCHOREND is dropped from ORDER -- "drift gate off" (saves one full
#                launch + WRF init + restart read, ~87 s on scc job 2325); the ledger notes then
#                carry drift_gate=off.
#   POINT_FUSED  1 (set by the orchestrator, winner-only plans only): when ORDER holds exactly ONE
#                run once ANCHOR/ANCHOREND are removed, skip the restart hand-off -- ONE wrf.exe
#                from the rebuilt wrfinput runs SPINUP_MINUTES of spin-up plus the STEPS window, and
#                s/step comes only from steps after the spin-up model time (configs.py
#                post_spinup_steps; radiation/history steps excluded by step_stats as always). No
#                restart is written or read, ANCHOR is not run, and ANCHOREND defaults to OFF (an
#                explicit ANCHOR_END=1 is ignored: there is no restart to bracket). Ignored with a
#                warning when RANKING_START!=spinup, MODE!=timing, SPINUP_MINUTES has several targets,
#                a window frame is requested, or more than one run remains.
#   WINDOW_EXACT 1: the measured window is exactly STEPS*dt via run_seconds instead of being rounded
#                UP to whole minutes (MIN=ceil(STEPS*dt/60): 110 steps at dt=10 ran 114). Default 0.
FUSED_ON=0
POINT_NOTE=""
POINT_ANCHOR_END=1
if [ "$MODE" = "timing" ]; then
  PF_REQ=${POINT_FUSED:-0}
  PF_MSG=""
  if [ "$PF_REQ" = "1" ]; then
    if [ "$RANKING_ACTIVE" != "1" ]; then PF_MSG="RANKING_START!=spinup"; PF_REQ=0
    elif [ "${#TARGET_MIN_ARR[@]}" -ne 1 ]; then PF_MSG="SPINUP_MINUTES has ${#TARGET_MIN_ARR[@]} targets"; PF_REQ=0; fi
  fi
  eval "$("$PYBIN" "$CAMPAIGN/configs.py" point-plan --fused "$PF_REQ" --anchor-end "${ANCHOR_END:-}" \
            --window-frame "$WINDOW_FRAME_ANY" "${ORDER[@]}")"
  ORDER=("${POINT_ORDER[@]}")
  FUSED_ON=$POINT_FUSED_ON
  POINT_ANCHOR_END=$POINT_ANCHOR_END
  [ -n "$PF_MSG" ] && POINT_NOTE="POINT_FUSED ignored: $PF_MSG"
  [ -n "$POINT_NOTE" ] && echo "WARNING: $POINT_NOTE" >&2
  [ "$FUSED_ON" = "1" ] && echo "POINT_FUSED: single-config fast path for cid=${ORDER[0]} (one wrf.exe: ${TARGET_MIN_ARR[0]} min spin-up + ${STEPS} step window, no restart hand-off, drift gate off)"
  [ "$POINT_ANCHOR_END" = "0" ] && [ "$FUSED_ON" != "1" ] && echo "ANCHOR_END=0: ANCHOREND dropped (drift gate off)"
  echo "point order (${#ORDER[@]} runs): ${ORDER[*]}"
fi

if [ "$RANKING_START" = "spinup" ] && [ "$MODE" = "timing" ] && [ "$FUSED_ON" != "1" ]; then
  if [ -z "${CFG_RANKS[ANCHOR]:-}" ]; then
    echo "ERROR: RANKING_START=spinup requires an ANCHOR config (CONFIG_FILE-driven run only)" >&2
    exit 1
  fi
  A_RANKS=${CFG_RANKS[ANCHOR]}; A_OMP=${CFG_OMP[ANCHOR]}; A_NX=${CFG_NX[ANCHOR]}
  A_NY=${CFG_NY[ANCHOR]}; A_NUMTILES=${CFG_NUMTILES[ANCHOR]}; A_DIST=${CFG_DIST[ANCHOR]}
  A_BLOCKTIME=${CFG_BLOCKTIME[ANCHOR]}
  A_NTASKS_PER_NODE=$(( A_RANKS / ALLOC_NODES ))
  # same case overrides / delete-keys / nest / tile arguments as the candidate runs' edit-namelist
  # below: the spin-up must run under the namelist the candidates restart into.
  A_SET_ARGS=()
  for kv in ${NAMELIST_OVERRIDES:-}; do A_SET_ARGS+=(--set "$kv"); done
  A_DELETE_ARGS=()
  for k in ${NAMELIST_DELETE_KEYS:-}; do A_DELETE_ARGS+=(--delete-key "$k"); done
  A_NEST_ARGS=()
  [ "$NEST_HISTORY" = "1" ] && A_NEST_ARGS+=(--nest-history-on)
  A_TILE_ARGS=(
    --numtiles-x "${CFG_NTILES_X[ANCHOR]:-0}" --numtiles-y "${CFG_NTILES_Y[ANCHOR]:-0}"
    --tile-sz-x "${CFG_TSZ_X[ANCHOR]:-0}" --tile-sz-y "${CFG_TSZ_Y[ANCHOR]:-0}"
    --tile-strategy "${CFG_TSTRAT[ANCHOR]:-0}"
  )
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
        "${A_SET_ARGS[@]}" "${A_DELETE_ARGS[@]}" "${A_NEST_ARGS[@]}" "${A_TILE_ARGS[@]}" \
        --write-restart-interval "$M" --restart-io-form "$RANKING_RESTART_IO_FORM"
      rm -f ./namelist.input.orig
      write_adios2_restart_xml "$SRC_DIR" "$(python3 "$CAMPAIGN/configs.py" restart-time "$CASE" "$M")" "$MAX_DOM"
      link_namelist_files
      ln -sf "$W/main/wrf.exe" ./wrf.exe
      rsl_rank0_only "$A_RANKS"
      configure_omp_env "$A_OMP" "$A_BLOCKTIME"
      echo "RANKING_SPINUP[m=$M]: launching anchor spin-up in $SRC_DIR"
      site_launch "$A_RANKS" "$A_NTASKS_PER_NODE" "$A_OMP" "$A_DIST"
      timeout "$SPINUP_TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 &
      WRF_PID=$!
      wait "$WRF_PID"
      RANKING_SPINUP_RC=$?
      WRF_PID=""
      run_node_hygiene reap 60
      if [ "$RANKING_SPINUP_RC" -ne 0 ] || ! ls wrfrst_d0*_* >/dev/null 2>&1; then
        if [ "$RANKING_STRICT_RESTART" = "1" ]; then
          echo "WARNING: RANKING_SPINUP[m=$M] failed (rc=$RANKING_SPINUP_RC) or wrote no wrfrst_d0* -- RANKING_STRICT_RESTART=1: every candidate that maps to it is SKIPPED (no t0 fallback)" >&2
        else
          echo "WARNING: RANKING_SPINUP[m=$M] failed (rc=$RANKING_SPINUP_RC) or wrote no wrfrst_d0* -- this target falls back to t0 for any candidate that maps to it" >&2
        fi
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
  # physics_namelists (header): this config's own pre-rendered namelist for this CASE replaces the
  # case template; every edit below (decomposition, history, restart) still applies on top of it.
  RUN_TEMPLATE="$NAMELIST_TEMPLATE"
  PHYS_NL="${CFG_PHYS_NL[$cid]:-}"
  if [ -n "$PHYS_NL" ]; then
    if [ -f "$PHYS_NL" ]; then
      RUN_TEMPLATE="$PHYS_NL"
    else
      echo "ERROR: cid=$cid physics_namelists[$CASE]=$PHYS_NL does not exist -- run SKIPPED" >&2
      write_restart_skip_row "${CASE}_${cid}_r${rep}_${J}" "$cid" "$rep" "physics_namelist_missing" "$CUR_TARGET_MIN"
      cd "$OUT"
      continue
    fi
  fi
  # window_frame (header): one end-of-window d01 history frame, MODE=timing only
  WF=0; WF_KEEP=(); WF_NOTE=""
  if [ "$MODE" = "timing" ]; then
    WF=${CFG_WINDOW_FRAME[$cid]:-0}
    [ "$WF" = "0" ] && WF=$WINDOW_FRAME
    case "$WF" in 0|1|2) ;; *) WF=1 ;; esac
  fi
  RUN_HISTORY_ARG="$HISTORY_MODE_ARG"
  [ "$WF" = "1" ] && RUN_HISTORY_ARG=end_d01
  [ "$WF" = "2" ] && RUN_HISTORY_ARG=mid_end_d01
  cp "$RUN_TEMPLATE" ./namelist.input.orig
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
  # Window length. Default: MIN = ceil(STEPS*dt/60) whole minutes (unchanged). FUSED_ON: spin-up +
  # STEPS*dt, exact via run_seconds. WINDOW_EXACT=1 (timing, no window frame): STEPS*dt exact.
  RUN_MIN_EFF=$MIN; RUN_SEC_ARGS=(); TIMEOUT_EFF=$TIMEOUT_S
  if [ "$FUSED_ON" = "1" ]; then
    TOTAL_S=$(( CUR_TARGET_MIN * 60 + STEPS * DT ))
    RUN_MIN_EFF=$(( TOTAL_S / 60 )); RUN_SEC_ARGS=(--run-seconds $(( TOTAL_S % 60 )))
    TIMEOUT_EFF=$(( TIMEOUT_S + SPINUP_TIMEOUT_S ))
  elif [ "${WINDOW_EXACT:-0}" = "1" ] && [ "$MODE" = "timing" ] && [ "$RUN_HISTORY_ARG" = "off" ]; then
    TOTAL_S=$(( STEPS * DT ))
    RUN_MIN_EFF=$(( TOTAL_S / 60 )); RUN_SEC_ARGS=(--run-seconds $(( TOTAL_S % 60 )))
  fi
  CASE_RESTART_ARGS=()   # a restart-only case starts every probe from its own wrfrst
  [ "$RESTART" = 1 ] && CASE_RESTART_ARGS=(--restart-from "$CASE_RESTART_TIME"
                                           --restart-io-form "$CASE_RESTART_IO_FORM")
  python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.orig ./namelist.input \
    --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
    --run-minutes "$RUN_MIN_EFF" "${RUN_SEC_ARGS[@]}" --max-dom "$MAX_DOM" \
    --history-mode "$RUN_HISTORY_ARG" --ranks "$ranks" \
    "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}" "${TILE_ARGS[@]}" \
    "${CASE_RESTART_ARGS[@]}"
  rm -f ./namelist.input.orig
  [ "$RESTART" = 1 ] && RANKING_RESTART_IO_FORM=$CASE_RESTART_IO_FORM \
    write_adios2_restart_xml . "$CASE_RESTART_TIME" "$MAX_DOM"
  # RANKING_START=spinup (see the setup block above): a SECOND, additive edit-namelist pass that
  # re-points this candidate's namelist.input at the shared anchor restart -- never changes the
  # FIRST pass above, so RANKING_START unset/"t0" leaves this candidate's rendered namelist
  # byte-identical to before this feature existed (the block below is unreachable then).
  RESTART_USED=0
  RESTART_SKIP_REASON="spinup_failed"
  if [ "$FUSED_ON" = "1" ]; then
    # POINT_FUSED: no restart file exists or is read; the measured state is the same post-spin-up
    # state, reached inside this one run. Recorded as restart_used=1 + fused=1 in the notes.
    echo "POINT_FUSED: cid=$cid runs spin-up ${CUR_TARGET_MIN} min + measurement in one wrf.exe"
    RESTART_USED=1
  elif [ -n "$RANKING_SPINUP_OK" ]; then
    if [ -n "$(diff <(physics_signature "$ANCHOR_NAMELIST") <(physics_signature ./namelist.input))" ]; then
      echo "RESTART_COMPAT_FALLBACK cid=$cid reason=physics_mismatch -- starting from t0 instead of the shared restart"
      RESTART_SKIP_REASON="physics_mismatch"
    else
      ln -sf "$RESTART_SRC_DIR"/wrfrst_d0*_* .
      cp ./namelist.input ./namelist.input.restart_orig
      python3 "$CAMPAIGN/configs.py" edit-namelist ./namelist.input.restart_orig ./namelist.input \
        --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" \
        --run-minutes "$RUN_MIN_EFF" "${RUN_SEC_ARGS[@]}" --max-dom "$MAX_DOM" \
        --history-mode "$RUN_HISTORY_ARG" --ranks "$ranks" \
        "${SET_ARGS[@]}" "${DELETE_ARGS[@]}" "${NEST_ARGS[@]}" "${TILE_ARGS[@]}" \
        --restart-from "$RESTART_TIME" --restart-io-form "$RANKING_RESTART_IO_FORM"
      RESTART_EDIT_RC=$?
      if [ "$RESTART_EDIT_RC" = "0" ]; then
        write_adios2_restart_xml . "$RESTART_TIME" "$MAX_DOM"
        rm -f ./namelist.input.restart_orig
        echo "RANKING_START=spinup: cid=$cid starts from the shared restart at $RESTART_TIME"
        RESTART_USED=1
      else
        # never measure a t0 run under the restart label: put the t0 namelist back, restart_used
        # stays 0 (SKIPPED under RANKING_STRICT_RESTART=1, an honest t0 fallback otherwise).
        mv -f ./namelist.input.restart_orig ./namelist.input
        rm -f ./wrfrst_d0*_*
        echo "RESTART_EDIT_FAILED cid=$cid rc=$RESTART_EDIT_RC -- not starting from the shared restart" >&2
        RESTART_SKIP_REASON="restart_edit_failed"
      fi
    fi
  fi
  if [ "$RANKING_ACTIVE" = "1" ] && [ "$RESTART_USED" != "1" ] && [ "$RANKING_STRICT_RESTART" = "1" ]; then
    write_restart_skip_row "${CASE}_${cid}_r${rep}_${J}" "$cid" "$rep" "$RESTART_SKIP_REASON" "$CUR_TARGET_MIN"
    cd "$OUT"
    continue
  fi
  link_namelist_files
  ln -sf "$W/main/wrf.exe" ./wrf.exe
  rsl_rank0_only "$ranks"

  configure_omp_env "$omp" "$blocktime"

  # A rank killed by SIGSEGV is retried once: f1-arm's Open MPI 5.0.10 crashes intermittently
  # in ob1's large-message path over btl/sm (MPI_Alltoallv -> mca_btl_sm_poll_handle_frag ->
  # mca_pml_ob1_send_request_schedule_once, haitang2017 2026-10-27), so the same configuration
  # on the same inputs runs on a second try. Without the retry a node sweep lost every
  # anchor and tune fell back to the uncalibrated proxy.
  for ATTEMPT in 1 2; do
  T0=$(date +%s.%N)
  site_launch "$ranks" "$ntasks_per_node" "$omp" "$dist"
  # background + explicit wait -- see the physics loop's own copy above (and the node-hygiene
  # block's job-740 comment) for why.
  CUR_RUN_ID="${CASE}_${cid}_r${rep}_${J}"
  timeout "$TIMEOUT_EFF" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 &
  WRF_PID=$!
  wait "$WRF_PID"
  RC=$?
  if [ "$ATTEMPT" = 1 ] && { [ "$RC" = 139 ] || grep -qa "exited on signal 11" wrf.out; }; then
    echo "RETRY cfg=$cid rep=$rep rc=$RC: a rank died on signal 11; running it once more"
    mv wrf.out wrf.out.attempt1
    if [ "${WTUNE_KEEP_ALL_RSL:-0}" = 1 ]; then mkdir -p attempt1 && mv rsl.* attempt1/; fi
    rm -rf wrfout_d* wrfrst_d* rsl.out.0000 rsl.error.0000
    continue
  fi
  break
  done
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

  # restart_used=1 must mean the run REALLY started at the shared restart (same check as the
  # physics loop's copy, which never sees RESTART_USED=1): a restart run's first d01 step is
  # stamped RESTART_TIME+dt, a t0 start earlier. POINT_FUSED reads no restart, so it is exempt.
  if [ "$RESTART_USED" = "1" ] && [ "$FUSED_ON" != "1" ] && [ -n "$RESTART_TIME" ]; then
    FIRST_STEP_TIME=$(grep -a -m1 "Timing for main" rsl.out.0000 2>/dev/null | sed -nE 's/.*time +([0-9_:-]+) .*/\1/p')
    if [ -n "$FIRST_STEP_TIME" ] && [[ ! "$FIRST_STEP_TIME" > "$RESTART_TIME" ]]; then
      echo "RESTART_CHECK_FAILED cid=$cid first_step=$FIRST_STEP_TIME restart_time=$RESTART_TIME" >&2
      RESTART_USED=0
    fi
  fi

  # step_stats() (campaign/configs.py) computes median (radiation-excluded, as
  # before) and mean_all (ALL d01 steps 2..n, radiation included -- see the
  # ledger's s_per_step_mean column). Only trust steps.txt when the run itself
  # reported success; a failed/truncated run's partial steps.txt is not a real
  # sample.
  if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then
    # POINT_FUSED: keep only the steps after the spin-up model time (steps_all.txt keeps the raw list)
    if [ "$FUSED_ON" = "1" ]; then
      mv steps.txt steps_all.txt
      "$PYBIN" "$CAMPAIGN/configs.py" fused-steps rsl.out.0000 --spin-seconds $(( CUR_TARGET_MIN * 60 )) --dt "$DT" > steps.txt
    fi
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

  DIFF_FIELDS=""

  # ---- window_frame: convert the end-of-window (and, level 2, window-mid) BP frames to NetCDF,
  # AFTER the timed region.
  # The s/step above comes only from "Timing for main" lines, which do not contain the history
  # write (WRF prints that as its own "Timing for Writing wrfout..." line), and the write lands at
  # the very last step anyway, so the measurement above never sees it; the conversion below runs
  # after wrf.exe returned. Direct python on the batch node: no srun (broken for MPI on scc).
  if [ "$WF" != "0" ]; then
    WF_BPS=$(ls -d wrfout_d01_* 2>/dev/null | grep -v '\.nc$' | sort)
    WF_NOTE=""
    if [ "$RC" = "0" ] && [ "$OK" = "1" ] && [ -n "$WF_BPS" ]; then
      for WF_BP in $WF_BPS; do
        "$PYBIN" "$CONV" --input "$RUNDIR/$WF_BP" --output "$RUNDIR/${WF_BP}.nc" > "convert_${WF_BP}.log" 2>&1
        if [ "$?" = "0" ] && [ -s "${WF_BP}.nc" ]; then
          WF_KEEP+=("${WF_BP}.nc"); rm -rf "$WF_BP"
        else
          rm -f "${WF_BP}.nc"; WF_NOTE="window_frame=none:convert_failed"; break
        fi
      done
      if [ -z "$WF_NOTE" ]; then
        WF_LAST=${WF_KEEP[${#WF_KEEP[@]}-1]}
        WF_TIME=${WF_LAST#wrfout_d01_}; WF_TIME=${WF_TIME%.nc}
        WF_NOTE="window_frame=$RUNDIR/$WF_LAST window_frame_time=$WF_TIME"
        [ "$WF" = "2" ] && [ "${#WF_KEEP[@]}" -ge 2 ] && WF_NOTE="$WF_NOTE window_frame_mid=$RUNDIR/${WF_KEEP[0]}"
      fi
    else
      WF_NOTE="window_frame=none:$([ "$RC" = "0" ] && [ "$OK" = "1" ] && echo no_frame || echo run_failed)"
    fi
    [ "${WF_NOTE#window_frame=none}" != "$WF_NOTE" ] && WF_KEEP=()
    echo "WINDOW_FRAME cid=$cid level=$WF $WF_NOTE"
  fi

  echo "RESULT case=$CASE config=$cid rep=$rep rc=$RC ok=$OK med=$MED mean=$MEAN n=$NMEAS excl=$NEXCL wall=$WALL diff_fields=$DIFF_FIELDS"

  # ---- ledger row ----
  # fidelity_label is reserved for PHYSICS-GRAY fidelity; STRICT diffwrf status goes to notes.
  NOTES="config=$cid n_excluded_steps=$NEXCL ref_config=$REF_CONFIG mode=$MODE io=adios2 build=$BUILD_ID"
  if [ "$WF" != "0" ]; then
    NOTES="$NOTES history=$RUN_HISTORY_ARG $WF_NOTE"
  else
    NOTES="$NOTES history=off"
  fi
  [ -n "$PHYS_NL" ] && NOTES="$NOTES physics_namelist=$PHYS_NL"
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
  if [ "$RANKING_ACTIVE" = "1" ]; then
    NOTES="$NOTES restart_target=$CUR_TARGET_MIN restart_used=${RESTART_USED:-0} $FOREIGN_LOAD_NOTE"
    [ "$FUSED_ON" = "1" ] && NOTES="$NOTES fused=1 spinup_s=$(( CUR_TARGET_MIN * 60 ))"
    [ "$POINT_ANCHOR_END" = "0" ] && NOTES="$NOTES drift_gate=off"
  elif [ "$FOREIGN_LOAD" = "1" ]; then
    NOTES="$NOTES $FOREIGN_LOAD_NOTE"
  fi
  export RUN_ID="${CASE}_${cid}_r${rep}_${J}" SRC_DATASET="strict_campaign" JOB_ID="$J" \
    REP_INDEX="$rep" CASE_RAW="$CASE" CASE_BASE="$CASE" CASE_FAMILY="$CASE_FAMILY" \
    HARDWARE_ID="$HARDWARE_ID" BINARY_KIND="pristine" COMPLIANCE_CLASS="STRICT" \
    E_WE="$E_WE" E_SN="$E_SN" E_VERT="$E_VERT" DX="$DX" DY="$DY" TIME_STEP="$DT" \
    MP_PHYSICS="$PHYS_MP_PHYSICS" CU_PHYSICS="$PHYS_CU_PHYSICS" BL_PBL_PHYSICS="$PHYS_BL_PBL_PHYSICS" \
    SF_SFCLAY_PHYSICS="$PHYS_SF_SFCLAY_PHYSICS" SF_SURFACE_PHYSICS="$PHYS_SF_SURFACE_PHYSICS" \
    RA_LW_PHYSICS="$PHYS_RA_LW_PHYSICS" RA_SW_PHYSICS="$PHYS_RA_SW_PHYSICS" \
    RADT="$PHYS_RADT" CUDT="$PHYS_CUDT" \
    HISTORY_INTERVAL="$(if [ "$WF" = "1" ]; then echo "$MIN"; elif [ "$WF" = "2" ]; then echo $(( MIN - MIN / 2 )); else echo 0; fi)" IO_FORM_HISTORY="14" \
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

  # ---- cleanup: keep rsl.out.0000, rsl.error.0000, namelist.input, steps.txt; delete everything else.
  [ "${WTUNE_KEEP_ALL_RSL:-0}" = 1 ] || \
    find . -maxdepth 1 -name 'rsl.*' ! -name 'rsl.out.0000' ! -name 'rsl.error.0000' -delete
  # (window_frame keeps its converted wrfout_d01_<Times>.nc: the orchestrator reads it afterwards)
  WF_FIND_ARGS=()
  for k in "${WF_KEEP[@]}"; do WF_FIND_ARGS+=(! -name "$k"); done
  find . -maxdepth 1 -name 'wrfout_d0*' ! -name "$(basename "${REF_NC:-}")" "${WF_FIND_ARGS[@]}" -exec rm -rf {} +
  # keep the launcher/stdout log of failed runs: smoke job 1074658 (maria1km) timed out with no rsl.* files
  # at all, and deleting wrf.out removed the only evidence of where it stalled
  if [ "$RC" = "0" ] && [ "$OK" = "1" ]; then rm -f wrf.out; fi

  cd "$OUT"
done

# The shared spin-up restarts this job wrote are only needed by this job's candidates (conus25:
# ~12 GB per spin-up; 23 of them had piled up to 251 GB on scc by 2026-09-28). Remove them unless
# WTUNE_KEEP_SPINUP_RESTART=1. Only this job's own "<case>_rankspinup_m<M>_<J>" dirs are touched,
# never a caller-supplied RESTART_SRC_DIR (the legacy reuse path).
if [ "${WTUNE_KEEP_SPINUP_RESTART:-0}" != "1" ]; then
  for d in "$OUT/${CASE}"_rankspinup_m*_"${J}"; do
    [ -d "$d" ] && rm -rf "$d"/wrfrst_d0*_*     # io_form 14 restarts are directories (rm -f left them)
  done
fi

# this job's fresh input copies are not read again (reflinks: freeing them is cheap)
if [ -n "${WTUNE_BP_FRESH_DIR:-}" ] && [ "${WTUNE_KEEP_FRESH_BP:-0}" != "1" ]; then
  rm -rf "$WTUNE_BP_FRESH_DIR"
fi

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

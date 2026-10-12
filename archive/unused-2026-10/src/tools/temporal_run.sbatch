#!/usr/bin/env bash
#SBATCH --job-name=wtune-temporal
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@90
# --signal=B:USR1@90 (job 740 postmortem, 2026-09; see campaign/strict_campaign.sbatch's own
# copy of this directive for the full writeup): Slurm delivers SIGUSR1 to the batch shell 90s
# before --time expires, allocation still fully alive. tools/temporal_validation.py's own
# build_long_sbatch_command/build_samples_sbatch_command/build_timeline_sbatch_command (the
# only builders of this script's sbatch command line) do not pass --signal, so this in-script
# directive is never overridden.
set -uo pipefail
#
# tools/temporal_run.sbatch -- WRF runner for decision 0053 step 2 (offline temporal
# validation): tools/temporal_validation.py's `submit-long`/`submit-samples` build the sbatch
# command line that invokes this script; a human can also `sbatch --export=...` it directly.
#
# WHY A SEPARATE SCRIPT, NOT A campaign/strict_campaign.sbatch MODE: that runner's STRICT/
# MODE=physics loops always force ADIOS2 restart I/O (io_form_restart=14, configs.py's
# _apply_adios2) and have no notion of "start from wrfrst at time T" (no restart=.true. path, no
# per-domain start_* rewrite). Retrofitting that in place would mean a new MODE branch touching
# namelist-edit calls, input staging AND the disk-quota/Thompson-table machinery of an
# already-dense, heavily relied-on 1000+ line script, for a run shape (netCDF restart, short
# restart-sampled windows) that script's own design deliberately excludes (see its ADIOS2 module
# docstring). A thin wrapper that reuses only campaign/sites/$SITE.sh's site_env/site_launch --
# the same site-abstraction layer strict_campaign.sbatch itself is built on -- is smaller and
# cannot regress that script's own (separately owned, untouched) behaviour at all.
#
# SCOPE LIMIT (accepted for this tool): unlike strict_campaign.sbatch, this script has no
# disk-quota guard and no Thompson-lookup-table shared cache. Both runs here are far smaller than
# the ADIOS2 history campaigns those guards were built for (a LONG run writes only a handful of
# netCDF restart snapshots, no BP history at all; a SAMPLES run is a ~30-step window with history
# off), so the risk they defend against is much smaller here -- but it is not zero, and this is
# unverified without a real cluster run (see tools/temporal_validation.py's final report).
#
# CASE-INPUT SCOPE LIMIT: netCDF wrfinput/wrfbdy only (bp_dir is never consulted) -- matches scc,
# the only site decision 0053 step 2 targets, where bp_dir is empty for every case already.
#
# Env: SITE (default scc), CASE (required), TV_MODE=long|samples|timeline (required).
#   TV_MODE=long:     HOURS, EVERY required.
#   TV_MODE=samples:  LONG_RUN_DIR, RESTART_HOURS, CONFIG_FILE required; STEPS (default 30),
#                      TIMEOUT_S (default 1200) optional.
#   TV_MODE=timeline: HOURS, RUN_MINUTES, EVERY, CONFIG_FILE required (state_timeline.py
#                      training-data extension, added on top of the LONG-run shape: same
#                      cold-start/full-length/netCDF-restart run, looped over CONFIG_FILE's
#                      up to-3 configs -- SAMPLES' own CONFIG_FILE loop convention, not LONG's
#                      single-anchor shape -- PLUS an auxiliary 2-D state history stream).
#                      RUN_MINUTES (job 587/588 postmortem) is the exact namelist run length in
#                      minutes -- HOURS is still required and still passed through to
#                      edit-namelist-timeline, but RUN_MINUTES is authoritative (see
#                      tools/temporal_validation.py's build_timeline_sbatch_command docstring).
#                      AUX_INTERVAL_MIN (default 30), AUX_STREAM (default: auto-detect a free
#                      stream, preferring 23), FRAMES_PER_AUXHIST (default 2000), AUX_FIELDS
#                      (default: `$TS default-fields`) optional. Backward compatible: long/samples
#                      callers and their env-var contracts are untouched by this addition.

SITE=${SITE:-scc}
export SITE
CAMPAIGN=${CAMPAIGN:-$HOME/wtune/campaign}
export CAMPAIGN
TOOLS=${TOOLS:-$HOME/wtune/tools}
source "$CAMPAIGN/sites/$SITE.sh"
PY="pixi run --manifest-path $PIXI_ENV/pixi.toml python"
TV="$PY $TOOLS/temporal_validation.py"
TS="$PY $TOOLS/state_timeline.py"   # TV_MODE=timeline plumbing only (aux-stream/iofields/namelist)

: "${CASE:?CASE env var is required}"
: "${TV_MODE:?TV_MODE env var is required (long|samples)}"

mkdir -p "$OUT"
site_env
J=$SLURM_JOB_ID
ALLOC_NODES=${SLURM_JOB_NUM_NODES:-1}

# ---- node hygiene: orphan wrf.exe reap (scc prte/pgid incident, 2026-09) -- see
# campaign/strict_campaign.sbatch's own copy of this block for the incident writeup and
# campaign/node_hygiene.sh for the per-node reap logic dispatched here (duplicated in
# miniature here rather than sourced, same rule this file's configure_omp_env already
# follows). This script writes no notes/metadata column anywhere (SAMPLES_LEDGER has none,
# long/timeline write no ledger row at all besides echoed RESULT lines), so a dirty node is
# logged only, never recorded in a ledger.
run_node_hygiene () {
  # run_node_hygiene <mode> <timeout_s>   mode: reap | census | reap+census. Sets REAP_N to
  # the total orphans reaped across every node, this call only.
  local mode="$1" tmo="$2" out
  site_hygiene "$mode"
  out=$(timeout "$tmo" "${SITE_HYGIENE_CMD[@]}" 2>&1) || true
  [ -n "$out" ] && echo "$out"
  REAP_N=$(printf '%s\n' "$out" | awk '/^REAPED_ORPHANS /{for(i=1;i<=NF;i++){split($i,kv,"="); if(kv[1]=="n") s+=kv[2]}} END{print s+0}')
}

# job 740 postmortem (2026-09; see campaign/strict_campaign.sbatch's own copy of this comment
# for the full writeup): a plain foreground `timeout ... "${SITE_LAUNCH_CMD[@]}"` defers every
# trap until that child exits, so Slurm's TIMEOUT SIGTERM was silently swallowed for the full
# 30s KillWait and SIGKILL hit before any trap ran -- every rank survived. Fix: launch each run
# in the BACKGROUND and `wait` on it explicitly (a trapped signal interrupts a blocked `wait`
# immediately, verified locally), plus `#SBATCH --signal=B:USR1@90` above for an early-warning
# signal 90s before the hard time limit, while the allocation is still alive.
WRF_PID=""
CUR_RUN_ID=""

stop_run_and_reap () {
  # stop_run_and_reap <kill_wait_s> <reap_tmo_s> -- see strict_campaign.sbatch's own copy for
  # the full contract. Safe to call with nothing in flight: the kill step is then a no-op.
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
  # CUR_RUN_ID is set only for the window between launching a run in the background and
  # capturing its RC (all three TV_MODE loops below). TV_MODE=samples is the only one of the
  # three with a ledger to write into (SAMPLES_LEDGER); long/timeline have no ledger row at
  # all (see the node-hygiene header comment above), so they are logged only, same as the
  # existing "missing restart file" row shape a few lines below in the samples loop.
  local sig="$1"
  [ -z "$CUR_RUN_ID" ] && return 0
  echo "INTERRUPTED case=$CASE tv_mode=$TV_MODE run=$CUR_RUN_ID reason=preempt_sig${sig}" >&2
  if [ "$TV_MODE" = samples ] && [ -n "${SAMPLES_LEDGER:-}" ]; then
    echo "$RESTART_HOURS,$cid,$ranks,$omp,$nx,$ny,$numtiles,$dist,interrupted,0," >> "$SAMPLES_LEDGER"
  fi
  CUR_RUN_ID=""
}

# NODE_HYGIENE_TRAP_DONE: USR1's own `exit 0` re-enters this function via the EXIT trap -- the
# guard makes that second entry a fast no-op (see strict_campaign.sbatch's own copy for the
# locally-verified transcript of this exact double-fire).
NODE_HYGIENE_TRAP_DONE=0
node_hygiene_trap () {
  local sig="$1"
  [ "$NODE_HYGIENE_TRAP_DONE" = "1" ] && return 0
  NODE_HYGIENE_TRAP_DONE=1
  if [ "$sig" = "USR1" ]; then
    stop_run_and_reap 10 30 >&2 || true   # T-90s, allocation alive: generous budget
  else
    stop_run_and_reap 3 15 >&2 || true    # real TERM/INT/EXIT: KillWait=30s total
  fi
  write_interrupted_row "$sig" >&2 || true
  [ "$sig" != "EXIT" ] && exit 0
}
trap 'node_hygiene_trap EXIT' EXIT
trap 'node_hygiene_trap TERM' TERM
trap 'node_hygiene_trap INT' INT
trap 'node_hygiene_trap USR1' USR1

# preflight (design point 1): the allocation is held before any WRF launch, so any wrf.exe
# found here is an orphan from a dead job -- reap is safe. Census still runs and logs
# NODE_DIRTY lines (design point 1b); with no notes/metadata column in this script's own
# output (see above) there is nothing to fold the result into, so it is logged only.
run_node_hygiene reap+census 90

echo "job $J case $CASE site $SITE tv_mode $TV_MODE nodes $SLURM_JOB_NODELIST"

eval "$($TV case-metadata "$CASE")"
read -ra INPUT_FILES_ARR <<< "$INPUT_FILES"

# ---- per-run OpenMP env (same rule as campaign/strict_campaign.sbatch's configure_omp_env,
# duplicated in miniature here rather than sourced -- that function lives inside
# strict_campaign.sbatch itself, not a shared file) ----------------------------------------
configure_omp_env () {
  local omp="$1" blocktime="$2"
  export OMP_NUM_THREADS="$omp"
  if [ "$SITE" = "scc" ]; then
    unset KMP_BLOCKTIME
    if [ "$omp" -gt 1 ]; then export OMP_PLACES=cores OMP_PROC_BIND=close
    else unset OMP_PLACES OMP_PROC_BIND; fi
  else
    if [ "$blocktime" = "default" ]; then unset KMP_BLOCKTIME
    else export KMP_BLOCKTIME=$blocktime; fi
  fi
}

link_static_run_files () {
  # everything main/wrf.exe's own run/ directory ships EXCEPT the case-specific files this
  # function's caller links separately -- identical filter to campaign/strict_campaign.sbatch's.
  local f b
  for f in "$W"/run/*; do
    b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;;
      *) [ -e "$b" ] || ln -sf "$f" "$b" ;; esac
  done
}

link_case_input_files () {
  local f
  for f in "${INPUT_FILES_ARR[@]}"; do
    [ -n "$f" ] && ln -sf "$PREP_DIR/$f" "$f"
  done
}

if [ "$TV_MODE" = long ]; then
  : "${HOURS:?HOURS env var is required for TV_MODE=long}"
  : "${EVERY:?EVERY env var is required for TV_MODE=long}"
  eval "$($TV anchor-config "$CASE")"

  RUNDIR="$OUT/temporal_${CASE}_long_${J}"
  rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"
  link_case_input_files
  link_static_run_files
  for ((d = 1; d <= MAX_DOM; d++)); do
    df=$(printf "d%02d" "$d")
    ln -sf "$PREP_DIR/wrfinput_${df}" "wrfinput_${df}"
  done
  ln -sf "$PREP_DIR/wrfbdy_d01" wrfbdy_d01

  cp "$NAMELIST_TEMPLATE" ./namelist.input.orig
  $TV edit-namelist-long ./namelist.input.orig ./namelist.input \
    --hours "$HOURS" --every "$EVERY" --nproc-x "$NX" --nproc-y "$NY" --numtiles "$NUMTILES" \
    --ranks "$RANKS" --max-dom "$MAX_DOM"
  rm -f ./namelist.input.orig
  ln -sf "$W/main/wrf.exe" ./wrf.exe

  configure_omp_env "$OMP" "$BLOCKTIME"
  site_launch "$RANKS" "$((RANKS / ALLOC_NODES))" "$OMP" "$DIST"
  # HOURS worth of wall time plus the same 3x-per_run watchdog multiplier
  # tools/scc_sweep.py's own estimate_time() uses, floored at 30 min.
  TIME_LIMIT_S=${TIME_LIMIT_S:-$(( HOURS * 3600 * 3 > 1800 ? HOURS * 3600 * 3 : 1800 ))}
  # background + explicit wait -- see the node-hygiene block's job-740 comment above for why.
  CUR_RUN_ID="${CASE}_long_${J}"
  T0=$(date +%s.%N)
  timeout "$TIME_LIMIT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 < /dev/null &
  WRF_PID=$!
  wait "$WRF_PID"
  RC=$?
  # RC>128 here is a narrow-race safety net only -- node_hygiene_trap already logged the
  # interruption and exited before this line otherwise (see strict_campaign.sbatch's copy).
  WRF_PID=""
  CUR_RUN_ID=""
  T1=$(date +%s.%N)
  WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

  # post-run reap (design point 2): mpirun returned (normal exit or watchdog TIMEOUT) --
  # reap any wrf.exe it left behind before the next run starts.
  run_node_hygiene reap 60
  [ "${REAP_N:-0}" -gt 0 ] && echo "WARN orphans_after_run cfg=long n=$REAP_N"

  OK=0
  if [ -f rsl.error.0000 ] && grep -qa "SUCCESS COMPLETE WRF" rsl.error.0000; then OK=1; fi
  echo "RESULT case=$CASE tv_mode=long hours=$HOURS every=$EVERY rc=$RC ok=$OK wall=$WALL"

  $TV parse-timing rsl.out.0000 > "$OUT/temporal_${CASE}_long_${J}_timing.csv"
  N_RST=$(ls -1 wrfrst_d01_* 2>/dev/null | wc -l)
  echo "LONG_RUN_DIR=$RUNDIR"
  echo "wrfrst_d01 files written: $N_RST"

  find . -maxdepth 1 -name 'rsl.*' ! -name 'rsl.out.0000' ! -name 'rsl.error.0000' -delete
  [ "$RC" = "0" ] && [ "$OK" = "1" ] && rm -f wrf.out
  echo "=== done (TV_MODE=long) case=$CASE run_dir=$RUNDIR"

elif [ "$TV_MODE" = samples ]; then
  : "${LONG_RUN_DIR:?LONG_RUN_DIR env var is required for TV_MODE=samples}"
  : "${RESTART_HOURS:?RESTART_HOURS env var is required for TV_MODE=samples}"
  : "${CONFIG_FILE:?CONFIG_FILE env var is required for TV_MODE=samples}"
  STEPS=${STEPS:-30}
  TIMEOUT_S=${TIMEOUT_S:-1200}
  export CONFIG_FILE

  RESTART_TS=""
  if [ "$RESTART_HOURS" != "0" ]; then
    RESTART_TS=$($TV restart-timestamp "$CASE" "$RESTART_HOURS")
  fi

  SAMPLES_LEDGER="$OUT/temporal_${CASE}_samples_h${RESTART_HOURS}_${J}.csv"
  echo "restart_hours,config_id,ranks,omp,nx,ny,numtiles,dist,rc,ok,steps" > "$SAMPLES_LEDGER"

  while IFS=$'\t' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy \
                          ntx nty tsx tsy tstrat src; do
    [ "$cid" = "config_id" ] && continue
    [ -z "$cid" ] && continue

    RUNDIR="$OUT/temporal_${CASE}_s${RESTART_HOURS}_${cid}_${J}"
    rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"
    link_case_input_files
    link_static_run_files
    ln -sf "$PREP_DIR/wrfbdy_d01" wrfbdy_d01

    if [ -z "$RESTART_TS" ]; then
      # t0: normal cold start, same wrfinput_* linking the LONG run itself uses.
      for ((d = 1; d <= MAX_DOM; d++)); do
        df=$(printf "d%02d" "$d")
        ln -sf "$PREP_DIR/wrfinput_${df}" "wrfinput_${df}"
      done
    else
      for ((d = 1; d <= MAX_DOM; d++)); do
        rf=$(printf "wrfrst_d%02d_%s" "$d" "$RESTART_TS")
        if [ ! -e "$LONG_RUN_DIR/$rf" ]; then
          echo "ERROR: missing restart file $LONG_RUN_DIR/$rf for T=+${RESTART_HOURS}h -- was the LONG run's --every a divisor of this offset, and did it reach this far?" >&2
          echo "$RESTART_HOURS,$cid,$ranks,$omp,$nx,$ny,$numtiles,$dist,,0," >> "$SAMPLES_LEDGER"
          cd "$OUT"
          continue 2
        fi
        ln -sf "$LONG_RUN_DIR/$rf" "$rf"
      done
    fi

    cp "$NAMELIST_TEMPLATE" ./namelist.input.orig
    $TV edit-namelist-sample ./namelist.input.orig ./namelist.input \
      --case "$CASE" --restart-hours "$RESTART_HOURS" --steps "$STEPS" \
      --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" --ranks "$ranks" --max-dom "$MAX_DOM"
    rm -f ./namelist.input.orig
    ln -sf "$W/main/wrf.exe" ./wrf.exe

    configure_omp_env "$omp" "$blocktime"
    site_launch "$ranks" "$((ranks / ALLOC_NODES))" "$omp" "$dist"
    # background + explicit wait -- see the node-hygiene block's job-740 comment above for why.
    CUR_RUN_ID="${CASE}_s${RESTART_HOURS}_${cid}_${J}"
    T0=$(date +%s.%N)
    timeout "$TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 < /dev/null &
    WRF_PID=$!
    wait "$WRF_PID"
    RC=$?
    # RC>128 here is a narrow-race safety net only -- see the TV_MODE=long block's own copy.
    WRF_PID=""
    CUR_RUN_ID=""
    T1=$(date +%s.%N)
    WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

    # post-run reap (design point 2): see the TV_MODE=long block's own copy above for why.
    run_node_hygiene reap 60
    [ "${REAP_N:-0}" -gt 0 ] && echo "WARN orphans_after_run cfg=$cid n=$REAP_N"

    OK=0
    if [ -f rsl.error.0000 ] && grep -qa "SUCCESS COMPLETE WRF" rsl.error.0000; then OK=1; fi
    STEPS_FLAT=""
    [ "$RC" = "0" ] && [ "$OK" = "1" ] && STEPS_FLAT=$($TV parse-timing rsl.out.0000 --flatten)
    echo "RESULT case=$CASE tv_mode=samples T=+${RESTART_HOURS}h config=$cid rc=$RC ok=$OK wall=$WALL"
    echo "$RESTART_HOURS,$cid,$ranks,$omp,$nx,$ny,$numtiles,$dist,$RC,$OK,$STEPS_FLAT" >> "$SAMPLES_LEDGER"

    find . -maxdepth 1 -name 'rsl.*' ! -name 'rsl.out.0000' ! -name 'rsl.error.0000' -delete
    [ "$RC" = "0" ] && [ "$OK" = "1" ] && rm -f wrf.out
    cd "$OUT"
  done < <($PY "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null)

  echo "SAMPLES_LEDGER=$SAMPLES_LEDGER"
  echo "=== done (TV_MODE=samples) case=$CASE T=+${RESTART_HOURS}h"

elif [ "$TV_MODE" = timeline ]; then
  : "${HOURS:?HOURS env var is required for TV_MODE=timeline}"
  # RUN_MINUTES (job 587/588 postmortem correction 5) is the authoritative namelist run length --
  # HOURS/HOURS*60 cannot represent a sub-hour case (isc21: 30 min) or a date-span-defined case
  # (lafe: all run_* keys 0) correctly; see tools/temporal_validation.py's
  # build_timeline_sbatch_command/resolve_run_minutes/read_run_length_hours docstrings for why.
  # HOURS itself is still required and still passed to edit-namelist-timeline (kept for any other
  # consumer / as a sanity cross-check), but --run-minutes is what actually sets the namelist.
  : "${RUN_MINUTES:?RUN_MINUTES env var is required for TV_MODE=timeline}"
  : "${EVERY:?EVERY env var is required for TV_MODE=timeline}"
  : "${CONFIG_FILE:?CONFIG_FILE env var is required for TV_MODE=timeline}"
  export CONFIG_FILE
  AUX_INTERVAL_MIN=${AUX_INTERVAL_MIN:-30}
  FRAMES_PER_AUXHIST=${FRAMES_PER_AUXHIST:-2000}
  AUX_FIELDS=${AUX_FIELDS:-$($TS default-fields)}

  # Aux-stream pick + Registry field validation are done ONCE per job (same case/build for every
  # config this job loops over below) against the untouched template -- not per-config.
  AUX_STREAM=${AUX_STREAM:-$($TS pick-aux-stream "$NAMELIST_TEMPLATE")}
  VALIDATED_FIELDS=$($TS check-iofields --wrf-tree "$W" --fields "$AUX_FIELDS")
  if [ -z "$VALIDATED_FIELDS" ]; then
    echo "ERROR: TV_MODE=timeline: every requested aux field was dropped by check-iofields (wrf-tree=$W, fields=$AUX_FIELDS) -- nothing to write to the aux stream" >&2
    exit 1
  fi
  echo "TIMELINE aux_stream=$AUX_STREAM aux_interval_min=$AUX_INTERVAL_MIN fields=$VALIDATED_FIELDS"

  while IFS=$'\t' read -r cid ranks omp nx ny numtiles dist blocktime patchx patchy \
                          ntx nty tsx tsy tstrat src; do
    [ "$cid" = "config_id" ] && continue
    [ -z "$cid" ] && continue

    RUNDIR="$OUT/temporal_${CASE}_timeline_${cid}_${J}"
    rm -rf "$RUNDIR"; mkdir -p "$RUNDIR"; cd "$RUNDIR"
    link_case_input_files
    link_static_run_files
    for ((d = 1; d <= MAX_DOM; d++)); do
      df=$(printf "d%02d" "$d")
      ln -sf "$PREP_DIR/wrfinput_${df}" "wrfinput_${df}"
    done
    ln -sf "$PREP_DIR/wrfbdy_d01" wrfbdy_d01

    $TS write-iofields ./iofields_timeline.txt --stream "$AUX_STREAM" --fields "$VALIDATED_FIELDS"

    cp "$NAMELIST_TEMPLATE" ./namelist.input.orig
    $TS edit-namelist-timeline ./namelist.input.orig ./namelist.input \
      --case "$CASE" --hours "$HOURS" --run-minutes "$RUN_MINUTES" --every "$EVERY" \
      --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" --ranks "$ranks" --max-dom "$MAX_DOM" \
      --aux-stream "$AUX_STREAM" --aux-interval "$AUX_INTERVAL_MIN" \
      --frames-per-auxhist "$FRAMES_PER_AUXHIST" --iofields-filename iofields_timeline.txt \
      || { echo "RESULT case=$CASE tv_mode=timeline config=$cid rc=90 ok=0 reason=namelist_edit_failed"; cd "$OUT"; continue; }
    # job 464 launched WRF with no namelist.input after this step failed; never do that again
    [ -s ./namelist.input ] || { echo "RESULT case=$CASE tv_mode=timeline config=$cid rc=91 ok=0 reason=no_namelist"; cd "$OUT"; continue; }
    rm -f ./namelist.input.orig
    ln -sf "$W/main/wrf.exe" ./wrf.exe

    configure_omp_env "$omp" "$blocktime"
    site_launch "$ranks" "$((ranks / ALLOC_NODES))" "$omp" "$dist"
    TIME_LIMIT_S=${TIME_LIMIT_S:-$(( HOURS * 3600 * 3 > 1800 ? HOURS * 3600 * 3 : 1800 ))}
    # background + explicit wait -- see the node-hygiene block's job-740 comment above for why.
    CUR_RUN_ID="${CASE}_timeline_${cid}_${J}"
    T0=$(date +%s.%N)
    timeout "$TIME_LIMIT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 < /dev/null &
    WRF_PID=$!
    wait "$WRF_PID"
    RC=$?
    # RC>128 here is a narrow-race safety net only -- see the TV_MODE=long block's own copy.
    WRF_PID=""
    CUR_RUN_ID=""
    T1=$(date +%s.%N)
    WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

    # post-run reap (design point 2): see the TV_MODE=long block's own copy above for why.
    run_node_hygiene reap 60
    [ "${REAP_N:-0}" -gt 0 ] && echo "WARN orphans_after_run cfg=$cid n=$REAP_N"

    OK=0
    if [ -f rsl.error.0000 ] && grep -qa "SUCCESS COMPLETE WRF" rsl.error.0000; then OK=1; fi
    echo "RESULT case=$CASE tv_mode=timeline config=$cid hours=$HOURS every=$EVERY rc=$RC ok=$OK wall=$WALL"

    $TV parse-timing rsl.out.0000 > "$OUT/temporal_${CASE}_timeline_${cid}_${J}_timing.csv"
    N_RST=$(ls -1 wrfrst_d01_* 2>/dev/null | wc -l)
    N_AUX=$(ls -1 "auxhist${AUX_STREAM}_d01_"* 2>/dev/null | wc -l)
    echo "TIMELINE_RUN_DIR config=$cid dir=$RUNDIR aux_stream=$AUX_STREAM aux_frames=$N_AUX wrfrst_d01_files=$N_RST"

    find . -maxdepth 1 -name 'rsl.*' ! -name 'rsl.out.0000' ! -name 'rsl.error.0000' -delete
    [ "$RC" = "0" ] && [ "$OK" = "1" ] && rm -f wrf.out
    cd "$OUT"
  done < <($PY "$CAMPAIGN/configs.py" "$CASE" 2>/dev/null)

  echo "=== done (TV_MODE=timeline) case=$CASE hours=$HOURS aux_stream=$AUX_STREAM"

else
  echo "TV_MODE must be 'long', 'samples' or 'timeline', got '$TV_MODE'" >&2
  exit 1
fi

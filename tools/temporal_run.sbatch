#!/usr/bin/env bash
#SBATCH --job-name=wtune-temporal
#SBATCH --time=02:00:00
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
#   TV_MODE=timeline: HOURS, EVERY, CONFIG_FILE required (state_timeline.py training-data
#                      extension, added on top of the LONG-run shape: same cold-start/full-length/
#                      netCDF-restart run, looped over CONFIG_FILE's up to-3 configs -- SAMPLES'
#                      own CONFIG_FILE loop convention, not LONG's single-anchor shape -- PLUS an
#                      auxiliary 2-D state history stream). AUX_INTERVAL_MIN (default 30),
#                      AUX_STREAM (default: auto-detect a free stream, preferring 23),
#                      FRAMES_PER_AUXHIST (default 2000), AUX_FIELDS (default:
#                      `$TS default-fields`) optional. Backward compatible: long/samples callers
#                      and their env-var contracts are untouched by this addition.

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
  T0=$(date +%s.%N)
  timeout "$TIME_LIMIT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1
  RC=$?
  T1=$(date +%s.%N)
  WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

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
    T0=$(date +%s.%N)
    timeout "$TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1
    RC=$?
    T1=$(date +%s.%N)
    WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

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
      --case "$CASE" --hours "$HOURS" --every "$EVERY" \
      --nproc-x "$nx" --nproc-y "$ny" --numtiles "$numtiles" --ranks "$ranks" --max-dom "$MAX_DOM" \
      --aux-stream "$AUX_STREAM" --aux-interval "$AUX_INTERVAL_MIN" \
      --frames-per-auxhist "$FRAMES_PER_AUXHIST" --iofields-filename iofields_timeline.txt
    rm -f ./namelist.input.orig
    ln -sf "$W/main/wrf.exe" ./wrf.exe

    configure_omp_env "$omp" "$blocktime"
    site_launch "$ranks" "$((ranks / ALLOC_NODES))" "$omp" "$dist"
    TIME_LIMIT_S=${TIME_LIMIT_S:-$(( HOURS * 3600 * 3 > 1800 ? HOURS * 3600 * 3 : 1800 ))}
    T0=$(date +%s.%N)
    timeout "$TIME_LIMIT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1
    RC=$?
    T1=$(date +%s.%N)
    WALL=$(python3 -c "print('%.1f' % ($T1 - $T0))")

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

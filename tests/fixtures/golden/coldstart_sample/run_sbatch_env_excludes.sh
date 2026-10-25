#!/usr/bin/env bash
#SBATCH --job-name=coldstart-sample
#SBATCH --partition=compute
#SBATCH --exclude=head,work5,work7,work9
#SBATCH -N 1 --exclusive --ntasks-per-node=72
#SBATCH --time=01:10:00
#SBATCH --signal=B:USR1@90
set -uo pipefail

# estimated 35 min (wps ~20min + 1 windows x ~15min/window); --time is 2x that

PLAN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOL=/opt/wtune/src/tools/coldstart_sample.py
SITE=scc
CAMPAIGN=${CAMPAIGN:-/opt/wtune/campaign}
source "$CAMPAIGN/sites/$SITE.sh"   # site_env / site_launch / site_hygiene -- reused, not reimplemented
source $HOME/wtune-scc/env.sh
export PATH=$NETCDF/bin:$PATH
ulimit -s unlimited
site_env
ALLOC_NODES=${SLURM_JOB_NUM_NODES:-1}
STATUS="$PLAN/windows_status.txt"
: > "$STATUS"

run_hygiene() {
  site_hygiene "$1"
  timeout 90 "${SITE_HYGIENE_CMD[@]}" 2>&1 || true
}

WRF_PID=""
CUR_WINDOW=""
kill_and_reap() {
  if [ -n "$WRF_PID" ] && kill -0 "$WRF_PID" 2>/dev/null; then
    kill -TERM "$WRF_PID" 2>/dev/null || true; sleep 3
    kill -0 "$WRF_PID" 2>/dev/null && kill -KILL "$WRF_PID" 2>/dev/null || true
  fi
  pkill -u "$USER" -x real.exe 2>/dev/null; pkill -u "$USER" -x wrf.exe 2>/dev/null
  run_hygiene reap
}

SIGNAL_TRAP_DONE=0
on_signal() {
  local sig="$1"
  [ "$SIGNAL_TRAP_DONE" = 1 ] && return 0
  SIGNAL_TRAP_DONE=1
  kill_and_reap
  if [ -n "$CUR_WINDOW" ]; then
    echo "WINDOW $CUR_WINDOW rc=1 ok=0 INTERRUPTED sig=$sig" >> "$STATUS"
  fi
  exit 3
}
on_exit() {
  [ "$SIGNAL_TRAP_DONE" = 1 ] && return 0
  run_hygiene reap
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap on_exit EXIT

run_hygiene reap+census

cd "$PLAN/wps"
GEO_CACHE="${WTUNE_GEO_CACHE:-$HOME/wtune-runs/geo_cache}/$(sed -n '/&share/,/^\//p;/&geogrid/,/^\//p' namelist.wps | grep -vE 'start_date|end_date|interval_seconds' | sha1sum | cut -c1-16)"
GEO_LOCKED=0
if command -v flock >/dev/null 2>&1 && mkdir -p "$(dirname "$GEO_CACHE")" 2>/dev/null && ( : >> "$GEO_CACHE.lock" ) 2>/dev/null; then
  exec 9>>"$GEO_CACHE.lock" && flock -w 900 9 && GEO_LOCKED=1 && echo "=== geogrid: lock $GEO_CACHE.lock held ($(date -u +%FT%TZ)) ==="
fi
GEO_CACHED=1; for f in geo_em.d01.nc; do [ -s "$GEO_CACHE/$f" ] || GEO_CACHED=0; done
if [ $GEO_CACHED = 1 ]; then
  echo "=== geogrid: cached $GEO_CACHE ($(date -u +%FT%TZ)) ==="
  for f in geo_em.d01.nc; do ln -sf "$GEO_CACHE/$f" .; done
  echo 'Successful completion of geogrid (cached)' > geogrid.log.cached; rc=0
else
echo "=== geogrid.exe ($(date -u +%FT%TZ)) ==="
if nm -D --undefined-only ./geogrid.exe 2>/dev/null | grep -qi mpi_init; then mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core ./geogrid.exe; else ./geogrid.exe; fi
rc=$?; pkill -u "$USER" -x geogrid.exe 2>/dev/null
fi
GEO_EM_EXPECTED=(geo_em.d01.nc)
GEO_EM_MISSING=()
for f in "${GEO_EM_EXPECTED[@]}"; do [ -f "$f" ] || GEO_EM_MISSING+=("$f"); done
if [ $rc -ne 0 ] || [ ${#GEO_EM_MISSING[@]} -gt 0 ] || ! grep -q 'Successful completion' geogrid.log* 2>/dev/null; then
  echo "FATAL geogrid rc=$rc missing=${GEO_EM_MISSING[*]:-none}"; exit 1
fi
if [ $GEO_CACHED = 0 ]; then mkdir -p "$GEO_CACHE" && cp -f geo_em.d01.nc "$GEO_CACHE/" 2>/dev/null; fi
if [ $GEO_LOCKED = 1 ]; then flock -u 9 2>/dev/null; exec 9>&-; fi

if [ 0 = 0 ]; then
echo "=== link_grib.csh ($(date -u +%FT%TZ)) ==="
rm -f GRIBFILE.*
if [ -s "$PLAN/grib_manifest.txt" ]; then
  mapfile -t GRIB_FILES < <(sed "s#^#$PLAN/grib/#" "$PLAN/grib_manifest.txt")
else
  shopt -s nullglob
  GRIB_FILES=("$PLAN"/grib/*.grib2 "$PLAN"/grib/*.grib "$PLAN"/grib/*.grb)
  shopt -u nullglob
fi
if [ ${#GRIB_FILES[@]} -eq 0 ]; then
  echo "FATAL: no forcing files in $PLAN/grib -- run 'coldstart_sample.py fetch --plan $PLAN' first"; exit 1
fi
./link_grib.csh "${GRIB_FILES[@]}"
[ $? -ne 0 ] && { echo "FATAL link_grib.csh failed"; exit 1; }
fi

run_window() {
  local k="$1" prefix="$2" vtable="$3"; shift 3
  local n_ungrib="$1"; shift
  local expected=(); for ((i = 0; i < n_ungrib; i++)); do expected+=("$1"); shift; done
  local n_metem="$1"; shift
  local expected_met=(); for ((i = 0; i < n_metem; i++)); do expected_met+=("$1"); shift; done
  local rc missing=() missing_met=() real_path
  CUR_WINDOW="$k"
  LAST_REAL_PATH=""
  LAST_WRF_STATUS="skipped"
  LAST_ML_STATUS="off"
  LAST_ML_TARGETS=""
  export OMP_NUM_THREADS=1
  cd "$PLAN/wps" || return 1
  echo "=== window $k: wps namelist + ungrib (prefix=$prefix vtable=$vtable) ($(date -u +%FT%TZ)) ==="
  python3 "$TOOL" fix-wps-namelist --plan "$PLAN" --win "$k" || return 1
  if [ "$vtable" = none ]; then rc=0
  else
  if [ -e "$vtable" ]; then ln -sf "$vtable" Vtable
  else echo "window $k: $vtable not staged in $PLAN/wps (is \$WPS_DIR set?)" >&2; return 1; fi
  rm -f "${prefix}:"*
  ./ungrib.exe
  rc=$?; pkill -u "$USER" -x ungrib.exe 2>/dev/null
  fi
  for f in "${expected[@]}"; do [ -f "$f" ] || missing+=("$f"); done
  if [ $rc -ne 0 ] || [ ${#missing[@]} -gt 0 ]; then
    echo "window $k: ungrib rc=$rc missing=${missing[*]:-none}" >&2; return 1
  fi

  echo "=== window $k: metgrid.exe ($(date -u +%FT%TZ)) ==="
  if nm -D --undefined-only ./metgrid.exe 2>/dev/null | grep -qi mpi_init; then mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core ./metgrid.exe; else ./metgrid.exe; fi
  rc=$?; pkill -u "$USER" -x metgrid.exe 2>/dev/null
  for f in "${expected_met[@]}"; do [ -f "$f" ] || missing_met+=("$f"); done
  if [ $rc -ne 0 ] || [ ${#missing_met[@]} -gt 0 ] || ! grep -q 'Successful completion' metgrid.log* 2>/dev/null; then
    echo "window $k: metgrid rc=$rc missing=${missing_met[*]:-none}" >&2; return 1
  fi

  echo "=== window $k: real.exe ($(date -u +%FT%TZ)) ==="
  cd "$PLAN/win_${k}/real" || return 1
  rm -f met_em.d0*.*.nc
  ln -sf "$PLAN"/wps/met_em.d0*.*.nc .
  for f in met_em.d0*.*.nc; do [ "$f" = "${f//:/_}" ] || ln -sf "$f" "${f//:/_}"; done
  python3 "$TOOL" fix-real-namelist --plan "$PLAN" --win "$k" || return 1
  timeout 600 mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core ./real.exe
  rc=$?; pkill -u "$USER" -x real.exe 2>/dev/null
  real_path=decomp
  if [ $rc -eq 124 ] || [ ! -f wrfinput_d01 ] || [ ! -f wrfbdy_d01 ]; then
    echo "window $k: real.exe decomposed attempt rc=$rc -- retrying serial (nproc_x=nproc_y=1)" >&2
    python3 "$TOOL" fix-real-namelist --plan "$PLAN" --win "$k" --serial || return 1
    timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core ./real.exe
    rc=$?; pkill -u "$USER" -x real.exe 2>/dev/null
    real_path=serial
  fi
  LAST_REAL_PATH="$real_path"
  [ $rc -ne 0 ] && return 1
  [ -f wrfinput_d01 ] && [ -f wrfbdy_d01 ] || return 1

  echo "=== window $k: wrf.exe ($(date -u +%FT%TZ)) ==="
  cd "$PLAN/win_${k}/wrf" || return 1
  ln -sf ../real/wrfinput_d0* .
  ln -sf ../real/wrfbdy_d01 .
  shopt -s nullglob; WRFLOWINP=(../real/wrflowinp_d0*); shopt -u nullglob
  [ ${#WRFLOWINP[@]} -gt 0 ] && ln -sf ../real/wrflowinp_d0* .
  TBL_CACHE="${WTUNE_TABLE_CACHE:-$HOME/wtune-runs/thompson_tables}/$(sha1sum < "$(readlink -f "$WRF_TREE/main/wrf.exe")" | cut -c1-16)"
  shopt -s nullglob; for f in "$TBL_CACHE"/qr_acr_q*.dat "$TBL_CACHE"/freezeH2O.dat; do [ -e "$f" ] || continue; [ -e "$(basename "$f")" ] || ln -s "$f" .; done; shopt -u nullglob
  python3 "$TOOL" fix-wrf-namelist --plan "$PLAN" --win "$k" --nproc-x 9 --nproc-y 8 --numtiles 1 || return 1
  export OMP_NUM_THREADS=1
  unset OMP_PLACES OMP_PROC_BIND
  site_launch 72 72 1 block
  timeout 900 "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1 &
  WRF_PID=$!
  ( po=0; while kill -0 "$WRF_PID" 2>/dev/null; do if grep -lqs 'FATAL CALLED' rsl.error.* 2>/dev/null; then sleep 30; kill -TERM "$WRF_PID" 2>/dev/null; sleep 20; kill -KILL "$WRF_PID" 2>/dev/null; echo "window $k: WRF FATAL CALLED -- launch stopped" >&2; break; fi; if grep -qs 'Problem opening' rsl.error.0000 && ! grep -qs 'Timing for main' rsl.error.0000; then po=$((po+10)); if [ $po -ge 120 ]; then kill -TERM "$WRF_PID" 2>/dev/null; sleep 20; kill -KILL "$WRF_PID" 2>/dev/null; echo "window $k: WRF stuck at init after 'Problem opening' -- launch stopped" >&2; break; fi; fi; sleep 10; done ) &
  FATAL_WATCH_PID=$!
  wait "$WRF_PID"
  rc=$?; WRF_PID=""
  kill "$FATAL_WATCH_PID" 2>/dev/null; wait "$FATAL_WATCH_PID" 2>/dev/null
  pkill -u "$USER" -x wrf.exe 2>/dev/null
  [ $rc -ne 0 ] && return 1
  shopt -s nullglob; TBLS=(qr_acr_q*.dat freezeH2O.dat); shopt -u nullglob
  for f in "${TBLS[@]}"; do [ -L "$f" ] || [ -e "$TBL_CACHE/$f" ] || { mkdir -p "$TBL_CACHE" && cp -f "$f" "$TBL_CACHE/.$f.$$" && mv -f "$TBL_CACHE/.$f.$$" "$TBL_CACHE/$f"; }; done
  LAST_WRF_STATUS="ok"
  return 0
}

if run_window 0 GFS Vtable.GFS 1 GFS:2019-11-26_15 1 met_em.d01.2019-11-26_15:00:00.nc; then
  echo "WINDOW 0 rc=0 ok=1 real=$LAST_REAL_PATH wrf=$LAST_WRF_STATUS ml=$LAST_ML_STATUS ml_targets=$LAST_ML_TARGETS" >> "$STATUS"
else
  echo "WINDOW 0 rc=1 ok=0 real=$LAST_REAL_PATH wrf=$LAST_WRF_STATUS ml=$LAST_ML_STATUS ml_targets=$LAST_ML_TARGETS" >> "$STATUS"
fi
CUR_WINDOW=""

run_hygiene reap+census
echo "=== DONE ($(date -u +%FT%TZ)) ==="
cat "$STATUS"

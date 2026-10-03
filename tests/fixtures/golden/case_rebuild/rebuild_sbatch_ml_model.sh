#!/usr/bin/env bash
#SBATCH --job-name=case-rebuild
#SBATCH --partition=compute
#SBATCH --exclude=head
#SBATCH -N 1 --exclusive --ntasks-per-node=72
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@90
#SBATCH --output=/scratch/rb/run-%j.out
set -uo pipefail

TOOL=<TOOL>
TOOL_SCM=/opt/wtune/src/tools/state_correction_model.py
ML_MODEL=/plan/report.json
GRIB_PY=${GRIB_PYTHON:-$HOME/wtune/envs/grib/.pixi/envs/default/bin/python}
SITE=scc
CAMPAIGN=${CAMPAIGN:-$HOME/wtune/campaign}
source "$CAMPAIGN/sites/$SITE.sh"   # site_env / site_launch / site_hygiene -- reused, not reimplemented
source ~/wtune-scc/env.sh
export PATH=$NETCDF/bin:$PATH
ulimit -s unlimited
site_env
REBUILD_DIR=/scratch/rb
ALLOC_NODES=${SLURM_JOB_NUM_NODES:-1}
STATUS="$REBUILD_DIR/rebuild_status.txt"
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
    echo "STAGE $CUR_WINDOW rc=1 ok=0 INTERRUPTED sig=$sig" >> "$STATUS"
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

cd "$REBUILD_DIR/wps"
export OMP_NUM_THREADS=1
CUR_WINDOW=geogrid
echo "=== geogrid.exe ($(date -u +%FT%TZ)) ==="
mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core ./geogrid.exe
rc=$?; pkill -u "$USER" -x geogrid.exe 2>/dev/null
GEO_EM_MISSING=()
for f in geo_em.d01.nc; do [ -f "$f" ] || GEO_EM_MISSING+=("$f"); done
if [ $rc -ne 0 ] || [ ${#GEO_EM_MISSING[@]} -gt 0 ] || ! grep -q 'Successful completion' geogrid.log* 2>/dev/null; then
  echo "FATAL geogrid rc=$rc missing=${GEO_EM_MISSING[*]:-none}"; exit 1
fi

CUR_WINDOW=link_grib
if [ -s "$REBUILD_DIR/grib_manifest.txt" ]; then
  mapfile -t GRIB_FILES < <(sed "s#^#$REBUILD_DIR/grib/#" "$REBUILD_DIR/grib_manifest.txt")
else
  shopt -s nullglob
  GRIB_FILES=("$REBUILD_DIR"/grib/*.grib2 "$REBUILD_DIR"/grib/*.grib "$REBUILD_DIR"/grib/*.grb)
  shopt -u nullglob
fi
if [ ${#GRIB_FILES[@]} -eq 0 ]; then
  echo "FATAL: no forcing files in $REBUILD_DIR/grib -- run 'case_rebuild.py rebuild fetch --out $REBUILD_DIR' first"; exit 1
fi
rm -f GRIBFILE.*
./link_grib.csh "${GRIB_FILES[@]}"
[ $? -ne 0 ] && { echo "FATAL link_grib.csh failed"; exit 1; }
python3 "$TOOL" rebuild fix-span-wps-namelist --out "$REBUILD_DIR" || exit 1
if [ -e Vtable.GFS ]; then ln -sf Vtable.GFS Vtable; else echo "FATAL: Vtable.GFS not staged"; exit 1; fi
rm -f "GFS:"*
CUR_WINDOW=ungrib
./ungrib.exe
rc=$?; pkill -u "$USER" -x ungrib.exe 2>/dev/null
UNGRIB_MISSING=()
for f in GFS:2020-01-01_00 GFS:2020-01-01_03 GFS:2020-01-01_06; do [ -f "$f" ] || UNGRIB_MISSING+=("$f"); done
[ $rc -ne 0 ] || [ ${#UNGRIB_MISSING[@]} -gt 0 ] && { echo "FATAL ungrib rc=$rc missing=${UNGRIB_MISSING[*]:-none}"; exit 1; }

CUR_WINDOW=metgrid
echo "=== metgrid.exe ($(date -u +%FT%TZ)) ==="
mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core ./metgrid.exe
rc=$?; pkill -u "$USER" -x metgrid.exe 2>/dev/null
MET_MISSING=()
for f in met_em.d01.2020-01-01_00:00:00.nc met_em.d01.2020-01-01_03:00:00.nc met_em.d01.2020-01-01_06:00:00.nc; do [ -f "$f" ] || MET_MISSING+=("$f"); done
if [ $rc -ne 0 ] || [ ${#MET_MISSING[@]} -gt 0 ] || ! grep -q 'Successful completion' metgrid.log* 2>/dev/null; then
  echo "FATAL metgrid rc=$rc missing=${MET_MISSING[*]:-none}"; exit 1
fi

CUR_WINDOW=real
echo "=== real.exe ($(date -u +%FT%TZ)) ==="
cd "$REBUILD_DIR/real"
rm -f met_em.d0*.*.nc; ln -sf "$REBUILD_DIR"/wps/met_em.d0*.*.nc .
for f in met_em.d0*.*.nc; do [ "$f" = "${f//:/_}" ] || ln -sf "$f" "${f//:/_}"; done
python3 "$TOOL" rebuild fix-span-real-namelist --out "$REBUILD_DIR" || exit 1
timeout 600 mpirun -np 72 --map-by ppr:72:node:PE=1 --bind-to core ./real.exe
rc=$?; pkill -u "$USER" -x real.exe 2>/dev/null
if [ $rc -eq 124 ] || [ ! -f wrfinput_d01 ] || [ ! -f wrfbdy_d01 ]; then
  echo "real.exe decomposed attempt rc=$rc -- retrying serial" >&2
  python3 "$TOOL" rebuild fix-span-real-namelist --out "$REBUILD_DIR" --serial || exit 1
  timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core ./real.exe
  rc=$?; pkill -u "$USER" -x real.exe 2>/dev/null
fi
[ $rc -ne 0 ] && { echo "FATAL real.exe rc=$rc"; exit 1; }
[ -f wrfinput_d01 ] && [ -f wrfbdy_d01 ] || { echo "FATAL: no wrfinput/wrfbdy"; exit 1; }
ML_OUT=$("$GRIB_PY" "$TOOL_SCM" apply --model "$ML_MODEL" --targets-from-report --wrfinput wrfinput_d01 --out wrfinput_d01.ml 2>&1)
ml_rc=$?
ML_TARGETS=$(printf '%s\n' "$ML_OUT" | grep '^APPLIED_TARGETS=' | tail -1 | sed 's/^APPLIED_TARGETS=//')
if [ $ml_rc -eq 0 ] && [ -f wrfinput_d01.ml ]; then
  mv wrfinput_d01 wrfinput_d01.noml && mv wrfinput_d01.ml wrfinput_d01
  echo "ml=applied ml_targets=$ML_TARGETS" > "$REBUILD_DIR/ml_status.txt"
else
  rm -f wrfinput_d01.ml
  echo "ml=skipped ml_targets=$ML_TARGETS" > "$REBUILD_DIR/ml_status.txt"
  printf '%s\n' "$ML_OUT" >&2
fi
CUR_WINDOW=""

run_hygiene reap+census
echo "=== DONE ($(date -u +%FT%TZ)) ==="

#!/usr/bin/env bash
#SBATCH --job-name=nc2bp
#SBATCH --partition=compute
#SBATCH --exclude=head
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=00:40:00
# Pre-convert one campaign case's netCDF inputs into the BP cache strict_campaign.sbatch reads
# (2026-10-03: ADIOS2 for every WRF input stream). Optional: strict_campaign converts on first use
# by itself; this only moves that one-off cost out of a probe allocation. SAME code path:
# configs.py stage-bp-inputs -> src/wtune_bp.convert_inputs (src/tools/wrf_nc2bp.py under the
# envs/adios2 python, $WTUNE_ADIOS2_PYTHON overrides; BP4; manifest-cached; lockf-serialized, so a
# probe job starting meanwhile waits instead of converting twice). Converts wrfinput_d0N, wrfbdy_d01
# and the aux inputs the case namelist reads (wrflowinp when sst_update=1, wrffdda, ...) from the
# case's prep_dir (cases.json + campaign/cases.$SITE.json overlay) into
# ${WTUNE_BP_CACHE:-$HOME/wtune-runs/bp-cache}/<case>/<sha1(prep_dir)[:12]>/ -- wrfbdy/wrflowinp/
# wrffdda one BP per record (<cache>/<name>/<name>_<date>.bp: WRF 4.8's ADIOS2 reader returns
# record 1 for every read of a multi-record BP). CHECK=1 (default) then
# round-trips every converted file back to netCDF and compares (wtune_bp inputs --check, needs
# netCDF4: the envs/wtune python, $WTUNE_PY overrides).
# usage (scc): sbatch --job-name=nc2bp-<case> --export=ALL,CASE=<case> campaign/nc2bp.sbatch
#   CASES=<a>:<b>:... converts several cases in one job (wtune tune, before its first probes;
#   CASES_JSON selects a generated case file). other sites: add -p/-A/--exclude and SITE=<site>.
set -uo pipefail
SITE=${SITE:-scc}
export SITE
CAMPAIGN=${CAMPAIGN:-$HOME/wtune/campaign}
export CAMPAIGN
CASES=${CASES:-${CASE:-}}
: "${CASES:?CASE (or CASES=a:b) required, e.g. --export=ALL,CASE=karina}"
# PIXI_ENV (and PATH to pixi) from the site file, the same interpreter strict_campaign.sbatch uses
source "$CAMPAIGN/sites/$SITE.sh"
PYBIN=$(pixi run --manifest-path "$PIXI_ENV/pixi.toml" python -c 'import sys; print(sys.executable)' 2>/dev/null | tail -1)
if [ -z "$PYBIN" ] || [ ! -x "$PYBIN" ]; then
  echo "FATAL: no python from PIXI_ENV=$PIXI_ENV (got '$PYBIN')" >&2; exit 1
fi
convert_case() {  # case
  local CASE="$1" PREP_DIR NAMELIST_TEMPLATE STAGE_OUT t0
  eval "$("$PYBIN" - "$CASE" <<'PY'
import os, sys
sys.path.insert(0, os.environ["CAMPAIGN"])
import configs
c = configs.CASES.get(sys.argv[1])
if c is None:
    print('echo "unknown case %s" >&2; exit 1' % sys.argv[1])
    sys.exit(0)
print("PREP_DIR=%s" % configs._shq(c["prep_dir"]))
print("NAMELIST_TEMPLATE=%s" % configs._shq(
    c.get("namelist_template") or os.path.join(c["prep_dir"], "namelist.input")))
PY
)"
  echo "case $CASE site $SITE prep_dir $PREP_DIR namelist $NAMELIST_TEMPLATE"
  t0=$(date +%s)
  if ! STAGE_OUT=$("$PYBIN" "$CAMPAIGN/configs.py" stage-bp-inputs "$CASE" "$NAMELIST_TEMPLATE" "$PREP_DIR"); then
    echo "=== FAIL: ADIOS2 input staging failed for $CASE (see above)"; return 1
  fi
  eval "$STAGE_OUT"
  echo "staged into $BP_DIR in $(( $(date +%s) - t0 )) s; converted: ${BP_CONVERTED:-none}; aux streams as BP: ${WTUNE_AUX_BP_STREAMS:-none}; per record: ${WTUNE_PER_RECORD_BP:-none}"
  [ "${CHECK:-1}" = 1 ] && [ -n "$BP_CONVERTED" ] || return 0
  WTUNE_PY=${WTUNE_PY:-$(dirname "$CAMPAIGN")/envs/wtune/.pixi/envs/default/bin/python}
  BP_PYMOD=${WTUNE_SRC:-$(dirname "$CAMPAIGN")/src}/wtune_bp.py
  if [ -x "$WTUNE_PY" ] && [ -f "$BP_PYMOD" ]; then
    # shellcheck disable=SC2086
    "$WTUNE_PY" "$BP_PYMOD" inputs "$PREP_DIR" --cache "$BP_DIR" --names $BP_CONVERTED --check || return 1
  else
    echo "WARNING: round-trip check skipped: $WTUNE_PY or $BP_PYMOD missing" >&2
  fi
}
fail=0
IFS=':' read -ra CASE_LIST <<< "$CASES"
for c in "${CASE_LIST[@]}"; do convert_case "$c" || fail=1; done
echo "=== done fail=$fail"
exit "$fail"

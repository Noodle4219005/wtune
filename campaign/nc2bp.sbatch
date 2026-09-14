#!/usr/bin/env bash
#SBATCH --job-name=nc2bp
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/campaign/nc2bp-%x-%j.out
# Convert one campaign case's netCDF inputs (wrfinput_d0N for N <= max_dom, wrfbdy_d01) to BP5 next to them,
# for io_form_input/boundary = 14. Serial converter (wtune/tools/wrf_nc2bp.py, validated by job 1071692 and the
# 1072441 regression gate). Existing .bp outputs are skipped. Ends with a readback of steps/vars/Times.
# usage: sbatch --job-name=nc2bp-<case> --export=ALL,CASE=<case> campaign/nc2bp.sbatch
set -uo pipefail
ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
CONV=/home/u5727520/wrf-stack/wtune/tools/wrf_nc2bp.py
: "${CASE:?CASE required}"
read -r SRC MAXDOM < <(python3 - "$CASE" <<'PY'
import json, sys
c = json.load(open("/home/u5727520/wrf-stack/wtune-autotuner/campaign/cases.json"))["cases"][sys.argv[1]]
print(c["prep_dir"], c.get("max_dom", 1))
PY
)
echo "case $CASE src $SRC max_dom $MAXDOM"
FILES="wrfbdy_d01"; for d in $(seq 1 "$MAXDOM"); do FILES="$FILES wrfinput_d0$d"; done
cd "$ENV" || exit 1
fail=0
for f in $FILES; do
  [ -f "$SRC/$f" ] || { echo "MISSING $SRC/$f"; fail=1; continue; }
  [ -d "$SRC/$f.bp" ] && { echo "skip $f (exists)"; continue; }
  s=$(date +%s)
  pixi run python "$CONV" "$SRC/$f" "$SRC/$f.bp" --engine BP5 || { echo "FAIL $f"; rm -rf "$SRC/$f.bp"; fail=1; continue; }
  echo "$f -> $(du -sh "$SRC/$f.bp" | cut -f1) in $(( $(date +%s) - s )) s"
done
pixi run python - "$SRC" $FILES <<'PY'
import sys, os, adios2
src = sys.argv[1]
for f in sys.argv[2:]:
    p = os.path.join(src, f + ".bp")
    if not os.path.isdir(p):
        print("READBACK missing %s" % p); continue
    # streaming ("r") mode: Stream.steps() is not allowed in ReadRandomAccess ("rra") mode
    # (this exact bug already hit job 1071692 and again jobs 1074578-1074582)
    t, nv = [], 0
    with adios2.Stream(p, "r") as s:
        for _ in s.steps():
            nv = max(nv, len(s.available_variables()))
            t.append(s.read("Times"))
    print("READBACK %s: steps=%d vars=%d Times=%s..%s" % (f, len(t), nv, t[:1], t[-1:]))
PY
echo "=== done fail=$fail"

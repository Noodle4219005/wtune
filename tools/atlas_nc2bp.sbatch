#!/usr/bin/env bash
#SBATCH --job-name=nc2bp
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home/u5727520/wrf-bench/atlas/nc2bp-%j.out
# atlas netCDF inputs -> BP5 (io_form_input/boundary = 14). Serial, one Time slice at a time.
# usage: sbatch --export=ALL,SRC=<case dir>,OUT=<bp dir>,FILES="wrfinput_d01 wrfbdy_d01" nc2bp.sbatch
set -uo pipefail
SRC=${SRC:?}; OUT=${OUT:?}; FILES=${FILES:-"wrfinput_d01 wrfbdy_d01"}
ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
CONV=/home/u5727520/wrf-stack/wtune/tools/wrf_nc2bp.py
mkdir -p "$OUT"; cd "$ENV" || exit 1
for f in $FILES; do
  [ -d "$OUT/$f.bp" ] && { echo "skip $f (exists)"; continue; }
  s=$(date +%s)
  pixi run python "$CONV" "$SRC/$f" "$OUT/$f.bp" --engine BP5 || { echo "FAIL $f"; rm -rf "$OUT/$f.bp"; continue; }
  echo "$f -> $(du -sh "$OUT/$f.bp" | cut -f1) in $(( $(date +%s) - s )) s"
done
ls -la "$OUT"
# readback check: variable count, Times steps
pixi run python - "$OUT" <<'PY'
import sys, os, adios2
out = sys.argv[1]
for f in sorted(os.listdir(out)):
    with adios2.FileReader(os.path.join(out, f)) as r:
        v = r.available_variables()
        print(f"{f}: vars={len(v)} Times_steps={v['Times']['AvailableStepsCount']} first={r.read('Times')}")
PY

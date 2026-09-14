#!/usr/bin/env bash
#SBATCH --job-name=fppreflight
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/fingerprint/fppreflight-%j.out
#
# Preflight for the phase fingerprinter before the 4-node validation job:
#  1. the repo test suite (tools/run_tests.sbatch stages, run inline);
#  2. real-data classifier check: perf data recorded by gate 1 (job 1074101, pristine wrf.exe, one rank per file)
#     -> perf script -> fingerprint.py --preflight-perf-script. Pass: parse rate >= 99% and MICRO, MPI and
#     DYNAMICS_OTHER all present with non-zero share for every file.
set -uo pipefail
REPO=${REPO:-/home/u5727520/wrf-stack/wtune-autotuner}
PIXI="pixi run --manifest-path /home/u5727520/wrf-stack/wtune-pixi-env-20260830/pixi.toml"
G1=/home/u5727520/wrf-bench/wscore/g1_V13_1074101/g1
T=/dev/shm/fppreflight.$SLURM_JOB_ID; mkdir -p "$T"
cd "$REPO" || exit 2
echo "repo=$REPO head=$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
bash tools/run_tests.sbatch; tests_rc=$?
echo "=== tests rc=$tests_rc"
for f in "$G1"/perf.r*.data.gz; do
  r=$(basename "$f" .data.gz)
  gunzip -c "$f" > "$T/$r.data"
  perf script -i "$T/$r.data" -F tid,time,period,event,ip,sym,dso 2> "$T/$r.err" | gzip > "$T/$r.txt.gz"
  echo "perf script $r lines=$(gunzip -c "$T/$r.txt.gz" | wc -l) err_lines=$(wc -l < "$T/$r.err")"
  rm -f "$T/$r.data"
done
$PIXI python fingerprint/fingerprint.py --preflight-perf-script "$T"/perf.r*.txt.gz; pre_rc=$?
echo "=== preflight rc=$pre_rc"

# End-to-end sampler check with the EXACT production perf command lines (job 1074372 lesson): a fake
# single-rank "wrf.exe" (python3 exec'd under that name so pgrep -x matches) burns CPU and appends
# "Timing for main" lines; sampler_node.sh attaches, stops, waits for exit, runs perf script; then
# fingerprint.py must parse a non-empty stream. Pass: DONE lines>0, n_parsed>0, unparsed=0, nranks=1.
E=$T/e2e; mkdir -p "$E/fp"
ln -sf "$(command -v python3)" "$E/wrf.exe"
printf ' nproc_x=1, nproc_y=1, numtiles=2,\n' > "$E/namelist.input"
( cd "$E" && SLURM_PROCID=0 exec -a wrf.exe ./wrf.exe -c '
import time
t0 = time.time()
with open("rsl.out.0000", "a") as f:
    for step in range(12):
        end = time.time() + 1.0
        x = 0
        while time.time() < end:
            x += sum(i * i for i in range(2000))
        f.write("Timing for main: time 2026-01-01_00:00:%02d on domain   1:    1.00000 elapsed seconds\n" % step)
        f.flush()
' ) &
FAKE=$!
POLL_S=1 bash fingerprint/sampler_node.sh "$E" "$E/fp" 250 2 8
wait $FAKE
cat "$E"/fp/meta.*.txt | grep -vE '^PID'
E2E_OUT=$($PIXI python fingerprint/fingerprint.py --sampler-dir "$E/fp" --run-dir "$E" --omp-threads 1 --out "$E/fingerprint" 2>&1); e2e_rc=$?
echo "$E2E_OUT" | tail -3
$PIXI python - "$E/fingerprint.json" <<'PY'; e2e_check=$?
import json, sys
m = json.load(open(sys.argv[1]))["meta"]
print("E2E n_parsed=%s n_unparsed=%s nranks=%s" % (m.get("n_parsed"), m.get("n_unparsed"), m.get("nranks")))
sys.exit(0 if (m.get("n_parsed") or 0) > 0 and m.get("n_unparsed") == 0 and m.get("nranks") == 1 else 1)
PY
echo "=== e2e rc=$e2e_rc check=$e2e_check"
rm -rf "$T"
[ "$tests_rc" = 0 ] && [ "$pre_rc" = 0 ] && [ "$e2e_rc" = 0 ] && [ "$e2e_check" = 0 ] && echo "PREFLIGHT PASS" || echo "PREFLIGHT FAIL"

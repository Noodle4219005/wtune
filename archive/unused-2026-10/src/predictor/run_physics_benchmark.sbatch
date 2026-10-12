#!/bin/bash
#SBATCH --job-name=wtune-physics-bench
#SBATCH --partition=ct112
#SBATCH --account=ACD110018
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predictor/physics_bench_%j.out
#SBATCH --error=/home/u5727520/wrf-bench/wscore/predictor/physics_bench_%j.err
# (stdout/stderr must be plain files in an existing directory: a per-job subdirectory does not
# exist yet at launch and the job fails in seconds -- job 1077878)
#
# PHYSICS-GRAY predictor benchmark (decision 0051, predictor/physics.py). Same shape as
# run_benchmark.sbatch (STRICT phase A): test -> benchmark, but over the physics-campaign ledger
# contract instead of the STRICT one. Written but not auto-submitted.
#
# Usage: sbatch --export=ALL,PHYSICS_LEDGERS='/path/ledger_a.csv /path/ledger_b.csv' \
#          predictor/run_physics_benchmark.sbatch
#        sbatch --export=ALL,PHYSICS_LEDGERS='/home/u5727520/wrf-bench/wscore/campaign/ledger_*physics*.csv' \
#          predictor/run_physics_benchmark.sbatch
# Env:   PHYSICS_LEDGERS (required) -- space-separated paths and/or shell globs to physics-
#        campaign ledger CSVs (source_dataset=strict_campaign, mode=physics rows). Refuses to run
#        (exit 1, before touching Slurm output/pytest) when this expands to zero files -- an
#        UNCONFIRMED-everything PHYSICS-GRAY report from an accidentally-empty ledger set is worse
#        than a job that never ran.
#
# Not gated by the cluster-batch cost line in AGENTS.md/CLAUDE.md ("N jobs x nodes x hours"):
# this is 1 node x <=30 min of pure CPU Python (no WRF, no MPI ranks), well under a single
# node-hour (22.4 units), so it is reported here for completeness rather than as a cost decision.
set -euo pipefail

REPO=/home/u5727520/wrf-stack/wtune-autotuner
PIXI_MANIFEST=/home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml
OUT_ROOT=/home/u5727520/wrf-bench/wscore/predictor
BENCH_OUT="$OUT_ROOT/physics_bench_${SLURM_JOB_ID:-manual}"

: "${PHYSICS_LEDGERS:?PHYSICS_LEDGERS env var is required, e.g. --export=ALL,PHYSICS_LEDGERS='/path/ledger_*.csv' -- refusing to run the PHYSICS-GRAY benchmark with no ledger data}"

# Expand PHYSICS_LEDGERS (space-separated paths/globs) BEFORE doing anything else -- an empty
# expansion means every glob missed and every candidate benchmark run below would be exercising
# physics.py's "no physics rows" branch only, not the actual PHYSICS-GRAY models.
shopt -s nullglob
LEDGER_FILES=( $PHYSICS_LEDGERS )
shopt -u nullglob
if [ "${#LEDGER_FILES[@]}" -eq 0 ]; then
  echo "ERROR: PHYSICS_LEDGERS='$PHYSICS_LEDGERS' expanded to zero files -- nothing to benchmark" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$BENCH_OUT"
cd "$REPO"

PIXIRUN="pixi run --manifest-path $PIXI_MANIFEST python"

echo "physics ledgers (${#LEDGER_FILES[@]}): ${LEDGER_FILES[@]}"

echo "== 1/2: pytest predictor/tests =="
$PIXIRUN -m pytest predictor/tests -v

echo "== 2/2: physics.py benchmark -> $BENCH_OUT =="
# physics.py's own CLI (predictor/physics.py --ledger ... --out ...) takes exactly one ledger
# path, so concatenate the expanded files into one CSV first (header from the first file only).
MERGED="$BENCH_OUT/merged_ledger.csv"
head -n 1 "${LEDGER_FILES[0]}" > "$MERGED"
for f in "${LEDGER_FILES[@]}"; do
  tail -n +2 "$f" >> "$MERGED"
done
$PIXIRUN predictor/physics.py --ledger "$MERGED" --out "$BENCH_OUT"

echo "done: $BENCH_OUT"

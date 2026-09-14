#!/bin/bash
#SBATCH --job-name=wtune-predictor-bench
#SBATCH --partition=ct112
#SBATCH --account=ACD110018
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/predictor/bench_%j.out
#SBATCH --error=/home/u5727520/wrf-bench/wscore/predictor/bench_%j.err
#
# Phase A of the WTune configuration predictor: test -> build ledger -> benchmark.
# Written but NOT submitted (offline-benchmark-only task; no job submission from this session).
#
# Not gated by the cluster-batch cost line in AGENTS.md/CLAUDE.md ("N jobs x nodes x hours"):
# this is 1 node x <=30 min of pure CPU Python (no WRF, no MPI ranks), i.e. well under a single
# node-hour (22.4 units), so it is reported here for completeness rather than as a cost decision.
set -euo pipefail

REPO=/home/u5727520/wrf-stack/wtune-autotuner
PIXI_MANIFEST=/home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml
WIKI_DATA=/home/u5727520/my-llm-wiki/data
OUT_ROOT=/home/u5727520/wrf-bench/wscore/predictor
BENCH_OUT="$OUT_ROOT/bench_${SLURM_JOB_ID}"

mkdir -p "$OUT_ROOT" "$BENCH_OUT"
cd "$REPO"

PIXIRUN="pixi run --manifest-path $PIXI_MANIFEST python"

echo "== 1/3: pytest predictor/tests =="
$PIXIRUN -m pytest predictor/tests -v

echo "== 2/3: build ledger =="
# campaign ledgers (strict_campaign.sbatch output). Default = only the cf448 STRICT campaign jobs
# 1075293-1075304. Never glob all ledger_*.csv: the 1074xxx smoke ledgers include runs taken
# before the nested-timing fixes (1074506-1074517) and single-config smokes on icpnq.
CAMPAIGN_DIR=/home/u5727520/wrf-bench/wscore/campaign
CAMPAIGN_GLOB=${CAMPAIGN_GLOB:-"$CAMPAIGN_DIR/ledger_*_107529[3-9].csv $CAMPAIGN_DIR/ledger_*_107530[0-4].csv"}
shopt -s nullglob
CAMPAIGN_LEDGERS=( $CAMPAIGN_GLOB )
shopt -u nullglob
echo "campaign ledgers: ${#CAMPAIGN_LEDGERS[@]}"
$PIXIRUN predictor/ledger.py build --wiki-data "$WIKI_DATA" --out predictor/ledger/runs.csv \
  --campaign-ledger "${CAMPAIGN_LEDGERS[@]}"
$PIXIRUN predictor/ledger.py validate --ledger predictor/ledger/runs.csv

echo "== 3/3: benchmark -> $BENCH_OUT =="
$PIXIRUN predictor/benchmark.py --ledger predictor/ledger/runs.csv --out "$BENCH_OUT"

echo "done: $BENCH_OUT"

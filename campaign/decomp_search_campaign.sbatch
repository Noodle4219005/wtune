#!/bin/bash
#SBATCH --job-name=decompsearch
#SBATCH --account=ACD110018
#SBATCH --partition=cf448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=01:30:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/campaign/decompsearch-%x-%j.out
#
# Find one case's best decomposition inside ONE allocation (no queue wait between rounds):
#   R0  STRICT templates (TEMPLATES, 448-core cases), or a balanced seed design when TEMPLATES is empty
#   R1.. D3 propose -> measure -> ingest, until D3 says STOP or ROUNDS is reached
#   CF  rerun the top CONFIRM_TOP measured decompositions once more, then rank everything
# Every WRF run goes through strict_campaign.sbatch unchanged, so ledgers keep the campaign schema.
#
# Usage: sbatch --export=ALL,CASE=zhengzhou campaign/decomp_search_campaign.sbatch
#        sbatch -p cf112 -N1 --time=00:45:00 --export=ALL,CASE=severe2023,TEMPLATES= campaign/decomp_search_campaign.sbatch
# Env:   CASE (required), TEMPLATES (default C01+C02+C07+C08+C10; C01 is the y baseline),
#        ROUNDS (3), BATCH (8), REPS (1), CONFIRM_TOP (3), STATE_DIR, PRIOR_LEDGERS,
#        EXTRA_LEDGERS (appended to PRIOR_LEDGERS, e.g. this case's earlier ledgers),
#        TEMPLATES=none (skip R0 entirely: this case's baseline rows come from EXTRA_LEDGERS),
#        MAX_OMP / MIN_TILE_ROWS / TILE_SHAPES (restrict D3 proposals: crashing thread counts,
#        hanging tile depths, tile geometries such as 1d_y only).
set -uo pipefail
: "${CASE:?CASE env var is required}"

# SITE/REPO (decision 0050 step 8): same site-conditional-default + campaign/sites/$SITE.sh
# pattern as strict_campaign.sbatch, so OUT/HARDWARE_ID/CORES_PER_NODE here are guaranteed
# identical to what the nested strict_campaign.sbatch calls below (run_campaign) compute
# themselves, instead of a second, driftable copy of the same per-site absolute paths.
SITE=${SITE:-f1}
export SITE
if [ "$SITE" = "scc" ]; then
  REPO=${REPO:-$HOME/wtune}
else
  REPO=${REPO:-/home/u5727520/wrf-stack/wtune-autotuner}
fi
export CAMPAIGN="$REPO/campaign"
source "$CAMPAIGN/sites/$SITE.sh"   # OUT, HARDWARE_ID, CORES_PER_NODE (site_env/site_launch unused here)
STATE_DIR=${STATE_DIR:-/home/u5727520/wrf-bench/wscore/decomp/search}
PY="pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml python"
TEMPLATES=${TEMPLATES-C01+C02+C07+C08+C10}
ROUNDS=${ROUNDS:-3}
BATCH=${BATCH:-8}
REPS=${REPS:-1}
CONFIRM_TOP=${CONFIRM_TOP:-3}
NODES=${SLURM_JOB_NUM_NODES:?}
TOTAL_CORES=$(( NODES * CORES_PER_NODE ))
J=$SLURM_JOB_ID
LEDGER="$OUT/ledger_${CASE}_${J}.csv"      # strict_campaign.sbatch appends every round here
L=$OUT
PRIOR_LEDGERS=${PRIOR_LEDGERS:-"$L/ledger_isc21_1075293.csv $L/ledger_maria1km_1075294.csv $L/ledger_lafe_1075295.csv $L/ledger_conus25_1075296.csv $L/ledger_karina_1075297.csv $L/ledger_conus12_1075298.csv $L/ledger_karina_1078036.csv $L/ledger_karina_1078093.csv $L/ledger_lafe_1078094.csv $L/ledger_karina_1078158.csv $L/ledger_lafe_1078238.csv"}
PRIOR_LEDGERS="$PRIOR_LEDGERS ${EXTRA_LEDGERS:-}"
PROPOSE_LIMITS=""
[ -n "${MAX_OMP:-}" ] && PROPOSE_LIMITS="$PROPOSE_LIMITS --max-omp $MAX_OMP"
[ -n "${MIN_TILE_ROWS:-}" ] && PROPOSE_LIMITS="$PROPOSE_LIMITS --min-tile-rows $MIN_TILE_ROWS"
[ -n "${TILE_SHAPES:-}" ] && PROPOSE_LIMITS="$PROPOSE_LIMITS --tile-shapes $TILE_SHAPES"
D="$STATE_DIR/${CASE}_${J}"
mkdir -p "$D"
STATE="$D/state.json"
cd "$REPO" || exit 1
echo "limits:${PROPOSE_LIMITS:- none} extra_ledgers=${EXTRA_LEDGERS:-none}"
echo "decompsearch case=$CASE job=$J nodes=$NODES cores=$TOTAL_CORES hw=$HARDWARE_ID templates='${TEMPLATES}' rounds=$ROUNDS batch=$BATCH reps=$REPS confirm_top=$CONFIRM_TOP dir=$D"

run_campaign () {  # env for one strict_campaign.sbatch pass; it uses $SLURM_JOB_ID -> same $LEDGER
  env CASE="$CASE" MODE=timing REPS="$1" "${@:2}" bash "$REPO/campaign/strict_campaign.sbatch"
}

if [ "$TEMPLATES" = "none" ]; then
  echo "===== R0 skipped (TEMPLATES=none)"
elif [ -n "$TEMPLATES" ]; then
  echo "===== R0 templates $TEMPLATES"
  run_campaign "$REPS" CONFIGS="$TEMPLATES" 2>&1 | grep -E "^RESULT|ERROR|FATAL|ABORT"
else
  # no templates for this allocation (e.g. 1 node): balanced shape per thread count instead, so D3's
  # first proposal is trained on this case's own rows rather than extrapolated from 448-core cases
  echo "===== R0 seed design"
  $PY predictor/decomp_space.py seed "$CASE" --out "$D/seed.json" --total-cores "$TOTAL_CORES" \
      --nodes "$NODES" --hardware-id "$HARDWARE_ID" 2>&1 | grep -v "WARN cache"
  jq -r '.configs[] | [.id,.ranks,.omp,.nx,.ny,.numtiles] | @tsv' "$D/seed.json"
  run_campaign "$REPS" CONFIG_FILE="$D/seed.json" 2>&1 | grep -E "^RESULT|ERROR|FATAL|ABORT"
fi

for r in $(seq 1 "$ROUNDS"); do
  P="$D/proposal_r${r}.json"
  echo "===== R$r propose"
  LEDGER_ARGS="$PRIOR_LEDGERS"; [ -f "$LEDGER" ] && LEDGER_ARGS="$LEDGER_ARGS $LEDGER"  # absent until a run happens (TEMPLATES=none)
  $PY predictor/decomp_search.py propose --case "$CASE" --ledger $LEDGER_ARGS \
      --state "$STATE" --batch "$BATCH" --out "$P" --total-cores "$TOTAL_CORES" --nodes "$NODES" \
      --hardware-id "$HARDWARE_ID" --reps "$REPS" $PROPOSE_LIMITS 2>&1 | grep -vE "WARN cache|^sbatch "
  [ -f "$P" ] || { echo "R$r: no proposal (stopped)"; break; }
  jq -r '.configs[] | [.id,.ranks,.omp,.nx,.ny,.numtiles,.tile_shape,.dist,.blocktime] | @tsv' "$P"
  echo "===== R$r measure"
  run_campaign "$REPS" CONFIG_FILE="$P" 2>&1 | grep -E "^RESULT|ERROR|FATAL|ABORT"
  # ingest only this round's rows: ingest() appends, so re-reading earlier rounds would double them
  RL="$D/ledger_r${r}.csv"
  { head -1 "$LEDGER"; grep "^${CASE}_R$(printf %02d "$r")D" "$LEDGER"; } > "$RL"
  $PY predictor/decomp_search.py ingest --state "$STATE" --ledger "$RL" 2>&1 | grep -v "WARN cache"
done

if [ "$CONFIRM_TOP" -gt 0 ]; then
  echo "===== CF confirm top $CONFIRM_TOP"
  $PY campaign/decomp_summary.py --case "$CASE" --ledger ${EXTRA_LEDGERS:-} "$LEDGER" --top "$CONFIRM_TOP" \
      --confirm-out "$D/confirm.json" > /dev/null 2>&1
  [ -f "$D/confirm.json" ] && run_campaign 1 CONFIG_FILE="$D/confirm.json" 2>&1 | grep -E "^RESULT|ERROR|FATAL|ABORT"
fi

echo "===== FINAL ranking ($CASE, job $J)"
RANK_LEDGERS="${EXTRA_LEDGERS:-}"; [ -f "$LEDGER" ] && RANK_LEDGERS="$RANK_LEDGERS $LEDGER"
$PY campaign/decomp_summary.py --case "$CASE" --ledger $RANK_LEDGERS --tsv "$D/ranking.tsv" 2>&1 | grep -v "WARN cache"

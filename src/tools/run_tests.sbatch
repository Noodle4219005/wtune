#!/usr/bin/env bash
#SBATCH --job-name=wtune-tests
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/autotuner/wtune-tests-%j.out
#
# Full WTune test suite for a worktree/branch, on a compute node (never on the login node).
# usage: sbatch [--export=ALL,REPO=/path/to/worktree] tools/run_tests.sbatch
# Pass = every stage prints PASS and the final line is "ALL STAGES PASS".
set -uo pipefail
REPO=${REPO:-/home/u5727520/wrf-stack/wtune-autotuner}
PIXI="pixi run --manifest-path /home/u5727520/wrf-stack/wtune-pixi-env-20260830/pixi.toml"
PIXI_PRED="pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml"
cd "$REPO" || exit 2
echo "repo=$REPO branch=$(git branch --show-current) head=$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
fail=0
stage() {
  local name=$1; shift
  echo "=== $name: $*"
  if "$@"; then echo "=== $name PASS"; else echo "=== $name FAIL rc=$?"; fail=1; fi
}
stage py_compile $PIXI python -m py_compile wtune.py src/wtune_*.py tests/test_wtune_*.py
# Root tests. test_fidelity_gate.py does NOT match test_wtune_*.py and was silently skipped
# until 2026-09-20; name it explicitly rather than widening the glob into wtune_*.py sources.
stage pytest $PIXI python -m pytest -q tests/test_wtune_*.py tests/test_fidelity_gate.py $(ls archive/plov/tests/test_*.py 2>/dev/null)
stage pytest_campaign $PIXI python -m pytest -q campaign tests/tools archive/experiments
# predictor/tests needs the OTHER pixi env (lightgbm/catboost). Running it under $PIXI turns
# 18 skips into 19 failures that look exactly like a regression -- see predictor/run_benchmark.sbatch:20.
stage pytest_predictor $PIXI_PRED python -m pytest -q tests/predictor
stage pytest_deprecated $PIXI python -m pytest -q archive/deprecated
stage selftest $PIXI python wtune.py selftest
stage compliance_validate $PIXI python wtune.py compliance validate
# The transition controller has checks that already fail on the base branch; this stage passes when the
# branch introduces no new failing check relative to a clean export of BASE (default origin/main).
BASE=${BASE:-origin/main}
transition_no_regression() {
  local tmp; tmp=$(mktemp -d /tmp/wtune-base-XXXXXX)
  git archive "$BASE" | tar -x -C "$tmp"
  (cd "$tmp" && $PIXI python src/wtune_transition_controller.py selftest) | grep -E ' FAIL' | sed 's/ \[.*//; s/ *FAIL$//' | sort > "$tmp/base.fail"
  $PIXI python src/wtune_transition_controller.py selftest | grep -E ' FAIL' | sed 's/ \[.*//; s/ *FAIL$//' | sort > "$tmp/branch.fail"
  echo "base ($BASE $(git rev-parse --short "$BASE")) failing checks: $(wc -l < "$tmp/base.fail")"; sed 's/^/  base: /' "$tmp/base.fail"
  echo "branch failing checks: $(wc -l < "$tmp/branch.fail")"; sed 's/^/  branch: /' "$tmp/branch.fail"
  local new; new=$(comm -13 "$tmp/base.fail" "$tmp/branch.fail")
  rm -rf "$tmp"
  if [ -n "$new" ]; then echo "new failures vs base:"; echo "$new" | sed 's/^/  NEW: /'; return 1; fi
  echo "no new failures vs base"
}
if [ -f src/wtune_transition_controller.py ]; then
  stage transition_no_regression transition_no_regression
fi
if [ "$fail" = 0 ]; then echo "ALL STAGES PASS"; else echo "SOME STAGES FAILED"; fi
exit $fail

#!/usr/bin/env bash
set -euo pipefail
WT_TEST_BIN=/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/.pixi/envs/default/bin
WT_TEST_TMP=/home/u5727520/wrf-stack/wtune/artifacts/20260913-node-local-physics/root-check
exec env CC="$WT_TEST_BIN/x86_64-conda-linux-gnu-cc" \
  FC="$WT_TEST_BIN/x86_64-conda-linux-gnu-gfortran" \
  TMPDIR="$WT_TEST_TMP" TMP="$WT_TEST_TMP" TEMP="$WT_TEST_TMP" \
  PYTHONDONTWRITEBYTECODE=1 "$WT_TEST_BIN/python" "$@"

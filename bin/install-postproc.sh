#!/usr/bin/env bash
# bin/install-postproc.sh -- install the post-processing tools of `wtune post` that are not in
# envs/wtune: MET + nccmp into the spack environment envs/postproc/, ParaView into the pixi
# environment envs/paraview/; then check that each one runs. Needs an initialized spack
# (`spack` on PATH) and pixi. Run it on a compute node (`sbatch bin/install-postproc.sbatch`):
# the ParaView check renders offscreen through the node's EGL.
#
#   bin/install-postproc.sh            install + check
#   bin/install-postproc.sh --check    only check an existing install
set -uo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="$repo/envs/postproc"
jobs="${SPACK_BUILD_JOBS:-$(nproc)}"
pixi_bin="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"
check_only=0
case "${1:-}" in
  --check) check_only=1 ;;
  "") ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

command -v spack >/dev/null 2>&1 || { echo "spack not on PATH -- source spack's setup-env.sh first" >&2; exit 1; }

if [ "$check_only" = 0 ]; then
  # always re-concretize after a spec change so the lockfile matches spack.yaml
  spack -e "$env_dir" concretize -f || exit 1
  spack -e "$env_dir" install -j "$jobs" || exit 1   # no --fail-fast: one run reports every failing package
  "$pixi_bin" install --manifest-path "$repo/envs/paraview/pixi.toml" || exit 1
fi

view="$env_dir/.spack-env/view"
pv="$repo/envs/paraview/.pixi/envs/default/bin"
fail=0
check() {   # name, command...
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "OK    $name"; else echo "FAIL  $name ($*)"; fail=1; fi
}

# MET tools print "Start <tool> V12.1.1" and then exit non-zero on a bare -version
# (captured, not piped: under pipefail the tool's own non-zero exit would fail the pipe)
met_ok() { local out; out="$("$view/bin/$1" -version 2>&1)"; [[ "$out" == *"Start $1 V"* ]]; }
for t in tc_pairs tc_rmw tc_stat grid_stat point_stat ensemble_stat; do
  check "MET $t" met_ok "$t"
done
check "nccmp" "$view/bin/nccmp" --version
shot="$(mktemp -d)/pv_check.png"
check "ParaView pvbatch (offscreen)" "$pv/pvbatch" --force-offscreen-rendering \
  -c "from paraview.simple import *; Sphere(); Show(); SaveScreenshot('$shot')"
echo "export WTUNE_POSTPROC_VIEW=$view"
echo "export WTUNE_PARAVIEW_BIN=$pv"
exit $fail

#!/usr/bin/env bash
#SBATCH --job-name=bp2nc-diff
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/home/u5727520/wrf-bench/atlas/bp2nc-diff-%j.out
# Retry a BP wrfout -> nc conversion off the login node, then diffwrf against a reference nc.
# usage: sbatch --export=ALL,BP=<bp dir>,REF=<nc file> bp2nc_diff.sbatch   (writes $BP.nc, overwrites)
set -uo pipefail
BP=${BP:?}; REF=${REF:?}
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
CONV=/home/u5727520/SC26-WRF-p0/scripts/vendor/wrf-adios2-to-netcdf4/convert.py
module load intel/2024_01_46 2>/dev/null || true
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
rm -f "$BP.nc"
s=$(date +%s)
( cd "$ENV" && pixi run python "$CONV" --input "$BP" --output "$BP.nc" ) 2>&1 | grep -v 'WARN cache' | tail -5
echo "convert rc=${PIPESTATUS[0]} in $(( $(date +%s) - s )) s -> $(ls -la "$BP.nc" 2>/dev/null)"
[ -f "$BP.nc" ] || exit 1
echo "--- diffwrf $BP.nc  vs  $REF"
"$W/external/io_netcdf/diffwrf" "$BP.nc" "$REF" 2>&1 | tail -15

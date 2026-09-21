#!/usr/bin/env bash
#SBATCH --job-name=diffctl
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/diffctl-%j.out
# diffwrf parser positive control with a SAME-DATE perturbed frame (diffwrf refuses to compare
# different dates). Expect differing_fields >= 2 (T gross, QVAPOR 1 ulp) and 0 for self.
module load intel/2024_01_46 2>/dev/null || true
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF; A=/home/u5727520/wrf-bench/wscore
mkdir -p $A/diffctl; cd $A/diffctl
REF=$A/V13a_1073854/wrfout_d01_2026-09-02_00:10:00
PERT=$A/diffctl/wrfout_d01_2026-09-02_00:10:00
# pixi python must not inherit the Intel LD_LIBRARY_PATH
env -u LD_LIBRARY_PATH /home/u5727520/.pixi/bin/pixi run --manifest-path /home/u5727520/wrf-stack/wtune-pixi-env-20260830/pixi.toml python /home/u5727520/wrf-stack/wtune/rtctl/score/perturb.py "$REF" "$PERT" || { echo FATAL perturb; exit 2; }
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
count () { awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' "$1"; }
$W/external/io_netcdf/diffwrf $REF $PERT > ref_vs_pert.txt 2>&1; echo "ref vs perturbed differing_fields=$(count ref_vs_pert.txt)"
$W/external/io_netcdf/diffwrf $REF $REF > ref_vs_ref.txt 2>&1; echo "ref vs ref differing_fields=$(count ref_vs_ref.txt)"
sed -n '1,12p' ref_vs_pert.txt
rm -f "$PERT"

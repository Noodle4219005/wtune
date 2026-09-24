#!/usr/bin/env bash
# f1 site (x86 Sapphire Rapids, 112 cores/node, Intel MPI). Decision 0050 step 8 minimal
# hardware abstraction. Every value and command below is copied byte-for-byte from
# strict_campaign.sbatch as it stood before SITE existed -- this file must never change what
# f1 runs. Sourced by strict_campaign.sbatch (and, for the plain variables only, by
# campaign/decomp_search_campaign.sbatch) once CAMPAIGN is known.
#
# Defines:
#   CORES_PER_NODE, W, OUT, HARDWARE_ID, PIXI_ENV, CONV, CASES_JSON_DEFAULT -- plain variables.
#   site_env()     -- module load + MPI/NETCDF/LD_LIBRARY_PATH exports; call once per job, after
#                      mkdir -p "$OUT".
#   site_launch()  -- see below; call once per WRF run.

CORES_PER_NODE=112
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
OUT=/home/u5727520/wrf-bench/wscore/campaign
# derived from the node pool (icpnq = ct448 SNC 8x14, icpnp = cf448 2 NUMA) so ledgers from
# different pools never share a hardware_id
HARDWARE_ID=f1-spr-$(hostname -s | sed 's/[0-9]*$//')
PIXI_ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
# vendored into this repo 2026-09-15: the old external path
# /home/u5727520/SC26-WRF-p0/scripts/vendor/wrf-adios2-to-netcdf4/convert.py disappeared, and
# physics smoke 1076771-1076776 silently produced no netCDF / no fidelity at all.
CONV=${CONV:-/home/u5727520/wrf-stack/wtune-autotuner/src/tools/wrf_adios2_to_netcdf4/convert.py}
CASES_JSON_DEFAULT=/home/u5727520/wrf-stack/wtune-autotuner/campaign/cases.json

site_env () {
  module load intel/2024_01_46 2>/dev/null || true
  export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false
  for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
  export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
}

# site_launch <ranks> <tpn> <omp> <dist>: fills the global array SITE_LAUNCH_CMD with the exact
# launch command (launcher + binary). Caller runs it as:
#   timeout "$TIMEOUT_S" "${SITE_LAUNCH_CMD[@]}" > wrf.out 2>&1
site_launch () {
  local ranks="$1" tpn="$2" omp="$3" dist="$4"
  SITE_LAUNCH_CMD=(srun --mpi=pmi2 --distribution="$dist" -n "$ranks" \
    --ntasks-per-node="$tpn" -c "$omp" ./wrf.exe)
}

# site_hygiene <mode>: no-op. site_launch above uses srun (native Slurm task launch), not
# mpirun/prte, so ranks stay in the job step's own process group the whole time and Slurm's
# own tracking reaps them on TIMEOUT/kill like any other step. f1 has no equivalent of the scc
# orphan-wrf.exe incident (Open MPI 5's prte spawning ranks outside Slurm's process-group
# tracking under ProctrackType=pgid, see campaign/sites/scc.sh's site_hygiene) to guard
# against. Kept as a callable no-op so strict_campaign.sbatch/temporal_run.sbatch need no
# per-site branch of their own.
site_hygiene () {
  SITE_HYGIENE_CMD=(true)
}

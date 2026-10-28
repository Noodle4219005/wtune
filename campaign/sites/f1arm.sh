#!/usr/bin/env bash
# f1-arm site (Grace CPU Superchip: 144 Neoverse-V2 cores/node in two 72-core NUMA domains,
# SMT off; Open MPI 5.0.10 from a Spack view, nvhpc 25.5). The toolchain comes from the
# machine's env_source ($HOME/wtune-f1arm/env.sh). f1 refuses a job without --account
# (machine.account) and has no login node inside its partitions (machine.login_nodes '').
#
# Defines the same names as scc.sh: CORES_PER_NODE, W, OUT, HARDWARE_ID, OMP_RUNTIME, PIXI_ENV,
# CONV, CASES_JSON_DEFAULT, site_env(), site_launch(), site_hygiene().

CORES_PER_NODE=${WTUNE_CORES_PER_NODE:-144}
NUMA_PER_NODE=${WTUNE_NUMA_PER_NODE:-2}
W=${WRF_TREE:-$HOME/wrf-env/nvhpc-arm/wrf/nvhpc-adios2-tuned-dmpar/WRF}
OUT=${WTUNE_CAMPAIGN_OUT:-$HOME/wtune-runs/campaign}
HARDWARE_ID=${WTUNE_HARDWARE_ID:-f1-grace}
OMP_RUNTIME=nvhpc
QUOTA_FS=home2           # mmlsquota filesystem: writes stop at the 100 G soft limit
# 100 G quotas: the default 60 GiB (sized for scc home1, 1000 G) aborted every probe at 75 G used
# (haitang2017, 2026-10-28); a probe writes rank 0's rsl and no history
MIN_FREE_GIB=${MIN_FREE_GIB:-5}
export PATH=$HOME/.pixi/bin:$PATH
PIXI_ENV=${WTUNE_PIXI_ENV:-$HOME/wtune-envs/campaign}
CONV=${CONV:-$(dirname "$CAMPAIGN")/src/wtune/tools/wrf_adios2_to_netcdf4/convert.py}
CASES_JSON_DEFAULT=$CAMPAIGN/cases.json

site_env () {
  source "${WTUNE_ENV_SOURCE:-$HOME/wtune-f1arm/env.sh}"
  ulimit -s unlimited      # nvhpc -Mstack_arrays: automatic arrays live on the stack
  # this Open MPI has no UCX: keep its TCP on the IPoIB network, not the 1 Gb/s eno1
  # (conus12 at 4 nodes: 0.462 -> 0.053 s/step)
  export OMPI_MCA_btl_tcp_if_include=${OMPI_MCA_btl_tcp_if_include:-10.200.0.0/14}
  # no shared-memory BTL (~3 % more s/step on one node, 3.01 vs 2.93). It was blamed for the
  # haitang2017 first-step SIGSEGVs, but those were mp_physics 7 writing refl_10cm with
  # do_radar_ref = 0 (fixed by forcing do_radar_ref = 1); btl/sm has not been re-measured since
  export OMPI_MCA_btl=${OMPI_MCA_btl:-self,tcp}
}

# site_launch <ranks> <tpn> <omp> <dist>: fills SITE_LAUNCH_CMD, same contract as scc.sh's.
# Each die owns its memory, so the ranks of a node are split evenly over its NUMA domains;
# a per-node rank count the domains do not divide is mapped per node instead.
site_launch () {
  local ranks="$1" tpn="$2" omp="$3" dist="$4"
  local nn="${SLURM_JOB_NUM_NODES:-1}"
  [ $(( tpn * nn )) -lt "$ranks" ] && tpn=$(( (ranks + nn - 1) / nn ))
  local map="ppr:${tpn}:node:PE=${omp}"
  if [ $(( tpn % NUMA_PER_NODE )) -eq 0 ]; then
    map="ppr:$(( tpn / NUMA_PER_NODE )):numa:PE=${omp}"
  fi
  SITE_LAUNCH_CMD=(mpirun -np "$ranks" --map-by "$map" --bind-to core)
  if [ "$dist" = "cyclic" ]; then
    SITE_LAUNCH_CMD+=(--rank-by node)
  fi
  SITE_LAUNCH_CMD+=(./wrf.exe)
}

# site_hygiene <mode>: no-op. f1's Slurm tracks a job's processes in a cgroup
# (ProctrackType=proctrack/cgroup), so nothing outlives the job.
site_hygiene () {
  SITE_HYGIENE_CMD=(true)
}

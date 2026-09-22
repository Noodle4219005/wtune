#!/usr/bin/env bash
# scc site (GH200, 72 Neoverse-V2 cores/node, SMT off, one NUMA node with memory; Open MPI
# 5.0.10 / HPC-X 2.50 via nvhpc 26.5). Decision 0050 step 8 minimal hardware abstraction.
#
# MEASURED (real MPI program, 2 nodes, inside a Slurm allocation):
#   - plain `mpirun -np N` works across nodes.
#   - `srun --mpi=pmix` is BROKEN (PMIx help files missing).
#   - `srun --mpi=pmi2` silently turns every rank into a singleton -- do not use srun here at all.
#   - `--cpu-set 0-71` breaks multi-node placement under OMPI5 -- do not add it; OMP_PLACES/
#     OMP_PROC_BIND (in site_launch) handle in-rank thread pinning instead of a raw cpu-set.
#   - OpenMP here is nvhpc's runtime, not Intel's: KMP_BLOCKTIME does not exist as a concept
#     under it (record it as "n/a" in the ledger; strict_campaign.sbatch's configure_omp_env
#     never exports it on this site).
#
# No `module` command on scc -- the whole toolchain/env comes from $HOME/wtune-scc/env.sh.
# Repo at $HOME/wtune; output dir $HOME/wtune-runs/campaign; login node `head` is a member of
# the (only) partition `compute`, so every scc sbatch command must add --exclude=head
# (predictor/experiment.py's --site scc path does this; see SITE_TABLE there).
#
# Defines the same four names as f1.sh: CORES_PER_NODE, W, OUT, HARDWARE_ID, PIXI_ENV, CONV,
# CASES_JSON_DEFAULT, site_env(), site_launch().

CORES_PER_NODE=72
W=${WRF_TREE:-$HOME/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF}
OUT=$HOME/wtune-runs/campaign
HARDWARE_ID=scc-gh200
# pixi lives in ~/.pixi/bin, which is not on PATH inside a batch job (smoke job 437: "pixi: command not found").
export PATH=$HOME/.pixi/bin:$PATH
PIXI_ENV=$HOME/wtune-envs/campaign
CONV=${CONV:-$HOME/wtune/tools/wrf_adios2_to_netcdf4/convert.py}
CASES_JSON_DEFAULT=$HOME/wtune/campaign/cases.json

site_env () {
  # nvhpc 26.5 + HPC-X 2.50 / Open MPI 5.0.10; sets NETCDF/HDF5/PNETCDF/ADIOS2_DIR/BLOSC2_LIB,
  # WRF_TREE, OMP_STACKSIZE=1G, and PATH/LD_LIBRARY_PATH for all of the above.
  source "$HOME/wtune-scc/env.sh"
}

# site_launch <ranks> <tpn> <omp> <dist>: fills the global array SITE_LAUNCH_CMD, same contract
# as f1.sh's. --map-by ppr:<tpn>:node:PE=<omp> --bind-to core places tpn ranks per node with omp
# cores bound to each rank; --rank-by node reproduces WRF's cyclic across-node rank order for
# dist=cyclic (block is OMPI5's own default rank-by, so the block case adds nothing extra).
site_launch () {
  local ranks="$1" tpn="$2" omp="$3" dist="$4"
  SITE_LAUNCH_CMD=(mpirun -np "$ranks" --map-by "ppr:${tpn}:node:PE=${omp}" --bind-to core)
  if [ "$dist" = "cyclic" ]; then
    SITE_LAUNCH_CMD+=(--rank-by node)
  fi
  SITE_LAUNCH_CMD+=(./wrf.exe)
}

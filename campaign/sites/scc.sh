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

CORES_PER_NODE=${WTUNE_CORES_PER_NODE:-72}
W=${WRF_TREE:-$HOME/wrf-stack/wrf/nvhpc-fast-adios2-dmsm/WRF}
OUT=$HOME/wtune-runs/campaign
HARDWARE_ID=${WTUNE_HARDWARE_ID:-scc-gh200}
OMP_RUNTIME=nvhpc        # no KMP_BLOCKTIME: threads are pinned with OMP_PLACES/OMP_PROC_BIND
# pixi lives in ~/.pixi/bin, which is not on PATH inside a batch job (smoke job 437: "pixi: command not found").
export PATH=$HOME/.pixi/bin:$PATH
PIXI_ENV=$HOME/wtune-envs/campaign
# derived from CAMPAIGN (this checkout's campaign/ dir, exported by the sbatch script) so the
# converter comes from the SAME tree as the script: $HOME/wtune has no src/wtune/tools at all, and the
# tree wtune deploys to is $HOME/wtune-src. Equals the old value for CAMPAIGN=$HOME/wtune/campaign.
CONV=${CONV:-$(dirname "$CAMPAIGN")/src/wtune/tools/wrf_adios2_to_netcdf4/convert.py}
CASES_JSON_DEFAULT=$CAMPAIGN/cases.json

site_env () {
  # nvhpc 26.5 + HPC-X 2.50 / Open MPI 5.0.10; sets NETCDF/HDF5/PNETCDF/ADIOS2_DIR/BLOSC2_LIB,
  # WRF_TREE, OMP_STACKSIZE=1G, and PATH/LD_LIBRARY_PATH for all of the above.
  source "${WTUNE_ENV_SOURCE:-$HOME/wtune-scc/env.sh}"
  # The nvhpc build uses -Mstack_arrays: automatic arrays live on the stack, and the 8 MiB default
  # segfaulted every rank (rc=139, smoke job 439). run-case.sbatch already did this.
  ulimit -s unlimited
}

# site_launch <ranks> <tpn> <omp> <dist>: fills the global array SITE_LAUNCH_CMD, same contract
# as f1.sh's. --map-by ppr:<tpn>:node:PE=<omp> --bind-to core places tpn ranks per node with omp
# cores bound to each rank; --rank-by node reproduces WRF's cyclic across-node rank order for
# dist=cyclic (block is OMPI5's own default rank-by, so the block case adds nothing extra).
site_launch () {
  local ranks="$1" tpn="$2" omp="$3" dist="$4"
  # Callers pass tpn = ranks / nodes with integer division, which floors. When the legal anchor
  # rank count does not divide by the node count (c34 121 ranks on 2 nodes -> ppr:60 maps only 120)
  # mpirun refuses with rc=213 before WRF starts (jobs 760, 773). Round up instead: 61+60 fits.
  # Computed here because this file is sourced at run time, so already-queued jobs pick it up.
  local nn="${SLURM_JOB_NUM_NODES:-1}"
  [ $(( tpn * nn )) -lt "$ranks" ] && tpn=$(( (ranks + nn - 1) / nn ))
  SITE_LAUNCH_CMD=(mpirun -np "$ranks" --map-by "ppr:${tpn}:node:PE=${omp}" --bind-to core)
  if [ "$dist" = "cyclic" ]; then
    SITE_LAUNCH_CMD+=(--rank-by node)
  fi
  SITE_LAUNCH_CMD+=(./wrf.exe)
}

# site_hygiene <mode>: fills SITE_HYGIENE_CMD with the per-node dispatch for
# campaign/node_hygiene.sh (mode: reap | census | reap+census). scc's Slurm has
# ProctrackType=pgid and no TaskPlugin; Open MPI 5's prte launcher puts the ranks it spawns
# into their own process group, so a job that ends on TIMEOUT -- or whose per-run watchdog
# kills mpirun -- leaves those ranks invisible to Slurm's cleanup. Node census (2026-09) found
# orphaned wrf.exe owned by our user on 8/12 compute nodes, 8-20h old, each spinning a full
# core on CPU 64 -- pure-MPI runs on those nodes ran ~35% slower, hybrid MPI+OpenMP 20-130x
# slower. srun is unreliable here (see the site_launch note above), so hygiene is dispatched
# the same way as WRF itself: mpirun can launch a non-MPI executable, and
# --map-by ppr:1:node runs exactly one copy of the helper per node in the allocation.
site_hygiene () {
  local mode="$1"
  SITE_HYGIENE_CMD=(mpirun -np "$ALLOC_NODES" --map-by "ppr:1:node" "$CAMPAIGN/node_hygiene.sh" "$mode")
}

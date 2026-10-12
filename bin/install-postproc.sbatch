#!/bin/bash
# Batch wrapper for bin/install-postproc.sh: builds MET / ParaView / VAPOR / nccmp on one
# compute node. Site-specific options (partition, excluded login nodes, spack setup) go on the
# sbatch command line or in SPACK_SETUP, e.g. on scc:
#   SPACK_SETUP=~/load_spack.sh sbatch --exclude=head bin/install-postproc.sbatch
#SBATCH --job-name=wtune-postproc-install
#SBATCH -N 1 --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=install-postproc-%j.out
set -uo pipefail
# shellcheck disable=SC1090
[ -n "${SPACK_SETUP:-}" ] && source "$SPACK_SETUP"
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
bin/install-postproc.sh "$@"

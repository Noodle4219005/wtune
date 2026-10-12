#!/usr/bin/env bash
# run_std_benchmarks.sh -- established benchmarks for the machine profile (decision 0052):
# OSU micro-benchmarks (MPI) and IOR (POSIX / MPI-IO / PnetCDF / HDF5), sized after WRF.
# The hand-written machine/bench/* cover clock, STREAM, WRF-like kernels and OpenMP; MPI and
# parallel I/O come from these standard tools so the numbers are comparable across machines.
#
# Needs a 2-node allocation (inter-node MPI):
#   sbatch -N2 --ntasks-per-node=<cores> --exclusive run_std_benchmarks.sh
# Env:
#   OSU_DIR   libexec/osu-micro-benchmarks/mpi of an OSU install     (required)
#   IOR       path to the ior binary                                  (required)
#   IO_DIR    directory to write IOR files to -- the SAME filesystem WRF writes on (required)
#   OUT_DIR   where raw outputs go (default .)
#   CORES     ranks per node for full-node tests (default $SLURM_NTASKS_PER_NODE or 72)
#   IOR_BLOCK per-rank IOR block (default 20m: 144 ranks x 20 MiB ~= one CONUS 2.5 km frame)
#   MPIRUN    default mpirun
# ADIOS2 is measured through WRF itself (io_form 14), not here.
set -uo pipefail
: "${OSU_DIR:?}" "${IOR:?}" "${IO_DIR:?}"
OUT_DIR=${OUT_DIR:-.}
CORES=${CORES:-${SLURM_NTASKS_PER_NODE:-72}}
IOR_BLOCK=${IOR_BLOCK:-20m}
MPIRUN=${MPIRUN:-mpirun}
NODES=${SLURM_JOB_NUM_NODES:-1}
NP=$((NODES * CORES))
mkdir -p "$OUT_DIR" "$IO_DIR"
TAG=$(hostname -s)-$(date +%Y%m%d-%H%M%S)
log() { echo "== $*" >&2; }

osu() {  # osu <label> <mpirun args...> -- <binary> [args]
  local label=$1; shift
  log "osu $label"
  "$MPIRUN" "$@" > "$OUT_DIR/osu_${label}.txt" 2>&1 || log "osu $label rc=$?"
}
P=$OSU_DIR/pt2pt; C=$OSU_DIR/collective
osu latency_intra -np 2 --map-by core --bind-to core "$P/osu_latency"
osu bw_intra      -np 2 --map-by core --bind-to core "$P/osu_bw"
if [ "$NODES" -ge 2 ]; then
  osu latency_inter -np 2 --map-by node --bind-to core "$P/osu_latency"
  osu bw_inter      -np 2 --map-by node --bind-to core "$P/osu_bw"
  osu bibw_inter    -np 2 --map-by node --bind-to core "$P/osu_bibw"
  # every core on node 1 paired with one on node 2: closest to a full-node halo exchange
  osu mbw_mr_inter  -np "$((2 * CORES))" --map-by "ppr:${CORES}:node" --bind-to core "$P/osu_mbw_mr" -m 1:4194304
fi
osu allreduce_full -np "$NP" --map-by "ppr:${CORES}:node" --bind-to core "$C/osu_allreduce" -m 8:1048576
osu alltoall_full  -np "$NP" --map-by "ppr:${CORES}:node" --bind-to core "$C/osu_alltoall" -m 8:65536

ior() {  # ior <label> <np> <ppr> <ior args...>
  local label=$1 np=$2 ppr=$3; shift 3
  log "ior $label"
  rm -rf "$IO_DIR/ior_$label"*
  "$MPIRUN" -np "$np" --map-by "ppr:${ppr}:node" --bind-to core "$IOR" "$@" \
    -o "$IO_DIR/ior_$label" > "$OUT_DIR/ior_${label}.txt" 2>&1 || log "ior $label rc=$?"
  rm -rf "$IO_DIR/ior_$label"*
}
# single writer: WRF io_form 2 funnels a whole frame through rank 0
ior posix_single 1 1 -a POSIX -b 2880m -t 4m -w -e -i 3
# all ranks, one shared file (io_form 11 = PnetCDF collective; MPI-IO and HDF5 for comparison)
ior posix_fpp     "$NP" "$CORES" -a POSIX -F -b "$IOR_BLOCK" -t 4m -w -e -i 3
ior mpiio_shared  "$NP" "$CORES" -a MPIIO -c -b "$IOR_BLOCK" -t 4m -w -e -i 3
ior ncmpi_shared  "$NP" "$CORES" -a NCMPI -c -b "$IOR_BLOCK" -t 4m -w -e -i 3
ior hdf5_shared   "$NP" "$CORES" -a HDF5  -c -b "$IOR_BLOCK" -t 4m -w -e -i 3

log "done: raw outputs in $OUT_DIR (tag $TAG)"

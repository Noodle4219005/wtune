#!/usr/bin/env bash
#SBATCH --job-name=schedbench
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/fingerprint/schedbench-%j.out
# Empirical check: can any runtime environment setting change iteration ownership of an OpenMP loop compiled
# exactly like WRF's micro tile loop (no SCHEDULE clause -> __kmpc_for_static_init_4, schedule 34)?
set -uo pipefail
S=/home/u5727520/wrf-stack/wtune/rtctl/fingerprint/schedbench
B=/home/u5727520/wrf-bench/wscore/fingerprint/schedbench_$SLURM_JOB_ID; mkdir -p "$B"; cd "$B"
module load intel/2024_01_46 2>/dev/null || true
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
FL="-qopenmp -fpp -auto -O3 -xSAPPHIRERAPIDS -qopt-zmm-usage=high -ip -fp-model precise -align all -fno-alias"
ifx $FL "$S/sched_bench.F90" -o sb_noclause || exit 1
ifx $FL -DRUNTIME_SCHED "$S/sched_bench.F90" -o sb_runtime || exit 1
for x in sb_noclause sb_runtime; do
  echo "== ABI $x: $(ldd $x | awk '/libiomp5/{print $3}')"
  objdump -d --no-show-raw-insn -M intel $x | grep -B8 -E "call.*__kmpc_(for_static_init|dispatch_init)" | grep -E "edx|__kmpc" | sed 's/^/  /'
done
export OMP_NUM_THREADS=2 OMP_PROC_BIND=false KMP_BLOCKTIME=0 OMP_STACKSIZE=512M
run() { local tag=$1; shift; for x in sb_noclause sb_runtime; do env "$@" ./$x "$tag"; done; }
run base
run omp_dynamic1        OMP_SCHEDULE=dynamic,1
run omp_guided1         OMP_SCHEDULE=guided,1
run omp_static_steal    OMP_SCHEDULE=static_steal
run kmp_static_greedy   KMP_SCHEDULE=static,greedy
run kmp_static_balanced KMP_SCHEDULE=static,balanced
run kmp_force_mono_dyn  KMP_FORCE_MONOTONIC_DYNAMIC_SCHEDULE=1 OMP_SCHEDULE=dynamic,1
echo "=== done"

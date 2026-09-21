#!/usr/bin/env bash
#SBATCH --job-name=ompknobs
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/fingerprint/ompknobs-%j.out
# Completes the ABI gate: compiler options that could change the schedule of an OpenMP loop without a clause,
# and static-loop variants the linked libiomp5 implements. Read-only.
set -uo pipefail
module load intel/2024_01_46 2>/dev/null || true
echo "== ifx $(ifx --version 2>&1 | head -1)"
ifx -help 2>&1 | grep -iE "openmp|sched|parallel|par-" | sed 's/^/  /'
echo "== ifort present? $(command -v ifort || echo no)"
command -v ifort >/dev/null && ifort -help 2>&1 | grep -iE "par-schedule|parallel|qopenmp" | head -20 | sed 's/^/  /'
L=/home/u5727520/wrf-stack/intel/lib/libiomp5.so
echo "== libiomp5 strings: static variants / steal"
strings -a "$L" | grep -iE "^(static|dynamic|guided|trapezoidal|steal|greedy|balanced|static_steal|static_balanced|static_greedy)[a-z_,]*$" | sort -u | sed 's/^/  /'
strings -a "$L" | grep -iE "KMP_SCHEDULE|static_steal|balanced_chunked" | sort -u | head -20 | sed 's/^/  /'
echo "=== done"

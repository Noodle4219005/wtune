#!/usr/bin/env bash
#SBATCH --job-name=ompabi
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/fingerprint/ompabi-%j.out
#
# Gate "OpenMP ABI of the hot micro loop" for the pristine production wrf.exe. Read-only: no WRF run.
#  1. which OpenMP loop-init entry points every function calls (__kmpc_for_static_init_* vs __kmpc_dispatch_*)
#  2. the schedule constant passed at each loop-init call inside microphysics / Thompson code
#     (LLVM/Intel kmp_sched: 33 static chunked, 34 static, 35 dynamic, 36 guided, 37 runtime, 38 auto)
#  3. SCHEDULE clauses present in the pristine WRF source (a schedule(runtime) loop would obey OMP_SCHEDULE)
#  4. schedule-related env names the linked libiomp5 knows, and schedule-related ifx options
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
EXE=$W/main/wrf.exe
O=/home/u5727520/wrf-bench/wscore/fingerprint/ompabi_$SLURM_JOB_ID
mkdir -p "$O"
module load intel/2024_01_46 2>/dev/null || true
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

echo "== binary"
echo "exe=$EXE sha256=$(sha256sum "$EXE" | cut -c1-16)"
readelf -h "$EXE" | grep -E "Type:"
readelf -p .comment "$EXE" 2>/dev/null | grep -v "^String dump" | sort -u | head -8
echo "defined text symbols: $(nm --defined-only "$EXE" | awk '$2 ~ /[tT]/' | wc -l)"

echo "== OpenMP runtime"
ldd "$EXE" | grep -iE "omp"
LIBOMP=$(ldd "$EXE" | awk '/libiomp5/{print $3; exit}')
if [ -n "$LIBOMP" ]; then
  echo "libiomp5=$LIBOMP sha256=$(sha256sum "$LIBOMP" | cut -c1-16)"
  strings -a "$LIBOMP" | grep -E '^(KMP|OMP)_[A-Z_]*SCHED[A-Z_]*$' | sort -u
fi
echo "== ifx schedule/openmp options"
ifx -help 2>&1 | grep -iE "sched|qopenmp" | head -20

echo "== census: OpenMP loop-init calls per function (whole binary)"
objdump -d --no-show-raw-insn -M intel "$EXE" | awk '
  /^[0-9a-f]+ <.*>:$/ { fn = $2; gsub(/[<>:]/, "", fn); next }
  /call/ && /__kmpc_(for_static_init|dispatch_init|dispatch_next|dist_for_static_init|fork_call)/ {
    match($0, /<__kmpc_[a-z_0-9]+/); c = substr($0, RSTART + 1, RLENGTH - 1); n[fn "," c]++ }
  END { for (k in n) print k "," n[k] }' | sort > "$O/kmpc_calls_by_function.csv"
awk -F, '{t[$2] += $3; f[$2]++} END {for (k in t) printf "  %-32s calls=%d functions=%d\n", k, t[k], f[k]}' "$O/kmpc_calls_by_function.csv" | sort
echo "functions calling a dispatch (dynamic/guided/runtime) entry:"
awk -F, '$2 ~ /dispatch/ {print "  " $0}' "$O/kmpc_calls_by_function.csv" | head -40
echo "micro/Thompson functions with OpenMP calls:"
grep -iE "microphysics|mp_gt_driver|thompson" "$O/kmpc_calls_by_function.csv" | sed 's/^/  /'

echo "== schedule argument at loop-init calls in micro/Thompson code (3rd arg = edx)"
nm -S --defined-only "$EXE" | awk 'NF == 4 && tolower($4) ~ /microphysics_driver|mp_gt_driver|module_mp_thompson/' > "$O/micro_symbols.txt"
echo "micro symbols: $(wc -l < "$O/micro_symbols.txt")"
while read -r addr size typ name; do
  s=$((16#$addr)); e=$((s + 16#$size))
  objdump -d --no-show-raw-insn -M intel --start-address=$s --stop-address=$e "$EXE" \
    | grep -B12 -E "call.*__kmpc_(for_static_init|dispatch_init|fork_call)" > "$O/ctx.tmp" || true
  if [ -s "$O/ctx.tmp" ]; then echo "### $name"; cat "$O/ctx.tmp"; fi
done < "$O/micro_symbols.txt" > "$O/micro_kmpc_context.txt"
rm -f "$O/ctx.tmp"
grep -E "^###|edx|__kmpc" "$O/micro_kmpc_context.txt" | head -80

echo "== pristine source: OpenMP loop directives and SCHEDULE clauses"
grep -n -i -A3 '^ *!\$OMP *PARALLEL *DO' "$W/phys/module_microphysics_driver.F" | head -30
echo "PARALLEL DO directives in phys+dyn_em+share: $(grep -rhi '^ *!\$OMP *PARALLEL *DO' "$W"/phys/*.F "$W"/dyn_em/*.F "$W"/share/*.F 2>/dev/null | wc -l)"
echo "SCHEDULE clauses (any !\$OMP line) by kind:"
grep -rhi '!\$OMP' "$W"/phys/*.F "$W"/dyn_em/*.F "$W"/share/*.F "$W"/frame/*.F 2>/dev/null \
  | grep -ioE 'schedule *\( *[a-z_]+' | tr 'A-Z' 'a-z' | tr -d ' ' | sort | uniq -c
echo "files with a SCHEDULE clause:"
grep -rli '!\$OMP.*schedule' "$W"/phys/*.F "$W"/dyn_em/*.F "$W"/share/*.F "$W"/frame/*.F 2>/dev/null | sed 's/^/  /'
sha256sum "$O/kmpc_calls_by_function.csv" | cut -c1-16
echo "=== done"

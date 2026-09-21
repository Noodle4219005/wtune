#!/usr/bin/env bash
#SBATCH --job-name=hwprobe
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/hwprobe-%j.out
#
# Capability probe on an icpnq compute node (same pool as ct448) for the slack-stealing gates:
# SMT, RDT/resctrl, cpufreq/turbo, user-mode perf events (stat / record / attach / symbols),
# NUMA page migration permission and cost. Read-only except for its own dummy processes.
set -u
module load intel/2024_01_46 2>/dev/null || true
P=/home/u5727520/wrf-stack/wtune/rtctl
D=/home/u5727520/wrf-bench/wscore/hwprobe-$SLURM_JOB_ID; mkdir -p "$D"; cd "$D"
C=/sys/devices/system/cpu
echo "== node $(hostname) kernel $(uname -r)"
lscpu | grep -E 'Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|L3 cache'
echo "== SMT control=$(cat $C/smt/control 2>&1) active=$(cat $C/smt/active 2>&1) cpu0 siblings=$(cat $C/cpu0/topology/thread_siblings_list 2>&1)"
echo "== RDT cpu flags: $(grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -xE 'cat_l3|cdp_l3|mba|cqm|cqm_llc|cqm_occup_llc|cqm_mbm_total|cqm_mbm_local|rdt_a' | tr '\n' ' ')"
echo "== resctrl mounted=$(mount | grep -c resctrl) dir=$(ls -ld /sys/fs/resctrl 2>&1) contents=[$(ls /sys/fs/resctrl 2>&1 | tr '\n' ' ')]"
echo "== cpufreq driver=$(cat $C/cpu0/cpufreq/scaling_driver 2>&1) governor=$(cat $C/cpu0/cpufreq/scaling_governor 2>&1) max=$(cat $C/cpu0/cpufreq/scaling_max_freq 2>&1) cur=$(cat $C/cpu0/cpufreq/scaling_cur_freq 2>&1) boost=$(cat $C/cpufreq/boost 2>&1) intel_pstate=[$(ls $C/intel_pstate 2>&1 | tr '\n' ' ')]"
echo "== powercap=[$(ls /sys/class/powercap 2>&1 | tr '\n' ' ')] energy_uj=$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>&1)"
echo "== perf paranoid=$(cat /proc/sys/kernel/perf_event_paranoid) $(perf --version 2>&1)"
echo "== uncore IMC events listed: $(perf list 2>/dev/null | grep -ciE 'uncore_imc|unc_m_cas')"

icx -O2 -qopenmp "$P/poc/elastic_bench.c" -o eb || { echo FATAL icx; exit 2; }
echo "start 0" > st
echo "== perf stat (user mode, 2-thread dummy)"
OMP_NUM_THREADS=2 perf stat -x, -e cycles:u,instructions:u,ref-cycles:u,LLC-loads:u,LLC-load-misses:u,cpu/event=0xa3,umask=0x14,name=stalls_l3_miss/u ./eb 8 30 st > /dev/null 2> stat.txt
cat stat.txt

echo "== perf record, fixed periods, per-event per-symbol aggregation"
OMP_NUM_THREADS=2 perf record -q -o rec.data -e cycles:u -c 2000003 -e instructions:u -c 4000037 -e ref-cycles:u -c 2000003 -e LLC-load-misses:u -c 10007 ./eb 8 30 st > /dev/null 2> rec.err
echo "record rc=$? $(tail -1 rec.err)"
perf script -i rec.data -F event,period,sym 2>/dev/null | awk '{ev=$1; sub(":$","",ev); p=$2; $1="";$2=""; s[ev" "$0]+=p} END{for(k in s) print s[k], k}' | sort -k2,2 -k1,1nr | awk '{c[$2]++} c[$2]<=2'

echo "== perf record attach (-p) to a running process"
OMP_NUM_THREADS=2 ./eb 8 400 st > /dev/null & EP=$!
sleep 1
timeout -s INT 3 perf record -q -o att.data -p $EP -e cycles:u -c 2000003 -e instructions:u -c 4000037 2> att.err
kill $EP 2>/dev/null; wait $EP 2>/dev/null
echo "attach samples: $(perf script -i att.data -F tid 2>/dev/null | wc -l) distinct tids: $(perf script -i att.data -F tid 2>/dev/null | sort -u | wc -l) err: $(tail -1 att.err)"

echo "== NUMA page migration (own process, 4 GB, running)"
cat > memhog.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
int main(int argc, char **argv) {
    size_t n = (size_t)atol(argv[1]) << 30; int secs = atoi(argv[2]);
    char *p = malloc(n); memset(p, 1, n);
    FILE *f = fopen("memhog.ready", "w"); fclose(f);
    time_t t0 = time(0); unsigned x = 0;
    while (time(0) - t0 < secs) { for (size_t i = 0; i < n; i += 4096) p[i] = (char)x; x++; usleep(100000); }
    return 0;
}
EOF
gcc -O2 memhog.c -o memhog
numasum () { awk '{kb=4; for(i=1;i<=NF;i++) if($i ~ /^kernelpagesize_kB=/){split($i,a,"="); kb=a[2]} for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split(substr($i,2),b,"="); s[b[1]]+=b[2]*kb}} END{for(k in s) if(s[k]>1048576) printf "node%s=%.2fGB ", k, s[k]/1048576; print ""}' /proc/$1/numa_maps; }
rm -f memhog.ready
numactl --cpunodebind=0 ./memhog 4 90 & MP=$!
for i in $(seq 1 200); do [ -f memhog.ready ] && break; sleep 0.1; done
echo "before: $(numasum $MP)"
echo "migratepages binary: $(command -v migratepages)"
S=$(date +%s.%N); migratepages $MP 0 1; RC=$?; E=$(date +%s.%N)
echo "migratepages 0->1 rc=$RC wall_s=$(python3 -c "print(round($E-$S,3))")"
echo "after:  $(numasum $MP)"
S=$(date +%s.%N); migratepages $MP 1 0; E=$(date +%s.%N)
echo "migratepages 1->0 wall_s=$(python3 -c "print(round($E-$S,3))") after: $(numasum $MP)"
kill $MP 2>/dev/null; wait $MP 2>/dev/null
echo "== done"

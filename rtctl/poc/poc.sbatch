#!/bin/bash
#SBATCH --job-name=elastic-poc
#SBATCH --account=ACD110018
#SBATCH --partition=ct112
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=4
#SBATCH --time=00:12:00
#SBATCH --output=/home/u5727520/wrf-bench/elastic-poc-%j.out
# Mechanism probe for external elastic-core lending. One node, 8 cores. No WRF.
set -u
module load intel/2024_01_46 2>/dev/null || true
P=/home/u5727520/wrf-stack/wtune/rtctl/poc
D=/home/u5727520/wrf-bench/elastic-poc-$SLURM_JOB_ID; mkdir -p "$D"; cd "$D"
echo "node $(hostname) job $SLURM_JOB_ID"
echo "sched_migration_cost_ns=$(cat /proc/sys/kernel/sched_migration_cost_ns) numa_balancing=$(cat /proc/sys/kernel/numa_balancing)"
icx -O2 -qopenmp "$P/elastic_bench.c" -o elastic_bench || { echo FATAL compile; exit 2; }
ldd ./elastic_bench | grep -i omp
CG=$(grep cpuset /proc/self/cgroup | cut -d: -f3)
CPUS=$(cat /sys/fs/cgroup/cpuset$CG/cpuset.cpus); echo "batch step cgroup $CG cpuset.cpus=$CPUS"
# expand the allocation's core list; c[0..7]
mapfile -t c < <(python3 -c "
import sys
out=[]
for part in '$CPUS'.split(','):
    a,_,b=part.partition('-'); out+=list(range(int(a),int(b or a)+1))
print('\n'.join(map(str,out)))")
for i in "${c[@]}"; do echo "core $i package $(cat /sys/devices/system/cpu/cpu$i/topology/physical_package_id) numa $(ls -d /sys/devices/system/cpu/cpu$i/node* | sed 's/.*node//')"; done
hexmask () { python3 -c "import sys; print(hex(sum(1<<int(x) for x in sys.argv[1:])))" "$@"; }
S0=$(hexmask ${c[0]} ${c[1]}); S1=$(hexmask ${c[2]} ${c[3]})      # same-socket pair (if c0..c3 share a package)
X1=$(hexmask ${c[4]} ${c[5]})                                      # cross pair (if c4,c5 are on the other package)
D0=$(hexmask ${c[0]} ${c[1]} ${c[2]} ${c[3]})
echo "masks S0=$S0 S1=$S1 X1=$X1 D0=$D0"

cat > task.sh <<'EOF'
#!/bin/bash
CG=$(grep cpuset /proc/self/cgroup | cut -d: -f3)
echo "rank $SLURM_PROCID cgroup $CG cpuset.cpus=$(cat /sys/fs/cgroup/cpuset$CG/cpuset.cpus 2>&1) Cpus_allowed_list=$(grep Cpus_allowed_list /proc/self/status | cut -f2)" >> "$ARM.cg"
if [ "$SLURM_PROCID" = 0 ]; then
  exec ./elastic_bench "$NCHUNKS" "$NREG" "$ARM.status" > "$ARM.bench" 2>&1
else
  grep Cpus_allowed_list /proc/self/status | cut -f2 > "$ARM.donor"
  sleep 8
fi
EOF
chmod +x task.sh

NREG=40
run () {  # ARM OMP MASK0 MASK1 NCHUNKS BLOCKTIME BIND(0/1) MODE(none|widen|move) [--all]
  export ARM=$1 OMP_NUM_THREADS=$2 NCHUNKS=$5 NREG KMP_BLOCKTIME=$6
  if [ "$7" = 1 ]; then export OMP_PROC_BIND=true OMP_PLACES=cores; else export OMP_PROC_BIND=false; unset OMP_PLACES; fi
  rm -f "$ARM.status" "$ARM.donor" "$ARM.cg"
  echo "=== arm $ARM OMP=$2 masks $3,$4 nchunks=$5 blocktime=$6 bind=$7 mode=$8 ${9:-} ==="
  # each step takes the whole 8-core allocation so mask_cpu can pick any subset of it
  srun -n 2 --cpus-per-task=4 --cpu-bind=mask_cpu:$3,$4 ./task.sh &
  SP=$!
  if [ "$8" != none ]; then
    timeout 60 python3 "$P/expander.py" "$ARM.status" "$ARM.donor" $NREG 6 "$8" ${9:-} > "$ARM.expander" 2>&1
  fi
  wait $SP
  cat "$ARM.cg"
  awk '/^region/{k=$2; w=$4; if(k>=3){ if(k%2){a[++na]=w}else{b[++nb]=w} } }
       END{asort(a); asort(b); printf "  odd(acted) median %.2f  even(control) median %.2f  (n=%d/%d)\n", a[int((na+1)/2)], b[int((nb+1)/2)], na, nb}' "$ARM.bench"
  [ -f "$ARM.expander" ] && { grep -E 'expander:|errs' "$ARM.expander" | head -3; grep -E 'on_donor' "$ARM.expander" | awk '{print "  "$1,$2,$3,$4,$5,$6,$7,$8}' | head -4; }
  grep -E '^region (3|4|39) ' "$ARM.bench" | cut -c1-140
  grep -E '^thread' "$ARM.bench" | awk '{printf "  %s %s migr %s nonvol %s\n",$1,$2,$10,$8}'
}

# --- short regions (~23 ms, one micro call), same-socket donor ---
run A_S    2 $S0 $S1 8 0 0 none
run B_S    4 $S0 $S1 8 0 0 none
run D_S    4 $D0 $X1 8 0 0 none
run CW_S   4 $S0 $S1 8 0 0 widen
run CM_S   4 $S0 $S1 8 0 0 move
run CMb_S  4 $S0 $S1 8 0 1 move
# --- cross-socket donor ---
run CW_X   4 $S0 $X1 8 0 0 widen
run CM_X   4 $S0 $X1 8 0 0 move
# --- long regions (~92 ms) ---
run BL_S   4 $S0 $S1 32 0 0 none
run CML_S  4 $S0 $S1 32 0 0 move --all
run DL_S   4 $D0 $X1 32 0 0 none
echo "=== done ==="

#!/usr/bin/env bash
#SBATCH --job-name=g1attrib
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:25:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/g1-%j.out
#
# Gate 1 (slack stealing): hot-vs-cold hardware attribution on pristine V (wrf.exe, OMP=2, numtiles=13,
# production cyclic placement). One same-SNC-domain hot/cold pair per node, perf-attached from outside
# during steps 5..35; micro windows reconstructed from sample timestamps. No WRF change, no controller.
# Env: EV (perf event spec, from the hwprobe result), SPEC (rank:role list), STEPS.
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore
S=/home/u5727520/wrf-stack/wtune/rtctl/score
STEPS=${STEPS:-40}
SPEC=${SPEC:-"12:hot,44:cold,9:hot,49:cold,10:hot,42:cold,11:hot,51:cold"}
EV=${EV:?set EV to the perf event spec validated by hwprobe}
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0 OMP_NUM_THREADS=2
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
J=$SLURM_JOB_ID; D=$OUT/g1_V13_$J
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$"); MIN=$(( (STEPS * DT + 59) / 60 ))
echo "job $J nodes $SLURM_JOB_NODELIST spec $SPEC"
echo "EV: $EV"
rm -rf "$D"; mkdir -p "$D"; cd "$D"
ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
for f in "$W"/run/*; do b=$(basename "$f"); case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
cp "$PREP/namelist.input" .
sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=13," namelist.input
ln -sf "$W/main/wrf.exe" ./wrf.exe
T0=$(date +%s.%N)
timeout 300 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 ./wrf.exe > wrf.out 2>&1 &
WP=$!
sleep 3
timeout 900 srun --overlap -N 4 -n 4 --ntasks-per-node=1 -c 1 --mem=4G --cpu-bind=none bash "$S/g1_node.sh" "$D" "$D/g1" "$EV" "$SPEC" > g1_collect.out 2>&1 &
CP=$!
wait $WP; RC=$?; T1=$(date +%s.%N)
wait $CP; echo "collector rc=$?"
grep "Timing for main" rsl.out.0000 | awk '{print $9}' > steps.txt
python3 - "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
print("RESULT tag=V13_g1 rc=%s steps=%d med=%.5f srun_wall=%.1f ok=%d" % (rc, len(v), st.median(v) if v else -1, float(t1) - float(t0),
      int(any("SUCCESS COMPLETE WRF" in l for l in open("rsl.error.0000")))))
PY
cat g1/meta.*.txt
echo "host,rank,role,class,cycles,instructions,ref_cycles,llc_misses,ipc,eff_ghz,mpki,n_windows,median_window_ms,median_micro_instr_per_window,window_util" > g1/summary.csv
cat g1/summary.*.csv >> g1/summary.csv
echo "--- summary (micro and all) ---"
awk -F, 'NR==1 || $4=="micro" || $4=="all"' g1/summary.csv | column -s, -t
tail -3 g1_collect.out
rm -f wrfout_d01_*
echo "=== done"

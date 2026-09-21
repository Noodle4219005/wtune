#!/usr/bin/env bash
#SBATCH --job-name=ntabab
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:45:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/numtiles/abab-%j.out
#
# Competition-safe branch: pristine wrf.exe, OMP=2, cyclic, 40 steps; numtiles 13 / 12 / 14 in a
# balanced order (each 3x, positions rotated) so drift cancels. Every arm is diffwrf'd against the
# first V13 final frame. No PLOV, no wrapper, no controller.
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore/numtiles
STEPS=${STEPS:-40}
ORDER=${ORDER:-"13 12 14 14 12 13 13 12 14"}
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0 OMP_NUM_THREADS=2
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
J=$SLURM_JOB_ID
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$"); MIN=$(( (STEPS * DT + 59) / 60 ))
echo "job $J nodes $SLURM_JOB_NODELIST order [$ORDER] wrf.exe sha256 $(sha256sum $W/main/wrf.exe | cut -c1-16) wraps=$(nm $W/main/wrf.exe | grep -c __wrap_)"
REF=; i=0
for NT in $ORDER; do
  i=$((i+1)); TAG=V${NT}_r$i; D=$OUT/${TAG}_$J
  rm -rf "$D"; mkdir -p "$D"; cd "$D"
  ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
  for f in "$W"/run/*; do b=$(basename "$f"); case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$PREP/namelist.input" .
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=$NT," namelist.input
  ln -sf "$W/main/wrf.exe" ./wrf.exe
  T0=$(date +%s.%N)
  timeout 300 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 ./wrf.exe > wrf.out 2>&1
  RC=$?; T1=$(date +%s.%N)
  grep "Timing for main" rsl.out.0000 | awk '{print $9}' > steps.txt
  python3 - "$TAG" "$NT" "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
tag, nt, rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
ok = int(any("SUCCESS COMPLETE WRF" in l for l in open("rsl.error.0000")))
print("RESULT tag=%s numtiles=%s rc=%s ok=%d steps=%d med=%.5f med_skip1=%.5f mean_skip1=%.5f srun_wall=%.1f"
      % (tag, nt, rc, ok, len(v), st.median(v) if v else -1, st.median(v[1:]) if len(v) > 1 else -1,
         st.mean(v[1:]) if len(v) > 1 else -1, float(t1) - float(t0)))
PY
  rm -f wrfout_d01_*_00:00:00
  F=$(ls wrfout_d01_* 2>/dev/null | sort | tail -1)
  if [ -z "$REF" ] && [ "$NT" = 13 ] && [ -n "$F" ]; then REF=$D/$F; echo "  REF=$REF"
  elif [ -n "$REF" ] && [ -n "$F" ]; then
    "$W/external/io_netcdf/diffwrf" "$REF" "$F" > diffwrf_vs_ref.txt 2>&1
    echo "DIFF $TAG vs first V13 differing_fields=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' diffwrf_vs_ref.txt)"
    rm -f "$F"
  fi
  cd "$OUT"
done
[ -n "$REF" ] && rm -f "$REF"
echo "=== summary (median of per-arm medians) ==="
grep '^RESULT' "$OUT/abab-$J.out" 2>/dev/null | awk '{for(i=1;i<=NF;i++){split($i,a,"="); v[a[1]]=a[2]} n[v["numtiles"]]++; s[v["numtiles"]]=s[v["numtiles"]]" "v["med"]} END{for(k in n) print "numtiles " k ": " s[k]}'
echo "=== done"

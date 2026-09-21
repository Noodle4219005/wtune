#!/usr/bin/env bash
#SBATCH --job-name=wscore2
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:50:00
#SBATCH --output=/home/u5727520/wrf-bench/wscore/s2-%j.out
#
# Score stage 2: pristine wrf.exe + external elastic-core controller (elasticd, one small overlapping
# task per node, started after the WRF step exists) against V and P on the same allocation.
# No PLOV in any E arm. home1 sits at its quota: every arm deletes its t=0 frame, diffs its final
# frame against V13a immediately, then deletes it.
# Env overrides: STEPS, ARMS, PLAN (prefix of make_plan.py outputs), NT_E, DPH, RATIO, HOTABS.
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
PREP=/home/u5727520/wrf-bench/stprep_1070876
OUT=/home/u5727520/wrf-bench/wscore
BIN=/home/u5727520/wrf-stack/wtune/rtctl/score/bin/elasticd
PROF=$OUT/profile_A_micro_1070173.csv
STEPS=${STEPS:-40}; NT_E=${NT_E:-12}; DPH=${DPH:-1}; RATIO=${RATIO:-1.5}
HOTABS=${HOTABS:-19.2}
ARMS=${ARMS:-"V13a EA0 EA1 EA1m EA2 P13 V13b"}
PLAN=${PLAN:-$OUT/plan_dph2}
SAMPLE=${SAMPLE:-0}
TSBIN=/home/u5727520/wrf-stack/wtune/rtctl/score/bin/tsample
SRANKS=
if [ "$SAMPLE" = 1 ]; then
  # per node (cyclic: node = rank % 4): the lowest hot rank of the plan, its donors, one neutral rank >= 200
  SRANKS=$(python3 - "$PLAN.plan" <<'PY'
import sys
pairs = [tuple(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()]
inplan = {r for p in pairs for r in p}
out = []
for node in range(4):
    hots = sorted({h for h, _ in pairs if h % 4 == node})
    if hots:
        out.append(hots[0]); out += [d for h, d in pairs if h == hots[0]]
    out.append(next(r for r in range(200, 224) if r % 4 == node and r not in inplan))
print(",".join(map(str, out)))
PY
)
  echo "sample ranks: $SRANKS"
fi
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
J=$SLURM_JOB_ID
DT=$(grep -oE "time_step *= *[0-9]+" "$PREP/namelist.input" | head -1 | grep -oE "[0-9]+$")
MIN=$(( (STEPS * DT + 59) / 60 ))
REF=
echo "job $J nodes $SLURM_JOB_NODELIST steps $STEPS arms [$ARMS] plan $PLAN"
echo "plan: $(wc -l < $PLAN.plan) slots sha256 $(sha256sum $PLAN.plan | cut -c1-16); multiprog: $(cut -c1-120 $PLAN.multiprog | tr '\n' '|')"
echo "wrf.exe sha256 $(sha256sum $W/main/wrf.exe | cut -c1-16) wrap=$(nm $W/main/wrf.exe | grep -cE '__wrap_')  elasticd sha256 $(sha256sum $BIN | cut -c1-16)  profile sha256 $(sha256sum $PROF | cut -c1-16)"
echo "home1: $(mmlsquota -u $USER home1 2>/dev/null | tail -1)"

arm () {  # TAG EXE OMP NUMTILES PLOV(0|1) CTLMODE(none|static|dynamic) [extra elasticd args]
  local TAG=$1 EXE=$2 OMPN=$3 NT=$4 PLOV=$5 CTL=$6; shift 6
  local EXTRA="$*" D=$OUT/s2_${TAG}_$J
  rm -rf "$D"; mkdir -p "$D"; cd "$D" || return 1
  ln -sf "$PREP/wrfinput_d01" .; ln -sf "$PREP/wrfbdy_d01" .
  for f in "$W"/run/*; do b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$PREP/namelist.input" . || { echo "RESULT tag=$TAG FATAL cannot write run dir"; cd "$OUT"; return 1; }
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/; s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=$NT," namelist.input
  ln -sf "$EXE" ./wrf.exe
  unset $(env | grep -oE '^WTIMER_[A-Z_0-9]+' | tr '\n' ' ') 2>/dev/null
  if [ "$PLOV" = 1 ]; then set -a; . /home/u5727520/wrf-stack/wtune/plov/production.env; set +a; export WTIMER_DIR="$D" WTIMER_DEBUG_RANK=0; fi
  export OMP_NUM_THREADS=$OMPN
  local LAUNCH="./wrf.exe"
  [ -n "${MPCONF:-}" ] && { cp "$MPCONF" mp.conf; LAUNCH="--multi-prog mp.conf"; }
  local T0 T1 RC WP CP=
  T0=$(date +%s.%N)
  # a clean 40-step arm takes ~105 s; 300 s bounds a stalled history write (job 1073932 V13a hung 900 s)
  timeout 300 srun --mpi=pmi2 --distribution=cyclic -n 224 --ntasks-per-node=56 -c 2 $LAUNCH > wrf.out 2>&1 &
  WP=$!
  if [ "$CTL" != none ]; then
    # after WRF's step exists; -n 4 is required, otherwise srun inherits SLURM_NTASKS=448
    sleep 3
    srun --overlap -N 4 -n 4 --ntasks-per-node=1 -c 1 --mem=1G --cpu-bind=none bash -c \
      "exec $BIN --exe wrf.exe --omp $OMPN --rpn 56 --profile $PROF --col A_micro_total_s --rsl $D/rsl.out.0000 --start-step 2 --donors-per-hot $DPH --hot-ratio $RATIO --hot-abs $HOTABS --mode $CTL --max-s 1200 $EXTRA > $D/ctl.\$SLURMD_NODENAME.out 2>&1" &
    CP=$!
  fi
  local SPID=
  if [ "$SAMPLE" = 1 ]; then
    sleep 1
    srun --overlap -N 4 -n 4 --ntasks-per-node=1 -c 1 --mem=1G --cpu-bind=none bash -c \
      "exec $TSBIN --exe wrf.exe --ranks $SRANKS --rsl $D/rsl.out.0000 --out $D/tsample.\$SLURMD_NODENAME.csv --interval-us 5000 --max-s 900 > $D/tsample.\$SLURMD_NODENAME.log 2>&1" &
    SPID=$!
  fi
  wait $WP; RC=$?; T1=$(date +%s.%N)
  if [ -n "$CP" ]; then sleep 2; kill -TERM $CP 2>/dev/null; wait $CP 2>/dev/null; fi
  if [ -n "$SPID" ]; then kill -TERM $SPID 2>/dev/null; wait $SPID 2>/dev/null; grep -h TSAMPLE tsample.*.log 2>/dev/null; fi
  local OK
  OK=$(grep -c "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null)
  grep "Timing for main" rsl.out.0000 2>/dev/null | awk '{print $9}' > steps.txt
  python3 - "$TAG" "$OMPN" "$NT" "$PLOV" "$CTL" "$OK" "$RC" "$T0" "$T1" <<'PY'
import sys, statistics as st
tag, omp, nt, plov, ctl, ok, rc, t0, t1 = sys.argv[1:]
v = [float(x) for x in open("steps.txt") if x.strip()]
w = v[1:] if len(v) > 1 else v
print("RESULT tag=%s omp=%s numtiles=%s plov=%s ctl=%s ok=%s rc=%s steps=%d med=%.5f med_skip1=%.5f mean_skip1=%.5f sum=%.3f srun_wall=%.1f"
      % (tag, omp, nt, plov, ctl, ok, rc, len(v), st.median(v) if v else -1, st.median(w) if w else -1,
         st.mean(w) if w else -1, sum(v), float(t1) - float(t0)))
PY
  if [ "$CTL" != none ]; then
    grep -h '^TOTAL' ctl.*.out 2>/dev/null
    grep -h 'FATAL\|restore\|short_teams\|ELASTICD plan\|pin_donor_master' ctl.*.out 2>/dev/null | sort | uniq -c | head -12
    python3 - <<'PY'
import glob, re
hot, don = [], []
for f in glob.glob("ctl.*.out"):
    for l in open(f):
        m = re.match(r"RANKCPU host=\S+ rank=(\d+) role=(\d) score=([\d.]+) nteam=\d+ home=\S+ master_s=([\d.]+) workers_s=([\d.]+)", l)
        if m and m[2] == "1": hot.append(float(m[5]))
        if m and m[2] == "2": don.append(float(m[5]))
if hot: print("  RANKCPU hot n=%d mean workers_s=%.2f | donors n=%d mean workers_s=%.2f" % (len(hot), sum(hot)/len(hot), len(don), sum(don)/max(1, len(don))))
PY
  fi
  grep -iE "segmentation|signal|forrtl|backtrace|quota" wrf.out rsl.error.0000 2>/dev/null | head -3
  rm -f wrfout_d01_*_00:00:00
  local F
  F=$(ls wrfout_d01_* 2>/dev/null | sort | tail -1)
  if [ "$TAG" = V13a ]; then
    [ -n "$F" ] && REF=$D/$F
    echo "  REF=${REF:-none}"
  elif [ -n "$REF" ] && [ -n "$F" ] && [ "$(basename "$REF")" = "$F" ]; then
    "$W/external/io_netcdf/diffwrf" "$REF" "$F" > diffwrf_vs_V13a.txt 2>&1
    echo "DIFF $TAG vs V13a differing_fields=$(awk '/Field +Ndifs/{h=1;next} h && NF>=3 && $2 ~ /^[0-9]+$/ {n++} END{print n+0}' diffwrf_vs_V13a.txt)"
    rm -f "$F"
  else
    echo "DIFF $TAG vs V13a SKIP ref=${REF:-none} f=${F:-none}"
  fi
  cd "$OUT"
}

for A in $ARMS; do
  echo "=== $A ==="
  case $A in
    V13a|V13b) arm $A $W/main/wrf.exe 2 13 0 none ;;
    E0)        arm $A $W/main/wrf.exe 4 $NT_E 0 none ;;
    P13)       arm $A $W/main/wrf_wtimer.exe 2 13 1 none ;;
    EA0)       MPCONF=$PLAN.multiprog arm $A $W/main/wrf.exe 2 13 0 none ;;
    EA1)       MPCONF=$PLAN.multiprog arm $A $W/main/wrf.exe 2 13 0 static --omp 4 --plan $PLAN.plan ;;
    EA1m)      MPCONF=$PLAN.multiprog arm $A $W/main/wrf.exe 2 13 0 static --omp 4 --plan $PLAN.plan --pin-donor-master ;;
    EA2)       MPCONF=$PLAN.multiprog arm $A $W/main/wrf.exe 2 13 0 dynamic --omp 4 --plan $PLAN.plan --pin-donor-master ;;
  esac
done
[ -n "$REF" ] && rm -f "$REF"
echo "home1: $(mmlsquota -u $USER home1 2>/dev/null | tail -1)"
echo "=== done ==="

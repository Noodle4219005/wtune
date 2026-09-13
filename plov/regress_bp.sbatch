#!/usr/bin/env bash
#SBATCH --job-name=plov-regress
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=56 --cpus-per-task=2
#SBATCH --time=00:50:00
#SBATCH --output=/home/u5727520/wrf-bench/regress/plov-regress-%j.out
# Standing regression gate for every libwt change, all-ADIOS2 (BP5 inputs + BP5 history, no netCDF I/O in WRF).
# karina_48h d01 (mp=8 Thompson, RRTMG radt=15, YSU, cu=0), 400 steps, 224 ranks x 2 threads, 8x28 decomposition.
#   A     : nothing staged, --distribution=block                  (reference)
#   Ec    : production staging (rows 4, adapt75), cyclic          (the proven production shape)
#   Ec224 : same as Ec + adios2_numaggregators=224                (BP5+cyclic readability probe, job 1071708/1071747)
# PASS = every arm SUCCESS COMPLETE, micro_refused=0 on Ec/Ec224, every BP wrfout converts, diffwrf(A,Ec) and
# diffwrf(A,Ec224) list zero differing fields. Inputs: $BPIN/wrfinput_d01.bp + wrfbdy_d01.bp (tools/wrf_nc2bp.py from
# /home/u5727520/wrf-bench/stprep_1070876, job 1072361). Binary provenance: $EXE.build (written by build.sbatch).
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
T=/home/u5727520/wrf-stack/wtune/plov
BPIN=${BPIN:-/home/u5727520/wrf-bench/karina_48h/bp}
NL=${NL:-/home/u5727520/wrf-bench/stprep_1070876/namelist.input}
ROOT=/home/u5727520/wrf-bench/regress
STEPS=${STEPS:-400}
EXE=${EXE:-$W/main/wrf_wtimer.exe}
ENV=/home/u5727520/wrf-stack/wtune-pixi-env-20260830
CONV=/home/u5727520/SC26-WRF-p0/scripts/vendor/wrf-adios2-to-netcdf4/convert.py
ARMS=${ARMS:-"A Ec Ec224"}
for s in "$T"/wtimer.c "$T"/wt_micro.c "$T"/wt_rad.c "$T"/wt_alloc.c "$T"/wtimer.S "$T"/wt_call.S "$T"/wt_argmap.h; do
  if [ "$s" -nt "$EXE" ]; then echo "STALE BINARY: $s is newer than $EXE"; ls -l "$s" "$EXE"; exit 3; fi
done
echo "binary $EXE $(stat -c %y "$EXE")"; [ -f "$EXE.build" ] && head -3 "$EXE.build"
[ -d "$BPIN/wrfinput_d01.bp" ] && [ -d "$BPIN/wrfbdy_d01.bp" ] || { echo "FATAL: BP inputs missing under $BPIN"; exit 4; }
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$ROOT"

arm () {  # $1=tag $2=stage rows (0 = off) $3=none|adapt75 $4=block|cyclic $5=adios2_numaggregators (empty = WRF default 0)
  local TAG=$1 ROWS=$2 D=$ROOT/rg_${1}_$SLURM_JOB_ID
  rm -rf "$D"; mkdir -p "$D"; cd "$D" || return 1
  ln -sf "$BPIN/wrfinput_d01.bp" wrfinput_d01.bp; ln -sf "$BPIN/wrfbdy_d01.bp" wrfbdy_d01.bp
  for f in "$W"/run/*; do b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$NL" namelist.input
  local DT MIN; DT=$(grep -oE "time_step=[0-9]+" namelist.input | grep -oE "[0-9]+$"); MIN=$(( STEPS * DT / 60 ))
  [ $(( MIN * 60 )) -eq $(( STEPS * DT )) ] || { echo "FATAL: STEPS*DT not whole minutes"; return 2; }
  sed -i "s/\brun_minutes=[0-9]*/run_minutes=$MIN/; s/ history_interval=[0-9]*/ history_interval=$MIN/" namelist.input
  sed -i "s/io_form_input=[0-9]*/io_form_input=14/; s/io_form_boundary=[0-9]*/io_form_boundary=14/; s/io_form_history=[0-9]*/io_form_history=14/" namelist.input
  sed -i "s/^&time_control$/\&time_control\n input_inname=\"wrfinput_d<domain>.bp\", bdy_inname=\"wrfbdy_d<domain>.bp\",/" namelist.input
  if [ -n "${5:-}" ]; then printf '&namelist_adios2\n adios2_numaggregators=%s,\n/\n' "$5" >> namelist.input; fi
  grep -c inname namelist.input | grep -q 1 || { echo "FATAL: inname not inserted"; return 2; }
  ln -sf "$EXE" ./run.exe
  export OMP_NUM_THREADS=2 WTIMER_DIR="$D" WTIMER_DEBUG_RANK=0
  unset WTIMER_RAD_ROWS WTIMER_RAD_BUF WTIMER_RAD_ISPLIT WTIMER_RAD_FULLJ WTIMER_RAD_SHARE WTIMER_LEWI WTIMER_STAGE_ROWS WTIMER_STAGE_DEBUG WTIMER_STAGE_SHARE WTIMER_STAGE_ADAPT WTIMER_STAGE_CHUNKS WTIMER_STAGE_MAX_PCT WTIMER_STAGE_GRID
  [ "$ROWS" != 0 ] && export WTIMER_STAGE_ROWS=$ROWS WTIMER_STAGE_DEBUG=1
  [ "${3:-}" = adapt75 ] && export WTIMER_STAGE_SHARE=1 WTIMER_STAGE_ADAPT=1 WTIMER_STAGE_CHUNKS=6 WTIMER_STAGE_MAX_PCT=75
  local s=$(date +%s)
  timeout 900 srun --mpi=pmi2 --distribution=${4:-block} -n 224 --ntasks-per-node=56 -c 2 ./run.exe > wrf.out 2>&1
  echo "  srun rc=$? wall=$(( $(date +%s) - s )) s distribution=${4:-block} numaggregators=${5:-default}"
  local OK N MED
  OK=$(grep -c "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null || echo 0)
  N=$(grep -c "Timing for main" rsl.out.0000 2>/dev/null || echo 0)
  MED=$(grep "Timing for main" rsl.out.0000 | awk '{print $9}' | sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.5f",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  echo "  $TAG rows=$ROWS ok=$OK steps=$N med=${MED:-NA} s/step"
  echo "  non-rad-step median: $(grep "Timing for main" rsl.out.0000 | awk '(NR-1)%60!=0{print $9}' | sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.5f (n=%d)",v[int((NR+1)/2)],NR}')"
  grep -h "^# micro_calls" wtimer.*.csv 2>/dev/null | sed 's/[a-z_]*=//g; s/#//' | awk '{c[NR]=$1;s[NR]=$2;r[NR]=$3} END{if(!NR){print "  micro footer: ranks=0"; exit} cm=c[1];cM=c[1];sm=s[1];sM=s[1];rm=r[1];rM=r[1];for(k=2;k<=NR;k++){if(c[k]<cm)cm=c[k];if(c[k]>cM)cM=c[k];if(s[k]<sm)sm=s[k];if(s[k]>sM)sM=s[k];if(r[k]<rm)rm=r[k];if(r[k]>rM)rM=r[k]} printf "  micro footer: ranks=%d calls %d..%d staged %d..%d refused %d..%d\n",NR,cm,cM,sm,sM,rm,rM}'
  grep -h "^# stage_\|^# micro_sig\|malformed\|refused" wtimer.0.csv 2>/dev/null | head -8
  grep -h "selfcheck\|stage geometry\|\[wtimer\] share:\|\[wtimer\] adapt" rsl.error.0000 2>/dev/null | head -12
  grep -iE "segmentation|signal|forrtl|backtrace|adios2 error" wrf.out rsl.error.0000 2>/dev/null | grep -v 'Varname' | head -5
  grep -h "^# rad_calls" wtimer.0.csv 2>/dev/null | head -1
  grep -h "^# stage_flags\|^# stage" wtimer.0.csv 2>/dev/null | head -4
  echo "  outputs: $(ls -d wrfout_d01_* 2>/dev/null | tr '\n' ' ')"
}
for a in $ARMS; do
  case $a in
    A)     echo "=== A: nothing staged, block ==="; arm A 0 none block ;;
    Ec)    echo "=== Ec: micro adapt75, cyclic ==="; arm Ec 4 adapt75 cyclic ;;
    Ec224) echo "=== Ec224: micro adapt75, cyclic, adios2_numaggregators=224 ==="; arm Ec224 4 adapt75 cyclic 224 ;;
  esac
done

# --- convert each arm's last BP wrfout to netCDF on one compute node, then diffwrf against A ---
conv () {  # $1=tag -> prints path of .nc or empty
  local D=$ROOT/rg_${1}_$SLURM_JOB_ID BP
  BP=$(ls -d "$D"/wrfout_d01_* 2>/dev/null | grep -v '\.nc$' | sort | tail -1); [ -n "$BP" ] || return 1
  local s=$(date +%s)
  srun -N1 -n1 -c 8 bash -c "cd $ENV && pixi run python $CONV --input $BP --output $BP.nc" > "$D/convert.log" 2>&1
  local rc=$?
  echo "  convert $1: rc=$rc in $(( $(date +%s) - s )) s $(tail -1 "$D/convert.log" | cut -c1-160)" >&2
  [ $rc -eq 0 ] || rm -f "$BP.nc"   # convert.py leaves a partial .nc behind on failure (1072441 Ec: 132 KB)
  [ $rc -eq 0 ] && [ -f "$BP.nc" ] && echo "$BP.nc"
}
echo; echo "=== BIT-IDENTITY (convert BP -> nc, diffwrf vs A) ==="
FA=$(conv A)
for Y in $ARMS; do
  [ "$Y" = A ] && continue
  FY=$(conv $Y)
  if [ -n "$FA" ] && [ -n "$FY" ]; then
    DF=$ROOT/rg_${Y}_$SLURM_JOB_ID/diffwrf_A_${Y}.txt
    "$W/external/io_netcdf/diffwrf" "$FA" "$FY" > "$DF" 2>&1
    echo "  A vs $Y: differing-field lines: $(grep -c '^ *[A-Z0-9_]\+ \+[0-9]' "$DF")  (full: $DF)"; head -12 "$DF"
  else echo "  A vs $Y: SKIP (A=${FA:-none} $Y=${FY:-none})"; fi
done
echo; echo "=== VERDICT ==="
for a in $ARMS; do
  D=$ROOT/rg_${a}_$SLURM_JOB_ID
  printf "  %-6s ok=%s refused_max=%s converted=%s\n" "$a" "$(grep -c 'SUCCESS COMPLETE WRF' $D/rsl.error.0000 2>/dev/null)" \
    "$(grep -h '^# micro_calls' $D/wtimer.*.csv 2>/dev/null | sed 's/.*micro_refused=\([0-9]*\).*/\1/' | sort -n | tail -1)" \
    "$(ls $D/wrfout_d01_*.nc 2>/dev/null | wc -l)"
done

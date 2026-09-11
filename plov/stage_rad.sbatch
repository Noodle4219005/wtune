#!/usr/bin/env bash
#SBATCH --job-name=wtrad
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:25:00
#SBATCH --output=/home/u5727520/wrf-stack/wtimer/rad-%j.out
#
# PLOV stage 1 gate: in-place dense-rectangle staging of the last R j-rows of every rank's own
# patch through the corrected 402-argument map (gen_argmap.py). No migration yet. The gate is
# bit-identity against the unstaged arm plus micro_staged == micro_calls on every rank; the
# self-check line from rank 0 must say STAGING ON.
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
T=/home/u5727520/wrf-stack/wtimer
SRC=/home/u5727520/wrf-bench/karina_48h
STEPS=130   # rad steps at 1, 61, 121 (stepra=60)
# Fail closed on a stale binary: 1070522/1070551 ran R1 code because build.sbatch
# had relinked a different tree. The exe must be newer than every wtimer source.
EXE=$W/main/wrf_wtimer.exe
for s in "$T"/wtimer.c "$T"/wt_micro.c "$T"/wt_rad.c; do
  if [ "$s" -nt "$EXE" ]; then echo "STALE BINARY: $s is newer than $EXE"; ls -l "$s" "$EXE"; exit 3; fi
done
echo "binary $EXE $(stat -c %y "$EXE") rad_share symbols: $(nm "$EXE" 2>/dev/null | grep -c wt_rad_share)"
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

P=/home/u5727520/wrf-bench/stprep_$SLURM_JOB_ID
rm -rf "$P"; mkdir -p "$P"; cd "$P"
for f in "$SRC"/met_em.d0*.nc "$SRC"/geo_em.d0*.nc; do [ -e "$f" ] && ln -sf "$(readlink -f "$f")" .; done
cp "$SRC/namelist.input" .
for f in "$W"/run/*; do b=$(basename "$f")
  case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
ln -sf "$W"/main/real.exe real.exe
DT=$(grep -oE "time_step *= *[0-9]+" namelist.input | head -1 | grep -oE "[0-9]+$")
MIN=$(( (STEPS * DT + 59) / 60 ))
sed -i "s/\brun_days=[0-9]*/run_days=0/;s/\brun_hours=[0-9]*/run_hours=0/;s/\brun_minutes=[0-9]*/run_minutes=$MIN/" namelist.input
sed -i "s/ history_interval=[0-9]*/ history_interval=$MIN/;s/ restart_interval=[0-9]*/ restart_interval=999999/" namelist.input
sed -i -e "s/ io_form_history=[0-9]*/ io_form_history=14/" -e "s/ io_form_restart=[0-9]*/ io_form_restart=14/" namelist.input
srun --mpi=pmi2 -n 448 --ntasks-per-node=112 ./real.exe > real.out 2>&1
SZI=$(stat -c %s wrfinput_d01 2>/dev/null || echo 0); SZB=$(stat -c %s wrfbdy_d01 2>/dev/null || echo 0)
[ "$SZI" -ge 500000000 ] && [ "$SZB" -ge 300000000 ] || { echo "FATAL prep: $SZI $SZB"; exit 4; }
sed -i "/^ *nproc_x=/d; /^ *nproc_y=/d; /^ *numtiles=/d; /&domains/a\\ nproc_x=8, nproc_y=28, numtiles=13," namelist.input
echo "prep ok"

HF=$P/hostfile.lpt
mk_hostfile () {  # $1=profile dir: LPT over per-rank unstaged micro totals, 56 ranks per node -> $HF
  python3 - "$1" "$HF" $(scontrol show hostnames "$SLURM_JOB_NODELIST") <<'EOP'
import sys,glob,re
d,hf,*nodes=sys.argv[1:]
load={}
for f in glob.glob(d+"/wtimer.*.csv"):
    r=int(re.search(r"wtimer\.(\d+)\.csv",f).group(1))
    load[r]=sum(float(l.split(',')[1]) for l in open(f) if l[:1].isdigit())
n=len(load); per=n//len(nodes); sums=[0.0]*len(nodes); cnt=[0]*len(nodes); host=[None]*n
for r in sorted(load,key=lambda r:-load[r]):
    k=min((k for k in range(len(nodes)) if cnt[k]<per),key=lambda k:sums[k])
    host[r]=nodes[k]; sums[k]+=load[r]; cnt[k]+=1
open(hf,"w").write("".join(h+"\n" for h in host))
print("  LPT hostfile: %d ranks, node sums %s, ranks/node %s"%(n," ".join("%.1f"%x for x in sums),cnt))
EOP
}

mk_hostfile_xpair () {  # x-pair placement: rank r -> node ((r mod 8) div 2); every N-S edge on-node, E-W off-node only at 3 of 7 column boundaries
  python3 - "$HF" $(scontrol show hostnames "$SLURM_JOB_NODELIST") <<'EOP'
import sys
hf,*nodes=sys.argv[1:]
host=[nodes[(r%8)//2] for r in range(224)]
open(hf,"w").write("".join(h+"\n" for h in host))
print("  xpair hostfile: 224 ranks, ranks/node %s"%[host.count(n) for n in nodes])
EOP
}

arm () {  # $1=tag $2=stage rows (0 = off) $3=share|adapt|adapt75 $4=srun --distribution (default block; arbitrary reads $HF)
  local TAG=$1 ROWS=$2 D=/home/u5727520/wrf-bench/st_${1}_$SLURM_JOB_ID
  rm -rf "$D"; mkdir -p "$D"; cd "$D" || return 1
  unset SLURM_HOSTFILE
  if [ "${4:-}" = arbitrary ]; then
    export SLURM_HOSTFILE=$HF
    srun --mpi=pmi2 --distribution=arbitrary -n 224 --ntasks-per-node=56 -c 2 bash -c 'echo $SLURM_PROCID $SLURMD_NODENAME' | sort -n | awk '{print $2}' > probe.txt
    if diff -q probe.txt "$HF" >/dev/null; then echo "  placement matches hostfile"; else echo "  FATAL placement != hostfile ($(diff probe.txt "$HF" | grep -c '^[<>]') lines)"; return 5; fi
  fi
  for f in "$P"/met_em.d0*.nc "$P"/geo_em.d0*.nc "$P"/wrfinput_d01 "$P"/wrfbdy_d01; do
    [ -e "$f" ] && ln -sf "$f" "$(basename "$f")"; done
  for f in "$W"/run/*; do b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  cp "$P/namelist.input" .; sed -i -e "s/ io_form_history=[0-9]*/ io_form_history=2/" namelist.input
  ln -sf "$W"/main/wrf_wtimer.exe ./run.exe
  export OMP_NUM_THREADS=2 WTIMER_DIR="$D" WTIMER_DEBUG_RANK=0
  unset WTIMER_RAD_ROWS WTIMER_RAD_BUF WTIMER_RAD_ISPLIT WTIMER_RAD_FULLJ WTIMER_RAD_SHARE WTIMER_RAD_SLOT_BYTES WTIMER_LEWI WTIMER_STAGE_ROWS WTIMER_STAGE_DEBUG WTIMER_STAGE_SHARE WTIMER_STAGE_ADAPT WTIMER_STAGE_CHUNKS WTIMER_STAGE_MAX_PCT
  [ "$ROWS" != 0 ] && export WTIMER_STAGE_ROWS=$ROWS WTIMER_STAGE_DEBUG=1
  [ "${3:-}" = share ] && export WTIMER_STAGE_SHARE=1
  [ "${3:-}" = adapt ] && export WTIMER_STAGE_SHARE=1 WTIMER_STAGE_ADAPT=1 WTIMER_STAGE_CHUNKS=4
  [ "${3:-}" = adapt75 ] && export WTIMER_STAGE_SHARE=1 WTIMER_STAGE_ADAPT=1 WTIMER_STAGE_CHUNKS=6 WTIMER_STAGE_MAX_PCT=75
  [ "${5:-}" = rad4 ] && export WTIMER_RAD_ROWS=4 WTIMER_STAGE_DEBUG=1   # $5=rad4 stages the radiation driver too
  [ "${5:-}" = rad8 ] && export WTIMER_RAD_ROWS=8 WTIMER_STAGE_DEBUG=1 WTIMER_RAD_SLOT_BYTES=16777216   # 8-row tail: 8 x 0.93 MB + 141 KB header
  [ "${5:-}" = rad31 ] && export WTIMER_RAD_ROWS=31 WTIMER_STAGE_DEBUG=1 WTIMER_RAD_SLOT_BYTES=33554432   # R2b: near-whole tile (32/33-row tiles), 32 MiB shm slot
  [ "${3:-}" = adapt75r8 ] && export WTIMER_STAGE_SHARE=1 WTIMER_STAGE_ADAPT=1 WTIMER_STAGE_CHUNKS=3 WTIMER_STAGE_MAX_PCT=75
  # $6 = pass-2 geometry variant (G0 #4, see wt_rad.c): rn/rs run on the original arrays, rh/rf use the buffer
  case "${6:-}" in
    rn) export WTIMER_RAD_BUF=0 WTIMER_RAD_ISPLIT=-1 ;;
    rs) export WTIMER_RAD_BUF=0 WTIMER_RAD_ISPLIT=32 ;;
    rh) export WTIMER_RAD_BUF=1 WTIMER_RAD_ISPLIT=32 WTIMER_RAD_FULLJ=0 ;;
    rf) export WTIMER_RAD_BUF=1 WTIMER_RAD_ISPLIT=32 WTIMER_RAD_FULLJ=1 ;;
    rhs) export WTIMER_RAD_BUF=1 WTIMER_RAD_ISPLIT=32 WTIMER_RAD_FULLJ=0 WTIMER_RAD_SHARE=1 ;;  # R2: rh + node-local row stealing
  esac
  timeout 400 srun --mpi=pmi2 --distribution=${4:-block} -n 224 --ntasks-per-node=56 -c 2 ./run.exe > wrf.out 2>&1
  echo "  distribution=${4:-block} rank:node for ranks 0-9: $(srun --distribution=${4:-block} -n 224 --ntasks-per-node=56 -c 2 bash -c 'echo $SLURM_PROCID:$SLURM_NODEID' | sort -t: -k1,1n | head -10 | tr "\n" " ")"
  echo "  srun rc=$?"
  local OK N MED
  OK=$(grep -c "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null || echo 0)
  N=$(grep -c "Timing for main" rsl.out.0000 2>/dev/null || echo 0)
  MED=$(grep "Timing for main" rsl.out.0000 | awk '{print $9}' | sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.5f",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  echo "  $TAG rows=$ROWS ok=$OK steps=$N med=${MED:-NA} s/step"
  echo "  rad readback over ranks: calls steps staged refused (min..max) | share_out share_in (min..max, sum) | wait_us sum/max"
  grep -h "^# rad_calls" wtimer.*.csv 2>/dev/null | sed 's/[a-z_]*=//g; s/#//' | awk '{for(k=1;k<=6;k++){if(NR==1||$k<m[k])m[k]=$k;if(NR==1||$k>M[k])M[k]=$k} so+=$5;si+=$6;sw+=$7;if($7>mw)mw=$7} END{if(!NR){print "  ranks=0";exit} printf "  ranks=%d calls %d..%d steps %d..%d staged %d..%d refused %d..%d | share_out %d..%d sum=%d share_in %d..%d sum=%d | wait_us sum=%d max=%d\n",NR,m[1],M[1],m[2],M[2],m[3],M[3],m[4],M[4],m[5],M[5],so,m[6],M[6],si,sw,mw}'
  grep -h "selfcheck\|stage geometry\|rad share\|\[wtimer\] share:\|\[wtimer\] adapt\|\[wtimer\] mark\|\[wtimer\] [23]d " rsl.error.0000 2>/dev/null | head -60
  echo "  micro readback over ranks (calls staged refused share_out share_in): min/max and sums"
  grep -h "^# micro_calls" wtimer.*.csv 2>/dev/null | sed 's/[a-z_]*=//g; s/#//' | awk '{c[NR]=$1;s[NR]=$2;r[NR]=$3;o[NR]=$4;i[NR]=$5;so+=o[NR];si+=i[NR];sw+=$6;sk+=$7;ro+=$8;ci+=$9;if($6>mw)mw=$6;if($7>mk)mk=$7} END{if(!NR){print "  ranks=0"; exit} cm=c[1];cM=c[1];sm=s[1];sM=s[1];rm=r[1];rM=r[1];om=o[1];oM=o[1];im=i[1];iM=i[1];for(k=2;k<=NR;k++){if(c[k]<cm)cm=c[k];if(c[k]>cM)cM=c[k];if(s[k]<sm)sm=s[k];if(s[k]>sM)sM=s[k];if(r[k]<rm)rm=r[k];if(r[k]>rM)rM=r[k];if(o[k]<om)om=o[k];if(o[k]>oM)oM=o[k];if(i[k]<im)im=i[k];if(i[k]>iM)iM=i[k]} printf "  ranks=%d calls %d..%d staged %d..%d refused %d..%d share_out %d..%d sum=%d share_in %d..%d sum=%d | wait_ms sum=%.1f max=%.1f work_ms sum=%.1f max=%.1f | rows_out sum=%d chunks_in sum=%d\n",NR,cm,cM,sm,sM,rm,rM,om,oM,so,im,iM,si,sw,mw,sk,mk,ro,ci}'
  python3 - <<'PY'
import glob,statistics as st
tot=[];med=[]
for f in glob.glob("wtimer.*.csv"):
    v=[float(l.split(',')[1]) for l in open(f) if l[:1].isdigit()]
    if v: tot.append(sum(v)); med.append(st.median(v))
if tot: print("  micro per call: median-of-rank-medians %.4f s; rank total median %.2f max %.2f s"%(st.median(med),st.median(tot),max(tot)))
wf=[]
for f in glob.glob("wtimer.*.csv"):
    w=[float(p[7]) for p in (l.split(',') for l in open(f) if l[:1].isdigit()) if len(p)>7 and p[7].strip()]  # call 0 has an empty mpi_wait field
    if w: wf.append(1000*sum(w)/len(w))
if wf: print("  mpi_wait ms/step over ranks: floor(min) %.2f median %.2f max %.2f"%(min(wf),st.median(wf),max(wf)))
PY
  grep -iE "segmentation|signal|forrtl|backtrace" wrf.out 2>/dev/null | head -5
}
echo "=== A: nothing staged, 130 steps ===" ; arm A 0 none block
echo "=== RHS4: rad4 + share, cyclic (shape-only compat fix; pass1=29 rows + 4 per-row chunks) ===" ; arm RHS4 0 none cyclic rad4 rhs
echo "=== RHS8: rad8 + share, cyclic (8-row tail covers the ~5-row spread; 8 extra driver entries) ===" ; arm RHS8 0 none cyclic rad8 rhs
echo "=== RHS31: rad31 + share, cyclic (mechanism stress: 31 per-row chunks, most stealing) ===" ; arm RHS31 0 none cyclic rad31 rhs

diff_pair () {
  local X=$1 Y=$2 FX FY
  echo; echo "=== BIT-IDENTITY $X vs $Y ==="
  FX=$(ls -t "/home/u5727520/wrf-bench/st_${X}_$SLURM_JOB_ID"/wrfout_d01_* 2>/dev/null | head -1)
  FY=$(ls -t "/home/u5727520/wrf-bench/st_${Y}_$SLURM_JOB_ID"/wrfout_d01_* 2>/dev/null | head -1)
  if [ -n "$FX" ] && [ "$(basename "$FX")" = "$(basename "$FY")" ]; then
    local D="/home/u5727520/wrf-bench/st_${Y}_$SLURM_JOB_ID/diffwrf_${X}_${Y}.txt"
    "$W/external/io_netcdf/diffwrf" "$FX" "$FY" > "$D" 2>&1   # full list kept; 1070403's tail -16 hid the head
    echo "  differing-field lines: $(grep -c '^ *[A-Z0-9_]\+ \+[0-9]' "$D")  (full: $D)"
    head -40 "$D"
  else echo "  SKIP: $(basename "${FX:-none}") vs $(basename "${FY:-none}")"; fi
}
for Y in RHS4 RHS8 RHS31; do diff_pair A $Y; done
echo; echo "=== rad-step wall time (Timing for main at steps 1/61/121) and median of the rest ==="
for X in A RHS4 RHS8 RHS31; do
  F=/home/u5727520/wrf-bench/st_${X}_$SLURM_JOB_ID/rsl.out.0000
  R=$(grep "Timing for main" "$F" | awk '{v[NR]=$9} END{for(i=1;i<=NR;i++)if(i==1||i==61||i==121)printf " step%d=%.3f",i,v[i]}')
  M=$(grep "Timing for main" "$F" | awk 'NR!=1&&NR!=61&&NR!=121{print $9}' | sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.5f (n=%d)",v[int((NR+1)/2)],NR}')
  echo "  $X$R | other-steps med=$M"
done

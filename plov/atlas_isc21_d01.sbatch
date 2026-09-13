#!/usr/bin/env bash
#SBATCH --job-name=isc21d01
#SBATCH --account=ACD110018
#SBATCH --partition=ct448
#SBATCH --nodes=4 --ntasks-per-node=112
#SBATCH --time=00:25:00
#SBATCH --output=/home/u5727520/wrf-bench/atlas/isc21/plov-d01-%j.out
#
# CP-1 ISC21 3-domain, d01-only (max_dom=1), 400 steps (dt=15 -> 100 min sim, radt=3 -> 33 rad steps).
# Two arms in one job, same inputs, same decomposition (wtune_shape: 7x32, numtiles=12):
#   A  = block placement, no staging (count-only; also the imbalance census)
#   Ec = production.env: adapt75 staging + node-local stealing, cyclic placement
# Pass: diffwrf(A,Ec) empty; Ec footer staged>0 + STAGING ON; report s/step delta.
# Derived from wtune/plov/stage.sbatch; no real.exe (atlas ships wrfinput/wrfbdy).
set -uo pipefail
W=/home/u5727520/wrf-stack/wrf/intel-adios2-dmsm/WRF
SRC=/home/u5727520/wrf-bench/atlas/isc21/WRF_ISC21_3_DOMAIN
ROOT=/home/u5727520/wrf-bench/atlas/isc21
STEPS=400
NPX=7; NPY=32; NT=12
module load intel/2024_01_46 2>/dev/null || true
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so OMP_STACKSIZE=512M OMP_PROC_BIND=false KMP_BLOCKTIME=0
for L in NETCDF PNETCDF HDF5 ADIOS2; do export $L=/home/u5727520/wrf-stack/intel; done
export LD_LIBRARY_PATH=/home/u5727520/wrf-stack/intel/lib64:/home/u5727520/wrf-stack/intel/lib:${LD_LIBRARY_PATH:-}

P=$ROOT/prep_d01_$SLURM_JOB_ID
rm -rf "$P"; mkdir -p "$P"; cd "$P"
cp "$SRC/namelist.input" .
DT=$(grep -oE "time_step *= *[0-9]+" namelist.input | head -1 | grep -oE "[0-9]+$")
MIN=$(( (STEPS * DT + 59) / 60 ))
sed -i -E "s/^ *run_minutes *=.*/   run_minutes = $MIN,/" namelist.input
sed -i -E "s/^ *max_dom *=.*/   max_dom = 1,/" namelist.input
sed -i -E "s/^ *history_interval_s *=.*/   history_interval = $MIN,/" namelist.input
sed -i -E "s/^ *io_form_history *=.*/   io_form_history = 2,/" namelist.input
sed -i -E "s/^ *io_form_auxhist24 *=.*/   io_form_auxhist24 = 0,/" namelist.input
sed -i -E "/^ *iofields_filename/d" namelist.input
sed -i -E "/^ *nproc_x *=/d; /^ *nproc_y *=/d; /^ *numtiles *=/d" namelist.input
sed -i -E "s/^ *max_dom = 1,/   max_dom = 1,\n   nproc_x = $NPX, nproc_y = $NPY, numtiles = $NT,/" namelist.input
grep -nE "run_minutes|max_dom|history_interval|io_form|nproc|numtiles|auxhist24" namelist.input
echo "prep ok (steps=$STEPS dt=$DT min=$MIN)"

arm () {  # $1=tag $2=stage rows (0 = off) $3=share|adapt|adapt75 $4=srun --distribution (default block)
  local TAG=$1 ROWS=$2 D=$ROOT/st_${1}_$SLURM_JOB_ID
  rm -rf "$D"; mkdir -p "$D"; cd "$D" || return 1
  for f in "$SRC"/wrfinput_d01 "$SRC"/wrfbdy_d01; do ln -sf "$f" "$(basename "$f")"; done
  for f in "$W"/run/*; do b=$(basename "$f")
    case "$b" in namelist.input|wrfout*|wrfinput*|wrfbdy*|wrfrst*|rsl.*) ;; *) [ -e "$b" ] || ln -sf "$f" "$b";; esac; done
  # Thompson tables: WRF rewrites them in place -> copy the case's own, never symlink
  for t in qr_acr_qgV2.dat qr_acr_qsV2.dat freezeH2O.dat; do rm -f "$t"; cp "$SRC/$t" "$t"; done
  cp "$P/namelist.input" .
  ln -sf "$W"/main/wrf_wtimer.exe ./run.exe
  export OMP_NUM_THREADS=2 WTIMER_DIR="$D" WTIMER_DEBUG_RANK=0
  unset WTIMER_LEWI WTIMER_STAGE_ROWS WTIMER_STAGE_DEBUG WTIMER_STAGE_SHARE WTIMER_STAGE_ADAPT WTIMER_STAGE_CHUNKS WTIMER_STAGE_MAX_PCT WTIMER_RAD_ROWS
  [ "$ROWS" != 0 ] && export WTIMER_STAGE_ROWS=$ROWS WTIMER_STAGE_DEBUG=1
  [ "${3:-}" = share ] && export WTIMER_STAGE_SHARE=1
  [ "${3:-}" = adapt75 ] && export WTIMER_STAGE_SHARE=1 WTIMER_STAGE_ADAPT=1 WTIMER_STAGE_CHUNKS=6 WTIMER_STAGE_MAX_PCT=75
  local T0=$(date +%s)
  timeout 900 srun --mpi=pmi2 --distribution=${4:-block} -n 224 --ntasks-per-node=56 -c 2 ./run.exe > wrf.out 2>&1
  echo "  srun rc=$? wall=$(( $(date +%s) - T0 ))s"
  echo "  distribution=${4:-block} rank:node for ranks 0-9: $(srun --distribution=${4:-block} -n 224 --ntasks-per-node=56 -c 2 bash -c 'echo $SLURM_PROCID:$SLURM_NODEID' | sort -t: -k1,1n | head -10 | tr "\n" " ")"
  local OK N MED
  OK=$(grep -c "SUCCESS COMPLETE WRF" rsl.error.0000 2>/dev/null || echo 0)
  N=$(grep -c "Timing for main" rsl.out.0000 2>/dev/null || echo 0)
  MED=$(grep "Timing for main" rsl.out.0000 | awk '{print $9}' | sort -n | awk '{v[NR]=$1} END{if(NR)printf "%.5f",(NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  echo "  $TAG rows=$ROWS ok=$OK steps=$N med=${MED:-NA} s/step"
  grep -h "selfcheck\|stage geometry\|\[wtimer\] share:\|\[wtimer\] adapt\|\[wtimer\] mark\|\[wtimer\] [23]d " rsl.error.0000 2>/dev/null | head -40
  echo "  micro readback over ranks (calls staged refused share_out share_in): min/max and sums"
  grep -h "^# micro_calls" wtimer.*.csv 2>/dev/null | sed 's/[a-z_]*=//g; s/#//' | awk '{c[NR]=$1;s[NR]=$2;r[NR]=$3;o[NR]=$4;i[NR]=$5;so+=o[NR];si+=i[NR];sw+=$6;sk+=$7;ro+=$8;ci+=$9;if($6>mw)mw=$6;if($7>mk)mk=$7} END{if(!NR){print "  ranks=0"; exit} cm=c[1];cM=c[1];sm=s[1];sM=s[1];rm=r[1];rM=r[1];om=o[1];oM=o[1];im=i[1];iM=i[1];for(k=2;k<=NR;k++){if(c[k]<cm)cm=c[k];if(c[k]>cM)cM=c[k];if(s[k]<sm)sm=s[k];if(s[k]>sM)sM=s[k];if(r[k]<rm)rm=r[k];if(r[k]>rM)rM=r[k];if(o[k]<om)om=o[k];if(o[k]>oM)oM=o[k];if(i[k]<im)im=i[k];if(i[k]>iM)iM=i[k]} printf "  ranks=%d calls %d..%d staged %d..%d refused %d..%d share_out %d..%d sum=%d share_in %d..%d sum=%d | wait_ms sum=%.1f max=%.1f work_ms sum=%.1f max=%.1f | rows_out sum=%d chunks_in sum=%d\n",NR,cm,cM,sm,sM,rm,rM,om,oM,so,im,iM,si,sw,mw,sk,mk,ro,ci}'
  # imbalance census: per physics kind, per-step max/mean and max/median over ranks; rank totals
  python3 - "$TAG" <<'PY'
import glob,sys,statistics as st
tag=sys.argv[1]; kinds=["micro","radiation","pre_radiation","pbl","surface","cumulus"]
rows={}  # rank -> list of per-call tuples
for f in glob.glob("wtimer.*.csv"):
    r=int(f.split('.')[1]); v=[]
    for l in open(f):
        if l[:1].isdigit():
            p=l.rstrip('\n').split(','); v.append([float(x) if x.strip() else 0.0 for x in p[1:7]]+[float(p[7]) if len(p)>7 and p[7].strip() else 0.0])
    rows[r]=v
if not rows: print("  census: no csv"); sys.exit()
n=min(len(v) for v in rows.values()); R=sorted(rows)
print("  census %s: ranks=%d steps=%d"%(tag,len(R),n))
for k,name in enumerate(kinds+["mpi_wait"]):
    per_step_mx=[];per_step_mn=[];per_step_md=[];tot=[]
    for r in R: tot.append(sum(rows[r][i][k] for i in range(n)))
    for i in range(n):
        s=[rows[r][i][k] for r in R]
        if name!="mpi_wait" and sum(s)==0: continue
        per_step_mx.append(max(s)); per_step_mn.append(sum(s)/len(s)); per_step_md.append(st.median(s))
    if not per_step_mx or sum(per_step_mn)==0: continue
    act=len(per_step_mx)
    print("  %-14s active_steps=%d  sum(max)=%.2fs sum(mean)=%.2fs -> max/mean=%.3f  median-step max/median=%.3f  rank-total min/med/max=%.2f/%.2f/%.2f s  (max-mean summed = %.2fs recoverable ceiling)"%(
        name,act,sum(per_step_mx),sum(per_step_mn),sum(per_step_mx)/sum(per_step_mn),st.median([a/b if b>0 else 0 for a,b in zip(per_step_mx,per_step_md)]),min(tot),st.median(tot),max(tot),sum(per_step_mx)-sum(per_step_mn)))
# where is the micro load: per-node sums (block: 56 consecutive ranks; cyclic: r mod 4) and top-8 ranks
mt={r:sum(rows[r][i][0] for i in range(n)) for r in R}
top=sorted(R,key=lambda r:-mt[r])[:8]
print("  micro top-8 ranks: "+" ".join("%d:%.1f"%(r,mt[r]) for r in top))
print("  micro node sums if block(56/node): %s ; if cyclic(r mod 4): %s"%(
    ["%.1f"%sum(mt[r] for r in R if r//56==k) for k in range(4)],["%.1f"%sum(mt[r] for r in R if r%4==k) for k in range(4)]))
PY
  grep -iE "segmentation|signal|forrtl|backtrace" wrf.out 2>/dev/null | head -5
}
echo "=== A: block placement, no staging (count-only census), $STEPS steps ===" ; arm A 0 none block
echo "=== Ec: production adapt75 + cyclic placement, $STEPS steps ===" ; arm Ec 4 adapt75 cyclic

echo; echo "=== BIT-IDENTITY A vs Ec ==="
FX=$(ls -t "$ROOT/st_A_$SLURM_JOB_ID"/wrfout_d01_* 2>/dev/null | head -1)
FY=$(ls -t "$ROOT/st_Ec_$SLURM_JOB_ID"/wrfout_d01_* 2>/dev/null | head -1)
if [ -n "$FX" ] && [ -n "$FY" ] && [ "$(basename "$FX")" = "$(basename "$FY")" ]; then
  "$W/external/io_netcdf/diffwrf" "$FX" "$FY" 2>&1 | tail -16
else echo "  SKIP: $(basename "${FX:-none}") vs $(basename "${FY:-none}")"; fi
for X in A Ec; do grep "Timing for main" "$ROOT/st_${X}_$SLURM_JOB_ID/rsl.out.0000" | awk -v t=$X '{v[NR]=$9} END{for(q=0;q<4;q++){n=0;delete w;for(i=q*100+1;i<=(q+1)*100&&i<=NR;i++)w[++n]=v[i];asort(w);printf "  %s steps %d-%d med=%.5f\n",t,q*100+1,(q+1)*100,w[int((n+1)/2)]}}'; done
for X in A Ec; do grep "Timing for main" "$ROOT/st_${X}_$SLURM_JOB_ID/rsl.out.0000" | awk -v t=$X '{s+=$9;n++} END{if(n)printf "  %s mean over %d steps = %.5f s/step (total %.1f s)\n",t,n,s/n,s}'; done

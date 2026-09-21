#!/usr/bin/env bash
# Gate-1 per-node collector: one overlapping task per node, never touches WRF (perf attach + /proc reads).
# args: RUNDIR OUTDIR EVENTSPEC RANKSPEC(rank:role,...)
set -u
RD=$1; OD=$2; EV=$3; SPEC=$4
H=$(hostname); A=/home/u5727520/wrf-stack/wtune/rtctl/score/g1_aggregate.py
mkdir -p "$OD"
steps () { awk '/Timing for main/{n++} END{print n+0}' "$RD/rsl.out.0000" 2>/dev/null; }
declare -A ROLE PID REC
for kv in ${SPEC//,/ }; do ROLE[${kv%%:*}]=${kv##*:}; done
until [ "$(steps)" -ge 5 ]; do sleep 0.2; done
for p in $(pgrep -u "$USER" -x wrf.exe); do
  r=$(tr '\0' '\n' < /proc/$p/environ | awk -F= '$1=="SLURM_PROCID"{print $2}')
  [ -n "${ROLE[$r]:-}" ] || continue
  PID[$r]=$p
  echo "META host=$H rank=$r role=${ROLE[$r]} pid=$p home=$(grep Cpus_allowed_list /proc/$p/status | cut -f2) threads=$(ls /proc/$p/task | wc -l) step=$(steps)" >> "$OD/meta.$H.txt"
  perf record -q -o "$OD/perf.r$r.data" -p "$p" $EV 2> "$OD/perf.r$r.err" &
  REC[$r]=$!
done
[ ${#PID[@]} -eq 0 ] && { echo "META host=$H no selected ranks" >> "$OD/meta.$H.txt"; exit 0; }
until [ "$(steps)" -ge 20 ]; do sleep 0.2; done
for r in "${!PID[@]}"; do
  p=${PID[$r]}
  home_cpu=$(grep Cpus_allowed_list /proc/$p/status | cut -f2 | cut -d- -f1 | cut -d, -f1)
  home_node=$(for n in /sys/devices/system/node/node*; do python3 -c "
import sys
c=int(sys.argv[1]); s=open(sys.argv[2]).read().strip()
for part in s.split(','):
    a,_,b=part.partition('-')
    if int(a)<=c<=int(b or a): print(sys.argv[3])" "$home_cpu" "$n/cpulist" "${n##*node}"; done)
  awk -v r="$r" -v h="$H" -v hn="$home_node" '{kb=4; for(i=1;i<=NF;i++) if($i ~ /^kernelpagesize_kB=/){split($i,a,"="); kb=a[2]}
       for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split(substr($i,2),b,"="); s[b[1]]+=b[2]*kb; t+=b[2]*kb}}
       END{printf "NUMA host=%s rank=%s home_node=%s total_gb=%.2f local_frac=%.4f nodes=", h, r, hn, t/1048576, (t? s[hn]/t : 0);
           for(k in s) if(s[k]>10240) printf "%s:%.2f ", k, s[k]/1048576; print ""}' /proc/$p/numa_maps >> "$OD/meta.$H.txt"
done
until [ "$(steps)" -ge 35 ]; do sleep 0.2; done
for r in "${!REC[@]}"; do kill -INT "${REC[$r]}" 2>/dev/null; done
wait
echo "META host=$H perf stopped at step $(steps)" >> "$OD/meta.$H.txt"
for r in "${!REC[@]}"; do
  # sym requires ip in perf script; the aggregator strips the ip token
  perf script -i "$OD/perf.r$r.data" -F tid,time,period,event,ip,sym,dso 2>/dev/null | python3 "$A" "$H" "$r" "${ROLE[$r]}" >> "$OD/summary.$H.csv"
done
# compress only this node's ranks: the output directory is shared by all four nodes
for r in "${!REC[@]}"; do gzip -f "$OD/perf.r$r.data"; done

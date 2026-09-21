#!/usr/bin/env python3
"""Per-step, per-node summary of a predictive-PLOV arm from plov.<rank>.csv traces.

Trace line (one per micro call, written by wt_micro.c when WTIMER_TRACE=1):
  step,mode,role,entry_tsc,own_done_tsc,exit_tsc,pred_ms,rows_pub,chunks_pub,chunks_out,chunks_in,
  work_in_ms,assigned_out_ms,assigned_in_ms,wait_ms,idle_ms,pack_ms,commit_ms
  role: O owner, D donor, N neutral; *_tsc are absolute rdtsc ticks (same clock for every rank of a node).

Node = rank % 4 (all arms run with --distribution=cyclic). Completion of rank r in step s =
exit_tsc - min(entry_tsc over the node's ranks in that step). The objective is the node MAX completion.

usage: plov_trace_summary.py ARMDIR TAG [TSC_HZ=2.0e9] [WARM=3]
prints one SUMMARY line and writes ARMDIR/plov_steps.csv (step,node,max_ms,min_ms,spread_ms,owner_wait_ms,
helper_idle_ms,overhead_ms,chunks_out,work_in_ms,assigned_in_ms,pred_medape_pct)
"""
import collections
import csv
import glob
import re
import statistics
import sys

arm, tag = sys.argv[1], sys.argv[2]
HZ = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0e9
WARM = int(sys.argv[4]) if len(sys.argv) > 4 else 3
ms = lambda ticks: ticks / HZ * 1e3

rows = collections.defaultdict(dict)   # step -> rank -> dict
prev_actual = collections.defaultdict(dict)
for f in glob.glob(arm + "/plov.*.csv"):
    r = int(re.search(r"plov\.(\d+)\.csv$", f).group(1))
    for d in csv.DictReader(open(f)):
        s = int(d["step"])
        rows[s][r] = d
if not rows:
    print("SUMMARY tag=%s no trace files" % tag)
    sys.exit(0)

steps_out, per = [], collections.defaultdict(list)
for s in sorted(rows):
    if s < WARM:
        continue
    for n in range(4):
        rs = [r for r in rows[s] if r % 4 == n]
        if not rs:
            continue
        e0 = min(int(rows[s][r]["entry_tsc"]) for r in rs)
        comp = {r: ms(int(rows[s][r]["exit_tsc"]) - e0) for r in rs}
        own = {r: ms(int(rows[s][r]["own_done_tsc"]) - e0) for r in rs}
        wait = sum(float(rows[s][r]["wait_ms"]) for r in rs)
        idle = sum(float(rows[s][r]["idle_ms"]) for r in rs)
        over = sum(float(rows[s][r]["pack_ms"]) + float(rows[s][r]["commit_ms"]) for r in rs)
        cout = sum(int(rows[s][r]["chunks_out"]) for r in rs)
        win = sum(float(rows[s][r]["work_in_ms"]) for r in rs)
        ain = sum(float(rows[s][r]["assigned_in_ms"]) for r in rs)
        ape = []
        for r in rs:
            p = float(rows[s][r]["pred_ms"])
            a = own[r] if rows[s][r]["role"] != "D" else comp[r]
            if p > 0 and (s - 1) in rows and r in rows[s - 1]:
                act = ms(int(rows[s][r]["exit_tsc"]) - int(rows[s][r]["entry_tsc"]))
                if act > 0:
                    ape.append(abs(p - act) / act * 100)
        rec = dict(step=s, node=n, max_ms=round(max(comp.values()), 3), min_ms=round(min(comp.values()), 3),
                   spread_ms=round(max(comp.values()) - min(comp.values()), 3), owner_wait_ms=round(wait, 3),
                   helper_idle_ms=round(idle, 3), overhead_ms=round(over, 3), chunks_out=cout,
                   work_in_ms=round(win, 3), assigned_in_ms=round(ain, 3),
                   pred_medape_pct=round(statistics.median(ape), 3) if ape else "")
        steps_out.append(rec)
        for k in ("max_ms", "spread_ms", "owner_wait_ms", "helper_idle_ms", "overhead_ms", "chunks_out", "work_in_ms", "assigned_in_ms"):
            per[k].append(rec[k])
        if rec["pred_medape_pct"] != "":
            per["pred_medape_pct"].append(rec["pred_medape_pct"])

with open(arm + "/plov_steps.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(steps_out[0]))
    w.writeheader()
    w.writerows(steps_out)


def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] if s else float("nan")


print("SUMMARY tag=%s node_steps=%d node_max_ms_med=%.2f node_spread_ms_med=%.2f p90=%.2f owner_wait_ms_med=%.2f "
      "helper_idle_ms_med=%.2f overhead_ms_med=%.2f chunks_out_med=%.1f work_in_ms_med=%.2f assigned_in_ms_med=%.2f pred_medape_pct=%.2f"
      % (tag, len(steps_out), q(per["max_ms"], .5), q(per["spread_ms"], .5), q(per["spread_ms"], .9), q(per["owner_wait_ms"], .5),
         q(per["helper_idle_ms"], .5), q(per["overhead_ms"], .5), q(per["chunks_out"], .5), q(per["work_in_ms"], .5),
         q(per["assigned_in_ms"], .5), q(per["pred_medape_pct"], .5) if per["pred_medape_pct"] else float("nan")))

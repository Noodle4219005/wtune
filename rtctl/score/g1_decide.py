#!/usr/bin/env python3
"""Gate-1 verdict: why is the hot rank slower than its same-SNC-domain cold partner?

usage: g1_decide.py RUNDIR   (reads RUNDIR/g1/summary.csv and RUNDIR/g1/meta.*.txt)

Pre-registered rule (journal 2026-09-13): if hot and cold differ by < 5% in micro IPC, micro effective GHz
and micro LLC MPKI, and hot/cold micro instructions-per-window tracks the hot/cold micro window-length
ratio (within 10%), then A (work amount) dominates and OS-level slack stealing has little headroom;
a NUMA local fraction ~1.0 on hot ranks leaves D (and gate 4) with nothing to fix.
"""
import csv
import glob
import re
import sys

rd = sys.argv[1]
rows = list(csv.DictReader(open(rd + "/g1/summary.csv")))
micro = {int(r["rank"]): r for r in rows if r["class"] == "micro"}
allc = {int(r["rank"]): r for r in rows if r["class"] == "all"}
numa = {}
for f in glob.glob(rd + "/g1/meta.*.txt"):
    for l in open(f):
        m = re.match(r"NUMA host=\S+ rank=(\d+) home_node=(\S*) total_gb=([\d.]+) local_frac=([\d.]+)", l)
        if m:
            numa[int(m.group(1))] = (float(m.group(3)), float(m.group(4)), m.group(2))
pairs = [(12, 44), (9, 49), (10, 42), (11, 51)]


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return float("nan")


def pct(a, b):
    return (a / b - 1) * 100 if b else float("nan")


print("%-9s %6s %6s %6s %7s %7s %7s %8s %8s %8s %9s %9s" % (
    "pair", "Wratio", "Iratio", "I/W", "dIPC%", "dGHz%", "dMPKI%", "hot_util", "cold_util", "micro%hot", "locH", "locC"))
verdict_a = []
for h, c in pairs:
    if h not in micro or c not in micro:
        print("%d/%d missing" % (h, c))
        continue
    H, C = micro[h], micro[c]
    wr = f(H, "median_window_ms") / f(C, "median_window_ms")
    ir = f(H, "median_micro_instr_per_window") / f(C, "median_micro_instr_per_window")
    d_ipc, d_ghz, d_mpki = pct(f(H, "ipc"), f(C, "ipc")), pct(f(H, "eff_ghz"), f(C, "eff_ghz")), pct(f(H, "mpki"), f(C, "mpki"))
    share = f(H, "cycles") / f(allc[h], "cycles") * 100 if h in allc else float("nan")
    print("%-9s %6.2f %6.2f %6.2f %7.1f %7.1f %7.1f %8s %9s %8.1f %9s %9s" % (
        "%d/%d" % (h, c), wr, ir, ir / wr, d_ipc, d_ghz, d_mpki, H["window_util"], C["window_util"], share,
        "%.3f" % numa[h][1] if h in numa else "?", "%.3f" % numa[c][1] if c in numa else "?"))
    verdict_a.append(abs(d_ipc) < 5 and abs(d_ghz) < 5 and abs(d_mpki) < 5 and abs(ir / wr - 1) < 0.10)
print()
print("A dominates (all criteria) in %d of %d pairs" % (sum(verdict_a), len(verdict_a)))
locs = [numa[h][1] for h, _ in pairs if h in numa]
if locs:
    print("hot-rank NUMA local fraction: min %.3f  (gate 4 has a target only if clearly < 1)" % min(locs))

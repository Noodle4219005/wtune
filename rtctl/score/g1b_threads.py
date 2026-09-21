#!/usr/bin/env python3
"""Gate 1b: per-thread busy fraction inside each microphysics window, counting ALL samples (incl. libm).

stdin : perf script -F tid,time,period,event,ip,sym,dso  for one rank
argv  : RANK ROLE
stdout: one summary line per rank:
  rank role n_windows median_window_ms  thread busy medians (sorted high->low)  median_max_over_min_busy
  median_libm_share_in_window  median_window_util_all_classes

Window = run of micro-symbol samples (any thread) with gaps < 60 ms (same as g1_aggregate.py).
Thread busy fraction in a window = that thread's cycles (all classes, samples inside the window) /
(effective Hz x window seconds), effective Hz = window cycles / window ref-cycles x 2.0e9.
"""
import collections
import re
import statistics
import sys

rank, role = sys.argv[1:3]
LINE = re.compile(r"^\s*(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+?):?\s+(.*)$")
samples = []  # (t, tid, kind, period, is_micro, is_libm)
for line in sys.stdin:
    m = LINE.match(line)
    if not m:
        continue
    tid, t, period, ev, rest = m.groups()
    e = ev.lower()
    kind = "ref" if "ref-cycles" in e else "cyc" if "cycles" in e else None
    if kind is None:
        continue
    sym, _, dso = rest.rpartition(" (")
    head, _, tail = sym.strip().partition(" ")
    sym = tail.strip() if tail and re.fullmatch(r"[0-9a-fA-F]+", head) else sym.strip()
    micro = sym.startswith("module_mp_thompson") or sym.startswith("module_microphysics_driver")
    libm = bool(re.search(r"libimf|libm[.-]|libsvml", dso))
    samples.append((float(t), int(tid), kind, int(period), micro, libm))
samples.sort()
micro_t = [s[0] for s in samples if s[4]]
windows, start, last = [], None, None
for t in micro_t:
    if start is None:
        start = last = t
    elif t - last > 0.060:
        windows.append((start, last))
        start = last = t
    else:
        last = t
if start is not None:
    windows.append((start, last))

busy_hi, busy_lo, ratios, libm_share, util_all, durs = [], [], [], [], [], []
j = 0
for (a, b) in windows:
    d = b - a
    if d < 0.005:
        continue
    while j < len(samples) and samples[j][0] < a:
        j += 1
    k = j
    cyc_tid, cyc, ref, lib = collections.Counter(), 0, 0, 0
    while k < len(samples) and samples[k][0] <= b:
        t, tid, kind, p, mic, lm = samples[k]
        if kind == "cyc":
            cyc_tid[tid] += p
            cyc += p
            if lm:
                lib += p
        else:
            ref += p
        k += 1
    if not ref or len(cyc_tid) == 0:
        continue
    hz = cyc / ref * 2.0e9
    busy = sorted((c / (hz * d) for c in cyc_tid.values()), reverse=True)
    top2 = (busy + [0.0, 0.0])[:2]
    busy_hi.append(top2[0])
    busy_lo.append(top2[1])
    ratios.append(top2[0] / top2[1] if top2[1] > 0 else float("inf"))
    libm_share.append(lib / cyc if cyc else 0.0)
    util_all.append(sum(top2) / 2)
    durs.append(d * 1e3)

med = lambda v: statistics.median(v) if v else float("nan")
print("rank=%s role=%s windows=%d median_window_ms=%.1f busy_thread1=%.2f busy_thread2=%.2f max_over_min=%.2f libm_share=%.2f util_all=%.2f"
      % (rank, role, len(durs), med(durs), med(busy_hi), med(busy_lo), med([r for r in ratios if r != float("inf")]),
         med(libm_share), med(util_all)))

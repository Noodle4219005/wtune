#!/usr/bin/env python3
"""Gate-1 hot-vs-cold attribution: aggregate one rank's `perf script` stream.

stdin : perf script -F tid,time,period,event,sym,dso   (user-mode samples of one WRF rank)
argv  : HOST RANK ROLE
stdout: CSV rows (no header) -> host,rank,role,class,cycles,instructions,ref_cycles,llc_misses,ipc,eff_ghz,mpki,
        n_windows,median_window_ms,median_micro_instr_per_window,window_util

Classes: micro (module_mp_thompson_*, module_microphysics_driver_*), libm (libimf/libm/libsvml DSOs),
mpi (libmpi DSO), other, all. A micro window is a run of micro samples (any thread) with gaps < 60 ms;
window_util = micro cycles in windows / (eff_hz * window_s * 2 threads).
Effective frequency = cycles/ref-cycles x 2.0 GHz (ref-cycles tick at the 2.0 GHz base, as validated
against scaling_cur_freq in knowledge/wrf/2026-08-21-the-cores-are-already-at-96-percent-of-turbo.md).
"""
import re
import statistics
import sys

host, rank, role = sys.argv[1:4]
LINE = re.compile(r"^\s*(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+?):?\s+(.*)$")
EV = ("cycles", "instructions", "ref_cycles", "llc_misses")
tot = {c: dict.fromkeys(EV, 0) for c in ("micro", "libm", "mpi", "other", "all")}
micro_samples = []  # (time_s, event, period)


def event_key(e):
    e = e.lower()
    if "ref-cycles" in e or "ref_cycles" in e:
        return "ref_cycles"
    if "instructions" in e:
        return "instructions"
    if "llc" in e or "cache-misses" in e:
        return "llc_misses"
    if "cycles" in e:
        return "cycles"
    return None


for line in sys.stdin:
    m = LINE.match(line)
    if not m:
        continue
    _tid, t, period, ev, rest = m.groups()
    k = event_key(ev)
    if k is None:
        continue
    period = int(period)
    sym, _, dso = rest.rpartition(" (")
    sym = sym.strip() or rest.strip()
    dso = dso.rstrip(")")
    head, _, tail = sym.partition(" ")
    if tail and re.fullmatch(r"[0-9a-fA-F]+", head):  # perf script prints the ip before the symbol
        sym = tail.strip()
    if sym.startswith("module_mp_thompson") or sym.startswith("module_microphysics_driver"):
        cls = "micro"
        micro_samples.append((float(t), k, period))
    elif re.search(r"libimf|libm[.-]|libsvml", dso):
        cls = "libm"
    elif "libmpi" in dso:
        cls = "mpi"
    else:
        cls = "other"
    tot[cls][k] += period
    tot["all"][k] += period

micro_samples.sort()
windows, cur = [], []
for s in micro_samples:
    if cur and s[0] - cur[-1][0] > 0.060:
        windows.append(cur)
        cur = []
    cur.append(s)
if cur:
    windows.append(cur)
durs, instr, util_num, util_den = [], [], 0.0, 0.0
for w in windows:
    d = max(w[-1][0] - w[0][0], 1e-3)
    cyc = sum(p for _, k, p in w if k == "cycles")
    ref = sum(p for _, k, p in w if k == "ref_cycles")
    ins = sum(p for _, k, p in w if k == "instructions")
    durs.append(d * 1e3)
    instr.append(ins)
    if ref:
        util_num += cyc
        util_den += (cyc / ref * 2.0e9) * d * 2
for cls, v in tot.items():
    ipc = v["instructions"] / v["cycles"] if v["cycles"] else float("nan")
    ghz = v["cycles"] / v["ref_cycles"] * 2.0 if v["ref_cycles"] else float("nan")
    mpki = v["llc_misses"] / v["instructions"] * 1e3 if v["instructions"] else float("nan")
    extra = ["", "", "", ""]
    if cls == "micro" and durs:
        extra = [len(durs), round(statistics.median(durs), 2), int(statistics.median(instr)),
                 round(util_num / util_den, 3) if util_den else ""]
    print(",".join(map(str, [host, rank, role, cls, v["cycles"], v["instructions"], v["ref_cycles"], v["llc_misses"],
                             round(ipc, 3), round(ghz, 3), round(mpki, 3)] + extra)))

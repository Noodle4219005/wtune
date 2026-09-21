#!/usr/bin/env python3
"""Did the helper (borrowed) OpenMP workers actually work, and where?

For each arm of a score job, reads tsample.<node>.csv (per-thread exec_ns / last CPU every ~5 ms
during the timestep phase), META home masks from tsample.<node>.log, and the borrower TIDs from
elasticd's PLAN lines, then reports over steps 3..37:
  - per thread: CPU rate (cores), CPU rate spent on CPUs outside the rank's home mask
  - per rank: total CPU rate across its threads (2 cores is the ceiling without borrowing)
  - thread churn: TIDs first seen after the window start or last seen before its end
  - whether each PLAN borrower TID exists in the sampled rank at all

usage: analyze_tsample.py JOBID [ARM ...]
"""
import collections
import csv
import glob
import re
import sys

ROOT = "/home/u5727520/wrf-bench/wscore"
job = sys.argv[1]
arms = sys.argv[2:] or ["V13a", "EA0", "EA1", "EA2"]
LO, HI = 3, 37


def cpulist(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


for arm in arms:
    d = "%s/s2_%s_%s" % (ROOT, arm, job)
    home, meta_threads = {}, {}
    for lg in glob.glob(d + "/tsample.*.log"):
        for l in open(lg):
            m = re.match(r"META rank=(\d+) pid=(\d+) home=(\S+) threads=(\d+)", l)
            if m:
                home[int(m.group(1))] = cpulist(m.group(3))
                meta_threads[int(m.group(1))] = int(m.group(4))
    borrowers = {}
    for c in glob.glob(d + "/ctl.*.out"):
        for l in open(c):
            m = re.match(r"PLAN slot=\d+ hot=(\d+)\(.*?\) tid=(\d+) donor=(\d+)", l)
            if m:
                borrowers[int(m.group(2))] = (int(m.group(1)), int(m.group(3)))
    series = collections.defaultdict(list)
    for f in glob.glob(d + "/tsample.*.csv"):
        for r in csv.DictReader(open(f)):
            series[(int(r["rank"]), int(r["tid"]))].append(
                (float(r["t_ms"]), int(r["step"]), r["state"], int(r["cpu"]), int(r["exec_ns"])))
    print("== %s (job %s) steps %d..%d, sampled ranks %s" % (arm, job, LO, HI, sorted(home)))
    if not series:
        print("   no samples")
        continue
    win_t = [x[0] for s in series.values() for x in s if LO <= x[1] <= HI]
    t_lo, t_hi = min(win_t), max(win_t)
    by_rank = collections.defaultdict(list)
    for (rank, tid), s in series.items():
        s.sort()
        w = [x for x in s if LO <= x[1] <= HI]
        churn = []
        if s[0][0] > t_lo + 50:
            churn.append("born@step%d" % s[0][1])
        if s[-1][0] < t_hi - 50:
            churn.append("gone@step%d" % s[-1][1])
        if len(w) < 2:
            by_rank[rank].append((tid, None, None, churn, 0))
            continue
        span = (w[-1][0] - w[0][0]) / 1000.0
        total = w[-1][4] - w[0][4]
        off = sum(b[4] - a[4] for a, b in zip(w, w[1:]) if rank in home and b[3] not in home[rank])
        by_rank[rank].append((tid, total / 1e9 / span, off / 1e9 / span, churn, len(w)))
    for rank in sorted(by_rank):
        rows = sorted(by_rank[rank])
        tot = sum(r[1] for r in rows if r[1] is not None)
        off = sum(r[2] for r in rows if r[2] is not None)
        print("  rank %3d home %-12s threads(meta) %s  rank_cpu %.2f cores  of which off-home %.2f" % (
            rank, ",".join(map(str, sorted(home.get(rank, [])))), meta_threads.get(rank, "?"), tot, off))
        for tid, rate, offr, churn, n in rows:
            tag = "  <-- BORROWER hot %d -> donor %d" % borrowers[tid] if tid in borrowers else ""
            if rate is None:
                print("      tid %d: <2 samples in window %s%s" % (tid, " ".join(churn), tag))
            else:
                print("      tid %d: cpu %.2f cores, off-home %.2f, samples %d %s%s" % (tid, rate, offr, n, " ".join(churn), tag))
    sampled_tids = {tid for (_, tid) in series}
    sampled_hot = {h for h, _ in borrowers.values() if h in home}
    missing = [(tid, hd) for tid, hd in borrowers.items() if hd[0] in sampled_hot and tid not in sampled_tids]
    if borrowers:
        print("  PLAN borrowers of sampled hot ranks not present among their threads during steps: %s" % (missing or "none"))

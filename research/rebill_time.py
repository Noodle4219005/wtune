#!/usr/bin/env python3
"""Fine-grained re-billing of per-rank cost, aligned by TIME instead of by call index.

wtune_phase_analysis.py aligns phases on kind=3 (MPI_Allreduce) because "segment k is the same
phase on every rank" is false -- cross-rank correlation of the per-segment series is +0.22, so
indexing by call number produces garbage. But two tagged runs produced kind=0 only: in this
configuration every intercepted call is MPI_Wait and Allreduce never appears inside a 60,000
call window. The anchor does not exist.

tend_s does. Every call carries an absolute time from a barrier-synced t0 at ~3.5 ms
resolution against a ~0.7 s timestep, so binning on absolute time aligns ranks with no anchor
call at all. A bin well under one timestep is a phase slice in all but name.

Streams the file and accumulates straight into the bin matrices. The first version built
Python lists of all 26.88M rows and was OOM-killed in 7 seconds; nothing about the analysis
needs the rows retained.
"""
import sys, os
import numpy as np

def main():
    path = sys.argv[1]
    bins_ms = [float(x) for x in sys.argv[2:]] or [10.0, 50.0, 200.0, 1000.0]
    # pass 1: extents only
    nr, tmin, tmax, nrow = 0, None, None, 0
    with open(path) as fh:
        fh.readline()
        for line in fh:
            f = line.split(',')
            if len(f) < 6: continue
            r = int(f[0]); t = float(f[5])
            nr = max(nr, r + 1); nrow += 1
            if tmin is None or t < tmin: tmin = t
            if tmax is None or t > tmax: tmax = t
    span = tmax - tmin
    print("%s: %d rows, %d ranks, span %.1f s" % (os.path.basename(path), nrow, nr, span))
    mats = {}
    for bm in bins_ms:
        nb = max(1, int(np.ceil(span / (bm / 1000.0))) + 1)
        mats[bm] = np.zeros((nr, nb), dtype=np.float32)
    tot = np.zeros(nr)
    # pass 2: accumulate
    with open(path) as fh:
        fh.readline()
        for line in fh:
            f = line.split(',')
            if len(f) < 6: continue
            r = int(f[0]); c = float(f[2]); t = float(f[5]) - tmin
            tot[r] += c
            for bm, M in mats.items():
                b = int(t / (bm / 1000.0))
                if b < M.shape[1]: M[r, b] += c
    print("  pooled M (max-mean)/span = %5.1f%%" % (100 * (tot.max() - tot.mean()) / span))
    print()
    print("  %-10s %10s %12s %12s %10s" % ("bin", "bins", "billed s", "floor s", "prize"))
    for bm in sorted(mats):
        M = mats[bm]
        billed = float(M.max(axis=0).sum()); floor = float(M.mean(axis=0).sum())
        print("  %-10s %10d %12.1f %12.1f %9.1f%%"
              % ("%.0f ms" % bm, M.shape[1], billed, floor,
                 100 * (billed - floor) / billed if billed else 0))
    print()
    print("READING: a bin far below one timestep bills the per-PHASE structure; a bin far above")
    print("  it collapses to the pooled statistic. Where the prize stops growing as the bin")
    print("  shrinks is the granularity at which the imbalance actually lives.")
    return 0

sys.exit(main())

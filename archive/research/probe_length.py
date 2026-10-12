#!/usr/bin/env python3
"""How long must you actually RUN before the per-rank cost pattern is known?

The alternative to predicting per-rank cost from weather features is to measure it: run the
first K seconds of a segment, read the per-rank cost, and use that to rank the hot ranks for
the rest. If a short probe wins, most of the ML model is unnecessary for the hot-spot question.

Two facts make this worth testing rather than assuming:
  * persistence (last segment's MEASURED cost) already scores 82% against the model's 85%,
    so measurement is nearly as good as prediction even when it is a segment out of date;
  * the probe is very nearly FREE. Those steps have to be run anyway. Write the restart at
    the end of the probe instead of at the segment boundary and the probe's compute is not
    discarded -- the only cost is the 11.6 s restart, against 17.6-152.8 s recoverable.

PRE-REGISTERED PREDICTION (written before any number is seen):
  radiation is called every radt=15 min of model time = 60 steps at dt=15 s = ~36 s of wall.
  The dominant imbalance in this model IS the solar terminator, i.e. radiation. So a probe
  shorter than ~36 s samples zero radiation calls and must systematically mis-rank the hot
  ranks -- while looking perfectly well-behaved. Expect a step change in hit rate somewhere
  around 3-4 buckets, not a smooth curve.

usage: probe_length.py [case ...]     (default: every probe_* case)
"""
import sys, os, glob
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np

RESULTS = '/home/u5727520/wrf-results'
TOPFRAC = 0.10


def load_buckets(case):
    """-> comp[rank, bucket] for one case, or None."""
    p = f'{RESULTS}/{case}/bucket_load.csv'
    if not os.path.exists(p):
        return None
    rows = np.genfromtxt(p, delimiter=',', names=True)
    nr = int(rows['rank'].max()) + 1
    nb = int(rows['bucket'].max()) + 1
    comp = np.zeros((nr, nb))
    for r, b, c in zip(rows['rank'], rows['bucket'], rows['comp_s']):
        comp[int(r), int(b)] = c
    # trailing buckets are empty once the run ends; the shim clamps overflow into the last one
    used = comp.sum(axis=0) > 0
    if used.sum() < 8:
        return None
    last = int(np.where(used)[0].max())
    return comp[:, :last]          # drop the clamped final bucket


def hit(score, truth, topk):
    a = set(np.argsort(truth)[-topk:]); b = set(np.argsort(score)[-topk:])
    return 100.0 * len(a & b) / topk


def main():
    cases = sys.argv[1:] or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(f'{RESULTS}/probe_*/bucket_load.csv'))
    mats = {}
    for c in cases:
        m = load_buckets(c)
        if m is not None:
            mats[c] = m
    if not mats:
        print("no usable probe cases"); return 2

    KS = [1, 2, 3, 4, 6, 9, 12, 18, 30]
    print("bucket = 10 s of wall ~= 17 steps at 0.6 s/step")
    print("radiation every 60 steps ~= 36 s ~= 3.6 buckets  <-- predicted threshold")
    print()
    print("%-22s %6s %8s  %s" % ("case", "ranks", "buckets", "  ".join("K=%-2d" % k for k in KS)))
    allrows = []
    for c, comp in sorted(mats.items()):
        nr, nb = comp.shape
        topk = max(1, int(round(TOPFRAC * nr)))
        row = []
        for K in KS:
            if K >= nb - 2:
                row.append(float('nan')); continue
            probe = comp[:, :K].sum(axis=1)          # what a K-bucket probe would measure
            rest = comp[:, K:].sum(axis=1)           # what the remaining run actually costs
            row.append(hit(probe, rest, topk))
        allrows.append(row)
        print("%-22s %6d %8d  %s" % (c, nr, nb,
              "  ".join("%4.1f" % v if np.isfinite(v) else "  --" for v in row)))
    A = np.array(allrows, float)
    print("%-22s %6s %8s  %s" % ("MEAN", "", "",
          "  ".join("%4.1f" % v for v in np.nanmean(A, axis=0))))
    print()
    print("seconds of wall probed:  " + "  ".join("%4d" % (k * 10) for k in KS))
    print()
    print("compare: GBRT own+geo on this corpus = 48.3%% hot-spot hit rate")
    print("         persistence (previous segment)      = 82%% in the closed-loop replay")
    print("         random                              = %.0f%%" % (100 * TOPFRAC))
    print()
    print("READING: a step change near K=4 confirms the radiation-interval floor -- the probe")
    print("         must span at least one radiation call. A smooth curve refutes it and means")
    print("         the imbalance is carried by something sampled every step.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""How long should a segment be, if the placement is rebuilt from the previous one?

WHY THIS IS OFFLINE

The measured relief was real but partial: node imbalance fell 10.4% -> 6.2%, yet the hot ranks
ended up [71, 16, 9, 4] across the nodes instead of the [26.7, 24.4, 24.4, 24.4] the placement
was built for. The reason is turnover -- about a quarter of the hot set changes over a 6 h
segment, and spread_map only moves the ranks that WERE hot, leaving the neighbours that become
hot next.

Shorter segments track better and cost more boundaries. Finding the balance by running is
expensive; the tracking half of it does not need a run at all. The default arm's 48 h profile
is a per-rank cost time series at 10 s resolution, so any segment length can be cut out of it
offline and scored.

THE CONFOUND THIS SEPARATES

Shorter segments also give NOISIER hot-set estimates: a 2 h segment holds ~19 usable buckets
against a 6 h segment's ~58. Measured overlap between adjacent segments came out LOWER for 2 h
(56%) than for 6 h (75%), which is backwards for a decaying signal and is the estimator, not
the weather. So the estimation window is swept INDEPENDENTLY of the segment length: a short
segment can still be planned from a long look-back, and only then does the segment-length
number mean "how stale", rather than "how stale and how noisy at once".

WHAT IS ASSUMED, AND IT IS NOT FREE

Scoring uses the DEFAULT arm's cost trajectory as ground truth for every candidate placement.
That assumes a rank's cost does not depend on where it sits -- which is false, because the
label is time-not-in-MPI and includes contention. The bias is toward optimism: a real run's
hot set shifts partly BECAUSE of the placement. So these numbers bound the tracking quality
from above and the winner still needs one confirmatory run.
"""
import os
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_place as wp
from decay_gate import load_bucket_csv

RES = '/home/u5727520/wrf-results'
NR, NN, NPN = 448, 4, 112
TOPK = max(1, int(round(0.10 * NR)))
ARMS = ['follow_AB_1A', 'follow_BA_2A']          # the two complete default-placement 48 h runs


def load_arm(d):
    """-> per-rank cost per 10 s bucket over the whole run, segments concatenated in order."""
    cols = []
    for k in range(64):
        f = os.path.join(RES, d, 'bucket_seg%02d.csv' % k)
        if not os.path.isfile(f):
            break
        comp, bsec = load_bucket_csv(f)
        if comp.shape[0] != NR:
            continue
        used = np.where(comp.sum(axis=0) > 0)[0]
        if used.size < 3:
            continue
        # drop bucket 0 (initialisation) and the partial tail of each segment
        cols.append(comp[:, 1:int(used.max())])
    return (np.hstack(cols), bsec) if cols else (None, None)


def hot(mat):
    t = mat.sum(axis=1)
    return set(np.argsort(t)[-TOPK:]), t


def score(mat, seg_buckets, look_buckets):
    """-> (mean hot share on the busiest node, mean node imbalance, mean overlap) over segments.

    seg_buckets  how long a segment is
    look_buckets how much history immediately before the boundary is used to plan. It is NOT
                 capped at seg_buckets: a controller standing at a boundary has every earlier
                 segment's profile on disk, so it may look back further than one segment. The
                 first version did cap it, which silently made the "hold the look-back fixed"
                 sweep identical to the main one and proved nothing.
    """
    n = mat.shape[1]
    nseg = n // seg_buckets
    if nseg < 3:
        return None
    shares, imbs, ovs = [], [], []
    for k in range(1, nseg):
        prev = mat[:, max(0, k * seg_buckets - look_buckets):k * seg_buckets]
        cur = mat[:, k * seg_buckets:(k + 1) * seg_buckets]
        hp, _ = hot(prev)
        hc, tc = hot(cur)
        node = wp.spread_map(NR, NN, np.array(sorted(hp)))
        shares.append(max(100.0 * sum(1 for r in hc if node[r] == q) / TOPK
                          for q in range(NN)))
        nl = np.array([tc[node == q].sum() for q in range(NN)])
        imbs.append(100.0 * (nl.max() - nl.mean()) / nl.mean())
        ovs.append(100.0 * len(hp & hc) / TOPK)
    return float(np.mean(shares)), float(np.mean(imbs)), float(np.mean(ovs)), nseg


mats = []
for d in ARMS:
    m, bsec = load_arm(d)
    if m is not None:
        mats.append((d, m, bsec))
        print('%-16s %d buckets of %.0f s = %.1f h of wall' % (d, m.shape[1], bsec,
                                                              m.shape[1] * bsec / 3600))
if not mats:
    print('no complete default arm found')
    sys.exit(2)

# the default map, for reference: no placement at all
d0, m0, bsec0 = mats[0]
BASE = np.arange(NR) // NPN
nseg6 = m0.shape[1] // int(round(6 * 3600 / (48 * 3600 / m0.shape[1] * 8)))  # unused, see below

print('\nsegment length sweep -- placement rebuilt from the previous segment')
print('the run is 48 model hours; a segment of L model-hours uses L/48 of the buckets\n')
print('%8s %7s | %11s %11s %9s | %11s' % ('seg (h)', 'nseg', 'busiest node',
                                          'node imbal', 'overlap', 'default imbal'))
for segh in (1, 2, 3, 4, 6, 8, 12):
    rows = []
    for d, m, _b in mats:
        sb = int(round(m.shape[1] * segh / 48.0))
        if sb < 6:
            continue
        r = score(m, sb, sb)                       # look-back = the whole previous segment
        if r:
            # what the DEFAULT map gives on the same segments, as the thing to beat
            n = m.shape[1] // sb
            dimb = []
            for k in range(1, n):
                cur = m[:, k * sb:(k + 1) * sb]
                t = cur.sum(axis=1)
                nl = np.array([t[BASE == q].sum() for q in range(NN)])
                dimb.append(100.0 * (nl.max() - nl.mean()) / nl.mean())
            rows.append(r + (float(np.mean(dimb)),))
    if rows:
        a = np.array([[x[0], x[1], x[2], x[4]] for x in rows]).mean(axis=0)
        print('%8d %7d | %10.1f%% %10.2f%% %8.0f%% | %10.2f%%'
              % (segh, rows[0][3], a[0], a[1], a[2], a[3]))

# Now the look-back really is held fixed and can reach back across boundaries, so the only
# thing changing is how STALE the plan is when the segment starts -- not how many buckets it
# was estimated from. If the numbers here track the ones above, shorter segments help because
# the plan is fresher; if they flatten out, the gain was mostly a better-averaged estimate.
print('\nlook-back held at 6 model-hours (can span earlier segments): only STALENESS varies')
print('%8s | %11s %11s %9s' % ('seg (h)', 'busiest node', 'node imbal', 'overlap'))
for segh in (1, 2, 3, 4, 6, 8, 12):
    rows = []
    for d, m, _b in mats:
        sb = int(round(m.shape[1] * segh / 48.0))
        lb = int(round(m.shape[1] * 6.0 / 48.0))
        if sb < 6:
            continue
        r = score(m, sb, lb)
        if r:
            rows.append(r)
    if rows:
        a = np.array([[x[0], x[1], x[2]] for x in rows]).mean(axis=0)
        print('%8d | %10.1f%% %10.2f%% %8.0f%%' % (segh, a[0], a[1], a[2]))

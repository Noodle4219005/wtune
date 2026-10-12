#!/usr/bin/env python3
"""Can a residual model supply the drift that copying one anchor forward misses?

THE SCHEME BEING TESTED

Simulate one short window, copy it forward, and let a model correct the drift between
segments -- repeat until the whole run is covered. Only ONE real simulation, so nothing has
to be integrated to reach a later anchor.

segment_cost_predict.py already bracketed what such a model is worth:

    k=1 hold (copy forward, NO drift correction)   9.5% total error   <- the floor to beat
    k=8 (re-measure every segment, unaffordable)   1.0% total error   <- the ceiling

So the model's entire value lives in that interval. Beating 9.5% means it helps; reaching
~1.0% means it has extracted everything there is.

THIS IS THE CEILING RUN, NOT THE DEPLOYABLE ONE

Features here come from the run's OWN wrfout frames, which in deployment do not exist yet --
that is the whole point of predicting. If a model cannot close the gap even with the true
future weather handed to it, no cheap surrogate for those features (coarse twin, met_em) can
rescue it, and the design is answered before any new simulation is spent.

TARGET

log(rate_j / rate_anchor) -- a RELATIVE drift, per segment. Every absolute quantity (machine,
rank count, e_vert, I/O backend, the case's own difficulty) cancels in the ratio, which is
exactly the half that the cross-case model in decisions/0045 could not predict.
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
from segment_cost_predict import run_profile, step_costs, ANCHOR_FRAC
from cost_model_v2 import FIELDS, SPREAD_OF

BENCH = '/home/u5727520/wrf-bench'
RESULTS = '/home/u5727520/wrf-results'
FRAME = re.compile(r'features_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})\.csv$')


def frame_features(path):
    with open(path) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        if not all(f in hdr for f in FIELDS):
            return None
        col = [hdr.index(f) for f in FIELDS]
        rows = []
        for r in rd:
            if len(r) != len(hdr):
                continue
            try:
                rows.append([float(r[c]) for c in col])
            except ValueError:
                continue
    if len(rows) < 10:
        return None
    X = np.asarray(rows, float)
    v = [float(X[:, j].mean()) for j in range(len(FIELDS))]
    for nm in SPREAD_OF:
        c = X[:, FIELDS.index(nm)]
        m = c.mean()
        v.append(float(np.percentile(c, 90) / m) if m > 0 else 1.0)
    return np.array(v)


def build(run, anchor_frac=None):
    """-> list of segments: (features, true mean rate), first is the anchor segment."""
    prof = run_profile(os.path.join(BENCH, run))
    if prof is None:
        return None
    h, w = prof
    _, clipped, _ = step_costs(h, w)
    dh = np.diff(h, prepend=h[0])
    dh[dh <= 0] = np.median(dh[dh > 0])
    rate = clipped / dh

    fdir = os.path.join(RESULTS, run)
    fr = []
    for p in sorted(glob.glob(os.path.join(fdir, 'features_*.csv'))):
        m = FRAME.search(p)
        if not m:
            continue
        f = frame_features(p)
        if f is not None:
            fr.append((p, f))
    if len(fr) < 3:
        return None
    n = len(fr)
    span = h[-1] / (n - 1)                     # frames bound n-1 segments
    segs = []
    for j in range(n - 1):
        a, b = j * span, (j + 1) * span
        m = (h >= a) & (h < b)
        if m.sum() < 20:
            return None
        segs.append((fr[j][1], float(np.mean(rate[m]))))
    # the anchor is the first anchor_frac of segment 0, not all of it -- that is what gets
    # simulated. It is a parameter because the window length, not the drift model, turned out
    # to be the dominant lever: at k=1 the error is almost pure downward BIAS (err 9.5% with
    # bias -9.5% at 2%, err 3.6% / -3.6% at 8%), because cost drifts upward and t=0 is always
    # the cheapest point. A longer window both shrinks the bias and is trivially affordable.
    aw = (ANCHOR_FRAC if anchor_frac is None else anchor_frac) * h[-1]
    ma = h < aw
    if ma.sum() < 10:
        return None
    anchor_rate = float(np.mean(rate[ma]))
    return segs, anchor_rate, span


def main():
    af = float(os.environ.get('ANCHOR_FRAC', ANCHOR_FRAC))
    runs = []
    for d in sorted(os.listdir(RESULTS)):
        if not os.path.isdir(os.path.join(RESULTS, d)):
            continue
        b = build(d, af)
        if b:
            runs.append((d,) + b)
    if len(runs) < 8:
        print('only %d usable runs' % len(runs))
        return 2
    print('anchor window = %.1f%% of the run' % (100 * af))
    nseg = sum(len(s) - 1 for _, s, _, _ in runs)
    print('%d runs, %d predicted segments (segment 0 is the anchor), %d features\n'
          % (len(runs), nseg, len(runs[0][1][0][0])))

    # rows: one per predicted segment. x = [anchor features, log ratio of segment features]
    X, Y, RID = [], [], []
    for i, (name, segs, arate, span) in enumerate(runs):
        f0 = segs[0][0]
        for j in range(1, len(segs)):
            fj = segs[j][0]
            ratio = np.log(np.maximum(fj, 1e-9) / np.maximum(f0, 1e-9))
            X.append(np.concatenate([f0, ratio, [j * span]]))
            Y.append(np.log(segs[j][1] / arate))
            RID.append(i)
    X, Y, RID = np.array(X), np.array(Y), np.array(RID)

    def totals(pred_log):
        """-> per-run |predicted total / true total - 1| in percent."""
        out = []
        for i, (name, segs, arate, span) in enumerate(runs):
            m = RID == i
            true = sum(s[1] for s in segs[1:]) * span + segs[0][1] * span
            p = arate * np.exp(pred_log[m])
            pt = float(p.sum()) * span + arate * span
            out.append(abs(pt / true - 1) * 100)
        return np.array(out)

    hold = totals(np.zeros(len(Y)))                       # copy forward, no correction
    oracle = totals(Y)                                    # the true drift, handed over
    from cost_model_v2 import fit_predict
    nf = len(runs[0][1][0][0])

    def loro(cols):
        pred = np.zeros(len(Y))
        for i in range(len(runs)):
            te = RID == i
            if te.sum() == 0:
                continue
            p, _ = fit_predict(X[~te][:, cols], Y[~te], X[te][:, cols], 'lgbm')
            pred[te] = p
        return totals(pred)

    # The split that decides whether this is deployable. f0 and the time offset are BOTH known
    # the moment the 2% anchor finishes -- the anchor run produces its own wrfout frame. The
    # ratio block is the future weather, which by definition does not exist yet. If most of the
    # skill sits in the deployable half, no coarse twin is needed at all.
    DEPLOY = np.concatenate([np.arange(nf), [2 * nf]])        # f0 + elapsed model hours
    ALL = np.arange(X.shape[1])
    # Control that has to be run before claiming the weather features do anything: elapsed
    # model hours ALONE. Cost drifts upward as a run develops, and a model given only the
    # clock can capture that without knowing any weather. The hot-spot model was long credited
    # with using weather when half its signal was three geometry columns; the same mistake is
    # available here and costs nothing to rule out.
    TIME_ONLY = np.array([2 * nf])
    tonly = loro(TIME_ONLY)
    dep = loro(DEPLOY)
    full = loro(ALL)

    print('%-46s %8s %8s' % ('', 'mean', 'worst'))
    for lab, v in (('copy forward, no correction (floor)', hold),
                   ('+ residual, elapsed model hours ONLY (control)', tonly),
                   ('+ residual, ANCHOR features only  [deployable]', dep),
                   ('+ residual, anchor + future weather [ceiling]', full),
                   ('+ TRUE drift (oracle)', oracle)):
        print('%-46s %7.1f%% %7.1f%%' % (lab, float(v.mean()), float(v.max())))
    gap = max(hold.mean() - oracle.mean(), 1e-9)
    print("\n  gap closed:  time-only %.0f%%   anchor-only %.0f%%   +future weather %.0f%%"
          % (100 * (hold.mean() - tonly.mean()) / gap,
             100 * (hold.mean() - dep.mean()) / gap,
             100 * (hold.mean() - full.mean()) / gap))
    print('  per-segment drift being predicted: sd of log(rate_j/rate_anchor) = %.3f '
          '(%.1f%% in rate)' % (float(Y.std()), 100 * (np.exp(Y.std()) - 1)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

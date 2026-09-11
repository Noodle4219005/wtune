#!/usr/bin/env python3
"""Simulate a little, predict a lot, repeat: how well does that recover a whole run's cost?

THE TARGET

Not seconds. A RELATIVE cost, where 120 is reliably slower than 60. That is a much weaker
requirement than absolute wall clock and it removes the part the cross-case model failed at:
machine, e_vert, rank count and I/O backend all sit in the measured anchor instead of having
to be predicted (see decisions/0045).

THE CONSTRAINT NOBODY CAN DESIGN AROUND

To profile model-hour 12 you must have integrated to model-hour 12. There is no seek. At
~98 s of wall per model-hour on 4 nodes, a 5-minute budget buys about 3 model-hours, so
anchors CANNOT be sprinkled across a 24 h run at full resolution. Anything that claims to is
either using a cheaper twin of the run or is wrong.

So this measures the CEILING first, on runs that already exist: given anchors wherever we
like -- which no 5-minute budget can actually deliver -- how good does the alternating scheme
get? If the ceiling is poor, a coarse-grid twin cannot rescue it and the design is dead
before any new simulation is spent. If the ceiling is good, the remaining question is only
how to obtain the anchors cheaply, which is a separate and answerable one.

WHAT IS MEASURED

sim_to_wall() gives (model time, cumulative wall) per step from rsl.out.0000, so the true
cost of every model-hour is known exactly. Predictors:

  const1     one anchor at the start, held over the whole run. The honest floor: this is what
             "profile the first minute and extrapolate" gets, and the 48 h decay curve
             predicts it should drift badly.
  hold-k     k anchors spread evenly; each holds until the next. The user's alternating
             scheme with perfect reachability. Sweeping k prices what each extra anchor buys.
  linear-k   same anchors, linearly interpolated between them instead of held.

Metrics are the ones the target actually needs:
  * total    |predicted total / actual total - 1|
  * order    over ALL runs, Spearman of predicted vs actual total, plus the number of pairs
             differing by >= 2x whose predicted order is WRONG. That last number is the
             literal reading of "120 must beat 60"; a Spearman of 0.9 can still hide it.
"""
import glob
import os
import re
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg

BENCH = '/home/u5727520/wrf-bench'
ANCHOR_FRAC = 0.02      # each anchor simulates 2% of the run: at 24 h that is ~29 model-min
BUDGET_S = 300.0        # the 5-minute promise


def run_profile(case_dir):
    """-> (model hours, cumulative wall seconds), or None if the trace cannot be trusted.

    THE GATE MATTERS MORE THAN THE PARSE. wrf-bench directories are restaged in place
    (`rm -rf` then re-link), so rsl.out.0000 is whatever ran there LAST, not necessarily the
    run whose profile is wanted. An audit against the harvested wall times found 15 of 76
    disagreeing: some rsl files were truncated to a few hundred steps of a killed run, and
    building a predictor on them produced 254% "errors" that were the corpus, not the model.

    Self-contained check, needing nothing but the directory: the trace must span the run
    length the namelist asked for. n_steps * time_step should equal run_hours.
    """
    rsl = os.path.join(case_dir, 'rsl.out.0000')
    nlp = os.path.join(case_dir, 'namelist.input')
    if not (os.path.isfile(rsl) and os.path.isfile(nlp)):
        return None
    tw = seg.sim_to_wall(rsl)
    if len(tw) < 200:
        return None
    t0 = tw[0][0]
    h = np.array([(t - t0).total_seconds() / 3600.0 for t, _ in tw])
    w = np.array([c for _, c in tw])
    txt = open(nlp, errors='ignore').read()

    def g(k):
        m = re.search(r'\b%s\s*=\s*([0-9]+)' % k, txt)
        return int(m.group(1)) if m else None

    want = (g('run_hours') or 0) + 24 * (g('run_days') or 0)
    dt = g('time_step')
    if not want or not dt:
        return None
    if abs(h[-1] - want) / want > 0.05:            # truncated or stale trace
        return None
    # Span alone is not enough. deep_conv_ot__mp6bl5 spans 11.6 h in 821 steps while its
    # time_step is 15 s, which would need 2784 -- the file has GAPS, so the cumulative wall it
    # reports is a fraction of the real one and the run looks 4x cheap. Checking step count
    # against time_step catches exactly that, and it is the check this docstring already
    # claimed to do.
    if abs(len(h) * dt / 3600.0 - want) / want > 0.05:
        return None
    return h, w


WRITE_MULT = 5.0        # a step costing more than this x the median is a history write


def step_costs(h, w):
    """-> (per-step wall cost, clipped cost, write surcharge).

    A history write happens INSIDE a time step, so one step in every history_interval costs
    tens of times a normal one. Those steps are not weather and they are not noise: they are a
    known, config-determined constant that every case in a comparison shares. Averaging them
    into an anchor is what made 4 anchors WORSE than 1 (mean 39%, worst 653%) -- with anchors
    at 0/25/50/75% of the run they land on write steps far more often than random, and holding
    an inflated rate over a quarter of the run inflates the total enormously.
    """
    c = np.diff(w, prepend=w[0])
    med = float(np.median(c[c > 0])) if np.any(c > 0) else 0.0
    clipped = np.minimum(c, WRITE_MULT * med) if med > 0 else c
    return c, clipped, float(c.sum() - clipped.sum())


def predict(h, w, k, mode):
    """k anchors, each simulating ANCHOR_FRAC of the run, then held or interpolated.

    Predicts the COMPUTE total (writes excluded); the write surcharge is returned separately
    because it is recoverable from the namelist, not from the weather.
    """
    total_h = h[-1]
    _, clipped, _ = step_costs(h, w)
    dh = np.diff(h, prepend=h[0])
    dh[dh <= 0] = np.median(dh[dh > 0]) if np.any(dh > 0) else 1.0
    rate = clipped / dh                            # wall seconds per model hour, writes clipped
    span = total_h / k
    awin = ANCHOR_FRAC * total_h
    ax, ay = [], []
    for i in range(k):
        s = i * span
        m = (h >= s) & (h < s + awin)
        if m.sum() < 2:
            m = (h >= s) & (h < s + 2 * awin)
        if m.sum() < 2:
            continue
        ax.append(float(h[m].mean()))
        # MEAN of the already-clipped rate, not the median. Radiation fires every radt (15 min
        # here), so an expensive radiation step is ~1 step in 60. A median throws all of them
        # away and the anchor reports only the cheap steps, which is why every prediction came
        # out systematically LOW (251 vs 331, 535 vs 641, 691 vs 820 ...). Clipping already
        # removed the history writes, so what is left for the mean to average is real work.
        #
        # This is the same constraint decay_gate.py documents for its measurement windows: a
        # window narrower than one radiation call misses it while reporting a perfectly
        # well-behaved cost field. At 2% of a 12 h run the window is 14.4 min against radt=15,
        # so it can contain zero radiation calls -- hence the ANCHOR_FRAC sweep below.
        ay.append(float(np.mean(rate[m])))
    if not ax:
        return None, 0.0
    ax, ay = np.array(ax), np.array(ay)
    if mode == 'hold':
        idx = np.clip(np.searchsorted(ax, h, side='right') - 1, 0, len(ay) - 1)
        pr = ay[idx]
    else:
        pr = np.interp(h, ax, ay)
    return float(np.trapezoid(pr, h)), len(ax) * awin


def main():
    cases = sorted(glob.glob(os.path.join(BENCH, '*')))
    rows = []
    for c in cases:
        if not os.path.isdir(c):
            continue
        p = run_profile(c)
        if p is None:
            continue
        h, w = p
        if h[-1] < 5.0:                             # need a run long enough to segment
            continue
        raw, clipped, wr = step_costs(h, w)
        rows.append({'case': os.path.basename(c), 'h': h, 'w': w,
                     'actual': float(w[-1]), 'compute': float(clipped.sum()),
                     'write': wr, 'model_h': float(h[-1])})
    if len(rows) < 6:
        print('only %d usable runs' % len(rows))
        return 2

    print('%d runs with per-step timing, %.0f-%.0f model hours\n'
          % (len(rows), min(r['model_h'] for r in rows), max(r['model_h'] for r in rows)))

    ws = 100.0 * np.median([r['write'] / r['actual'] for r in rows])
    print('history writes are %.1f%% of wall (median) and are excluded from the target: they '
          'are set by\nhistory_interval, not by the weather, and every case in a comparison '
          'carries the same term.\n' % ws)

    def report(label, preds):
        err = [abs(p / r['compute'] - 1) * 100 for p, r in zip(preds, rows) if p]
        a = np.array([r['compute'] for r, p in zip(rows, preds) if p])
        b = np.array([p for p in preds if p])
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        rho = np.corrcoef(ra, rb)[0, 1]
        bad = sum(1 for i in range(len(a)) for j in range(len(a))
                  if a[i] >= 2 * a[j] and b[i] <= b[j])
        npair = sum(1 for i in range(len(a)) for j in range(len(a)) if a[i] >= 2 * a[j])
        print('  %-14s total err mean %5.1f%%  worst %5.1f%%   rho %+.3f   '
              '2x-pairs ordered wrong: %d/%d'
              % (label, float(np.mean(err)), float(np.max(err)), rho, bad, npair))

    global ANCHOR_FRAC
    print('anchor window = %.0f%% of the run each' % (100 * ANCHOR_FRAC))
    for mode in ('hold', 'linear'):
        for k in (1, 2, 4, 8):
            preds = [predict(r['h'], r['w'], k, mode)[0] for r in rows]
            report('%s k=%d' % (mode, k), preds)
        print()

    # How long must one anchor window be? Radiation every radt makes this a threshold, not a
    # smooth trade: below one radiation interval the window can contain zero radiation calls
    # and the anchor is biased low no matter how many anchors are taken.
    base = ANCHOR_FRAC
    # Swept at k=1 as well as k=4, because that is the only deployable arm: k>=2 needs anchors
    # at model-hours you can only reach by integrating there. And the drift decomposition says
    # this is where the money is -- segment-to-segment drift is only 1.5% in rate, while the
    # 2% anchor differs from its OWN segment by much more, so most of what looked like drift is
    # just an unrepresentative window.
    for kk in (1, 4):
        print('anchor-window sweep (k=%d, hold) -- 12 h run, radt=15 min => 1 call per 2.1%% '
              'of run:' % kk)
        for f in (0.005, 0.01, 0.02, 0.04, 0.08, 0.16):
            ANCHOR_FRAC = f
            preds = [predict(r['h'], r['w'], kk, 'hold')[0] for r in rows]
            e = [abs(p / r['compute'] - 1) * 100 for p, r in zip(preds, rows) if p]
            sgn = [(p / r['compute'] - 1) * 100 for p, r in zip(preds, rows) if p]
            w = [abs(p / r['compute'] - 1) * 100 for p, r in zip(preds, rows) if p]
            print('  window %4.1f%% (%5.1f model-min at 12 h)  err %5.1f%%  worst %5.1f%%  '
                  'bias %+5.1f%%  simulated %4.1f%%'
                  % (100 * f, f * 12 * 60, float(np.mean(e)), float(np.max(w)),
                     float(np.mean(sgn)), 100 * kk * f))
        print()
    ANCHOR_FRAC = base

    # More anchors barely helped (23.8% -> 21.0% from k=1 to k=8) while the worst case stayed
    # at ~254%. Anchors fix DRIFT; an error that survives eight of them is structural, so name
    # the offenders instead of guessing at a mechanism.
    preds = [predict(r['h'], r['w'], 8, 'hold')[0] for r in rows]
    bad = sorted(((abs(p / r['compute'] - 1) * 100, r, p) for p, r in zip(preds, rows) if p),
                 key=lambda t: -t[0])[:6]
    print('worst runs at k=8 (where extra anchors do not help):')
    print('  %-26s %7s %9s %9s %8s %8s' % ('run', 'err', 'compute', 'pred', 'model_h', 'ramp'))
    for e, r, p in bad:
        _, clipped, _ = step_costs(r['h'], r['w'])
        n = len(clipped)
        first = float(np.median(clipped[:max(2, n // 50)]))     # first 2%
        rest = float(np.median(clipped[max(2, n // 50):]))
        print('  %-26s %6.0f%% %9.0f %9.0f %8.1f %7.2fx'
              % (r['case'], e, r['compute'], p, r['model_h'],
                 (first / rest) if rest > 0 else float('nan')))
    allramp = []
    for r in rows:
        _, clipped, _ = step_costs(r['h'], r['w'])
        n = len(clipped)
        a = float(np.median(clipped[:max(2, n // 50)]))
        b = float(np.median(clipped[max(2, n // 50):]))
        if b > 0:
            allramp.append(a / b)
    print('  first-2%% median step cost / rest, over all runs: median %.2fx  max %.2fx'
          % (float(np.median(allramp)), float(np.max(allramp))))
    print()

    # what does the 5-minute promise actually buy? Express the budget as the fraction of each
    # run that can be integrated inside BUDGET_S at the run's own measured rate.
    print('the 5-minute budget, per run (fraction of the run that fits in %.0f s):' % BUDGET_S)
    fr = [(r['case'], r['model_h'], BUDGET_S / (r['actual'] / r['model_h']), r['actual'])
          for r in rows]
    fr.sort(key=lambda t: -t[1])
    for name, mh, budget_h, act in fr[:6]:
        print('  %-26s %5.1f model-h, wall %6.0f s -> %.2f model-h affordable (%.1f%% of run)'
              % (name, mh, act, budget_h, 100 * budget_h / mh))
    med = float(np.median([t[2] / t[1] for t in fr]))
    print('  median affordable fraction: %.1f%%  (k anchors of %.0f%% each need k*%.0f%%)'
          % (100 * med, 100 * ANCHOR_FRAC, 100 * ANCHOR_FRAC))
    return 0


if __name__ == '__main__':
    sys.exit(main())

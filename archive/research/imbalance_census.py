#!/usr/bin/env python3
"""Which cases could rebalancing ever help? Two axes, both measured, both required.

The hypothesis under test: the transition controller never paid off because CONUS is unusually
uniform, and a genuinely imbalanced case would show an effect.

That is right in spirit but incomplete, because this repo has already measured two things that
constrain it:

  * wall time is SUM over phases of the per-phase MAX, not the max of each rank's total
    (wrf-real-bottleneck-synchronization-granularity). A rank that is hot in radiation and
    cold in microphysics contributes to neither bound.
  * each phase is hot in a DIFFERENT region, so no single decomposition balances them all
    (wrf-mpi-decomposition-loadbalance-why-all-fail, reason 2).

So "imbalanced" is not sufficient. The imbalance also has to be PERSISTENT -- if the hot set
turns over every few minutes, no static reallocation at a restart boundary can chase it.

    axis A  MAGNITUDE    max/mean of per-rank compute, and the recoverable seconds
    axis B  PERSISTENCE  tau, the decay constant of the per-rank pattern, measured from
                         consecutive windows of the run itself

Only the top-right corner -- large AND persistent -- can pay. A case that is large but
short-tau is the case the controller would chase and lose to; that is a different failure from
a case that is simply flat, and it needs a different answer.

usage: imbalance_census.py [case ...]      (default: every case with a bucket_load.csv)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from decay_gate import load_bucket_csv, windows_from_buckets, measure_decay, DEFAULT_WINDOW_S

RESULTS = '/home/u5727520/wrf-results'
SWITCH_COST_S = 11.6      # measured ADIOS2 restart: read 10.69 + write 0.90


def census(case):
    p = os.path.join(RESULTS, case, 'bucket_load.csv')
    if not os.path.exists(p):
        return None
    try:
        comp, bsec = load_bucket_csv(p)
    except Exception:
        return None
    used = comp.sum(axis=0) > 0
    if used.sum() < 3:
        return None
    last = int(np.where(used)[0].max())
    core = comp[:, 1:last] if last > 1 else comp
    tot = core.sum(axis=1)
    if tot.sum() <= 0:
        return None
    mx, mean = float(tot.max()), float(tot.mean())
    row = {'case': case, 'ranks': core.shape[0], 'bsec': bsec,
           'wall_s': float(core.sum(axis=0).max() if core.shape[1] else 0) or None,
           'span_s': core.shape[1] * bsec,
           'ratio': mx / mean if mean else float('nan'),
           'recoverable_s': mx - mean,
           'p99_over_median': float(np.percentile(tot, 99) / max(np.median(tot), 1e-9)),
           'tau_s': None, 'hit1': None}
    # tau needs fine buckets: a 300 s bucket cannot resolve a pattern that turns over in 300 s
    if bsec <= 60:
        w, width = windows_from_buckets(comp, bsec, DEFAULT_WINDOW_S)
        if w.shape[1] >= 3:
            d = measure_decay(w, width)
            row['tau_s'] = d['tau_s']
            row['hit1'] = d['hit'][0] if d['hit'] else None
    return row


def main():
    cases = sys.argv[1:] or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(RESULTS, '*', 'bucket_load.csv')))
    rows = [r for r in (census(c) for c in cases) if r]
    if not rows:
        print('no usable cases'); return 2

    rows.sort(key=lambda r: -r['recoverable_s'])
    print('MAGNITUDE — how much is there to recover?   (switch costs %.1f s)' % SWITCH_COST_S)
    print('%-28s %6s %8s %10s %12s %10s' %
          ('case', 'ranks', 'span_s', 'max/mean', 'recoverable', 'pays?'))
    for r in rows:
        pays = 'yes' if r['recoverable_s'] > SWITCH_COST_S else 'no'
        print('%-28s %6d %8.0f %10.3f %11.1f s %10s' %
              (r['case'], r['ranks'], r['span_s'], r['ratio'], r['recoverable_s'], pays))

    fine = [r for r in rows if r['tau_s']]
    if fine:
        print()
        print('PERSISTENCE — does the hot set stay put?   (fine-bucket cases only)')
        print('%-28s %10s %10s %12s   %s' %
              ('case', 'max/mean', 'tau_s', 'hit@40s', 'verdict'))
        for r in sorted(fine, key=lambda x: -(x['tau_s'] or 0)):
            big = r['recoverable_s'] > SWITCH_COST_S
            persistent = r['tau_s'] > 1800
            if big and persistent:
                v = 'CANDIDATE — large and persistent'
            elif big:
                v = 'chases: large but turns over (tau %.0f s)' % r['tau_s']
            elif persistent:
                v = 'persistent but too small to pay'
            else:
                v = 'flat and fleeting — nothing to do'
            print('%-28s %10.3f %10.0f %11.1f%%   %s' %
                  (r['case'], r['ratio'], r['tau_s'], r['hit1'] or -1, v))
    print()
    print('READING')
    print('  Only large AND persistent can pay. Large-but-short-tau is the case the controller')
    print('  would chase and lose to -- a different failure from a case that is simply flat,')
    print('  and it needs a different answer.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Is the cost-drift trajectory a property of the CASE or of the CONFIGURATION?

This decides whether a cheap coarse-grid twin can supply the drift at all, and it is free to
answer from runs already on disk.

    drift_j = log(rate_j / rate_0)   -- how a segment's cost compares to the run's first one

Same case, different mp/bl scheme = same weather trajectory, different fine-scale state and
different per-step work. So:

  * within-case spread SMALL  -> drift is set by the large-scale weather evolution. A 12 km
    twin follows that same evolution, so ITS OWN cost trajectory has the same shape, and the
    3 km anchor only needs scaling by it. No ndown, no mid-run restart, no second anchor.

  * within-case spread LARGE  -> drift is set by things the configuration changes, i.e. by
    fine-scale detail a coarse twin does not have. Then only a faithful state helps, and
    ndown from 12 km is unlikely to deliver one.

The comparison is within-case spread against between-case spread. If they are the same size,
knowing the case tells you nothing about the drift and the whole surrogate idea is dead
regardless of which mechanism is responsible.
"""
import os
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
from residual_drift import build, RESULTS

runs = []
for d in sorted(os.listdir(RESULTS)):
    if not os.path.isdir(os.path.join(RESULTS, d)):
        continue
    b = build(d)
    if b:
        segs, arate, span = b
        r0 = segs[0][1]
        drift = np.array([np.log(s[1] / r0) for s in segs])
        base = d.split('__')[0]                      # strip the __mpXblY variant suffix
        runs.append({'run': d, 'case': base, 'drift': drift, 'n': len(drift)})

by_case = {}
for r in runs:
    by_case.setdefault(r['case'], []).append(r)
multi = {k: v for k, v in by_case.items() if len(v) >= 3}

print('%d runs, %d distinct cases, %d cases with >=3 variants\n'
      % (len(runs), len(by_case), len(multi)))

if not multi:
    print('no case has >=3 variants with usable profiles')
    sys.exit(1)

# compare like with like: only segments both members of a pair actually have
def rms_diff(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return None
    return float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))


within, between = [], []
names = sorted(multi)
for c in names:
    v = multi[c]
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            d = rms_diff(v[i]['drift'], v[j]['drift'])
            if d is not None:
                within.append(d)
for a in range(len(names)):
    for b in range(a + 1, len(names)):
        for ra in multi[names[a]]:
            for rb in multi[names[b]]:
                d = rms_diff(ra['drift'], rb['drift'])
                if d is not None:
                    between.append(d)

within, between = np.array(within), np.array(between)
print('RMS difference between two drift trajectories, in log-rate:')
print('  SAME case, different mp/bl  : mean %.4f  median %.4f  (n=%d pairs)'
      % (within.mean(), np.median(within), len(within)))
print('  DIFFERENT cases             : mean %.4f  median %.4f  (n=%d pairs)'
      % (between.mean(), np.median(between), len(between)))
print('  ratio between/within        : %.2fx' % (between.mean() / max(within.mean(), 1e-9)))
print()
print('  for scale: sd of the drift itself is %.4f'
      % float(np.std(np.concatenate([r['drift'] for r in runs]))))
print()
print('per-case drift trajectories (log rate vs segment 0):')
for c in names:
    v = multi[c]
    n = min(r['n'] for r in v)
    M = np.array([r['drift'][:n] for r in v])
    print('  %-18s n=%d variants  mean %s' % (c, len(v), np.round(M.mean(0), 3)))
    print('  %-18s              sd   %s' % ('', np.round(M.std(0), 3)))

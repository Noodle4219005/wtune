#!/usr/bin/env python3
"""How many cases per weather regime does the model actually need?

Every measurement so far says regime COVERAGE dominates everything else:
  9 same-weather training cases -> 80.8% hit rate
  1 same-weather training case  -> 39.4%
  1, and a different decomposition -> 21.7%   (2x random)
That is a bigger effect than the per-bl head (+12pt), the geometry features (+4pt), the loss
function (+2pt) or the model class (KNN, -7pt) -- and it is the one thing whose cost is not
yet known, because nobody has measured where it saturates.

karina has 9 same-weather variants, so the curve can be drawn for free: hold one out, train
on k of the remaining 8, sweep k. If it flattens at 3 the corpus can be small and cheap; if
it climbs all the way to 8, the data route is expensive and the architecture needs rethinking
instead. Either answer converts the corpus budget from a guess into arithmetic.

usage: saturation.py <case>...
"""
import sys, os, re, itertools
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg, wtune_oracle_train as tr

TOPK, FIT, SEG_HOURS = 45, dict(depth=3, n_est=80, lr=0.10), 3.0
RNG = np.random.default_rng(0)
NDRAW = 6                       # random subsets per k, so a lucky pick cannot set the shape


def hit(score, true, sgv):
    o = []
    for k in np.unique(sgv):
        ix = np.where(sgv == k)[0]
        if len(ix) < TOPK:
            continue
        a = set(ix[np.argsort(true[ix])[-TOPK:]]); b = set(ix[np.argsort(score[ix])[-TOPK:]])
        o.append(100.0 * len(a & b) / TOPK)
    return float(np.mean(o)) if o else np.nan


rows = []
for c in sys.argv[1:]:
    r, why = seg.build_case(c)
    if r is None:
        continue
    X, y, cs, sg, names = r
    keep = [i for i, q in enumerate(names) if not q.startswith('geo_')]   # geometry is not portable
    nl = f'{seg.RESULTS}/{c}/namelist.input'
    t = open(nl).read() if os.path.exists(nl) else ''
    m = re.search(r'time_step=(\d+)', t)
    dt = int(m.group(1)) if m else 15
    rate = cs / (SEG_HOURS * 3600.0 / dt)
    smean = np.empty_like(rate)
    for k in np.unique(sg):
        smean[sg == k] = rate[sg == k].mean()
    rows.append(dict(case=c, X=X[:, keep], rate=rate, shape=rate / smean, seg=sg))
n = len(rows)
print("=== 飽和曲線：同 regime 訓練樣本數 vs 熱點命中率（已排除幾何特徵）===")
print("%4s %10s %10s %s" % ("k", "命中率", "標準差", "(每個 k 取 %d 組隨機子集 x %d 個留出)" % (NDRAW, n)))
prev = None
for k in range(1, n):
    vals = []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        seen = set()
        for _ in range(NDRAW):
            pick = tuple(sorted(RNG.choice(others, size=k, replace=False)))
            if pick in seen:
                continue
            seen.add(pick)
            Xtr = np.vstack([rows[j]['X'] for j in pick])
            ytr = np.concatenate([rows[j]['shape'] for j in pick])
            p = tr.gbrt_pred(tr.gbrt_fit(Xtr, ytr, **FIT), rows[i]['X'])
            vals.append(hit(p, rows[i]['rate'], rows[i]['seg']))
    mu = float(np.mean(vals))
    print("%4d %9.1f%% %9.1f%%  %s" % (k, mu, float(np.std(vals)),
          "" if prev is None else ("%+.1f" % (mu - prev))))
    prev = mu

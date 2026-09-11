#!/usr/bin/env python3
"""Is cost(case, config) separable, and does imbalance rank configs the way cost does?

Two questions, both of which decide what the cost model can be:

1. SEPARABILITY. If cost(case, mp, bl) ~ base(case) * f(mp) * g(bl), then a handful of runs
   pins every combination, including ones nobody has run -- which is the entire point of a
   model here. If the interaction term is large, the model needs the full cross product and
   is far less useful. Fit in log space so the multiplicative form is linear, and report the
   residual against a model that has ONLY main effects.

2. DOES IMBALANCE RANK CONFIGS? WTune's existing target is a per-rank imbalance statistic.
   Imbalance is a RATIO (max/mean) and cost is an ABSOLUTE quantity, so a scheme that makes
   every rank uniformly slower looks BETTER balanced and IS more expensive. If that happens
   in the data, the current objective cannot be used to choose a config, which is the whole
   argument for re-pointing the model at cost.

Runs are grouped by (case, io_form_history, io_form_restart, history_interval): netCDF and
ADIOS2 wall clocks are not comparable, and mixing them once produced a confident conclusion
that a controlled sweep reversed.
"""
import csv
import itertools
import math
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else '/home/u5727520/wrf-results/derived-data/config-cost.csv'
IOKEY = ('io_form_history', 'io_form_restart', 'history_interval')


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):                      # average ties, or a flat column lies
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float('nan')


rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if not r['variant'].startswith('mp') or not r['s_per_model_h']:
            continue
        r['cost'] = float(r['s_per_model_h'])
        r['M'] = float(r['imbalance_M_pct']) if r['imbalance_M_pct'] else None
        r['mp'] = int(r['mp_physics'])
        r['bl'] = int(r['bl_pbl_physics'])
        r['grp'] = (r['case'],) + tuple(r[k] for k in IOKEY)
        rows.append(r)

groups = {}
for r in rows:
    groups.setdefault(r['grp'], []).append(r)
groups = {k: v for k, v in groups.items() if len(v) >= 4}

print('=== 2. does imbalance rank configs the way cost does? ===')
print('%-18s %5s  %8s   %s' % ('case', 'n', 'rho', 'counterexample (more balanced AND more expensive)'))
for k in sorted(groups):
    g = [r for r in groups[k] if r['M'] is not None]
    if len(g) < 4:
        continue
    rho = spearman([r['M'] for r in g], [r['cost'] for r in g])
    ce = ''
    for a, b in itertools.combinations(g, 2):
        lo, hi = (a, b) if a['cost'] < b['cost'] else (b, a)
        if hi['M'] < lo['M']:                       # costs more, yet looks better balanced
            ce = 'mp%dbl%d M=%.1f cost=%.0f  vs  mp%dbl%d M=%.1f cost=%.0f (+%.0f%%)' % (
                hi['mp'], hi['bl'], hi['M'], hi['cost'],
                lo['mp'], lo['bl'], lo['M'], lo['cost'], 100 * (hi['cost'] / lo['cost'] - 1))
            break
    print('%-18s %5d  %+8.3f   %s' % (k[0], len(g), rho, ce or '-'))

print()
print('=== 1. separability: cost ~ base(case) * f(mp) * g(bl), fitted in log space ===')
# alternating least squares on log cost. Main effects only -- no interaction term. What the
# residual comes to IS the interaction, which is the number that decides the model's shape.
lc = {id(r): math.log(r['cost']) for g in groups.values() for r in g}
allr = [r for g in groups.values() for r in g]
base = {k: 0.0 for k in groups}
fmp = {m: 0.0 for m in {r['mp'] for r in allr}}
gbl = {b: 0.0 for b in {r['bl'] for r in allr}}
for _ in range(200):
    for d, key in ((base, 'grp'), (fmp, 'mp'), (gbl, 'bl')):
        for k in d:
            sel = [r for r in allr if r[key] == k]
            if sel:
                d[k] = sum(lc[id(r)] - base[r['grp']] - fmp[r['mp']] - gbl[r['bl']]
                           for r in sel) / len(sel) + d[k]
    m = sum(fmp.values()) / len(fmp)               # absorb the gauge freedom into base
    for k in fmp:
        fmp[k] -= m
    for k in base:
        base[k] += m
    m = sum(gbl.values()) / len(gbl)
    for k in gbl:
        gbl[k] -= m
    for k in base:
        base[k] += m

res = [lc[id(r)] - base[r['grp']] - fmp[r['mp']] - gbl[r['bl']] for r in allr]
rms = math.sqrt(sum(x * x for x in res) / len(res))
spread = max(lc.values()) - min(lc.values())
print('  n = %d runs in %d comparable groups' % (len(allr), len(groups)))
print('  multiplicative main effects (relative cost, 1.00 = group mean):')
print('    mp_physics: ' + '  '.join('%d=%.3f' % (k, math.exp(v)) for k, v in sorted(fmp.items())))
print('    bl_pbl    : ' + '  '.join('%d=%.3f' % (k, math.exp(v)) for k, v in sorted(gbl.items())))
print('  residual RMS = %.4f in log space = %.2f%% in cost' % (rms, 100 * (math.exp(rms) - 1)))
print('  total spread = %.2fx, so main effects explain the bulk iff the residual is small'
      % math.exp(spread))
worst = sorted(zip(res, allr), key=lambda t: -abs(t[0]))[:4]
print('  worst residuals (where an interaction term would be needed):')
for v, r in worst:
    print('    %-16s mp%d bl%d  actual %.1f  model %.1f  (%+.1f%%)'
          % (r['case'], r['mp'], r['bl'], r['cost'], r['cost'] / math.exp(v), 100 * (math.exp(v) - 1)))

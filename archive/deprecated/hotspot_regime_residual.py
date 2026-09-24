#!/usr/bin/env python3
"""Per-regime heads and the residual form, against the control that decides what they mean.

THE CONTROL COMES FIRST

Every predicted hot-spot map in the regime figure looks more like the OTHER predictions than
like its own target, which says the model is largely replaying a shared positional prior. So
the first arm here is that prior ALONE: base[rank] = the mean normalised cost at each rank
position over the training cases, with no weather input whatsoever. Whatever it scores is the
part of the current 56.0% that owes nothing to the features -- and any arm that fails to beat
it is not doing what it claims.

THE RESIDUAL FORM

    cost[rank] = base[rank] + G(weather)
                 \________/   \________/
                 recomputable   learned

Subtracting the prior explicitly forces the model to spend its splits on the weather-driven
deviation instead of rediscovering the prior. Decision 0032 measured this at 46.9% against
51.9% for feeding geometry as plain columns -- worse, but with the virtue of moving the
decomposition dependence into a table. It is worth re-testing now because the features have
changed (union, +6.6 points) and because the flatness diagnosis points straight at it.

PER-REGIME HEADS

The prior art (2026-09-05-mynn-needs-its-own-head) got 57.6% -> 75.7% from a per-SCHEME head
on a third of the data. Scheme is the right partition to try because it is known from the
namelist at inference time, unlike weather type. But the corpus is thin: splitting by
bl_pbl_physics leaves 5 / 3 / 3 weathers per head, so leave-one-weather-out trains a head on
as few as TWO weathers. Scheme-as-a-feature is carried alongside as the data-efficient
version of the same idea -- the tree can split on it when it helps without partitioning.
"""
import os
import re
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
from stencil_scale import base_of

PX, PY = 8, 56
NR = PX * PY
TOPK = max(1, int(round(0.10 * NR)))
CASES = ['karina_48h', 'karina_cat4', 'deep_conv_ot', 'arctic_sbl', 'marine_cao', 'polar_low',
         'extratrop_cyc', 'edouard_landfall', 'atmos_river', 'dry_conv', 'lowell_cat4',
         'marie_ts', 'decay_karina_cat4', 'decay_arctic_sbl', 'decay_deep_conv_ot']
LGBM = dict(num_leaves=31, min_child_samples=40, learning_rate=0.06, n_estimators=300,
            colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
            n_jobs=8, random_state=0, verbose=-1)
KNOBS = ('mp_physics', 'bl_pbl_physics', 'ra_lw_physics')


def weather_of(case):
    b = base_of(case)
    b = re.sub(r'^(decay|probe)_', '', b)
    b = re.sub(r'_\d+h$', '', b)
    b = re.sub(r'_(cat\d|\d+km)$', '', b)
    return b


def knobs_of(case):
    p = os.path.join(seg.RESULTS, case, 'namelist.input')
    txt = open(p).read() if os.path.exists(p) else ''
    out = []
    for k in KNOBS:
        m = re.search(r'\b%s\s*=\s*(\d+)' % k, txt)
        out.append(float(m.group(1)) if m else np.nan)
    return out


def norm(cs, sg):
    out = np.empty_like(cs, dtype=float)
    for g in np.unique(sg):
        m = sg == g
        out[m] = cs[m] / cs[m].mean()
    return out


data = {}
for c in CASES:
    r, why = seg.build_case(c)
    if r is None:
        continue
    X, y, cs, sg, names = r
    data[c] = dict(case=c, X=X, cs=cs, sg=sg, names=list(names), w=weather_of(c),
                   k=knobs_of(c), y=norm(cs, sg))
UNION = sorted(set.union(*(set(d['names']) for d in data.values())))
print('%d cases, %d weathers, %d union features' % (
    len(data), len({d['w'] for d in data.values()}), len(UNION)))
for b in sorted({d['k'][1] for d in data.values()}):
    ws = {d['w'] for d in data.values() if d['k'][1] == b}
    print('  bl_pbl=%.0f : %d cases, %d weathers' % (
        b, sum(1 for d in data.values() if d['k'][1] == b), len(ws)))


def mat(d, with_knobs):
    out = np.full((d['X'].shape[0], len(UNION)), np.nan)
    idx = {n: i for i, n in enumerate(d['names'])}
    for j, c in enumerate(UNION):
        if c in idx:
            out[:, j] = d['X'][:, idx[c]]
    if with_knobs:
        out = np.hstack([out, np.tile(d['k'], (out.shape[0], 1))])
    return out


def prior_from(train):
    """base[rank]: mean normalised cost at each rank POSITION over the training cases.
    Recomputable, weather-free, and the thing the model may merely be replaying."""
    acc = np.zeros(NR)
    n = 0
    for d in train:
        for g in np.unique(d['sg']):
            m = d['sg'] == g
            if m.sum() == NR:
                acc += d['y'][m]
                n += 1
    return acc / max(n, 1)


def hits(pred, d):
    out = []
    for g in np.unique(d['sg']):
        m = d['sg'] == g
        if m.sum() != NR:
            continue
        a = set(np.argsort(d['cs'][m])[-TOPK:])
        b = set(np.argsort(pred[m])[-TOPK:])
        out.append(100.0 * len(a & b) / TOPK)
    return out


def run(arm, train, te):
    from lightgbm import LGBMRegressor
    base = prior_from(train)
    if arm == 'prior':
        return np.tile(base, te['X'].shape[0] // NR)
    wk = arm.endswith('+knobs')
    if arm.startswith('head'):
        train = [d for d in train if d['k'][1] == te['k'][1]] or train
    Xtr = np.vstack([mat(d, wk) for d in train])
    Xte = mat(te, wk)
    if arm.startswith('resid'):
        ytr = np.concatenate([d['y'][:] - np.tile(base, d['X'].shape[0] // NR)
                              for d in train])
        m = LGBMRegressor(**LGBM).fit(Xtr, ytr)
        return np.tile(base, te['X'].shape[0] // NR) + np.asarray(m.predict(Xte))
    ytr = np.concatenate([d['y'] for d in train])
    m = LGBMRegressor(**LGBM).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte))


ARMS = ['prior', 'plain', 'plain+knobs', 'head', 'head+knobs', 'resid', 'resid+knobs']
LABEL = {'prior': 'positional prior ONLY, no weather',
         'plain': 'LightGBM + union features  [current best]',
         'plain+knobs': '  + scheme numbers as features',
         'head': 'per-scheme head (split by bl_pbl_physics)',
         'head+knobs': '  per-scheme head + scheme features',
         'resid': 'residual: base[rank] + G(weather)',
         'resid+knobs': '  residual + scheme features'}

res = {a: {} for a in ARMS}
for w in sorted({d['w'] for d in data.values()}):
    train = [d for d in data.values() if d['w'] != w]
    for te in [d for d in data.values() if d['w'] == w]:
        for a in ARMS:
            h = hits(run(a, train, te), te)
            if h:
                res[a][te['case']] = float(np.mean(h))

print()
print('%-44s %7s %7s %7s' % ('arm', 'mean', 'median', 'worst'))
for a in ARMS:
    v = np.array(list(res[a].values()))
    print('%-44s %6.1f%% %6.1f%% %6.1f%%'
          % (LABEL[a], v.mean(), np.median(v), v.min()))

print('\nper case:')
cs_ = sorted(res['plain'])
print('%-20s %s' % ('case', ' '.join('%7s' % a[:7] for a in ARMS)))
for c in cs_:
    print('%-20s %s' % (c, ' '.join('%6.1f%%' % res[a][c] if c in res[a] else '     - '
                                    for a in ARMS)))

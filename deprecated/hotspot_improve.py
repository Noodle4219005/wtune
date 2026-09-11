#!/usr/bin/env python3
"""Four ways to make the hot-spot model better, on identical folds.

WHY THERE IS ROOM

Two independent runs of the same case agree on the top 10% of ranks at 73-96%
(hotspot_truth.py). The model scores 29-67%. The measurement is not the ceiling.

WHAT THE DIAGNOSIS SAYS TO TRY

  union    Cross-case training intersects feature NAMES, and polar_low and marie_ts carry
           only 11 raw columns -- no PBLH, HFX, LH, UST, WMAX, PBLFRAC. The intersection
           therefore strips the entire surface/boundary-layer block from EVERY case. The
           worst regime in the corpus is the cold-air outbreak at 29%, which is driven by
           surface heat flux over warm water, and the model cannot see HFX. Raw frames are
           deleted after featurisation so the two thin cases cannot be rebuilt -- but
           LightGBM learns a default direction per split for missing values, so the union
           can be used directly with NaN.

  lgbm     The hand-rolled GBRT has no regularisation of any kind. Isolated here on the SAME
           20 columns so the learner is separated from the features.

  rank     The metric is set membership in the top 10%, which is ordinal, but training
           minimises squared error on a value. That is visible in the maps: every predicted
           field comes out FLATTER than its target, which is what squared loss does. Train
           lambdarank on binary relevance (in the true top 10% or not), one group per
           (case, segment), optimising NDCG at 45.

  both     union + rank together.

The fold is leave-one-WEATHER-out: base_of() folds decay_/probe_/_14x64/__mpXblY variants
onto their parent, and weather_of() additionally folds karina_48h onto karina, which base_of
misses because that run was prepared from its own met_em tree. Skipping the second step
inflates the hurricane by 11 points.
"""
import re
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr
from stencil_scale import base_of

PX, PY = 8, 56
NR = PX * PY
TOPFRAC = 0.10
TOPK = max(1, int(round(TOPFRAC * NR)))
GBRT = dict(depth=3, n_est=80, lr=0.10)
CASES = ['karina_48h', 'karina_cat4', 'deep_conv_ot', 'arctic_sbl', 'marine_cao', 'polar_low',
         'extratrop_cyc', 'edouard_landfall', 'atmos_river', 'dry_conv', 'lowell_cat4',
         'marie_ts', 'decay_karina_cat4', 'decay_arctic_sbl', 'decay_deep_conv_ot']


def weather_of(case):
    b = base_of(case)
    b = re.sub(r'^(decay|probe)_', '', b)
    b = re.sub(r'_\d+h$', '', b)
    b = re.sub(r'_(cat\d|\d+km)$', '', b)
    return b


def shape_target(cs, sg):
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
    data[c] = dict(case=c, X=X, cs=cs, sg=sg, names=list(names), w=weather_of(c))
print('%d cases, %d weathers' % (len(data), len({d['w'] for d in data.values()})))

INTER = sorted(set.intersection(*(set(d['names']) for d in data.values())))
UNION = sorted(set.union(*(set(d['names']) for d in data.values())))
print('intersection %d cols, union %d cols' % (len(INTER), len(UNION)))


def matrix(d, cols):
    """Case features re-indexed onto `cols`; absent columns become NaN, which LightGBM
    handles natively by learning a default branch. The hand-rolled GBRT cannot, so it only
    ever runs on the intersection."""
    out = np.full((d['X'].shape[0], len(cols)), np.nan)
    idx = {n: i for i, n in enumerate(d['names'])}
    for j, c in enumerate(cols):
        if c in idx:
            out[:, j] = d['X'][:, idx[c]]
    return out


def hit_of(pred, d):
    hits = []
    for g in np.unique(d['sg']):
        m = d['sg'] == g
        if m.sum() != NR:
            continue
        t, p = d['cs'][m], pred[m]
        a, b = set(np.argsort(t)[-TOPK:]), set(np.argsort(p)[-TOPK:])
        hits.append(100.0 * len(a & b) / TOPK)
    return hits


def fit_predict(arm, cols, train, test):
    Xtr = np.vstack([matrix(d, cols) for d in train])
    Xte = matrix(test, cols)
    if arm == 'gbrt':
        y = np.concatenate([shape_target(d['cs'], d['sg']) for d in train])
        m = tr.gbrt_fit(Xtr, y, **GBRT)
        return tr.gbrt_pred(m, Xte)
    import lightgbm as lgb
    if arm == 'value':
        y = np.concatenate([shape_target(d['cs'], d['sg']) for d in train])
        m = lgb.LGBMRegressor(num_leaves=31, min_child_samples=40, learning_rate=0.06,
                              n_estimators=300, colsample_bytree=0.8, subsample=0.8,
                              subsample_freq=1, reg_lambda=1.0, n_jobs=8, random_state=0,
                              verbose=-1).fit(Xtr, y)
        return np.asarray(m.predict(Xte))
    # lambdarank: one group per (case, segment), relevance 1 for the true top 10%
    ys, groups = [], []
    for d in train:
        for g in np.unique(d['sg']):
            mm = d['sg'] == g
            if mm.sum() != NR:
                continue
            t = d['cs'][mm]
            lab = np.zeros(NR, dtype=int)
            lab[np.argsort(t)[-TOPK:]] = 1
            ys.append(lab)
            groups.append(NR)
    keep = np.concatenate([np.where(d['sg'] == g)[0] + off
                           for off, d in zip(np.cumsum([0] + [x['X'].shape[0]
                                                              for x in train])[:-1], train)
                           for g in np.unique(d['sg']) if (d['sg'] == g).sum() == NR])
    m = lgb.LGBMRanker(objective='lambdarank', num_leaves=31, min_child_samples=40,
                       learning_rate=0.06, n_estimators=300, colsample_bytree=0.8,
                       subsample=0.8, subsample_freq=1, reg_lambda=1.0, n_jobs=8,
                       random_state=0, verbose=-1, label_gain=[0, 1],
                       eval_at=[TOPK]).fit(Xtr[keep], np.concatenate(ys), group=groups)
    return np.asarray(m.predict(Xte))


ARMS = [('baseline: hand-rolled GBRT, intersected features', 'gbrt', INTER),
        ('LightGBM, same intersected features', 'value', INTER),
        ('LightGBM + UNION features (NaN for absent)', 'value', UNION),
        ('LightGBM ranker, intersected features', 'rank', INTER),
        ('LightGBM ranker + UNION features', 'rank', UNION)]

res = {}
for label, arm, cols in ARMS:
    per = {}
    for w in sorted({d['w'] for d in data.values()}):
        test_cases = [d for d in data.values() if d['w'] == w]
        train = [d for d in data.values() if d['w'] != w]
        if len(train) < 5:
            continue
        for te in test_cases:
            h = hit_of(fit_predict(arm, cols, train, te), te)
            if h:
                per.setdefault(te['case'], []).append(float(np.mean(h)))
    flat = [np.mean(v) for v in per.values()]
    res[label] = per
    print('%-46s mean %5.1f%%  median %5.1f%%  worst %5.1f%%  (%d cases)'
          % (label, float(np.mean(flat)), float(np.median(flat)), float(np.min(flat)),
             len(flat)))

print('\nper case (mean over its segments):')
names = [l for l, _, _ in ARMS]
allc = sorted(res[names[0]])
print('%-20s %s' % ('case', ' '.join('%9s' % n.split(',')[0][:9] for n in names)))
for c in allc:
    print('%-20s %s' % (c, ' '.join('%8.1f%%' % np.mean(res[n][c]) if c in res[n] else '     -  '
                                    for n in names)))

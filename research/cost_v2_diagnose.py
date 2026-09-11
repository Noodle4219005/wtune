#!/usr/bin/env python3
"""Why does B(weather) fail to replace base(case)? Sample count, or the wrong features?

cost_model_v2.py transplanted the hot-spot architecture and got WORSE with weather than
without: 38.4% MAPE against 28.4% for structure alone. Two explanations, and they call for
opposite fixes, so guessing is not acceptable:

  (A) not enough cases. The hot-spot model has ~448 samples per case -- one per rank -- so
      26 features sit on ~16k rows. The cost model has ONE sample per case: 30 features on
      27 rows. If that is the binding constraint, the learning curve is still falling
      steeply at 27 cases and the fix is more cases.

  (B) wrong features. Domain-mean weather may simply not carry the information; the fix is
      then different features, and more cases would buy nothing.

The discriminator is the learning curve. Train on k cases, test on held-out cases, sweep k.
Still-falling at the right edge => (A). Flat => (B).

A second axis runs alongside it: the corpus is not homogeneous. It mixes 2 h floor probes,
6 h probe runs, a nested 1 km case and a 48 h run with the 12 h production runs. Those differ
in kind, not in weather, so they are reported separately -- a model scored on a corpus whose
spread it was never meant to explain is being blamed for the wrong thing.
"""
import csv
import math
import os
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle_train as tr
from cost_model_v2 import case_features, RESULTS, CSV, fit_predict

STRUCT = ['e_vert', 'e_we', 'e_sn', 'dx', 'n_ranks', 'time_step', 'model_h',
          'io_form_history', 'numtiles', 'nproc_x']
# runs that differ in KIND, not in weather: synthetic decomposition floors, short probes,
# and the one nested case. Keeping them makes the target span 35x and blames the weather
# features for structure they were never given.
ODD = ('floor_', 'probe_', 'decay_nest')


def load(homogeneous):
    out = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if not r['s_per_model_h']:
                continue
            if homogeneous:
                if any(r['run'].startswith(p) for p in ODD):
                    continue
                if r.get('model_h') != '12.0':
                    continue
            f, fn = case_features(os.path.join(RESULTS, r['run']))
            if f is None:
                continue
            try:
                sv = [float(r[k]) for k in STRUCT]
                mp, bl = int(r['mp_physics']), int(r['bl_pbl_physics'])
            except (KeyError, ValueError, TypeError):
                continue
            out.append({'case': r['case'], 'mp': mp, 'bl': bl,
                        'y': math.log(float(r['s_per_model_h'])),
                        'x': np.array(sv + f, float)})
    return out, len(STRUCT)


def main_effects(train):
    fmp = {d['mp']: 0.0 for d in train}
    gbl = {d['bl']: 0.0 for d in train}
    off = {d['case']: 0.0 for d in train}
    for _ in range(200):
        for d, key in ((off, 'case'), (fmp, 'mp'), (gbl, 'bl')):
            for k in d:
                sel = [t for t in train if t[key] == k]
                if sel:
                    d[k] += sum(t['y'] - off[t['case']] - fmp[t['mp']] - gbl[t['bl']]
                                for t in sel) / len(sel)
        for d in (fmp, gbl):
            m = sum(d.values()) / len(d)
            for k in d:
                d[k] -= m
            for k in off:
                off[k] += m
    return fmp, gbl


def score(data, cols, ncase=None, seed=0, learner='lgbm'):
    """Leave-one-case-out MAPE. ncase caps how many TRAINING cases are used, so sweeping it
    traces the learning curve; the test case is always fully held out.

    Defaults to LightGBM: on identical folds and features it beat the hand-rolled learner on
    every arm by ~20% relative (23.2% vs 28.4% structure-only), so a learning curve traced
    with the unregularised one measures the learner's weakness as much as the corpus's.
    """
    cases = sorted({d['case'] for d in data})
    rng = np.random.default_rng(seed)
    errs = []
    for c in cases:
        pool = [x for x in cases if x != c]
        if ncase is not None and ncase < len(pool):
            pool = list(rng.choice(pool, ncase, replace=False))
        train = [d for d in data if d['case'] in pool]
        test = [d for d in data if d['case'] == c]
        if len(train) < 4:
            continue
        fmp, gbl = main_effects(train)
        res = np.array([t['y'] - fmp.get(t['mp'], 0.0) - gbl.get(t['bl'], 0.0) for t in train])
        p, _ = fit_predict(np.array([t['x'][cols] for t in train]), res,
                           np.array([t['x'][cols] for t in test]), learner)
        for b, t in zip(p, test):
            pred = math.exp(b + fmp.get(t['mp'], 0.0) + gbl.get(t['bl'], 0.0))
            errs.append(abs(pred / math.exp(t['y']) - 1) * 100)
    return (float(np.mean(errs)), float(np.median(errs)), len(errs)) if errs else (float('nan'),) * 3


for homogeneous in (False, True):
    data, ns = load(homogeneous)
    if len(data) < 8:
        print('%s corpus too small (%d)' % ('homogeneous' if homogeneous else 'full', len(data)))
        continue
    cases = sorted({d['case'] for d in data})
    nfeat = len(data[0]['x'])
    print('=== %s corpus: %d runs, %d cases, %d features ==='
          % ('HOMOGENEOUS (12 h, no floor/probe/nest)' if homogeneous else 'FULL',
             len(data), len(cases), nfeat))
    ALL = np.arange(nfeat)
    ST = np.arange(ns)
    print('  learning curve (training cases -> LOOCO MAPE), averaged over 3 draws:')
    print('  %8s %14s %14s' % ('k cases', 'structure only', 'struct+weather'))
    for k in (5, 10, 15, 20, len(cases) - 1):
        if k > len(cases) - 1:
            continue
        a = np.mean([score(data, ST, k, s)[0] for s in range(3)])
        b = np.mean([score(data, ALL, k, s)[0] for s in range(3)])
        print('  %8d %13.1f%% %13.1f%%' % (k, a, b))
    print()

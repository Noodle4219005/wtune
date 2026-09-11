#!/usr/bin/env python3
"""Cost model v2: the hot-spot model's architecture, re-pointed at cost(case, config).

WHAT IS BEING TRANSPLANTED

The hot-spot model predicts per-rank cost as

    cost[rank] = base[rank] + G(17 weather fields of that rank's patch)      GBRT depth 3
                 \_______/   \_________________________________________/
                 recomputable lookup          learned residual

That split is the part worth keeping: the term that depends on a discrete structural choice
is a table you can recompute without a model, and the model only has to learn what the table
cannot express. Its consumer is gone -- every rebalancing actuator is closed -- so the same
shape is re-pointed at the objective that still has one:

    log cost = f(mp) + g(bl) + B(17 weather fields of the CASE + its grid shape)
               \___________/   \_______________________________________________/
               recomputable      learned residual, GBRT depth 3
               config lookup

WHY THIS IS NOT COSMETIC

cost_separability.py fits base(case) as a FREE PARAMETER, one per case. That is why the
operational recipe needs one run of a new case to pin it: the model has nothing to say about
a case it has never seen. Replacing that free parameter with B(weather) is what removes the
run. The honest test is therefore leave-one-CASE-out with ZERO runs of the held-out case,
and it is a much harder bar than the 3.06% that one anchor run buys.

Both numbers are reported side by side, plus two baselines that must be beaten before the
weather features can be said to do anything:
  * corpus median   -- predict every case as the middle of the corpus
  * structure only  -- grid shape and run length, no weather at all

The hot-spot model's own history is the reason those baselines are here: three geometry
columns with no weather scored ~40% hit rate on their own, and for a long time that was
mistaken for the model working.
"""
import csv
import glob
import math
import os
import re
import sys

sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle_train as tr

RESULTS = '/home/u5727520/wrf-results'
CSV = '/home/u5727520/wrf-results/derived-data/config-cost.csv'
GBRT = dict(depth=3, n_est=80, lr=0.10, min_leaf=2)   # depth/n_est/lr as the hot-spot model;
                                                      # min_leaf 10 -> 2, the corpus is cases
                                                      # not ranks and 10 would never split

# LightGBM was absent when the hot-spot model was written -- wtune_models.py:44 says "No sklearn
# in this env", and the hand-rolled gbrt_fit in wtune_oracle_train.py exists for that reason
# alone, not by design. Added to the pixi env on 2026-09-07 (lightgbm 4.7.0, scipy 1.17.1;
# numpy stayed at 2.4.6 and python at 3.11, so nothing already measured moves).
#
# It matters more here than it did there. The hand-rolled learner has NO regularisation of any
# kind: no L1/L2, no column or row subsampling, no minimum split gain, no early stopping. On
# ~16k rank-rows that is survivable. On 27 case-rows with 30 columns it is the textbook setup
# for fitting noise -- the first suspect for weather features making the model WORSE (38.4%
# against 28.4% for structure alone). Both learners run on identical folds and features so the
# question gets answered instead of assumed.
#
# n_jobs is pinned LOW on purpose. LightGBM defaults to one thread per core; on a 112-core
# node, spawning 112 threads to fit 27 rows is pure contention, and it hung a login-node probe
# outright. Tiny data wants few threads.
#
# sklearn-style parameter names throughout (colsample_bytree, subsample, reg_lambda ...) --
# mixing them with the native aliases (feature_fraction, bagging_fraction, lambda_l2) makes
# LightGBM warn and silently prefer one of the two.
LGBM = dict(num_leaves=4, min_child_samples=3, learning_rate=0.05, n_estimators=200,
            colsample_bytree=0.6, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
            n_jobs=4, random_state=0, verbose=-1)


def fit_predict(Xtr, ytr, Xte, learner='gbrt'):
    """-> (predictions on Xte, fitted model). learner in {'gbrt', 'lgbm'}."""
    if learner == 'lgbm':
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(**LGBM).fit(Xtr, ytr)
        return np.asarray(m.predict(Xte)), m
    m = tr.gbrt_fit(Xtr, ytr, **GBRT)
    return tr.gbrt_pred(m, Xte), m
SPREAD_OF = ('radiation', 'micro', 'surf')            # the three the hot-spot model gives
                                                      # neighbourhood columns to
FEATSTAMP = re.compile(r'features_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})\.csv$')

# The canonical 17, selected BY NAME. Frame schemas are not uniform across the corpus and not
# even within a run: karina_48h carries 21 columns (extra QKE fields) and at least one run
# mixes an 11-column frame with 17-column ones, so positional stacking silently misaligns
# fields or crashes. Indexing by name is the only safe read.
FIELDS = ('radiation', 'micro', 'surf', 'TCWV', 'LWP', 'RWP', 'IWP', 'SWP', 'GWP',
          'nspecies', 'mixed_phase', 'PBLH', 'HFX', 'LH', 'UST', 'WMAX', 'PBLFRAC')


def case_features(run_dir):
    """Per-rank feature frames -> one case-level vector.

    Frame 0 is dropped: it is the initial state, before any physics has run, and its file is
    visibly smaller than the rest. Averaging it in would dilute exactly the fields that decide
    cost.

    Two statistics per field, mirroring the hot-spot model's own-plus-neighbourhood split:
    the mean (how much work there is) and p90/mean (how unevenly it is spread). A case whose
    convection is concentrated costs differently from one with the same total spread evenly.
    """
    frames = sorted(p for p in glob.glob(os.path.join(run_dir, 'features_*.csv'))
                    if FEATSTAMP.search(p))
    if len(frames) < 2:
        return None, None
    acc = []
    for p in frames[1:]:
        with open(p) as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            if not all(f in hdr for f in FIELDS):      # frame lacks the canonical set
                continue
            col = [hdr.index(f) for f in FIELDS]
            rows = []
            for r in rd:
                if len(r) != len(hdr):
                    continue
                try:
                    rows.append([float(r[c]) for c in col])
                except ValueError:
                    continue
        if rows:
            acc.append(np.asarray(rows, float))
    if not acc:
        return None, None
    X = np.vstack(acc)                                 # (ranks x frames) x len(FIELDS)
    feats, fnames = [], []
    for j, nm in enumerate(FIELDS):
        feats.append(float(X[:, j].mean()))
        fnames.append('mean_' + nm)
    for nm in SPREAD_OF:
        v = X[:, FIELDS.index(nm)]
        m = v.mean()
        feats.append(float(np.percentile(v, 90) / m) if m > 0 else 1.0)
        fnames.append('p90rel_' + nm)
    return feats, fnames


def main():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if not r['s_per_model_h']:
                continue
            rows.append(r)

    # structural columns: these move ABSOLUTE cost more than any weather field does, so a model
    # that omits them is not learning weather, it is learning which cases have 80 levels.
    STRUCT = ['e_vert', 'e_we', 'e_sn', 'dx', 'n_ranks', 'time_step', 'model_h',
              'io_form_history', 'numtiles', 'nproc_x']
    data = []
    for r in rows:
        f, fn = case_features(os.path.join(RESULTS, r['run']))
        if f is None:
            continue
        sv = []
        ok = True
        for k in STRUCT:
            try:
                sv.append(float(r[k]))
            except (KeyError, ValueError, TypeError):
                ok = False
                break
        if not ok:
            continue
        try:
            mp, bl = int(r['mp_physics']), int(r['bl_pbl_physics'])
        except (KeyError, ValueError, TypeError):
            continue
        data.append({'run': r['run'], 'case': r['case'], 'mp': mp, 'bl': bl,
                     'y': math.log(float(r['s_per_model_h'])),
                     'x': np.array(sv + f, float), 'nstruct': len(sv),
                     'names': STRUCT + fn})
    if len(data) < 8:
        print('not enough usable runs: %d' % len(data))
        return 2

    cases = sorted({d['case'] for d in data})
    print('corpus: %d runs, %d distinct cases, %d features (%d structural + %d weather)'
          % (len(data), len(cases), len(data[0]['x']), data[0]['nstruct'],
             len(data[0]['x']) - data[0]['nstruct']))
    print('cases: ' + ' '.join(cases))
    print()

    def fit_main_effects(train):
        """f(mp), g(bl) with a per-case offset absorbed, so the main effects are not
        contaminated by which cases happen to carry which schemes."""
        fmp = {d['mp']: 0.0 for d in train}
        gbl = {d['bl']: 0.0 for d in train}
        off = {d['case']: 0.0 for d in train}
        for _ in range(300):
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

    def predict(train, test, cols, learner='gbrt'):
        fmp, gbl = fit_main_effects(train)
        # residual = what the recomputable config lookup cannot explain
        res = np.array([t['y'] - fmp.get(t['mp'], 0.0) - gbl.get(t['bl'], 0.0) for t in train])
        Xtr = np.array([t['x'][cols] for t in train])
        Xte = np.array([t['x'][cols] for t in test])
        base, _ = fit_predict(Xtr, res, Xte, learner)
        return [math.exp(b + fmp.get(t['mp'], 0.0) + gbl.get(t['bl'], 0.0))
                for b, t in zip(base, test)]

    nstruct = data[0]['nstruct']
    ALL = np.arange(len(data[0]['x']))
    STRUCT_ONLY = np.arange(nstruct)

    def looco(cols=None, anchor=False, median=False, learner='gbrt'):
        errs = []
        for c in cases:
            train = [d for d in data if d['case'] != c]
            test = [d for d in data if d['case'] == c]
            if median:
                p = math.exp(float(np.median([t['y'] for t in train])))
                pred = [p] * len(test)
            else:
                pred = predict(train, test, cols, learner)
            if anchor:
                # one run of the held-out case is allowed: use it to correct the level,
                # exactly the recipe cost_separability.py validated
                k = 0
                scale = math.exp(test[k]['y']) / pred[k]
                pred = [p * scale for p in pred]
                pairs = [(p, t) for i, (p, t) in enumerate(zip(pred, test)) if i != k]
            else:
                pairs = list(zip(pred, test))
            for p, t in pairs:
                errs.append(abs(p / math.exp(t['y']) - 1) * 100)
        return errs

    print('%-44s %8s %8s %8s' % ('', 'MAPE', 'median', 'worst'))
    print('  baseline (no learner):')
    e = looco(median=True)
    print('%-44s %7.1f%% %7.1f%% %7.1f%%'
          % ('    corpus median', float(np.mean(e)), float(np.median(e)), float(np.max(e))))
    for learner in ('gbrt', 'lgbm'):
        tag = 'hand-rolled GBRT (no regularisation)' if learner == 'gbrt' \
              else 'LightGBM 4.7.0 (regularised)'
        print('  %s:' % tag)
        for label, kw in [
                ('structure only (no weather at all)', dict(cols=STRUCT_ONLY)),
                ('FULL: structure + 17 weather  [v2]', dict(cols=ALL)),
                ('FULL + one anchor run of the case', dict(cols=ALL, anchor=True)),
        ]:
            e = looco(learner=learner, **kw)
            print('%-44s %7.1f%% %7.1f%% %7.1f%%'
                  % ('    ' + label, float(np.mean(e)), float(np.median(e)), float(np.max(e))))
    print()
    print('  (cost_separability.py, base fitted per case + one anchor: 3.06%% mean)')

    # which columns does the regularised learner actually spend splits on? The hot-spot model
    # was long believed to use weather when ~half its signal was three geometry columns, so
    # "the features are in the vector" is not evidence that they are used.
    fmp, gbl = fit_main_effects(data)
    res = np.array([t['y'] - fmp[t['mp']] - gbl[t['bl']] for t in data])
    _, m = fit_predict(np.array([t['x'] for t in data]), res,
                       np.array([data[0]['x']]), 'lgbm')
    gain = m.booster_.feature_importance('gain')   # LGBMRegressor wraps the Booster
    names = data[0]['names']
    order = np.argsort(gain)[::-1]
    tot = gain.sum() or 1.0
    print('\n  LightGBM split gain, top 10 of %d columns:' % len(names))
    for i in order[:10]:
        kind = 'struct ' if i < nstruct else 'weather'
        print('    %-7s %-22s %5.1f%%' % (kind, names[i], 100.0 * gain[i] / tot))
    wshare = 100.0 * gain[nstruct:].sum() / tot
    print('    -> weather columns take %.1f%% of total split gain' % wshare)
    return 0


if __name__ == '__main__':
    sys.exit(main())

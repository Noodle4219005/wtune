#!/usr/bin/env python3
# DEPRECATED 2026-09-11 (moved to deprecated/, see deprecated/README.md): the withdrawal
# below assumed the consumer would be restart-boundary rebalancing. That lever is closed
# (restart ~56 s > 7-17 s recoverable) and the live rebalancer, PLOV, is measurement-driven
# (it needs no prediction of which rank is hot). Kept runnable for the studies it produced.
# ─────────────── DEPRECATION WITHDRAWN 2026-09-08 ───────────────
# This module was marked DEPRECATED on 2026-09-07 on the grounds that the per-rank hot-spot
# target had no consumer, because all eight rebalancing actuators were closed.
#
# That reasoning does not hold. The consumer is rebalancing at restart boundaries, which is
# the project's actual objective, and three of the eight closures are ones the actuator
# inventory itself flags as SOFT:
#
#   * rank placement / pairing -- closed by an ANALYSIS built on a profiler with a known,
#     unfixed t0-bin bug; a later pairing simulation put the ceiling at 12.2-16.6% on
#     high-M cases, HIGHER than the analysis that closed it could see
#   * dynamic re-pin "worth 0.8%" -- explicitly withdrawn by its own author
#   * the "8.82% perfect-balance prize" that several closures are measured against -- later
#     re-derived from the raw ledger as 15.7%
#
# A second mistake fed the same conclusion: cost was found to be nearly state-independent,
# but that was measured on DOMAIN-AGGREGATE cost, and aggregation is precisely the operation
# that destroys per-rank redistribution information. The same corpus shows the per-rank
# pattern moving a great deal -- hot-spot identity decays 93.7% -> 54.3% over 16 model-hours.
# Total cost being flat and the pattern moving are not in conflict.
#
# So this stays live: it is the SENSOR for the lever, not a leftover.
# ────────────────────────────────────────────────────────────────
"""Settle the wtune model ARCHITECTURE: which features, how much data, which learner.

stencil_scale.py established the shape of the problem on 15 weather groups:

    own        17 cols  40.6%      geometry absent
    cur        26 cols  51.8%      geometry + 4-neighbour cross   <- production
    cur_nogeo  23 cols  41.3%      -10.5  <-- geometry is worth more than everything else
    wide3      41 cols  54.9%      +3.1
    wideall   139 cols  54.6%      +2.8   <-- 98 extra columns buy nothing

So the architecture hinges on ONE question: the 10.5-point geometry term is three columns
bundled together, and they are not the same kind of thing.

    geo_i, geo_j   normalised ABSOLUTE rank coordinates -- a lookup table on position.
                   Known not to transfer across decompositions (+4.4 at 8x56, -0.6 at 14x64).
    geo_nnb        halo neighbour count, i.e. "how far from the domain edge" -- a BOUNDARY
                   feature, translation-equivariant, and expressible by any convolutional
                   model through padding.

If the signal is mostly geo_nnb, a decomposition-independent model is possible and worth
building. If it is mostly geo_i/geo_j, the model is structurally tied to one decomposition
and the density-field idea is dead. Nothing measured so far separates them.

modes:
  features  every feature component measured separately, with PAIRED per-fold differences
            and a bootstrap CI -- "+3.1" with no spread is not evidence.
  curve     learning curve over the number of training WEATHER groups: is accuracy still
            rising at 14, i.e. data-limited, or flat, i.e. architecture-limited?
  models    learner comparison at fixed features: GBRT depth/n_est, kNN, ridge.

usage: wtune_arch.py <features|curve|models>
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle_train as tr
from stencil_scale import raw_case, base_of, box_stats, grad_mag, nbr_cross, hit

FIT = dict(depth=3, n_est=80, lr=0.10)
NBR_OF = ('radiation', 'micro', 'surf')
SCALES = (1, 2, 4)
MAXROWS = int(os.environ.get('WTUNE_MAXROWS', '8000'))
SEED = 20260905
BOOT = 2000


def feats(Xb, names, PX, PY, comps):
    """Assemble a design matrix from explicitly named components.

    comps subset of {'ij', 'nnb', 'nbr4', 'ms3'}. Keeping them separate is the whole point:
    'ij' is the non-transferable half of what used to be called "geometry".
    """
    NR = PX * PY
    I = np.arange(NR) % PX; J = np.arange(NR) // PX
    cols = [Xb]
    if 'ij' in comps:
        cols.append(np.column_stack([I / (PX - 1.0), J / (PY - 1.0)]))
    if 'nnb' in comps:
        nnb = ((I > 0).astype(float) + (I < PX - 1) + (J > 0) + (J < PY - 1))
        cols.append(nnb.reshape(-1, 1))
    if 'nbr4' in comps:
        for k in NBR_OF:
            if k in names:
                m, mx = nbr_cross(Xb[:, names.index(k)], PX, PY)
                cols.append(np.column_stack([m, mx]))
    if 'ms3' in comps:
        for k in NBR_OF:
            if k not in names:
                continue
            v = Xb[:, names.index(k)]
            per = []
            for r in SCALES:
                m, sd = box_stats(v, r, PX, PY)
                per += [m, sd]
            per.append(grad_mag(v, PX, PY))
            cols.append(np.column_stack(per))
    return np.nan_to_num(np.hstack(cols), nan=0.0, posinf=0.0, neginf=0.0)


def load_corpus():
    cases = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob('/home/u5727520/wrf-results/*/bucket_load.csv'))
    built = {}
    for c in cases:
        b = raw_case(c)
        if b:
            built[c] = b
    from collections import Counter
    cnt = Counter()
    for c, (segs, _, _) in built.items():
        for _X, nm, _r in segs:
            cnt[tuple(nm)] += 1
    schema = list(cnt.most_common(1)[0][0])
    # PROJECT supersets onto the modal schema by NAME instead of dropping them. The corpus's
    # stored CSVs were normalised to 17 columns, but the live featurizer still emits 21
    # (QKEMAX, QKEINT, QKELEV, ELMAX are extra). An equality test silently discarded every
    # freshly-run case -- which is how four brand-new 16x28 twins failed to appear in a corpus
    # count that stayed at exactly 38. Projection keeps them and changes nothing for the rest.
    kept = {}
    dropped = 0
    for c, (segs, PX, PY) in built.items():
        keep = []
        for Xb, nm, rate in segs:
            if list(nm) == schema:
                keep.append((Xb, nm, rate))
            elif set(schema).issubset(nm):
                idx = [list(nm).index(k) for k in schema]
                keep.append((Xb[:, idx], schema, rate))
            else:
                dropped += 1
        if keep:
            kept[c] = (keep, PX, PY)
    if dropped:
        print("  (%d segments dropped: schema is not a superset)" % dropped)
    groups = {}
    for c in kept:
        groups.setdefault(base_of(c), []).append(c)
    return kept, groups, schema


def train_eval(built, train_cases, test_cases, comps, fit=None, learner='gbrt', seed=0):
    """-> (mean hit%, mean cost err%) over the test cases' segments."""
    Xtr, ytr = [], []
    for c in train_cases:
        segs, PX, PY = built[c]
        for Xb, nm, rate in segs:
            Xtr.append(feats(Xb, nm, PX, PY, comps))
            ytr.append(rate / rate.mean())
    if not Xtr:
        return float('nan'), float('nan')
    Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
    if MAXROWS and len(ytr) > MAXROWS:
        idx = np.random.RandomState(SEED + seed).choice(len(ytr), MAXROWS, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if learner == 'gbrt':
        mdl = tr.gbrt_fit(Xtr, ytr, **(fit or FIT))
        pred = lambda X: tr.gbrt_pred(mdl, X)
    elif learner == 'ridge':
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
        A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
        w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)
        pred = lambda X: np.column_stack([np.ones(len(X)), (X - mu) / sd]) @ w
    elif learner.startswith('knn'):
        k = int(learner[3:])
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
        Z = (Xtr - mu) / sd

        def pred(X, Z=Z, ytr=ytr, k=k, mu=mu, sd=sd):
            Q = (X - mu) / sd
            out = np.empty(len(Q))
            for i in range(len(Q)):                     # corpus is small; exact kNN is fine
                d = ((Z - Q[i]) ** 2).sum(1)
                out[i] = ytr[np.argpartition(d, k)[:k]].mean()
            return out
    else:
        raise ValueError(learner)
    hits, errs = [], []
    for c in test_cases:
        segs, PX, PY = built[c]
        topk = max(1, int(round(0.10 * PX * PY)))
        for Xb, nm, rate in segs:
            p = pred(feats(Xb, nm, PX, PY, comps))
            hits.append(hit(p, rate, topk))
            truth = rate / rate.mean()
            errs.append(100.0 * np.median(np.abs(p - truth) / np.maximum(truth, 1e-9)))
    return float(np.mean(hits)), float(np.mean(errs))


def paired_ci(diffs, seed=1):
    """Bootstrap CI on the MEAN PAIRED difference. Pairing matters: folds differ wildly in
    difficulty, so an unpaired comparison is dominated by which folds happen to be easy."""
    rs = np.random.RandomState(seed)
    d = np.asarray(diffs, float)
    bs = [rs.choice(d, len(d), replace=True).mean() for _ in range(BOOT)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mode_features(built, groups):
    SETS = [
        ('own',            set()),
        ('own+nnb',        {'nnb'}),                  # boundary only  -- transferable
        ('own+ij',         {'ij'}),                   # absolute coords only -- not
        ('own+geo',        {'ij', 'nnb'}),
        ('cur',            {'ij', 'nnb', 'nbr4'}),    # production
        ('own+nnb+nbr4',   {'nnb', 'nbr4'}),
        ('wide3',          {'ij', 'nnb', 'ms3'}),
        ('CAND',           {'nnb', 'ms3'}),           # decomposition-independent candidate
    ]
    names = sorted(groups)
    print("weather groups=%d  folds=%d (all)  maxrows=%d" % (len(groups), len(names), MAXROWS))
    per = {}
    for tag, comps in SETS:
        fold_hits = []
        for gi, g in enumerate(names):
            held = set(groups[g])
            trn = [c for c in built if c not in held]
            h, e = train_eval(built, trn, groups[g], comps, seed=gi)
            fold_hits.append(h)
        per[tag] = np.array(fold_hits)
        print("  %-14s done  mean=%.1f%%" % (tag, np.nanmean(fold_hits)), flush=True)
    print()
    base = per['cur']
    print("%-14s %8s %10s %22s %s" % ("feature set", "mean", "vs cur", "95% CI of paired diff", "folds better"))
    for tag, v in per.items():
        d = v - base
        ok = np.isfinite(d)
        lo, hi = paired_ci(d[ok]) if tag != 'cur' and ok.sum() > 2 else (0.0, 0.0)
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else ("--" if tag != 'cur' else "")
        print("%-14s %7.1f%% %+9.1f %10.1f .. %+6.1f   %2d/%-2d  %s"
              % (tag, np.nanmean(v), np.nanmean(d), lo, hi, int((d[ok] > 0).sum()), int(ok.sum()), sig))
    print()
    print("READING: 'own+nnb' vs 'own+ij' says which half of the old geometry term carries the")
    print("signal. If own+nnb ~ own+geo, the model can be made decomposition-independent.")


def mode_curve(built, groups):
    names = sorted(groups)
    rs = np.random.RandomState(SEED)
    comps = {'ij', 'nnb', 'ms3'}
    print("learning curve, features=wide3, %d weather groups" % len(names))
    print("%6s %8s %10s" % ("k", "reps", "hit%"))
    for k in (3, 5, 7, 9, 11, len(names) - 1):
        if k >= len(names):
            continue
        vals = []
        for rep in range(5):
            perm = list(rs.permutation(names))
            trn_g, tst_g = perm[:k], perm[k:k + 3]
            trn = [c for g in trn_g for c in groups[g]]
            tst = [c for g in tst_g for c in groups[g]]
            h, _ = train_eval(built, trn, tst, comps, seed=rep)
            vals.append(h)
        print("%6d %8d %9.1f%%  (sd %.1f)" % (k, len(vals), np.mean(vals), np.std(vals)), flush=True)
    print()
    print("READING: still rising at the largest k -> DATA-limited, buy more weather cases.")
    print("         flat from mid k                -> ARCHITECTURE-limited, more data wastes money.")


def mode_models(built, groups):
    names = sorted(groups)
    comps = {'ij', 'nnb', 'ms3'}
    CONFS = [
        ('gbrt d2 n80',  'gbrt', dict(depth=2, n_est=80,  lr=0.10)),
        ('gbrt d3 n80',  'gbrt', dict(depth=3, n_est=80,  lr=0.10)),
        ('gbrt d4 n80',  'gbrt', dict(depth=4, n_est=80,  lr=0.10)),
        ('gbrt d3 n160', 'gbrt', dict(depth=3, n_est=160, lr=0.07)),
        ('ridge',        'ridge', None),
        ('knn5',         'knn5', None),
        ('knn20',        'knn20', None),
    ]
    print("learner comparison at features=wide3, %d folds" % len(names))
    print("%-14s %10s %12s" % ("learner", "hit%", "cost err%"))
    for tag, learner, fit in CONFS:
        hs, es = [], []
        for gi, g in enumerate(names):
            held = set(groups[g])
            trn = [c for c in built if c not in held]
            h, e = train_eval(built, trn, groups[g], comps, fit=fit, learner=learner, seed=gi)
            hs.append(h); es.append(e)
        print("%-14s %9.1f%% %11.1f%%" % (tag, np.nanmean(hs), np.nanmean(es)), flush=True)


def mean_profile(built, cases, PX, PY):
    """The static per-rank cost profile at one decomposition: mean of rate/mean(rate) over
    every training segment. No weather in it at all -- this is what the geometry is."""
    acc = []
    for c in cases:
        segs, px, py = built[c]
        if (px, py) != (PX, PY):
            continue
        for _X, _nm, rate in segs:
            acc.append(rate / rate.mean())
    return np.mean(acc, axis=0) if acc else None


def mode_residual(built, groups):
    """Predict the DEVIATION from a static per-rank profile instead of the level.

    geo_i/geo_j are worth ~12 points, and the likeliest reason is that a static per-rank cost
    profile exists -- edge ranks are cheap, some rows are reliably expensive -- and the trees
    are spending their capacity crudely reconstructing that lookup table from coordinates.

    The residual formulation hands it over for free:
        base[rank] = mean over training segments of rate/mean(rate)
        target     = rate/mean(rate) - base[rank]
        prediction = base[rank] + G(weather)

    Two things this could buy, neither yet measured:
      * the model spends its capacity on weather rather than on geometry, and
      * base is a lookup table that can be RECOMPUTED at any decomposition, so the
        decomposition-specific part stops being baked into the trained model.

    The first row is the diagnostic that was never run: how far does base alone get, with no
    model at all? If base alone is already close to the production model, most of what the
    model currently does is reproduce a table.
    """
    names = sorted(groups)
    ARMS = [
        ('base only',   None),                    # no model at all
        ('own',         (set(), False)),
        ('own+geo',     ({'ij', 'nnb'}, False)),  # current production
        ('resid own',   (set(), True)),
        ('resid +geo',  ({'ij', 'nnb'}, True)),
    ]
    print("leave-one-weather-out over %d groups; base = static per-rank profile\n" % len(names))
    print("%-14s %8s %10s %s" % ("arm", "cols", "hit%", " vs own+geo"))
    res = {}
    for tag, spec in ARMS:
        hits, ncol = [], 0
        for gi, g in enumerate(names):
            held = set(groups[g])
            trn = [c for c in built if c not in held]
            for c in groups[g]:
                segs, PX, PY = built[c]
                topk = max(1, int(round(0.10 * PX * PY)))
                base = mean_profile(built, trn, PX, PY)
                if base is None:
                    continue
                if spec is None:
                    for _X, _nm, rate in segs:
                        hits.append(hit(base, rate, topk))
                    continue
                comps, residual = spec
                Xtr, ytr = [], []
                for tc in trn:
                    tsegs, tpx, tpy = built[tc]
                    tbase = base if (tpx, tpy) == (PX, PY) else mean_profile(built, trn, tpx, tpy)
                    if tbase is None:
                        continue
                    for Xb, nm, rate in tsegs:
                        Xtr.append(feats(Xb, nm, tpx, tpy, comps))
                        y = rate / rate.mean()
                        ytr.append(y - tbase if residual else y)
                if not Xtr:
                    continue
                Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
                ncol = Xtr.shape[1]
                if MAXROWS and len(ytr) > MAXROWS:
                    idx = np.random.RandomState(SEED + gi).choice(len(ytr), MAXROWS, replace=False)
                    Xtr, ytr = Xtr[idx], ytr[idx]
                mdl = tr.gbrt_fit(Xtr, ytr, **FIT)
                for Xb, nm, rate in segs:
                    p = tr.gbrt_pred(mdl, feats(Xb, nm, PX, PY, comps))
                    if residual:
                        p = base + p
                    hits.append(hit(p, rate, topk))
        res[tag] = float(np.mean(hits)) if hits else float('nan')
        print("%-14s %8s %9.1f%%" % (tag, ncol if ncol else "-", res[tag]), flush=True)
    print()
    b = res.get('own+geo', float('nan'))
    for tag, v in res.items():
        if tag != 'own+geo':
            print("  %-14s %+6.1f vs own+geo" % (tag, v - b))
    print()
    print("READING")
    print("  'base only' high  -> most of the model's output is a static lookup table, and the")
    print("                       decomposition-specific part can be recomputed rather than trained.")
    print("  'resid own' >= 'own+geo' with fewer columns -> the residual framing works and the")
    print("                       geometry columns were a crude stand-in for the profile.")


def mode_decomp(built, groups):
    """Does the geo_i/geo_j term actually break across a DECOMPOSITION change?

    The features run says absolute rank coordinates carry essentially all of the geometry
    signal (own+ij 45.9% vs own+nnb 35.4%). If that is a coordinate lookup rather than
    physics, a model trained at 8x56 must degrade when tested at 14x64 -- and a model
    WITHOUT those columns must not. karina_14x64 is the same storm as karina_cat4 run at a
    different decomposition, so this isolates decomposition from weather.
    """
    # _16x28 (6 h) and _16x28h12 (12 h, matched to the controls) both count as variants.
    test = [c for c in built if re.search(r'_\d+x\d+(h\d+)?$', c)]
    if not test:
        print("no decomposition-variant case in the corpus"); return
    print("decomposition transfer test")
    print("%-22s %-16s %10s %10s" % ("test case", "features", "hit%", "cost err%"))
    for tc in sorted(test):
        base = base_of(tc)
        # train on every case that is NOT this weather at all -- so the only thing being
        # measured is transfer, not memorisation of the same storm at another decomposition
        trn = [c for c in built if base_of(c) != base]
        _s, PX, PY = built[tc]
        for tag, comps in (('with ij', {'ij', 'nnb', 'ms3'}), ('NO ij', {'nnb', 'ms3'})):
            h, e = train_eval(built, trn, [tc], comps)
            print("%-22s %-16s %9.1f%% %9.1f%%   (%dx%d)" % (tc, tag, h, e, PX, PY))
        # and the same weather's 8x56 sibling as the control, same training set
        sib = [c for c in built if base_of(c) == base and c != tc]
        if sib:
            for tag, comps in (('with ij', {'ij', 'nnb', 'ms3'}), ('NO ij', {'nnb', 'ms3'})):
                h, e = train_eval(built, trn, sib[:1], comps)
                _s2, px2, py2 = built[sib[0]]
                print("%-22s %-16s %9.1f%% %9.1f%%   (%dx%d) CONTROL"
                      % (sib[0], tag, h, e, px2, py2))
    print()
    print("READING: if 'with ij' drops at 14x64 but holds at 8x56 while 'NO ij' is flat across")
    print("both, the geometry term is a per-decomposition lookup and the model must be")
    print("retrained for every decomposition it is asked to advise on.")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'features'
    built, groups, schema = load_corpus()
    print("corpus: %d dirs, %d weather groups, %d fields" % (len(built), len(groups), len(schema)))
    {'features': mode_features, 'curve': mode_curve, 'models': mode_models,
     'decomp': mode_decomp, 'residual': mode_residual}[mode](built, groups)
    return 0


if __name__ == '__main__':
    sys.exit(main())

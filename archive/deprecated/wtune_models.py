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
"""Which model class actually predicts the hot ranks? GBRT vs KNN vs linear.

The question is not academic. Roughly half the per-rank signal is a FIXED POSITIONAL PRIOR:
three geometry features (rank i, rank j, halo neighbour count), identical in every case, score
+0.380 Spearman / ~40% hit rate on their own, with no weather information at all. A prior of
that shape is a lookup table, and KNN *is* a lookup table -- while a depth-3 GBRT has to spend
its handful of splits reconstructing one.

Pre-registered prediction (written before running): KNN wins on within-case ordering, where
the positional prior dominates, and ties or loses on leave-one-case-out, because LOCO is
out-of-distribution by construction -- a held-out regime's feature vectors can sit off the
training manifold entirely, and KNN does not extrapolate, it just returns whatever is nearest.

Metric is the operational one: of the 45 genuinely most expensive ranks in a segment (top 10%),
how many are in the predicted top 45. Random = 10%.

usage: wtune_models.py [case ...]
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

SEG_HOURS = 3.0
TOPK = 45
GBRT = dict(depth=3, n_est=80, lr=0.10)


def rows_for(case):
    r, why = seg.build_case(case)
    if r is None:
        return None
    X, y, cs, sg, names = r
    nl = f'{seg.RESULTS}/{case}/namelist.input'
    txt = open(nl).read() if os.path.exists(nl) else ''
    m = re.search(r'time_step=(\d+)', txt)
    dt = int(m.group(1)) if m else 15
    return dict(case=case, X=X, names=names, rate=cs / (SEG_HOURS * 3600.0 / dt), seg=sg)


def knn_predict(Xtr, ytr, Xte, k, chunk=512):
    """Brute-force k-NN regression on z-scored features. No sklearn in this env, and at
    ~16k x 20 a chunked distance matrix is cheaper than a tree index would be to maintain."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    an = (A * A).sum(1)
    out = np.empty(len(B))
    for i in range(0, len(B), chunk):
        b = B[i:i + chunk]
        d = an[None, :] - 2.0 * b @ A.T + (b * b).sum(1)[:, None]
        idx = np.argpartition(d, k, axis=1)[:, :k]
        out[i:i + chunk] = ytr[idx].mean(1)
    return out


def ridge_predict(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(Xtr)), (Xtr - mu) / sd])
    B = np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd])
    G = A.T @ A + lam * np.eye(A.shape[1])
    G[0, 0] -= lam
    return B @ np.linalg.solve(G, A.T @ ytr)


def hit_rate(score, true, sg):
    hits = []
    for k in np.unique(sg):
        idx = np.where(sg == k)[0]
        if len(idx) < TOPK:
            continue
        ht = set(idx[np.argsort(true[idx])[-TOPK:]])
        hp = set(idx[np.argsort(score[idx])[-TOPK:]])
        hits.append(100.0 * len(ht & hp) / TOPK)
    return float(np.mean(hits)) if hits else np.nan


def main():
    want = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(f'{seg.RESULTS}/*')
                                  if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
    rows = [r for r in (rows_for(c) for c in want) if r]
    n = len(rows)
    if n < 4:
        print("need >= 4 cases; got %d" % n); return 2
    X = np.vstack([r['X'] for r in rows])
    rate = np.concatenate([r['rate'] for r in rows])
    sgs = np.concatenate([r['seg'] for r in rows])
    cid = np.concatenate([np.full(len(r['rate']), i) for i, r in enumerate(rows)])
    grp = sgs + 1000 * cid
    smean = np.empty_like(rate)
    for g in np.unique(grp):
        smean[grp == g] = rate[grp == g].mean()
    shape = rate / smean
    names = rows[0]['names']
    geo = [i for i, nm in enumerate(names) if nm.startswith('geo_')]

    MODELS = [
        ("GBRT d3/80 (現行)", lambda a, b, c: tr.gbrt_pred(tr.gbrt_fit(a, b, **GBRT), c), None),
        ("KNN k=10", lambda a, b, c: knn_predict(a, b, c, 10), None),
        ("KNN k=50", lambda a, b, c: knn_predict(a, b, c, 50), None),
        ("KNN k=200", lambda a, b, c: knn_predict(a, b, c, 200), None),
        ("KNN k=50 只用幾何", lambda a, b, c: knn_predict(a, b, c, 50), geo),
        ("ridge 線性", lambda a, b, c: ridge_predict(a, b, c), None),
    ]

    print("=== 模型比較（特徵/折/目標完全相同，只換模型）===")
    print("樣本 %s，%d 個 case\n" % (str(X.shape), n))
    print("%-22s %14s %16s" % ("模型", "LOCO 命中率", "case 內 5-fold 命中率"))
    res = {}
    for nm, fn, cols in MODELS:
        Xc = X[:, cols] if cols else X
        loco, within = [], []
        for i in range(n):
            trm, tem = cid != i, cid == i
            loco.append(hit_rate(fn(Xc[trm], shape[trm], Xc[tem]), rate[tem], sgs[tem]))
            # within-case: 5 deterministic folds inside the SAME case, so the model has seen
            # this regime -- isolates "is the mapping learnable" from "does it transfer"
            ix = np.where(tem)[0]
            pw = np.empty(len(ix))
            for f in range(5):
                te = np.zeros(len(ix), bool); te[f::5] = True
                pw[te] = fn(Xc[ix[~te]], shape[ix[~te]], Xc[ix[te]])
            within.append(hit_rate(pw, rate[ix], sgs[ix]))
        res[nm] = (np.mean(loco), np.mean(within))
        print("%-22s %13.1f%% %15.1f%%" % (nm, res[nm][0], res[nm][1]))
    best_l = max(res, key=lambda k: res[k][0])
    best_w = max(res, key=lambda k: res[k][1])
    print("\n-> LOCO 最佳: %s (%.1f%%)   case 內最佳: %s (%.1f%%)"
          % (best_l, res[best_l][0], best_w, res[best_w][1]))
    return 0


if __name__ == '__main__':
    sys.exit(main())

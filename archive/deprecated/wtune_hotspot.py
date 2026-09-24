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
"""Optimise for the metric that rebalancing actually uses: WHICH ranks are the hot ones.

The shape model is trained to predict every rank's cost. Rebalancing only ever moves the
expensive tail, and the operational metric is therefore

    hit rate = |predicted top-45 AND true top-45| / 45        (random = 10%)

Regression on the whole distribution is not that objective. A squared-error fit spends most
of its effort on the 90% of ranks that will never be moved. So before spending 4-node cluster
hours on more cases, test whether simply changing the TARGET lifts the hit rate on the data
already in hand.

Targets compared (features, folds and model identical throughout; only y changes):
  shape    cost / segment mean                    -- the current baseline
  logshape log of the same                        -- equalises relative deviations
  pct      within-segment percentile rank 0..1    -- purely ordinal, segment-balanced
  top10    1 if in the segment's top 10%, else 0  -- the metric itself, as a classifier
  top25    1 if in the segment's top 25%          -- softer, 4x more positive examples

Reports hit rate AND per-rank cost error, because they are different jobs: a classifier can
rank the tail well while saying nothing about seconds. If the split is real, the answer is
two heads -- a classifier to choose the ranks, the regressor to price them.

usage: wtune_hotspot.py [case ...]
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

SEG_HOURS = 3.0
FIT = dict(depth=3, n_est=80, lr=0.10)
TOPK = 45


def rows_for(case):
    r, why = seg.build_case(case)
    if r is None:
        return None
    X, y, cs, sg, names = r
    s = seg.schemes(case)
    nl = f'{seg.RESULTS}/{case}/namelist.input'
    txt = open(nl).read() if os.path.exists(nl) else ''
    m = re.search(r'time_step=(\d+)', txt)
    dt = int(m.group(1)) if m else 15
    return dict(case=case, X=X, rate=cs / (SEG_HOURS * 3600.0 / dt), seg=sg,
                L=float(s['e_vert']), mp=s['mp'], bl=s['bl'])


def per_seg(fn, v, sg):
    out = np.empty(len(v), float)
    for k in np.unique(sg):
        m = sg == k
        out[m] = fn(v[m])
    return out


def loo_lin(F, y):
    n = len(y)
    A = np.column_stack([np.ones(n), F])
    p = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        c, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        p[i] = A[i] @ c
    return p


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

    shape = rate / per_seg(lambda v: np.full(len(v), v.mean()), rate, sgs + 1000 * cid)
    grp = sgs + 1000 * cid
    pct = per_seg(lambda v: np.argsort(np.argsort(v)) / (len(v) - 1.0), rate, grp)
    top10 = (pct >= 1 - TOPK / 448.0).astype(float)
    top25 = (pct >= 0.75).astype(float)
    TARGETS = (("shape (現行)", shape), ("logshape", np.log(shape)),
               ("pct 段內百分位", pct), ("top10 分類", top10), ("top25 分類", top25))

    mean_rate = np.array([r['rate'].mean() for r in rows])
    rate_pred = loo_lin(np.array([[r['L']] for r in rows], float), mean_rate)

    print("=== 目標函數比較（特徵/折/模型完全相同，只換 y）===")
    print("%-16s %10s %14s" % ("訓練目標", "熱點命中率", "每-rank 成本誤差"))
    scores = {}
    for nm, yt in TARGETS:
        sc = np.empty(len(yt))
        for i in range(n):
            trm, tem = cid != i, cid == i
            sc[tem] = tr.gbrt_pred(tr.gbrt_fit(X[trm], yt[trm], **FIT), X[tem])
        scores[nm] = sc
        hits, errs = [], []
        for i in range(n):
            tem = cid == i
            hits.append(hit_rate(sc[tem], rate[tem], sgs[tem]))
            if nm == "shape (現行)":
                pred = rate_pred[i] * sc[tem]
                errs.append(100 * np.mean(np.abs(pred - rate[tem]) / rate[tem]))
        print("%-16s %9.1f%% %14s" % (nm, np.mean(hits),
                                      "%.1f%%" % np.mean(errs) if errs else "—(非成本尺度)"))

    best = max(scores, key=lambda k: np.mean(
        [hit_rate(scores[k][cid == i], rate[cid == i], sgs[cid == i]) for i in range(n)]))
    print("\n-> 最佳選擇器: %s" % best)

    print("\n=== 最佳選擇器的逐測資命中率（對照現行）===")
    print("%-18s %5s %4s %4s %12s %12s %8s" % ("測資", "L", "mp", "bl", "現行", best, "改善"))
    for i, r in enumerate(rows):
        tem = cid == i
        a = hit_rate(scores["shape (現行)"][tem], rate[tem], sgs[tem])
        b = hit_rate(scores[best][tem], rate[tem], sgs[tem])
        print("%-18s %5.0f %4d %4d %11.1f%% %11.1f%% %+7.1f" % (r['case'], r['L'], r['mp'], r['bl'], a, b, b - a))
    ha = np.mean([hit_rate(scores["shape (現行)"][cid == i], rate[cid == i], sgs[cid == i]) for i in range(n)])
    hb = np.mean([hit_rate(scores[best][cid == i], rate[cid == i], sgs[cid == i]) for i in range(n)])
    print("%-18s %5s %4s %4s %11.1f%% %11.1f%% %+7.1f  <- 平均（亂猜 10.0%%）"
          % ("", "", "", "", ha, hb, hb - ha))
    return 0


if __name__ == '__main__':
    sys.exit(main())

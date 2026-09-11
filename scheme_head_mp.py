#!/usr/bin/env python3
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
"""Is one shared mapping the wrong model? Test a per-scheme head against it.

Three measurements on the same MYNN case pin the problem down:
  93.3%  run-to-run reproducibility  -> the pattern is deterministic
  86.1%  within-case 5-fold hit rate -> the FEATURES already describe it
  42.2%  leave-one-case-out hit rate -> what fails is TRANSFER

So the gap is not a missing variable (QKE and EL_PBL were tried and did not help; the
features are already sufficient in-case). It is that a single shared regressor fits one
mapping to cases whose mappings differ by PBL scheme.

Compared here, same folds and features throughout:
  shared    train on every other case            (current behaviour)
  same-mp   train only on other cases with the SAME mp_physics
Same-bl sees ~1/3 the data, so if it still wins, the scheme really is a separate regime.
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg, wtune_oracle_train as tr

TOPK, FIT = 45, dict(depth=3, n_est=80, lr=0.10)
SEG_HOURS = 3.0


def hit(score, true, sgv):
    out = []
    for k in np.unique(sgv):
        ix = np.where(sgv == k)[0]
        if len(ix) < TOPK:
            continue
        a = set(ix[np.argsort(true[ix])[-TOPK:]]); b = set(ix[np.argsort(score[ix])[-TOPK:]])
        out.append(100.0 * len(a & b) / TOPK)
    return float(np.mean(out)) if out else np.nan


want = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(f'{seg.RESULTS}/*')
                              if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
rows = []
for c in want:
    r, why = seg.build_case(c)
    if r is None:
        continue
    X, y, cs, sg, names = r
    nl = f'{seg.RESULTS}/{c}/namelist.input'
    t = open(nl).read() if os.path.exists(nl) else ''
    m = re.search(r'time_step=(\d+)', t)
    dt = int(m.group(1)) if m else 15
    rows.append(dict(case=c, X=X, rate=cs / (SEG_HOURS * 3600.0 / dt), seg=sg,
                     mp=seg.schemes(c)['mp'], names=names))
n = len(rows)
X = np.vstack([r['X'] for r in rows])
rate = np.concatenate([r['rate'] for r in rows])
sgs = np.concatenate([r['seg'] for r in rows])
cid = np.concatenate([np.full(len(r['rate']), i) for i, r in enumerate(rows)])
grp = sgs + 1000 * cid
sm = np.empty_like(rate)
for g in np.unique(grp):
    sm[grp == g] = rate[grp == g].mean()
shape = rate / sm
bl = np.array([r['mp'] for r in rows])

print("=== 共用映射 vs 同方案 head（LOCO，特徵/折完全相同）===")
print("%-26s %3s %12s %12s %8s" % ("case", "bl", "shared", "same-mp", "差"))
A, B = [], []
for i in range(n):
    te = cid == i
    m_all = cid != i
    m_bl = (cid != i) & np.isin(cid, np.where(bl == bl[i])[0])
    a = hit(tr.gbrt_pred(tr.gbrt_fit(X[m_all], shape[m_all], **FIT), X[te]), rate[te], sgs[te])
    if m_bl.sum() == 0:
        b = np.nan
    else:
        b = hit(tr.gbrt_pred(tr.gbrt_fit(X[m_bl], shape[m_bl], **FIT), X[te]), rate[te], sgs[te])
    A.append(a); B.append(b)
    print("%-26s %3d %11.1f%% %11.1f%% %+7.1f" % (rows[i]['case'], bl[i], a, b, b - a))
print("%-26s %3s %11.1f%% %11.1f%% %+7.1f  <- 平均" % ("", "", np.nanmean(A), np.nanmean(B),
                                                       np.nanmean(B) - np.nanmean(A)))
for v in sorted(set(bl)):
    k = [i for i in range(n) if bl[i] == v]
    print("  mp=%d (%d case): shared %.1f%%  same-mp %.1f%%"
          % (v, len(k), np.nanmean([A[i] for i in k]), np.nanmean([B[i] for i in k])))

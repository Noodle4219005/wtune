#!/usr/bin/env python3
"""Does MYNN's own prognostic turbulence close the hot-spot gap? Within-case, so it is free.

The surface proxies (PBLH/HFX/LH/UST/WMAX/PBLFRAC) did not: the MYNN gap went 18.6 -> 21.3
points. A repeat run then showed the per-rank pattern is 93.3% reproducible, so the missing
information exists and those proxies were simply the wrong variables. QKE (=2*TKE) and EL_PBL
are MYNN's own state, and QKELEV -- the number of levels with QKE above a quiescence
threshold -- is the loop count the scheme actually pays for.

This compares feature sets on ONE case with 5 deterministic within-case folds, which needs
only the feature CSVs, no frames and no cluster run.
"""
import sys, numpy as np
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import wtune_segment_train as seg, wtune_oracle_train as tr

CASE = sys.argv[1] if len(sys.argv) > 1 else 'karina_cat4__mp10bl5'
TOPK, FIT = 45, dict(depth=3, n_est=80, lr=0.10)
QKE_COLS = ('QKEMAX', 'QKEINT', 'QKELEV', 'ELMAX')

r, why = seg.build_case(CASE)
if r is None:
    print("SKIP:", why); sys.exit(2)
X, y, cs, sg, names = r
print("%s: X%s" % (CASE, X.shape))
print("欄位: %s" % names)

def hit(score, true, sgv):
    out = []
    for k in np.unique(sgv):
        ix = np.where(sgv == k)[0]
        if len(ix) < TOPK:
            continue
        a = set(ix[np.argsort(true[ix])[-TOPK:]]); b = set(ix[np.argsort(score[ix])[-TOPK:]])
        out.append(100.0 * len(a & b) / TOPK)
    return float(np.mean(out))

smean = np.empty_like(cs)
for k in np.unique(sg):
    smean[sg == k] = cs[sg == k].mean()
shape = cs / smean

qk = [i for i, n in enumerate(names) if n in QKE_COLS or n.startswith(('nbrmean_QKE','nbrmax_QKE'))]
base = [i for i in range(len(names)) if i not in qk]
print("QKE 相關欄位 %d 個: %s\n" % (len(qk), [names[i] for i in qk]))

for label, cols in (("不含 QKE (%d 欄)" % len(base), base), ("含 QKE (%d 欄)" % len(names), list(range(len(names))))):
    Xc = X[:, cols]
    pred = np.empty(len(shape))
    for f in range(5):
        te = np.zeros(len(shape), bool); te[f::5] = True
        pred[te] = tr.gbrt_pred(tr.gbrt_fit(Xc[~te], shape[~te], **FIT), Xc[te])
    print("%-22s case 內 5-fold 熱點命中率 = %.1f%%   Spearman = %+.3f"
          % (label, hit(pred, cs, sg), tr._spearman(pred, shape)))

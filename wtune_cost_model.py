#!/usr/bin/env python3
"""Two-stage COST model: a case-level RATE times a per-rank SHAPE.

    comp[r] / step  =  scale(config)  x  shape(weather, geometry)[r]
                       ^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^
                       10 samples        17920 samples
                       namelist only     weather + halo geometry

Predicting a rate rather than a duration is the whole point: every decision wtune makes needs
a ratio or an ordering, never an absolute second, and

    total_wall = n_steps * rate,   n_steps = sim_hours * 3600 / dt   <- exact arithmetic.

Why TWO stages and not one joint fit. A single structural regression
`rate[r] = a_dyn*L + a_rad*L*rad[r] + ...` was tried first and lost badly to the GBRT
(per-rank LOCO +0.142 vs +0.555, case-level 15.1% vs 7.9%), and its fitted coefficients were
physically wrong -- radiation clipped to zero, Morrison ranked cheaper than Thompson. The
reason is structural, not a tuning failure: the oracle's phase loads are normalised PER CASE,
so their magnitude is not comparable across cases, and least squares on the absolute rate is
then dominated by the between-case level effect. The scale term absorbs everything and the
within-case pattern never gets fitted. Splitting the two questions removes that competition:
each stage sees only the variation it is supposed to explain.

Stage 1 (scale): mean rate per case from e_vert / mp / bl. Namelist only -- weather does not
                 enter, which is itself a measured result, not an assumption.
Stage 2 (shape): per-rank rate divided by its own case-segment mean, from weather + halo
                 geometry. Dividing out the mean is what makes 17920 rows usable for one
                 question instead of 10 rows for two.

Reported: Spearman (is the ordering right?) AND relative MAE on the absolute per-rank rate
(is the number right?). The controller needs the second one -- it moves seconds, not ranks.

usage: wtune_cost_model.py [--extrap] [case ...]
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

NR = seg.NR
SEG_HOURS = 3.0
FIT = dict(depth=3, n_est=80, lr=0.10)
# Geometry (geo_i/geo_j/geo_nnb) is EXCLUDED by default. It raises accuracy on the
# decomposition it was trained on (+4.4pp) and is worth -0.6pp on a different one -- and
# changing decomposition is precisely what the transition controller exists to do, so a model
# that leans on it is unreliable at the one moment it is consulted. Measured 2026-09-05:
# weather-only scores an identical 80.6% at 8x56 and at 14x64, so dropping geometry is what
# makes the number decomposition-independent. Set WTUNE_USE_GEO=1 to study the same-shape case.
USE_GEO = os.environ.get('WTUNE_USE_GEO') == '1'
PURE = os.environ.get('WTUNE_MEASURED_SHARE') is None   # default: no measured value leaks in


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
    rate = cs / (SEG_HOURS * 3600.0 / dt)          # compute seconds per model step, per rank
    return dict(case=case, X=X, names=names, rate=rate, seg=sg, L=float(s['e_vert']),
                mp=s['mp'], bl=s['bl'], ra=s['ra'], dt=dt)


# Stage 2 is fitted PER PBL SCHEME. Measured: a same-bl head lifts MYNN from 57.6% to 75.7%
# hot-spot hit rate while leaving YSU and MYJ unchanged -- and it does so on a THIRD of the
# training data, which is the signature of a genuinely separate regime rather than of more
# capacity. The same test on mp_physics goes the other way (63.4% -> 61.8%), so the split
# axis is bl and only bl.
#
# It lines up with the two-stage decomposition exactly: mp changes the per-step RATE
# (WSM6 < Thompson < Morrison, which stage 1 already handles), while bl changes the per-rank
# SHAPE, because a TKE-predicting scheme and a K-profile scheme have different algorithm
# structure. Rate and shape are the two stages.
# REFUTED on a diverse corpus, 2026-09-05. Keep it available, but OFF by default.
#   karina-dominated pool (11 cases, 9 sharing one storm): bl=5 57.6% -> 75.7%, +18.1
#   diverse pool (23 cases, 4 weather regimes):            bl=5 44.7% -> 43.1%, -1.6
#                                                          overall 48.3% -> 44.6%, -3.7
# The gain was never scheme purity. In the old pool a same-bl training set happened to be
# same-WEATHER as well, so "train on your own scheme" smuggled in "train on your own storm".
# With genuinely different regimes the extra data from other schemes is worth more than
# scheme purity, and splitting throws it away.
PER_BL_HEAD = os.environ.get('WTUNE_PER_BL_HEAD') == '1'


def fit_shape(Xtr, ytr, bltr, Xte, blte, fit_kw):
    """Pooled across schemes by default; per-PBL-scheme only when explicitly requested."""
    m = np.ones(len(bltr), bool)
    if PER_BL_HEAD:
        sel = bltr == blte
        if sel.sum() >= 448:                # at least one whole case to stand on
            m = sel
    return tr.gbrt_pred(tr.gbrt_fit(Xtr[m], ytr[m], **fit_kw), Xte)


def seg_mean(rate, sg):
    """Mean rate of the segment each row belongs to (the local scale to divide out)."""
    out = np.empty_like(rate)
    for k in np.unique(sg):
        m = sg == k
        out[m] = rate[m].mean()
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


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    extrap = '--extrap' in sys.argv[1:]
    want = argv or sorted(os.path.basename(d) for d in glob.glob(f'{seg.RESULTS}/*')
                          if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
    rows = [r for r in (rows_for(c) for c in want) if r]
    if len(rows) < 4:
        print("need >= 4 cases; got %d" % len(rows)); return 2
    n = len(rows)
    X = np.vstack([r['X'] for r in rows])
    if not USE_GEO:
        keep = [i for i, q in enumerate(rows[0]['names']) if not q.startswith('geo_')]
        X = X[:, keep]
        print("特徵: 已排除 %d 個幾何欄，使用 %d 欄（decomposition 無關）"
              % (len(rows[0]['names']) - len(keep), len(keep)))
    rate = np.concatenate([r['rate'] for r in rows])
    sgm = np.concatenate([seg_mean(r['rate'], r['seg']) for r in rows])
    cid = np.concatenate([np.full(len(r['rate']), i) for i, r in enumerate(rows)])
    shape = rate / sgm                              # ~1.0, dimensionless, comparable across cases

    print("=== 兩階段成本模型 ===")
    print("階段 1（率）  : %d 個 case x 3 個 namelist 欄位" % n)
    print("階段 2（形狀）: %d 列 x %d 個特徵" % X.shape)
    print("\n%-18s %5s %4s %4s %4s %10s %8s" % ("case", "L", "mp", "bl", "dt", "率(ms/步)", "形狀範圍"))
    for r in rows:
        print("%-18s %5.0f %4d %4d %4d %10.1f %8s"
              % (r['case'], r['L'], r['mp'], r['bl'], r['dt'], 1000 * r['rate'].mean(),
                 "%.2f-%.2f" % ((r['rate'] / seg_mean(r['rate'], r['seg'])).min(),
                                (r['rate'] / seg_mean(r['rate'], r['seg'])).max())))

    # ---------- stage 1: the RATE, from the namelist only ----------
    mean_rate = np.array([r['rate'].mean() for r in rows])
    L = np.array([[r['L']] for r in rows], float)
    Lmpbl = np.array([[r['L'], r['mp'], r['bl']] for r in rows], float)
    print("\n=== 階段 1：case 平均率（leave-one-out）===")
    for nm, F in (("平均基準", None), ("e_vert", L), ("e_vert+mp+bl", Lmpbl)):
        p = (np.array([np.mean(np.delete(mean_rate, i)) for i in range(n)])
             if F is None else loo_lin(F, mean_rate))
        e = np.abs(p - mean_rate).mean()
        print("  %-14s MAE %7.1f ms/step  (%.1f%%)" % (nm, 1000 * e, 100 * e / mean_rate.mean()))
    # use whichever stage-1 form actually won LOO, rather than the most elaborate one:
    # at n=10, e_vert alone beats e_vert+mp+bl, so mp/bl are costing more in variance than
    # they buy in bias.
    cands = {"e_vert": loo_lin(L, mean_rate), "e_vert+mp+bl": loo_lin(Lmpbl, mean_rate)}
    best = min(cands, key=lambda k: np.abs(cands[k] - mean_rate).mean())
    rate_pred = cands[best]
    print("  -> 合成使用: %s" % best)

    # ---------- stage 2: the SHAPE, from weather + geometry ----------
    print("\n=== 階段 2：per-rank 形狀（LOCO）===")
    blrow = np.concatenate([np.full(len(r['rate']), r['bl']) for r in rows])
    sp, sh_pred = [], np.empty_like(shape)
    for i in range(n):
        trm, tem = cid != i, cid == i
        sh_pred[tem] = fit_shape(X[trm], shape[trm], blrow[trm], X[tem], rows[i]['bl'], FIT)
        sp.append(tr._spearman(sh_pred[tem], shape[tem]))
        print("  %-18s rho=%+.3f" % (rows[i]['case'], sp[-1]))
    print("  %-18s %+.3f  <- mean" % ("", np.mean(sp)))

    # ---------- the number the controller actually consumes ----------
    print("\n=== 合成：絕對 per-rank 率（%s）===" % ("share=1，無洩漏" if PURE else "share 用實測值"))
    print("%-18s %10s %10s %9s %8s" % ("case", "實測均值", "模型均值", "相對MAE", "最大rank"))
    rels, rels_hot = [], []
    for i, r in enumerate(rows):
        tem = cid == i
        # share = each segment's rate relative to the case mean. Measured share is a LEAK, so
        # report both: the measured one for reference and share=1 for a leak-free composite.
        # wtune_segshare.py measured share = 1.00 +/- 0.09 with NO predictor (solar, frame,
        # segment index, persistence) beating the constant, so 1 is the honest choice.
        share_meas = seg_mean(r['rate'], r['seg']) / r['rate'].mean()
        share = np.ones_like(share_meas) if PURE else share_meas
        pred = rate_pred[i] * share * sh_pred[tem]
        rel = np.abs(pred - rate[tem]) / rate[tem].mean()
        rels.append(rel.mean())
        k = np.argsort(rate[tem])[-45:]                       # the hottest 10% of ranks
        rels_hot.append((np.abs(pred[k] - rate[tem][k]) / rate[tem][k]).mean())
        print("%-18s %10.1f %10.1f %8.1f%% %7.1f%%"
              % (r['case'], 1000 * rate[tem].mean(), 1000 * pred.mean(),
                 100 * rel.mean(), 100 * rels_hot[-1]))
    print("%-18s %10s %10s %8.1f%% %7.1f%%"
          % ("平均", "", "", 100 * np.mean(rels), 100 * np.mean(rels_hot)))

    if extrap:
        print("\n=== 外推：只用 e_vert<=60 的 case 擬合階段 1，預測 80/100 ===")
        tr_ix = np.array([r['L'] <= 60 for r in rows])
        if tr_ix.sum() >= 3 and (~tr_ix).sum():
            # Training on L<=60 leaves only TWO distinct level counts, so a 3-predictor fit
            # has almost nothing to determine the slope with. Report both forms: if the plain
            # e_vert line extrapolates and the 3-predictor one does not, the failure is
            # over-parameterisation at n=6, not the model form.
            for nm, F in (("e_vert", L), ("e_vert+mp+bl", Lmpbl)):
                A = np.column_stack([np.ones(tr_ix.sum()), F[tr_ix]])
                c, *_ = np.linalg.lstsq(A, mean_rate[tr_ix], rcond=None)
                errs = []
                for i in np.where(~tr_ix)[0]:
                    pv = np.concatenate([[1.0], F[i]]) @ c
                    errs.append(100 * (pv / mean_rate[i] - 1))
                    print("  %-13s %-16s L=%-4.0f 實測 %.1f / 模型 %.1f  (%+.0f%%)"
                          % (nm, rows[i]['case'], rows[i]['L'], 1000 * mean_rate[i],
                             1000 * pv, errs[-1]))
                print("  %-13s 平均偏誤 %+.0f%%, 平均絕對 %.0f%%\n"
                      % (nm, np.mean(errs), np.mean(np.abs(errs))))
    return 0


if __name__ == '__main__':
    sys.exit(main())

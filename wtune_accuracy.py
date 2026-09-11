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
"""Per-case, per-segment accuracy in PLAIN PERCENTAGES. No rho, no z-scores.

Every number here answers a question someone can act on:

  1. 每-rank 成本誤差 %   mean(|predicted - measured| / measured) over the 448 ranks of that
                          segment. "If I say rank r will burn X seconds, how far off am I?"
  2. 熱點命中率 %         of the 45 genuinely most expensive ranks (top 10%), how many are in
                          my predicted top 45. This is what rebalancing actually needs -- you
                          move the hot ranks, so finding them is the job.
  3. 總時間誤差 %         predicted vs measured wall time of the whole run.

Every case is evaluated LEAVE-ONE-CASE-OUT: the model never saw it. No measured quantity of
the evaluated case leaks into its own prediction (share is fixed at 1, not read back).

usage: wtune_accuracy.py [case ...]
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

SEG_HOURS = 3.0
FIT = dict(depth=3, n_est=80, lr=0.10)
TOPK = 45                                     # top 10% of 448 ranks


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
    smap = seg.sim_to_wall(f'{seg.BENCH}/{case}/rsl.out.0000')
    return dict(case=case, X=X, rate=cs / (SEG_HOURS * 3600.0 / dt), seg=sg,
                L=float(s['e_vert']), n_steps=len(smap), loop_wall=smap[-1][1], dt=dt)


def seg_mean(v, sg):
    out = np.empty_like(v)
    for k in np.unique(sg):
        out[sg == k] = v[sg == k].mean()
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
    shape = rate / np.concatenate([seg_mean(r['rate'], r['seg']) for r in rows])

    # stage 1: case cost rate from e_vert alone, leave-one-out
    mean_rate = np.array([r['rate'].mean() for r in rows])
    Lcol = np.array([[r['L']] for r in rows], float)
    rate_pred = loo_lin(Lcol, mean_rate)

    # stage 2: per-rank shape, leave-one-case-out
    sh_pred = np.empty_like(shape)
    for i in range(n):
        trm, tem = cid != i, cid == i
        sh_pred[tem] = tr.gbrt_pred(tr.gbrt_fit(X[trm], shape[trm], **FIT), X[tem])

    ks = sorted(np.unique(sgs))
    print("=" * 96)
    print("每個測資的分段成本預測準確度（全部留一法：被評估的 case 從未參與訓練）")
    print("=" * 96)
    print("\n【1】每-rank 成本誤差 %   —— 我說某個 rank 要花多少秒，平均差幾 %")
    print("%-18s %8s %8s %8s %8s | %8s" % ("測資", "第1段", "第2段", "第3段", "第4段", "全案"))
    err_all = []
    for i, r in enumerate(rows):
        tem = cid == i
        pred = rate_pred[i] * sh_pred[tem]           # share fixed at 1: no measured value used
        true, sgv = rate[tem], sgs[tem]
        cells = []
        for k in ks:
            m = sgv == k
            cells.append(100 * np.mean(np.abs(pred[m] - true[m]) / true[m]) if m.any() else np.nan)
        whole = 100 * np.mean(np.abs(pred - true) / true)
        err_all.append(whole)
        print("%-18s %7.1f%% %7.1f%% %7.1f%% %7.1f%% | %7.1f%%"
              % (r['case'], *cells, whole))
    print("%-18s %8s %8s %8s %8s | %7.1f%%  <- 平均" % ("", "", "", "", "", np.mean(err_all)))

    print("\n【2】熱點命中率 %   —— 真正最貴的 45 個 rank（前 10%），我找到幾個")
    print("%-18s %8s %8s %8s %8s | %8s" % ("測資", "第1段", "第2段", "第3段", "第4段", "全案"))
    hit_all = []
    for i, r in enumerate(rows):
        tem = cid == i
        pred = rate_pred[i] * sh_pred[tem]
        true, sgv = rate[tem], sgs[tem]
        cells = []
        for k in ks:
            m = np.where(sgv == k)[0]
            if len(m) < TOPK:
                cells.append(np.nan); continue
            ht = set(m[np.argsort(true[m])[-TOPK:]])
            hp = set(m[np.argsort(pred[m])[-TOPK:]])
            cells.append(100.0 * len(ht & hp) / TOPK)
        hit_all.append(np.nanmean(cells))
        print("%-18s %7.1f%% %7.1f%% %7.1f%% %7.1f%% | %7.1f%%"
              % (r['case'], *cells, hit_all[-1]))
    print("%-18s %8s %8s %8s %8s | %7.1f%%  <- 平均（亂猜是 10.0%%）"
          % ("", "", "", "", "", np.mean(hit_all)))

    print("\n【3】總執行時間誤差 %   —— 整個 run 要跑多久")
    print("%-18s %10s %10s %9s %9s" % ("測資", "實測(分)", "預測(分)", "誤差", "誤差%"))
    wall = np.array([r['loop_wall'] for r in rows])
    per_step = wall / np.array([r['n_steps'] for r in rows])
    ps_pred = loo_lin(Lcol, per_step)                    # ms/step from e_vert, leave-one-out
    tot_err = []
    for i, r in enumerate(rows):
        pw = ps_pred[i] * r['n_steps']
        tot_err.append(100 * abs(pw - wall[i]) / wall[i])
        print("%-18s %10.1f %10.1f %+8.1f %8.1f%%"
              % (r['case'], wall[i] / 60, pw / 60, (pw - wall[i]) / 60, tot_err[-1]))
    print("%-18s %10s %10s %9s %8.1f%%  <- 平均" % ("", "", "", "", np.mean(tot_err)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

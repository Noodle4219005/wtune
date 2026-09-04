#!/usr/bin/env python3
"""Case-level TOTAL-RUNTIME prediction for the 12 h campaign.

The user's operating constraint is "two random large weather cases inside ~10 h", so what has
to be predicted is a 40-80 minute run, not a 20 second one. The earlier runtime work
(MAE 1.07 s on 2 h runs) was measured at a scale where the answer barely mattered.

Decomposition, on purpose:

    total_wall  =  n_steps  x  wall_per_step  +  init

`n_steps = sim_hours * 3600 / dt` is EXACT ARITHMETIC from the namelist -- predicting it with
a model would be measuring our own ability to divide. So the model predicts only
`wall_per_step`, and the honest question is: does weather add anything on top of the purely
structural predictor (vertical levels)? Reported as three nested models under leave-one-out,
so "the features help" has to beat "levels alone", not just beat the mean.

Target comes from rsl.out.0000: the sum of every `Timing for` elapsed value, i.e. the solver
time loop including history writes -- the part that scales. Init is reported separately.

usage: wtune_runtime12.py [case ...]
"""
import sys, os, glob, re
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg

RESULTS = seg.RESULTS
BENCH = seg.BENCH


def case_row(case):
    rsl = f'{BENCH}/{case}/rsl.out.0000'
    feats = seg.load_features(f'{RESULTS}/{case}')
    if not os.path.exists(rsl) or not feats:
        return None
    smap = seg.sim_to_wall(rsl)
    if len(smap) < 100:
        return None
    n_steps = len(smap)
    loop_wall = smap[-1][1]
    s = seg.schemes(case)
    nl = f'{RESULTS}/{case}/namelist.input'
    txt = open(nl).read() if os.path.exists(nl) else ''
    dt = int(re.search(r'time_step=(\d+)', txt).group(1)) if re.search(r'time_step=(\d+)', txt) else -1
    X = np.vstack([m for _, m, _ in feats]).mean(axis=0)      # domain- and time-mean drivers
    return dict(case=case, n_steps=n_steps, loop_wall=loop_wall,
                wps=loop_wall / n_steps, e_vert=s['e_vert'], dt=dt,
                mp=s['mp'], bl=s['bl'], ra=s['ra'], X=X, names=feats[0][2])


def loo_linear(F, y):
    """Leave-one-out prediction from an ordinary least-squares fit on columns F (n, k)."""
    n = len(y)
    pred = np.empty(n)
    A = np.column_stack([np.ones(n)] + [F[:, j] for j in range(F.shape[1])]) if F.size else np.ones((n, 1))
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        pred[i] = A[i] @ coef
    return pred


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    want = sys.argv[1:] or sorted(os.path.basename(d) for d in glob.glob(f'{RESULTS}/*')
                                  if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
    rows = [r for r in (case_row(c) for c in want) if r]
    if len(rows) < 4:
        print("need >= 4 cases; got %d" % len(rows)); return 2
    names = rows[0]['names']
    print("%-18s %7s %9s %8s %6s %4s  %s" % ("case", "steps", "loop_s", "ms/step", "e_vert", "dt", "mp/bl/ra"))
    for r in rows:
        print("%-18s %7d %9.1f %8.2f %6d %4d  %d/%d/%d"
              % (r['case'], r['n_steps'], r['loop_wall'], 1000 * r['wps'], r['e_vert'], r['dt'],
                 r['mp'], r['bl'], r['ra']))

    y = np.array([r['wps'] for r in rows])
    lev = np.array([[r['e_vert']] for r in rows], float)
    Xf = np.array([r['X'] for r in rows])
    n = len(rows)

    print("\n=== 預測 wall_per_step（leave-one-out）===")
    base = np.array([np.mean(np.delete(y, i)) for i in range(n)])
    print("%-34s MAE %7.2f ms  (%.1f%%)" % ("0. 平均基準", 1000 * mae(base, y), 100 * mae(base, y) / y.mean()))
    pl = loo_linear(lev, y)
    print("%-34s MAE %7.2f ms  (%.1f%%)" % ("1. 只用 e_vert（純結構）", 1000 * mae(pl, y), 100 * mae(pl, y) / y.mean()))

    # 2. e_vert + the single weather feature that helps most, chosen INSIDE each fold
    pred = np.empty(n)
    picks = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        best, bj = None, -1
        for j in range(Xf.shape[1]):             # inner LOO on the training folds only
            F = np.column_stack([lev[:, 0], Xf[:, j]])
            inner = loo_linear(F[m], y[m])
            e = mae(inner, y[m])
            if best is None or e < best:
                best, bj = e, j
        picks.append(names[bj])
        F = np.column_stack([lev[:, 0], Xf[:, bj]])
        A = np.column_stack([np.ones(n), F[:, 0], F[:, 1]])
        coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        pred[i] = A[i] @ coef
    print("%-34s MAE %7.2f ms  (%.1f%%)" % ("2. e_vert + 天氣特徵（巢狀選擇）",
                                            1000 * mae(pred, y), 100 * mae(pred, y) / y.mean()))
    from collections import Counter
    print("   每折選中的特徵: %s" % dict(Counter(picks)))

    # 3. Pure NAMELIST model. Four cases share e_vert=45 yet span 444-653 ms/step (1.47x), so
    # levels cannot be the whole story -- and the ordering is WSM6 < Thompson < Morrison, i.e.
    # the microphysics scheme's own prognostic-variable count. No selection, no weather.
    mpv = np.array([[r['mp']] for r in rows], float)
    p3 = loo_linear(np.column_stack([lev[:, 0], mpv[:, 0]]), y)
    print("%-34s MAE %7.2f ms  (%.1f%%)" % ("3. e_vert + mp（純 namelist）",
                                            1000 * mae(p3, y), 100 * mae(p3, y) / y.mean()))
    blv = np.array([[r['bl']] for r in rows], float)
    p4 = loo_linear(np.column_stack([lev[:, 0], mpv[:, 0], blv[:, 0]]), y)
    print("%-34s MAE %7.2f ms  (%.1f%%)" % ("4. e_vert + mp + bl",
                                            1000 * mae(p4, y), 100 * mae(p4, y) / y.mean()))
    # levels multiply per-column physics work, so the interaction is the physical form
    p5 = loo_linear(np.column_stack([lev[:, 0] * mpv[:, 0]]), y)
    print("%-34s MAE %7.2f ms  (%.1f%%)" % ("5. e_vert x mp（交互項）",
                                            1000 * mae(p5, y), 100 * mae(p5, y) / y.mean()))
    best = min([(mae(p, y), nm, p) for p, nm in
                ((base, "mean"), (pl, "e_vert"), (pred, "e_vert+weather"),
                 (p3, "e_vert+mp"), (p4, "e_vert+mp+bl"), (p5, "e_vert*mp"))])
    print("   -> 最佳: %s" % best[1])
    pred = best[2]

    print("\n=== 換算成整段執行時間（n_steps 由 namelist 精確得出，不是預測出來的）===")
    print("%-18s %9s %9s %9s" % ("case", "實際 min", "模型 min", "誤差"))
    err = []
    for i, r in enumerate(rows):
        act = r['loop_wall'] / 60.0
        prd = pred[i] * r['n_steps'] / 60.0
        err.append(abs(prd - act))
        print("%-18s %9.1f %9.1f %+8.1f" % (r['case'], act, prd, prd - act))
    tot = np.array([r['loop_wall'] / 60.0 for r in rows])
    print("%-18s %9s %9s %8.1f min (%.1f%%)" % ("MAE", "", "", np.mean(err), 100 * np.mean(err) / tot.mean()))
    return 0


if __name__ == '__main__':
    sys.exit(main())

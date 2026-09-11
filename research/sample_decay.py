#!/usr/bin/env python3
"""How fast does a per-rank cost MEASUREMENT go stale, and can the model fill the gap?

This replaces an earlier design that was wrong. That one asked "do the first K buckets predict
the rest", which probes exactly ONE position -- the start of the run -- and so confounds probe
length with the fact that the start is special: spin-up transients, one particular solar
terminator position that never recurs, cold caches. The corrected design samples PERIODICALLY
across the whole run and scores every (source window, horizon) pair, which measures staleness
rather than startedness.

Three questions, in the order that decides the architecture:

  1. DECAY      a window of width W measured at t, scored against the top ranks at t+delta.
                Averaged over every t. Gives the re-sampling interval the controller needs.
  2. WIDTH      sweep W. Radiation fires every 60 steps (~36 s at 0.6 s/step) and the dominant
                imbalance in this model is the solar terminator, so a window narrower than one
                radiation call should mis-rank while looking well-behaved.
  3. FILL-IN    given a sample that is delta old, does the weather model IMPROVE on it?
                  A  stale sample alone            (persistence at lag delta)
                  B  model alone                   (GBRT on weather features)
                  C  model WITH the stale sample as an extra feature   <-- the hybrid
                If C beats both, the architecture is periodic cheap measurement plus
                model-based interpolation, not one or the other.

Needs bucket_load.csv written at fine resolution (BUCKET_SEC=10).

usage: sample_decay.py [case ...]      (default: every decay_* case)
"""
import sys, os, glob, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle_train as tr

RESULTS = '/home/u5727520/wrf-results'
BENCH = '/home/u5727520/wrf-bench'
TOPFRAC = 0.10
BSEC = 10.0
STAMP = re.compile(r'features_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})\.csv$')
FIT = dict(depth=3, n_est=80, lr=0.10)


def load_buckets(case):
    p = f'{RESULTS}/{case}/bucket_load.csv'
    if not os.path.exists(p):
        return None
    rows = np.genfromtxt(p, delimiter=',', names=True)
    nr = int(rows['rank'].max()) + 1
    nb = int(rows['bucket'].max()) + 1
    comp = np.zeros((nr, nb))
    for r, b, c in zip(rows['rank'], rows['bucket'], rows['comp_s']):
        comp[int(r), int(b)] = c
    used = comp.sum(axis=0) > 0
    if used.sum() < 20:
        return None
    # Two buckets are not real 10 s windows and must both go.
    #  * the LAST one, because the shim clamps everything past the end into it;
    #  * the FIRST one, because charge() only splits a span across ONE bucket edge -- a span
    #    crossing several edges is charged entirely to the bucket it started in. Steady-state
    #    steps are ~0.6 s so this never bites there, but initialisation is long and lands
    #    wholly in bucket 0 (observed: comp=19.99 s inside a 10 s bucket).
    last = int(np.where(used)[0].max())
    return comp[:, 1:last]


def hit(score, truth, topk):
    a = set(np.argsort(truth)[-topk:]); b = set(np.argsort(score)[-topk:])
    return 100.0 * len(a & b) / topk


def decay_table(comp, widths, lags, topk):
    """mean hit% over every source position t, for each (W, lag)."""
    nr, nb = comp.shape
    out = np.full((len(widths), len(lags)), np.nan)
    for iw, W in enumerate(widths):
        for il, L in enumerate(lags):
            vals = []
            t = 0
            while t + W + L + W <= nb:
                src = comp[:, t:t + W].sum(axis=1)
                tgt = comp[:, t + W + L:t + W + L + W].sum(axis=1)
                if src.sum() > 0 and tgt.sum() > 0:
                    vals.append(hit(src, tgt, topk))
                t += max(1, W)                    # non-overlapping source windows
            if vals:
                out[iw, il] = float(np.mean(vals))
    return out


def load_weather(case, nr):
    """-> list of (datetime, X, names) frames for the case, or []."""
    out = []
    for p in sorted(glob.glob(f'{RESULTS}/{case}/features_*.csv')):
        m = STAMP.search(p)
        if not m:
            continue
        y, mo, d, h, mi = map(int, m.groups())
        A = np.genfromtxt(p, delimiter=',', names=True)
        if A.shape[0] != nr:
            continue
        nm = list(A.dtype.names)
        out.append((datetime.datetime(y, mo, d, h, mi), np.column_stack([A[k] for k in nm]), nm))
    return out


def fill_in_test(case, comp, W, lags, topk):
    """Q3: does the model add anything to a stale sample, and vice versa?

    Trained WITHIN the case across source positions -- this asks whether the two information
    sources are complementary, not whether the model transfers across cases (already measured
    elsewhere at 48.3%).
    """
    nr, nb = comp.shape
    frames = load_weather(case, nr)
    if len(frames) < 2:
        return None
    # Map each bucket to the most recent weather frame. Frames are HISTH=2 h apart; the run is
    # 12 h of model time compressed into ~1730 s of wall, so bucket -> frame is a linear map.
    span = nb * BSEC
    t0 = frames[0][0]
    fsec = np.array([(f[0] - t0).total_seconds() for f in frames], float)
    fsec = fsec / max(fsec[-1], 1e-9) * span
    geo_i = (np.arange(nr) % 8) / 7.0
    geo_j = (np.arange(nr) // 8) / 55.0

    rows = []
    for L in lags:
        A_, B_, C_, D_ = [], [], [], []
        Xtr, ytr, Xte_all = [], [], []
        recs = []
        t = 0
        while t + W + L + W <= nb:
            src = comp[:, t:t + W].sum(axis=1)
            tgt = comp[:, t + W + L:t + W + L + W].sum(axis=1)
            if src.sum() > 0 and tgt.sum() > 0:
                fi = int(np.searchsorted(fsec, (t + W + L) * BSEC, side='right')) - 1
                fi = max(0, min(fi, len(frames) - 1))
                Xw = frames[fi][1]
                recs.append((src, tgt, Xw))
            t += max(1, W)
        if len(recs) < 4:
            continue
        # leave-one-position-out across source positions
        for k in range(len(recs)):
            trn = [r for j, r in enumerate(recs) if j != k]
            src, tgt, Xw = recs[k]
            def design(s, X, with_src):
                cols = [X, geo_i[:, None], geo_j[:, None]]
                if with_src:
                    cols.append((s / max(s.mean(), 1e-12))[:, None])
                return np.nan_to_num(np.hstack(cols))
            # D is the RESIDUAL arm. B and C ask the model to reproduce the whole cost field;
            # D asks it only for the CHANGE since the sample, with the sample added back:
            #     cost_hat(t+L) = sample + G(weather, sample)
            # The sample is already ~83% correct at short lag, so it is a strong identity path
            # -- and GBRT is exactly the model class that cannot represent identity cheaply,
            # since piecewise-constant trees need many splits to approximate y = x. Feeding the
            # sample as one feature among twenty (arm C) makes the model rediscover that
            # mapping; predicting the residual makes it free.
            for tag, with_src, residual, acc in (('B', False, False, B_),
                                                 ('C', True, False, C_),
                                                 ('D', True, True, D_)):
                Xt = np.vstack([design(s, X, with_src) for s, _g, X in trn])
                if residual:
                    yt = np.concatenate([g / g.mean() - s / s.mean() for s, g, _X in trn])
                else:
                    yt = np.concatenate([g / g.mean() for _s, g, _X in trn])
                mdl = tr.gbrt_fit(Xt, yt, **FIT)
                p = tr.gbrt_pred(mdl, design(src, Xw, with_src))
                if residual:
                    p = src / max(src.mean(), 1e-12) + p
                acc.append(hit(p, tgt, topk))
            A_.append(hit(src, tgt, topk))
        rows.append((L, float(np.mean(A_)), float(np.mean(B_)),
                     float(np.mean(C_)), float(np.mean(D_))))
    return rows


def main():
    cases = sys.argv[1:] or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(f'{RESULTS}/decay_*/bucket_load.csv'))
    mats = {}
    for c in cases:
        m = load_buckets(c)
        if m is not None:
            mats[c] = m
    if not mats:
        print("no usable fine-bucket case (need BUCKET_SEC=10 runs)"); return 2

    WIDTHS = [1, 2, 4, 6, 12]
    LAGS = [0, 2, 6, 12, 24, 48, 96]
    print("bucket = %.0f s of wall ~= %d steps at 0.6 s/step" % (BSEC, int(BSEC / 0.6)))
    print("radiation every 60 steps ~= 36 s ~= %.1f buckets" % (36.0 / BSEC))
    print("random baseline = %.0f%%\n" % (100 * TOPFRAC))

    for c, comp in sorted(mats.items()):
        nr, nb = comp.shape
        topk = max(1, int(round(TOPFRAC * nr)))
        print("=== %s : %d ranks, %d buckets (%.0f s of wall) ===" % (c, nr, nb, nb * BSEC))
        T = decay_table(comp, WIDTHS, LAGS, topk)
        print("%-10s %s" % ("width\\lag", "  ".join("%5d" % (l * BSEC) for l in LAGS)))
        print("%-10s %s" % ("(seconds)", "  ".join("%5s" % "" for _ in LAGS)))
        for iw, W in enumerate(WIDTHS):
            print("W=%-3d(%3ds) %s" % (W, int(W * BSEC),
                  "  ".join("%5.1f" % v if np.isfinite(v) else "    -" for v in T[iw])))
        print()

    print("################ FILL-IN: can the model improve a stale sample? ################")
    print("%-20s %6s  %8s %8s %8s %10s   %s"
          % ("case", "lag_s", "A stale", "B model", "C both", "D residual", "best"))
    for c, comp in sorted(mats.items()):
        nr, _ = comp.shape
        topk = max(1, int(round(TOPFRAC * nr)))
        rows = fill_in_test(c, comp, W=6, lags=[6, 24, 96], topk=topk)
        if not rows:
            print("%-20s (no usable weather frames)" % c); continue
        for L, a, b, cc, d in rows:
            names = ['A stale', 'B model', 'C both', 'D residual']
            vals = [a, b, cc, d]
            best = names[int(np.argmax(vals))]
            print("%-20s %6d  %8.1f %8.1f %8.1f %10.1f   %s"
                  % (c, int(L * BSEC), a, b, cc, d, best), flush=True)
    print()
    print("READING")
    print("  decay:   how far right the row stays high = how long a measurement stays valid,")
    print("           i.e. the controller's required re-sampling interval.")
    print("  width:   a jump between W=4 (40 s) and below confirms the radiation-interval floor.")
    print("  fill-in: C or D > max(A,B) means periodic measurement + model interpolation beats")
    print("           both pure prediction and pure probing -- that is then the architecture.")
    print("  D > C:   confirms the residual framing matters, i.e. the win comes from giving the")
    print("           model a free identity path rather than from the sample's information.")
    print("           D ~= C would refute that and mean GBRT absorbs the sample fine as a feature.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

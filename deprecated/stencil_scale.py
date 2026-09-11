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
"""Is the RECEPTIVE FIELD the binding constraint on per-rank cost prediction?

Per-rank cost is dense prediction on a 2D field: the input is a multi-channel image (17
physical fields sampled onto the PY x PX rank grid) and the output is one scalar per rank.
Seen that way, the production model is a per-pixel regressor plus exactly ONE hand-made
convolution: `_nbr_stats` (wtune_segment_train.py:119-158) is a 4-neighbour cross kernel
returning mean and max, applied to 3 of the 17 fields.

Two facts say the spatial axis is under-exploited:
  * that single hand-made stencil helped, and
  * the geometry features (geo_i, geo_j) are an ABSOLUTE-COORDINATE lookup, which is exactly
    why they failed to transfer across decompositions (+4.4 at 8x56, -0.6 at 14x64).

So before anyone builds a CNN, this asks the cheap version of the question: widen the
receptive field with multi-scale box statistics (r=1,2,4 -> 3x3, 5x5, 9x9) plus a gradient
magnitude, and see whether cross-case transfer improves. Costs zero cluster time -- it reuses
the features_*.csv already on disk.

usage: stencil_scale.py [case ...]        (default: every case with a bucket_load.csv)
"""
import sys, os, glob, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

SEG_HOURS = 3.0
FIT = dict(depth=3, n_est=80, lr=0.10)
# _best_split (wtune_oracle_train.py:64) scans ncol x 9 thresholds per node, so runtime is
# linear in BOTH rows and columns. 'wideall' carries ~139 columns against 'cur's 26, and full
# LOCO over every case would run for hours. Cap rows and folds instead -- both are applied
# IDENTICALLY to every feature set with a fixed seed, so the comparison between sets stays
# fair even though each absolute number is noisier than a full fit would give.
MAXROWS = int(os.environ.get('WTUNE_MAXROWS', '8000'))
FOLDS = int(os.environ.get('WTUNE_FOLDS', '10'))
SEED = 20260905
NBR_OF = ('radiation', 'micro', 'surf')          # the 3 fields the production stencil uses
SCALES = (1, 2, 4)                                # box radii -> 3x3, 5x5, 9x9
STAMP = re.compile(r'features_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})\.csv$')


def box_stats(v, r, PX, PY):
    """Mean and std over a (2r+1)^2 box, edges clamped (not wrapped: the domain is not periodic).

    448 cells and at most 81 offsets, so the naive gather is far cheaper than an integral image
    and has no accumulation-order subtleties.
    """
    g = v.reshape(PY, PX)
    J, I = np.mgrid[0:PY, 0:PX]
    acc = np.zeros_like(g); acc2 = np.zeros_like(g); n = 0
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            s = g[np.clip(J + dj, 0, PY - 1), np.clip(I + di, 0, PX - 1)]
            acc += s; acc2 += s * s; n += 1
    m = acc / n
    return m.ravel(), np.sqrt(np.maximum(acc2 / n - m * m, 0.0)).ravel()


def grad_mag(v, PX, PY):
    g = v.reshape(PY, PX)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy).ravel()


def nbr_cross(v, PX, PY):
    """The production stencil, reproduced here so the comparison is like-for-like."""
    g = v.reshape(PY, PX)
    acc = np.zeros_like(g); cnt = np.zeros_like(g); mx = np.full_like(g, -np.inf)
    for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sj = slice(max(0, dj), PY + min(0, dj)); si = slice(max(0, di), PX + min(0, di))
        tj = slice(max(0, -dj), PY + min(0, -dj)); ti = slice(max(0, -di), PX + min(0, -di))
        acc[tj, ti] += g[sj, si]; cnt[tj, ti] += 1
        mx[tj, ti] = np.maximum(mx[tj, ti], g[sj, si])
    cnt[cnt == 0] = 1
    mx[~np.isfinite(mx)] = 0.0
    return (acc / cnt).ravel(), mx.ravel()


def base_of(case):
    """The WEATHER a case directory belongs to -- resolved from PROVENANCE, not from the name.

    The _14x64 decomposition variants have no geo_em of their own; their met_em are symlinks
    into the parent case, and following that link is ground truth. A name-only rule gets this
    wrong in a way that leaks: karina_14x64 strips to 'karina', but its met_em points at
    karina_cat4 -- so the same storm landed in two different folds and each leaked into the
    other's training set.

        karina_14x64          -> karina_cat4   (via symlink, NOT 'karina')
        atmos_river_14x64     -> atmos_river   (via symlink)
        deep_conv_ot__mp10bl2 -> deep_conv_ot  (scheme suffix)
    """
    for p in sorted(glob.glob(os.path.join(seg.BENCH, case, 'met_em.d01.*')))[:1]:
        parent = os.path.basename(os.path.dirname(os.path.realpath(p)))
        if parent and parent != case:
            case = parent
    return re.sub(r'_\d+x\d+$', '', case.split('__')[0])


def decomp_of(case):
    for p in (f'{seg.RESULTS}/{case}/namelist.input', f'{seg.BENCH}/{case}/namelist.input'):
        if os.path.exists(p):
            t = open(p).read()
            gx = re.search(r'nproc_x=(\d+)', t); gy = re.search(r'nproc_y=(\d+)', t)
            if gx and gy:
                return int(gx.group(1)), int(gy.group(1))
    return 8, 56


def raw_case(case):
    """-> list of (Xraw, names, rate) per segment, plus PX, PY. No stencils applied yet."""
    PX, PY = decomp_of(case)
    NR = PX * PY
    out = f'{seg.RESULTS}/{case}'
    bk, rsl = f'{out}/bucket_load.csv', f'{seg.BENCH}/{case}/rsl.out.0000'
    if not (os.path.exists(bk) and os.path.exists(rsl)):
        return None
    rows = np.genfromtxt(bk, delimiter=',', names=True)
    nb = int(rows['bucket'].max()) + 1
    # Bucket width is NOT always 300 s. Diagnostic runs use BUCKET_SEC=10, and hardcoding 300
    # silently mis-maps their segment windows, producing garbage rates that then poison every
    # model trained on a corpus glob. Derive it from t_start_s, which the shim writes.
    bsec = 300.0
    if 't_start_s' in rows.dtype.names:
        ts = np.unique(rows['t_start_s'])
        if len(ts) > 1:
            d = np.diff(np.sort(ts))
            d = d[d > 0]
            if len(d):
                bsec = float(np.median(d))
    comp = np.zeros((NR, nb)); wait = np.zeros((NR, nb))
    for r, b, c, w in zip(rows['rank'], rows['bucket'], rows['comp_s'], rows['wait_s']):
        if int(r) < NR:
            comp[int(r), int(b)] = c; wait[int(r), int(b)] = w
    smap = seg.sim_to_wall(rsl)
    if not smap:
        return None
    sim = np.array([s for s, _ in smap]); wall = np.array([w for _, w in smap])
    span = float((comp + wait).sum(axis=1).max())
    head = float(np.clip(span - wall[-1], 0.0, 0.5 * span))

    def w_of(t):
        i = int(np.searchsorted(sim, t))
        return head + (0.0 if i <= 0 else wall[min(i, len(wall) - 1)])

    feats = []
    for p in sorted(glob.glob(os.path.join(out, 'features_*.csv'))):
        m = STAMP.search(p)
        if not m:
            continue
        y, mo, d, h, mi = map(int, m.groups())
        A = np.genfromtxt(p, delimiter=',', names=True)
        nm = list(A.dtype.names)
        if A.shape[0] != NR:
            continue
        feats.append((datetime.datetime(y, mo, d, h, mi), np.column_stack([A[k] for k in nm]), nm))
    if len(feats) < 2:
        return None
    nl = f'{seg.RESULTS}/{case}/namelist.input'
    t = open(nl).read() if os.path.exists(nl) else ''
    mm = re.search(r'time_step=(\d+)', t)
    dt = int(mm.group(1)) if mm else 15
    segs = []
    for k in range(len(feats) - 1):
        t0, Xb, nm = feats[k]
        edges = np.arange(nb + 1) * bsec
        w0, w1 = w_of(t0), w_of(feats[k + 1][0])
        ov = np.clip(np.minimum(edges[1:], w1) - np.maximum(edges[:-1], w0), 0.0, None) / 300.0
        if ov.sum() <= 0:
            continue
        c = comp @ ov
        c[0] = np.median(c)                      # rank 0 also carries I/O; not a physics cost
        segs.append((Xb, nm, c / (SEG_HOURS * 3600.0 / dt)))
    return (segs, PX, PY) if segs else None


def make_features(Xb, names, PX, PY, mode):
    """Assemble one segment's design matrix under the named feature set."""
    NR = PX * PY
    I = np.arange(NR) % PX; J = np.arange(NR) // PX
    nnb = ((I > 0).astype(float) + (I < PX - 1) + (J > 0) + (J < PY - 1))
    geo = np.column_stack([I / (PX - 1.0), J / (PY - 1.0), nnb])
    cols = [Xb]
    if mode in ('cur', 'wide3', 'wideall'):
        cols.append(geo)
    if mode in ('cur', 'cur_nogeo'):
        for k in NBR_OF:
            if k in names:
                m, mx = nbr_cross(Xb[:, names.index(k)], PX, PY)
                cols.append(np.column_stack([m, mx]))
    if mode in ('wide3', 'wide3_nogeo', 'wideall', 'wideall_nogeo'):
        fields = NBR_OF if mode.startswith('wide3') else tuple(names)
        for k in fields:
            if k not in names:
                continue
            v = Xb[:, names.index(k)]
            per = []
            for r in SCALES:
                m, sd = box_stats(v, r, PX, PY)
                per += [m, sd]
            per.append(grad_mag(v, PX, PY))
            cols.append(np.column_stack(per))
    # A NaN anywhere would propagate silently through _best_split's means and poison the
    # comparison rather than fail, so clamp here. Absent fields are legitimately all-zero.
    return np.nan_to_num(np.hstack(cols), nan=0.0, posinf=0.0, neginf=0.0)


def hit(score, true, topk):
    a = set(np.argsort(true)[-topk:]); b = set(np.argsort(score)[-topk:])
    return 100.0 * len(a & b) / topk


def main():
    argv = sys.argv[1:]
    cases = argv or sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(f'{seg.RESULTS}/*/bucket_load.csv'))
    built = {}
    for c in cases:
        b = raw_case(c)
        if b:
            built[c] = b
    if len(built) < 3:
        print("need at least 3 usable cases, got %d" % len(built)); return 2
    # One common feature schema. Filter PER SEGMENT against the MODAL schema, not per case
    # against the first case's: the corpus still holds 11-column CSVs from before QKE/EL_PBL
    # were always emitted, and a case can mix widths across its own segments. Checking only
    # segs[0] let those through and vstack failed with a bare shape error.
    from collections import Counter
    cnt = Counter()
    for c, (segs, _, _) in built.items():
        for _Xb, nm, _r in segs:
            cnt[tuple(nm)] += 1
    schema = list(cnt.most_common(1)[0][0])
    kept = {}
    dropped_seg = dropped_case = 0
    for c, (segs, PX, PY) in built.items():
        keep = [s for s in segs if list(s[1]) == schema]
        dropped_seg += len(segs) - len(keep)
        if keep:
            kept[c] = (keep, PX, PY)
        else:
            dropped_case += 1
    built = kept
    print("cases=%d (dropped %d)  segments dropped=%d  fields=%d  scales=%s"
          % (len(built), dropped_case, dropped_seg, len(schema), SCALES))
    if len(built) < 3:
        print("need at least 3 usable cases after schema filtering"); return 2

    # LEAVE-ONE-*WEATHER*-OUT, not leave-one-directory-out. 36 of the 56 corpus entries are
    # scheme variants (__mpXblY) or decomposition variants (_14x64) of the SAME weather case.
    # Holding out deep_conv_ot__mp10bl1 while eight deep_conv_ot siblings stay in training is
    # within-case evaluation wearing a cross-case label -- and plain sorted()[:10] made SEVEN of
    # ten folds deep_conv_ot variants. The wiki already recorded this exact artifact once (the
    # per-bl head that scored +18.1 on a karina-dominated pool and -3.7 on a diverse one).
    groups = {}
    for c in built:
        groups.setdefault(base_of(c), []).append(c)
    fold_bases = sorted(groups)
    if FOLDS > 0:
        fold_bases = fold_bases[:FOLDS]
    print("weather groups=%d (from %d dirs); folds=%d  maxrows=%d  seed=%d"
          % (len(groups), len(built), len(fold_bases), MAXROWS, SEED))
    print("  held-out weather: %s" % ", ".join(fold_bases))
    MODES = ['own', 'cur', 'cur_nogeo', 'wide3', 'wide3_nogeo', 'wideall', 'wideall_nogeo']
    print("%-16s %8s %10s %12s" % ("feature set", "ncol", "hit%", "cost err%"))
    base_hit = None
    for mode in MODES:
        hits, errs, ncol = [], [], 0
        for held_base in fold_bases:                        # leave-one-WEATHER-out
            held_cases = set(groups[held_base])
            Xtr, ytr = [], []
            for c, (segs, PX, PY) in built.items():
                if c in held_cases:                         # every sibling variant held out too
                    continue
                for Xb, nm, rate in segs:
                    Xtr.append(make_features(Xb, nm, PX, PY, mode))
                    ytr.append(rate / rate.mean())
            if not Xtr:
                continue
            Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
            ncol = Xtr.shape[1]
            if MAXROWS and len(ytr) > MAXROWS:
                # seeded on the fold, not the mode, so every feature set trains on the SAME rows
                # fold INDEX, not hash(str): PYTHONHASHSEED randomisation would make the
                # subsample differ between runs and silently break reproducibility.
                idx = np.random.RandomState(SEED + fold_bases.index(held_base)).choice(
                    len(ytr), MAXROWS, replace=False)
                Xtr, ytr = Xtr[idx], ytr[idx]
            mdl = tr.gbrt_fit(Xtr, ytr, **FIT)
            for c in groups[held_base]:
                segs, PX, PY = built[c]
                topk = max(1, int(round(0.10 * PX * PY)))
                for Xb, nm, rate in segs:
                    p = tr.gbrt_pred(mdl, make_features(Xb, nm, PX, PY, mode))
                    hits.append(hit(p, rate, topk))
                    truth = rate / rate.mean()
                    errs.append(100.0 * np.median(np.abs(p - truth) / np.maximum(truth, 1e-9)))
        h, e = float(np.mean(hits)), float(np.mean(errs))
        if base_hit is None and mode == 'cur':
            base_hit = h
        mark = ""
        if base_hit is not None and mode != 'cur':
            mark = "  (%+.1f vs cur)" % (h - base_hit)
        print("%-16s %8d %9.1f%% %11.1f%%%s" % (mode, ncol, h, e, mark), flush=True)
    print()
    print("random baseline for hit%% = 10.0 (top-10%% of ranks)")
    print("PASS: a wide* set beats 'cur' by >3.0 points of hit rate -> receptive field was the")
    print("      binding constraint and a convolutional model is worth building.")
    print("FAIL: within +/-3.0 -> spatial context is already saturated by the 4-neighbour cross;")
    print("      the transfer gap is NOT a receptive-field problem and a CNN will not fix it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

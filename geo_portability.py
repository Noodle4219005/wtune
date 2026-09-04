#!/usr/bin/env python3
"""Does the geometry prior survive a different decomposition, or is it fitted to 8x56?

This is the largest unquantified risk in the per-rank model. Three geometry features --
rank i, rank j, halo neighbour count -- carry roughly half the signal: on their own, with no
weather at all, they score +0.380 Spearman and ~58% hot-spot hit rate. They are also
IDENTICAL in every case seen so far, because every case ran at 8x56 on 4 nodes.

If that prior is a real property of a halo exchange (edge ranks have fewer neighbours, and
row index tracks a latitude band), it should transfer to 14x64 on 8 nodes. If it is a lookup
table fitted to one grid, it will not -- and half the reported accuracy is an artefact that
evaporates the moment the controller changes node count, which is the very thing the
controller exists to do.

Both features are already scale-free by construction: geo_i/(PX-1) and geo_j/(PY-1) lie in
[0,1] either way, and the neighbour count is 2/3/4 in both. So the comparison is clean --
what differs is meaning, not range.

Self-contained on purpose: wtune_segment_train fixes PX/PY at import time, and this needs two
decompositions in one process.

usage: geo_portability.py <train-case>... -- <test-case>
"""
import sys, os, glob, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_segment_train as seg
import wtune_oracle_train as tr

SEG_HOURS = 3.0
FIT = dict(depth=3, n_est=80, lr=0.10)
NBR_OF = ('radiation', 'micro', 'surf')
STAMP = re.compile(r'features_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})\.csv$')


def decomp_of(case):
    """(PX, PY) straight from the namelist the runner archived beside the results."""
    for p in (f'{seg.RESULTS}/{case}/namelist.input', f'{seg.BENCH}/{case}/namelist.input'):
        if os.path.exists(p):
            t = open(p).read()
            gx = re.search(r'nproc_x=(\d+)', t)
            gy = re.search(r'nproc_y=(\d+)', t)
            if gx and gy:
                return int(gx.group(1)), int(gy.group(1))
    return 8, 56


def nbr_stats(v, PX, PY):
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


def build(case):
    """-> X (with geometry LAST), shape target, absolute rate, segment ids, PX, PY."""
    PX, PY = decomp_of(case)
    NR = PX * PY
    out = f'{seg.RESULTS}/{case}'
    bk = f'{out}/bucket_load.csv'
    rsl = f'{seg.BENCH}/{case}/rsl.out.0000'
    if not (os.path.exists(bk) and os.path.exists(rsl)):
        return None
    rows = np.genfromtxt(bk, delimiter=',', names=True)
    nb = int(rows['bucket'].max()) + 1
    comp = np.zeros((NR, nb)); wait = np.zeros((NR, nb))
    for r, b, c, w in zip(rows['rank'], rows['bucket'], rows['comp_s'], rows['wait_s']):
        if int(r) < NR:
            comp[int(r), int(b)] = c; wait[int(r), int(b)] = w
    bsec = 300.0
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
        feats.append((datetime.datetime(y, mo, d, h, mi), np.column_stack([A[k] for k in nm]), nm))
    if len(feats) < 2:
        return None
    i_ = np.arange(NR) % PX; j_ = np.arange(NR) // PX
    nnb = ((i_ > 0).astype(float) + (i_ < PX - 1) + (j_ > 0) + (j_ < PY - 1))
    Xs, ys, rs, sg, names = [], [], [], [], None
    nl = f'{seg.RESULTS}/{case}/namelist.input'
    t = open(nl).read() if os.path.exists(nl) else ''
    mm = re.search(r'time_step=(\d+)', t)
    dt = int(mm.group(1)) if mm else 15
    for k in range(len(feats) - 1):
        t0, Xb, nm = feats[k]
        edges = np.arange(nb + 1) * bsec
        w0, w1 = w_of(t0), w_of(feats[k + 1][0])
        ov = np.clip(np.minimum(edges[1:], w1) - np.maximum(edges[:-1], w0), 0.0, None) / bsec
        if ov.sum() <= 0:
            continue
        c = comp @ ov
        c[0] = np.median(c)
        rate = c / (SEG_HOURS * 3600.0 / dt)
        cols = [Xb]
        ex = ['geo_i', 'geo_j', 'geo_nnb']
        cols.append(np.column_stack([i_ / (PX - 1.0), j_ / (PY - 1.0), nnb]))
        for q in NBR_OF:
            if q in nm:
                a, b2 = nbr_stats(Xb[:, nm.index(q)], PX, PY)
                cols.append(np.column_stack([a, b2])); ex += ['nbrmean_' + q, 'nbrmax_' + q]
        Xs.append(np.hstack(cols)); ys.append(rate / rate.mean()); rs.append(rate)
        sg.append(np.full(NR, k)); names = nm + ex
    if not Xs:
        return None
    return (np.vstack(Xs), np.concatenate(ys), np.concatenate(rs), np.concatenate(sg),
            names, PX, PY)


def hit(score, true, sgv, topk):
    o = []
    for k in np.unique(sgv):
        ix = np.where(sgv == k)[0]
        if len(ix) < topk:
            continue
        a = set(ix[np.argsort(true[ix])[-topk:]]); b = set(ix[np.argsort(score[ix])[-topk:]])
        o.append(100.0 * len(a & b) / topk)
    return float(np.mean(o)) if o else float('nan')


def main():
    argv = sys.argv[1:]
    cut = argv.index('--')
    train, test = argv[:cut], argv[cut + 1:]
    TR = [b for b in (build(c) for c in train) if b]
    TE = [(c, build(c)) for c in test]
    TE = [(c, b) for c, b in TE if b]
    if not TR or not TE:
        print("need at least one usable train and test case"); return 2
    names = TR[0][4]
    TR = [b for b in TR if b[4] == names]
    # Per-case feature standardisation. to_ranks takes a MEAN per patch, so the LEVEL is
    # comparable across decompositions -- but the SPREAD is not: a 14x64 patch averages over
    # ~14x64 cells where an 8x56 patch averages ~16x112, so the smaller patches produce a
    # systematically wider feature distribution. That mismatch, not the rank geometry, is the
    # obvious suspect for cross-decomposition transfer collapsing to 21.7%.
    STD = os.environ.get('WTUNE_STD_FEATS') == '1'
    def z(A):
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        return (A - mu) / sd
    X = np.vstack([z(b[0]) if STD else b[0] for b in TR])
    Y = np.concatenate([b[1] for b in TR])
    if STD:
        print("(特徵已在每個 case 內標準化)")
    gi = [i for i, n in enumerate(names) if n.startswith('geo_')]
    nogeo = [i for i in range(len(names)) if i not in gi]
    print("訓練: %d 個 case，%s，分解 %dx%d" % (len(TR), X.shape, TR[0][5], TR[0][6]))
    print("%-22s %6s %10s %12s %12s %8s" % ("測試 case", "PXxPY", "ranks", "含幾何", "不含幾何", "幾何貢獻"))
    for c, b in TE:
        Xt, yt, rt, st, nt, px, py = b
        if STD:
            Xt = z(Xt)
        if nt != names:
            print("%-22s 特徵集不同，跳過" % c); continue
        topk = max(1, int(round(0.10 * px * py)))
        a = hit(tr.gbrt_pred(tr.gbrt_fit(X, Y, **FIT), Xt), rt, st, topk)
        d = hit(tr.gbrt_pred(tr.gbrt_fit(X[:, nogeo], Y, **FIT), Xt[:, nogeo]), rt, st, topk)
        print("%-22s %3dx%-3d %9d %11.1f%% %11.1f%% %+7.1f" % (c, px, py, px * py, a, d, a - d))
    return 0


if __name__ == '__main__':
    sys.exit(main())

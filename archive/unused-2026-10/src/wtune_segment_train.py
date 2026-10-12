#!/usr/bin/env python3
"""Train wtune's per-rank cost model on SEGMENTED features + SEGMENTED labels.

Why segmented at all: over a 12 h run the solar terminator sweeps 180 degrees, so the per-rank
hot-spot pattern inverts mid-run. A whole-run label is the mean of two opposite patterns, and a
controller that acts on it acts on something that was never true at any instant. So every
2-hour window gets its own feature frame (wrfout) and its own label (5-minute wall buckets from
blockbar_bucket), and the model is trained on (window features -> window cost).

The join is the delicate part: features are indexed by SIMULATION time, labels by WALL time.
rsl.out.0000 carries both -- every "Timing for ..." line gives a sim timestamp and the wall
seconds that step/write consumed -- so a cumulative sum of those elapsed values is an exact
sim->wall map. The one thing it cannot see is initialisation before the first step, which is
recovered as (total bucket wall) - (sum of all timed wall), i.e. the unaccounted head.

usage: wtune_segment_train.py [case ...]      (default: every case with both inputs present)
"""
import sys, os, glob, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle_train as tr

RESULTS = '/home/u5727520/wrf-results'
BENCH = '/home/u5727520/wrf-bench'
# Decomposition is a PARAMETER, not a constant: the geometry features (geo_i/geo_j/geo_nnb)
# carry roughly half the per-rank signal, and whether that prior survives a different
# nproc_x x nproc_y is unverified. Testing it requires reading a 14x64 run with the same code.
PX = int(os.environ.get('WTUNE_PX', 8))
PY = int(os.environ.get('WTUNE_PY', 56))
NR_ = PX * PY
NR = PX * PY
tr.PX, tr.PY = PX, PY

TIMING = re.compile(r'Timing for (\S+).*?time (\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})?.*?:\s+([0-9.]+) elapsed')
WROUT = re.compile(r'Timing for Writing wrfout_d01_(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}).*?:\s+([0-9.]+) elapsed')
MAIN = re.compile(r'Timing for main: time (\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}) on domain\s+1:\s+([0-9.]+) elapsed')
FEATSTAMP = re.compile(r'features_(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})\.csv$')


def _dt(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d_%H:%M:%S')


def sim_to_wall(rsl_path):
    """-> (sim_datetime, cumulative wall seconds at the END of that step), ascending.

    ONLY `Timing for main` lines. WRF performs the history write INSIDE the time step
    (med_last_solve_io is called from the integrate loop), so `Timing for Writing wrfout`
    is already contained in that step's main timing -- adding it double counts. Same for
    `Timing for processing lateral boundary`, a sub-phase of the step.

    Verified on karina_cat4: main sum 1300.5 s + pre-loop init 17.2 s = 1317.7 s, which is
    exactly the per-rank total the bucket profiler measured. Adding the 187.9 s of write
    timings gave 1488.4 s -- 13% too long -- which stretched every window and pushed the last
    segment half-way past the end of the data, making segment 3 look ~2x cheap in ALL ten
    cases. A systematic effect identical across ten different latitudes and seasons is the
    signature of a harness bug, not weather.
    """
    out, cum = [], 0.0
    with open(rsl_path, errors='ignore') as fh:
        for ln in fh:
            if 'Timing for main' not in ln:
                continue
            m = MAIN.search(ln)
            if m:
                cum += float(m.group(2))
                out.append((_dt(m.group(1)), cum))
    return out


def load_buckets(path):
    """-> comp[NR, NB], wait[NR, NB], bucket_sec."""
    rows = np.genfromtxt(path, delimiter=',', names=True)
    nb = int(rows['bucket'].max()) + 1
    comp = np.zeros((NR, nb))
    wait = np.zeros((NR, nb))
    for r, b, c, w in zip(rows['rank'], rows['bucket'], rows['comp_s'], rows['wait_s']):
        comp[int(r), int(b)] = c
        wait[int(r), int(b)] = w
    tstart = rows['t_start_s']
    bsec = float(np.median(np.diff(np.unique(tstart)))) if np.unique(tstart).size > 1 else 300.0
    return comp, wait, bsec


def window_label(comp, wait, bsec, w0, w1):
    """-> (comp_frac, comp_seconds) per rank over wall window [w0, w1).

    Two targets, because they are not the same question:
      comp_frac = comp/(comp+wait) is what earlier work used, but it conflates "I am slow"
                  with "my neighbours are fast" -- a rank's wait is set by someone else.
      comp_s    = absolute compute seconds, which is what a load balancer actually moves.
    """
    nb = comp.shape[1]
    edges = np.arange(nb + 1) * bsec
    ov = np.clip(np.minimum(edges[1:], w1) - np.maximum(edges[:-1], w0), 0.0, None)
    frac = ov / bsec
    if frac.sum() <= 0:
        return None
    c = comp @ frac
    w = wait @ frac
    tot = c + w
    good = tot > 0
    if good.sum() < NR * 0.9:
        return None
    y = np.where(good, c / np.where(good, tot, 1.0), np.nan)
    y[0] = np.nanmedian(y)                       # rank 0 is the serial-I/O outlier
    cs = c.copy()
    cs[0] = np.median(cs)
    return (np.nan_to_num(y, nan=float(np.nanmedian(y))), cs)


# --- decomposition geometry: fixed by nproc_x/nproc_y, identical in every case ---
_R = np.arange(NR)
_I = _R % PX                                     # WRF rank order is row-major in x
_J = _R // PX
_NNB = ((_I > 0).astype(float) + (_I < PX - 1) + (_J > 0) + (_J < PY - 1))


def _nbr_stats(v):
    """Mean and max of v over each rank's 4-neighbourhood (excluding itself)."""
    g = v.reshape(PY, PX)
    acc = np.zeros_like(g)
    cnt = np.zeros_like(g)
    mx = np.full_like(g, -np.inf)
    for dj, di in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        sl_j = slice(max(0, dj), PY + min(0, dj))
        sl_i = slice(max(0, di), PX + min(0, di))
        tj = slice(max(0, -dj), PY + min(0, -dj))
        ti = slice(max(0, -di), PX + min(0, -di))
        acc[tj, ti] += g[sl_j, sl_i]
        cnt[tj, ti] += 1
        mx[tj, ti] = np.maximum(mx[tj, ti], g[sl_j, sl_i])
    cnt[cnt == 0] = 1
    mx[~np.isfinite(mx)] = 0.0
    return (acc / cnt).ravel(), mx.ravel()


NBR_OF = ('radiation', 'micro', 'surf')


def augment_frame(X, names):
    """Add halo geometry and neighbourhood aggregates.

    A rank's WAIT is produced by its neighbours, not by itself, and edge/corner ranks exchange
    with 3/2 neighbours instead of 4 -- neither fact is expressible in a feature vector that
    only describes the rank's own column of atmosphere.
    """
    cols = [X]
    extra = ['geo_i', 'geo_j', 'geo_nnb']
    cols.append(np.column_stack([_I / (PX - 1.0), _J / (PY - 1.0), _NNB]))
    for k in NBR_OF:
        if k not in names:
            continue
        v = X[:, names.index(k)]
        m, mx = _nbr_stats(v)
        cols.append(np.column_stack([m, mx]))
        extra += ['nbrmean_' + k, 'nbrmax_' + k]
    return np.hstack(cols), list(names) + extra


def load_features(out_dir):
    got = []
    for p in sorted(glob.glob(os.path.join(out_dir, 'features_*.csv'))):
        m = FEATSTAMP.search(p)
        if not m:
            continue
        y, mo, d, h, mi = map(int, m.groups())
        X = np.genfromtxt(p, delimiter=',', names=True)
        names = list(X.dtype.names)
        M = np.column_stack([X[n] for n in names])
        got.append((datetime.datetime(y, mo, d, h, mi), M, names))
    return got


def schemes(case):
    """mp / bl / ra from the namelist the runner archived next to the results."""
    for p in (f'{RESULTS}/{case}/namelist.input', f'{BENCH}/{case}/namelist.input'):
        if os.path.exists(p):
            txt = open(p).read()
            g = lambda k: int(re.search(k + r'=(\d+)', txt).group(1)) if re.search(k + r'=(\d+)', txt) else -1
            return {'mp': g('mp_physics'), 'bl': g('bl_pbl_physics'), 'ra': g('ra_lw_physics'),
                    'e_vert': g('e_vert')}
    return {'mp': -1, 'bl': -1, 'ra': -1, 'e_vert': -1}


def build_case(case):
    out = f'{RESULTS}/{case}'
    bk = f'{out}/bucket_load.csv'
    rsl = f'{BENCH}/{case}/rsl.out.0000'
    feats = load_features(out)
    if not os.path.exists(bk):
        return None, "no bucket_load.csv"
    if not os.path.exists(rsl):
        return None, "no rsl.out.0000 in %s" % os.path.dirname(rsl)
    if len(feats) < 2:
        return None, "only %d feature frame(s); need >= 2 to define a window" % len(feats)
    comp, wait, bsec = load_buckets(bk)
    smap = sim_to_wall(rsl)
    if not smap:
        return None, "rsl.out.0000 has no parsable 'Timing for main' lines"
    sim = np.array([s for s, _ in smap])
    wall = np.array([w for _, w in smap])
    # Init before the time loop is in no Timing line. Recover it as (wall span the buckets
    # cover) - (wall the Timing lines account for). Deriving it from summed comp+wait instead
    # makes a shim accounting quirk blow the offset up and then EVERY window silently misses.
    # every second of wall is charged to either comp or wait, so the per-rank total IS the
    # wall the profiler covered -- an exact anchor, unlike nb*bsec whose last bucket is partial
    bucket_span = float((comp + wait).sum(axis=1).max())
    head = float(np.clip(bucket_span - wall[-1], 0.0, 0.5 * bucket_span))
    if wall[-1] > bucket_span * 1.02:
        print("  %-18s WARN timed loop %.0fs exceeds profiled wall %.0fs -- double counting?"
              % (case, wall[-1], bucket_span))

    def w_of(t):
        i = int(np.searchsorted(sim, t))
        return head + (0.0 if i <= 0 else wall[min(i, len(wall) - 1)])

    rows, ys, cs_all, segs, dropped = [], [], [], [], []
    names = feats[0][2]
    aug_names = None
    for k in range(len(feats) - 1):              # last frame opens no window
        t0, X, nm = feats[k]
        t1 = feats[k + 1][0]
        if nm != names:
            dropped.append("seg%d: feature set differs" % k); continue
        lab = window_label(comp, wait, bsec, w_of(t0), w_of(t1))
        if lab is None:
            dropped.append("seg%d: no bucket overlaps wall [%.0f,%.0f) of span %.0f"
                           % (k, w_of(t0), w_of(t1), bucket_span)); continue
        y, cs = lab
        Xa, aug_names = augment_frame(X, names)
        rows.append(Xa); ys.append(y); cs_all.append(cs); segs.append(k)
    if not rows:
        return None, "; ".join(dropped) or "no segments built"
    if dropped:
        print("  %-18s dropped %d segment(s): %s" % (case, len(dropped), dropped[0]))
    return (np.vstack(rows), np.concatenate(ys), np.concatenate(cs_all),
            np.repeat(segs, NR), aug_names), None


def zscore_by_case(y, cid):
    """Standardise the target inside each case.

    The controller only ever ranks ranks WITHIN one running case, so between-case level is
    not signal -- it is a distractor. Pooled cases here differ in mean compute fraction from
    0.68 to 0.82 (deeper columns are more compute-bound and wait less), and a depth-2 GBRT
    spends its few splits separating those means instead of learning the within-case pattern.
    Spearman on a held-out case is invariant to that case's own standardisation, so this
    changes training only, never the metric.
    """
    z = np.empty_like(y)
    for i in np.unique(cid):
        m = cid == i
        s = y[m].std()
        z[m] = (y[m] - y[m].mean()) / (s if s > 0 else 1.0)
    return z


def loco(X, y, ytrain, cid, fit_kw):
    """Mean leave-one-case-out Spearman. Trains on `ytrain`, always SCORES against raw `y`."""
    out = []
    for i in np.unique(cid):
        trm, tem = cid != i, cid == i
        if tem.sum() == 0 or trm.sum() == 0:
            continue
        m = tr.gbrt_fit(X[trm], ytrain[trm], **fit_kw)
        out.append(tr._spearman(tr.gbrt_pred(m, X[tem]), y[tem]))
    return np.array(out)


def report(Xall, yall, csall, cid, seg, cases, names, nbase):
    print("\nfeatures: base %d, augmented %d  %s" % (nbase, len(names), names[nbase:]))
    print("pooled X%s over %d cases, %d segments" % (Xall.shape, len(cases), len(np.unique(seg))))

    # targets. Metric is always raw comp_frac, so all three are directly comparable.
    yz = zscore_by_case(yall, cid)
    grp = cid * (int(seg.max()) + 1) + seg
    csz = zscore_by_case(csall, grp)          # per (case, segment): segment durations differ
    TARGETS = (("comp_frac", yall), ("comp_frac z-case", yz), ("comp_s z-case", csz))
    geo_ix = [i for i, n in enumerate(names) if n.startswith('geo_')]
    nbr_ix = [i for i, n in enumerate(names) if n.startswith('nbr')]
    base_ix = list(range(nbase))
    # The decisive control: geo_* is IDENTICAL in every case, so a geo-only model contains no
    # weather information at all. If it already scores near the full model, then the per-rank
    # "weather" model is mostly a fixed positional lookup for this 8x56 decomposition.
    FEATS = (("GEO ONLY %d" % len(geo_ix), geo_ix),
             ("base %d" % nbase, base_ix),
             ("base+nbr %d" % (nbase + len(nbr_ix)), base_ix + nbr_ix),
             ("base+geo %d" % (nbase + len(geo_ix)), base_ix + geo_ix),
             ("all %d" % len(names), list(range(len(names)))))
    ALLCAPS = (("d2/40", dict(depth=2, n_est=40, lr=0.15)),
               ("d3/80", dict(depth=3, n_est=80, lr=0.10)),
               ("d4/120", dict(depth=4, n_est=120, lr=0.08)))
    # the depth-4 sweep is ~60% of the total cost; WTUNE_FAST=1 drops it so a bounded job
    # always returns something rather than being killed with a block-buffered empty file
    CAPS = ALLCAPS[:2] if os.environ.get("WTUNE_FAST") else ALLCAPS

    print("\n=== 消融：特徵集 x 訓練標籤 x 模型容量（值為 mean LOCO Spearman）===")
    hdr = "%-16s %-18s" + " %8s" * len(CAPS)      # width follows CAPS, not a hardcoded 3
    print(hdr % ("特徵", "標籤", *[c[0] for c in CAPS]))
    best = (-9, None)
    for fname, fsl in FEATS:
        for tname, yt in TARGETS:
            row = []
            for cname, kw in CAPS:
                v = loco(Xall[:, fsl], yall, yt, cid, kw).mean()
                row.append(v)
                if v > best[0]:
                    best = (v, (fname, fsl, tname, yt, cname, kw))
            print(("%-16s %-18s" + " %+8.3f" * len(row)) % (fname, tname, *row))
    bv, (fname, fsl, tname, yt, cname, kw) = best
    print("-> 最佳: %s / %s / %s = %+.3f" % (fname, tname, cname, bv))

    X = Xall[:, fsl]
    print("\n=== 最佳設定的逐 case LOCO ===")
    for i, c in enumerate(cases):
        trm, tem = cid != i, cid == i
        m = tr.gbrt_fit(X[trm], yt[trm], **kw)
        print("%-20s rho=%+.3f  (n=%d)"
              % (c, tr._spearman(tr.gbrt_pred(m, X[tem]), yall[tem]), tem.sum()))

    print("\n=== leave-one-SCHEME-out ===")
    for key in ('mp', 'bl', 'ra'):
        vals = sorted({schemes(c)[key] for c in cases})
        if len(vals) < 2:
            continue
        accs = []
        for v in vals:
            hold = np.array([schemes(cases[i])[key] == v for i in cid])
            if hold.sum() == 0 or (~hold).sum() == 0:
                continue
            m = tr.gbrt_fit(X[~hold], yt[~hold], **kw)
            rho = tr._spearman(tr.gbrt_pred(m, X[hold]), yall[hold])
            accs.append(rho)
            print("  hold out %s=%-3d rho=%+.3f (n=%d)" % (key, v, rho, hold.sum()))
        if accs:
            print("  %s mean = %+.3f" % (key, np.mean(accs)))

    print("\n=== segment transfer (前半訓練 -> 後半測試) ===")
    half = seg.max() / 2.0
    a, b = seg <= half, seg > half
    if a.sum() and b.sum():
        m = tr.gbrt_fit(X[a], yt[a], **kw)
        print("  early->late rho=%+.3f" % tr._spearman(tr.gbrt_pred(m, X[b]), yall[b]))
        m = tr.gbrt_fit(X[b], yt[b], **kw)
        print("  late->early rho=%+.3f" % tr._spearman(tr.gbrt_pred(m, X[a]), yall[a]))

    # The gate is priced in SECONDS, not ratios: a 1.15x spread on 12 s of compute is 1.8 s and
    # never pays for a move, while the same ratio on 700 s is 105 s and might.
    print("\n=== 每段絕對可回收秒數 max-mean of comp_s（重平衡定價用）===")
    print("%-20s %8s %8s %8s %8s" % ("case", "seg0", "seg1", "seg2", "seg3"))
    for i, c in enumerate(cases):
        cells = []
        for k in sorted(np.unique(seg)):
            m = (cid == i) & (seg == k)
            cells.append("%8.1f" % (csall[m].max() - csall[m].mean()) if m.sum() else "%8s" % "-")
        print("%-20s %s" % (c, "".join(cells)))


def main():
    want = sys.argv[1:] or sorted(
        os.path.basename(d) for d in glob.glob(f'{RESULTS}/*')
        if os.path.isdir(d) and not os.path.basename(d).startswith('_'))
    Xs, ys, css, cid, segs, cases, names = [], [], [], [], [], [], None
    print("=== per-case segmented build ===")
    for c in want:
        r, why = build_case(c)
        if r is None:
            print("%-20s SKIP: %s" % (c, why))   # never fail silently: a silent zero is the bug
            continue
        X, y, cs, sg, nm = r
        names = names or nm
        if nm != names:
            print("%-20s SKIP: feature set differs" % c); continue
        Xs.append(X); ys.append(y); css.append(cs); segs.append(sg)
        cid.append(np.full(len(y), len(cases)))
        s = schemes(c)
        print("%-20s X%s segs=%d comp_frac=%.3f spread=%.3f comp_s=%.0f-%.0f mp%d/bl%d/ra%d/L%d"
              % (c, X.shape, len(np.unique(sg)), y.mean(), y.max() / y.mean(),
                 cs.min(), cs.max(), s['mp'], s['bl'], s['ra'], s['e_vert']))
        cases.append(c)
    if len(cases) < 2:
        print("need >= 2 cases with both features and buckets; got %d" % len(cases)); return 2
    nbase = len(names) - 3 - 2 * sum(1 for k in NBR_OF if k in names)
    report(np.vstack(Xs), np.concatenate(ys), np.concatenate(css), np.concatenate(cid),
           np.concatenate(segs), cases, names, nbase)
    return 0


if __name__ == '__main__':
    sys.exit(main())

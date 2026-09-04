#!/usr/bin/env python3
"""Case-level TOTAL-RUNTIME prediction (user's stated top priority: predict total time).

8 measured regimes. Case-level features = domain aggregate of the per-rank cost drivers.
Target = total wall time proxy = median per-rank elapsed_s (WRF syncs, so per-rank
elapsed ~ wall). Also predict the bottleneck = max per-rank compute.

Only 8 samples -> leave-one-out CV + a mean baseline; keep the model simple (ridge,
single-feature) and report honestly. GBRT/many-feature would just memorise 8 points.
"""
import sys, glob, os, re, datetime, csv
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle as oc
import wtune_oracle_train as tr
tr.PX, tr.PY = 6, 6
CASES = "tropical_conv cont_diurnal marine_sc midlat_cyclone orographic sh_winter subtropical_dry tropical_cyclone".split()
ROOT = "/home/u5727520/wrf-results"

def parse_doy_utc(fn):
    m = re.search(r'wrfout_d01_(\d{4})-(\d{2})-(\d{2})_(\d{2})', os.path.basename(fn))
    y, mo, d, h = map(int, m.groups())
    return datetime.date(y, mo, d).timetuple().tm_yday, h

Xc, elapsed, cmax, names = [], [], [], None
for c in CASES:
    wo = sorted(f for f in glob.glob(f"{ROOT}/{c}/wrfout_d01_*") if os.path.isfile(f))[0]
    mpi = f"{ROOT}/{c}/mpi_per_rank.csv"
    doy, utc0 = parse_doy_utc(wo)
    X, y, nm = tr.build_dataset(wo, mpi, doy, utc0, run_hours=1, stride=4)  # X:(36,11) y:per-rank compute
    names = names or nm
    Xc.append(X.mean(0))                       # case-level = domain-mean of each driver
    rows = list(csv.DictReader(open(mpi)))
    el = np.array([float(r['elapsed_s']) for r in rows])
    elapsed.append(float(np.median(el)))       # wall-time proxy
    cmax.append(float(y.max()))                # bottleneck rank compute
Xc = np.array(Xc); elapsed = np.array(elapsed); cmax = np.array(cmax)
print(f"case-level X{Xc.shape}; targets: elapsed {elapsed.min():.1f}-{elapsed.max():.1f}s, "
      f"cmax {cmax.min():.1f}-{cmax.max():.1f}s\n")

def loo_ridge(X, y, lam):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        mu = X[m].mean(0); sd = X[m].std(0); sd[sd == 0] = 1
        w = tr.ridge_fit((X[m]-mu)/sd, y[m], lam=lam)
        pred[i] = tr.ridge_pred(w, ((X[i:i+1]-mu)/sd))[0]
    return pred

for tgt, name in [(elapsed, "TOTAL elapsed (wall proxy)"), (cmax, "bottleneck max-compute")]:
    print(f"=== target: {name} ===")
    # single-feature Spearman ranking across the 8 cases
    sp = [(tr._spearman(Xc[:, j], tgt), names[j]) for j in range(Xc.shape[1])]
    sp.sort(key=lambda t: -abs(t[0]))
    print("  top case-level features by |Spearman| vs target:")
    for r, nm in sp[:5]:
        print(f"    {nm:12s} rho={r:+.3f}")
    # LOO ridge (all features, strong reg) vs mean baseline
    for lam in (3.0, 10.0):
        pred = loo_ridge(Xc, tgt, lam)
        mae = np.mean(np.abs(pred - tgt)); rho = tr._spearman(pred, tgt)
        print(f"  LOO ridge(lam={lam:>4}): MAE={mae:.2f}s  rho={rho:+.3f}")
    base = np.array([np.delete(tgt, i).mean() for i in range(len(tgt))])
    print(f"  LOO mean-baseline : MAE={np.mean(np.abs(base-tgt)):.2f}s  (target std {tgt.std():.2f}s)\n")

# single-best-feature LOO -- NESTED: the feature is chosen inside each fold, using only the
# training cases. Choosing it once on all 8 (as an earlier version did) leaks the held-out case
# into the selection and makes the reported MAE optimistic by an unknown amount.
print("=== single-best-feature LOO for TOTAL elapsed (nested selection) ===")
n = len(elapsed)
pred = np.zeros(n); chosen = []
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    jt = max(range(Xc.shape[1]), key=lambda j: abs(tr._spearman(Xc[m, j], elapsed[m])))
    chosen.append(names[jt])
    A = np.polyfit(Xc[m, jt], elapsed[m], 1); pred[i] = np.polyval(A, Xc[i, jt])
import collections
sel = collections.Counter(chosen)
print(f"  NESTED LOO MAE={np.mean(np.abs(pred-elapsed)):.2f}s  rho={tr._spearman(pred,elapsed):+.3f}")
print(f"  feature chosen per fold: {dict(sel)}  (stable selection => the headline is real)")
# the leaky version, reported alongside so the gap is visible
jall = max(range(Xc.shape[1]), key=lambda j: abs(tr._spearman(Xc[:, j], elapsed)))
pl = np.zeros(n)
for i in range(n):
    m = np.ones(n, bool); m[i] = False
    A = np.polyfit(Xc[m, jall], elapsed[m], 1); pl[i] = np.polyval(A, Xc[i, jall])
print(f"  (leaky, selection on all cases, '{names[jall]}'): MAE={np.mean(np.abs(pl-elapsed)):.2f}s"
      f"   <- optimism = {np.mean(np.abs(pl-elapsed))-np.mean(np.abs(pred-elapsed)):+.2f}s")
for c, a, p in zip(CASES, elapsed, pred):
    print(f"    {c:18s} actual={a:5.2f}s  LOOpred={p:5.2f}s  err={p-a:+.2f}s")

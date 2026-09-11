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
"""Cross-case generalization test for the WRF per-rank cost oracle.

8 measured 6x6=36-rank cases (diverse WPS regimes). For each: features from its
wrfout snapshot, truth = per-rank compute (elapsed_s - mpi_s) from mpi_per_rank.csv.

MAIN RESULT: leave-one-case-out (LOCO) CV -- train GBRT on 7 regimes, predict the
held-out regime's 36 per-rank compute. Spearman = "can we rank WHICH ranks are hot
in an UNSEEN regime" (the signal imbalance correction needs). MAE is reported too but
is inflated by per-case baseline shift (GBRT cannot extrapolate the absolute level of
a regime it never saw); Spearman is the fair generalization metric.
"""
import sys, glob, csv, os, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle as oc
import wtune_oracle_train as tr

tr.PX, tr.PY = 6, 6                       # override the 16x28 module global for 36-rank cases
CASES = "tropical_conv cont_diurnal marine_sc midlat_cyclone orographic sh_winter subtropical_dry tropical_cyclone".split()
ROOT = "/home/u5727520/wrf-results"

def parse_doy_utc(fn):
    m = re.search(r'wrfout_d01_(\d{4})-(\d{2})-(\d{2})_(\d{2})', os.path.basename(fn))
    y, mo, d, h = map(int, m.groups())
    return datetime.date(y, mo, d).timetuple().tm_yday, h

Xs, ys, cid, names = [], [], [], None
print("=== per-case build + within-case sanity (5-fold Spearman) ===")
for i, c in enumerate(CASES):
    wos = [f for f in glob.glob(f"{ROOT}/{c}/wrfout_d01_*") if os.path.isfile(f)]
    if not wos:
        print(f"{c:18s} NO wrfout"); continue
    wo = sorted(wos)[0]; mpi = f"{ROOT}/{c}/mpi_per_rank.csv"
    doy, utc0 = parse_doy_utc(wo)
    X, y, nm = tr.build_dataset(wo, mpi, doy, utc0, run_hours=1, stride=4)
    names = names or nm
    Xs.append(X); ys.append(y); cid.append(np.full(len(y), i))
    sp = tr.kfold_spearman(X, y, tr.gbrt_fit, tr.gbrt_pred, k=5)[0]
    print(f"{c:18s} X{X.shape} y[min/mean/max]={y.min():.2f}/{y.mean():.2f}/{y.max():.2f}  within-5fold rho={sp:+.3f}")

Xall = np.vstack(Xs); yall = np.concatenate(ys); cidall = np.concatenate(cid)
print(f"\nfeatures ({len(names)}): {names}")
print(f"pooled dataset: X{Xall.shape}")

print("\n=== leave-one-case-out (train 7 regimes -> predict held-out regime's 36 ranks) ===")
locos = []
for i, c in enumerate(CASES):
    trm = cidall != i; tem = cidall == i
    m = tr.gbrt_fit(Xall[trm], yall[trm])
    pred = tr.gbrt_pred(m, Xall[tem]); truth = yall[tem]
    sp = tr._spearman(pred, truth); mae = float(np.mean(np.abs(pred - truth)))
    # imbalance-ordering quality: correlation of (rank's deviation from case mean)
    sp_dev = tr._spearman(pred - pred.mean(), truth - truth.mean())
    locos.append(sp)
    print(f"{c:18s} LOCO rho={sp:+.3f}  imbal-order rho={sp_dev:+.3f}  MAE={mae:5.2f}s  (truth mean {truth.mean():.2f}s)")
print(f"mean LOCO Spearman = {np.mean(locos):+.3f}")

sp_pool = tr.kfold_spearman(Xall, yall, tr.gbrt_fit, tr.gbrt_pred, k=5)[0]
print(f"\npooled-8case 5-fold Spearman (within-mixed) = {sp_pool:+.3f}")
print("(LOCO << pooled would mean the model only interpolates within known regimes;")
print(" LOCO ~ pooled means the per-rank hot-spot signal transfers across regimes.)")

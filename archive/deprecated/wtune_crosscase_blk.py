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
"""Cross-case per-rank test for the BIG cases (3km / 901^2 / 448-rank / 2h).

Features: from each case's end-frame wrfout, tiled to 8x56=448 patches (WRF rank order).
Labels:   per-rank compute fraction = 1 - wait_frac, from blockbar_prof load map
          (grank lrank wait_frac). wall ~ uniform (WRF syncs) so compute_time prop (1-frac).
Rank 0 = serial-I/O outlier -> neutralised (median), same as the CONUS pipeline.
"""
import sys, glob, os, re, datetime
sys.path.insert(0, '/home/u5727520/wrf-stack/wtune')
import numpy as np
import wtune_oracle as oc
import wtune_oracle_train as tr

PX, PY, STRIDE = 8, 56, 2
tr.PX, tr.PY = PX, PY

def parse_doy_utc(fn):
    m = re.search(r'wrfout_d01_(\d{4})-(\d{2})-(\d{2})_(\d{2})', os.path.basename(fn))
    y, mo, d, h = map(int, m.groups())
    return datetime.date(y, mo, d).timetuple().tm_yday, h

def end_wrfout(case_dir):
    fs = [f for f in glob.glob(f"{case_dir}/wrfout_d01_*") if os.path.isfile(f)]
    return sorted(fs)[-1] if fs else None      # latest timestamp = most spun-up

def build_case(wrfout, loadmap, run_hours):
    doy, utc0 = parse_doy_utc(wrfout)
    feats = oc.extract_features(wrfout, stride=STRIDE)
    fields = oc.phase_cost_fields(feats, doy, utc0, run_hours=run_hours)
    cols = {}
    for k in ('radiation', 'micro', 'surf'):
        cols[k] = oc.to_ranks(fields[k], PX, PY).ravel()
    for k in ('TCWV', 'LWP', 'RWP', 'IWP', 'SWP', 'GWP', 'nspecies', 'mixed_phase'):
        if k in feats and feats[k] is not None:
            cols[k] = oc.to_ranks(np.asarray(feats[k], float), PX, PY).ravel()
    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    # label: per-rank compute fraction = 1 - wait_frac (blockbar load map: grank lrank frac)
    wf = np.zeros(PX * PY)
    for ln in open(loadmap):
        p = ln.split()
        if len(p) >= 3:
            r = int(p[0])
            if r < len(wf): wf[r] = float(p[2])
    comp = 1.0 - wf
    comp[0] = np.median(comp)                   # neutralise serial-I/O rank 0
    return X, comp, names

if __name__ == '__main__':
    CASES = os.environ.get("BIGCASES",
        "squall_line maritime_conv wpac_tc natl_cyclone sep_stratus amazon_conv "
        "glakes_lake_effect sahara_dry rockies_oro india_monsoon alaska_winter southern_ocean").split()
    ROOT = "/home/u5727520/wrf-results"
    Xs, ys, cid, names, used = [], [], [], None, []
    print("=== per-case build + within-case 5-fold Spearman ===")
    for i, c in enumerate(CASES):
        wo = end_wrfout(f"{ROOT}/{c}"); lm = f"{ROOT}/{c}/blockbar_load.txt"
        if not wo or not os.path.exists(lm):
            print(f"{c:20s} MISSING (wrfout={bool(wo)} loadmap={os.path.exists(lm)})"); continue
        X, y, nm = build_case(wo, lm, run_hours=2); names = names or nm
        Xs.append(X); ys.append(y); cid.append(np.full(len(y), len(used))); used.append(c)
        sp = tr.kfold_spearman(X, y, tr.gbrt_fit, tr.gbrt_pred, k=5)[0]
        print(f"{c:20s} X{X.shape} compute_frac[min/mean/max]={y.min():.3f}/{y.mean():.3f}/{y.max():.3f} "
              f"spread={y.max()/y.mean():.3f} within-5fold rho={sp:+.3f}")
    if len(used) >= 2:
        Xall = np.vstack(Xs); yall = np.concatenate(ys); cidall = np.concatenate(cid)
        print(f"\nfeatures({len(names)}): {names}\npooled X{Xall.shape} over {len(used)} cases")
        print("\n=== leave-one-case-out ===")
        loco = []
        for i, c in enumerate(used):
            trm = cidall != i; tem = cidall == i
            m = tr.gbrt_fit(Xall[trm], yall[trm]); pred = tr.gbrt_pred(m, Xall[tem]); truth = yall[tem]
            sp = tr._spearman(pred, truth); loco.append(sp)
            print(f"{c:20s} LOCO rho={sp:+.3f}  (truth spread {truth.max()/truth.mean():.3f})")
        print(f"mean LOCO Spearman = {np.mean(loco):+.3f}")
        print(f"pooled 5-fold Spearman = {tr.kfold_spearman(Xall, yall, tr.gbrt_fit, tr.gbrt_pred, k=5)[0]:+.3f}")

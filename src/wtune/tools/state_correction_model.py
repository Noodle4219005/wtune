#!/usr/bin/env python3
"""tools/state_correction_model.py -- ML state correction: cold-start wrfinput -> real WRF state.

NOT a timing/cost model (see tools/window_cost_model was abandoned for this reason -- see
inbox capture). The job here is narrower and different: a cold-start wrfinput_d01 built from
GFS/analog state at simulated time T (tools/coldstart_sample.py's real.exe output -- ALL
hydrometeors are exactly zero there, per wtune_oracle.py's documented RH-fallback finding) is
systematically biased relative to what a real WRF simulation's own state at T actually looks
like (the model has had time to develop its own clouds/precip/momentum structure that a
diagnostic cold start cannot see). This trains a small numpy-only regressor that predicts that
bias -- per-level residuals in T/QVAPOR/U/V plus cloud/precipitating hydrometeor amounts -- from
cheap column features of the cold-start file alone, using paired (cold-start wrfinput, WRF's own
restart wrfrst at the same T) truth. `apply` writes an optionally-corrected COPY of a wrfinput;
it is never consumed by any wall-time/cost code path in this repo.

CLI: python3 tools/state_correction_model.py <build-pairs|evaluate|apply> ...
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wtune import predict as predict         # solar_cos_zenith (exact same formula the rest of the
                                         # project uses for day/night weighting).

_WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
G_ACCEL = 9.81                          # same constant wtune_oracle.py uses for Z = geopotential/g


class StateCorrectionError(Exception):
    """User-facing input error (grid mismatch, missing required variable, bad CLI args)."""


# ---------------------------------------------------------------------------
# Physics helpers (Tetens saturation vapor pressure -- SAME constants/formula as
# wtune_oracle._derived_moisture, reused here verbatim rather than re-derived, for
# consistency with the rest of the project's RH proxy).
# ---------------------------------------------------------------------------

def temp_from_theta(theta_k, pressure_pa):
    """Actual temperature [K] from potential temperature theta=T+300 [K] and pressure [Pa]."""
    return theta_k * (np.asarray(pressure_pa, float) / 1.0e5) ** 0.2854


def saturation_qs(temp_k, pressure_pa):
    """Saturation mixing ratio [kg/kg] (Tetens es, same constants as wtune_oracle.py)."""
    temp_k = np.asarray(temp_k, float)
    es = 611.2 * np.exp(17.67 * (temp_k - 273.15) / (temp_k - 29.65))
    return 0.622 * es / np.maximum(np.asarray(pressure_pa, float) - es, 1.0)


def relative_humidity(theta_k, qv_kgkg, pressure_pa, clip=1.2):
    """RH = qv/qs, clipped to [0, clip] for use as a bounded ML feature. clip=None for the
    raw (unclipped) value needed by `apply`'s supersaturation check."""
    temp_k = temp_from_theta(theta_k, pressure_pa)
    qs = saturation_qs(temp_k, pressure_pa)
    qv = np.maximum(np.asarray(qv_kgkg, float), 1e-12)
    rh = qv / qs
    return rh if clip is None else np.clip(rh, 0.0, clip)


# ---------------------------------------------------------------------------
# Stagger / destagger
# ---------------------------------------------------------------------------

def destagger(arr, axis):
    """Stag (length N+1 along `axis`) -> mass point (length N): average of adjacent pairs.
    Used to bring U/V (and PH/PHB) onto the same mass grid as T/QVAPOR for feature/target
    extraction."""
    arr = np.asarray(arr)
    sl0 = [slice(None)] * arr.ndim
    sl1 = [slice(None)] * arr.ndim
    sl0[axis] = slice(0, -1)
    sl1[axis] = slice(1, None)
    return 0.5 * (arr[tuple(sl0)] + arr[tuple(sl1)])


def restagger(mass_arr, axis):
    """EXACT algebraic right-inverse of `destagger`: the unique stag array (length N+1) with
    stag[0] = mass[0] and (stag[k]+stag[k+1])/2 = mass[k] for every k -- i.e.
    destagger(restagger(m), axis) == m identically, for any m (see
    tools/test_state_correction_model.py's round-trip test). This is NOT a smoothing average;
    it is a linear system solved exactly (one degree of freedom, pinned at the west/south edge).
    Used only where an exact correction placement matters; `apply` itself restaggers predicted
    U/V residuals with `restagger_avg` below, per this task's explicit "restaggered by
    averaging" instruction."""
    mass_arr = np.asarray(mass_arr, float)
    n = mass_arr.shape[axis]
    out_shape = list(mass_arr.shape)
    out_shape[axis] = n + 1
    out = np.zeros(out_shape, dtype=float)

    def set_slice(a, axis_, i, val):
        sl = [slice(None)] * a.ndim
        sl[axis_] = i
        a[tuple(sl)] = val

    def get_slice(a, axis_, i):
        sl = [slice(None)] * a.ndim
        sl[axis_] = i
        return a[tuple(sl)]

    set_slice(out, axis, 0, get_slice(mass_arr, axis, 0))          # stag[0] = mass[0]
    for k in range(n):
        prev = get_slice(out, axis, k)
        mk = get_slice(mass_arr, axis, k)
        set_slice(out, axis, k + 1, 2.0 * mk - prev)                # stag[k+1] = 2*m[k]-stag[k]
    return out


def restagger_avg(mass_arr, axis):
    """Spread a mass-point field onto the (length N+1) staggered grid by averaging: interior
    stag points = mean of their two neighboring mass points, the two edge stag points = their
    single neighboring mass point (no extrapolation). Used by `apply` to place predicted
    dU/dV mass-point residuals back onto U/V's own native grid -- the spec's "U/V restaggered
    by averaging", distinct from the exact inverse `restagger` above (that one has no
    "smoothing" interpretation; this one does, and is what should be used for adding a noisy
    ML correction back onto a physical field)."""
    mass_arr = np.asarray(mass_arr, float)
    n = mass_arr.shape[axis]
    interior = 0.5 * (np.take(mass_arr, range(0, n - 1), axis=axis)
                       + np.take(mass_arr, range(1, n), axis=axis))
    first = np.take(mass_arr, [0], axis=axis)
    last = np.take(mass_arr, [n - 1], axis=axis)
    return np.concatenate([first, interior, last], axis=axis)


# ---------------------------------------------------------------------------
# netCDF I/O (lazy netCDF4 import -- same convention as coldstart_sample.py/wtune_oracle.py:
# this module must stay importable without netCDF4 for the pure-numpy pieces to be testable).
# ---------------------------------------------------------------------------

_CORE_VARS = ("T", "QVAPOR", "U", "V", "PH", "PHB", "P", "PB", "MU", "MUB", "TSK", "SMOIS",
              "HGT", "LANDMASK", "XLAT", "XLONG", "ZNU")
_HYDROMETEOR_VARS = ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")

FEATURE_NAMES = ["theta_K", "qvapor_kgkg", "rh", "pressure_pa", "height_agl_m", "u_ms", "v_ms",
                  "hgt_m", "landmask", "tsk_K", "lat_deg", "coszen", "eta", "level_idx_frac",
                  "mu_total_pa", "smois_top"]


class _RestartAliases:
    """wrfrst_d0N stores the prognostic fields per time level (U_1/U_2, V_*, PH_*, MU_*, ...);
    wrfinput uses the bare names. Reading NAME falls back to NAME_2 (the current level at restart)
    so both file kinds expose the same variable names. Verified on scc 2026-09-24: conus12
    wrfrst has U_2 V_2 PH_2 MU_2 but T, P, PB, QVAPOR, TSK, SMOIS under their bare names."""

    def __init__(self, ds):
        self._ds = ds
        self.dimensions = ds.dimensions
        self.variables = _AliasVars(ds.variables)

    def __getattr__(self, name):
        return getattr(self._ds, name)

    def close(self):
        self._ds.close()


class _AliasVars(dict):
    def __init__(self, raw):
        super().__init__(raw)
        for k in list(raw):
            if k.endswith("_2") and k[:-2] not in raw:
                self[k[:-2]] = raw[k]


def _open_nc(path):
    import netCDF4
    return _RestartAliases(netCDF4.Dataset(path))


def _read_times(ds):
    """[datetime, ...] from a WRF file's Times(Time,DateStrLen) char variable. Decoding mirrors
    tools/state_timeline.py's read_times() (tobytes() fast path, per-char fallback for a masked
    array) -- same proven approach, kept local rather than imported since that file is owned by
    another concurrent agent."""
    out = []
    for row in ds.variables["Times"][:]:
        if hasattr(row, "tobytes"):
            s = row.tobytes().decode("utf-8", "replace")
        else:
            s = "".join(c.decode("utf-8", "replace") if isinstance(c, (bytes, bytearray))
                         else str(c) for c in row)
        s = s.split("\x00")[0]
        out.append(datetime.datetime.strptime(s, _WRF_TIME_FMT))
    return out


def check_grid_match(ds_truth, ds_input, tol_deg=1e-2):
    """Raises StateCorrectionError unless both files describe the same grid: dims
    (bottom_top/south_north/west_east), DX/DY global attrs, and SW-corner XLAT/XLONG."""
    for dim in ("bottom_top", "south_north", "west_east"):
        a, b = len(ds_truth.dimensions[dim]), len(ds_input.dimensions[dim])
        if a != b:
            raise StateCorrectionError("grid mismatch: dim %r truth=%d input=%d" % (dim, a, b))
    for attr in ("DX", "DY"):
        a = float(getattr(ds_truth, attr))
        b = float(getattr(ds_input, attr))
        if abs(a - b) > 1.0:
            raise StateCorrectionError("grid mismatch: %s truth=%r input=%r" % (attr, a, b))
    lat_a = float(ds_truth.variables["XLAT"][0, 0, 0])
    lat_b = float(ds_input.variables["XLAT"][0, 0, 0])
    lon_a = float(ds_truth.variables["XLONG"][0, 0, 0])
    lon_b = float(ds_input.variables["XLONG"][0, 0, 0])
    if abs(lat_a - lat_b) > tol_deg or abs(lon_a - lon_b) > tol_deg:
        raise StateCorrectionError(
            "grid mismatch: SW corner truth=(%.4f,%.4f) input=(%.4f,%.4f)"
            % (lat_a, lon_a, lat_b, lon_b))


def _available_hydrometeors(ds_truth, ds_input):
    return [v for v in _HYDROMETEOR_VARS
            if v in ds_truth.variables and v in ds_input.variables]


def _require_core(ds, path):
    missing = [v for v in _CORE_VARS if v not in ds.variables]
    if missing:
        raise StateCorrectionError("%s: missing required variable(s) %s" % (path, missing))


def _column_features(ds, j, i, t_dt, nz):
    """One column's FEATURE_NAMES rows (nz, len(FEATURE_NAMES)), from ds (INPUT file) alone."""
    theta = np.asarray(ds.variables["T"][0, :, j, i], float) + 300.0
    qv = np.asarray(ds.variables["QVAPOR"][0, :, j, i], float)
    p = (np.asarray(ds.variables["P"][0, :, j, i], float)
         + np.asarray(ds.variables["PB"][0, :, j, i], float))
    ph_stag = (np.asarray(ds.variables["PH"][0, :, j, i], float)
               + np.asarray(ds.variables["PHB"][0, :, j, i], float))
    z_mass = destagger(ph_stag / G_ACCEL, axis=0)
    hgt = float(ds.variables["HGT"][0, j, i])
    height_agl = z_mass - hgt
    u_mass = 0.5 * (np.asarray(ds.variables["U"][0, :, j, i], float)
                     + np.asarray(ds.variables["U"][0, :, j, i + 1], float))
    v_mass = 0.5 * (np.asarray(ds.variables["V"][0, :, j, i], float)
                     + np.asarray(ds.variables["V"][0, :, j + 1, i], float))
    landmask = float(ds.variables["LANDMASK"][0, j, i])
    tsk = float(ds.variables["TSK"][0, j, i])
    lat = float(ds.variables["XLAT"][0, j, i])
    lon = float(ds.variables["XLONG"][0, j, i])
    coszen = predict.solar_cos_zenith(lat, lon, t_dt)
    eta = np.asarray(ds.variables["ZNU"][0, :], float)
    mu_total = float(ds.variables["MU"][0, j, i]) + float(ds.variables["MUB"][0, j, i])
    smois_top = float(ds.variables["SMOIS"][0, 0, j, i])
    rh = relative_humidity(theta, qv, p)
    level_idx_frac = np.arange(nz, dtype=float) / max(nz - 1, 1)

    cols = {
        "theta_K": theta, "qvapor_kgkg": qv, "rh": rh, "pressure_pa": p,
        "height_agl_m": height_agl, "u_ms": u_mass, "v_ms": v_mass,
        "hgt_m": np.full(nz, hgt), "landmask": np.full(nz, landmask),
        "tsk_K": np.full(nz, tsk), "lat_deg": np.full(nz, lat),
        "coszen": np.full(nz, coszen), "eta": eta, "level_idx_frac": level_idx_frac,
        "mu_total_pa": np.full(nz, mu_total), "smois_top": np.full(nz, smois_top),
    }
    return np.column_stack([cols[n] for n in FEATURE_NAMES]), p


def _column_truth(ds, j, i, hydrometeor_vars):
    theta = np.asarray(ds.variables["T"][0, :, j, i], float) + 300.0
    qv = np.asarray(ds.variables["QVAPOR"][0, :, j, i], float)
    u_mass = 0.5 * (np.asarray(ds.variables["U"][0, :, j, i], float)
                     + np.asarray(ds.variables["U"][0, :, j, i + 1], float))
    v_mass = 0.5 * (np.asarray(ds.variables["V"][0, :, j, i], float)
                     + np.asarray(ds.variables["V"][0, :, j + 1, i], float))
    nz = theta.shape[0]
    cloud = np.zeros(nz)
    precip = np.zeros(nz)
    if "QCLOUD" in hydrometeor_vars:
        cloud = cloud + np.asarray(ds.variables["QCLOUD"][0, :, j, i], float)
    if "QICE" in hydrometeor_vars:
        cloud = cloud + np.asarray(ds.variables["QICE"][0, :, j, i], float)
    if "QRAIN" in hydrometeor_vars:
        precip = precip + np.asarray(ds.variables["QRAIN"][0, :, j, i], float)
    if "QSNOW" in hydrometeor_vars:
        precip = precip + np.asarray(ds.variables["QSNOW"][0, :, j, i], float)
    if "QGRAUP" in hydrometeor_vars:
        precip = precip + np.asarray(ds.variables["QGRAUP"][0, :, j, i], float)
    return theta, qv, u_mass, v_mass, cloud, precip


# ---------------------------------------------------------------------------
# build-pairs
# ---------------------------------------------------------------------------

def build_pairs(pairs, out_path, n_columns=3000, seed=0):
    """pairs: [(case_id, truth_rst_path, input_wrfinput_path), ...]. Writes `out_path` (.npz)
    with X (n_rows, len(FEATURE_NAMES)), per-target arrays, pressure_pa, case_id, and
    (pair_idx, col_j, col_i, level_idx) metadata; one row per sampled-column x level."""
    rng = np.random.default_rng(seed)
    X_all, dT_all, dQV_all, dU_all, dV_all = [], [], [], [], []
    cloud_all, precip_all, case_all, pidx_all, j_all, i_all, lvl_all = [], [], [], [], [], [], []
    hyd_report = {}

    for pidx, (case_id, truth_path, input_path) in enumerate(pairs):
        ds_t = _open_nc(truth_path)
        ds_in = _open_nc(input_path)
        try:
            check_grid_match(ds_t, ds_in)
            _require_core(ds_t, truth_path)
            _require_core(ds_in, input_path)
            hyd = _available_hydrometeors(ds_t, ds_in)
            hyd_report[case_id] = hyd
            nz = len(ds_in.dimensions["bottom_top"])
            ny = len(ds_in.dimensions["south_north"])
            nx = len(ds_in.dimensions["west_east"])
            times = _read_times(ds_in)
            t_dt = times[0]

            n_cells = ny * nx
            n_pick = min(n_columns, n_cells)
            flat = rng.choice(n_cells, size=n_pick, replace=False)
            js, iss = np.unravel_index(flat, (ny, nx))

            for j, i in zip(js.tolist(), iss.tolist()):
                feats, p = _column_features(ds_in, j, i, t_dt, nz)
                in_theta = feats[:, FEATURE_NAMES.index("theta_K")]
                in_qv = feats[:, FEATURE_NAMES.index("qvapor_kgkg")]
                in_u = feats[:, FEATURE_NAMES.index("u_ms")]
                in_v = feats[:, FEATURE_NAMES.index("v_ms")]
                t_theta, t_qv, t_u, t_v, cloud, precip = _column_truth(ds_t, j, i, hyd)

                X_all.append(feats)
                dT_all.append(t_theta - in_theta)
                dQV_all.append(t_qv - in_qv)
                dU_all.append(t_u - in_u)
                dV_all.append(t_v - in_v)
                cloud_all.append(cloud)
                precip_all.append(precip)
                case_all.extend([case_id] * nz)
                pidx_all.extend([pidx] * nz)
                j_all.extend([j] * nz)
                i_all.extend([i] * nz)
                lvl_all.extend(list(range(nz)))
        finally:
            ds_t.close()
            ds_in.close()
        print("build-pairs: case=%s pair=%d columns=%d hydrometeors=%s"
              % (case_id, pidx, n_pick, hyd_report[case_id]))

    X = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, len(FEATURE_NAMES)))
    dT = np.concatenate(dT_all) if dT_all else np.zeros(0)
    dQV = np.concatenate(dQV_all) if dQV_all else np.zeros(0)
    dU = np.concatenate(dU_all) if dU_all else np.zeros(0)
    dV = np.concatenate(dV_all) if dV_all else np.zeros(0)
    cloud = np.concatenate(cloud_all) if cloud_all else np.zeros(0)
    precip = np.concatenate(precip_all) if precip_all else np.zeros(0)
    pressure_pa = X[:, FEATURE_NAMES.index("pressure_pa")] if len(X) else np.zeros(0)

    np.savez_compressed(
        out_path, X=X, feature_names=np.array(FEATURE_NAMES),
        dT=dT, dQV=dQV, dU=dU, dV=dV, cloud=cloud, precip=precip, pressure_pa=pressure_pa,
        case_id=np.array(case_all, dtype="<U128"),
        pair_idx=np.array(pidx_all, dtype=int), col_j=np.array(j_all, dtype=int),
        col_i=np.array(i_all, dtype=int), level_idx=np.array(lvl_all, dtype=int),
        hydrometeors_by_case=np.array(json.dumps(hyd_report)))
    return {"n_rows": int(len(X)), "n_pairs": len(pairs), "hydrometeors": hyd_report}


# ---------------------------------------------------------------------------
# ML core: ridge with nested (inner-LOO) alpha selection, LOCO harness, kNN, two-part
# presence/amount model for hydrometeors. One pure-numpy ridge (ridge_fit/ridge_pred below);
# repo policy: no scipy/sklearn.
# ---------------------------------------------------------------------------

RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


# ridge_fit / ridge_pred: verbatim from the archived wtune_oracle_train.py
def ridge_fit(X, y, lam=1.0):
    Xa = np.column_stack([np.ones(len(X)), X])
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1]); A[0, 0] -= lam
    w = np.linalg.solve(A, Xa.T @ y)
    return w


def ridge_pred(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


def _standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def _inner_loco_rmse(X, y, groups, alpha):
    """Pooled RMSE from an INNER leave-one-CASE-out sweep over (X, y, groups) -- each inner
    fold standardizes on its own inner-training subset only. T17 (2026-09-24): replaces the old
    leave-one-SAMPLE-out alpha search (_ridge_loo_mae, O(n^2 d): n refits of an O(n d) ridge
    solve each), which took 2h17m/49m on real ~2e6/3.4e5-row datasets and had to be killed.
    Inner LOCO with (say) 4 training cases is 4 fits total, independent of n. Returns None if
    fewer than 2 distinct cases are present (the caller falls back to a fixed alpha)."""
    groups = np.asarray(groups)
    cases = sorted(set(groups.tolist()))
    if len(cases) < 2:
        return None
    preds = np.empty(len(y), dtype=float)
    for c in cases:
        train = groups != c
        test = groups == c
        mu, sd = _standardize_fit(X[train])
        w = ridge_fit((X[train] - mu) / sd, y[train], lam=alpha)
        preds[test] = ridge_pred(w, (X[test] - mu) / sd)
    return _rmse(preds, y)


def fit_ridge_nested(X, y, groups=None, alphas=RIDGE_ALPHAS):
    """Ridge with alpha chosen by INNER leave-one-CASE-out on (X, y, groups) -- see
    _inner_loco_rmse. `groups` identifies the case each row of X/y belongs to WITHIN this
    training fold; callers (loco_predict) must never pass the OUTER held-out LOCO case's rows
    in here, so `groups` here can only ever contain the remaining training cases. Falls back to
    alpha=1.0 (announced) when `groups` is None or has fewer than 2 distinct cases (T17(a))."""
    best_alpha, best_rmse = None, None
    if groups is not None:
        for a in alphas:
            rmse_a = _inner_loco_rmse(X, y, groups, a)
            if rmse_a is not None and (best_rmse is None or rmse_a < best_rmse):
                best_alpha, best_rmse = a, rmse_a
    if best_alpha is None:
        best_alpha = 1.0
        n_cases = len(set(np.asarray(groups).tolist())) if groups is not None else 0
        print("fit_ridge_nested: training fold has %d case(s) (< 2) -- alpha fallback = 1.0"
              % n_cases, flush=True)
    mu, sd = _standardize_fit(X)
    Xs = (X - mu) / sd
    w = ridge_fit(Xs, y, lam=best_alpha)
    return {"kind": "ridge", "w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
            "alpha": best_alpha}


def ridge_predict_model(model, X):
    mu = np.asarray(model["mu"])
    sd = np.asarray(model["sd"])
    w = np.asarray(model["w"])
    Xs = (X - mu) / sd
    return ridge_pred(w, Xs)


def knn_predict(X_train, y_train, X_test, k=3, chunk_size=2000):
    """Plain-numpy kNN regression: mean of the k nearest (Euclidean) training labels. T17(b)
    (2026-09-24): distances are computed in chunks of at most `chunk_size` test rows at a time
    instead of one (n_test, n_train, d) broadcast tensor -- the unchunked version is O(n_test *
    n_train * d) memory, which is what made this unusable at real dataset sizes. Numerically
    IDENTICAL to the unchunked computation (same per-row nearest-k selection and mean, just
    computed in slices)."""
    k_eff = min(k, len(X_train))
    n_test = len(X_test)
    out = np.empty(n_test, dtype=float)
    for start in range(0, n_test, max(1, chunk_size)):
        end = min(start + chunk_size, n_test)
        chunk = X_test[start:end]
        d = np.sqrt(((chunk[:, None, :] - X_train[None, :, :]) ** 2).sum(-1))
        idx = np.argsort(d, axis=1)[:, :k_eff]
        out[start:end] = y_train[idx].mean(axis=1)
    return out


def fit_knn(X, y, groups=None, k=3, max_train_rows=20000, seed=0):
    """T17(b) (2026-09-24): fits on a reproducible random subsample of at most `max_train_rows`
    training rows -- an unbounded training set makes knn_predict's per-test-row distance/sort
    cost O(n_train), unusable at real dataset sizes (~1e5-1e6 rows). `subsample_size` is
    recorded in the returned model (and therefore in evaluate()'s report, when this model kind
    is used for a target). `groups` is accepted (unused -- kNN does no alpha/case selection)
    only so loco_predict's uniform fit_fn(X, y, groups) calling convention works for it too."""
    n = len(X)
    if n > max_train_rows:
        rng = np.random.default_rng(seed)
        sub = rng.choice(n, size=max_train_rows, replace=False)
        X, y = X[sub], y[sub]
    mu, sd = _standardize_fit(X)
    return {"kind": "knn", "X": ((X - mu) / sd).tolist(), "y": y.tolist(), "mu": mu.tolist(),
            "sd": sd.tolist(), "k": k, "subsample_size": int(len(X)), "seed": seed}


def knn_predict_model(model, X):
    mu = np.asarray(model["mu"])
    sd = np.asarray(model["sd"])
    Xtr = np.asarray(model["X"])
    ytr = np.asarray(model["y"])
    Xs = (X - mu) / sd
    return knn_predict(Xtr, ytr, Xs, k=model["k"])


def loco_predict(X, y, groups, fit_fn, predict_fn, verbose=False):
    """Leave-one-CASE-out: for every distinct value in `groups`, fit on every OTHER group's
    rows and predict this group's rows. Returns an array the same length/order as y. fit_fn's
    training input NEVER includes a row from the group being predicted this iteration. fit_fn is
    called as fit_fn(X_train, y_train, groups_train) -- T17(a): groups_train lets fit_ridge_nested
    (and fit_two_part) run their OWN inner leave-one-CASE-out alpha selection without ever
    seeing the row being held out this outer iteration (groups_train structurally excludes it).
    `verbose` (T17(d)): prints one flushed progress line per outer fold, for a long run."""
    groups = np.asarray(groups)
    cases = sorted(set(groups.tolist()))
    preds = np.empty(len(y), dtype=float)
    for i, c in enumerate(cases):
        train = groups != c
        test = groups == c
        model = fit_fn(X[train], y[train], groups[train])
        preds[test] = predict_fn(model, X[test])
        if verbose:
            print("  LOCO fold %d/%d (held out %s): %d train rows, %d test rows"
                  % (i + 1, len(cases), c, int(train.sum()), int(test.sum())), flush=True)
    return preds


def fit_two_part(X, y_amount, groups=None, threshold=1e-5, alphas=RIDGE_ALPHAS):
    """Presence (ridge-regressed 0/1, thresholded at predict time) + amount-where-present
    (ridge on log1p(amount*1e6)) -- the spec's "logistic-ish ... ridge on features then
    threshold" option. T17(c): both parts use the SAME inner leave-one-CASE-out alpha
    selection as fit_ridge_nested (`groups` sliced to the amount-where-present mask for the
    second part)."""
    presence = (y_amount > threshold).astype(float)
    clf = fit_ridge_nested(X, presence, groups, alphas=alphas)
    mask = y_amount > threshold
    reg = fit_ridge_nested(X[mask], np.log1p(y_amount[mask] * 1e6),
                           groups[mask] if groups is not None else None, alphas=alphas) \
        if mask.sum() >= 3 else None
    return {"kind": "two_part", "clf": clf, "reg": reg, "threshold": threshold}


def two_part_predict(model, X):
    """Predicted amount (kg/kg), zero where the classifier predicts no condensate."""
    presence = ridge_predict_model(model["clf"], X) > 0.5
    amt = np.zeros(len(X))
    if model["reg"] is not None:
        log_amt = np.clip(ridge_predict_model(model["reg"], X), -50, 50)
        amt = np.expm1(log_amt) / 1e6
    amt = np.clip(amt, 0.0, None)
    return np.where(presence, amt, 0.0)


LEVEL_GROUPS = [
    # NOTE (deviation from the task's literal wording, which said "low <850 hPa, mid 850-400,
    # high >400" -- self-contradictory since that makes 'high' a superset of 'mid' and puts the
    # near-surface column in the 'low' *pressure* bucket instead of the 'low' *altitude* one.
    # Implemented with the standard meteorological sense instead: low = near-surface = high
    # pressure.
    ("low", lambda p: p > 85000.0),
    ("mid", lambda p: (p <= 85000.0) & (p > 40000.0)),
    ("high", lambda p: p <= 40000.0),
]

RESIDUAL_TARGETS = ["dT", "dQV", "dU", "dV"]
CONDENSATE_TARGETS = ["cloud", "precip"]
CONDENSATE_THRESHOLD = 1e-5


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def _csi(pred_present, truth_present):
    hits = int(np.sum(pred_present & truth_present))
    misses = int(np.sum(~pred_present & truth_present))
    fa = int(np.sum(pred_present & ~truth_present))
    denom = hits + misses + fa
    if denom == 0:
        return 1.0  # nothing predicted, nothing observed -- vacuously perfect on this slice
    return hits / denom


def is_improved(model_rmse, baseline_rmse, n_cases, min_relative=0.10, min_cases=4):
    if n_cases < min_cases or baseline_rmse <= 0:
        return False
    return (baseline_rmse - model_rmse) / baseline_rmse >= min_relative


def is_improved_condensate(model_csi, baseline_csi, n_cases, min_abs=0.10, min_cases=4):
    if n_cases < min_cases:
        return False
    return (model_csi - baseline_csi) >= min_abs


def _subsample_rows_per_pair(pair_idx, max_rows_per_pair, seed=0):
    """T17(e) (2026-09-24): boolean mask selecting at most `max_rows_per_pair` rows per distinct
    `pair_idx` value, chosen reproducibly (seeded RNG) -- for a quick evaluate() run on a
    subset. `max_rows_per_pair=None` -> mask selects everything (no-op)."""
    n = len(pair_idx)
    if max_rows_per_pair is None:
        return np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    for p in sorted(set(pair_idx.tolist())):
        rows = np.where(pair_idx == p)[0]
        if len(rows) > max_rows_per_pair:
            rows = rng.choice(rows, size=max_rows_per_pair, replace=False)
        mask[rows] = True
    return mask


def evaluate(dataset_npz, out_path, max_rows_per_pair=None, seed=0):
    """`max_rows_per_pair` (T17(e), default None = all rows): reproducibly subsamples to at
    most this many rows per pair_idx before evaluating, for a quick run. `seed` (T17(a)/(b)):
    used for this subsampling AND for fit_knn's own training-row subsampling (kNN is not
    currently wired into any target here, but the seed threads through for when it is)."""
    t0 = time.time()
    d = np.load(dataset_npz, allow_pickle=False)
    row_mask = _subsample_rows_per_pair(d["pair_idx"], max_rows_per_pair, seed=seed)
    X = d["X"][row_mask]
    case_id = d["case_id"][row_mask]
    pressure = d["pressure_pa"][row_mask]
    target_arrays = {tgt: d[tgt][row_mask] for tgt in RESIDUAL_TARGETS + CONDENSATE_TARGETS}
    n_cases = len(set(case_id.tolist()))
    n_rows = len(X)
    if max_rows_per_pair is not None:
        print("evaluate: --max-rows-per-pair=%d (seed=%d) -> %d/%d rows kept"
              % (max_rows_per_pair, seed, n_rows, len(row_mask)), flush=True)
    report = {"n_cases": n_cases, "n_rows": n_rows, "targets": {}}

    for name, mask_fn in LEVEL_GROUPS:
        lvl_mask = mask_fn(pressure)
        if not lvl_mask.any():
            continue
        Xs, cases_s = X[lvl_mask], case_id[lvl_mask]

        for tgt in RESIDUAL_TARGETS:
            y = target_arrays[tgt][lvl_mask]
            print("evaluate: %s__%s (n=%d)" % (tgt, name, len(y)), flush=True)
            preds = loco_predict(Xs, y, cases_s, fit_ridge_nested, ridge_predict_model,
                                 verbose=True)
            baseline_rmse = _rmse(np.zeros_like(y), y)   # baseline = uncorrected (residual 0)
            model_rmse = _rmse(preds, y)
            improved = is_improved(model_rmse, baseline_rmse, n_cases)
            model_full = fit_ridge_nested(Xs, y, cases_s)
            report["targets"]["%s__%s" % (tgt, name)] = {
                "kind": "residual", "level_group": name, "n": int(len(y)),
                "baseline_rmse": baseline_rmse, "model_rmse": model_rmse,
                "relative_reduction": (baseline_rmse - model_rmse) / baseline_rmse
                if baseline_rmse > 0 else None,
                "improved": improved, "model": model_full,
                "ridge_coeffs_standardized": dict(zip(FEATURE_NAMES, model_full["w"][1:])),
            }

        for tgt in CONDENSATE_TARGETS:
            y = target_arrays[tgt][lvl_mask]
            print("evaluate: %s__%s (n=%d)" % (tgt, name, len(y)), flush=True)
            truth_present = y > CONDENSATE_THRESHOLD
            preds = loco_predict(Xs, y, cases_s, fit_two_part, two_part_predict, verbose=True)
            pred_present = preds > CONDENSATE_THRESHOLD
            baseline_rmse = _rmse(np.zeros_like(y), y)   # baseline = 0 amount everywhere
            model_rmse = _rmse(preds, y)
            baseline_csi = _csi(np.zeros_like(truth_present, dtype=bool), truth_present)
            model_csi = _csi(pred_present, truth_present)
            improved = is_improved_condensate(model_csi, baseline_csi, n_cases)
            model_full = fit_two_part(Xs, y, cases_s)
            report["targets"]["%s__%s" % (tgt, name)] = {
                "kind": "condensate", "level_group": name, "n": int(len(y)),
                "baseline_rmse": baseline_rmse, "model_rmse": model_rmse,
                "baseline_csi": baseline_csi, "model_csi": model_csi,
                "domain_mean_error": float(np.mean(preds) - np.mean(y)),
                "improved": improved, "model": model_full,
            }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("decision rule: residual target improved iff LOCO RMSE drops >=10%% vs the "
          "uncorrected baseline AND n_cases>=4; condensate target improved iff LOCO CSI rises "
          ">=0.10 (absolute) over the baseline (which predicts no condensate anywhere) AND "
          "n_cases>=4. n_cases=%d -> %s" % (n_cases, "meets the case-count floor" if n_cases >= 4
                                             else "BELOW the case-count floor: report every "
                                             "target as 'not enough evidence' regardless of MAE"))
    for key, t in report["targets"].items():
        verdict = "IMPROVED" if t["improved"] else "not enough evidence"
        print("  %-16s n=%-6d %s" % (key, t["n"], verdict))
    elapsed = time.time() - t0
    print("evaluate: total wall time %.1fs" % elapsed, flush=True)
    return report


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def _predict_grid(model_entry, feats_flat, mask_flat):
    """model_entry: the report['targets'][...] dict (may be None if that level-group/target
    wasn't fit). Returns a full-length (len(feats_flat),) array, zero outside mask_flat."""
    out = np.zeros(len(feats_flat))
    if model_entry is None or not mask_flat.any():
        return out
    model = model_entry["model"]
    Xm = feats_flat[mask_flat]
    if model["kind"] == "ridge":
        out[mask_flat] = ridge_predict_model(model, Xm)
    elif model["kind"] == "two_part":
        out[mask_flat] = two_part_predict(model, Xm)
    else:
        raise StateCorrectionError("unknown model kind %r" % model["kind"])
    return out


def apply_correction(report_path, wrfinput_in, wrfinput_out, targets_from_report=False):
    """Writes a corrected COPY of wrfinput_in to wrfinput_out, using report_path's fitted
    per-target models. `targets_from_report` (K5, 2026-09-24): apply ONLY the targets whose
    evaluate() verdict is 'improved' (report['targets'][k]['improved'] is True) -- a target
    evaluate() found NOT to help (LOCO RMSE/CSI didn't clear the bar) must never be applied
    blind just because a model happens to exist for it. If NONE of the report's targets are
    improved, nothing is written at all (no wrfinput_out, no netCDF4 import even attempted) and
    this returns None -- the caller (this module's own CLI, or coldstart_sample/case_rebuild's
    sbatch ML-apply step) must treat that as 'skip, keep the original', not an error."""
    if os.path.abspath(wrfinput_in) == os.path.abspath(wrfinput_out):
        raise StateCorrectionError("apply: --out must differ from --wrfinput (refusing to "
                                    "overwrite the input in place)")
    if os.path.exists(wrfinput_out):
        raise StateCorrectionError("apply: --out %r already exists (refusing to overwrite)"
                                    % wrfinput_out)
    with open(report_path) as f:
        report = json.load(f)

    if targets_from_report:
        improved = {k: v for k, v in report.get("targets", {}).items() if v.get("improved")}
        if not improved:
            print("apply: --targets-from-report: no target in %s has verdict 'improved' -- "
                  "nothing written" % report_path)
            print("APPLIED_TARGETS=")
            return None
        report = dict(report)
        report["targets"] = improved

    import netCDF4
    import shutil
    shutil.copyfile(wrfinput_in, wrfinput_out)
    ds = netCDF4.Dataset(wrfinput_out, "r+")
    try:
        _require_core(ds, wrfinput_out)
        nz = len(ds.dimensions["bottom_top"])
        ny = len(ds.dimensions["south_north"])
        nx = len(ds.dimensions["west_east"])
        times = _read_times(ds)
        t_dt = times[0]

        feats = np.empty((nz, ny, nx, len(FEATURE_NAMES)))
        pressure = np.empty((nz, ny, nx))
        for j in range(ny):
            for i in range(nx):
                f_ji, p_ji = _column_features(ds, j, i, t_dt, nz)
                feats[:, j, i, :] = f_ji
                pressure[:, j, i] = p_ji
        feats_flat = feats.reshape(-1, len(FEATURE_NAMES))
        pressure_flat = pressure.reshape(-1)

        dT = np.zeros(nz * ny * nx)
        dQV = np.zeros(nz * ny * nx)
        dU = np.zeros(nz * ny * nx)
        dV = np.zeros(nz * ny * nx)
        cloud = np.zeros(nz * ny * nx)
        precip = np.zeros(nz * ny * nx)
        for name, mask_fn in LEVEL_GROUPS:
            lvl_mask = mask_fn(pressure_flat)
            dT += _predict_grid(report["targets"].get("dT__%s" % name), feats_flat, lvl_mask)
            dQV += _predict_grid(report["targets"].get("dQV__%s" % name), feats_flat, lvl_mask)
            dU += _predict_grid(report["targets"].get("dU__%s" % name), feats_flat, lvl_mask)
            dV += _predict_grid(report["targets"].get("dV__%s" % name), feats_flat, lvl_mask)
            cloud += _predict_grid(report["targets"].get("cloud__%s" % name), feats_flat,
                                    lvl_mask)
            precip += _predict_grid(report["targets"].get("precip__%s" % name), feats_flat,
                                     lvl_mask)

        dT = dT.reshape(nz, ny, nx)
        dQV = dQV.reshape(nz, ny, nx)
        dU = dU.reshape(nz, ny, nx)
        dV = dV.reshape(nz, ny, nx)
        cloud = cloud.reshape(nz, ny, nx)
        precip = precip.reshape(nz, ny, nx)

        theta0 = ds.variables["T"][0] + 300.0
        qv0 = np.asarray(ds.variables["QVAPOR"][0], float)
        theta_new = theta0 + dT
        qv_new = np.maximum(qv0 + dQV, 0.0)

        # supersaturation clamp: move any excess above RH=1.0 into cloud water instead of
        # letting QVAPOR sit supersaturated (spec: "do not create supersaturation above RH
        # 100%; move excess to cloud water").
        p_total = (np.asarray(ds.variables["P"][0], float)
                   + np.asarray(ds.variables["PB"][0], float))
        rh_new = relative_humidity(theta_new, qv_new, p_total, clip=None)
        super_mask = rh_new > 1.0
        if super_mask.any():
            qs = saturation_qs(temp_from_theta(theta_new, p_total), p_total)
            excess = np.where(super_mask, qv_new - qs, 0.0)
            qv_new = np.where(super_mask, qs, qv_new)
            cloud = cloud + excess

        ds.variables["T"][0] = (theta_new - 300.0).astype(ds.variables["T"].dtype)
        ds.variables["QVAPOR"][0] = qv_new.astype(ds.variables["QVAPOR"].dtype)

        u0 = np.asarray(ds.variables["U"][0], float)
        v0 = np.asarray(ds.variables["V"][0], float)
        u_new = u0 + restagger_avg(dU, axis=2)      # west_east_stag axis
        v_new = v0 + restagger_avg(dV, axis=1)      # south_north_stag axis
        ds.variables["U"][0] = u_new.astype(ds.variables["U"].dtype)
        ds.variables["V"][0] = v_new.astype(ds.variables["V"].dtype)

        if "QCLOUD" in ds.variables:
            qc0 = np.asarray(ds.variables["QCLOUD"][0], float)
            ds.variables["QCLOUD"][0] = np.clip(qc0 + cloud, 0.0, None).astype(
                ds.variables["QCLOUD"].dtype)
        if "QRAIN" in ds.variables:
            qr0 = np.asarray(ds.variables["QRAIN"][0], float)
            ds.variables["QRAIN"][0] = np.clip(qr0 + precip, 0.0, None).astype(
                ds.variables["QRAIN"].dtype)

        applied_keys = sorted(report["targets"].keys())
        ds.setncattr("STATE_CORRECTION_MODEL", os.path.basename(report_path))
        ds.setncattr("STATE_CORRECTION_VARS", "T,QVAPOR,U,V,QCLOUD,QRAIN")
        ds.setncattr("STATE_CORRECTION_TARGETS", ",".join(applied_keys))
    finally:
        ds.close()
    # K5 (2026-09-24): a stable, bash-grep-able marker line -- coldstart_sample/case_rebuild's
    # sbatch ML-apply step (running under the grib-env python, not /usr/bin/python3) captures
    # this to record which targets it actually applied in windows_status.txt / the rebuild
    # report, without needing netCDF4/ncdump just to read the attr back.
    print("APPLIED_TARGETS=%s" % ",".join(applied_keys))
    return wrfinput_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="state_correction_model.py", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build-pairs")
    sp.add_argument("--pair", nargs=3, action="append", required=True,
                     metavar=("CASE", "RST", "WRFINPUT"),
                     help="CASE id (used for leave-one-case-out grouping), the WRF restart "
                     "(truth) file, and the cold-start wrfinput (input) file at the same "
                     "simulated time T. Repeatable.")
    sp.add_argument("--out", required=True)
    sp.add_argument("--columns", type=int, default=3000)
    sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("evaluate")
    sp.add_argument("--pairs", required=True, help="pairs.npz from build-pairs")
    sp.add_argument("--out", required=True)
    sp.add_argument("--max-rows-per-pair", type=int, default=None,
                    help="T17(e): reproducibly subsample to at most N rows per pair, for a "
                    "quick run (default: all rows)")
    sp.add_argument("--seed", type=int, default=0,
                    help="T17: seed for --max-rows-per-pair subsampling and for kNN's own "
                    "training-row subsampling")

    sp = sub.add_parser("apply")
    sp.add_argument("--model", required=True, help="report.json from evaluate")
    sp.add_argument("--wrfinput", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--targets-from-report", action="store_true",
                     help="apply ONLY the targets evaluate() found improved (report.json's own "
                     "targets[*].improved) -- refuse a target's correction just because a "
                     "model happens to exist for it. Writes nothing if none are improved.")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "build-pairs":
            stats = build_pairs(args.pair, args.out, n_columns=args.columns, seed=args.seed)
            print("build-pairs: wrote %s (%d rows, %d pairs)"
                  % (args.out, stats["n_rows"], stats["n_pairs"]))
            return 0
        if args.cmd == "evaluate":
            evaluate(args.pairs, args.out, max_rows_per_pair=args.max_rows_per_pair,
                    seed=args.seed)
            return 0
        if args.cmd == "apply":
            out = apply_correction(args.model, args.wrfinput, args.out,
                                    targets_from_report=args.targets_from_report)
            if out is None:
                return 0   # --targets-from-report with nothing improved: already explained
            print("apply: wrote %s" % out)
            return 0
    except StateCorrectionError as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

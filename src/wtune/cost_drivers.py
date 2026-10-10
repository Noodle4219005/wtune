"""wtune_cost_drivers -- weather cost indicators c(t) for probe-point selection and weighting
(decision 0060 in the wiki: "WTune 依天氣成本指標選取預測點並加權").

Why: coldstart_sample.select_windows places probe points by sun position / time only, and the
whole-run integrator (coldstart_sample.predicted_compute_total) implicitly weights every point by
piecewise-linear interpolation. Neither looks at the WEATHER, whose cost (microphysics, convection)
varies by tens of percent along the run (elevconv: +44% at 10 UTC). This module computes, without
running WRF, an hourly cost proxy c(t) from pluggable data sources and derives (a) which times to
probe (coldstart_sample.select_windows(cost_lookup=...)) and (b) two alternative whole-run weightings
(weighted_totals, attached by wtune_probe_plan.combine). It can only fix the TIME placement of the
points; it cannot fix a reconstruction that measures low at the peak (decision 0060, H2).

Provider interface (ordered fallback):

    provider.name                      "case_forcing" | "era5" | "gfs" | "stage4" | "mrms"
    provider.fetch(domain_bounds, t_start, t_end, interval_h) -> (series, provenance)
        domain_bounds  (lat0, lat1, lon0, lon1), lon -180..180   (d01 bounds for now)
        series         {datetime: {indicator: float}}  (any subset of INDICATORS; sparse in time is
                       fine -- build_cost_series interpolates to the hourly grid)
        provenance     dict: source, fields used, files/URLs, auth, notes
    Raises ProviderError on failure (never returns fake data).

Indicators (INDICATORS), aggregated over the domain, per time:
    cloud_col    mean total-column cloud liquid + ice water, kg m-2
    precip_area  fraction of points with precipitation > 1 mm/h
    cape_area    fraction of points with CAPE > 1000 J/kg
    sunlit       fraction of points with the sun up (computed here, wtune_predict.sunlit_fraction)
    wind_max     max 10 m wind speed, m/s (CFL risk)

build_cost_series() asks providers in order and keeps each indicator from the FIRST provider that
supplies it; a provider that raises is recorded in provenance and skipped; it stops once the four
weather indicators are all covered. It never raises.

c(t): each indicator is z-scored over the run's hourly grid (zero-variance ones are dropped), then
c = sum_i w_i z_i / sum_i w_i. Default w_i come from the physics cost shares of the case profile
(wtune_case._build_cost_structure 'phases_ranked' weights, which come from COST_WEIGHT) when given,
else DEFAULT_PHASE_WEIGHTS: cloud_col and precip_area <- microphysics (split in two), cape_area <-
cumulus, sunlit <- radiation, wind_max <- dynamics. A second series c_noradiation drops 'sunlit':
the regression target s_normal EXCLUDES radiation steps (those go through r(t)), so regressing it on
a sun term would double count.
"""
from __future__ import annotations

import datetime
import glob
import math
import os
import re
import statistics
import sys

# `python -m wtune_cost_drivers` from src/ must resolve coldstart_sample, era5_intermediate, ...
# the way wtune_tune / wtune_rebuild_predict do (tools/ and predictor/ are top-level import roots).
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from wtune import predict as predict  # noqa: E402

INDICATORS = ("cloud_col", "precip_area", "cape_area", "sunlit", "wind_max")
WEATHER_INDICATORS = ("cloud_col", "precip_area", "cape_area", "wind_max")
PRECIP_MM_H = 1.0
CAPE_J_KG = 1000.0
G = 9.80665
DEFAULT_SOURCES = ("case_forcing", "era5", "gfs", "stage4", "mrms")
# physics cost shares used when no case cost structure is given (COST_WEIGHT scale: low=1, med=2,
# high=4); cloud_col and precip_area split the microphysics share.
# dynamics is a mostly FIXED cost (baseline RK3 + advection; its profile weight is 5 for a typical
# case), and 10 m wind only modulates it through CFL/acoustic substepping -- so wind_max gets a
# fraction of it, else a nearly constant-cost term would dominate c.
WIND_SHARE_OF_DYNAMICS = 0.2
DEFAULT_PHASE_WEIGHTS = {"microphysics": 2.0, "cumulus": 1.0, "radiation": 1.0, "dynamics": 1.0}


class ProviderError(Exception):
    pass


# ---------------------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------------------
def _hours(t_start, t_end):
    n = int(math.floor((t_end - t_start).total_seconds() / 3600.0 + 1e-9))
    return [t_start + datetime.timedelta(hours=h) for h in range(n + 1)]


def _interp(xs, ys, x):
    """Linear interpolation, clamped at both ends. xs ascending."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    f = (x - xs[lo]) / float(xs[hi] - xs[lo])
    return ys[lo] + f * (ys[hi] - ys[lo])


def _zscore(vals):
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    return None if sd < 1e-12 else [(v - m) / sd for v in vals]


def indicator_weights(cost_structure=None):
    """{indicator: weight}. cost_structure: wtune_case's expected_cost_structure dict
    ('phases_ranked': [{'phase', 'weight'}, ...]) or None for DEFAULT_PHASE_WEIGHTS."""
    ph = dict(DEFAULT_PHASE_WEIGHTS)
    if cost_structure:
        for r in cost_structure.get("phases_ranked", []):
            if r.get("phase") in ph and r.get("weight") is not None:
                ph[r["phase"]] = float(r["weight"])
    return {"cloud_col": ph["microphysics"] / 2.0, "precip_area": ph["microphysics"] / 2.0,
            "cape_area": ph["cumulus"], "sunlit": ph["radiation"],
            "wind_max": WIND_SHARE_OF_DYNAMICS * ph["dynamics"]}


class CostSeries:
    """Hourly indicators + cost proxy on the grid t0, t0+1h, ...; see module docstring."""

    def __init__(self, t0, times, indicators, weights, provenance, used):
        self.t0, self.times = t0, times
        self.indicators = indicators          # {name: [hourly floats]}
        self.weights = weights                # {name: weight actually applied in c}
        self.provenance = provenance          # [{provider, status, ...}]
        self.used = used                      # {indicator: provider name}
        self.c = self.c_noradiation = None
        self.dropped = []                     # constant (zero-variance) indicators
        self._build()

    def _combine(self, skip=()):
        zs, ws = [], []
        for name, vals in self.indicators.items():
            if name in skip or name in self.dropped:
                continue
            z = _zscore(vals)
            if z is None:
                if name not in self.dropped:
                    self.dropped.append(name)
                continue
            zs.append(z)
            ws.append(self.weights.get(name, 1.0))
        if not zs or sum(ws) <= 0:
            return None
        sw = float(sum(ws))
        return [sum(w * z[i] for w, z in zip(ws, zs)) / sw for i in range(len(self.times))]

    def _build(self):
        self.c = self._combine()
        self.c_noradiation = self._combine(skip=("sunlit",))
        # A c made of 'sunlit' alone is just the time-based day/night rule in disguise (its peak
        # lands on daytime hours): cost mode needs at least one WEATHER indicator that varies.
        if not any(n in WEATHER_INDICATORS and n not in self.dropped for n in self.indicators):
            self.c = self.c_noradiation = None

    @property
    def usable(self):
        return self.c is not None

    def why_unusable(self):
        """Loud, specific reason (or None when usable): names every provider's error/status."""
        if self.usable:
            return None
        det = "; ".join("%s: %s" % (p.get("provider"), p.get("error") or p.get("status"))
                        for p in self.provenance) or "no provider configured"
        return ("no weather indicator (cloud_col, precip_area, cape_area, wind_max) varies over the "
                "run (%s)" % det)

    def value_at(self, t, series="c"):
        vals = self.c if series == "c" else self.c_noradiation
        if vals is None:
            return None
        xs = [(x - self.t0).total_seconds() for x in self.times]
        return _interp(xs, vals, (t - self.t0).total_seconds())

    def mean_over(self, t, window_s, series="c", n=8):
        """Mean of c over [t, t+window_s] (n+1 samples)."""
        vs = [self.value_at(t + datetime.timedelta(seconds=window_s * i / float(n)), series)
              for i in range(n + 1)]
        return None if vs[0] is None else statistics.fmean(vs)

    def cost_lookup(self, candidates, window_s):
        """{candidate: mean c over its measured window} for coldstart_sample.select_windows."""
        return {c: self.mean_over(c, window_s) for c in candidates} if self.usable else {}

    def precip_lookup(self, candidates, window_s):
        """{candidate: mean precip_area over its window} (select_windows' precip_lookup), or {}."""
        vals = self.indicators.get("precip_area")
        if not vals:
            return {}
        xs = [(x - self.t0).total_seconds() for x in self.times]
        out = {}
        for c in candidates:
            s = [_interp(xs, vals, (c - self.t0).total_seconds() + window_s * i / 8.0)
                 for i in range(9)]
            out[c] = statistics.fmean(s)
        return out

    def summary(self):
        return dict(sources=self.used, weights={k: v for k, v in self.weights.items()
                                                 if k in self.indicators and k not in self.dropped},
                    dropped_constant=list(self.dropped), provenance=self.provenance,
                    hours=len(self.times))


def build_cost_series(providers, bounds, t_start, t_end, sample_points=None, weights=None,
                      cost_structure=None, interval_h=1):
    """CostSeries over the hourly grid [t_start .. t_end]. Never raises: provider failures are in
    .provenance; .usable is False when no varying indicator was obtained."""
    times = _hours(t_start, t_end)
    wts = dict(indicator_weights(cost_structure))
    wts.update(weights or {})
    xs = [(t - t_start).total_seconds() for t in times]
    got, used, prov = {}, {}, []
    for p in providers:
        if all(i in got for i in WEATHER_INDICATORS):
            break
        name = getattr(p, "name", type(p).__name__)
        try:
            series, pv = p.fetch(bounds, t_start, t_end, interval_h)
            pv = dict(pv or {})
        except Exception as e:  # noqa: BLE001 -- ordered fallback, by design
            prov.append(dict(provider=name, status="failed", error="%s: %s" % (type(e).__name__, e)))
            continue
        pts = sorted(series.items())
        taken = []
        for ind in WEATHER_INDICATORS:
            if ind in got:
                continue
            pairs = [((t - t_start).total_seconds(), v[ind]) for t, v in pts
                     if ind in v and v[ind] is not None and math.isfinite(v[ind])]
            if not pairs:
                continue
            px, py = zip(*pairs)
            got[ind] = [_interp(list(px), list(py), x) for x in xs]
            used[ind] = name
            taken.append(ind)
        pv.update(provider=name, status="ok" if taken else "no-usable-indicator",
                  indicators_used=taken, n_times=len(pts))
        prov.append(pv)
    if sample_points:
        got["sunlit"] = [predict.sunlit_fraction(sample_points, t) for t in times]
        used["sunlit"] = "computed (wtune_predict.sunlit_fraction)"
    return CostSeries(t_start, times, got, wts, prov, used)


# ---------------------------------------------------------------------------------------
# weighted totals (called from wtune_probe_plan.combine)
# ---------------------------------------------------------------------------------------
def _integrate(step_s, dt, stepra, run_start, sample_points, r_night, r_day):
    """sum_i s_i + sum over radiation steps (r(t_i)-1) s_i, r(t)=r_night+(r_day-r_night)*sunlit(t)
    -- the arithmetic of coldstart_sample.predicted_compute_total, on caller-supplied s_i."""
    n = len(step_s)
    flags = predict.classify_steps(n, stepra)
    total = sum(step_s)
    for i in range(n):
        if not flags[i]:
            continue
        t_i = run_start + datetime.timedelta(seconds=(i + 1) * dt)
        frac = predict.sunlit_fraction(sample_points, t_i) if sample_points else None
        r_t = (r_night + (r_day - r_night) * frac) if frac is not None else (r_night + r_day) / 2.0
        total += (r_t - 1.0) * step_s[i]
    return total


def _step_hour(i, dt, n_hours):
    return min(int((i + 0.5) * dt // 3600), n_hours - 1)


def weighted_totals(cs, measurements, run_start, run_end, dt, stepra, sample_points, r_night,
                    r_day):
    """{'stratified_compute_s', 'strata', 'regression_compute_s' (None if <3 points), 'alpha',
    'beta', 'r2', 'n_points'} -- compute-only totals (combine() adds init_s).

    Stratified: the H = ceil(run hours) hourly values c_h = c(run_start + h) are split into
    K = len(measurements) quantile strata (edges at j/K quantiles; ties stay together), w_k =
    hours_k / H. Each measured point belongs to the stratum of c at its own time; s_k = mean
    s_normal of its points; a stratum with no point takes the s of the point whose c is nearest the
    stratum's median c. Every step i inside hour h gets s_{k(h)}; the total is
    sum_i s_i + sum_{radiation steps} (r(t_i)-1) s_i, i.e. EXACTLY the radiation treatment of
    predicted_compute_total, so with r == 1 it equals N_steps * sum_k w_k s_k.

    Regression: least squares s = alpha + beta c on the points (x = c_noradiation at the point's
    time), beta clamped >= 0 (then alpha = mean s); every step in hour h gets
    max(0, alpha + beta c_noradiation(h)), same radiation treatment; R^2 = 1 - SSres/SStot
    (None when SStot = 0)."""
    steps_n = int(round((run_end - run_start).total_seconds() / dt))
    n_hours = max(1, int(math.ceil(steps_n * dt / 3600.0 - 1e-9)))
    out = dict(stratified_compute_s=None, strata=None, regression_compute_s=None, alpha=None,
               beta=None, r2=None, n_points=len(measurements))
    if not cs.usable or not measurements:
        return out
    c_h = [cs.value_at(run_start + datetime.timedelta(hours=h), "c") for h in range(n_hours)]
    K = max(1, len(measurements))
    srt = sorted(c_h)
    edges = [srt[min(n_hours - 1, int(j * n_hours / float(K)))] for j in range(1, K)]

    def stratum(c):
        return sum(1 for e in edges if c >= e)

    k_of_h = [stratum(c) for c in c_h]
    pts = [(cs.value_at(pm.at, "c"), pm.s_normal) for pm in measurements]
    s_k, strata = {}, []
    for k in range(K):
        hs = [h for h in range(n_hours) if k_of_h[h] == k]
        if not hs:
            continue
        mine = [s for c, s in pts if stratum(c) == k]
        med = statistics.median(c_h[h] for h in hs)
        s = statistics.fmean(mine) if mine else min(pts, key=lambda p: abs(p[0] - med))[1]
        s_k[k] = s
        strata.append(dict(k=k + 1, hours=len(hs), weight=len(hs) / float(n_hours), s=s,
                           n_points=len(mine), c_median=med))
    step_s = [s_k[k_of_h[_step_hour(i, dt, n_hours)]] for i in range(steps_n)]
    out["stratified_compute_s"] = _integrate(step_s, dt, stepra, run_start, sample_points,
                                             r_night, r_day)
    out["strata"] = strata
    if len(measurements) >= 3:
        xs = [cs.value_at(pm.at, "c_noradiation") for pm in measurements]
        ys = [pm.s_normal for pm in measurements]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 1e-12 else 0.0
        beta = max(0.0, beta)
        alpha = my - beta * mx if beta > 0 else my
        sst = sum((y - my) ** 2 for y in ys)
        ssr = sum((y - (alpha + beta * x)) ** 2 for x, y in zip(xs, ys))
        out["r2"] = (1.0 - ssr / sst) if sst > 1e-18 else None
        cr_h = [cs.value_at(run_start + datetime.timedelta(hours=h), "c_noradiation")
                for h in range(n_hours)]
        step_r = [max(0.0, alpha + beta * cr_h[_step_hour(i, dt, n_hours)]) for i in range(steps_n)]
        out["regression_compute_s"] = _integrate(step_r, dt, stepra, run_start, sample_points,
                                                 r_night, r_day)
        out["alpha"], out["beta"] = alpha, beta
    return out


def attach_weighted_totals(result, cs, measurements, run_start, run_end, dt, stepra,
                           sample_points):
    """Adds stratified_total_s / regression_total_s / regression_r2 / cost_indicators to a
    combine() result dict (totals include init_s, like total_s; total_s itself is untouched).
    spread_pct = (max - min) / total_s over {total_s, stratified, regression} -- a credibility
    indicator, not an error bar. Never raises."""
    try:
        w = weighted_totals(cs, measurements, run_start, run_end, dt, stepra, sample_points,
                            result["r_night"], result["r_day"])
        init_s = result.get("init_s") or 0.0
        strat = None if w["stratified_compute_s"] is None else w["stratified_compute_s"] + init_s
        reg = None if w["regression_compute_s"] is None else w["regression_compute_s"] + init_s
        vals = [v for v in (result["total_s"], strat, reg) if v is not None]
        result["stratified_total_s"] = strat
        result["regression_total_s"] = reg
        result["regression_r2"] = w["r2"]
        info = cs.summary()
        info.update(strata=w["strata"], regression_alpha=w["alpha"], regression_beta=w["beta"],
                    spread_pct=(100.0 * (max(vals) - min(vals)) / result["total_s"]
                                if len(vals) > 1 and result["total_s"] else None))
        result["cost_indicators"] = info
    except Exception as e:  # noqa: BLE001 -- informational; must never break combine()
        result["cost_indicators"] = dict(error="%s: %s" % (type(e).__name__, e))
    return result


# ---------------------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------------------
def _np():
    import numpy as np
    return np


def _in_bounds(lat, lon, bounds):
    np = _np()
    lat0, lat1, lon0, lon1 = bounds
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    return (lat >= lat0) & (lat <= lat1) & (lon180 >= lon0) & (lon180 <= lon1)


class CaseForcingProvider:
    """(a) The case's own forcing as WRF sees it: met_em.d01.* NetCDF files (cloud_col from QC+QI
    over PRES, wind_max from the lowest metgrid level's UU/VV) and/or WPS-intermediate files
    <PREFIX>:YYYY-MM-DD_HH (the ERA5 rebuild path's output, tools/era5_intermediate.py ->
    scenario/wps_intermediate.read_fields: QC/QI pressure levels, UU/VV at the 200100 Pa surface
    flag). No network. Gives cloud_col and wind_max only (no precipitation or CAPE in either)."""
    name = "case_forcing"

    def __init__(self, dirs, prefix=None):
        self.dirs = [d for d in dirs if d]
        self.prefix = prefix

    def fetch(self, bounds, t_start, t_end, interval_h):
        series, files = {}, []
        for t, path, kind in self._files(t_start, t_end):
            try:
                vals = self._met_em(path) if kind == "met_em" else self._intermediate(path, bounds)
            except Exception as e:  # noqa: BLE001
                raise ProviderError("%s: %s: %s" % (path, type(e).__name__, e))
            if vals:
                series.setdefault(t, {}).update(vals)
                files.append(path)
        if not series:
            series, files, note = self._wrf_inputs(t_start, t_end)
            if not series:
                raise ProviderError("no readable met_em.d01.*, <prefix>:YYYY-MM-DD_HH, wrfinput_d01 "
                                    "or wrfbdy_d01 in %s for %s..%s" % (self.dirs, t_start, t_end))
            return series, dict(source="case wrfinput_d01/wrfbdy_d01 (FALLBACK: no met_em or "
                                       "intermediate files; no network)", files=files, auth="none",
                                fields=note)
        return series, dict(source="case forcing files (no network)", files=files, auth="none",
                            fields="QC+QI (cloud_col), UU/VV lowest level (wind_max)")

    def _wrf_inputs(self, t_start, t_end):
        """FALLBACK when no met_em / intermediate files exist. wrfbdy_d01 (when present) gives
        the boundary state at every bdy time (Times[i]; BXS/BXE/BYS/BYE hold the VALUE at that
        time, BT* the tendency; the values are MASS-COUPLED, so they are divided by the domain-
        mean MU+MUB of wrfinput_d01 -- an approximation, terrain varies MU by ~10%): cloud_col = sum_k mean_edge(QCLOUD+QICE)_k dp_k / g with dp_k from
        the domain-mean P+PB profile of wrfinput_d01, wind_max = max(|U|, |V|) at the lowest level
        of the outermost boundary cells. Weak signal: it describes the lateral boundary, not the
        interior (a convective peak inside the domain is invisible to it). wrfinput_d01 alone gives
        t0 only (QCLOUD+QICE over P+PB, max of U10/V10). Returns (series, files, field note)."""
        np = _np()
        try:
            import netCDF4
        except ImportError:
            return {}, [], ""
        wi_path = self._find("wrfinput_d01")
        bdy_path = self._find("wrfbdy_d01")
        series, files, notes, pmean, mu_mean = {}, [], [], None, None
        wi_t0 = wi_row = None
        if wi_path:
            try:
                with netCDF4.Dataset(wi_path) as nc:
                    t0 = _wrf_time(nc.variables["Times"][0])
                    pres = np.asarray(nc.variables["P"][0], dtype=float) + \
                        np.asarray(nc.variables["PB"][0], dtype=float)
                    pmean = pres.reshape(pres.shape[0], -1).mean(axis=1)
                    if "MU" in nc.variables and "MUB" in nc.variables:
                        mu_mean = float(np.mean(np.asarray(nc.variables["MU"][0], dtype=float) +
                                                np.asarray(nc.variables["MUB"][0], dtype=float)))
                    row = {}
                    if "QCLOUD" in nc.variables:
                        q = np.asarray(nc.variables["QCLOUD"][0], dtype=float).clip(min=0.0)
                        if "QICE" in nc.variables:
                            q = q + np.asarray(nc.variables["QICE"][0], dtype=float).clip(min=0.0)
                        row["cloud_col"] = float(np.mean(_column_integral(q, pres)))
                    if "U10" in nc.variables and "V10" in nc.variables:
                        row["wind_max"] = float(np.max(np.hypot(
                            np.asarray(nc.variables["U10"][0], dtype=float),
                            np.asarray(nc.variables["V10"][0], dtype=float))))
                    if row and t_start <= t0 <= t_end:
                        wi_t0, wi_row = t0, row
                        files.append(wi_path)
                        notes.append("wrfinput_d01 (t0 only): QCLOUD+QICE over P+PB, U10/V10")
            except Exception:  # noqa: BLE001 -- fallback of a fallback: try wrfbdy alone
                pass
        if bdy_path and mu_mean:       # wrfbdy fields are mass-coupled: needs wrfinput's MU+MUB
            try:
                with netCDF4.Dataset(bdy_path) as nc:
                    times = [_wrf_time(x) for x in nc.variables["Times"][:]]
                    nz = nc.variables["U_BXS"].shape[2] if "U_BXS" in nc.variables else None
                    w = None
                    if pmean is not None and nz is not None and len(pmean) == nz:
                        pi = np.concatenate([[pmean[0] + 0.5 * (pmean[0] - pmean[1])],
                                             0.5 * (pmean[1:] + pmean[:-1]),
                                             [pmean[-1] - 0.5 * (pmean[-2] - pmean[-1])]])
                        w = np.abs(np.diff(pi)) / G                       # kg m-2 per unit q
                    for i, t in enumerate(times):
                        if not (t_start <= t <= t_end):
                            continue
                        row = {}
                        if w is not None and "QCLOUD_BXS" in nc.variables:
                            qk = []
                            for side in ("BXS", "BXE", "BYS", "BYE"):
                                tot = np.asarray(nc.variables["QCLOUD_" + side][i, 0], dtype=float)
                                if "QICE_" + side in nc.variables:
                                    tot = tot + np.asarray(nc.variables["QICE_" + side][i, 0],
                                                           dtype=float)
                                qk.append(tot.clip(min=0.0).mean(axis=1) / mu_mean)
                            row["cloud_col"] = float(np.mean(qk, axis=0) @ w)
                        if "U_BXS" in nc.variables and "V_BYS" in nc.variables:
                            row["wind_max"] = float(max(
                                np.abs(np.asarray(nc.variables[n][i, 0, 0], dtype=float)).max()
                                for n in ("U_BXS", "U_BXE", "V_BYS", "V_BYE")) / mu_mean)
                        if row:
                            series.setdefault(t, {}).update(
                                {k: v for k, v in row.items() if k not in series.get(t, {})})
                    files.append(bdy_path)
                    notes.append("wrfbdy_d01 boundary values at each bdy time (weak: edge cells "
                                 "only, decoupled by domain-mean MU+MUB); needs wrfinput_d01")
            except Exception:  # noqa: BLE001
                pass
        if wi_row:      # wrfinput fills only what the boundary file did not give at t0 (the two
            # measure different things -- whole-domain vs edge cells -- and must not be mixed)
            series[wi_t0] = dict(wi_row, **series.get(wi_t0, {}))
        return series, files, "; ".join(notes)

    def _find(self, name):
        for d in self.dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    def _files(self, t_start, t_end):
        out = {}
        for d in self.dirs:
            for p in glob.glob(os.path.join(d, "met_em.d01.*.nc")):
                m = re.search(r"met_em\.d01\.(\d{4}-\d\d-\d\d)_(\d\d)[:_](\d\d)[:_](\d\d)\.nc$", p)
                if m:
                    t = datetime.datetime.strptime(m.group(1) + m.group(2) + m.group(3) + m.group(4),
                                                   "%Y-%m-%d%H%M%S")
                    out.setdefault(t, (p, "met_em"))
            for p in glob.glob(os.path.join(d, (self.prefix or "*") + ":????-??-??_??")):
                try:
                    t = datetime.datetime.strptime(p.rsplit(":", 1)[1], "%Y-%m-%d_%H")
                except ValueError:
                    continue
                out.setdefault(t, (p, "intermediate"))
        return [(t, p, k) for t, (p, k) in sorted(out.items()) if t_start <= t <= t_end]

    @staticmethod
    def _met_em(path):
        np = _np()
        import netCDF4
        with netCDF4.Dataset(path) as nc:
            res = {}
            pres = np.asarray(nc.variables["PRES"][0], dtype=float) if "PRES" in nc.variables else None
            if pres is not None and "QC" in nc.variables:
                q = np.nan_to_num(np.asarray(nc.variables["QC"][0], dtype=float)).clip(min=0.0)
                if "QI" in nc.variables:
                    q = q + np.nan_to_num(np.asarray(nc.variables["QI"][0], dtype=float)).clip(min=0.0)
                res["cloud_col"] = float(np.mean(_column_integral(q, pres)))
            if "UU" in nc.variables and "VV" in nc.variables and pres is not None:
                k0 = int(np.argmax(pres.reshape(pres.shape[0], -1).mean(axis=1)))   # lowest level
                u = np.asarray(nc.variables["UU"][0, k0], dtype=float)
                v = np.asarray(nc.variables["VV"][0, k0], dtype=float)
                ny, nx = min(u.shape[0], v.shape[0]), min(u.shape[1], v.shape[1])
                res["wind_max"] = float(np.nanmax(np.hypot(u[:ny, :nx], v[:ny, :nx])))
            return res

    @staticmethod
    def _intermediate(path, bounds):
        np = _np()
        from wtune.scenario import wps_intermediate as wi
        recs = wi.read_fields(path)
        res, levels = {}, {"QC": {}, "QI": {}}
        uu = vv = None
        for r in recs:
            if r.iproj != 0:
                continue
            lat = r.startlat + r.deltalat * np.arange(r.ny)
            lon = r.startlon + r.deltalon * np.arange(r.nx)
            mask = _in_bounds(lat[:, None] + 0 * lon[None, :], lon[None, :] + 0 * lat[:, None], bounds)
            if not mask.any():
                continue
            slab = np.asarray(r.slab, dtype=float)
            if r.field in levels:
                levels[r.field][r.xlvl] = np.where(mask, np.nan_to_num(slab).clip(min=0.0), 0.0), mask
            elif r.field == "UU" and r.xlvl == wi.SURFACE_PA:
                uu = (slab, mask)
            elif r.field == "VV" and r.xlvl == wi.SURFACE_PA:
                vv = (slab, mask)
        qc = levels["QC"]
        if qc:
            plev = sorted(qc, reverse=True)
            q = np.stack([qc[p][0] + (levels["QI"][p][0] if p in levels["QI"] else 0.0)
                          for p in plev])
            mask = qc[plev[0]][1]
            pres = np.broadcast_to(np.array(plev)[:, None, None], q.shape)
            res["cloud_col"] = float(_column_integral(q, pres)[mask].mean())
        if uu is not None and vv is not None:
            res["wind_max"] = float(np.nanmax(np.where(uu[1], np.hypot(uu[0], vv[0]), np.nan)))
        return res


def _wrf_time(chars):
    """datetime from a WRF 'Times' character row (b'2022-05-12_00:00:00' as a char array)."""
    txt = b"".join(chars.tolist()).decode() if hasattr(chars, "tolist") and not isinstance(chars, bytes) \
        else chars.decode()
    return datetime.datetime.strptime(txt.strip("\x00 "), "%Y-%m-%d_%H:%M:%S")


def _column_integral(q, pres):
    """Column integral of q (kg/kg) over pressure, kg m-2: trapezoid in p, per column.
    q, pres: (nlev, ny, nx), levels in any consistent order."""
    np = _np()
    dp = np.abs(np.diff(pres, axis=0))
    return (0.5 * (q[1:] + q[:-1]) * dp).sum(axis=0) / G


class Era5Provider:
    """(b) ERA5 single-level fields from the public ARCO store, via the repo's one reader
    tools/era5_intermediate.open_arco() (era5_intermediate.py:204; also post/recipes/era5.py:31
    open_era5 + :48 check_time_covered, post/fetch/era5.py = `wtune post fetch era5`) and its
    crop_indices() (:115). Anonymous (gcsfs token='anon'), hourly, needs internet (login node on
    scc, not compute nodes). Variables (first present name wins; absent ones are reported):
    total_column_cloud_liquid_water + total_column_cloud_ice_water (cloud_col), total_precipitation
    (precip_area, m/h -> mm/h), convective_available_potential_energy (cape_area),
    10m u/v components (wind_max)."""
    name = "era5"
    _V = dict(cl="total_column_cloud_liquid_water", ci="total_column_cloud_ice_water",
              tp="total_precipitation", cape="convective_available_potential_energy",
              u10="10m_u_component_of_wind", v10="10m_v_component_of_wind")

    def __init__(self, ds=None, margin=0.0):
        self._ds, self.margin = ds, margin

    def fetch(self, bounds, t_start, t_end, interval_h):
        np = _np()
        try:
            ds = self._ds
            if ds is None:
                from wtune.post.recipes import era5 as recipe_era5
                ds = recipe_era5.open_era5()
                recipe_era5.check_time_covered(ds, t_start)
                recipe_era5.check_time_covered(ds, t_end)
            import wtune.tools.era5_intermediate as era5_mod
            lat_idx, lon_idx, _lv = era5_mod.crop_indices(ds["latitude"].values,
                                                          ds["longitude"].values, bounds, self.margin)
        except Exception as e:  # noqa: BLE001
            raise ProviderError("ERA5 ARCO unavailable: %s: %s" % (type(e).__name__, e))
        times = [t_start + datetime.timedelta(hours=h)
                 for h in range(0, int((t_end - t_start).total_seconds() // 3600) + 1,
                                max(1, int(round(interval_h))))]
        have = {k: v for k, v in self._V.items() if v in ds}
        missing = sorted(set(self._V.values()) - set(have.values()))
        if not have:
            raise ProviderError("ERA5 store has none of the expected variables %s" % self._V)
        # Nearest-hour lookup in numpy + isel: label-based ds.sel(..., method="nearest") raises
        # InvalidIndexError("Reindexing only valid with uniquely valued Index objects") when the
        # store's time index (or a caller-supplied one) holds duplicate stamps.
        tarr = np.asarray(ds["time"].values).astype("datetime64[s]")

        def nearest(t):
            d = np.abs(tarr - np.datetime64(t, "s"))
            i = int(np.argmin(d))
            if d[i] > np.timedelta64(3600, "s"):
                raise ProviderError("ERA5 has no time within 1 h of %s" % t)
            return i
        times = sorted(set(times))                         # one fetch per distinct hour

        def one(t):
            snap = ds.isel(time=nearest(t))

            def grid(key):
                a = np.asarray(snap[have[key]].values, dtype=float)
                return a[..., lat_idx, :][..., lon_idx]
            row = {}
            if "cl" in have:
                row["cloud_col"] = float(np.mean(grid("cl") + (grid("ci") if "ci" in have else 0.0)))
            if "tp" in have:
                row["precip_area"] = float(np.mean(grid("tp") * 1000.0 > PRECIP_MM_H))
            if "cape" in have:
                row["cape_area"] = float(np.mean(grid("cape") > CAPE_J_KG))
            if "u10" in have and "v10" in have:
                row["wind_max"] = float(np.max(np.hypot(grid("u10"), grid("v10"))))
            return t, row
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as ex:     # one global chunk per variable-hour
            series = dict(ex.map(one, times))
        return series, dict(source="ARCO-ERA5 gcs://gcp-public-data-arco-era5 (0.25 deg, hourly)",
                            auth="anonymous", variables=have, missing_variables=missing,
                            note="precip_area from total_precipitation (1 h accumulation)")


class GfsProvider:
    """(c) NOAA GFS 0.25 deg on the AWS open bucket noaa-gfs-bdp-pds (anonymous), per-message
    byte-range .idx subsetting through the existing tools/metdata.fetch_message/fetch_idx_message
    (metdata.py:371/:410) and metdata.open_grib_field (:429, cfgrib). Valid time t is served from
    the nearest cycle at/before t (metdata.gfs_cycle_for :121): f000 analysis at 00/06/12/18Z,
    f003/f006.. forecast between; APCP via metdata.resolve_apcp_request (f000 -> f003 bucket).
    Default cadence 3 h (build_cost_series interpolates to hourly). cfgrib shortName/typeOfLevel
    mapping of CWAT/CAPE is NOT yet verified against a downloaded message (same caveat as
    metdata's own STATE_FIELD_SPECS)."""
    name = "gfs"
    SPECS = dict(
        cwat=dict(var="CWAT", level="entire atmosphere (considered as a single layer)",
                  cfgrib_short="cwat", cfgrib_level="atmosphereSingleLayer"),
        cape=dict(var="CAPE", level="surface", cfgrib_short="cape", cfgrib_level="surface"),
    )

    def __init__(self, cache_dir=None, source="aws", step_h=3):
        self.cache_dir = cache_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "wtune_gfs_cache")
        self.source, self.step_h = source, step_h

    def fetch(self, bounds, t_start, t_end, interval_h):
        try:
            import wtune.tools.metdata as md
        except Exception as e:  # noqa: BLE001
            raise ProviderError("tools.metdata not importable: %s" % e)
        np = _np()
        step = max(self.step_h, int(round(interval_h)))
        t = t_start.replace(minute=0, second=0, microsecond=0)
        series, errs = {}, []
        while t <= t_end:
            row = {}
            cyc, fh = md.gfs_cycle_for(t)
            specs = dict(md.STATE_FIELD_SPECS)
            specs.update(self.SPECS)
            def get(key, c=cyc, f=fh):
                path = md.fetch_message(key, c, f, self.source, self.cache_dir, specs=specs)
                la, lo, v = md.open_grib_field(path, specs[key]["cfgrib_short"],
                                               typeOfLevel_filter=specs[key]["cfgrib_level"])
                la, lo, v = md.crop_bbox(la, lo, v, bounds[0], bounds[1], bounds[2], bounds[3],
                                         margin_deg=0.0)
                return np.asarray(v, dtype=float)
            for ind, key, fn in (("cloud_col", "cwat", lambda v: float(np.mean(v))),
                                 ("cape_area", "cape", lambda v: float(np.mean(v > CAPE_J_KG)))):
                try:
                    row[ind] = fn(get(key))
                except Exception as e:  # noqa: BLE001
                    errs.append("%s %s: %s: %s" % (t.isoformat(), key, type(e).__name__, e))
            try:
                u, v = get("u10"), get("v10")
                row["wind_max"] = float(np.max(np.hypot(u, v)))
            except Exception as e:  # noqa: BLE001
                errs.append("%s wind: %s: %s" % (t.isoformat(), type(e).__name__, e))
            try:
                ac, af, win, _w = md.resolve_apcp_request(cyc, fh)
                path = md.fetch_message("apcp", ac, af, self.source, self.cache_dir, specs=specs)
                la, lo, v = md.open_grib_field(path, specs["apcp"]["cfgrib_short"],
                                               typeOfLevel_filter=specs["apcp"]["cfgrib_level"])
                la, lo, v = md.crop_bbox(la, lo, v, bounds[0], bounds[1], bounds[2], bounds[3],
                                         margin_deg=0.0)
                row["precip_area"] = float(np.mean(np.asarray(v, dtype=float) / win > PRECIP_MM_H))
            except Exception as e:  # noqa: BLE001
                errs.append("%s apcp: %s: %s" % (t.isoformat(), type(e).__name__, e))
            if row:
                series[t] = row
            t += datetime.timedelta(hours=step)
        if not series:
            raise ProviderError("GFS: no field obtained (%s)" % "; ".join(errs[:3]))
        return series, dict(source="NOAA GFS 0.25deg, AWS noaa-gfs-bdp-pds", auth="anonymous",
                            cadence_h=step, errors=errs[:10])


def _grib_precip_area(path, bounds):
    """Fraction of in-bounds points with > 1 mm of 1 h accumulation (Stage IV / MRMS QPE GRIB2
    via cfgrib; 2-D or 1-D lat/lon; negative values = missing)."""
    np = _np()
    import xarray as xr
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        v = np.asarray(ds[list(ds.data_vars)[0]].values, dtype=float)
        lat = np.asarray(ds["latitude"].values, dtype=float)
        lon = np.asarray(ds["longitude"].values, dtype=float)
    finally:
        ds.close()
    if lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    m = _in_bounds(lat, lon, bounds) & np.isfinite(v) & (v >= 0)
    if not m.any():
        raise ProviderError("no valid precipitation points inside the domain")
    return float(np.mean(v[m] > PRECIP_MM_H))


class Stage4Provider:
    """(d) NCEP Stage IV 1 h QPE, CONUS, from the IEM archive (anonymous HTTPS), reusing
    post/fetch/stage4.py url_for (:61), path_for (:65) and post/sources.download (:317). Precip
    only (precip_area). 2001-present; Puerto Rico/Hawaii/Alaska not covered."""
    name = "stage4"

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "wtune_stage4")

    def fetch(self, bounds, t_start, t_end, interval_h):
        from wtune.post.fetch import stage4
        from wtune.post import sources
        if not stage4._overlaps(bounds):
            raise ProviderError("domain %s outside Stage IV CONUS %s" % (bounds, stage4.CONUS_BBOX))
        series, urls = {}, []
        for t in stage4.valid_times(t_start, t_end, 1):
            dest = os.path.join(self.cache_dir, "ST4.%s.01h.grib" % t.strftime("%Y%m%d%H"))
            sha, _dl = sources.download(stage4.url_for(t, 1), dest)
            if sha is None:
                continue
            urls.append(stage4.url_for(t, 1))
            series[t] = {"precip_area": _grib_precip_area(dest, bounds)}
        if not series:
            raise ProviderError("no Stage IV 1 h file for %s..%s" % (t_start, t_end))
        return series, dict(source="NCEP Stage IV (IEM archive)", auth="anonymous", n_files=len(urls))


class MrmsProvider:
    """(d) MRMS MultiSensor_QPE_01H_Pass2 from AWS noaa-mrms-pds (anonymous; 2020-10-14 onward),
    reusing post/fetch/mrms.py pick_region (:50), list_keys (:61), choose_keys (:95), _fetch_gz
    (:109). Precip only (precip_area)."""
    name = "mrms"

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "wtune_mrms")

    def fetch(self, bounds, t_start, t_end, interval_h):
        from wtune.post.fetch import mrms
        if t_end < mrms.FIRST_DAY:
            raise ProviderError("MRMS starts %s" % mrms.FIRST_DAY.date())
        region = mrms.pick_region(bounds)
        if region is None:
            raise ProviderError("domain %s outside every MRMS sub-domain" % (bounds,))
        pname = mrms.PRODUCTS["qpe"]
        series = {}
        for t in mrms.targets(t_start, t_end, "qpe", 60):
            keys = mrms.choose_keys(mrms.list_keys("%s/%s/%s/" % (region, pname, t.strftime("%Y%m%d"))),
                                    [t], datetime.timedelta(minutes=5))
            if not keys:
                continue
            dest = os.path.join(self.cache_dir, os.path.basename(keys[0])[:-3])
            sha, _dl = mrms._fetch_gz(keys[0], dest, None)
            if sha:
                series[t] = {"precip_area": _grib_precip_area(dest, bounds)}
        if not series:
            raise ProviderError("no MRMS QPE file for %s..%s" % (t_start, t_end))
        return series, dict(source="MRMS QPE_01H_Pass2 (%s)" % region, auth="anonymous")


# IMERG is deliberately absent: GPM IMERG needs NASA Earthdata credentials, and this module never
# embeds or reads credentials. Name it in probe.indicator_sources and it is reported as skipped.
UNSUPPORTED = {"imerg": "IMERG requires NASA Earthdata credentials; not implemented (no credentials "
                        "are ever hardcoded)"}


class _Skipped:
    def __init__(self, name, why):
        self.name, self.why = name, why

    def fetch(self, *a, **k):
        raise ProviderError(self.why)


def default_providers(names=None, case_dir=None, grib_dir=None, cache_dir=None):
    """Providers in the order of `names` (default DEFAULT_SOURCES). Unknown or unsupported names
    become always-failing providers so provenance says why they were skipped."""
    out = []
    for n in names or DEFAULT_SOURCES:
        if n == "case_forcing":
            out.append(CaseForcingProvider([case_dir, grib_dir,
                                            case_dir and os.path.join(case_dir, "wps")]))
        elif n == "era5":
            out.append(Era5Provider())
        elif n == "gfs":
            out.append(GfsProvider(cache_dir))
        elif n == "stage4":
            out.append(Stage4Provider(cache_dir and os.path.join(cache_dir, "stage4")))
        elif n == "mrms":
            out.append(MrmsProvider(cache_dir and os.path.join(cache_dir, "mrms")))
        else:
            out.append(_Skipped(n, UNSUPPORTED.get(n, "unknown indicator source %r" % n)))
    return out


def bounds_from_points(sample_points, margin=0.0):
    """(lat0, lat1, lon0, lon1) of [(lat, lon), ...] (wrfinput_d01 sample points), or None."""
    if not sample_points:
        return None
    la = [p[0] for p in sample_points]
    lo = [p[1] for p in sample_points]
    return (min(la) - margin, max(la) + margin, min(lo) - margin, max(lo) + margin)


def case_cost_structure(case_dir, namelist_path=None, cores_per_node=None):
    """wtune_case's expected_cost_structure for the case (physics-scheme cost shares), or None."""
    try:
        from wtune import case as wtune_case
        nl, wi = wtune_case.find_case_inputs(case_dir, namelist=namelist_path)
        args = (nl, wi) if cores_per_node is None else (nl, wi, cores_per_node)
        return wtune_case.build_profile(*args)["expected_cost_structure"]
    except Exception:  # noqa: BLE001
        return None


def cost_series_for_case(case_dir, run_start, run_end, sources=None, grib_dir=None,
                         cost_structure=None, cache_dir=None, providers=None, namelist_path=None):
    """CostSeries for a case's own d01 span [run_start, run_end]; None only when the domain
    bounds cannot be determined (no wrfinput_d01). cost_structure None -> taken from the case
    profile (case_cost_structure; falls back to DEFAULT_PHASE_WEIGHTS). Never raises."""
    try:
        if cost_structure is None:
            cost_structure = case_cost_structure(case_dir, namelist_path)
        sample_points, _src = predict.read_wrfinput_sample_points(case_dir)
        bounds = bounds_from_points(sample_points)
        if bounds is None:
            return None
        provs = providers if providers is not None else default_providers(
            sources, case_dir, grib_dir, cache_dir)
        return build_cost_series(provs, bounds, run_start, run_end, sample_points,
                                 cost_structure=cost_structure)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------
# point plan by CASE TYPE (user requirement: the points follow the kind of case, not the tuning
# level). Labels come from the decision-0057 classifier -- post/caseinfo.detect_types, or its
# namelist fallback wtune_advise.det_case_types -- never re-derived here (case_types_for below).
# ---------------------------------------------------------------------------------------
# (band of a t0-only plan, band of the type's base set), percent of the predicted run time.
# Rule-of-thumb numbers; the sources are:
#   common/synoptic/frontal  t0 only, 15 %, plus combine()'s day/night radiation r(t) formula
#   convection  10 % with the peak point; t0-only 25 % -- elevconv: truth step cost +44 % at 10 UTC
#               (knowledge/experiments/2026-09-28-rebuild-underestimates-convective-peak-spinup-
#               length-test-elevconv.md); a t0-only plan never sees it
#   tc          10 % with the max-intensity point; t0-only 25 % (assumed, no TC measurement yet)
#   terrain     10 % with the max moisture/precip point; t0-only 20 % (ASSUMED, no evidence)
#   climate     10 % with 4-6 month-stratified points; t0-only 25 % (ASSUMED, no evidence)
# conus25: a t0-only probe under-predicted the run by 9.6 % (wtune_probe_plan module docstring).
CASE_BANDS = {"common": (15.0, 15.0), "synoptic": (15.0, 15.0), "frontal": (15.0, 15.0),
              "convection": (25.0, 10.0), "tc": (25.0, 10.0), "terrain": (20.0, 10.0),
              "climate": (25.0, 10.0)}
SPECIFIC_TYPES = ("tc", "convection", "terrain", "climate")
STRATUM_BAND_FACTOR = 0.8        # each extra stratum point multiplies the band (unvalidated)
CLIMATE_RUN_HOURS = 30 * 24


def case_types_for(case):
    """Labels from the existing classifier for a wtune_advise Case object (reuse, not a second
    implementation): post.caseinfo.detect_types when wrfout frames exist, else its namelist
    fallback. Returns a list of type labels."""
    from wtune import advise as wtune_advise
    return list(wtune_advise.det_case_types(case)["types"])


def _grid_time(cs, t):
    """Snap t onto the hourly grid of cs (when given)."""
    if cs is None:
        return t
    h = round((t - cs.t0).total_seconds() / 3600.0)
    return cs.t0 + datetime.timedelta(hours=max(0, min(len(cs.times) - 1, h)))


def _peak_time(cs, names):
    """Hour with max sum of z-scored `names` indicators (constants dropped), or None."""
    if cs is None:
        return None
    zs = [z for z in (_zscore(cs.indicators[n]) for n in names if n in cs.indicators) if z]
    if not zs:
        return None
    tot = [sum(z[i] for z in zs) for i in range(len(cs.times))]
    return cs.times[max(range(len(tot)), key=lambda i: (tot[i], -i))]


def _type_points(t, run_start, run_end, cs, best_track, window_s):
    """([(time, reason)], [rationale], low_confidence) for ONE specific type's base set
    (beyond t0)."""
    pts, why, low = [], [], False
    last = run_end - datetime.timedelta(seconds=window_s)
    if t == "convection":
        pk = _peak_time(cs, ("precip_area", "cape_area"))
        if pk:
            pts.append((pk, "convective_peak"))
            why.append("convection: max precip_area+cape_area hour %s" % pk.isoformat())
        else:
            low = True
            why.append("convection: LOW CONFIDENCE -- no precip/CAPE indicator, peak hour unknown")
    elif t == "terrain":
        pk = _peak_time(cs, ("precip_area",)) or _peak_time(cs, ("cloud_col",))
        if pk:
            pts.append((pk, "orographic_peak"))
            why.append("terrain: max precip_area (else cloud_col) hour %s" % pk.isoformat())
        else:
            low = True
            why.append("terrain: LOW CONFIDENCE -- no precip/cloud indicator for the moisture peak")
    elif t == "tc":
        rows = [r for r in (best_track or []) if run_start <= r["time"] <= run_end]
        with_p = [r for r in rows if r.get("pmin_hpa")]
        if with_p:
            r = min(with_p, key=lambda r: (r["pmin_hpa"], r["time"]))
            pts.append((r["time"], "tc_max_intensity"))
            why.append("tc: best-track min pressure %.0f hPa at %s" % (r["pmin_hpa"], r["time"].isoformat()))
        else:
            pk = _peak_time(cs, ("cloud_col", "wind_max"))
            if pk:
                pts.append((pk, "tc_max_intensity"))
                why.append("tc: no best-track pressure; max cloud_col+wind_max hour %s" % pk.isoformat())
            else:
                low = True
                why.append("tc: LOW CONFIDENCE -- no best track and no indicators, intensity peak unknown")
        land = [r for r in rows if r.get("landfall") or r.get("record_id") == "L"]
        if land:
            pts.append((land[0]["time"], "tc_landfall"))
            why.append("tc: landfall flag in the track at %s" % land[0]["time"].isoformat())
        elif rows:
            why.append("tc: landfall not determinable (track rows carry no land flag / record id L)")
    return [(min(max(tt, run_start), last), r) for tt, r in pts], why, low


def _climate_points(run_start, run_end, cs, window_s):
    days = (run_end - run_start).total_seconds() / 86400.0
    n = max(4, min(6, int(round(days / 30.0))))
    span = (run_end - run_start) / n
    last = run_end - datetime.timedelta(seconds=window_s)
    out = []
    for k in range(n):
        lo, hi = run_start + span * k, run_start + span * (k + 1)
        if k == 0:
            out.append((run_start, "t0"))
            continue
        mid = lo + span / 2
        if cs is not None and cs.usable:
            cand = [t for t in cs.times if lo <= t < hi]
            if cand:
                vals = sorted(cs.value_at(t) for t in cand)
                med = vals[len(vals) // 2]
                mid = min(cand, key=lambda t: (abs(cs.value_at(t) - med), t))
        out.append((min(_grid_time(cs, mid), last), "stratum %d/%d" % (k + 1, n)))
    return out


def plan_points_for_case(case_types, run_start, run_end, precision_min, T_est_s, indicators=None,
                         best_track=None, window_s=0.0, min_gap_s=1800.0, max_points=6):
    """Probe points by case type. Returns {points: [(time, reason)], band_pct, rationale: [str],
    needs_rebuild: [time], confidence: "normal"|"low", meets_precision: bool, types: [...]}.

    case_types: labels from the decision-0057 classifier (case_types_for). 'precip'/'moisture'/
    'pbl'/... are ignored (always detected, no point rule); 'common'/'synoptic'/'frontal' alone
    mean t0 only. Per specific type (CASE_BANDS; sources in the comment above it):
      convection  t0 + max precip_area/CAPE hour (indicators; none -> low confidence, no point)
      tc          t0 + best-track min-pressure time (else max cloud_col/wind_max hour), + landfall
                  time when the track rows carry `landfall` / record_id 'L'
      terrain     t0 + max precip_area (else cloud_col) hour
      climate or run >= 30 days   4-6 equal time strata (month-sized), one hour per stratum nearest
                  the stratum's median c (midpoint without indicators); t0 is stratum 1
    Several labels: union of the base sets, merged with wtune_probe_plan's min-gap rule; the band
    is the WORST band among the present specific types (common's 15 % is ignored when a specific
    type is present).

    Precision rule: ladder [t0 only] -> [base set] -> [base + k strata points, band x 0.8**k, up to
    max_points]; the first rung with band_pct/100 * T_est_s/60 <= precision_min wins, i.e. the
    smallest set that is precise enough; if none is, the last rung with meets_precision False. A
    t0-only rung carries the worst t0-only band of the present types (convection 25 %, ...), so a
    short run with a loose precision stays at t0 even for a convective case. The extra strata
    points are cost-quantile strata of `indicators` (coldstart_sample._select_by_cost), else
    farthest-in-time. t0 uses phase-1 data (no rebuild); every other point needs a rebuild."""
    types = set(case_types or ["common"])
    duration_h = (run_end - run_start).total_seconds() / 3600.0
    if duration_h >= CLIMATE_RUN_HOURS:
        types.add("climate")
    specific = [t for t in SPECIFIC_TYPES if t in types]
    cs = indicators if (indicators is not None and indicators.usable) else None
    if indicators is not None and cs is None and hasattr(indicators, "indicators"):
        cs = indicators if any(n in indicators.indicators for n in INDICATORS) else None
    rationale, low = [], False
    t0pt = [(run_start, "t0")]

    if not specific:
        label = "/".join(sorted(types & {"common", "synoptic", "frontal"})) or "common"
        rationale.append("%s: t0 only (phase-1 data, no rebuild) + the day/night radiation r(t) "
                         "formula of combine(); band %.0f%%" % (label, CASE_BANDS["common"][1]))
        ladder_base = t0pt
        band_t0 = band_base = CASE_BANDS["common"][1]
    else:
        base = list(t0pt)
        for t in specific:
            if t == "climate":
                base = _climate_points(run_start, run_end, cs, window_s)
                rationale.append("climate: %d time strata (month-sized), one hour per stratum%s; "
                                 "no evidence slug, rule of thumb" % (
                                     len(base), " by median c" if cs else " at the midpoint"))
                continue
            pts, why, lw = _type_points(t, run_start, run_end, cs, best_track, window_s)
            base += pts
            rationale += why
            low = low or lw
        band_t0 = max(CASE_BANDS[t][0] for t in specific)
        # a type whose peak could not be located has not earned its base-set band
        band_base = max(CASE_BANDS[t][1] if not (low and t in ("convection", "tc", "terrain"))
                        else CASE_BANDS[t][0] for t in specific)
        ladder_base = base
    # merge by the existing min-gap rule
    from wtune import probe_plan as _pp

    def merged(pts):
        m = _pp._merge_close_points(sorted((t, r, None) for t, r in pts), min_gap_s)
        return [(t, r) for t, r, _s in m]
    ladder_base = merged(ladder_base)
    rungs = []
    if len(ladder_base) > 1 or "climate" not in specific:
        rungs.append((t0pt, band_t0, "t0 only"))
    if len(ladder_base) > 1 or specific:
        rungs.append((ladder_base, band_base, "base set"))
    if "climate" in specific:
        rungs = [r for r in rungs if r[2] != "t0 only"] or [(ladder_base, band_base, "base set")]
    # escalation: extra strata points on top of the base set
    chosen = {t for t, _r in ladder_base}
    cand = [t for t in (cs.times if cs else [run_start + datetime.timedelta(hours=h)
                                              for h in range(int(duration_h) + 1)])
            if t <= run_end - datetime.timedelta(seconds=window_s) and t not in chosen]
    extra = []
    for k in range(1, max(0, max_points - len(ladder_base)) + 1):
        if not cand:
            break
        if cs is not None:
            from wtune.tools import coldstart_sample as _cs_mod
            lk = {t: cs.value_at(t) for t in cand + list(chosen) + [e[0] for e in extra]}
            pool = sorted(set(cand + [x for x in chosen] + [e[0] for e in extra]))
            sel = _cs_mod._select_by_cost(pool, lk, len(chosen) + len(extra) + 1) or []
            new = [(t, r) for t, r in sel if t not in chosen and t not in {e[0] for e in extra}]
            if not new:
                break
            pick = (new[0][0], "stratum+")
        else:
            allp = sorted(chosen | {e[0] for e in extra})
            far = max(cand, key=lambda c: (min(abs((c - p).total_seconds()) for p in allp), c))
            pick = (far, "farthest")
        extra.append(pick)
        cand = [c for c in cand if abs((c - pick[0]).total_seconds()) >= min_gap_s]
        rungs.append((merged(ladder_base + extra), band_base * STRATUM_BAND_FACTOR ** k,
                      "base + %d strata" % k))
    win = None
    for pts, band, name in rungs:
        if band / 100.0 * T_est_s / 60.0 <= precision_min:
            win = (pts, band, name, True)
            break
    if win is None:
        pts, band, name = rungs[-1]
        win = (pts, band, name, False)
    pts, band, name, ok = win
    rationale.append("precision: %s -> band %.1f%% x T_est %.1f min = %.1f min %s precision %.1f min"
                     % (name, band, T_est_s / 60.0, band / 100.0 * T_est_s / 60.0,
                        "<=" if ok else ">", precision_min))
    if low:
        rationale.append("LOW CONFIDENCE: band reverted to the t0-only value for the type(s) whose "
                         "peak could not be placed")
    return dict(points=pts, band_pct=band, rationale=rationale,
                needs_rebuild=[t for t, _r in pts if t != run_start],
                confidence="low" if low else "normal", meets_precision=ok, types=sorted(types))


def main(argv=None):
    """Dry run, no WRF: `python -m wtune_cost_drivers CASE_DIR [--sources a,b] [--spinup-min 20]
    [--max-points 5]` prints the indicator provenance, c(t) per hour and the points
    plan_points(points_by='cost-indicators') would pick."""
    import argparse
    import copy
    ap = argparse.ArgumentParser(prog="wtune_cost_drivers", description=main.__doc__)
    ap.add_argument("case_dir")
    ap.add_argument("--namelist", default=None)
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    ap.add_argument("--grib-dir", default=None)
    ap.add_argument("--spinup-min", type=float, default=20.0)
    ap.add_argument("--max-points", type=int, default=5)
    a = ap.parse_args(argv)
    from wtune import config as wcfg
    from wtune import probe_plan as pp
    nl = predict.parse_namelist(a.namelist or os.path.join(a.case_dir, "namelist.input"))
    _n, run_s, dt, _note = predict.get_d01_steps(nl)
    run_start = predict.get_time_control_datetime(nl, "start")
    run_end = run_start + datetime.timedelta(seconds=run_s)
    stepra = predict.steps_per_radiation_call(float(predict.g(nl, "physics", "radt", None)), dt)
    cs = cost_series_for_case(a.case_dir, run_start, run_end, [s for s in a.sources.split(",") if s],
                              a.grib_dir, namelist_path=a.namelist)
    if cs is None:
        print("no domain bounds (wrfinput_d01 missing?)")
        return 1
    for p in cs.provenance:
        print("provider", p)
    print("used", cs.used, "usable", cs.usable, "dropped", cs.dropped, "weights", cs.weights)
    if not cs.usable:
        print("WARNING: cost mode unusable -- %s" % cs.why_unusable())
    for i, t in enumerate(cs.times):
        print(t.isoformat(), "c=%s" % ("%.3f" % cs.c[i] if cs.c else "-"),
              {k: round(v[i], 3) for k, v in cs.indicators.items()})
    cfg = copy.deepcopy(wcfg.DEFAULTS["probe"])
    cfg.update(points_by="cost-indicators", max_points=a.max_points, sources=["rebuild"])
    pts, notes = pp.plan_points(nl, run_start, run_end, dt, stepra, a.spinup_min * 60.0, cfg,
                                case_dir=a.case_dir, forcing_interval_h=1, indicators=cs)
    for p in pts:
        print("point", p.at.isoformat(), p.reason, p.source)
    for n in notes:
        print("note", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

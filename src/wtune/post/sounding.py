"""post/sounding.py -- `wtune post sounding RUN_DIR --point lat,lon --time T --out FILE.png`:
a MetPy Skew-T at the nearest grid column (wrf.ll_to_xy), from wrf-python model-level T/Td/
wind, with surface-parcel CAPE/CIN (metpy.calc.surface_based_cape_cin) printed on the
figure and the parcel profile with CAPE/CIN shaded (add_parcel). metpy/matplotlib/wrf-python are all imported lazily, inside functions only.
"""
from __future__ import annotations

import os

from wtune.post.extract import frame_ref


class SoundingError(RuntimeError):
    pass


def _wrf():
    import wrf
    return wrf


def _netcdf4():
    import netCDF4
    return netCDF4


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _profile(ds, wrf_mod, j, i, timeidx=0):
    """Returns (p_hpa, tc_degC, td_degC, u_ms, v_ms) 1-D arrays, bottom to top."""
    p = wrf_mod.getvar(ds, "pressure", timeidx=timeidx)
    tc = wrf_mod.getvar(ds, "tc", timeidx=timeidx)
    td = wrf_mod.getvar(ds, "td", timeidx=timeidx, units="degC")
    ua = wrf_mod.getvar(ds, "ua", timeidx=timeidx, units="m s-1")
    va = wrf_mod.getvar(ds, "va", timeidx=timeidx, units="m s-1")
    return (p[:, j, i].values, tc[:, j, i].values, td[:, j, i].values,
            ua[:, j, i].values, va[:, j, i].values)


def add_parcel(skew, p_q, tc_q, td_q, lw=1.0):
    """Surface-based parcel trace (metpy.calc.parcel_profile, black dashed) with CAPE (red) and
    CIN (blue) shaded on `skew`. p_q/tc_q/td_q: pint quantities, surface first, pressure
    decreasing. Returns the parcel profile, or None when it cannot be built (fewer than 3 finite
    levels, non-monotonic pressure)."""
    import numpy as np
    import metpy.calc as mpcalc
    ok = np.isfinite(p_q.magnitude) & np.isfinite(tc_q.magnitude) & np.isfinite(td_q.magnitude)
    if ok.sum() < 3:
        return None
    p_q, tc_q, td_q = p_q[ok], tc_q[ok], td_q[ok]
    if np.any(np.diff(p_q.magnitude) >= 0):
        return None
    try:
        prof = mpcalc.parcel_profile(p_q, tc_q[0], td_q[0]).to("degC")
        skew.plot(p_q, prof, "k", lw=lw, ls="--", label="sfc parcel")
        skew.shade_cin(p_q, tc_q, prof, td_q, alpha=0.3)
        skew.shade_cape(p_q, tc_q, prof, alpha=0.3)
    except Exception:            # metpy raises on degenerate (e.g. saturated-everywhere) columns
        return None
    return prof


def make_sounding(run_dir, lat, lon, time_str, out_path, domain=1):
    """Writes a Skew-T PNG/SVG for the nearest grid column at time_str. Returns
    {"out", "lat", "lon", "grid_j", "grid_i", "cape_j_kg", "cin_j_kg"}."""
    import numpy as np
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units as mpunits

    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    plt = _matplotlib()

    path, ti = frame_ref(run_dir, domain, time_str)
    if not os.path.exists(path):
        raise SoundingError("no such frame: %s" % path)

    with netcdf4.Dataset(path) as ds:
        xy = wrf_mod.ll_to_xy(ds, lat, lon, timeidx=ti)
        i, j = int(xy[0]), int(xy[1])
        p, tc, td, u, v = _profile(ds, wrf_mod, j, i, ti)

    order = np.argsort(p)[::-1]        # SkewT wants pressure decreasing (surface first)
    p, tc, td, u, v = p[order], tc[order], td[order], u[order], v[order]

    p_q = p * mpunits.hPa
    tc_q = tc * mpunits.degC
    td_q = td * mpunits.degC
    u_q = u * mpunits("m/s")
    v_q = v * mpunits("m/s")

    cape, cin = mpcalc.surface_based_cape_cin(p_q, tc_q, td_q)

    fig = plt.figure(figsize=(7, 8))
    # standard Skew-T log-p (decision 0064): isotherms skewed 45 deg, dry/moist adiabats and
    # mixing-ratio lines as background, T red / Td green, barbs in kt
    skew = SkewT(fig, rotation=45)
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 50)
    skew.plot_dry_adiabats(alpha=0.25, lw=0.6)
    skew.plot_moist_adiabats(alpha=0.25, lw=0.6)
    skew.plot_mixing_lines(alpha=0.3, lw=0.6)
    skew.plot(p_q, tc_q, "r")
    skew.plot(p_q, td_q, "g")
    add_parcel(skew, p_q, tc_q, td_q)
    skew.plot_barbs(p_q[::3], u_q[::3], v_q[::3], xloc=0.94,   # default 1.0 clips half the barb
                    plot_units=mpunits.knots)
    skew.ax.set_xlabel("temperature (°C)")
    skew.ax.set_ylabel("pressure (hPa)")
    from wtune.post.conventions import path_time_label
    skew.ax.set_title("%s  d%02d  (%.3f, %.3f)  %s\nCAPE=%.0f J/kg  CIN=%.0f J/kg  "
                      "(surface-based parcel; barbs in kt)"
                      % (os.path.basename(os.path.normpath(run_dir)), domain, lat, lon,
                         path_time_label(path, time_str),
                         cape.magnitude, cin.magnitude), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return dict(out=out_path, lat=lat, lon=lon, grid_j=j, grid_i=i,
               cape_j_kg=float(cape.magnitude), cin_j_kg=float(cin.magnitude))


# ---------------------------------------------------------------------------------------
# model-vs-radiosonde comparison (pbl recipe)
# ---------------------------------------------------------------------------------------

G = 9.80665
RI_CRIT = 0.25          # bulk-Richardson critical number, Seibert et al. (2000, Atmos. Env. 34)
ZMIN_M = 100.0          # PBL heights below this are reported as ZMIN_M (sonde surface level is noisy)
SHEAR2_FLOOR = 1.0      # (m/s)^2 floor of the bulk-Richardson denominator (see pblh_bulk_ri)
_COLUMN_KEYS = ("p", "t", "td", "u", "v", "z")


def model_column(ds, wrf_mod, j, i, timeidx=0):
    """Model column at grid index (j, i): dict of 1-D arrays bottom to top --
    p (hPa), t/td (degC), u/v (m/s, earth-relative), z (m AGL) -- plus scalars
    ter (m MSL) and pblh_model (m, the PBL scheme's own PBLH variable)."""
    import numpy as np
    p, t, td, u, v = _profile(ds, wrf_mod, j, i, timeidx)
    z = wrf_mod.getvar(ds, "z", timeidx=timeidx, units="m")[:, j, i].values
    ter = float(wrf_mod.getvar(ds, "ter", timeidx=timeidx)[j, i])
    pblh = float(np.asarray(ds.variables["PBLH"][timeidx])[j, i]) if "PBLH" in ds.variables else float("nan")
    return dict(p=p, t=t, td=td, u=u, v=v, z=z - ter, ter=ter, pblh_model=pblh)


def obs_column(sounding, elev_m):
    """IGRA sounding (dict of arrays from post.fetch.igra.read_soundings) -> the same column
    dict as model_column, bottom to top, only levels with p, T and Td. IGRA reports geopotential
    height on only some levels, so height AGL is rebuilt for every level by integrating the
    hypsometric equation (metpy.calc.thickness_hydrostatic on virtual temperature) up from the
    station elevation -- consistent between levels and independent of the sparse GPH field."""
    import numpy as np
    import metpy.calc as mpcalc
    from metpy.units import units as mpunits

    p = np.asarray(sounding["pressure_hPa"], float)
    t = np.asarray(sounding["temp_C"], float)
    td = np.asarray(sounding["dewpoint_C"], float)
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(td)
    order = np.argsort(-p[ok])
    sel = lambda a: np.asarray(a, float)[ok][order]      # noqa: E731
    p, t, td = sel(p), sel(t), sel(td)
    wdir, wspd = sel(sounding["wdir_deg"]), sel(sounding["wspd_ms"])
    keep = np.concatenate([[True], np.diff(p) < 0])       # drop repeated pressures
    p, t, td, wdir, wspd = p[keep], t[keep], td[keep], wdir[keep], wspd[keep]
    if p.size < 5:
        return None
    u, v = mpcalc.wind_components(wspd * mpunits("m/s"), wdir * mpunits.degree)
    tv = mpcalc.virtual_temperature(t * mpunits.degC,
                                    mpcalc.mixing_ratio_from_relative_humidity(
                                        p * mpunits.hPa, t * mpunits.degC,
                                        mpcalc.relative_humidity_from_dewpoint(
                                            t * mpunits.degC, td * mpunits.degC))).to("K").magnitude
    z = np.zeros_like(p)
    for k in range(1, p.size):
        tv_mean = 0.5 * (tv[k] + tv[k - 1])
        z[k] = z[k - 1] + 287.05 * tv_mean / G * np.log(p[k - 1] / p[k])
    return dict(p=p, t=t, td=td, u=np.asarray(u.magnitude), v=np.asarray(v.magnitude), z=z,
                ter=float(elev_m), pblh_model=float("nan"))


def pblh_bulk_ri(col, ri_crit=RI_CRIT):
    """PBL height (m AGL) from the bulk-Richardson-number method: the lowest height where
        Ri_b(z) = g/theta_v(sfc) * (theta_v(z) - theta_v(sfc)) * z / ((u-u_sfc)^2 + (v-v_sfc)^2)
    first reaches ri_crit = 0.25 (Seibert et al. 2000; Vogelezang & Holtslag 1996), linearly
    interpolated between levels. theta_v from metpy.calc.virtual_potential_temperature.
    No established package computes PBL height from a profile (MetPy has none; wrf-python only
    returns the PBL scheme's own PBLH), hence this short routine, applied identically to the
    radiosonde and the model column so the two are comparable. The denominator is floored at
    SHEAR2_FLOOR (Vogelezang-Holtslag's surface friction-velocity term, unavailable from a
    sounding, is what normally keeps a shear-free mixed layer from dividing by zero). NaN if
    Ri_b never exceeds ri_crit at or above ZMIN_M (100 m; shallower layers are returned as 100 m, since the surface level of a sonde is a noisy reference) below the top of the profile or the profile has < 5 levels."""
    import numpy as np
    import metpy.calc as mpcalc
    from metpy.units import units as mpunits

    p, t, td, z = col["p"], col["t"], col["td"], col["z"]
    if len(p) < 5:
        return float("nan")
    mixr = mpcalc.mixing_ratio_from_relative_humidity(
        p * mpunits.hPa, t * mpunits.degC,
        mpcalc.relative_humidity_from_dewpoint(t * mpunits.degC, td * mpunits.degC))
    thv = mpcalc.virtual_potential_temperature(p * mpunits.hPa, t * mpunits.degC, mixr).to("K").magnitude
    def _fill(a):      # sondes report wind on a subset of levels: interpolate in height, else Ri is NaN
        a = np.asarray(a, float)
        fin = np.isfinite(a)
        return np.interp(z, z[fin], a[fin]) if fin.sum() >= 2 else np.zeros_like(a)
    u, v = _fill(col["u"]), _fill(col["v"])
    shear2 = np.maximum((u - u[0]) ** 2 + (v - v[0]) ** 2, SHEAR2_FLOOR)
    dz = z - z[0]
    ri = G / thv[0] * (thv - thv[0]) * dz / shear2
    for k in range(1, len(ri)):
        if np.isfinite(ri[k]) and ri[k] >= ri_crit and z[k] >= ZMIN_M:
            if not np.isfinite(ri[k - 1]) or ri[k - 1] >= ri_crit:
                return float(max(z[k], ZMIN_M))
            f = (ri_crit - ri[k - 1]) / (ri[k] - ri[k - 1])
            return float(max(z[k - 1] + f * (z[k] - z[k - 1]), ZMIN_M))
    return float("nan")


def profile_stats(model, obs, p_top=500.0):
    """Model minus obs bias/RMSE of T, Td, wind speed at the observed pressure levels between
    the observed surface pressure and p_top (hPa); the model column is interpolated linearly in
    ln p onto each observed level (numpy.interp). Returns {var: {"bias","rmse","n"}}."""
    import numpy as np
    lp_m = np.log(model["p"])[::-1]                       # np.interp wants increasing x
    sel = (obs["p"] <= model["p"][0]) & (obs["p"] >= p_top)
    lp_o = np.log(obs["p"][sel])
    out = {}
    m_spd = np.hypot(model["u"], model["v"])
    o_spd = np.hypot(obs["u"], obs["v"])[sel]
    for name, m_arr, o_arr in (("T", model["t"], obs["t"][sel]), ("Td", model["td"], obs["td"][sel]),
                               ("wspd", m_spd, o_spd)):
        m_i = np.interp(lp_o, lp_m, m_arr[::-1])
        d = m_i - o_arr
        d = d[np.isfinite(d)]
        out[name] = dict(bias=float(d.mean()) if d.size else float("nan"),
                         rmse=float(np.sqrt((d ** 2).mean())) if d.size else float("nan"), n=int(d.size))
    return out


def skewt_overlay(fig, rect, model, obs, title, p_top=200.0, text=None):
    """Skew-T (metpy.plots.SkewT) of model (solid) and radiosonde (dashed) T (red) and Td
    (green), wind barbs of each on the right margin, in the sub-rectangle `rect` of `fig`
    (SkewT's `subplot` argument). `text`: optional monospace stats box inside the axes."""
    import numpy as np
    from metpy.plots import SkewT
    from metpy.units import units as mpunits

    skew = SkewT(fig, subplot=rect, rotation=45)
    for col, ls, lab, xloc in ((model, "-", "WRF", 1.02), (obs, "--", "sonde", 1.10)):
        if col is None:
            continue
        skew.plot(col["p"] * mpunits.hPa, col["t"] * mpunits.degC, "r", ls=ls, lw=1.4, label="T " + lab)
        skew.plot(col["p"] * mpunits.hPa, col["td"] * mpunits.degC, "g", ls=ls, lw=1.4, label="Td " + lab)
        ok = np.isfinite(col["u"]) & np.isfinite(col["v"]) & (col["p"] >= p_top)
        idx = np.flatnonzero(ok)[::2 if col is model else 1]
        skew.plot_barbs(col["p"][idx] * mpunits.hPa, col["u"][idx] * mpunits("m/s"),
                        col["v"][idx] * mpunits("m/s"), xloc=xloc, length=5, linewidth=0.6,
                        plot_units=mpunits.knots)   # barbs in kt (decision 0064)
    if model is not None:
        add_parcel(skew, model["p"] * mpunits.hPa, model["t"] * mpunits.degC,
                   model["td"] * mpunits.degC, lw=0.9)
    skew.ax.set_ylim(1020, p_top)
    skew.ax.set_xlim(-30, 40)
    skew.plot_dry_adiabats(alpha=0.25, lw=0.6)
    skew.plot_moist_adiabats(alpha=0.25, lw=0.6)
    skew.plot_mixing_lines(alpha=0.3, lw=0.6)
    skew.ax.set_title(title, fontsize=11)
    skew.ax.set_xlabel("T (°C); barbs in kt", fontsize=9)
    skew.ax.set_ylabel("p (hPa)", fontsize=9)
    skew.ax.tick_params(labelsize=9)
    if text:
        skew.ax.text(0.02, 0.98, text, transform=skew.ax.transAxes, fontsize=9, va="top", ha="left",
                     family="monospace", zorder=10,
                     bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))
    return skew

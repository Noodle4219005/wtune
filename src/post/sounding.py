"""post/sounding.py -- `wtune post sounding RUN_DIR --point lat,lon --time T --out FILE.png`:
a MetPy Skew-T at the nearest grid column (wrf.ll_to_xy), from wrf-python model-level T/Td/
wind, with surface-parcel CAPE/CIN (metpy.calc.surface_based_cape_cin) printed on the
figure. metpy/matplotlib/wrf-python are all imported lazily, inside functions only.
"""
from __future__ import annotations

import os


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

    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    if not os.path.exists(path):
        raise SoundingError("no such frame: %s" % path)

    with netcdf4.Dataset(path) as ds:
        xy = wrf_mod.ll_to_xy(ds, lat, lon)
        i, j = int(xy[0]), int(xy[1])
        p, tc, td, u, v = _profile(ds, wrf_mod, j, i)

    order = np.argsort(p)[::-1]        # SkewT wants pressure decreasing (surface first)
    p, tc, td, u, v = p[order], tc[order], td[order], u[order], v[order]

    p_q = p * mpunits.hPa
    tc_q = tc * mpunits.degC
    td_q = td * mpunits.degC
    u_q = u * mpunits("m/s")
    v_q = v * mpunits("m/s")

    cape, cin = mpcalc.surface_based_cape_cin(p_q, tc_q, td_q)

    fig = plt.figure(figsize=(7, 8))
    skew = SkewT(fig)
    skew.plot(p_q, tc_q, "r")
    skew.plot(p_q, td_q, "g")
    skew.plot_barbs(p_q[::3], u_q[::3], v_q[::3], xloc=0.94)   # default 1.0 clips half the barb
    skew.ax.set_title("%s  d%02d  %s  (%.3f,%.3f)\nCAPE=%.0f J/kg  CIN=%.0f J/kg"
                      % (os.path.basename(os.path.normpath(run_dir)), domain, time_str,
                         lat, lon, cape.magnitude, cin.magnitude), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return dict(out=out_path, lat=lat, lon=lon, grid_j=j, grid_i=i,
               cape_j_kg=float(cape.magnitude), cin_j_kg=float(cin.magnitude))

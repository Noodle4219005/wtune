"""post/surface_diag.py -- surface wind gust and visibility from a wrfout frame, for comparison
with station gusts and visibility (post/agent_obs.py, stations_extra). WRF writes neither by
default, so both are the standard post-processor diagnostics:

- gust: WSPD10MAX when the run wrote it (nwp_diagnostics = 1: the 10 m wind maximum over the
  history interval); otherwise NCEP UPP's CALGUST -- the wind at the top of the PBL mixed down,
  gust = w10 + (w_pbl - w10) * (1 - min(0.5, PBLH / 2000 m)), never below w10.
- visibility: Stoelinga and Warner (1999, JAM 38:385), as in UPP's CALVIS, at the lowest model
  level: extinction beta (1/km) = 144.7 C_cloud^0.88 + 1.1 C_rain^0.75 + 163.9 C_ice
  + 10.36 C_snow^0.7776 with C in g/m3 (graupel counted as snow), vis = 3.912 / beta, capped at
  VIS_CAP_M. Clear air and haze are not represented, so the comparison is about fog, cloud
  base and precipitation, not dry-air visibility.
"""
from __future__ import annotations

RD = 287.04
VIS_CAP_M = 16093.0          # 10 statute miles, the highest value US ASOS reports


def gust(ds, timeidx=0):
    """(m/s 2-D, method text)."""
    import numpy as np
    if "WSPD10MAX" in ds.variables:
        return np.asarray(ds.variables["WSPD10MAX"][timeidx], "f8"), "WSPD10MAX"
    import wrf
    z = np.asarray(wrf.getvar(ds, "height_agl", timeidx=timeidx), "f8")
    ua = np.asarray(wrf.getvar(ds, "ua", timeidx=timeidx), "f8")
    va = np.asarray(wrf.getvar(ds, "va", timeidx=timeidx), "f8")
    pblh = np.asarray(ds.variables["PBLH"][timeidx], "f8")
    w10 = np.hypot(np.asarray(ds.variables["U10"][timeidx], "f8"),
                   np.asarray(ds.variables["V10"][timeidx], "f8"))
    inside = z <= pblh[None]
    k = np.maximum(inside.sum(axis=0) - 1, 0)          # highest mass level inside the PBL
    wpbl = np.take_along_axis(np.hypot(ua, va), k[None], axis=0)[0]
    g = w10 + (wpbl - w10) * (1.0 - np.minimum(0.5, pblh / 2000.0))
    return np.maximum(g, w10), "UPP CALGUST (PBL-top wind mixed down)"


def visibility(ds, timeidx=0):
    """m 2-D at the lowest model level, capped at VIS_CAP_M."""
    import numpy as np
    v = ds.variables

    def lev0(name):
        return np.asarray(v[name][timeidx, 0], "f8") if name in v else 0.0
    p = lev0("P") + lev0("PB")
    th = lev0("T") + 300.0
    tk = th * (p / 1.0e5) ** (RD / 1004.5)
    qv = lev0("QVAPOR")
    rho = p / (RD * tk * (1.0 + 0.61 * qv))
    g_m3 = 1.0e3 * rho                                   # kg/kg -> g/m3
    cw, cr, ci = (np.maximum(lev0(n), 0.0) * g_m3 for n in ("QCLOUD", "QRAIN", "QICE"))
    cs = (np.maximum(lev0("QSNOW"), 0.0) + np.maximum(lev0("QGRAUP"), 0.0)) * g_m3
    beta = 144.7 * cw ** 0.88 + 1.1 * cr ** 0.75 + 163.9 * ci + 10.36 * cs ** 0.7776
    with np.errstate(divide="ignore"):
        vis = np.where(beta > 0, 3.912 / beta * 1000.0, VIS_CAP_M)
    return np.minimum(vis, VIS_CAP_M)

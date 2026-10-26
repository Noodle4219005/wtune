"""Rainfall colour scales: the Central Weather Administration's (CWA, Taiwan) for a domain over
Taiwan, the WPC isohyet thresholds (decision 0064, post/obs_figs.py) elsewhere.

CWA's accumulated-rain legend (daily and hourly maps share it; 16 thresholds, 1 .. 300 mm and
an open top bin), colours sampled from cwa.gov.tw's QZJ8/QZT8 legend images on 2026-10-26 (a
few units of JPEG error; wiki taiwan-heavy-rain-wrf-figure-set-and-cwa-rain-colour-scale). The
WPC thresholds stop at 8 in = 203.2 mm, which a Taiwan typhoon passes within hours (haitang2017
d03 saturated it).
"""
from __future__ import annotations

CWA_LEVELS = [1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
CWA_COLOURS = ["#9cfdfe", "#01d2fd", "#00a5fd", "#0177fd", "#26a31b", "#00f92e", "#fefd31",
               "#ffd328", "#ffa71e", "#ff2b06", "#d92203", "#aa1701", "#ab20a3", "#dc2dd2",
               "#fe38fa"]                          # one per bin [CWA_LEVELS[k], CWA_LEVELS[k+1])
CWA_OVER = "#fed6fd"                               # >= 300 mm
TAIWAN_BOX = (20.0, 27.0, 116.0, 124.0)            # lat_min, lat_max, lon_min, lon_max


def over_taiwan(lat, lon) -> bool:
    """True when the grid's centre point lies in TAIWAN_BOX."""
    import numpy as np
    lat, lon = np.asarray(lat, dtype="f8"), np.asarray(lon, dtype="f8")
    j, i = lat.shape[0] // 2, lat.shape[1] // 2
    return (TAIWAN_BOX[0] <= lat[j, i] <= TAIWAN_BOX[1]
            and TAIWAN_BOX[2] <= lon[j, i] <= TAIWAN_BOX[3])


def cwa_cmap(levels=None):
    """(ListedColormap, BoundaryNorm) of CWA's colours for `levels` (a prefix of CWA_LEVELS,
    default all), the bins above the last level drawn in the next official colour."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    levels = list(levels or CWA_LEVELS)
    n = len(levels) - 1
    cm = ListedColormap(CWA_COLOURS[:n], name="cwa_rain")
    cm.set_over(CWA_COLOURS[n] if n < len(CWA_COLOURS) else CWA_OVER)
    cm.set_under((1, 1, 1, 0))
    return cm, BoundaryNorm(levels, n)             # above the top level: the cmap's over colour


def rain_scale(lat, lon, hi_mm):
    """(levels, cmap, scale name) for an accumulation map of a grid whose maximum is hi_mm:
    the CWA or WPC thresholds up to one level above hi_mm (at least two levels)."""
    if over_taiwan(lat, lon):
        upto = [lv for lv in CWA_LEVELS if lv <= hi_mm]
        lev = CWA_LEVELS[:max(2, len(upto) + 1)]
        return [float(v) for v in lev], cwa_cmap(lev)[0], "CWA"
    from wtune.post.obs_figs import PRECIP_LEVELS, precip_cmap
    upto = [lv for lv in PRECIP_LEVELS if lv <= hi_mm]
    return list(PRECIP_LEVELS[:max(2, len(upto) + 1)]), precip_cmap(), "WPC"

"""post/conventions.py -- the drawing conventions every WTune figure shares (decision 0064):
units, operational colour tables and level sets, and the init/valid/lead time label. One place,
so a figure builder never re-derives them."""
from __future__ import annotations

from datetime import datetime

KT_PER_MS = 1.943844

# NWS WSR-88D reflectivity table, 5 dBZ steps from 5 to 75 (MetPy ctables 'NWSReflectivity')
DBZ_LEVELS = list(range(5, 80, 5))
# 2-5 km updraft helicity, m2 s-2 (SPC/HREF practice: >= 75 marks rotating storms)
UH_LEVELS = [25, 50, 75, 100, 150, 200, 300]
# integrated vapour transport, kg m-1 s-1; 250 = atmospheric-river threshold (Ralph et al. 2019,
# CW3E AR scale)
IVT_LEVELS = [100, 150, 200, 250, 300, 400, 500, 600, 750, 1000, 1250]
# diverging map for precipitation / moisture differences: wet = green-blue, dry = brown
WET_DRY_CMAP = "BrBG"

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
UNIT_LABELS = {"m2/s2": "m$^2$ s$^{-2}$", "kg m^-1 s^-1": "kg m$^{-1}$ s$^{-1}$"}


def reflectivity_colors():
    """(levels, colours) of the NWS reflectivity table: one colour per 5 dBZ bin, 5..75+."""
    from metpy.plots import ctables
    norm, cmap = ctables.registry.get_with_steps("NWSReflectivity", 5, 5)
    return DBZ_LEVELS, [cmap(i) for i in range(cmap.N)]


def reflectivity_cmap():
    """ListedColormap of the NWS table with white below 5 dBZ (for pcolormesh/BoundaryNorm)."""
    from matplotlib.colors import ListedColormap
    _, colors = reflectivity_colors()
    # one colour per bin plus the >= 75 dBZ overflow colour, so BoundaryNorm(extend="max")
    # has len(levels) colours available
    cm = ListedColormap(colors[:len(DBZ_LEVELS)], name="nws_reflectivity")
    cm.set_under("white")
    cm.set_over(colors[len(DBZ_LEVELS) - 1])
    return cm


def _parse(t):
    if isinstance(t, datetime):
        return t
    t = str(t).replace("T", "_")[:19]
    for fmt in (WRF_TIME_FMT, "%Y-%m-%d_%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    return None


def init_time(ds):
    """Simulation start (WRF's SIMULATION_START_DATE global attribute), or None."""
    try:
        return _parse(ds.getncattr("SIMULATION_START_DATE"))
    except Exception:
        return None


def time_label(init, valid):
    """'init 2019-01-28 00Z  valid 2019-01-31 00Z (F072)' -- the forecast-chart convention;
    falls back to the valid time alone when the init time is unknown."""
    v = _parse(valid)
    i = _parse(init) if init is not None else None
    if v is None:
        return str(valid)
    vs = v.strftime("%Y-%m-%d %HZ")
    if i is None:
        return "valid " + vs
    lead_h = int(round((v - i).total_seconds() / 3600.0))
    return "init %s  valid %s (F%03d)" % (i.strftime("%Y-%m-%d %HZ"), vs, lead_h)


def ds_time_label(ds, valid):
    return time_label(init_time(ds), valid)


def path_time_label(path, valid):
    """time_label() for a wrfout path (opens it only to read SIMULATION_START_DATE)."""
    try:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            return ds_time_label(ds, valid)
    except Exception:
        return time_label(None, valid)

"""post/analysis_kit.py -- the pieces an agent's analysis/*.py script kept rewriting during the
2026-10-26 mock competition (haitang2017, MeiYu-2020), handed to every script as `ctx.kit`
(decision 0078): rain colour scales, contingency scores with absolute or percentile
(normalized) thresholds, box means, nearest grid points, a Hovmoller along a line, track
error, and the observation/model side-by-side map. numpy only, except obs_model_panels.
"""
from __future__ import annotations

import math

from wtune.post.rainscale import CWA_LEVELS, cwa_cmap, rain_scale  # noqa: F401  re-exported

EARTH_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance(s) in km; numpy arrays broadcast."""
    import numpy as np
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2)
    return 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_index(lat2d, lon2d, lat, lon):
    """(j, i, distance km) of the grid cell nearest to (lat, lon)."""
    import numpy as np
    d = haversine_km(np.asarray(lat2d), np.asarray(lon2d), lat, lon)
    j, i = np.unravel_index(np.nanargmin(d), d.shape)
    return int(j), int(i), float(d[j, i])


def match_points(lat2d, lon2d, field, points, max_km=None):
    """[(value, distance km)] of `field` at the cell nearest each (lat, lon) of `points`;
    value None when farther than max_km (e.g. a gauge outside the domain)."""
    out = []
    for lat, lon in points:
        j, i, d = nearest_index(lat2d, lon2d, lat, lon)
        out.append((None if max_km is not None and d > max_km else float(field[j][i]), d))
    return out


def box_mean(lat2d, lon2d, field, box):
    """NaN-aware mean of `field` over the cells inside box = [lat_min, lat_max, lon_min,
    lon_max]; NaN when no cell is inside."""
    import numpy as np
    lat, lon = np.asarray(lat2d), np.asarray(lon2d)
    inside = (lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3])
    vals = np.asarray(field, dtype="f8")[inside]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def contingency(obs, fcst, threshold, fcst_threshold=None):
    """2x2 table and scores of the event `value >= threshold` over the points where both are
    finite. fcst_threshold: a separate threshold for the forecast (the normalized case).
    dict(hits, misses, false_alarms, correct_negatives, n, pod, far, csi, fbias, ets)."""
    import numpy as np
    o, f = np.asarray(obs, dtype="f8").ravel(), np.asarray(fcst, dtype="f8").ravel()
    ok = np.isfinite(o) & np.isfinite(f)
    oe = o[ok] >= threshold
    fe = f[ok] >= (threshold if fcst_threshold is None else fcst_threshold)
    a = int(np.sum(oe & fe))
    b = int(np.sum(~oe & fe))
    c = int(np.sum(oe & ~fe))
    d = int(np.sum(~oe & ~fe))
    n = a + b + c + d
    div = lambda x, y: x / y if y else float("nan")   # noqa: E731
    a_r = div((a + b) * (a + c), n)
    return dict(hits=a, misses=c, false_alarms=b, correct_negatives=d, n=n,
                pod=div(a, a + c), far=div(b, a + b), csi=div(a, a + b + c),
                fbias=div(a + b, a + c), ets=div(a - a_r, a + b + c - a_r))


def scores_by_threshold(obs, fcst, thresholds, normalized=False):
    """[dict(threshold, fcst_threshold, **contingency)] per threshold. normalized=True:
    thresholds are percentiles (e.g. 50, 75, 90, 95) taken from each field separately, so a
    displaced or too-weak rain area is scored on its pattern rather than scoring ~0 at every
    fixed amount."""
    import numpy as np
    o, f = np.asarray(obs, dtype="f8"), np.asarray(fcst, dtype="f8")
    both = np.isfinite(o) & np.isfinite(f)
    out = []
    for t in thresholds:
        if normalized:
            ot, ft = float(np.percentile(o[both], t)), float(np.percentile(f[both], t))
        else:
            ot = ft = float(t)
        out.append(dict(threshold=ot, fcst_threshold=ft, percentile=t if normalized else None,
                        **contingency(o, f, ot, ft)))
    return out


def line_points(lat0, lon0, lat1, lon1, n=100):
    """n (lat, lon) points evenly along the straight lon/lat line, and their distance (km)
    from the first point."""
    import numpy as np
    lats, lons = np.linspace(lat0, lat1, n), np.linspace(lon0, lon1, n)
    return lats, lons, haversine_km(lat0, lon0, lats, lons)


def hovmoller(fields, lat2d, lon2d, lat0, lon0, lat1, lon1, n=100, halfwidth_cells=0):
    """(distance km [n], values [len(fields), n]) of a time sequence of 2-D fields sampled at
    the nearest cell along the line (lat0, lon0) -> (lat1, lon1); halfwidth_cells > 0 averages
    a (2h+1)^2 block around each cell."""
    import numpy as np
    lats, lons, dist = line_points(lat0, lon0, lat1, lon1, n)
    cells = [nearest_index(lat2d, lon2d, la, lo)[:2] for la, lo in zip(lats, lons)]
    h = int(halfwidth_cells)
    out = np.full((len(fields), n), np.nan)
    for t, f in enumerate(fields):
        f = np.asarray(f, dtype="f8")
        for k, (j, i) in enumerate(cells):
            out[t, k] = np.nanmean(f[max(0, j - h):j + h + 1, max(0, i - h):i + h + 1])
    return dist, out


def track_error(model, best):
    """[(time, distance km)] at the times both tracks have; model/best: [(time, lat, lon)]
    with comparable time keys (e.g. 'YYYY-MM-DD_HH:MM:SS' strings or datetimes)."""
    ref = {t: (la, lo) for t, la, lo in best}
    return [(t, float(haversine_km(la, lo, *ref[t]))) for t, la, lo in model if t in ref]


def obs_model_panels(obs, model, levels, title, cmap=None, norm=None, labels=("observed",
                     "model"), cbar_label="mm", extent=None, figsize=(12.0, 5.6)):
    """(fig, axes): two maps side by side on ONE colour scale. obs/model: (lat, lon, field)
    with 1-D or 2-D lat/lon (the observation grid need not be the model's). Default scale: the
    CWA one when the model grid is over Taiwan, else WPC (rain_scale). Save it with
    ctx.save_figure(fig, name)."""
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm
    from wtune.post.basemap import add_base_map
    from wtune.post.plotting import _matplotlib, _set_cartopy_data_dir
    plt = _matplotlib()
    _set_cartopy_data_dir()
    if cmap is None:
        lat_m, lon_m = (np.asarray(a) for a in model[:2])
        if lat_m.ndim == 1:
            lon_m, lat_m = np.meshgrid(lon_m, lat_m)
        levels, cmap, _name = rain_scale(lat_m, lon_m, float(max(levels)))
    if norm is None:
        norm = BoundaryNorm(levels, cmap.N, extend="max" if cmap.N > len(levels) - 1
                            else "neither")
    if extent is None:
        la, lo = np.asarray(model[0]), np.asarray(model[1])
        extent = [float(np.nanmin(lo)), float(np.nanmax(lo)), float(np.nanmin(la)),
                  float(np.nanmax(la))]
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained",
                             subplot_kw={"projection": ccrs.PlateCarree()})
    mesh = None
    for ax, (lat, lon, field), lab in zip(axes, (obs, model), labels):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        f = np.ma.masked_invalid(np.asarray(field, dtype="f8"))
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_less(f, levels[0]), cmap=cmap, norm=norm,
                             transform=ccrs.PlateCarree(), shading="auto")
        add_base_map(ax, ccrs, terrain=False, nest_rings=[], caption=False)
        ax.set_title(lab, fontsize=10)
    fig.colorbar(mesh, ax=list(axes), shrink=0.85, label=cbar_label, extend="max")
    fig.suptitle(title, fontsize=11)
    return fig, axes


def fmt(value, decimals=1):
    """The string an assessment quotes for a CSV number: `wtune post check` matches decimals at
    the precision written, so quote with the same rounding the CSV can reproduce."""
    return "nan" if value is None or (isinstance(value, float) and math.isnan(value)) else (
        "%.*f" % (decimals, value))

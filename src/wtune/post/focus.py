"""Where and when a recipe samples a point figure (cross-section, time-height, skew-T) and draws
its one-time maps: the heaviest post-spin-up rain and its peak hour.

The recipes used the last frame's single wettest cell (moisture) or strongest echo (skew-T) at
the last frame. On haitang2017 d03 that put the cross-section on the domain's west edge in
Guangdong and the maps at F072, after the rain had ended. Here the rain accumulated after the
spin-up is smoothed over ~30 km, cells within the boundary zone plus that width are skipped,
an optional box (project.yaml post.focus: [lat_min, lat_max, lon_min, lon_max]) restricts the
search, and the peak is the frame interval with the largest mean rain rate around that point.
"""
from __future__ import annotations

from datetime import datetime

SMOOTH_KM = 30.0


def _hours(a, b):
    from wtune.post.caseinfo import WRF_TIME_FMT
    return (datetime.strptime(b, WRF_TIME_FMT) - datetime.strptime(a, WRF_TIME_FMT)
            ).total_seconds() / 3600.0


def rain_focus(run_dir, domain, spinup_hours=6.0, box=None, smooth_km=SMOOTH_KM):
    """dict(lat, lon, j, i, peak_time, peak_rate_mm_h, total_mm, start, end, rule); None when
    the domain has no frames. total_mm: the smoothed post-spin-up accumulation at the point."""
    import netCDF4
    import numpy as np
    from scipy.ndimage import uniform_filter
    from wtune.post import caseinfo
    from wtune.post.extract import accumulated_rain, list_frame_refs
    frames = list_frame_refs(run_dir, domain=domain)
    if not frames:
        return None
    after = [f for f in frames if _hours(frames[0][0], f[0]) >= spinup_hours] or frames
    if len(after) < len(frames):
        after = [frames[len(frames) - len(after) - 1]] + after   # the accumulation starts there
    rains = []
    for _t, path, ti in after:
        with netCDF4.Dataset(path) as ds:
            rains.append(np.asarray(accumulated_rain(ds, ti), dtype="f8"))
            if len(rains) == 1:
                lat = np.asarray(ds.variables["XLAT"][ti])
                lon = np.asarray(ds.variables["XLONG"][ti])
                dx_km = float(getattr(ds, "DX", 3000.0)) / 1000.0
    k = max(1, int(round(smooth_km / dx_km)) | 1)
    total = uniform_filter(rains[-1] - rains[0], size=k, mode="nearest")
    ok = np.zeros(total.shape, bool)
    try:
        m = caseinfo.spec_bdy_width(run_dir) + k
    except (caseinfo.CaseInfoError, OSError):
        m = caseinfo.DEFAULT_SPEC_BDY_WIDTH + k
    ok[m:total.shape[0] - m, m:total.shape[1] - m] = True
    in_box = False
    if box:
        inside = ((lat >= box[0]) & (lat <= box[1]) & (lon >= box[2]) & (lon <= box[3]))
        if (ok & inside).any():
            ok &= inside
            in_box = True
    if not ok.any():
        ok[:] = True
    j, i = np.unravel_index(np.nanargmax(np.where(ok, total, -np.inf)), total.shape)
    sl = (slice(max(0, j - k // 2), j + k // 2 + 1), slice(max(0, i - k // 2), i + k // 2 + 1))
    best, peak = -1.0, after[-1][0]
    for a, b, ra, rb in zip(after, after[1:], rains, rains[1:]):
        h = _hours(a[0], b[0])
        rate = float(np.nanmean(rb[sl] - ra[sl])) / h if h > 0 else 0.0
        if rate > best:
            best, peak = rate, b[0]
    rule = "heaviest post-spin-up rain (%g km mean)%s" % (smooth_km,
                                                          " in post.focus" if in_box else "")
    return dict(lat=float(lat[j, i]), lon=float(lon[j, i]), j=int(j), i=int(i), peak_time=peak,
                peak_rate_mm_h=max(best, 0.0), total_mm=float(total[j, i]),
                start=after[0][0], end=after[-1][0], rule=rule)

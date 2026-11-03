"""post/recipes/hovmoller.py -- precip_hovmoller: rain against distance along a line and time,
model and observation side by side, so a reader sees WHERE across the band the rain fell and
WHEN (the position and timing errors a map at one time cannot show).

Line: project.yaml cross_sections.hovmoller (`wtune project set --cross-section
hovmoller=lat0,lon0,lat1,lon1`), else through the rain core (post/focus.rain_focus inside
post.focus) ACROSS the mean 0-1.5 km wind there at the rain peak, HALF_KM each side: a band
lying along the low-level flow (south-west flow over Taiwan) then shows its cross-flow position
in time. Model: the finest domain's hourly rain, and its accumulation over each observation
window; observation: every agent gridded_precip dataset (wtune-obs/1) over those windows.
Each line point takes the nearest grid cell (a 3x3 block mean on the model grid); a point
farther than 1.5 cells from any cell (off the grid) is NaN. Values go to
precip_hovmoller_<run>.csv (panel, window, distance_km, mm).
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime, timedelta

from wtune.post import figdpi

WRF_FMT = "%Y-%m-%d_%H:%M:%S"
HALF_KM = 250.0
N_POINTS = 101
WIND_TOP_M = 1500.0
MODEL_HALFWIDTH = 1


def across_flow_line(lat, lon, u, v, half_km=HALF_KM):
    """(lat0, lon0, lat1, lon1) of the line through (lat, lon) perpendicular to the wind (u east,
    v north), half_km each side; the first end is to the right of the flow, the second to the
    left (south-west flow: south-east -> north-west). Calm: west -> east."""
    sp = math.hypot(u, v)
    px, py = ((-v / sp, u / sp) if sp >= 1.0 else (1.0, 0.0))
    dlat = half_km * py / 111.2
    dlon = half_km * px / (111.2 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


def sample_line(fields, lat2d, lon2d, line, n=N_POINTS, halfwidth=0):
    """(distance km from the line centre [n], values [len(fields), n]); NaN off the grid."""
    import numpy as np
    from wtune.post.analysis_kit import haversine_km, line_points, nearest_index
    from wtune.post.conventions import continuous_lons
    lat2d = np.asarray(lat2d, "f8")
    lon2d = continuous_lons(np.asarray(lon2d, "f8"))
    lats, lons, dist = line_points(*line, n=n)
    lons = (lons - float(lon2d.flat[lon2d.size // 2]) + 180.0) % 360.0 - 180.0 + \
        float(lon2d.flat[lon2d.size // 2])
    step = float(np.nanmedian(haversine_km(lat2d[:, :-1], lon2d[:, :-1], lat2d[:, 1:],
                                           lon2d[:, 1:]))) if lat2d.shape[1] > 1 else 50.0
    cells = []
    for la, lo in zip(lats, lons):
        j, i, d = nearest_index(lat2d, lon2d, la, lo)
        cells.append((j, i) if d <= 1.5 * step else None)
    h = int(halfwidth)
    out = np.full((len(fields), n), np.nan)
    for t, f in enumerate(fields):
        f = np.asarray(f, "f8")
        for k, c in enumerate(cells):
            if c is not None:
                j, i = c
                out[t, k] = np.nanmean(f[max(0, j - h):j + h + 1, max(0, i - h):i + h + 1])
    return dist - dist[-1] / 2.0, out


def _low_level_wind(path, ti, j, i):
    """Mean earth-relative wind (u, v) m/s of the levels below WIND_TOP_M AGL at (j, i)."""
    import netCDF4
    import numpy as np
    import wrf
    with netCDF4.Dataset(path) as ds:
        uv = np.asarray(wrf.getvar(ds, "uvmet", timeidx=ti))[:, :, j, i]
        z = np.asarray(wrf.getvar(ds, "height_agl", timeidx=ti))[:, j, i]
    keep = z <= max(WIND_TOP_M, float(z[0]))
    return float(uv[0][keep].mean()), float(uv[1][keep].mean())


def default_line(run_dir, domain, frames, spinup_hours, focus_box, map_time):
    """(line, rule) through the rain core across the low-level flow at the rain peak."""
    from wtune.post.focus import frame_at, rain_focus
    f = rain_focus(run_dir, domain, spinup_hours, box=focus_box)
    if not f:
        raise ValueError("no rain core on d%02d" % domain)
    t, path, ti = frame_at(frames, map_time or f["peak_time"])
    u, v = _low_level_wind(path, ti, f["j"], f["i"])
    line = across_flow_line(f["lat"], f["lon"], u, v)
    rule = ("through the rain core %.2fN %.2fE across the 0-1.5 km wind (%.1f m/s from %.0f deg) "
            "at %s" % (f["lat"], f["lon"], math.hypot(u, v),
                       (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0, t[:13] + "Z"))
    return line, rule


def model_series(frames):
    """(times [datetime], lat2d, lon2d, accumulated rain mm [frames])."""
    import netCDF4
    import numpy as np
    from wtune.post.extract import accumulated_rain
    times, acc, lat, lon = [], [], None, None
    for t, path, ti in frames:
        with netCDF4.Dataset(path) as ds:
            if lat is None:
                lat = np.asarray(ds.variables["XLAT"][ti])
                lon = np.asarray(ds.variables["XLONG"][ti])
            acc.append(np.asarray(accumulated_rain(ds, ti), "f8"))
        times.append(datetime.strptime(t, WRF_FMT))
    return times, lat, lon, acc


def _at(times, t, tol=timedelta(minutes=10)):
    best = min(range(len(times)), key=lambda k: abs(times[k] - t))
    return best if abs(times[best] - t) <= tol else None


def obs_panels(project_dir, times, acc, lat, lon, line):
    """[(name, label, windows [(start, end)], obs values [w, n], model values [w, n])] for every
    usable agent gridded_precip dataset; model accumulated over the same windows."""
    import numpy as np
    from wtune.post import agent_obs
    ok, _bad = agent_obs.usable(project_dir, "gridded_precip") if project_dir else ([], [])
    out = []
    for name, d, prov in ok:
        wins, ofields, mfields = [], [], []
        for (acc_h, end), path in sorted(agent_obs.grid_index(d).items(), key=lambda kv: kv[0][1]):
            start = end - timedelta(hours=acc_h)
            k0, k1 = _at(times, start), _at(times, end)
            if k0 is None or k1 is None:
                continue
            olat, olon, opr = agent_obs.read_grid(path)
            olon2, olat2 = np.meshgrid(olon, olat)
            wins.append((start, end))
            ofields.append((olat2, olon2, opr))
            mfields.append(acc[k1] - acc[k0])
        if not wins:
            continue
        dist, ov = None, []
        for olat2, olon2, opr in ofields:
            dist, v = sample_line([opr], olat2, olon2, line)
            ov.append(v[0])
        _d, mv = sample_line(mfields, lat, lon, line, halfwidth=MODEL_HALFWIDTH)
        out.append((name, name.upper(), wins, np.array(ov), mv))
    return out


def write_csv(path, dist, panels):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "start_utc", "end_utc", "distance_km", "mm"])
        for name, wins, values in panels:
            for (s, e), row in zip(wins, values):
                for d, v in zip(dist, row):
                    w.writerow([name, s.strftime("%Y-%m-%dT%H:%MZ"), e.strftime("%Y-%m-%dT%H:%MZ"),
                                "%.1f" % d, "" if v != v else "%.2f" % v])
    return path


def _edges(times):
    """Cell edges (datetimes) for windows [(start, end)]."""
    return [w[0] for w in times] + [times[-1][1]]


def plot(out_path, run_label, domain, dist, hourly_wins, hourly, obs, lat, lon, line, rule,
         spin_end, peak, across_flow=True):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import BoundaryNorm
    from wtune.post.rainscale import rain_scale
    ncol = 2 * len(obs) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.0 * ncol + 0.8, 6.4), sharey=True,
                             layout="constrained", squeeze=False)
    axes = axes[0]
    panels = []
    for name, label, wins, ov, mv in obs:
        hi = float(np.nanmax([np.nanmax(ov), np.nanmax(mv), 1.0]))
        hours = int(round((wins[0][1] - wins[0][0]).total_seconds() / 3600.0))
        panels += [(ov, wins, "%s, mm / %d h" % (label, hours), hi, False),
                   (mv, wins, "WRF d%02d, mm / %d h" % (domain, hours), hi, True)]
    panels.append((hourly, hourly_wins, "WRF d%02d, mm / h" % domain,
                   float(np.nanmax([np.nanmax(hourly), 1.0])), True))
    xlabel = ("distance from the line centre (km)\n- right of the flow  |  left +" if across_flow
              else "distance from the line centre (km)\n%.2fN %.2fE -  |  + %.2fN %.2fE" % line)
    xe = np.concatenate([[dist[0] - (dist[1] - dist[0]) / 2],
                         (dist[:-1] + dist[1:]) / 2, [dist[-1] + (dist[1] - dist[0]) / 2]])
    for ax, (vals, wins, title, hi, model) in zip(axes, panels):
        levels, cmap, _n = rain_scale(lat, lon, hi)
        norm = BoundaryNorm(levels, cmap.N, extend="max" if cmap.N > len(levels) - 1 else "neither")
        ye = mdates.date2num(_edges(wins))
        m = ax.pcolormesh(xe, ye, np.ma.masked_less(np.ma.masked_invalid(vals), levels[0]),
                          cmap=cmap, norm=norm, shading="flat")
        fig.colorbar(m, ax=ax, location="bottom", shrink=0.9, aspect=25, label="mm")
        ax.axvline(0.0, color="0.3", lw=0.8, ls=":")
        if spin_end is not None and model:
            ax.axhspan(ye[0], mdates.date2num(spin_end), facecolor="none", edgecolor="0.45",
                       hatch="///", lw=0, zorder=3)
        if peak is not None:
            ax.axhline(mdates.date2num(peak), color="k", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.yaxis_date()
        ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d %HZ"))
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("time (UTC); hatched = model spin-up, dashed = rain peak", fontsize=9)
    fig.suptitle("%s  rain along %.2fN %.2fE -> %.2fN %.2fE (%s)"
                 % (run_label, line[0], line[1], line[2], line[3], rule), fontsize=9, wrap=True)
    fig.savefig(out_path, dpi=figdpi.dpi(110))
    plt.close(fig)
    return out_path


def figure(run_dir, run_label, domain, frames, out_dir, spinup_hours, project_dir, focus_box,
           map_time, line_override=None):
    """The precip_hovmoller fig() or missing() entry."""
    import numpy as np
    from wtune.post.recipe import fig, missing
    if len(frames) < 3:
        return missing("precip_hovmoller", "wrfout_d%02d_*" % domain, "fewer than 3 frames")
    try:
        if line_override:
            line = tuple(float(line_override[k]) for k in ("lat0", "lon0", "lat1", "lon1"))
            rule = "project.yaml cross_sections.hovmoller"
        else:
            line, rule = default_line(run_dir, domain, frames, spinup_hours, focus_box, map_time)
        times, lat, lon, acc = model_series(frames)
        hourly_wins = list(zip(times[:-1], times[1:]))
        dt_h = np.array([(b - a).total_seconds() / 3600.0 for a, b in hourly_wins])
        dist, mv = sample_line([b - a for a, b in zip(acc[:-1], acc[1:])], lat, lon, line,
                               halfwidth=MODEL_HALFWIDTH)
        hourly = mv / dt_h[:, None]
        obs = obs_panels(project_dir, times, acc, lat, lon, line)
    except (ImportError, OSError, KeyError, ValueError) as e:
        return missing("precip_hovmoller", "wrfout_d%02d rain / agent gridded_precip" % domain,
                       "%s: %s" % (type(e).__name__, e))
    stem = os.path.join(out_dir, "precip_hovmoller_%s" % run_label)
    write_csv(stem + ".csv", dist, [("wrf_hourly", hourly_wins, hourly)]
              + [p for name, _l, wins, ov, mv_ in obs
                 for p in ((name, wins, ov), ("wrf_" + name, wins, mv_))])
    spin_end = times[0] + timedelta(hours=spinup_hours) if spinup_hours else None
    peak = datetime.strptime(map_time, WRF_FMT) if map_time else None
    path = plot(stem + ".png", run_label, domain, dist, hourly_wins, hourly, obs, lat, lon, line,
                rule, spin_end, peak, across_flow=not line_override)
    return fig("precip_hovmoller", path, run_label, domain="d%02d" % domain,
               valid_time="%s..%s" % (frames[0][0], frames[-1][0]),
               extra={"line": list(line), "rule": rule, "csv": stem + ".csv",
                      "observations": [o[0] for o in obs]})

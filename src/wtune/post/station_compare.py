"""post/station_compare.py -- model vs. observation at named stations, one series per station,
variable and domain (`wtune post stations`). Reads a wtune-obs/1 `stations` dataset
(post/agent_obs.py) and the run's wrfout files.

Per station and per domain whose grid contains it (point inside the outline of the domain's
XLAT/XLONG mass points):

- the model value is the nearest mass point (`nearest`) or the bilinear interpolation inside the
  grid cell holding the station (`bilinear`; the cell is found on the curvilinear XLAT/XLONG grid,
  falling back to nearest where no cell encloses the point). Winds are interpolated as
  earth-relative components (U10/V10 rotated by SINALPHA/COSALPHA) and speed / direction derived
  afterwards. No station-height correction is applied: `model_hgt_minus_elev_m` reports the
  mismatch instead.
- variables: t2 (T2, degC), td2 (from Q2 and PSFC, degC), wspd10 (m/s), wdir10 (meteorological,
  degrees from north clockwise), gust (WSPD10MAX when the run wrote it, else post/surface_diag's
  UPP-style estimate). slp is not supported.
  gust caveat: ASOS reports a gust only when one occurred, so the paired gust sample is
  conditional on gusty hours; station_stats.csv carries a `note` column saying so and the
  compare notes and the gust panel title repeat it.
- pairing: a model output time is paired with the observation of that station and variable
  nearest in time, if within `tol_min` (default 10 min: hourly ASOS reports at :51-:55 pair with
  the on-the-hour output). A grid point on the outermost ring is not compared.
- statistics: bias = mean(model - obs), RMSE; wind direction uses the circular difference
  d = ((model - obs + 180) % 360) - 180, and pairs whose observed wind speed is below `calm_kt`
  are left out of the direction statistics (counted in n_calm_excluded).
"""
from __future__ import annotations

import csv
import math
import os
from datetime import datetime

from wtune.post import agent_obs

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
TIME_FMT = agent_obs.TIME_FMT
VARIABLES = ("t2", "td2", "wspd10", "wdir10", "gust")
UNITS = {"t2": "degC", "td2": "degC", "wspd10": "m/s", "wdir10": "deg", "gust": "m/s"}
KT_MS = 1852.0 / 3600.0
SERIES_COLS = ("station", "domain", "variable", "time_utc", "obs", "model", "diff", "calm")
STATS_COLS = ("station", "domain", "variable", "n", "bias", "rmse", "n_calm_excluded",
              "model_hgt_minus_elev_m", "method", "grid_j", "grid_i", "model_hgt_m", "note")
# ASOS reports a gust only when one occurred ("M" = no gust, not a missing value), so the paired
# gust sample is conditional on gusty hours and its bias/RMSE are not a general gust skill.
GUST_SAMPLING_NOTE = "obs only when gusts occur; model gust source = %s"
FIGURE_ID = "station_series"


class StationError(Exception):
    pass


def circ_diff(model, obs):
    """Model minus observation for directions in degrees, in [-180, 180)."""
    return ((model - obs + 180.0) % 360.0) - 180.0


def wind_dir(u, v):
    """Meteorological direction (degrees from north, clockwise; where the wind blows from)."""
    import numpy as np
    return np.mod(180.0 + np.degrees(np.arctan2(u, v)), 360.0)


def dewpoint_c(q2, psfc):
    """Dew point (degC) from the 2 m mixing ratio (kg/kg) and surface pressure (Pa): vapour
    pressure e = w p / (0.622 + w), Bolton (1980) inverse of the saturation formula."""
    import numpy as np
    w = np.maximum(q2, 1e-8)
    e = np.maximum(w * psfc / (0.622 + w), 1e-3)             # Pa
    ln = np.log(e / 611.2)
    return 243.5 * ln / (17.67 - ln)


# -- geometry -------------------------------------------------------------------------------
def _inside(lat, lon, slat, slon):
    """True when (slat, slon) is inside the polygon through the grid's outer ring of points."""
    import numpy as np
    ring = np.concatenate([np.c_[lon[0, :], lat[0, :]], np.c_[lon[1:, -1], lat[1:, -1]],
                           np.c_[lon[-1, -2::-1], lat[-1, -2::-1]],
                           np.c_[lon[-2:0:-1, 0], lat[-2:0:-1, 0]]])
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = (y > slat) != (y2 > slat)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x + (slat - y) * (x2 - x) / (y2 - y)
    return bool(np.count_nonzero(cross & (slon < xi)) % 2)


def locate(lat, lon, slat, slon, method):
    """((j, i) of the nearest grid point, [(j, i, weight)] of the points that give the model
    value at the station), or None when the nearest point is on the outermost ring of the grid
    (the lateral-boundary zone is not compared). Distance: dlat^2 + (dlon cos lat)^2."""
    import numpy as np
    c = math.cos(math.radians(slat))
    d2 = (lat - slat) ** 2 + ((lon - slon) * c) ** 2
    i0, j0 = (int(k) for k in np.unravel_index(int(np.argmin(d2)), d2.shape))
    ny, nx = lat.shape
    if i0 in (0, ny - 1) or j0 in (0, nx - 1):
        return None
    near = (i0, j0)
    if method == "nearest":
        return near, [(i0, j0, 1.0)]
    px = (lon - slon) * c
    py = lat - slat
    for ci in (i0 - 1, i0):
        for cj in (j0 - 1, j0):
            if ci < 0 or cj < 0 or ci + 1 >= ny or cj + 1 >= nx:
                continue
            x = [px[ci, cj], px[ci, cj + 1], px[ci + 1, cj], px[ci + 1, cj + 1]]
            y = [py[ci, cj], py[ci, cj + 1], py[ci + 1, cj], py[ci + 1, cj + 1]]
            s = t = 0.5
            for _ in range(30):             # solve P(s, t) = 0 (the station) by Newton
                w = ((1 - t) * (1 - s), (1 - t) * s, t * (1 - s), t * s)
                fx = sum(a * b for a, b in zip(w, x))
                fy = sum(a * b for a, b in zip(w, y))
                dxs = (1 - t) * (x[1] - x[0]) + t * (x[3] - x[2])
                dxt = (1 - s) * (x[2] - x[0]) + s * (x[3] - x[1])
                dys = (1 - t) * (y[1] - y[0]) + t * (y[3] - y[2])
                dyt = (1 - s) * (y[2] - y[0]) + s * (y[3] - y[1])
                det = dxs * dyt - dxt * dys
                if det == 0:
                    break
                s -= (fx * dyt - fy * dxt) / det
                t -= (dxs * fy - dys * fx) / det
            if -1e-6 <= s <= 1 + 1e-6 and -1e-6 <= t <= 1 + 1e-6 and abs(fx) + abs(fy) < 1e-6:
                s, t = min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)
                return near, [(ci, cj, (1 - t) * (1 - s)), (ci, cj + 1, (1 - t) * s),
                              (ci + 1, cj, t * (1 - s)), (ci + 1, cj + 1, t * s)]
    return near, [(i0, j0, 1.0)]


def _interp(field, pts):
    return sum(w * float(field[i, j]) for i, j, w in pts)


# -- model side -----------------------------------------------------------------------------
def _frames(run_dir):
    """{domain: [(datetime, path, timeidx)]}."""
    from wtune.post import extract
    out = {}
    for (dom, tstr), (path, idx) in extract.frame_refs(run_dir).items():
        try:
            out.setdefault(dom, []).append((datetime.strptime(tstr, WRF_TIME_FMT), path, idx))
        except ValueError:
            continue
    return {d: sorted(v) for d, v in sorted(out.items())}


def _surface_fields(ds, idx, wanted):
    """{name: 2-D array}: t2, td2, u10, v10 (earth-relative), gust as needed; plus method text."""
    import numpy as np
    v = ds.variables

    def rd(name):
        return np.asarray(v[name][idx], "f8")
    out, method = {}, {}
    if "t2" in wanted:
        out["t2"] = rd("T2") - 273.15
    if "td2" in wanted:
        out["td2"] = dewpoint_c(rd("Q2"), rd("PSFC"))
    if wanted & {"wspd10", "wdir10"}:
        u, w = rd("U10"), rd("V10")
        if "SINALPHA" in v and "COSALPHA" in v:
            sa, ca = rd("SINALPHA"), rd("COSALPHA")
            u, w = u * ca - w * sa, w * ca + u * sa
        else:
            method["wind"] = "no SINALPHA/COSALPHA: U10/V10 taken as earth-relative"
        out["u10"], out["v10"] = u, w
    if "gust" in wanted:
        if "WSPD10MAX" in v:
            out["gust"] = rd("WSPD10MAX")
            method["gust"] = "WSPD10MAX (maximum over the history interval)"
        else:
            from wtune.post import surface_diag
            g, text = surface_diag.gust(ds, idx)
            out["gust"], method["gust"] = g, text
    return out, method


def _gust_source(methods):
    """WSPD10MAX or estimated (surface_diag's UPP-style gust) from _surface_fields' method text."""
    text = methods.get("gust")
    if text is None:
        return "unknown (no model frame paired)"
    return "WSPD10MAX" if text.startswith("WSPD10MAX") else "estimated"


def _stat(pairs):
    import numpy as np
    if not pairs:
        return 0, float("nan"), float("nan")
    d = np.asarray(pairs, "f8")
    return len(d), float(d.mean()), float(np.sqrt((d ** 2).mean()))


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else ("%.*f" % (nd, x))


# -- main -----------------------------------------------------------------------------------
def resolve_run_dir(path):
    """A wrfout directory, or a wtune project with exactly one run."""
    if os.path.isfile(os.path.join(path, "project.yaml")):
        import wtune.project
        runs = [p for _i, p in wtune.project.discover_runs(path)]
        if len(runs) != 1:
            raise StationError("%s has %d runs; pass the run directory itself" % (path, len(runs)))
        return runs[0]
    return path


def compare(run_dir, stations_ds_dir, station_ids, variables, domains=None, method="nearest",
            calm_kt=3.0, tol_min=None, out_dir=None):
    """Compare wrfout at stations with a `stations` dataset (see the module doc).

    Returns dict(series=[row], stats=[row], notes=[str], figures=[path], files=[path]); with
    `out_dir` also writes station_series.csv, station_stats.csv and one PNG per station and
    variable (stations/station_<ID>_<var>.png) and registers the PNGs in out_dir/figures.json.
    Raises StationError for input that cannot be compared."""
    run_dir = resolve_run_dir(run_dir)
    if method not in ("nearest", "bilinear"):
        raise StationError("method must be nearest or bilinear, not %r" % method)
    err, _warn = agent_obs.check(stations_ds_dir)
    if err:
        raise StationError("%s does not pass obs-check: %s" % (stations_ds_dir, err[0]))
    prov = agent_obs.read_provenance(stations_ds_dir) or {}
    if prov.get("kind") != "stations":
        raise StationError("%s is kind %r, not stations" % (stations_ds_dir, prov.get("kind")))
    rows = agent_obs.read_stations(os.path.join(stations_ds_dir, agent_obs.STATIONS_FILE))
    notes = []
    # the default is the whole dataset: every station and every variable the model side supports
    in_ds = sorted({r["id"] for r in rows})
    present = {r["var"] for r in rows}
    if not station_ids or list(station_ids) == ["all"]:
        station_ids = in_ds
    elif any(i not in station_ids for i in in_ds):
        notes.append("stations in the dataset but NOT analysed (partial analysis): %s"
                     % ", ".join(i for i in in_ds if i not in station_ids))
    if not variables or list(variables) == ["all"]:
        variables = [v for v in VARIABLES if v in present]
    bad = [v for v in variables if v not in VARIABLES]
    if bad:
        raise StationError("unsupported variable(s) %s; supported: %s"
                           % (", ".join(bad), ", ".join(VARIABLES)))
    skipped = sorted(present - set(variables))
    if skipped:
        notes.append("variables in the dataset but NOT compared: %s%s" % (
            ", ".join(skipped),
            " (unsupported on the model side)" if set(skipped) - set(VARIABLES) else ""))
    obs = {}                              # (station, var) -> sorted [(time, value)]
    pos = {}                              # station -> (lat, lon, elev)
    for r in rows:
        if r["id"] in station_ids:
            obs.setdefault((r["id"], r["var"]), []).append((r["time"], r["value"]))
            pos.setdefault(r["id"], (r["lat"], r["lon"], r["elev_m"]))
    for k in obs:
        obs[k].sort()
    for sid in station_ids:
        if sid in pos and pos[sid][2] is None:
            notes.append("station %s: no elev_m in the dataset, height difference left empty "
                         "(wtune event obs-convert asos --station ID=lat,lon,elev)" % sid)
        if sid not in pos:
            notes.append("station %s: not in the dataset" % sid)
    stations = [s for s in station_ids if s in pos]
    speed = {(s, t): v for (s, var), lst in obs.items() if var == "wspd10" for t, v in lst}
    frames = _frames(run_dir)
    if not frames:
        raise StationError("no wrfout frames in %s" % run_dir)
    doms = [d for d in frames if domains is None or d in domains]
    if not doms:
        raise StationError("domain(s) %s not in %s (has %s)" % (domains, run_dir, list(frames)))
    import netCDF4
    import numpy as np
    series, stats = [], []
    placed = {s: False for s in stations}
    for dom in doms:
        fr = frames[dom]
        with netCDF4.Dataset(fr[0][1]) as ds:
            lat = np.asarray(ds.variables["XLAT"][fr[0][2]], "f8")
            lon = np.asarray(ds.variables["XLONG"][fr[0][2]], "f8")
            hgt = np.asarray(ds.variables["HGT"][fr[0][2]], "f8")
        times = [f[0] for f in fr]
        tol = 10.0 if tol_min is None else tol_min
        if len(times) > 1:
            step = float(np.median(np.diff([t.timestamp() for t in times]))) / 60.0
            if step > 60.0:
                notes.append("d%02d: wrfout interval %g min -- only observations within %g min of "
                             "an output time are paired; set the history interval to 60 min or "
                             "less for hourly station data" % (dom, step, tol))
        else:
            notes.append("d%02d: one wrfout time only; pairing tolerance %g min" % (dom, tol))
        mine = {}
        for sid in stations:
            if _inside(lat, lon, pos[sid][0], pos[sid][1]):
                loc = locate(lat, lon, pos[sid][0], pos[sid][1], method)
                if loc is None:
                    notes.append("d%02d: station %s is in the outermost grid ring, not compared"
                                 % (dom, sid))
                    continue
                mine[sid] = loc
                placed[sid] = True
        if not mine:
            continue
        # which output times pair with which observation, per station and variable
        todo = {}                      # frame index -> {(station, var): (obs time, obs value)}
        for fi, (t, _p, _i) in enumerate(fr):
            for sid in mine:
                for var in variables:
                    lst = obs.get((sid, var))
                    if not lst:
                        continue
                    best = min(lst, key=lambda a: abs((a[0] - t).total_seconds()))
                    if abs((best[0] - t).total_seconds()) <= tol * 60.0:
                        todo.setdefault(fi, {})[(sid, var)] = best
        want = set(variables)
        cache, methods = {}, {}
        for fi in sorted(todo):
            t, path, idx = fr[fi]
            if path not in cache:
                for c in cache.values():
                    c.close()
                cache = {path: netCDF4.Dataset(path)}
            fields, m = _surface_fields(cache[path], idx, want)
            methods.update(m)
            for (sid, var), (ot, ov) in sorted(todo[fi].items()):
                pts = mine[sid][1]
                if var == "wspd10":
                    mv = math.hypot(_interp(fields["u10"], pts), _interp(fields["v10"], pts))
                elif var == "wdir10":
                    mv = float(wind_dir(_interp(fields["u10"], pts), _interp(fields["v10"], pts)))
                else:
                    mv = _interp(fields[var], pts)
                calm = 0
                if var == "wdir10":
                    d = float(circ_diff(mv, ov))
                    sp = speed.get((sid, ot))
                    calm = int(sp is not None and round(sp / KT_MS, 2) < calm_kt)   # stored m/s is rounded: 3 kt is not calm
                else:
                    d = mv - ov
                series.append(dict(station=sid, domain="d%02d" % dom, variable=var,
                                   time_utc=t.strftime(TIME_FMT), obs=ov, model=mv, diff=d,
                                   calm=calm))
        for c in cache.values():
            c.close()
        for k, text in methods.items():
            notes.append("d%02d %s: %s" % (dom, k, text))
        for sid, (near, pts) in sorted(mine.items()):
            h = _interp(hgt, pts)
            for var in variables:
                if (sid, var) not in obs:
                    continue
                rs = [r for r in series if (r["station"], r["domain"], r["variable"])
                      == (sid, "d%02d" % dom, var)]
                used = [r["diff"] for r in rs if not r["calm"]]
                n, bias, rmse = _stat(used)
                stats.append(dict(station=sid, domain="d%02d" % dom, variable=var, n=n, bias=bias,
                                  rmse=rmse, n_calm_excluded=sum(r["calm"] for r in rs),
                                  model_hgt_minus_elev_m=None if pos[sid][2] is None
                                  else h - pos[sid][2], method=method,
                                  grid_j=mine[sid][0][0], grid_i=mine[sid][0][1],
                                  model_hgt_m=h,
                                  note=GUST_SAMPLING_NOTE % _gust_source(methods)
                                  if var == "gust" else ""))
    gust_stats = [st for st in stats if st["variable"] == "gust"]
    if gust_stats:
        notes.append("gust sampling bias: ASOS reports a gust only when one occurred (M = no "
                     "gust), so these samples are gust-reports-only, not all hours -- "
                     + ", ".join("%s %s n=%d bias=%s" % (st["station"], st["domain"], st["n"],
                                                         _fmt(st["bias"], 2)) for st in gust_stats)
                     + "; read bias/RMSE as skill at gusty times, not overall")
    for sid, ok in placed.items():
        if not ok:
            notes.append("station %s (%.3f, %.3f): outside every selected domain" % (
                sid, pos[sid][0], pos[sid][1]))
    for var in variables:
        if not any(k[1] == var for k in obs):
            notes.append("%s: the dataset has no observation of it for the selected stations" % var)
    if "wdir10" in variables and not any(k[1] == "wspd10" for k in obs):
        notes.append("wdir10: no wspd10 observations, so calm winds cannot be excluded")
    result = dict(series=series, stats=stats, notes=notes, figures=[], files=[])
    if out_dir:
        _write(result, out_dir, run_dir, stations, variables, pos, calm_kt, method)
    return result


# -- outputs --------------------------------------------------------------------------------
def _write(result, out_dir, run_dir, stations, variables, pos, calm_kt, method):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "station_series.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SERIES_COLS)
        for r in result["series"]:
            w.writerow([r["station"], r["domain"], r["variable"], r["time_utc"], _fmt(r["obs"]),
                        _fmt(r["model"]), _fmt(r["diff"]), r["calm"]])
    result["files"].append(p)
    p = os.path.join(out_dir, "station_stats.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(STATS_COLS)
        for r in result["stats"]:
            w.writerow([r["station"], r["domain"], r["variable"], r["n"], _fmt(r["bias"]),
                        _fmt(r["rmse"]), r["n_calm_excluded"], _fmt(r["model_hgt_minus_elev_m"], 1),
                        r["method"], r["grid_j"], r["grid_i"], _fmt(r["model_hgt_m"], 1),
                        r["note"]])
    result["files"].append(p)
    figs = _draw(result, out_dir, stations, variables, pos, calm_kt, method)
    result["figures"] = figs
    _register(out_dir, run_dir, figs, result["notes"])


def _draw(result, out_dir, stations, variables, pos, calm_kt, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from wtune.post import figdpi
    os.makedirs(os.path.join(out_dir, "stations"), exist_ok=True)
    out = []
    for sid in stations:
        for var in variables:
            rows = [r for r in result["series"] if r["station"] == sid and r["variable"] == var]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(8, 3.8))
            circ = var == "wdir10"

            def line(t, y, **kw):
                y = np.asarray(y, "f8")
                cuts = [0]
                if circ and len(y) > 1:                  # break the line at 360 -> 0 jumps
                    cuts += list(np.where(np.abs(np.diff(y)) > 180.0)[0] + 1)
                cuts.append(len(y))
                for n, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                    if n:
                        kw.pop("label", None)
                    ax.plot(t[a:b], y[a:b], **kw)
            def lab(dom):
                st = [x for x in result["stats"] if (x["station"], x["domain"], x["variable"])
                      == (sid, dom, var)]
                return "%s (%.0f m)" % (dom, st[0]["model_hgt_m"]) if st else dom
            tobs = sorted({(r["time_utc"], r["obs"]) for r in rows if not r["calm"]})
            parse = lambda s: datetime.strptime(s, TIME_FMT)       # noqa: E731
            line([parse(a) for a, _ in tobs], [b for _, b in tobs], color="k", marker="o",
                 ms=3.5, lw=1.0, label="observation")
            summ = []
            for k, dom in enumerate(sorted({r["domain"] for r in rows})):
                dr = sorted((r for r in rows if r["domain"] == dom), key=lambda r: r["time_utc"])
                line([parse(r["time_utc"]) for r in dr], [r["model"] for r in dr], marker="s",
                     ms=2.5, lw=1.2, color="C%d" % k, label=lab(dom))
                st = [s for s in result["stats"] if (s["station"], s["domain"], s["variable"])
                      == (sid, dom, var)]
                if st:
                    s = st[0]
                    summ.append("%s bias %s RMSE %s n=%d%s" % (
                        dom, _fmt(s["bias"], 2), _fmt(s["rmse"], 2), s["n"],
                        " (%d calm excl.)" % s["n_calm_excluded"] if circ else ""))
            if circ:
                ax.set_ylim(0, 360)
                ax.set_yticks(range(0, 361, 90))
            ax.set_ylabel("%s (%s)" % (var, UNITS[var]))
            ax.set_title("%s  %s   %s\n%s" % (sid, var + (" (gust reports only)"
                                                          if var == "gust" else ""), " | ".join(summ), ("elev %.0f m, %s" % (pos[sid][2], method)
                                                    if pos[sid][2] is not None
                                                    else "elev unknown, %s" % method)), fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
            fig.autofmt_xdate()
            fig.tight_layout()
            path = os.path.join(out_dir, "stations", "station_%s_%s.png" % (sid, var))
            fig.savefig(path, dpi=figdpi.dpi())
            plt.close(fig)
            out.append(path)
    return out


def _register(out_dir, run_dir, figs, notes):
    """Add the PNGs to out_dir/figures.json (created when absent, otherwise extended)."""
    import json
    from wtune.post import recipe
    entries = [recipe.fig(FIGURE_ID, p, os.path.basename(os.path.normpath(run_dir)))
               for p in figs]
    idx = os.path.join(out_dir, recipe.INDEX_FILE)
    if os.path.isfile(idx):
        with open(idx) as fh:
            doc = json.load(fh)
        keep = [f for f in doc.get("figures") or [] if f.get("id") != FIGURE_ID]
        rel = []
        for e in entries:
            e = dict(e)
            e["path"] = os.path.relpath(e["path"], out_dir)
            rel.append(e)
        doc["figures"] = keep + rel
        with open(idx, "w") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False, default=str)
        return
    recipe.write_index(out_dir, [run_dir], [os.path.basename(os.path.normpath(run_dir))], {},
                       entries, [recipe.note("stations", n, level="info") for n in notes])

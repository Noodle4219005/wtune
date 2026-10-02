"""post/recipes/pbl.py -- the "pbl" figure set (decision 0057 section 2), rebuilt against real
WRF output (Katrina 12 km d01, hourly, 48 h; 2026-09-29):

  pbl_time_height     theta (K) vs time and height AGL 0-4 km at one interior LAND column, model
                      PBLH (the scheme's own PBLH variable) and the bulk-Richardson PBL height
                      of the model column, radiosonde bulk-Richardson PBLH as markers, T2 and
                      10 m wind underneath. Column = the radiosonde station with the largest
                      diurnal T2 range when IGRA data exist, else the interior land point with
                      the largest diurnal T2 range (never a sea or boundary-zone point).
  pbl_cross_terrain   W-E theta cross-section through that column, lateral boundary zone
                      excluded, at the time of the column's maximum model PBLH, PBLH overlaid.
  pbl_skewt_<id>      Skew-T of model vs radiosonde (IGRA), T/Td/wind, one panel per sounding
  pbl_height_compare  PBL height, radiosonde vs model, per station/sounding (bulk-Richardson on
                      both, plus the scheme's PBLH), with model-minus-obs T/Td/wspd statistics
                      written to pbl_obs_stats.csv

wrf-python getvar/vertcross for all model fields; MetPy for Skew-T and thermodynamics; IGRA
soundings come from `wtune post fetch igra` (post/fetch/igra.py) in <project>/input/obs/igra/.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

from post import tools
from post.extract import frame_ref, list_frame_refs

WRF_FMT = "%Y-%m-%d_%H:%M:%S"
TOP_AGL_M = 4000.0
COAST_BUFFER = 3          # cells: a "land" column must have land on every side this far out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _wrf():
    return __import__("wrf")


def _interior(ds, run_dir):
    from post import caseinfo
    ny, nx = ds.dimensions["south_north"].size, ds.dimensions["west_east"].size
    b = caseinfo._interior_bounds(ny, nx, caseinfo.spec_bdy_width(run_dir))
    return b if b else (0, ny, 0, nx)


def _obs_dir(run_dir, project_dir=None):
    """<project>/input/obs/igra if that exists. Project: explicit argument, else $WTUNE_PROJECT /
    a project.yaml above the run dir or the current directory."""
    cands = [project_dir]
    try:
        import wtune_project
        cands += [wtune_project.find_project(start_dir=run_dir), wtune_project.find_project()]
    except Exception:                      # no project / unresolvable env var: just no obs
        pass
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "input", "obs", "igra")):
            return os.path.join(c, "input", "obs", "igra")
    return None


def _load_stations(obs_dir):
    """[(id, name, lat, lon, elev, {datetime: sounding})] from stations.csv + <ID>.csv."""
    from post.fetch import igra
    out = []
    path = os.path.join(obs_dir, "stations.csv")
    if not os.path.exists(path):
        return out
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            f = os.path.join(obs_dir, row["id"] + ".csv")
            if os.path.exists(f):
                out.append((row["id"], row["name"], float(row["lat"]), float(row["lon"]),
                            float(row["elev_m"]), igra.read_soundings(f)))
    return out


def pick_column(run_dir, frames, stations=None):
    """(j, i, label, station_id|None): an interior LAND column with the largest T2 range over the
    run (a strong diurnal cycle = a boundary layer that actually grows and collapses). With
    `stations` ([(id, name, lat, lon, ...)]) only stations whose nearest column is interior land
    are candidates; without, every interior land column at least COAST_BUFFER cells from the
    coast (a sea-breeze/marine column is not the diagnostic wanted). Returns None when no column
    qualifies (caller reports missing, never a forced pick)."""
    import netCDF4
    import numpy as np
    from scipy.ndimage import minimum_filter
    wrf_mod = _wrf()
    with netCDF4.Dataset(frames[0][1]) as ds:
        j0, j1, i0, i1 = _interior(ds, run_dir)
        land = np.asarray(ds.variables["LANDMASK"][0]) == 1
        cands = []
        if stations:
            for st in stations:
                xy = wrf_mod.ll_to_xy(ds, st[2], st[3])
                i, j = int(xy[0]), int(xy[1])
                if j0 <= j < j1 and i0 <= i < i1 and land[j, i]:
                    cands.append((j, i, "%s %s" % (st[0][-5:], st[1].title()), st[0]))
        else:
            deep = minimum_filter(land, size=2 * COAST_BUFFER + 1, mode="constant", cval=False)
            mask = np.zeros_like(deep)
            mask[j0:j1, i0:i1] = True
            jj, ii = np.nonzero(deep & mask)
            step = max(1, len(jj) // 400)                 # <= ~400 candidates: bounds the I/O
            cands = [(int(j), int(i), "(j=%d,i=%d)" % (j, i), None) for j, i in zip(jj[::step], ii[::step])]
    if not cands:
        return None
    t2 = np.zeros((len(frames), len(cands)))
    for k, (_, path, ti) in enumerate(frames):
        with netCDF4.Dataset(path) as ds:
            arr = np.asarray(ds.variables["T2"][ti])
        t2[k] = [arr[c[0], c[1]] for c in cands]
    return cands[int(np.argmax(t2.max(axis=0) - t2.min(axis=0)))]


def column_series(frames, j, i):
    """Per frame (time, path, timeidx): (datetime, model column dict) at (j, i)."""
    import netCDF4
    from post import sounding
    wrf_mod = _wrf()
    out = []
    for time_str, path, ti in frames:
        with netCDF4.Dataset(path) as ds:
            col = sounding.model_column(ds, wrf_mod, j, i, ti)
            col["t2"] = float(ds.variables["T2"][ti, j, i])
            col["wspd10"] = float((ds.variables["U10"][ti, j, i] ** 2 + ds.variables["V10"][ti, j, i] ** 2) ** 0.5)
        out.append((datetime.strptime(time_str, WRF_FMT).replace(tzinfo=timezone.utc), col))
    return out


def plot_time_height(series, obs_pblh, label, lon, out_path, run_label, domain):
    """theta(t, z_AGL) + PBLH curves; obs_pblh = {datetime: m AGL} radiosonde bulk-Ri PBLH."""
    import numpy as np
    import matplotlib.dates as mdates
    from post import sounding
    plt = _plt()
    times = [t for t, _ in series]
    zg = np.arange(0.0, TOP_AGL_M + 1, 100.0)
    theta = np.full((len(zg), len(times)), np.nan)
    pblh_ri = []
    for n, (_, col) in enumerate(series):
        th = (col["t"] + 273.15) * (1000.0 / col["p"]) ** 0.2854
        theta[:, n] = np.interp(zg, col["z"], th)
        pblh_ri.append(sounding.pblh_bulk_ri(col))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    lo, hi = np.floor(np.nanmin(theta)), np.ceil(np.nanmax(theta))
    cf = ax.contourf(mdates.date2num(times), zg, theta, levels=np.arange(lo, hi + 1, 2.0), cmap="viridis")
    fig.colorbar(cf, ax=ax, label="theta (K)", pad=0.01)
    ax.plot(times, [c["pblh_model"] if c["pblh_model"] > 0 else np.nan for _, c in series], "r-", lw=1.8,
            label="model PBLH (scheme)")   # PBLH is 0 in the t=0 frame (not yet computed)
    ax.plot(times, pblh_ri, "w--", lw=1.6, label="model PBLH (bulk Ri=0.25)")
    if obs_pblh:
        ot = sorted(obs_pblh)
        ax.plot(ot, [obs_pblh[t] for t in ot], "kD", mfc="yellow", ms=7, ls="none",
                label="sonde PBLH (bulk Ri=0.25)")
    ax.set_ylim(0, TOP_AGL_M)
    ax.set_ylabel("height AGL (m)")
    ax.legend(fontsize=7, loc="upper left", ncol=3)
    ax.set_title("%s d%02d  %s (%.2f°E)  theta time-height, local solar time = UTC %+.1f h"
                 % (run_label, domain, label, lon, lon / 15.0), fontsize=10)
    ax2.plot(times, [c["t2"] - 273.15 for _, c in series], "r-")
    ax2.set_ylabel("T2 (degC)", color="r")
    ax3 = ax2.twinx()
    ax3.plot(times, [c["wspd10"] for _, c in series], "b-", lw=1)
    ax3.set_ylabel("10 m wind (m/s)", color="b")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hz"))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax2.set_xlabel("valid time (UTC)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _fill_below_valid(arr):
    """Fill NaN levels (below ground) with the lowest finite sample of their column, so contourf
    sees no masked corner (a masked corner drops the whole quad -> white wedges beside terrain);
    the terrain polygon drawn on top hides the fill."""
    import numpy as np
    arr = arr.copy()
    for c in range(arr.shape[1]):
        fin = np.flatnonzero(np.isfinite(arr[:, c]))
        if fin.size and fin[0] > 0:
            arr[:fin[0], c] = arr[fin[0], c]
    return arr


def plot_cross_section(run_dir, domain, time_str, j, i, out_path, run_label, label, halfwidth_km=500.0,
                       top_km=None):
    """W-E theta cross-section through row j, restricted to the domain interior columns."""
    import netCDF4
    import numpy as np
    wrf_mod = _wrf()
    plt = _plt()
    path, ti = frame_ref(run_dir, domain, time_str)
    with netCDF4.Dataset(path) as ds:
        j0, j1, i0, i1 = _interior(ds, run_dir)
        dx_km = float(ds.getncattr("DX")) / 1000.0
        half = int(round(halfwidth_km / dx_km))
        i0, i1 = max(i0, i - half), min(i1, i + half + 1)      # +-halfwidth around the column
        lat = np.asarray(ds.variables["XLAT"][ti])
        lon = np.asarray(ds.variables["XLONG"][ti])
        z = wrf_mod.getvar(ds, "z", timeidx=ti, units="m")
        theta = wrf_mod.getvar(ds, "theta", timeidx=ti)
        ter = wrf_mod.getvar(ds, "ter", timeidx=ti)
        top = (top_km * 1000.0 if top_km else
               float(np.ceil((np.nanmax(np.asarray(ter)[j, i0:i1]) + 3500.0) / 500.0) * 500.0))
        levels = np.arange(0.0, top + 1, 100.0)
        sp = wrf_mod.CoordPair(lat=float(lat[j, i0]), lon=float(lon[j, i0]))
        ep = wrf_mod.CoordPair(lat=float(lat[j, i1 - 1]), lon=float(lon[j, i1 - 1]))
        cross = wrf_mod.vertcross(theta, z, levels=levels, wrfin=ds, start_point=sp, end_point=ep,
                                  latlon=True)
        ter_line = np.asarray(wrf_mod.interpline(ter, wrfin=ds, start_point=sp, end_point=ep, latlon=True))
        pbl_line = np.asarray(wrf_mod.interpline(wrf_mod.getvar(ds, "PBLH", timeidx=ti), wrfin=ds,
                                                 start_point=sp, end_point=ep, latlon=True))
        lm_line = np.asarray(wrf_mod.interpline(wrf_mod.getvar(ds, "LANDMASK", timeidx=ti), wrfin=ds,
                                                start_point=sp, end_point=ep, latlon=True))
    arr = _fill_below_valid(np.asarray(cross))
    xs = np.linspace(0.0, dx_km * (i1 - 1 - i0), arr.shape[1])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cf = ax.contourf(xs, levels, arr, levels=np.arange(np.floor(np.nanmin(arr)), np.ceil(np.nanmax(arr)) + 1, 1.0),
                     cmap="viridis")
    fig.colorbar(cf, ax=ax, label="theta (K)", pad=0.01)
    ax.fill_between(xs, 0, ter_line, color="0.25", zorder=3)
    ax.fill_between(xs, 0, top * 0.02, where=lm_line < 0.5, color="tab:blue", zorder=5, label="water")
    ax.plot(xs, ter_line + pbl_line, "r-", lw=1.5, zorder=4, label="PBL top (scheme), MSL")
    xj = dx_km * (i - i0)
    ax.axvline(xj, color="w", ls=":", lw=1.2, zorder=4)
    ax.text(xj, top * 0.97, " " + label, color="w", fontsize=7, va="top", zorder=5)
    ax.set_ylim(0, top)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("distance west to east along row j=%d, interior only (km)" % j)
    ax.set_ylabel("height MSL (m)")
    ax.set_title("%s d%02d  W-E theta cross-section through %s  %s UTC" % (run_label, domain, label, time_str),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _station_comparisons(stations, frames_by_time, run_dir):
    """For every (station, sounding) with a matching hourly frame inside the domain interior:
    dict(sid, name, time, model col, obs col, stats, PBL heights)."""
    import netCDF4
    from post import sounding
    wrf_mod = _wrf()
    rows = []
    for sid, name, lat, lon, elev, sounds in stations:
        for t in sorted(sounds):
            ts = t.strftime(WRF_FMT)
            if ts not in frames_by_time:
                continue
            fpath, ti = frames_by_time[ts]
            with netCDF4.Dataset(fpath) as ds:
                j0, j1, i0, i1 = _interior(ds, run_dir)
                xy = wrf_mod.ll_to_xy(ds, lat, lon, timeidx=ti)
                i, j = int(xy[0]), int(xy[1])
                if not (j0 <= j < j1 and i0 <= i < i1):
                    continue
                model = sounding.model_column(ds, wrf_mod, j, i, ti)
            obs = sounding.obs_column(sounds[t], elev)
            if obs is None:
                continue
            rows.append(dict(sid=sid, name=name, time=t, model=model, obs=obs,
                             stats=sounding.profile_stats(model, obs),
                             pblh_obs=sounding.pblh_bulk_ri(obs),
                             pblh_model_ri=sounding.pblh_bulk_ri(model),
                             pblh_model=model["pblh_model"] if model["pblh_model"] > 0 else float("nan")))
    return rows


def _fmt(v):
    return "n/a" if v != v else "%.0f m" % v


def plot_skewt_station(rows, out_path, run_label, domain):
    """2x3 grid of Skew-Ts (>= 5.6 x 5.6 in per panel, fonts >= 9 pt), one panel per sounding;
    per-panel PBLH/rmse text box inside the axes, one legend for the whole figure."""
    from post import sounding
    plt = _plt()
    rows = sorted(rows, key=lambda r: r["time"])[:6]
    ncol = 3 if len(rows) > 2 else len(rows)
    nrow = (len(rows) + ncol - 1) // ncol
    fig = plt.figure(figsize=(6.2 * ncol, 5.0 * nrow + 0.6))
    skew = None
    for n, r in enumerate(rows):
        text = ("PBLH  sonde %s\n      WRF Ri %s\n      WRF scheme %s\n"
                "rmse 1000-500 hPa\n  T %.1f K  Td %.1f K\n  wspd %.1f m/s"
                % (_fmt(r["pblh_obs"]), _fmt(r["pblh_model_ri"]), _fmt(r["pblh_model"]),
                   r["stats"]["T"]["rmse"], r["stats"]["Td"]["rmse"], r["stats"]["wspd"]["rmse"]))
        skew = sounding.skewt_overlay(fig, (nrow, ncol, n + 1), r["model"], r["obs"],
                                      "%s  %s UTC" % (r["name"].title(), r["time"].strftime("%d %Hz")),
                                      text=text)
    if skew is not None:
        h, l = skew.ax.get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, fontsize=10)
    fig.suptitle("%s d%02d  WRF (solid) vs IGRA radiosonde (dashed); barbs: WRF inner, sonde outer"
                 % (run_label, domain), fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.subplots_adjust(wspace=0.4)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    return out_path


def plot_pblh_compare(rows, out_path, run_label, domain):
    plt = _plt()
    good = [r for r in rows if r["pblh_obs"] == r["pblh_obs"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    vals = [v for r in good for v in (r["pblh_obs"], r["pblh_model_ri"], r["pblh_model"]) if v == v]
    lim = max(500.0, 1.1 * max(vals)) if vals else 1000.0     # fit the data: 0-4 km would hide the signal
    ax.plot([0, lim], [0, lim], "k-", lw=0.8)
    markers = "osD^vP*X"
    for n, sid in enumerate(sorted({r["sid"] for r in rows})):
        rs = [r for r in good if r["sid"] == sid]
        if not rs:
            continue
        mk = markers[n % len(markers)]
        ob = [r["pblh_obs"] for r in rs]
        ax.scatter(ob, [r["pblh_model_ri"] for r in rs], marker=mk, c="tab:blue", s=40,
                   label="%s WRF bulk-Ri" % sid[-5:])
        ax.scatter(ob, [r["pblh_model"] for r in rs], marker=mk, edgecolors="tab:red", facecolors="none",
                   s=40, label="%s WRF scheme PBLH" % sid[-5:])
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("radiosonde PBL height, bulk Ri=0.25 (m AGL)")
    ax.set_ylabel("WRF PBL height (m AGL)")
    ax.set_title("%s d%02d  PBL height, WRF vs radiosonde (%d soundings)" % (run_label, domain, len(good)),
                 fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def write_stats_csv(rows, out_path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "name", "time_utc", "pblh_obs_m", "pblh_model_bulkri_m", "pblh_model_scheme_m",
                    "T_bias_K", "T_rmse_K", "Td_bias_K", "Td_rmse_K", "wspd_bias_ms", "wspd_rmse_ms", "n_levels"])
        for r in sorted(rows, key=lambda r: (r["sid"], r["time"])):
            s = r["stats"]
            w.writerow([r["sid"], r["name"], r["time"].strftime("%Y-%m-%dT%H:%MZ"),
                        "%.0f" % r["pblh_obs"], "%.0f" % r["pblh_model_ri"], "%.0f" % r["pblh_model"],
                        "%.2f" % s["T"]["bias"], "%.2f" % s["T"]["rmse"], "%.2f" % s["Td"]["bias"],
                        "%.2f" % s["Td"]["rmse"], "%.2f" % s["wspd"]["bias"], "%.2f" % s["wspd"]["rmse"],
                        s["T"]["n"]])
    return out_path


def figures(run_dirs, labels, out_dir, case_info, project_dir=None, xsec_top_km=None,
            xsec_halfwidth_km=None, **_ignored):
    from post.recipe import fig, missing
    run_dir, run_label = run_dirs[0], labels[0]
    pbl_domains = [d for d in case_info["domains"]
                   if (d["bl_pbl_physics"] or 0) == 0 or d["dx_km"] <= 1.0]
    domain = (max(pbl_domains, key=lambda d: d["domain"])["domain"] if pbl_domains
              else case_info["domains"][-1]["domain"])
    frames = list_frame_refs(run_dir, domain=domain)      # [(time, path, timeidx)]
    if len(frames) < 2:
        return [missing("pbl_time_height", "wrfout_d%02d_*" % domain,
                        "%d frame(s) found, need at least 2" % len(frames))]
    frames_by_time = {t: (path, ti) for t, path, ti in frames}
    dom = "d%02d" % domain
    span = "%s..%s" % (frames[0][0], frames[-1][0])

    obs_dir = _obs_dir(run_dir, project_dir)
    stations = _load_stations(obs_dir) if obs_dir else []
    out = []
    try:
        pick = pick_column(run_dir, frames, stations or None)
        if pick is None and stations:
            pick = pick_column(run_dir, frames, None)     # no station in the interior: fall back
        if pick is None:
            return [missing("pbl_time_height", "interior land column",
                            "no land column away from the coast and the lateral boundary zone in %s" % dom)]
        j, i, label, sid = pick
        import netCDF4
        with netCDF4.Dataset(frames[0][1]) as ds:
            lon = float(ds.variables["XLONG"][0, j, i])
        series = column_series(frames, j, i)
        obs_pblh = {}
        if sid:
            from post import sounding
            for st in stations:
                if st[0] == sid:
                    for t, s in st[5].items():
                        col = sounding.obs_column(s, st[4])
                        v = sounding.pblh_bulk_ri(col) if col is not None else float("nan")
                        if v == v:
                            obs_pblh[t] = v
        path = plot_time_height(series, obs_pblh, label, lon,
                                os.path.join(out_dir, "pbl_time_height.png"), run_label, domain)
        out.append(fig("pbl_time_height", path, run_label, domain=dom, valid_time=span,
                       extra={"column": label, "grid_ji": [j, i]}))
        tmax = max(series[1:], key=lambda s: s[1]["pblh_model"])[0].strftime(WRF_FMT)
        path = plot_cross_section(run_dir, domain, tmax, j, i,
                                  os.path.join(out_dir, "pbl_cross_terrain.png"), run_label, label,
                                  halfwidth_km=xsec_halfwidth_km or 500.0, top_km=xsec_top_km)
        out.append(fig("pbl_cross_terrain", path, run_label, domain=dom, valid_time=tmax))
    except (tools.ToolNotFoundError, ImportError) as e:
        return out + [missing("pbl_time_height", "wrf-python", str(e)),
                      missing("pbl_cross_terrain", "wrf-python", str(e))]

    if not stations:
        out.append(missing("pbl_skewt_obs", "IGRA radiosondes",
                           "no <project>/input/obs/igra/ found; run `wtune post fetch igra --project P "
                           "--run-dir R` (login node, needs internet)"))
        return out
    try:
        rows = _station_comparisons(stations, frames_by_time, run_dir)
        if not rows:
            out.append(missing("pbl_skewt_obs", "IGRA radiosondes",
                               "no sounding inside the %s interior matches an hourly wrfout frame" % dom))
            return out
        for sid_ in sorted({r["sid"] for r in rows}):
            rs = [r for r in rows if r["sid"] == sid_]
            path = plot_skewt_station(rs, os.path.join(out_dir, "pbl_skewt_%s.png" % sid_), run_label, domain)
            out.append(fig("pbl_skewt_obs", path, run_label, domain=dom,
                           valid_time=",".join(r["time"].strftime("%d%H") for r in rs), extra={"station": sid_}))
        csv_path = write_stats_csv(rows, os.path.join(out_dir, "pbl_obs_stats.csv"))
        path = plot_pblh_compare(rows, os.path.join(out_dir, "pbl_height_compare.png"), run_label, domain)
        out.append(fig("pbl_height_compare", path, run_label, domain=dom, valid_time=span,
                       extra={"stats_csv": csv_path}))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("pbl_skewt_obs", "metpy", str(e)))
    return out

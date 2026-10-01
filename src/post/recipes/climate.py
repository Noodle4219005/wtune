"""post/recipes/climate.py -- the "climate" figure set (decision 0057 section 2) for long runs
(type detected at run length >= 30 days).

  climate_annual_cycle   monthly means of T2 (land boxes) / SST (ocean box) and precipitation over
                         named regions, model vs ERA5 (T2, SST) and GPCP (precipitation)
  climate_bias_t2        seasonal (DJF/MAM/JJA/SON) T2 maps, model minus ERA5
  climate_bias_precip    seasonal precipitation maps, model minus GPCP
  climate_box_monthly.csv  the box time series behind the annual-cycle figure (processed data)

Pipeline. CDO does every time reduction: `cdo select` concatenates the run's wrfout frames
keeping only T2, SST, RAINC, RAINNC; `cdo monmean` gives monthly T2/SST; monthly precipitation
comes from `cdo monmax` of the running accumulation RAINC+RAINNC after `cdo shifttime` by one
output interval (so the value valid at 00 UTC on the 1st closes the PREVIOUS month) and a
first difference between consecutive months (the accumulation starts from 0 at the first frame).
The remaining steps are plain numpy on those small monthly fields: day-weighted seasonal means of
whole months, bilinear interpolation of the (regular-grid) references onto the model's
curvilinear grid -- CDO cannot remap onto a WRF grid without a hand-written grid description --
and cos(lat)-weighted box means. The lateral boundary zone (spec_bdy_width cells) is blanked in
every map and box statistic. (Seasonal means are taken from the monthly means with day weights
rather than with `cdo yseasmean`, which would silently average partial seasons.)

Spin-up. The first SPINUP_DAYS (default 30) after the first frame are discarded: only calendar
months that START at or after that instant and are complete (their closing 00 UTC frame exists)
are analysed. A December..December run therefore yields January..December of the analysis year.
Seasons need all three months (DJF = Jan + Feb + Dec of the analysis year).

References, produced by login-node fetch modules (compute nodes have no internet):
  ERA5 t2m/sst  post/fetch/era5_monthly.py -> <project>/input/era5/monthly/
  GPCP precip   post/fetch/gpcp_monthly.py -> <project>/input/obs/gpcp_monthly/
Precipitation reference is GPCP, not ERA5 tp: ERA5 tp is a forecast accumulation (model wet
biases over the tropical ocean and orography; a monthly mean needs every hourly chunk of the
ARCO store), GPCP is observation-based. Its 2.5 degree grid is coarse against the model; the
figures say so. SST only has a meaningful cycle when the run used sst_update=1: a constant SST
field is reported instead of drawn as a flat "annual cycle".
"""
from __future__ import annotations

import calendar
import csv
import glob
import os
from datetime import datetime, timedelta

from post import tools
from post.extract import list_frames

SPINUP_DAYS = 30.0
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}

# name, (lat0, lat1, lon0, lon1), surface ("land": LANDMASK==1 and T2; "sea": LANDMASK==0 and SST)
BOXES = [
    ("North China", (35.0, 45.0, 110.0, 120.0), "land"),
    ("South China", (20.0, 28.0, 105.0, 120.0), "land"),
    ("Taiwan", (21.5, 25.5, 119.5, 122.5), "land"),
    ("W North Pacific", (15.0, 30.0, 130.0, 150.0), "sea"),
]


# --------------------------------------------------------------------------------------
# pure helpers (unit-tested on synthetic data)
# --------------------------------------------------------------------------------------

def _next_month(y, m):
    return (y + 1, 1) if m == 12 else (y, m + 1)


def analysis_months(t_first, t_last, spinup_days=SPINUP_DAYS):
    """[(year, month)] of complete calendar months usable for statistics: the month starts at or
    after t_first + spinup_days and its closing frame (00 UTC on the next month's 1st, needed by
    the precipitation accumulation) exists, i.e. t_last >= start of the next month."""
    cutoff = t_first + timedelta(days=spinup_days)
    out = []
    y, m = t_first.year, t_first.month
    while datetime(y, m, 1) <= t_last:
        ny, nm = _next_month(y, m)
        if datetime(y, m, 1) >= cutoff and datetime(ny, nm, 1) <= t_last:
            out.append((y, m))
        y, m = ny, nm
    return out


def complete_seasons(months):
    """{name: [(y, m), ...]} for the seasons whose three calendar months each occur exactly once
    in `months` (12 months of one analysis year give all four; DJF = Jan + Feb + Dec of that
    year). A month occurring twice (a run longer than a year) makes the season ambiguous and it
    is left out rather than mixed."""
    out = {}
    for name, ms in SEASONS.items():
        sel = []
        for m in ms:
            cand = [k for k in months if k[1] == m]
            if len(cand) != 1:
                sel = None
                break
            sel.append(cand[0])
        if sel:
            out[name] = sorted(sel)
    return out


def monthly_totals(acc_at_month_end, first_acc=0.0):
    """Monthly totals from the accumulation valid at each month's end (ordered list of arrays):
    first differences, the first month starting from `first_acc` (0: WRF's RAINC/RAINNC are 0 at
    the first frame). A negative difference (accumulation reset by a restart or bucket) becomes
    NaN instead of a negative rain total."""
    import numpy as np
    out, prev = [], np.asarray(first_acc, dtype="f8")
    for a in acc_at_month_end:
        a = np.asarray(a, dtype="f8")
        d = a - prev
        out.append(np.where(d < -1e-6, np.nan, d))
        prev = a
    return out


def season_mean(monthly, keys):
    """Day-weighted mean of the monthly fields {(y, m): array} for the months in `keys`."""
    import numpy as np
    w = np.array([calendar.monthrange(y, m)[1] for y, m in keys], dtype="f8")
    stack = np.array([monthly[k] for k in keys], dtype="f8")
    return np.tensordot(w / w.sum(), stack, axes=1)


def box_mean(field, lat, lon, box, mask):
    """cos(lat)-weighted mean of `field` over the lat/lon box restricted to boolean `mask`
    (True = use); NaN cells are ignored. Returns (mean, n_cells); mean NaN when nothing valid."""
    import numpy as np
    lat0, lat1, lon0, lon1 = box
    sel = (lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1) & mask & np.isfinite(field)
    if not sel.any():
        return float("nan"), 0
    w = np.cos(np.radians(lat[sel]))
    return float((field[sel] * w).sum() / w.sum()), int(sel.sum())


def interp_to_grid(ref_lat, ref_lon, ref_field, lat2d, lon2d):
    """Bilinear interpolation of a regular lat/lon field onto model points (lat2d, lon2d).
    Longitudes are put in the reference's convention (0..360 or -180..180); NaN in the reference
    propagates (e.g. ERA5 SST over land); points outside the reference grid are NaN. scipy's
    RegularGridInterpolator: there is no CDO route onto a curvilinear WRF grid."""
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    ref_lat = np.asarray(ref_lat, dtype="f8")
    ref_lon = np.asarray(ref_lon, dtype="f8")
    fld = np.asarray(ref_field, dtype="f8")
    if ref_lat[0] > ref_lat[-1]:
        ref_lat, fld = ref_lat[::-1], fld[::-1]
    lon = np.mod(lon2d, 360.0) if ref_lon.max() > 180.0 else np.where(lon2d > 180.0, lon2d - 360.0, lon2d)
    f = RegularGridInterpolator((ref_lat, ref_lon), fld, bounds_error=False, fill_value=np.nan)
    return f(np.stack([lat2d.ravel(), lon.ravel()], axis=-1)).reshape(lat2d.shape)


def blank_boundary(field, width):
    """Copy of `field` (ny, nx) with the outer `width` cells set to NaN."""
    import numpy as np
    out = np.array(field, dtype="f8")
    if width > 0:
        out[:width, :] = np.nan
        out[-width:, :] = np.nan
        out[:, :width] = np.nan
        out[:, -width:] = np.nan
    return out


# --------------------------------------------------------------------------------------
# CDO stage
# --------------------------------------------------------------------------------------

def _cdo():
    return tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or envs/wtune/pixi.toml provides it.")


def _read_cdo_monthly(path, varname, ny, nx):
    """{(y, m): (ny, nx) array} from a cdo monthly-time-axis file."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as ds:
        # cdo names the axis after what it found in the WRF file: XTIME for real wrfout (the
        # variable carrying standard_name=time), "time" for files it wrote from other sources
        tv = next((v for v in ds.variables.values() if getattr(v, "standard_name", "") == "time"),
                  ds.variables.get("time"))
        times = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
        arr = np.ma.filled(ds.variables[varname][:], np.nan).astype("f8").reshape(len(times), ny, nx)
    return {(t.year, t.month): arr[i] for i, t in enumerate(times)}


def model_monthly(frames, work_dir, ny, nx, step_s):
    """CDO reductions of the run's frames: {"T2": {(y,m): arr}, "SST": ..., "PR_acc": ...} where
    PR_acc is the RAINC+RAINNC accumulation valid at the end of each month (after the
    one-interval shift, see the module docstring). The large concatenated file is deleted."""
    cdo = _cdo()
    os.makedirs(work_dir, exist_ok=True)
    merged = os.path.join(work_dir, "merged.nc")
    tools.run([cdo, "-s", "-O", "select,name=T2,SST,RAINC,RAINNC"] + [p for _, p in frames] + [merged])
    mmean = os.path.join(work_dir, "monmean.nc")
    tools.run([cdo, "-s", "-O", "monmean", "-selname,T2,SST", merged, mmean])
    rain = os.path.join(work_dir, "rain_monmax.nc")
    tools.run([cdo, "-s", "-O", "monmax", "-shifttime,-%dseconds" % step_s,
               "-expr,PR=RAINC+RAINNC", merged, rain])
    out = {"T2": _read_cdo_monthly(mmean, "T2", ny, nx),
           "SST": _read_cdo_monthly(mmean, "SST", ny, nx),
           "PR_acc": _read_cdo_monthly(rain, "PR", ny, nx)}
    os.remove(merged)
    return out


# --------------------------------------------------------------------------------------
# reference data
# --------------------------------------------------------------------------------------

def _project(run_dir, project_dir):
    if project_dir:
        return project_dir
    try:
        import wtune_project
        return wtune_project.find_project(start_dir=run_dir) or wtune_project.find_project()
    except Exception:                        # noqa: BLE001 -- no project: no references
        return None


def read_era5_monthly(project, var, months):
    """({(y, m): 2-D array K}, lat, lon) from <project>/input/era5/monthly, or (None, None, None)
    when any requested month is missing."""
    import netCDF4
    import numpy as np
    out, lat, lon = {}, None, None
    for (y, m) in months:
        f = os.path.join(project, "input", "era5", "monthly", "era5_%s_%04d%02d.nc" % (var, y, m))
        if not os.path.isfile(f):
            return None, None, None
        with netCDF4.Dataset(f) as ds:
            lat = np.asarray(ds.variables["lat"][:], dtype="f8")
            lon = np.asarray(ds.variables["lon"][:], dtype="f8")
            out[(y, m)] = np.ma.filled(ds.variables[var][:], np.nan).astype("f8")
    return out, lat, lon


def read_gpcp_monthly(project, months):
    """({(y, m): 2-D array mm/day}, lat, lon) from <project>/input/obs/gpcp_monthly, or Nones."""
    import netCDF4
    import numpy as np
    out, lat, lon = {}, None, None
    for (y, m) in months:
        cands = sorted(glob.glob(os.path.join(project, "input", "obs", "gpcp_monthly",
                                              "gpcp_v02r03_monthly_d%04d%02d_c*.nc" % (y, m))))
        if not cands:
            return None, None, None
        with netCDF4.Dataset(cands[-1]) as ds:
            lat = np.asarray(ds.variables["latitude" if "latitude" in ds.variables else "lat"][:], dtype="f8")
            lon = np.asarray(ds.variables["longitude" if "longitude" in ds.variables else "lon"][:], dtype="f8")
            out[(y, m)] = np.ma.filled(ds.variables["precip"][0], np.nan).astype("f8")
    return out, lat, lon


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_annual_cycle(months, series, out_path, run_label, notes):
    """series = {"T": {...}, "P": {...}}, each {"model": {box: arr|None}, "ref": {box: arr}|None,
    "ref_label", "why"}; one row per box: left temperature (T2 over land boxes, SST over the
    ocean box), right precipitation. A box without model data shows its reason in the panel."""
    import numpy as np
    plt = _plt()
    labels = ["%s %02d" % (calendar.month_abbr[m], y % 100) for y, m in months]
    x = np.arange(len(months))
    fig, axes = plt.subplots(len(BOXES), 2, figsize=(12, 3.1 * len(BOXES)), sharex=True)
    for r, (name, box, kind) in enumerate(BOXES):
        for c, key in enumerate(("T", "P")):
            ax = axes[r, c]
            s = series[key]
            mod = s["model"].get(name)
            if mod is None or not np.isfinite(mod).any():
                ax.text(0.5, 0.5, s["why"], ha="center", va="center", transform=ax.transAxes, fontsize=10)
            else:
                # temperatures in degC (decision 0064); the CSV keeps K
                off = 273.15 if key == "T" else 0.0
                ax.plot(x, np.asarray(mod) - off, "r-o", ms=4, lw=1.8, label="WRF")
                ref = (s["ref"] or {}).get(name)
                if ref is not None:
                    ax.plot(x, np.asarray(ref) - off, "k--s", ms=4, lw=1.4, label=s["ref_label"])
                if r == 0:
                    ax.legend(fontsize=8, loc="best")
            what = "precipitation (mm/day)" if key == "P" else ("SST (°C)" if kind == "sea" else "T2 (°C)")
            ax.set_ylabel(what, fontsize=9)
            ax.set_title("%s  %g-%gN %g-%gE  %s" % (name, box[0], box[1], box[2], box[3],
                                                    "land" if kind == "land" else "ocean"), fontsize=9)
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle("%s  monthly means, model vs reference%s" % (run_label, ("\n" + "; ".join(notes)) if notes else ""),
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_seasonal_bias(diffs, lat, lon, crs, landmask, out_path, run_label, var_label, ref_label, unit,
                       note=""):
    """2x2 maps of model - reference for the seasons in `diffs` ({name: 2-D array, NaN blanked});
    one symmetric level set for all panels, white zero bin, coastlines through post.plotting's
    Natural-Earth-or-LANDMASK logic. Seasons absent from `diffs` get an empty labelled panel."""
    import numpy as np
    import cartopy.crs as ccrs
    from post.recipes.synoptic import symmetric_levels
    from post.plotting import _set_cartopy_data_dir
    plt = _plt()
    finite = np.concatenate([d[np.isfinite(d)] for d in diffs.values()]) if diffs else np.array([])
    vmax = max(float(np.percentile(np.abs(finite), 98)), 1e-3) if finite.size else 1.0
    levels = symmetric_levels(vmax)
    # wet = green/blue, dry = brown for precipitation; warm = red for temperature (decision 0064)
    cmap = plt.get_cmap("BrBG" if "precip" in var_label.lower() else "RdBu_r")
    _set_cartopy_data_dir()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), subplot_kw={"projection": crs})
    cf = None
    for ax, name in zip(axes.ravel(), SEASONS):
        ax.set_facecolor(cmap(0.5))
        if name not in diffs:
            ax.set_title(name, fontsize=11)
            ax.text(0.5, 0.5, "season incomplete in run", transform=ax.transAxes, ha="center")
            continue
        d = diffs[name]
        cf = ax.contourf(lon, lat, np.ma.masked_invalid(d), levels=levels, cmap=cmap, extend="both",
                         transform=ccrs.PlateCarree())
        from post.basemap import add_base_map
        drawn = add_base_map(ax, ccrs, lats=lat, lons=lon, terrain=False, nest_rings=[],
                             caption=False)
        if not drawn["layers"] and landmask is not None:
            ax.contour(lon, lat, landmask, levels=[0.5], colors="k", linewidths=0.6,
                       transform=ccrs.PlateCarree())
        ok = np.isfinite(d)
        ax.set_title("%s   bias %+.2f  rmse %.2f %s" % (name, float(d[ok].mean()),
                                                        float(np.sqrt((d[ok] ** 2).mean())), unit), fontsize=10)
    if cf is not None:
        fig.colorbar(cf, ax=axes, ticks=levels, shrink=0.8, pad=0.02,
                     label="%s, model - %s (%s)" % (var_label, ref_label, unit))
    fig.suptitle("%s  seasonal %s bias vs %s%s" % (run_label, var_label, ref_label, note), fontsize=12)
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_box_csv(path, months, rows):
    """rows = [(box, quantity, source, array over months)] -> long CSV (processed data)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["box", "quantity", "source", "year_month", "value"])
        for box, q, src, arr in rows:
            for (y, m), v in zip(months, arr):
                w.writerow([box, q, src, "%04d-%02d" % (y, m), "" if v != v else "%.4f" % v])
    return path


def figures(run_dirs, labels, out_dir, case_info, project_dir=None, spinup_days=None, **_ignored):
    from post.recipe import fig, missing
    run_dir, run_label = run_dirs[0], labels[0]
    domain = case_info["domains"][0]["domain"]
    dom = "d%02d" % domain
    spinup_days = SPINUP_DAYS if spinup_days is None else float(spinup_days)
    ids = ("climate_annual_cycle", "climate_bias_t2", "climate_bias_precip")
    frames = list_frames(run_dir, domain=domain)
    if len(frames) < 2:
        return [missing(i, "wrfout_%s_* time series" % dom, "%d frame(s) found" % len(frames)) for i in ids]
    from post.caseinfo import WRF_TIME_FMT, spec_bdy_width
    times = [datetime.strptime(t, WRF_TIME_FMT) for t, _ in frames]
    step_s = int(round((times[1] - times[0]).total_seconds()))
    months = analysis_months(times[0], times[-1], spinup_days)
    seasons = complete_seasons(months)
    span = "%.1f d run, %d complete month(s) after the %g d spin-up" % (
        (times[-1] - times[0]).total_seconds() / 86400.0, len(months), spinup_days)
    if len(months) < 3:
        return [missing(i, "wrfout_%s_* time series" % dom,
                        "%s; need at least 3 (seasonal) / 12 (annual cycle)" % span) for i in ids]

    try:
        import netCDF4
        import numpy as np
        wrf_mod = __import__("wrf")
        with netCDF4.Dataset(frames[0][1]) as ds:
            lat = np.asarray(ds.variables["XLAT"][0], dtype="f8")
            lon = np.asarray(ds.variables["XLONG"][0], dtype="f8")
            land = np.asarray(ds.variables["LANDMASK"][0]) == 1
            crs = wrf_mod.get_cartopy(var=wrf_mod.getvar(ds, "T2", timeidx=0))
        ny, nx = lat.shape
        width = spec_bdy_width(run_dir)
        interior = np.isfinite(blank_boundary(np.ones((ny, nx)), width))
        mod = model_monthly(frames, os.path.join(out_dir, "_work"), ny, nx, step_s)
    except (tools.ToolNotFoundError, tools.ToolError, ImportError) as e:
        return [missing(i, "cdo / wrf-python", str(e)) for i in ids]

    order = sorted(mod["PR_acc"])
    totals = dict(zip(order, monthly_totals([mod["PR_acc"][k] for k in order])))
    t2 = {k: mod["T2"][k] for k in months if k in mod["T2"]}
    sst = {k: mod["SST"][k] for k in months if k in mod["SST"]}
    pr = {k: totals[k] / calendar.monthrange(*k)[1] for k in months if k in totals}      # mm/day
    if not (len(t2) == len(sst) == len(pr) == len(months)):
        return [missing(i, "CDO monthly output", "months missing from the CDO monthly files: %s"
                        % sorted(set(months) - set(t2) - set(pr))[:4]) for i in ids]

    hint = "run `wtune post fetch %s --project P --run-dir R` on a login node"
    project = _project(run_dir, project_dir)
    era_t2 = era_sst = gp = None
    if project:
        for var in ("t2m", "sst"):
            raw, r_lat, r_lon = read_era5_monthly(project, var, months)
            if raw:
                val = {k: interp_to_grid(r_lat, r_lon, v, lat, lon) for k, v in raw.items()}
                if var == "t2m":
                    era_t2 = val
                else:
                    era_sst = val
        raw, r_lat, r_lon = read_gpcp_monthly(project, months)
        if raw:
            gp = {k: interp_to_grid(r_lat, r_lon, v, lat, lon) for k, v in raw.items()}

    out = []
    if len(months) < 12:
        out.append(missing("climate_annual_cycle", "wrfout_%s_* time series" % dom,
                           "%s; the annual cycle needs 12" % span))
    else:
        def per_box(temp_land, temp_sea, precip):
            """{box: series} using each box's surface mask; a None source gives None."""
            res = {}
            for name, box, kind in BOXES:
                src = temp_land if kind == "land" else temp_sea
                if precip is not None:
                    src = precip
                if src is None:
                    res[name] = None
                    continue
                mask = (land if kind == "land" else ~land) & interior
                res[name] = np.array([box_mean(src[k], lat, lon, box, mask)[0] for k in months])
            return res

        first = np.array(sst[months[0]])
        sst_flat = max(float(np.nanmax(np.abs(sst[k] - first))) for k in months) < 1e-3
        notes = []
        mod_t = per_box(t2, None if sst_flat else sst, None)
        ref_t = per_box(era_t2, era_sst, None) if (era_t2 is not None and era_sst is not None) else None
        mod_p = per_box(None, None, pr)
        ref_p = per_box(None, None, gp) if gp is not None else None
        if ref_t is None:
            notes.append("ERA5 monthly missing: " + hint % "era5_monthly")
        if ref_p is None:
            notes.append("GPCP missing: " + hint % "gpcp_monthly")
        if sst_flat:
            notes.append("SST is constant in wrfout (sst_update=0): no ocean-box SST cycle")
        series = {"T": {"model": mod_t, "ref": ref_t, "ref_label": "ERA5 (T2 land / SST ocean)",
                        "why": "SST constant in wrfout (sst_update=0)"},
                  "P": {"model": mod_p, "ref": ref_p, "ref_label": "GPCP v2.3, 2.5 deg", "why": "no data"}}
        path = plot_annual_cycle(months, series, os.path.join(out_dir, "climate_annual_cycle.png"),
                                 run_label, notes)
        rows = []
        for name in mod_p:
            for q, m_arr, r_arr, r_src in (("T2_or_SST_K", mod_t.get(name), (ref_t or {}).get(name), "ERA5"),
                                           ("precip_mm_per_day", mod_p.get(name), (ref_p or {}).get(name), "GPCP")):
                if m_arr is not None:
                    rows.append((name, q, "WRF", m_arr))
                if r_arr is not None:
                    rows.append((name, q, r_src, r_arr))
        csv_path = write_box_csv(os.path.join(out_dir, "climate_box_monthly.csv"), months, rows)
        out.append(fig("climate_annual_cycle", path, run_label, domain=dom,
                       valid_time="%04d-%02d..%04d-%02d" % (months[0] + months[-1]),
                       extra={"csv": csv_path, "spinup_days": spinup_days, "notes": notes}))

    lm = land.astype(float)

    def bias(model_field, ref_field):
        return {n: blank_boundary(season_mean(model_field, ks) - season_mean(ref_field, ks), width)
                for n, ks in seasons.items()}

    if not seasons:
        for i in ids[1:]:
            out.append(missing(i, "complete season", "%s: no season has all three months" % span))
        return out
    tag = " (spin-up %g d excluded)" % spinup_days
    if era_t2 is None:
        out.append(missing("climate_bias_t2", "ERA5 monthly t2m", hint % "era5_monthly"))
    else:
        path = plot_seasonal_bias(bias(t2, era_t2), lat, lon, crs, lm, os.path.join(out_dir, "climate_bias_t2.png"),
                                  run_label, "T2", "ERA5", "°C", tag)
        out.append(fig("climate_bias_t2", path, run_label, domain=dom, valid_time=",".join(sorted(seasons))))
    if gp is None:
        out.append(missing("climate_bias_precip", "GPCP monthly precipitation", hint % "gpcp_monthly"))
    else:
        path = plot_seasonal_bias(bias(pr, gp), lat, lon, crs, lm, os.path.join(out_dir, "climate_bias_precip.png"),
                                  run_label, "precipitation", "GPCP (2.5 deg)", "mm/day", tag)
        out.append(fig("climate_bias_precip", path, run_label, domain=dom, valid_time=",".join(sorted(seasons))))
    return out

"""post/changemap.py -- consecutive and vs-t0 change maps, and period accumulations (decision
0058 section 2 update, 2026-09-29): differences are computed with `cdo sub`, never re-derived
in Python. For each field this module extracts the two contributing times into ONE NetCDF file
via post/extract.py (wrf-python's own field derivation -- the same engine `wtune post extract`
uses), then runs `cdo sub -seltimestep,2 FILE -seltimestep,1 FILE OUT` to difference them.
A period accumulation (e.g. 6 h rain) is the same operation: WRF's RAINC+RAINNC is already a
running total, so a period total is exactly (rain_acc(t1) - rain_acc(t0)) -- no new
accumulation logic, just the same cdo-sub call on the "rain_acc" field.

Output naming: <field>-chg<Nh>_d<domain>_fNNN_<validZ>.png (decision 0058, "命名在圖名後加間隔").
"""
from __future__ import annotations

import os
from datetime import datetime

from post import tools
from post.extract import UNITS
from post.timeseries import WRF_TIME_FMT, forecast_dir_name

# human-readable field names for the change-map colour bar label (2026-09-29 lead review
# item 3: "add units to change-map colour bars, e.g. 'SLP change (hPa)'").
_FIELD_DISPLAY = dict(slp="SLP", t2="T2", pw="PW", mcape="MCAPE", rain_acc="rainfall")

# fields this module knows how to difference; each must be a post/extract.py SURFACE_FIELDS
# entry (2-D, one value per grid cell per frame).
DEFAULT_CHANGE_FIELDS = ("slp", "t2", "pw", "mcape")


def _hours_between(t_prev, t_curr):
    dt = (datetime.strptime(t_curr, WRF_TIME_FMT) - datetime.strptime(t_prev, WRF_TIME_FMT))
    return int(round(dt.total_seconds() / 3600.0))


def _domain_crs(run_dir, domain):
    """The domain's own WRF map projection (wrf.get_cartopy on the first frame's global
    attributes) and that frame's path (for the LANDMASK fallback coastline); (None, None) when
    unavailable, in which case the map falls back to PlateCarree."""
    try:
        import netCDF4
        from post.extract import list_frames
        wrf_mod = __import__("wrf")
        frames = list_frames(run_dir, domain=domain)
        if not frames:
            return None, None
        with netCDF4.Dataset(frames[0][1]) as ds:
            return wrf_mod.get_cartopy(wrfin=ds), frames[0][1]
    except Exception:
        return None, None


def _change_scale(field, diffs):
    """(levels, cmap, extend, colour-bar label prefix) fixed for every map of one
    (field, mode): symmetric about 0 with NO boundary at 0 (the bin straddling 0 is one white
    band in RdBu_r) from the pooled 98th percentile of |diff|. rain_acc 'change' is never
    negative (it is the rainfall in the interval), so it gets a sequential 0-based map."""
    import numpy as np

    vals = np.concatenate([np.asarray(d, dtype="f8").ravel() for d in diffs])
    vals = vals[np.isfinite(vals)]
    if field == "rain_acc":
        # interval rainfall on the WPC isohyet thresholds in mm (decision 0064)
        from post.obs_figs import PRECIP_LEVELS, precip_cmap
        hi = float(np.percentile(vals, 99.5)) if vals.size else 1.0
        upto = [lv for lv in PRECIP_LEVELS if lv <= hi]
        levels = PRECIP_LEVELS[:max(2, len(upto) + 1)]
        return levels, precip_cmap(), "max"
    from post.recipes.synoptic import symmetric_levels
    vmax = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
    return symmetric_levels(max(vmax, 1e-6)), "RdBu_r", "both"


def _compute_diff(run_dir, field, domain, out_dir, t_prev, t_curr):
    """cdo sub of the field at t_curr minus t_prev (one extract file, seltimestep 2 - 1).
    Returns (lat, lon, diff) ndarrays; raises tools.ToolError/OSError/KeyError on failure."""
    from post.extract import extract
    cdo_bin = tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or "
                                 "envs/wtune/pixi.toml provides it.")
    nc_path = os.path.join(out_dir, "_chg_%s_%s_%s.nc"
                           % (field, t_prev.replace(":", ""), t_curr.replace(":", "")))
    diff_nc = nc_path.replace(".nc", "_diff.nc")
    try:
        extract(run_dir, [field], nc_path, domain=domain, times=[t_prev, t_curr],
               box=(-90.0, 90.0, -180.0, 180.0))
        tools.run([cdo_bin, "sub", "-seltimestep,2", nc_path, "-seltimestep,1", nc_path, diff_nc])
        import netCDF4
        import numpy as np
        with netCDF4.Dataset(diff_nc) as ds:
            return (np.asarray(ds.variables["XLAT"][:]), np.asarray(ds.variables["XLONG"][:]),
                    np.asarray(ds.variables[field][0]))
    finally:
        for p in (nc_path, diff_nc):
            try:
                os.remove(p)
            except OSError:
                pass


def _draw_change(run_dir, field, domain, out_path, t_prev, t_curr, interval_h, lat, lon, diff,
                 scale):
    from post.recipes.mapfig import save_map
    levels, cmap, extend = scale
    display = _FIELD_DISPLAY.get(field, field)
    unit = UNITS.get(field, "")
    if field == "rain_acc":
        fill_label = "%d-h rainfall (%s)" % (interval_h, unit)
        title = "%d-h rainfall  d%02d\n%s -> %s" % (interval_h, domain, t_prev, t_curr)
    else:
        fill_label = "%s change (%s)" % (display, unit) if unit else "%s change" % display
        title = "%s  %d-h change  d%02d\n%s -> %s" % (display, interval_h, domain, t_prev, t_curr)
    crs, sample = _domain_crs(run_dir, domain)
    ds = None
    try:
        if sample:
            import netCDF4
            ds = netCDF4.Dataset(sample)
        # same map style as every other figure: domain projection, Natural Earth coastlines
        # (LANDMASK contour fallback), labelled lat/lon gridlines
        save_map(out_path, lat, lon, title, fill_field=diff, fill_label=fill_label,
                 fill_cmap=cmap, fill_levels=levels, fill_extend=extend, ds=ds, crs=crs,
                 basemap="cartopy", gridline_labels=True)
    finally:
        if ds is not None:
            ds.close()


def change_map(run_dir, run_label, field, domain, out_dir, t_prev, t_curr, t0, fig_id=None,
               scale=None, precomputed=None):
    """One change-map figure: field value at t_curr minus at t_prev. `t0` (the series' first
    valid time) is only used to build the fNNN forecast tag in the output filename. `scale`:
    a fixed (levels, cmap, extend) shared with sibling maps (figures_for_change_maps passes
    one per field and mode); default: derived from this map alone. `precomputed`: the
    (lat, lon, diff) already computed. Returns a post/recipe.py fig()/missing() record."""
    from post.recipe import fig, missing

    interval_h = _hours_between(t_prev, t_curr)
    fig_id = fig_id or ("chg_%s" % field)
    tag = forecast_dir_name(datetime.strptime(t0, WRF_TIME_FMT), t_curr)
    out_name = "%s-chg%dh_d%02d_%s.png" % (field, interval_h, domain, tag)
    out_path = os.path.join(out_dir, out_name)

    try:
        lat, lon, diff = precomputed or _compute_diff(run_dir, field, domain, out_dir,
                                                      t_prev, t_curr)
        scale = scale or _change_scale(field, [diff])
        _draw_change(run_dir, field, domain, out_path, t_prev, t_curr, interval_h, lat, lon,
                     diff, scale)
        note = ("%d h rainfall (rain_acc(t) - rain_acc(t-%dh))" % (interval_h, interval_h)
                if field == "rain_acc" else "%d h change vs %s" % (interval_h, t_prev))
        return fig(fig_id, out_path, run_label, domain="d%02d" % domain, valid_time=t_curr,
                  extra={"forecast_tag": tag, "note": note + ", computed with cdo sub"})
    except tools.ToolNotFoundError as e:
        return missing(fig_id, "cdo", str(e), run=run_label)
    except (tools.ToolError, ImportError, KeyError, OSError) as e:
        return missing(fig_id, "cdo/wrf-python", str(e), run=run_label)


def figures_for_change_maps(run_dir, run_label, domain, out_dir, selected_times,
                            fields=DEFAULT_CHANGE_FIELDS):
    """selected_times: an ordered list of valid times (post/timeseries.select_times's own
    output). For each consecutive pair: one change map per field, plus one vs-t0 change map
    (skipped for the first pair, identical to the consecutive one) once 3+ times exist. Also
    one period-accumulation change map for rain_acc per consecutive pair (decision 0058:
    "時段量，例如 6 h 雨量"). Every diff is computed first (cdo sub), then ONE colour scale is
    fixed per (field, mode) -- consecutive vs since-t0 -- and shared by all its maps, so the
    frames are comparable. Returns post/recipe.py fig()/missing() records."""
    from post.recipe import missing
    if len(selected_times) < 2:
        return []
    t0 = selected_times[0]
    jobs = []   # (field, fig_id, mode, t_prev, t_curr)
    for i in range(1, len(selected_times)):
        t_prev, t_curr = selected_times[i - 1], selected_times[i]
        for field in fields:
            jobs.append((field, None, "consec", t_prev, t_curr))
        jobs.append(("rain_acc", "chg_rain_acc", "consec", t_prev, t_curr))
        if i >= 2:
            for field in fields:
                jobs.append((field, "chg_%s_vs_t0" % field, "vs_t0", t0, t_curr))

    computed, errors = {}, {}
    for j, (field, fig_id, mode, a, b) in enumerate(jobs):
        try:
            computed[j] = _compute_diff(run_dir, field, domain, out_dir, a, b)
        except tools.ToolNotFoundError as e:
            errors[j] = missing(fig_id or "chg_%s" % field, "cdo", str(e), run=run_label)
        except (tools.ToolError, ImportError, KeyError, OSError) as e:
            errors[j] = missing(fig_id or "chg_%s" % field, "cdo/wrf-python", str(e),
                                run=run_label)

    scales = {}
    for key in {(f, m) for (f, _, m, _, _) in jobs}:
        diffs = [computed[j][2] for j, job in enumerate(jobs)
                 if (job[0], job[2]) == key and j in computed]
        if diffs:
            scales[key] = _change_scale(key[0], diffs)

    out = []
    for j, (field, fig_id, mode, a, b) in enumerate(jobs):
        if j in errors:
            out.append(errors[j])
            continue
        out.append(change_map(run_dir, run_label, field, domain, out_dir, a, b, t0,
                              fig_id=fig_id, scale=scales[(field, mode)],
                              precomputed=computed[j]))
    return out

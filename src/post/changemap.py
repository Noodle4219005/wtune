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
_FIELD_DISPLAY = dict(slp="SLP", t2="T2", pw="PW", mcape="MCAPE", rain_acc="rain accumulation")

# fields this module knows how to difference; each must be a post/extract.py SURFACE_FIELDS
# entry (2-D, one value per grid cell per frame).
DEFAULT_CHANGE_FIELDS = ("slp", "t2", "pw", "mcape")


def _hours_between(t_prev, t_curr):
    dt = (datetime.strptime(t_curr, WRF_TIME_FMT) - datetime.strptime(t_prev, WRF_TIME_FMT))
    return int(round(dt.total_seconds() / 3600.0))


def change_map(run_dir, run_label, field, domain, out_dir, t_prev, t_curr, t0, fig_id=None):
    """One change-map figure: field value at t_curr minus at t_prev. `t0` (the series' first
    valid time) is only used to build the fNNN forecast tag in the output filename. Returns a
    post/recipe.py fig()/missing() record."""
    from post.recipe import fig, missing
    from post.extract import extract

    interval_h = _hours_between(t_prev, t_curr)
    fig_id = fig_id or ("chg_%s" % field)
    tag = forecast_dir_name(datetime.strptime(t0, WRF_TIME_FMT), t_curr)
    out_name = "%s-chg%dh_d%02d_%s.png" % (field, interval_h, domain, tag)
    out_path = os.path.join(out_dir, out_name)

    try:
        cdo_bin = tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or "
                                     "envs/wtune/pixi.toml provides it.")
    except tools.ToolNotFoundError as e:
        return missing(fig_id, "cdo", str(e), run=run_label)

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
            lat = np.asarray(ds.variables["XLAT"][:])
            lon = np.asarray(ds.variables["XLONG"][:])
            diff = np.asarray(ds.variables[field][0])

        display = _FIELD_DISPLAY.get(field, field)
        unit = UNITS.get(field, "")
        fill_label = "%s change (%s)" % (display, unit) if unit else "%s change" % display

        from post.recipes.mapfig import save_map
        save_map(out_path, lat, lon,
                 "%s  %d h change  d%02d  %s -> %s" % (field, interval_h, domain, t_prev, t_curr),
                 fill_field=diff, fill_label=fill_label, fill_cmap="RdBu_r",
                 basemap="landmask")
        return fig(fig_id, out_path, run_label, domain="d%02d" % domain, valid_time=t_curr,
                  extra={"forecast_tag": tag,
                        "note": "%d h change vs %s, computed with cdo sub" % (interval_h, t_prev)})
    except (tools.ToolError, tools.ToolNotFoundError, ImportError, KeyError, OSError) as e:
        return missing(fig_id, "cdo/wrf-python", str(e), run=run_label)
    finally:
        for p in (nc_path, diff_nc):
            try:
                os.remove(p)
            except OSError:
                pass


def figures_for_change_maps(run_dir, run_label, domain, out_dir, selected_times,
                            fields=DEFAULT_CHANGE_FIELDS):
    """selected_times: an ordered list of valid times (post/timeseries.select_times's own
    output). For each consecutive pair: one change map per field, plus one vs-t0 change map
    (skipped for the first pair, identical to the consecutive one) once 3+ times exist. Also
    one period-accumulation change map for rain_acc per consecutive pair (decision 0058:
    "時段量，例如 6 h 雨量"). Returns post/recipe.py fig()/missing() records."""
    if len(selected_times) < 2:
        return []
    t0 = selected_times[0]
    out = []
    for i in range(1, len(selected_times)):
        t_prev, t_curr = selected_times[i - 1], selected_times[i]
        for field in fields:
            out.append(change_map(run_dir, run_label, field, domain, out_dir, t_prev, t_curr, t0))
        out.append(change_map(run_dir, run_label, "rain_acc", domain, out_dir, t_prev, t_curr, t0,
                              fig_id="chg_rain_acc"))
        if i >= 2:
            for field in fields:
                out.append(change_map(run_dir, run_label, field, domain, out_dir, t0, t_curr, t0,
                                      fig_id="chg_%s_vs_t0" % field))
    return out

"""`wtune post preview`: before WRF runs, what `wtune post recipe` will draw and what it will be
compared with -- the figure list of the detected case types, the t=0 state from wrfinput, the
observations of the simulated window (best track, Stage IV, ERA5 z500) and the tune's run-time
estimate -- written to <project>/post/preview/ (figures/ flat, data/ the values of each figure as
netCDF, preview.html). No figure here is a model result: each names the data it was drawn from."""

from __future__ import annotations

import csv
import glob
import html
import json
import os
from datetime import timedelta

from wtune.post import caseinfo
from wtune.post.recipe import fig, figures_folder, load_catalog

KT_PER_MS = 1.0 / 0.514444


def expected_figures(types, catalog, every=None) -> list:
    """[(type, figure id, zh question)] of every catalog figure of the detected types, plus the
    time series ones when the project's post block sets `every`: what the report will hold."""
    from wtune.post.report import _TYPE_ID_PREFIX
    prefixes = [(t, _TYPE_ID_PREFIX.get(t, t + "_")) for t in types]
    if every:
        prefixes += [("timeseries", "timeseries_"), ("timeseries", "chg_")]
    return [(t, fid, ((catalog.get(fid) or {}).get("zh") or {}).get("question", ""))
            for t, p in prefixes for fid in sorted(catalog) if fid.startswith(p)]


def stage4_tiles(index, window) -> list:
    """Paths of the Stage IV files of the shortest accumulation that tiles `window` exactly
    (index: precip.stage4_index's {(hours, valid end): path}); [] when none does."""
    start, end = window
    span_h = (end - start).total_seconds() / 3600.0
    for acc in sorted({a for a, _ in index}):
        if span_h % acc:
            continue
        ends = [start + timedelta(hours=acc * (k + 1)) for k in range(int(span_h // acc))]
        if ends and all((acc, t) in index for t in ends):
            return [index[(acc, t)] for t in ends]
    return []


def _read_stage4(path):
    import numpy as np
    import xarray as xr
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        lon = np.asarray(ds["longitude"], dtype="f8")
        return (np.asarray(ds["latitude"], dtype="f8"), np.where(lon > 180, lon - 360, lon),
                np.asarray(ds[list(ds.data_vars)[0]], dtype="f8"))


def _write_data(out, fig_path, title, source, grids, valid_time=None):
    """out/data/<figures/ name>.nc: the plotted values of one figure. grids: [(group or None,
    lat, lon, [(name, values, units, long name)])], lat/lon 2-D; masked values become NaN."""
    import netCDF4
    import numpy as np
    parts = os.path.relpath(fig_path, out).split(os.sep)
    name = parts[-1] if len(parts) == 1 else "%s-%s" % (parts[0], parts[-1])
    path = os.path.join(out, "data", os.path.splitext(name)[0] + ".nc")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.title, ds.source, ds.Conventions = title, source, "CF-1.8"
        if valid_time:
            ds.valid_time = valid_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        for group, lat, lon, fields in grids:
            g = ds.createGroup(group) if group else ds
            g.createDimension("y", lat.shape[0])
            g.createDimension("x", lat.shape[1])
            coords = [("lat", lat, "degrees_north", "latitude"),
                      ("lon", lon, "degrees_east", "longitude")]
            for vname, values, units, long_name in coords + fields:
                v = g.createVariable(vname, "f4", ("y", "x"), zlib=True, fill_value=np.nan)
                v.units, v.long_name = units, long_name
                if vname not in ("lat", "lon"):
                    v.coordinates = "lat lon"
                v[:] = np.ma.filled(np.ma.asarray(values, dtype="f8"), np.nan)
    return path


def _geo_em_files(project_dir):
    """{dom: geo_em.dNN.nc} of the project's `wtune prep wps` (input/prep/wps/), else {}."""
    found = glob.glob(os.path.join(project_dir, "input", "prep", "wps", "geo_em.d[0-9][0-9].nc"))
    return {int(os.path.basename(p)[8:10]): p for p in found}


def _namelist_window(case_dir):
    """(t_start, t_end) of d01 from case/namelist.input (start_* and run_* / end_*), else
    (None, None)."""
    from datetime import datetime, timedelta
    from wtune import predict
    try:
        tc = predict.parse_namelist(os.path.join(case_dir, "namelist.input")).get("time_control", {})
        first = lambda k: int((tc.get(k) or [0])[0])  # noqa: E731
        t0 = datetime(first("start_year"), first("start_month"), first("start_day"),
                      first("start_hour"), first("start_minute"), first("start_second"))
        hours = first("run_days") * 24 + first("run_hours") + first("run_minutes") / 60.0
        if hours <= 0:
            t1 = datetime(first("end_year"), first("end_month"), first("end_day"),
                          first("end_hour"), first("end_minute"), first("end_second"))
        else:
            t1 = t0 + timedelta(hours=hours)
        return t0, t1
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def _domain_map_before_real(project_dir, case_dir, out):
    """common_domain_map before wrfinput exists: from geo_em (terrain shaded) after `wtune prep
    wps`, else from case/namelist.wps (domain boxes only)."""
    from wtune.post import domain_map
    path = os.path.join(out, "common_domain_map.png")
    geo = _geo_em_files(project_dir)
    if geo:
        crs, doms, terrain = domain_map.domains_from_files(geo)
        src = "geo_em（WPS geogrid）"
    else:
        nlw = os.path.join(case_dir, "namelist.wps")
        if not os.path.exists(nlw):
            raise FileNotFoundError("no wrfinput, geo_em or namelist.wps to draw the domains from")
        crs, doms = domain_map.domains_from_namelist(nlw)
        terrain, src = None, "namelist.wps"
    domain_map.draw(crs, doms, path, "domain/nest configuration (%s)" % src.split("（")[0],
                    terrain=terrain)
    return fig("common_domain_map", path, "preview", domain=1, extra=dict(source=src))


def _domain_map(case_dir, out):
    import netCDF4
    import numpy as np
    from wtune.post.recipes import common
    files = {d: caseinfo.initial_state(case_dir, d) for d in range(1, 10)}
    max_dom = max(d for d, p in files.items() if p)
    path = common.plot_domain_map(case_dir, max_dom, out, "wrfinput",
                                  frame_of=lambda d: ("t=0", files[d]) if files.get(d) else None)
    grids = []
    for d in range(1, max_dom + 1):
        with netCDF4.Dataset(files[d]) as ds:
            lat, lon, hgt = (np.asarray(ds.variables[k][0]) for k in ("XLAT", "XLONG", "HGT"))
        grids.append(("d%02d" % d, lat, lon, [("terrain", hgt, "m", "terrain height")]))
    data = _write_data(out, path, "domains and terrain", "wrfinput", grids)
    return fig("common_domain_map", path, "preview", domain=1,
               extra=dict(source="wrfinput（初始場）", data=data))


def _slp_wind_t0(case_dir, out, start):
    import netCDF4
    import numpy as np
    import wrf
    from wtune.post.recipes.mapfig import save_map
    path = os.path.join(out, "synoptic", "syn_slp_wind10_f000.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with netCDF4.Dataset(caseinfo.initial_state(case_dir)) as ds:
        slp = wrf.getvar(ds, "slp", timeidx=0)
        lats, lons = wrf.latlon_coords(slp, as_np=True)
        u, v = (np.asarray(ds.variables[k][0]) for k in ("U10", "V10"))
        save_map(path, lats, lons, "wrfinput d01 SLP (every 4 hPa) + 10 m wind (kt)  t=0 %s"
                 % start.strftime("%Y-%m-%d %HZ"),
                 fill_field=np.hypot(u, v) * KT_PER_MS, fill_label="10 m wind (kt)",
                 fill_cmap="YlOrRd", fill_levels=list(range(0, 75, 5)), fill_extend="max",
                 contour_field=np.asarray(slp), contour_levels=list(range(900, 1060, 4)),
                 contour_label_fmt="%d", u=u, v=v, barb_factor=KT_PER_MS,
                 barb_label="wind barbs in kt", ds=ds, crs=wrf.get_cartopy(var=slp),
                 gridline_labels=True, terrain=False)
        data = _write_data(out, path, "SLP and 10 m wind at t=0", "wrfinput d01", [(
            None, lats, lons, [("slp", np.asarray(slp), "hPa", "sea level pressure"),
                               ("u10", u, "m s-1", "10 m grid-relative x wind"),
                               ("v10", v, "m s-1", "10 m grid-relative y wind")])],
            valid_time=start)
    return fig("syn_slp_wind10_f000", path, "preview", domain=1, valid_time=start,
               extra=dict(source="wrfinput（初始場）", data=data))


def _best_track(out, storm_id, hurdat2, window):
    from wtune.post import atcf, tc
    tc_dir = os.path.join(out, "tc")
    os.makedirs(tc_dir, exist_ok=True)
    basin, cy, _year = tc.storm_id_parts(storm_id)
    bdeck = atcf.write_lines(atcf.hurdat2_to_bdeck_rows(tc.parse_hurdat2_full(hurdat2, storm_id)),
                             os.path.join(tc_dir, "best_track.dat"), basin=basin, cy=cy,
                             tech="BEST")
    path = tc._track_map_figure(tc_dir, {}, bdeck, storm_id, window=window)
    return fig("tc_track_map", path, "preview", extra=dict(source="HURDAT2（最佳路徑）", data=bdeck))


def _stage4_total(project_dir, case_dir, out, window):
    import numpy as np
    from wtune.post.obs_figs import PRECIP_LEVELS, precip_cmap
    from wtune.post.recipes.mapfig import save_map
    from wtune.post.recipes.precip import stage4_index
    tiles = stage4_tiles(stage4_index(project_dir), window)
    if not tiles:
        return None
    total = None
    for p in tiles:
        lat, lon, mm = _read_stage4(p)
        total = mm if total is None else total + mm
    la0, la1, lo0, lo1 = caseinfo._d01_bbox_and_window(case_dir)[:4]
    rows, cols = np.where((lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1))
    if not rows.size:
        return None
    sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    path = os.path.join(out, "precip", "precip_stage4_total.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_map(path, lat[sl], lon[sl], "Stage IV observed precipitation  %s - %s (%d files)"
             % (window[0].strftime("%Y-%m-%d %HZ"), window[1].strftime("%Y-%m-%d %HZ"), len(tiles)),
             fill_field=np.ma.masked_invalid(total[sl]), fill_label="mm", fill_cmap=precip_cmap(),
             fill_levels=PRECIP_LEVELS, fill_extend="max", gridline_labels=True, terrain=False)
    data = _write_data(out, path, "Stage IV precipitation total %s - %s"
                       % (window[0].strftime("%Y-%m-%d %HZ"), window[1].strftime("%Y-%m-%d %HZ")),
                       "Stage IV (%d files)" % len(tiles),
                       [(None, lat[sl], lon[sl], [("precip", np.ma.masked_invalid(total[sl]), "mm",
                                                   "accumulated precipitation")])],
                       valid_time=window[1])
    return fig("precip_stage4_total", path, "preview", valid_time=window[1],
               extra=dict(source="Stage IV（觀測雨量）", data=data))


def _era5_z500(project_dir, out, window):
    import numpy as np
    from wtune.post.recipes import era5
    from wtune.post.recipes.mapfig import save_map
    figs = []
    for p in sorted(glob.glob(os.path.join(project_dir, "input", "era5", "z500_*.nc"))):
        lat, lon, z, valid = era5.read_subset(p)
        if not window[0] <= valid <= window[1]:
            continue
        lons, lats = np.meshgrid(lon, lat)
        path = os.path.join(out, "synoptic", "syn_era5_z500_%s.png" % valid.strftime("%Y%m%d%H"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_map(path, lats, lons, "ERA5 500 hPa height (every 60 m)  %s"
                 % valid.strftime("%Y-%m-%d %HZ"), contour_field=z,
                 contour_levels=list(range(4800, 6000, 60)), contour_label_fmt="%d",
                 gridline_labels=True, terrain=False)
        data = _write_data(out, path, "ERA5 500 hPa geopotential height", "ERA5",
                           [(None, lats, lons, [("z500", z, "m", "500 hPa height")])],
                           valid_time=valid)
        figs.append(fig("syn_era5_z500", path, "preview", valid_time=valid,
                        extra=dict(source="ERA5（再分析）", data=data)))
    return figs


def run_preview(project_dir, project_doc, out) -> dict:
    """Writes out/ (per-type folders, figures/, expected_figures.csv, preview.html). Returns
    {out_dir, figures, expected, notes, report_path, figures_dir}."""
    import wtune.project
    case_dir = os.path.join(project_dir, "case")
    storm_id, hurdat2 = wtune.project.project_storm(project_doc)
    every = wtune.project.project_post_defaults(project_doc)["every"]
    os.makedirs(out, exist_ok=True)
    has_state = bool(caseinfo.initial_state(case_dir))
    figures, notes = [], []
    if has_state:
        info = caseinfo.detect_types([case_dir], hurdat2_path=hurdat2, storm_id=storm_id)
        window = caseinfo._d01_bbox_and_window(case_dir)[4:]
    else:                       # before `wtune prep real`: namelists (and geo_em) only
        info = dict(types=["common"], evidence={}, run_length_h=0.0, domains=[])
        window = _namelist_window(case_dir)
        notes.append("no wrfinput yet (wtune prep real): figure types are not detected and the "
                     "t=0 fields are not drawn; the domain map comes from %s"
                     % ("geo_em" if _geo_em_files(project_dir) else "namelist.wps"))
    expected = expected_figures(info["types"], load_catalog(), every=every)

    steps = [("domain map", lambda: [_domain_map(case_dir, out) if has_state
                                     else _domain_map_before_real(project_dir, case_dir, out)])]
    if has_state:
        steps.append(("t=0 SLP", lambda: [_slp_wind_t0(case_dir, out, window[0])]))
    if window[0] is not None:
        steps.append(("ERA5 z500", lambda: _era5_z500(project_dir, out, window)))
        if "tc" in info["types"] and storm_id and hurdat2:
            steps.append(("best track", lambda: [_best_track(out, storm_id, hurdat2, window)]))
        if has_state:
            steps.append(("Stage IV", lambda: [_stage4_total(project_dir, case_dir, out, window)]))
    for name, step in steps:
        try:
            figures += [f for f in step() if f and f["path"]]
        except Exception as e:                              # noqa: BLE001
            notes.append("%s skipped: %s: %s" % (name, type(e).__name__, e))
    if has_state and not any(f["id"] == "precip_stage4_total" for f in figures):
        notes.append("Stage IV: no fetched accumulation tiles %s - %s" % window)

    with open(os.path.join(out, "expected_figures.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "figure_id", "question"])
        w.writerows(expected)
    report = wtune.project.latest_tune_report(project_dir)
    summary = []
    if report:
        with open(report) as fh:
            summary = json.load(fh).get("summary", [])
    earlier = sorted(d for d in glob.glob(os.path.join(project_dir, "post", "*", "report.html"))
                     if os.path.dirname(d) != os.path.abspath(out))
    report_path = _write_html(out, info, window, expected, figures, summary, earlier, notes)
    from wtune.post.recipe import write_index
    write_index(out, [case_dir], ["preview"], info, figures,
                [dict(source="preview", level="info", text=n) for n in notes], kind="preview")
    return dict(out_dir=out, figures=figures, expected=expected, notes=notes,
                report_path=report_path, figures_dir=figures_folder(out, figures))


def _write_html(out, info, window, expected, figures, summary, earlier, notes):
    from wtune.post.report import _embed_image
    e = html.escape
    if window[0] is not None and window[1] is not None:
        span = "%s – %s（%.0f h）" % (window[0].strftime("%Y-%m-%d %HZ"),
                                    window[1].strftime("%Y-%m-%d %HZ"),
                                    (window[1] - window[0]).total_seconds() / 3600.0)
    else:
        span = "（namelist.input 讀不到時段）"
    rows = "".join("<tr><td>%s</td><td><code>%s</code></td><td>%s</td></tr>" % (e(t), e(f), e(q))
                   for t, f, q in expected)
    figs = "".join('<figure><img src="%s" style="max-width:100%%"><figcaption>%s：%s　數據：'
                   "<code>%s</code></figcaption></figure>"
                   % (_embed_image(f["path"]), e(f["source"]), e(os.path.basename(f["path"])),
                      e(os.path.relpath(f["data"], out) if f.get("data") else "—"))
                   for f in figures)
    parts = [
        "<!doctype html><meta charset='utf-8'><title>preview</title>",
        "<h1>預分析（還沒跑 WRF）</h1>",
        "<p>模擬時段 %s。偵測到的類型：%s。下面的圖都不是模式結果，而是初始場或"
        "觀測；跑完後拿 <code>post/&lt;id&gt;/figures/</code> 的同名圖來比。</p>"
        % (span, e(", ".join(info["types"]))),
        "<h2>預估</h2><pre>%s</pre>" % e("\n".join(summary) or "（沒有 tune 報告）"),
        "<h2>跑完後會畫的圖（%d 種）</h2><table border=1 cellpadding=4>"
        "<tr><th>類型</th><th>圖</th><th>回答的問題</th></tr>%s</table>" % (len(expected), rows),
        "<h2>現在就能看的</h2>%s" % figs,
    ]
    if earlier:
        parts.append("<h2>之前的後處理</h2><ul>%s</ul>"
                     % "".join("<li><code>%s</code></li>" % e(p) for p in earlier))
    if notes:
        parts.append("<h2>略過</h2><ul>%s</ul>" % "".join("<li>%s</li>" % e(n) for n in notes))
    path = os.path.join(out, "preview.html")
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    return path

"""post/recipes/precip.py -- the "precip" figure set (decision 0057 section 2): accumulated
precipitation (RAINC+RAINNC, `wtune post extract`'s own `rain_acc` field, drawn with
post/plotting.py's engine) is always produced (precip is always detected).

Observation comparison (needs `wtune post fetch stage4`, plus `stations` for the point
verification): the model accumulation over the longest window that (a) has an NCEP Stage IV
product ending at its end time, (b) lies after spin-up and (c) fits inside the run (24 h Stage IV
ends at 12 UTC, 6 h at 00/06/12/18, else 1 h) is verified against Stage IV with MET Grid-Stat:
Stage IV is OBS on its own ~4.8 km HRAP grid (`regrid.to_grid = OBS`, BUDGET), the model is first
moved from the curvilinear WRF grid to a lat/lon grid by `cdo remapcon`; verification is
restricted to the model domain's interior (spec_bdy_width cells excluded). Scores: FHO/CTC/CTS
(POD, FAR, CSI, ETS, frequency bias) at 1/10/25/50/100 mm, NBRCNT FSS at neighbourhood widths
1/3/5/9 points. Station verification (T2, Td2, 10 m wind, MSLP) against ISD-Lite with MET
ascii2nc + Point-Stat lives here too because precip is the one type every run has (the
"common" set is owned by another module) -- see post/met_pointstat.py.

Every missing observation figure says whether the data does not exist for this run (coverage)
or which fetch command to run.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta

from post import sources, tools
from post.extract import list_frame_refs

# WPC-style event thresholds in mm (0.01, 0.1, 0.25, 0.5, 1 in, then 50 mm; decision 0064)
THRESHOLDS_MM = (0.254, 2.54, 6.35, 12.7, 25.4, 50)
WIDTHS = (1, 3, 5, 9)
STAGE4_CELL_KM = 4.763
RADAR_RANGE_KM = 230


def _project_of(out_dir):
    """Walk-up fallback (absolute path) for callers that were not handed --project."""
    try:
        from wtune.project import find_project
        return find_project(os.path.abspath(out_dir))
    except Exception:
        return None


def stage4_index(project_dir):
    """{(accum_h, valid datetime): path} of the Stage IV files present under the project."""
    out = {}
    base = sources.dataset_dir(os.path.abspath(project_dir), "stage4") if project_dir else None
    if not base or not os.path.isdir(base):
        return out
    for name in os.listdir(base):
        if name.startswith("ST4.") and name.endswith(".grib"):
            try:
                out[(int(name[15:17]), datetime.strptime(name[4:14], "%Y%m%d%H"))] = \
                    os.path.join(base, name)
            except ValueError:
                pass
    return out


def pick_window(frame_times, spinup_hours, index):
    """(accum_h, valid time, path) of the longest Stage IV window inside the run after
    spin-up (latest one among that length), else None. frame_times: sorted datetimes."""
    if len(frame_times) < 2:
        return None
    have = set(frame_times)
    t0 = frame_times[0] + timedelta(hours=spinup_hours)
    for accum in (24, 6, 1):
        cands = sorted(t for (a, t) in index if a == accum
                       and t in have and t - timedelta(hours=accum) in have
                       and t - timedelta(hours=accum) >= t0)
        if cands:
            return accum, cands[-1], index[(accum, cands[-1])]
    return None


def stage4_missing_reason(project_dir, run_dir, frame_times, bbox, has_files):
    """Coverage ('data does not exist') vs 'fetch it' -- the missing() reason text."""
    from post.fetch import stage4
    if bbox is not None and not stage4._overlaps(bbox):
        return ("Stage IV does not exist for this region (coverage: CONUS HRAP grid only, no "
                "Puerto Rico/Hawaii/Alaska; run bbox %s)" % (tuple(round(x, 1) for x in bbox),))
    if frame_times and frame_times[-1] < datetime(2001, 1, 1):
        return "Stage IV does not exist before 2001 (coverage)"
    cmd = "wtune post fetch stage4 --project %s --run-dir %s" % (project_dir or "<project>", run_dir)
    where = (sources.dataset_dir(project_dir, "stage4") if project_dir
             else "<no wtune project found -- pass --project>")
    if has_files:
        return ("Stage IV files are present but none is a complete window after spin-up inside "
                "the run (24 h products end 12 UTC, 6 h at 00/06/12/18); fetch more with: " + cmd)
    return ("Stage IV not fetched yet (no ST4.*.grib in %s) -- run on a login node: " % where) + cmd


def _model_accum(ref0, ref1):
    """(lat, lon, mm) model precipitation between two wrfout frames: RAINC+RAINNC(t1) -
    RAINC+RAINNC(t0) -- WRF's cumulative rain, difference of two frames is plain arithmetic.
    ref0/ref1: (path, timeidx) (post.extract.frame_ref; a frames_per_outfile > 1 file holds
    several valid times, the accumulation AT a valid time is the value at its timeidx), or a
    bare path for timeidx 0."""
    import netCDF4
    import numpy as np
    from post.extract import accumulated_rain
    (path0, i0), (path1, i1) = [(r, 0) if isinstance(r, str) else r for r in (ref0, ref1)]
    with netCDF4.Dataset(path0) as d0, netCDF4.Dataset(path1) as d1:
        r0 = np.asarray(accumulated_rain(d0, i0), "f8")   # incl. the I_RAIN* bucket part
        r1 = np.asarray(accumulated_rain(d1, i1), "f8")
        lat = np.asarray(d1.variables["XLAT"][i1], "f8")
        lon = np.asarray(d1.variables["XLONG"][i1], "f8")
    return lat, lon, np.maximum(r1 - r0, 0.0)


def _write_csvs(out_dir, tag, cts, nbr):
    p1 = os.path.join(out_dir, "precip_categorical_%s.csv" % tag)
    with open(p1, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold_mm", "POD", "FAR", "CSI", "frequency_bias", "n_points"])
        for r in cts:
            w.writerow([r["thresh"], r["pod"], r["far"], r["csi"], r["fbias"], r["total"]])
    p2 = os.path.join(out_dir, "precip_fss_%s.csv" % tag)
    with open(p2, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold_mm", "width_points", "width_km", "FSS", "FBS", "forecast_rate",
                    "observed_rate"])
        for r in nbr:
            w.writerow([r["thresh"], r["width"], round(r["width"] * STAGE4_CELL_KM, 2), r["fss"],
                        r["fbs"], r["f_rate"], r["o_rate"]])
    return p1, p2


def _stage4_figures(run_dir, run_label, domain, frames, out_dir, spinup_hours, project_dir):
    from post.recipe import fig, missing
    from post import met_gridstat, obs_figs
    from post.caseinfo import WRF_TIME_FMT, spec_bdy_width

    times = [datetime.strptime(f[0], WRF_TIME_FMT) for f in frames]
    refs = {datetime.strptime(f[0], WRF_TIME_FMT): (f[1], f[2] if len(f) > 2 else 0)
            for f in frames}
    ids = ("precip_obs_compare", "precip_diff", "precip_ets_fbias", "precip_fss")
    index = stage4_index(project_dir)
    win = pick_window(times, spinup_hours, index)
    if win is None:
        bbox = None
        try:
            import netCDF4
            with netCDF4.Dataset(frames[0][1]) as ds:
                la, lo = ds.variables["XLAT"][0], ds.variables["XLONG"][0]
                bbox = (float(la.min()), float(la.max()), float(lo.min()), float(lo.max()))
        except Exception:
            pass
        reason = stage4_missing_reason(project_dir, run_dir, times, bbox, bool(index))
        return [missing(i, "NCEP Stage IV QPE", reason) for i in ids]

    accum, valid, s4_path = win
    t_start = valid - timedelta(hours=accum)
    tag = "A%02d" % accum
    work = os.path.join(out_dir, "gridstat_stage4_%s" % tag)
    os.makedirs(work, exist_ok=True)
    try:
        lat, lon, mm = _model_accum(refs[t_start], refs[valid])
        fcst_nc = met_gridstat.curvilinear_to_latlon(
            lat, lon, {"APCP_MODEL": (mm, "mm", "WRF accumulated precipitation")}, valid,
            os.path.join(work, "model_apcp.nc"), method="remapcon")
        poly = met_gridstat.write_interior_polygon(refs[valid][0], spec_bdy_width(run_dir),
                                                   os.path.join(work, "interior.poly"))
        mask_note = "model interior"
        sites_file = os.path.join(sources.dataset_dir(project_dir, "stage4"), "nexrad_sites.txt")
        if os.path.isfile(sites_file):
            from post.fetch.nexrad import parse_homr
            with open(sites_file) as fh:
                sites = parse_homr(fh.read())
            cov, dropped = met_gridstat.coverage_polygon(sites, RADAR_RANGE_KM, poly,
                                                         os.path.join(work, "coverage.poly"))
            if cov:
                poly, mask_note = cov, "model interior within %d km of a NEXRAD site" % RADAR_RANGE_KM
        res = met_gridstat.run_verification(
            fcst_nc, s4_path, poly, work, "APCP_MODEL", "APCP", [">=%g" % t for t in THRESHOLDS_MM],
            obs_level=tag, widths=WIDTHS, to_grid="OBS", regrid_method="BUDGET",
            model="WRF", obtype="STAGE4")
        cts = met_gridstat.parse_cts_table(res["cts"])
        nbr = met_gridstat.parse_nbrcnt_table(res["nbrcnt"])
        glat, glon, gf, go, _fn, _on = met_gridstat.read_pairs_fields(res["pairs_nc"])
    except (tools.ToolNotFoundError, tools.ToolError, met_gridstat.METError, ImportError,
            OSError, StopIteration) as e:
        return [missing(i, "MET Grid-Stat vs Stage IV", "verification failed: %s" % e) for i in ids]

    src = sources.source_tags(project_dir, ["stage4"])
    csv_cat, csv_fss = _write_csvs(out_dir, tag, cts, nbr)
    win_txt = "%s -> %s UTC (%d h)" % (t_start.strftime("%Y-%m-%d %H"), valid.strftime("%Y-%m-%d %H"), accum)
    common = dict(obs_sources=src, tool="MET grid_stat (model regridded to Stage IV grid, BUDGET)",
                  window=win_txt, csv=[csv_cat, csv_fss], mask=mask_note)
    stamp = valid.strftime(WRF_TIME_FMT)
    out = []
    p = obs_figs.side_by_side(
        glat, glon, gf, go, ("%s d%02d model" % (run_label, domain), "NCEP Stage IV (obs)"),
        "accumulated precipitation %s, matched points, %s" % (win_txt, mask_note),
        os.path.join(out_dir, "precip_obs_compare.png"), obs_figs.PRECIP_LEVELS, cbar_label="mm")
    out.append(fig("precip_obs_compare", p, run_label, domain="d%02d" % domain, valid_time=stamp, extra=common))
    import numpy as np
    dmax = float(np.nanpercentile(np.abs(gf - go), 98)) or 1.0
    p = obs_figs.diff_map(glat, glon, gf, go, "model - Stage IV, %s, %s" % (win_txt, mask_note),
                          os.path.join(out_dir, "precip_diff.png"), "mm", dmax)
    out.append(fig("precip_diff", p, run_label, domain="d%02d" % domain, valid_time=stamp, extra=common))
    p = obs_figs.scores_vs_threshold(cts, "Grid-Stat categorical scores vs Stage IV, %s (n=%d points)"
                                     % (win_txt, cts[0]["total"]),
                                     os.path.join(out_dir, "precip_ets_fbias.png"), "threshold (mm)")
    out.append(fig("precip_ets_fbias", p, run_label, domain="d%02d" % domain, valid_time=stamp, extra=common))
    p = obs_figs.fss_vs_scale(nbr, STAGE4_CELL_KM, "Grid-Stat FSS vs Stage IV, %s" % win_txt,
                              os.path.join(out_dir, "precip_fss.png"), " mm")
    out.append(fig("precip_fss", p, run_label, domain="d%02d" % domain, valid_time=stamp, extra=common))
    prov = sources.read_provenance(project_dir, "stage4") if project_dir else {}
    if prov is None or prov.get("failed"):
        # a partial download still gives figures, on whatever window the files allow
        from post.recipe import note
        n = len(prov["failed"]) if prov else None
        out.append(note("precip", "Stage IV download was incomplete (%s): this comparison uses %s. "
                        "Run `wtune post fetch stage4` again, then post again."
                        % ("%d file(s) failed" % n if n else "no provenance record", win_txt)))
    return out


def station_verification(run_dir, domain, frames, out_dir, spinup_hours, project_dir):
    """The one station Point-Stat path (precip and terrain both use it): (result dict, None, None)
    on success, else (None, dataset label, reason). `domain`/`frames` select the model grid."""
    from post import met_pointstat
    try:
        return met_pointstat.verify_run(run_dir, domain, frames, project_dir, out_dir,
                                        spinup_hours), None, None
    except met_pointstat.NoStations as e:
        return None, "NOAA ISD-Lite stations", str(e)
    except (tools.ToolNotFoundError, tools.ToolError, ImportError, OSError) as e:
        return None, "MET Point-Stat vs ISD-Lite", "verification failed: %s" % e
    except Exception as e:  # noqa: BLE001 -- MET/cdo/wrf-python failures must not drop the recipe
        return None, "MET Point-Stat vs ISD-Lite", "verification failed: %s: %s" % (type(e).__name__, e)


def _station_figures(run_dir, run_label, domain, frames, out_dir, spinup_hours, project_dir):
    from post.recipe import fig, missing
    ids = ("precip_stn_series", "precip_stn_map")
    res, dataset, reason = station_verification(run_dir, domain, frames, out_dir, spinup_hours,
                                                project_dir)
    if res is None:
        return [missing(i, dataset, reason) for i in ids]
    src = sources.source_tags(project_dir, ["stations"])
    extra = dict(obs_sources=src, tool="MET ascii2nc + point_stat (BILIN, interior mask)",
                 csv=res["csv"], summary=res["summary"])
    return [fig("precip_stn_series", res["series_png"], run_label, domain="d%02d" % domain,
                valid_time="%s..%s" % (res["t_first"], res["t_last"]), extra=extra),
            fig("precip_stn_map", res["map_png"], run_label, domain="d%02d" % domain,
                valid_time="%s..%s" % (res["t_first"], res["t_last"]), extra=extra)]


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6, project_dir=None):
    from post.recipe import fig, missing

    run_dir, run_label = run_dirs[0], labels[0]
    domain = case_info["domains"][-1]["domain"]   # finest domain, most informative rain field
    frames = list_frame_refs(run_dir, domain=domain)     # [(time, path, timeidx)]
    out = []
    if not frames:
        out.append(missing("precip_accum", "wrfout_d%02d_*" % domain, "no frames found"))
        return out
    last_time = frames[-1][0]
    try:
        from post.plotting import plot_field
        from post.timeseries import fixed_scale
        lev, cmap, extend = fixed_scale(run_dir, "rain_acc", domain, [last_time])
        written = plot_field(run_dir, "rain_acc", out_dir, domain=domain,
                             times=[last_time], basemap="cartopy", levels=lev, cmap=cmap,
                             extend=extend, gridline_labels=True)
        out.append(fig("precip_accum", written[0], run_label, domain="d%02d" % domain,
                       valid_time=last_time,
                       extra={"note": "accumulated since run start (spin-up period "
                                      "included in the accumulation, per WRF's own "
                                      "RAINC/RAINNC convention -- spin-up excluded: "
                                      "%g h means excluded from any time-series analysis "
                                      "above, not from this cumulative total)" % spinup_hours}))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("precip_accum", "wrf-python", str(e)))

    project_dir = os.path.abspath(project_dir) if project_dir else _project_of(out_dir)
    out += _stage4_figures(run_dir, run_label, domain, frames, out_dir, spinup_hours, project_dir)
    out += _station_figures(run_dir, run_label, domain, frames, out_dir, spinup_hours, project_dir)
    return out

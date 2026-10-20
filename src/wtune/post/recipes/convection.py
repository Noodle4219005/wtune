"""post/recipes/convection.py -- the "convection" figure set (decision 0057 section 2):
max column reflectivity (wrf-python `mdbz`), MCAPE/MCIN (`cape_2d`), 2-5 km updraft helicity
time-maximum (wrf-python `uhel`, bottom=2000/top=5000 m -- the standard Kain et al. 2008 UH
layer, and wrf-python's own default for this diagnostic, so no bespoke formula is needed here),
a representative-point Skew-T/hodograph (MetPy, via `wtune post sounding`'s own
post/sounding.py -- reused, not re-implemented), and (when radar obs are available) a
side-by-side reflectivity comparison plus MET Grid-Stat categorical scores and FSS against radar:
MRMS composite reflectivity when fetched (2020-10-14 onward), else a composite built from NEXRAD
Level II volumes (`wtune post fetch nexrad`; post/radar_composite.py, Py-ART gridding). Model
`mdbz` and the radar composite are put on one lat/lon grid (cdo remapnn) and scored by Grid-Stat at
20/35/45 dBZ (FHO/CTC/CTS + NBRCNT FSS) inside the model interior; cells beyond every radar's
range are missing, not "no echo". Scores of the sampled times are pooled by summing contingency
counts (CTC) / FBS terms. MODE object verification is not run. A missing comparison says whether
the data does not exist (coverage) or which fetch command to run.

MCAPE/MCIN maps draw cells where wrf-python has no value (no LFC, failed parcel) as a labelled
hatched "undefined" mask, not as white holes.

The finest domain with cu_physics=0 (the convection-permitting domain, per detect_convection())
is used for every field here.
"""
from __future__ import annotations

import os

from wtune.post import tools
from wtune.post.conventions import reflectivity_cmap
from wtune.post.extract import frame_ref, list_frame_refs, list_frames
from wtune.post.recipes.common import FIELD_ERRORS, error_reason


def _finest_convective_domain(domains):
    hits = [d for d in domains if d["dx_km"] <= 4.0 and (d["cu_physics"] or 0) == 0]
    return max(hits, key=lambda d: d["domain"])["domain"] if hits else domains[-1]["domain"]


def _post_spinup_frames(run_dir, domain, spinup_hours):
    from datetime import datetime
    from wtune.post.caseinfo import WRF_TIME_FMT
    frames = list_frame_refs(run_dir, domain=domain)      # [(time, path, timeidx)]
    if not frames:
        return frames
    t0 = datetime.strptime(frames[0][0], WRF_TIME_FMT)
    return [f for f in frames
           if (datetime.strptime(f[0], WRF_TIME_FMT) - t0).total_seconds() / 3600.0 >= spinup_hours] \
        or frames


UNDEF_LABEL = "undefined (no LFC / parcel fails)"
MCAPE_LEVELS = list(range(0, 4001, 250))
MCIN_LEVELS = list(range(0, 101, 10))       # almost the whole domain is < 40 J/kg


def _undefined(arr):
    """Cells wrf-python could not evaluate: NaN or the 9.97e36 fill value."""
    import numpy as np
    return ~np.isfinite(arr) | (np.abs(arr) > 1e10)


def _map_field(run_dir, domain, field, time_str, out_path, title, cmap="viridis",
               fill_label=""):
    """title may contain '{time}', replaced by the init/valid/lead label (decision 0064)."""
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    from wtune.post.recipes.mapfig import save_map
    path, ti = frame_ref(run_dir, domain, time_str)
    kw = {}
    with netCDF4.Dataset(path) as ds:
        if field == "mdbz":
            var = wrf_mod.getvar(ds, "mdbz", timeidx=ti)
        elif field == "mcape":
            var = wrf_mod.getvar(ds, "cape_2d", timeidx=ti)[0]
        elif field == "mcin":
            var = wrf_mod.getvar(ds, "cape_2d", timeidx=ti)[1]
        elif field == "uhel":
            var = wrf_mod.getvar(ds, "uhel", timeidx=ti, bottom=2000.0, top=5000.0)
        else:
            raise ValueError(field)
        lats, lons = wrf_mod.latlon_coords(var, as_np=True)
        crs = wrf_mod.get_cartopy(var=var)
        arr = np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(var, "f8")), np.nan), "f8")
        from wtune.post.conventions import ds_time_label, reflectivity_colors
        title = title.replace("{time}", ds_time_label(ds, time_str))
        if field == "mdbz":
            # NWS WSR-88D table, 5 dBZ steps (decision 0064)
            levels, colors = reflectivity_colors()
            arr = np.where(arr < levels[0], np.nan, arr)
            kw = dict(fill_levels=levels, fill_colors=colors, fill_extend="max")
        if field in ("mcape", "mcin"):
            undef = _undefined(arr)
            arr = np.where(undef, np.nan, arr)
            kw = dict(fill_levels=MCAPE_LEVELS if field == "mcape" else MCIN_LEVELS,
                      fill_extend="max", hatch_mask=undef, hatch_label=UNDEF_LABEL)
        save_map(out_path, lats, lons, title, fill_field=arr, fill_label=fill_label,
                 fill_cmap=cmap, ds=ds, crs=crs, **kw)
    return out_path


def _uh_time_max(run_dir, domain, frames):
    """Time-maximum of 2-5 km updraft helicity over the given frames -- a pointwise max over
    time, matching "2-5 km 上升螺旋度的時間最大值" (decision 0057 section 2), computed with
    plain numpy over wrf-python's own per-frame uhel field (no tool provides the time-max
    aggregation itself)."""
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    best = None
    lats = lons = crs = None
    for f in frames:                    # (time, path[, timeidx])
        path, ti = f[1], (f[2] if len(f) > 2 else 0)
        with netCDF4.Dataset(path) as ds:
            var = wrf_mod.getvar(ds, "uhel", timeidx=ti, bottom=2000.0, top=5000.0)
            arr = np.asarray(var)
            if best is None:
                best = arr.copy()
                lats, lons = wrf_mod.latlon_coords(var, as_np=True)
                crs = wrf_mod.get_cartopy(var=var)
            else:
                best = np.maximum(best, arr)
    return best, lats, lons, crs


def _max_reflectivity_point(run_dir, domain, time_str):
    import netCDF4
    import numpy as np
    wrf_mod = __import__("wrf")
    path, ti = frame_ref(run_dir, domain, time_str)
    with netCDF4.Dataset(path) as ds:
        mdbz = np.asarray(wrf_mod.getvar(ds, "mdbz", timeidx=ti))
        lat = np.asarray(ds.variables["XLAT"][ti])
        lon = np.asarray(ds.variables["XLONG"][ti])
        j, i = np.unravel_index(np.argmax(mdbz), mdbz.shape)
        return float(lat[j, i]), float(lon[j, i])


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6):
    from wtune.post.recipe import fig, missing

    run_dir, run_label = run_dirs[0], labels[0]
    domains = case_info["domains"]
    domain = _finest_convective_domain(domains)
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return [missing("conv_max_reflectivity", "wrfout_d%02d_*" % domain,
                        "no frames found for the convection-permitting domain")]
    post_spinup = _post_spinup_frames(run_dir, domain, spinup_hours)
    last_time = post_spinup[-1][0]

    out = []
    try:
        path = _map_field(run_dir, domain, "mdbz", last_time,
                          os.path.join(out_dir, "conv_max_reflectivity.png"),
                          "%s d%02d max column reflectivity (NWS table)  {time}" % (run_label, domain),
                          fill_label="reflectivity (dBZ)")
        out.append(fig("conv_max_reflectivity", path, run_label, domain="d%02d" % domain,
                       valid_time=last_time))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_max_reflectivity", "wrf-python", str(e)))
    except FIELD_ERRORS as e:
        out.append(missing("conv_max_reflectivity", "wrfout_d%02d" % domain, error_reason(e)))

    try:
        for field, label in (("mcape", "MCAPE (J/kg)"), ("mcin", "MCIN (J/kg)")):
            path = _map_field(run_dir, domain, field, last_time,
                              os.path.join(out_dir, "conv_%s.png" % field),
                              "%s d%02d %s  {time}" % (run_label, domain, label),
                              cmap="plasma", fill_label=label)
            out.append(fig("conv_%s" % field, path, run_label, domain="d%02d" % domain,
                           valid_time=last_time))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_mcape_mcin", "wrf-python", str(e)))
    except FIELD_ERRORS as e:
        out.append(missing("conv_mcape_mcin", "wrfout_d%02d" % domain, error_reason(e)))

    try:
        uh_max, lats, lons, crs = _uh_time_max(run_dir, domain, post_spinup)
        from wtune.post.recipes.mapfig import save_map
        path = os.path.join(out_dir, "conv_uh25_max.png")
        from wtune.post.conventions import UH_LEVELS, lead_range_label
        import numpy as np
        uh_plot = np.where(np.asarray(uh_max, dtype="f8") < UH_LEVELS[0], np.nan, uh_max)
        save_map(path, lats, lons,
                "%s d%02d 2-5 km AGL updraft helicity, time max over %s\n"
                "(spin-up %g h excluded; >= 75 m$^2$ s$^{-2}$ marks rotating storms)"
                % (run_label, domain,
                   lead_range_label(post_spinup[0][1], post_spinup[0][0], post_spinup[-1][0]),
                   spinup_hours),
                fill_field=uh_plot, fill_label="2-5 km UH (m$^2$ s$^{-2}$)", fill_cmap="Reds",
                fill_levels=UH_LEVELS, fill_extend="max", crs=crs)
        out.append(fig("conv_uh25_max", path, run_label, domain="d%02d" % domain,
                       valid_time="%s..%s" % (post_spinup[0][0], post_spinup[-1][0])))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_uh25_max", "wrf-python", str(e)))
    except FIELD_ERRORS as e:
        out.append(missing("conv_uh25_max", "wrfout_d%02d" % domain, error_reason(e)))

    try:
        lat, lon = _max_reflectivity_point(run_dir, domain, last_time)
        from wtune.post.sounding import make_sounding
        path = os.path.join(out_dir, "conv_skewt.png")
        make_sounding(run_dir, lat, lon, last_time, path, domain=domain)
        out.append(fig("conv_skewt", path, run_label, domain="d%02d" % domain,
                       valid_time=last_time,
                       extra={"point": "%.3f,%.3f (max-reflectivity column)" % (lat, lon)}))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_skewt", "MetPy", str(e)))
    except FIELD_ERRORS as e:
        out.append(missing("conv_skewt", "wrfout_d%02d" % domain, error_reason(e)))

    out += _radar_figures(run_dir, run_label, domain, case_info, out_dir, spinup_hours)
    return out


# --------------------------------------------------------------------------------------
# model vs radar composite reflectivity (MET Grid-Stat)
# --------------------------------------------------------------------------------------

DBZ_THRESHOLDS = (20, 35, 45)
WIDTHS = (1, 3, 5, 9)
MAX_RADAR_TIMES = 12
RADAR_IDS = ("conv_radar_compare", "conv_radar_scores", "conv_fss")


def _project_of(out_dir):
    try:
        from wtune.project import find_project
        return find_project(out_dir)
    except Exception:
        return None


def _nexrad_volumes(project_dir):
    """{radar: [(datetime, path)]} of the Level II volumes present under the project."""
    from wtune.post import sources
    from wtune.post.fetch.nexrad import key_time
    base = sources.dataset_dir(project_dir, "nexrad")
    out = {}
    if not os.path.isdir(base):
        return out
    for radar in sorted(os.listdir(base)):
        d = os.path.join(base, radar)
        if os.path.isdir(d):
            vols = [(key_time(f), os.path.join(d, f)) for f in os.listdir(d)]
            out[radar] = sorted((t, p) for t, p in vols if t is not None and not p.endswith("_MDM"))
    return out


def _mrms_files(project_dir):
    """{datetime: path} of MRMS reflectivity composites present under the project."""
    from wtune.post import sources
    from wtune.post.fetch.mrms import key_time, PRODUCTS
    base = sources.dataset_dir(project_dir, "mrms")
    out = {}
    for root, _dirs, files in os.walk(base):
        if PRODUCTS["refl"] in root:
            for f in files:
                t = key_time(f + ".gz")
                if t is not None and f.endswith(".grib2"):
                    out[t] = os.path.join(root, f)
    return out


def _nearest(vols, t, tol_min=8):
    best = min(vols, key=lambda v: abs((v[0] - t).total_seconds()), default=None)
    return best[1] if best and abs((best[0] - t).total_seconds()) <= tol_min * 60 else None


def radar_missing_reason(project_dir, run_dir, frame_times, bbox):
    """The missing() reason: coverage (data does not exist) or the fetch command."""
    from wtune.post.fetch import mrms
    t0 = frame_times[0] if frame_times else None
    cmd = "wtune post fetch nexrad --project %s --run-dir %s" % (project_dir or "<project>", run_dir)
    if t0 is not None and t0 >= mrms.FIRST_DAY:
        cmd += "   (or: wtune post fetch mrms --project %s --run-dir %s)" % (project_dir or "<project>", run_dir)
    elif t0 is not None:
        cmd += "   (MRMS does not exist before %s, so NEXRAD Level II is the only radar source)" % mrms.FIRST_DAY.date()
    return ("radar reflectivity not available locally for this run -- run on a login node: " + cmd)


def _pool_counts(per_time):
    """[{thresh, pod, far, csi, fbias, total}] from summed contingency counts of every time."""
    agg = {}
    for ctc in per_time:
        for r in ctc:
            a = agg.setdefault(r["thresh"], dict(fy_oy=0, fy_on=0, fn_oy=0, fn_on=0))
            for k in a:
                a[k] += r[k]
    out = []
    for th, a in sorted(agg.items()):
        hits, fa, miss = a["fy_oy"], a["fy_on"], a["fn_oy"]
        tot = hits + fa + miss + a["fn_on"]
        def ratio(n, d):
            return n / d if d else float("nan")
        out.append(dict(thresh=th, pod=ratio(hits, hits + miss), far=ratio(fa, hits + fa),
                        csi=ratio(hits, hits + fa + miss), fbias=ratio(hits + fa, hits + miss),
                        total=tot))
    return out


def _pool_fss(per_time):
    """Pooled FSS per (threshold, width): 1 - sum(FBS) / sum(FBS_ref), FBS_ref = FBS / (1 - FSS)
    recovered from MET's own FBS and FSS columns of every time (MET's NBRCNT aggregation rule);
    o_rate/f_rate are plain means."""
    agg = {}
    for rows in per_time:
        for r in rows:
            if not (r["fss"] == r["fss"] and r["fbs"] == r["fbs"]) or r["fss"] >= 1.0:
                continue
            a = agg.setdefault((r["thresh"], r["width"]), dict(fbs=0.0, ref=0.0, o=[], f=[]))
            a["fbs"] += r["fbs"]
            a["ref"] += r["fbs"] / (1.0 - r["fss"]) if r["fss"] < 1.0 else 0.0
            a["o"].append(r["o_rate"])
            a["f"].append(r["f_rate"])
    out = []
    for (th, w), a in sorted(agg.items()):
        out.append(dict(thresh=th, width=w, fss=(1.0 - a["fbs"] / a["ref"]) if a["ref"] > 0 else float("nan"),
                        fbs=a["fbs"], o_rate=sum(a["o"]) / len(a["o"]), f_rate=sum(a["f"]) / len(a["f"])))
    return out


def _radar_figures(run_dir, run_label, domain, case_info, out_dir, spinup_hours):
    from datetime import datetime
    import csv
    import netCDF4
    import numpy as np
    from wtune.post.recipe import fig, missing
    from wtune.post import met_gridstat, obs_figs, radar_composite, sources
    from wtune.post.caseinfo import WRF_TIME_FMT, spec_bdy_width

    frames = _post_spinup_frames(run_dir, domain, spinup_hours)
    times = [(datetime.strptime(t, WRF_TIME_FMT), (p, ti)) for t, p, ti in frames]
    project_dir = _project_of(out_dir)
    if not times:
        return [missing(i, "radar composite reflectivity", "no frames") for i in RADAR_IDS]
    bbox = None
    try:
        with netCDF4.Dataset(times[0][1][0]) as ds:
            la, lo = ds.variables["XLAT"][0], ds.variables["XLONG"][0]
            bbox = (float(la.min()), float(la.max()), float(lo.min()), float(lo.max()))
    except Exception:
        pass

    mrms = _mrms_files(project_dir) if project_dir else {}
    nex = _nexrad_volumes(project_dir) if project_dir else {}
    use = None
    picks = []
    if mrms:
        for t, p in times:
            f = _nearest([(k, v) for k, v in mrms.items()], t, tol_min=6)
            if f:
                picks.append((t, p, [f]))
        use = "mrms"
    if not picks and nex:
        for t, p in times:
            fl = [f for f in (_nearest(v, t) for v in nex.values()) if f]
            if fl:
                picks.append((t, p, fl))
        use = "nexrad"
    agent, notes = None, []
    if not picks and project_dir:
        from wtune.post import agent_obs
        from wtune.post.recipe import note
        ok, bad = agent_obs.usable(project_dir, "radar")
        notes = [note("convection", "agent dataset %s is not used: %s (wtune event obs-check)"
                      % (n, e)) for n, e in bad]
        for name, d, prov in ok:
            idx = agent_obs.snapshot_index(d, "radar")
            picks = [(t, p, [f]) for t, p in times for f in [agent_obs.nearest(idx, t, 8)] if f]
            if picks:
                use, agent = "agent", (name, prov)
                break
    if not picks:
        reason = radar_missing_reason(project_dir, run_dir, [t for t, _ in times], bbox)
        if nex or mrms:
            reason = ("radar files are present but none lies within +-8 min of a post-spin-up model "
                      "frame; fetch more: " + reason.split("run on a login node: ")[-1])
        return notes + [missing(i, "MRMS / NEXRAD composite reflectivity", reason) for i in RADAR_IDS]
    step = max(1, -(-len(picks) // MAX_RADAR_TIMES))
    picks = picks[::step]

    work = os.path.join(out_dir, "gridstat_radar")
    os.makedirs(work, exist_ok=True)
    per_ctc, per_nbr, rows_csv, best = [], [], [], None
    try:
        import wrf as wrf_mod
        sites = radar_composite.load_sites(project_dir) if use == "nexrad" else None
        poly = None
        for t, (path, ti), files in picks:
            tag = t.strftime("%Y%m%d%H%M")
            with netCDF4.Dataset(path) as ds:
                var = wrf_mod.getvar(ds, "mdbz", timeidx=ti)
                mlat, mlon = wrf_mod.latlon_coords(var, as_np=True)
                mdbz = np.asarray(np.ma.filled(np.ma.masked_invalid(np.asarray(var, "f8")), np.nan), "f8")
            if poly is None:
                poly = met_gridstat.write_interior_polygon(path, spec_bdy_width(run_dir),
                                                           os.path.join(work, "interior.poly"))
            fnc = met_gridstat.curvilinear_to_latlon(
                np.asarray(mlat, "f8"), np.asarray(mlon, "f8"),
                {"REFC_MODEL": (mdbz, "dBZ", "model max column reflectivity")}, t,
                os.path.join(work, "model_%s.nc" % tag), method="remapnn")
            griddes = fnc[:-3] + ".griddes"
            onc = os.path.join(work, "obs_%s.nc" % tag)
            if use == "nexrad":
                onc, n_rad = radar_composite.write_composite_latlon(
                    files, np.asarray(mlat, "f8"), np.asarray(mlon, "f8"), griddes, t, onc,
                    sites=sites)
            elif use == "agent":
                onc = agent_obs.regrid_snapshot(files[0], "refl", t, griddes, onc, "REFC_OBS",
                                                "dBZ", bbox, "remapnn")
                n_rad = 1
            else:
                onc, n_rad = radar_composite.mrms_on_latlon(files[0], griddes, t, onc), 1
            if onc is None:
                continue
            res = met_gridstat.run_verification(
                fnc, onc, poly, os.path.join(work, "gs_%s" % tag), "REFC_MODEL", "REFC_OBS",
                [">=%g" % x for x in DBZ_THRESHOLDS], widths=WIDTHS, to_grid="FCST",
                regrid_method="NEAREST", model="WRF", obtype="RADAR")
            ctc = met_gridstat.parse_ctc_counts(res["ctc"])
            nbr = met_gridstat.parse_nbrcnt_table(res["nbrcnt"])
            cts = met_gridstat.parse_cts_table(res["cts"])
            per_ctc.append(ctc)
            per_nbr.append(nbr)
            for r in cts:
                rows_csv.append([t.strftime("%Y-%m-%d %H:%M"), r["thresh"], r["pod"], r["far"], r["csi"],
                                 r["fbias"], r["total"], n_rad])
            glat, glon, gf, go, _a, _b = met_gridstat.read_pairs_fields(res["pairs_nc"])
            if not np.any(np.isfinite(gf) & np.isfinite(go)):
                continue                                  # no matched cells (obs entirely missing)
            score = int(np.sum(go >= 35))
            if best is None or score > best[0]:
                best = (score, t, glat, glon, gf, go, n_rad)
    except radar_composite.PyartUnavailable as e:
        return [missing(i, "NEXRAD composite reflectivity", str(e)) for i in RADAR_IDS]
    except (tools.ToolNotFoundError, tools.ToolError, met_gridstat.METError, ImportError, OSError,
            StopIteration, ValueError, KeyError) as e:
        return [missing(i, "MET Grid-Stat vs radar", "verification failed: %s" % e) for i in RADAR_IDS]
    if not per_ctc or best is None:
        return [missing(i, "radar composite reflectivity",
                        "no radar composite with cells inside the model interior could be built "
                        "(radar volumes unreadable, or the domain lies beyond every radar's range)")
                for i in RADAR_IDS]

    pooled_cts = _pool_counts(per_ctc)
    pooled_fss = _pool_fss(per_nbr)
    dx = [d["dx_km"] for d in case_info["domains"] if d["domain"] == domain]
    cell_km = dx[0] if dx and dx[0] else 3.0
    csv_ts = os.path.join(out_dir, "conv_radar_scores_by_time.csv")
    with open(csv_ts, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["valid_time_utc", "threshold_dBZ", "POD", "FAR", "CSI", "frequency_bias", "n_points", "n_radars"])
        w.writerows(rows_csv)
    csv_pool = os.path.join(out_dir, "conv_radar_scores_pooled.csv")
    with open(csv_pool, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold_dBZ", "POD", "FAR", "CSI", "frequency_bias", "n_points_all_times"])
        for r in pooled_cts:
            w.writerow([r["thresh"], r["pod"], r["far"], r["csi"], r["fbias"], r["total"]])
        w.writerow([])
        w.writerow(["threshold_dBZ", "width_points", "width_km", "FSS_pooled"])
        for r in pooled_fss:
            w.writerow([r["thresh"], r["width"], round(r["width"] * cell_km, 2), r["fss"]])

    if use == "agent":
        import textwrap
        obs_name = textwrap.shorten(agent[1].get("dataset") or agent[0], 48, placeholder="...")
        src = [agent_obs.source_tag(*agent)]
    else:
        obs_name = ("MRMS MergedReflectivityQCComposite" if use == "mrms"
                    else "NEXRAD Level II composite (Py-ART)")
        src = sources.source_tags(project_dir, [use])
    nt = len(per_ctc)
    common = dict(obs_sources=src, csv=[csv_ts, csv_pool], n_times=nt,
                  tool="MET grid_stat (both fields on one lat/lon grid, interior mask), pooled over %d time(s)" % nt)
    _s, tb, glat, glon, gf, go, n_rad = best
    out = list(notes)
    p = obs_figs.side_by_side(
        glat, glon, gf, go, ("%s d%02d model max column reflectivity" % (run_label, domain), obs_name + " (obs)"),
        "composite reflectivity %s UTC, matched points%s" % (
            tb.strftime("%Y-%m-%d %H:%M"), "" if use == "agent" else " (%d radar(s))" % n_rad),
        os.path.join(out_dir, "conv_radar_compare.png"), obs_figs.DBZ_LEVELS, cmap=reflectivity_cmap(),
        cbar_label="dBZ", extend="max")
    out.append(fig("conv_radar_compare", p, run_label, domain="d%02d" % domain,
                   valid_time=tb.strftime(WRF_TIME_FMT), extra=dict(common, tool=common["tool"])))
    p = obs_figs.scores_vs_threshold(pooled_cts, "Grid-Stat categorical scores vs %s, %d time(s) pooled" % (obs_name, nt),
                                     os.path.join(out_dir, "conv_radar_scores.png"), "reflectivity threshold (dBZ)")
    out.append(fig("conv_radar_scores", p, run_label, domain="d%02d" % domain,
                   valid_time="%s..%s" % (picks[0][0].strftime(WRF_TIME_FMT), picks[-1][0].strftime(WRF_TIME_FMT)),
                   extra=common))
    p = obs_figs.fss_vs_scale(pooled_fss, cell_km, "Grid-Stat FSS vs %s, %d time(s) pooled" % (obs_name, nt),
                              os.path.join(out_dir, "conv_fss.png"), " dBZ")
    out.append(fig("conv_fss", p, run_label, domain="d%02d" % domain,
                   valid_time="%s..%s" % (picks[0][0].strftime(WRF_TIME_FMT), picks[-1][0].strftime(WRF_TIME_FMT)),
                   extra=common))
    return out

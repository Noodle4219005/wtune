"""post/recipes/convection.py -- the "convection" figure set (decision 0057 section 2):
max column reflectivity (wrf-python `mdbz`), MCAPE/MCIN (`cape_2d`), 2-5 km updraft helicity
time-maximum (wrf-python `uhel`, bottom=2000/top=5000 m -- the standard Kain et al. 2008 UH
layer, and wrf-python's own default for this diagnostic, so no bespoke formula is needed here),
a representative-point Skew-T/hodograph (MetPy, via `wtune post sounding`'s own
post/sounding.py -- reused, not re-implemented), and (when radar obs are available) a
side-by-side reflectivity comparison plus MET Grid-Stat FSS. Radar obs (MRMS/NEXRAD composite)
are not fetched by this module (no downloader implemented this round -- decision 0057 section 3
lists only ERA5/HURDAT2/Stage IV as anonymously fetchable; MRMS needs a dated archive pull this
round does not implement), so that comparison and its FSS score are always reported missing with
that reason, never silently skipped.

The finest domain with cu_physics=0 (the convection-permitting domain, per detect_convection())
is used for every field here.
"""
from __future__ import annotations

import os

from post import tools
from post.extract import list_frames


def _finest_convective_domain(domains):
    hits = [d for d in domains if d["dx_km"] <= 4.0 and (d["cu_physics"] or 0) == 0]
    return max(hits, key=lambda d: d["domain"])["domain"] if hits else domains[-1]["domain"]


def _post_spinup_frames(run_dir, domain, spinup_hours):
    from datetime import datetime
    from post.caseinfo import WRF_TIME_FMT
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        return frames
    t0 = datetime.strptime(frames[0][0], WRF_TIME_FMT)
    return [(t, p) for t, p in frames
           if (datetime.strptime(t, WRF_TIME_FMT) - t0).total_seconds() / 3600.0 >= spinup_hours] \
        or frames


def _map_field(run_dir, domain, field, time_str, out_path, title, cmap="viridis",
               fill_label=""):
    import netCDF4
    wrf_mod = __import__("wrf")
    from post.recipes.mapfig import save_map
    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    with netCDF4.Dataset(path) as ds:
        if field == "mdbz":
            var = wrf_mod.getvar(ds, "mdbz", timeidx=0)
        elif field == "mcape":
            var = wrf_mod.getvar(ds, "cape_2d", timeidx=0)[0]
        elif field == "mcin":
            var = wrf_mod.getvar(ds, "cape_2d", timeidx=0)[1]
        elif field == "uhel":
            var = wrf_mod.getvar(ds, "uhel", timeidx=0, bottom=2000.0, top=5000.0)
        else:
            raise ValueError(field)
        lats, lons = wrf_mod.latlon_coords(var, as_np=True)
        crs = wrf_mod.get_cartopy(var=var)
        import numpy as np
        save_map(out_path, lats, lons, title, fill_field=np.asarray(var), fill_label=fill_label,
                fill_cmap=cmap, ds=ds, crs=crs)
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
    lats = lons = crs = ds_for_coast = None
    for _t, path in frames:
        with netCDF4.Dataset(path) as ds:
            var = wrf_mod.getvar(ds, "uhel", timeidx=0, bottom=2000.0, top=5000.0)
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
    path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
    with netCDF4.Dataset(path) as ds:
        mdbz = np.asarray(wrf_mod.getvar(ds, "mdbz", timeidx=0))
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        j, i = np.unravel_index(np.argmax(mdbz), mdbz.shape)
        return float(lat[j, i]), float(lon[j, i])


def figures(run_dirs, labels, out_dir, case_info, spinup_hours=6):
    from post.recipe import fig, missing

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
                          "%s d%02d max column reflectivity  %s" % (run_label, domain, last_time),
                          cmap="gist_ncar", fill_label="dBZ")
        out.append(fig("conv_max_reflectivity", path, run_label, domain="d%02d" % domain,
                       valid_time=last_time))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_max_reflectivity", "wrf-python", str(e)))

    try:
        for field, label in (("mcape", "MCAPE (J/kg)"), ("mcin", "MCIN (J/kg)")):
            path = _map_field(run_dir, domain, field, last_time,
                              os.path.join(out_dir, "conv_%s.png" % field),
                              "%s d%02d %s  %s" % (run_label, domain, label, last_time),
                              cmap="plasma", fill_label=label)
            out.append(fig("conv_%s" % field, path, run_label, domain="d%02d" % domain,
                           valid_time=last_time))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_mcape_mcin", "wrf-python", str(e)))

    try:
        uh_max, lats, lons, crs = _uh_time_max(run_dir, domain, post_spinup)
        from post.recipes.mapfig import save_map
        path = os.path.join(out_dir, "conv_uh25_max.png")
        save_map(path, lats, lons,
                "%s d%02d 2-5 km updraft helicity, time-max (spin-up excluded: %g h)"
                % (run_label, domain, spinup_hours),
                fill_field=uh_max, fill_label="UH 2-5km (m2/s2)", fill_cmap="Reds", crs=crs)
        out.append(fig("conv_uh25_max", path, run_label, domain="d%02d" % domain,
                       valid_time="%s..%s" % (post_spinup[0][0], post_spinup[-1][0])))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_uh25_max", "wrf-python", str(e)))

    try:
        lat, lon = _max_reflectivity_point(run_dir, domain, last_time)
        from post.sounding import make_sounding
        path = os.path.join(out_dir, "conv_skewt.png")
        make_sounding(run_dir, lat, lon, last_time, path, domain=domain)
        out.append(fig("conv_skewt", path, run_label, domain="d%02d" % domain,
                       valid_time=last_time,
                       extra={"point": "%.3f,%.3f (max-reflectivity column)" % (lat, lon)}))
    except (tools.ToolNotFoundError, ImportError) as e:
        out.append(missing("conv_skewt", "MetPy", str(e)))

    out.append(missing("conv_radar_compare", "MRMS/NEXRAD composite reflectivity",
                       "not fetched this round -- no dated-archive downloader implemented "
                       "(decision 0057 section 3 only anonymously fetches ERA5/HURDAT2/Stage IV)"))
    out.append(missing("conv_fss", "MET Grid-Stat FSS vs. radar obs",
                       "needs the radar obs comparison above, which is missing"))
    return out

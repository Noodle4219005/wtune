"""post/extract.py -- `wtune post extract RUN_DIR --vars ... --out FILE`: wrf-python
derived diagnostic fields as a point time-series CSV or a gridded NetCDF box. wrf-python
and netCDF4 are imported lazily (only inside functions), never at module import time, so
`wtune post inventory`/`compare` and `wtune --help` never need them installed.

Supported --vars:
  slp                sea-level pressure, hPa
  t2                 2 m temperature, K
  td2                2 m dewpoint, degC
  rh2                2 m relative humidity, %
  wspd10 / wdir10    10 m earth-relative wind speed (m/s) / direction (deg)
  rain_acc           accumulated precip, mm (RAINC + RAINNC)
  rain_rate          hourly precip rate, mm/h (finite difference of rain_acc between
                     consecutive extracted frames, divided by the frame spacing in hours;
                     the first frame has no rate and is written as an empty field)
  mdbz               max column reflectivity, dBZ
  mcape / mcin       2-D max CAPE / the CIN at the mcape level, J/kg (cape_2d)
  pw                 precipitable water, kg/m2 (wrf-python getvar 'pw')
  z, tc, ua, va, wspd  interpolated to each of --levels (hPa) via wrf.interplevel:
                     geopotential height (m), temperature (degC), earth-relative u/v
                     (m/s), wind speed (m/s)

--points "lat,lon[,name]" (repeatable) -> nearest grid point (wrf.ll_to_xy) time series,
written as CSV with a units header row. --box lat0,lat1,lon0,lon1 -> NetCDF subset with
zlib compression. Frames are extracted in a ProcessPoolExecutor (netCDF-C is not
thread-safe, so this must be processes, not threads).

Decision 0056 (2026-09-29 update, wrap established tools): wrf-python stays -- it is the
NCAR tool for these diagnostics, nothing in CDO/NCO computes slp/mcape/interpolated-level
fields from WRF's native eta levels. The --box spatial subsetting below stays self-written
too, for a narrower reason: it crops the wrf-python-DERIVED field already held in memory
(a numpy boolean mask over XLAT/XLONG), not a raw file variable -- there is no plain-file
subset step for CDO/NCO to replace it with, since the field being cropped does not exist
in any file until wrf-python computes it. `wtune post inventory` and `compare` do the
analogous file-level work (headers, netCDF diffing) through ncdump/cdo/diffwrf instead;
see their module docstrings.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"

SURFACE_FIELDS = ("slp", "t2", "td2", "rh2", "wspd10", "wdir10", "rain_acc", "rain_rate",
                  "mdbz", "mcape", "mcin", "pw")
LEVEL_FIELDS = ("z", "tc", "ua", "va", "wspd")
UNITS = dict(slp="hPa", t2="K", td2="degC", rh2="%", wspd10="m/s", wdir10="deg",
             rain_acc="mm", rain_rate="mm/h", mdbz="dBZ", mcape="J/kg", mcin="J/kg",
             pw="kg/m2", z="m", tc="degC", ua="m/s", va="m/s", wspd="m/s")


class ExtractError(RuntimeError):
    pass


def _wrf():
    import wrf
    return wrf


def _netcdf4():
    import netCDF4
    return netCDF4


def list_frames(run_dir, domain=1):
    """Sorted [(time_str, path), ...] for wrfout_d0<domain>_* in run_dir."""
    pattern = os.path.join(run_dir, "wrfout_d%02d_*" % domain)
    out = []
    rx = re.compile(r"wrfout_d\d+_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)$")
    for path in sorted(glob.glob(pattern)):
        m = rx.search(os.path.basename(path))
        if m:
            out.append((m.group(1), path))
    return out


def _surface_field(ds, wrf_mod, name, timeidx):
    if name == "slp":
        return wrf_mod.getvar(ds, "slp", timeidx=timeidx, units="hPa")
    if name == "t2":
        return wrf_mod.getvar(ds, "T2", timeidx=timeidx)
    if name == "td2":
        return wrf_mod.getvar(ds, "td2", timeidx=timeidx, units="degC")
    if name == "rh2":
        return wrf_mod.getvar(ds, "rh2", timeidx=timeidx)
    if name in ("wspd10", "wdir10"):
        wspd_wdir = wrf_mod.getvar(ds, "uvmet10_wspd_wdir", timeidx=timeidx, units="m s-1")
        return wspd_wdir[0] if name == "wspd10" else wspd_wdir[1]
    if name == "rain_acc":
        rainc = wrf_mod.getvar(ds, "RAINC", timeidx=timeidx)
        rainnc = wrf_mod.getvar(ds, "RAINNC", timeidx=timeidx)
        total = rainc + rainnc
        total.attrs = dict(rainc.attrs)    # xarray < 2025.11 drops attrs in arithmetic; get_cartopy needs 'projection'
        return total
    if name == "mdbz":
        return wrf_mod.getvar(ds, "mdbz", timeidx=timeidx)
    if name in ("mcape", "mcin"):
        cape2d = wrf_mod.getvar(ds, "cape_2d", timeidx=timeidx)
        return cape2d[0] if name == "mcape" else cape2d[1]
    if name == "pw":
        return wrf_mod.getvar(ds, "pw", timeidx=timeidx)
    raise ExtractError("unknown surface field %r" % name)


def _level_base_field(ds, wrf_mod, name, timeidx):
    if name == "z":
        return wrf_mod.getvar(ds, "z", timeidx=timeidx, units="m")
    if name == "tc":
        return wrf_mod.getvar(ds, "tc", timeidx=timeidx)
    if name == "ua":
        return wrf_mod.getvar(ds, "ua", timeidx=timeidx, units="m s-1")
    if name == "va":
        return wrf_mod.getvar(ds, "va", timeidx=timeidx, units="m s-1")
    if name == "wspd":
        return wrf_mod.getvar(ds, "wspd_wdir", timeidx=timeidx, units="m s-1")[0]
    raise ExtractError("unknown level field %r" % name)


def extract_frame(task):
    """Runs in a worker process (module-level for picklability). task = (path, time_str,
    surface_vars, level_vars, levels_hpa). Returns {"time": ..., fields...} with each
    surface field a 2-D ndarray and each level field a dict {level_hpa: 2-D ndarray}."""
    path, time_str, surface_vars, level_vars, levels_hpa = task
    import numpy as np
    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    out = {"time": time_str}
    with netcdf4.Dataset(path) as ds:
        for name in surface_vars:
            if name == "rain_rate":
                continue           # derived after all frames are gathered
            out[name] = np.asarray(_surface_field(ds, wrf_mod, name, 0))
        if level_vars and levels_hpa:
            pressure = wrf_mod.getvar(ds, "pressure", timeidx=0)
            for name in level_vars:
                base = _level_base_field(ds, wrf_mod, name, 0)
                out[name] = {lev: np.asarray(wrf_mod.interplevel(base, pressure, lev))
                            for lev in levels_hpa}
    return out


def _add_rain_rate(frames, time_fmt=WRF_TIME_FMT):
    import numpy as np
    from datetime import datetime
    if len(frames) < 2 or "rain_acc" not in frames[0]:
        for f in frames:
            f["rain_rate"] = None
        return
    frames[0]["rain_rate"] = None
    for i in range(1, len(frames)):
        dt_h = (datetime.strptime(frames[i]["time"], time_fmt)
               - datetime.strptime(frames[i - 1]["time"], time_fmt)).total_seconds() / 3600.0
        frames[i]["rain_rate"] = ((frames[i]["rain_acc"] - frames[i - 1]["rain_acc"]) / dt_h
                                  if dt_h > 0 else np.zeros_like(frames[i]["rain_acc"]))


def extract(run_dir, variables, out_path, domain=1, times=None, levels_hpa=None,
            points=None, box=None, workers=1):
    """variables: list of field names (SURFACE_FIELDS + LEVEL_FIELDS). times: optional list
    of WRF time strings to restrict to (default: every frame). points: [(lat, lon, name)];
    box: (lat0, lat1, lon0, lon1). Exactly one of points/box must be given; the output
    format follows --out's extension for box (.nc) and is always CSV for points."""
    if bool(points) == bool(box):
        raise ExtractError("extract() needs exactly one of points or box")

    surface_vars = [v for v in variables if v in SURFACE_FIELDS]
    level_vars = [v for v in variables if v in LEVEL_FIELDS]
    unknown = [v for v in variables if v not in SURFACE_FIELDS and v not in LEVEL_FIELDS]
    if unknown:
        raise ExtractError("unknown --vars %s" % unknown)
    if level_vars and not levels_hpa:
        raise ExtractError("--levels required for %s" % level_vars)
    needed_surface = [v for v in surface_vars if v != "rain_rate"]
    if "rain_rate" in surface_vars and "rain_acc" not in needed_surface:
        needed_surface.append("rain_acc")

    all_frames = list_frames(run_dir, domain=domain)
    if times:
        wanted = set(times)
        all_frames = [(t, p) for t, p in all_frames if t in wanted]
    if not all_frames:
        raise ExtractError("no wrfout_d%02d_* frames found in %s" % (domain, run_dir))

    tasks = [(path, t, needed_surface, level_vars, levels_hpa or []) for t, path in all_frames]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(extract_frame, tasks))
    else:
        frames = [extract_frame(t) for t in tasks]
    frames.sort(key=lambda f: f["time"])

    if "rain_rate" in surface_vars:
        _add_rain_rate(frames)

    if points:
        _write_points_csv(frames, points, variables, levels_hpa or [], out_path,
                          all_frames[0][1])
        return dict(kind="points", out=out_path, n_frames=len(frames), n_points=len(points))
    _write_box_netcdf(frames, box, variables, levels_hpa or [], out_path, all_frames[0][1])
    return dict(kind="box", out=out_path, n_frames=len(frames), box=box)


def _nearest_indices(sample_path, points):
    """[(j, i, lat, lon, name), ...] for each requested point via wrf.ll_to_xy."""
    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    out = []
    with netcdf4.Dataset(sample_path) as ds:
        for lat, lon, name in points:
            xy = wrf_mod.ll_to_xy(ds, lat, lon)
            i, j = int(xy[0]), int(xy[1])
            out.append((j, i, lat, lon, name))
    return out


def _field_value(frame, var, j, i, levels_hpa):
    if var in LEVEL_FIELDS:
        return {lev: float(frame[var][lev][j, i]) for lev in levels_hpa}
    val = frame.get(var)
    if val is None:
        return None
    return float(val[j, i])


def _write_points_csv(frames, points, variables, levels_hpa, out_path, sample_path):
    idx = _nearest_indices(sample_path, points)
    columns = []
    for var in variables:
        if var in LEVEL_FIELDS:
            for lev in levels_hpa:
                columns.append("%s_%dhPa" % (var, int(lev)))
        else:
            columns.append(var)

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        for j, i, lat, lon, name in idx:
            w.writerow(["# point", name, "lat=%.4f" % lat, "lon=%.4f" % lon,
                       "grid_j=%d" % j, "grid_i=%d" % i])
            w.writerow(["time"] + ["%s[%s]" % (c, UNITS.get(c.split("_")[0], "")) for c in columns])
            for f in frames:
                row = [f["time"]]
                for var in variables:
                    if var in LEVEL_FIELDS:
                        for lev in levels_hpa:
                            row.append(f[var][lev][j, i])
                    else:
                        v = f.get(var)
                        row.append("" if v is None else float(v[j, i]))
                w.writerow(row)


def _write_box_netcdf(frames, box, variables, levels_hpa, out_path, sample_path):
    netcdf4 = _netcdf4()
    import numpy as np
    lat0, lat1, lon0, lon1 = box
    with netcdf4.Dataset(sample_path) as ds:
        lat2d = np.asarray(ds.variables["XLAT"][0])
        lon2d = np.asarray(ds.variables["XLONG"][0])
    mask = (lat2d >= min(lat0, lat1)) & (lat2d <= max(lat0, lat1)) \
        & (lon2d >= min(lon0, lon1)) & (lon2d <= max(lon0, lon1))
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ExtractError("--box %s does not intersect the domain" % (box,))
    j0, j1, i0, i1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

    with netcdf4.Dataset(out_path, "w") as out:
        out.createDimension("time", len(frames))
        out.createDimension("south_north", j1 - j0)
        out.createDimension("west_east", i1 - i0)
        tvar = out.createVariable("time", str, ("time",))
        for k, f in enumerate(frames):
            tvar[k] = f["time"]
        xlat_var = out.createVariable("XLAT", "f4", ("south_north", "west_east"))
        xlat_var[:] = lat2d[j0:j1, i0:i1]
        xlat_var.standard_name = "latitude"
        xlat_var.units = "degrees_north"
        xlong_var = out.createVariable("XLONG", "f4", ("south_north", "west_east"))
        xlong_var[:] = lon2d[j0:j1, i0:i1]
        xlong_var.standard_name = "longitude"
        xlong_var.units = "degrees_east"
        # CF "coordinates" attribute on every data variable, naming XLAT/XLONG as its 2-D
        # curvilinear grid coordinates. Without this (and the standard_name/units above), cdo
        # cannot tell XLAT/XLONG apart from an ordinary (south_north, west_east) data field --
        # `cdo sub` on two otherwise-identical extractions was found to difference XLAT/XLONG
        # right along with the requested variable, zeroing them out (post/changemap.py's own
        # change-map figures came out blank until this was added; caught on the scc P7
        # validation run, 2026-09-29).
        for var in variables:
            if var in LEVEL_FIELDS:
                for lev in levels_hpa:
                    name = "%s_%dhPa" % (var, int(lev))
                    v = out.createVariable(name, "f4", ("time", "south_north", "west_east"),
                                           zlib=True, complevel=4)
                    v.units = UNITS.get(var, "")
                    v.coordinates = "XLAT XLONG"
                    for k, f in enumerate(frames):
                        v[k] = f[var][lev][j0:j1, i0:i1]
            else:
                v = out.createVariable(var, "f4", ("time", "south_north", "west_east"),
                                       zlib=True, complevel=4)
                v.units = UNITS.get(var, "")
                v.coordinates = "XLAT XLONG"
                for k, f in enumerate(frames):
                    val = f.get(var)
                    v[k] = (np.full((j1 - j0, i1 - i0), np.nan, dtype="f4") if val is None
                            else val[j0:j1, i0:i1])

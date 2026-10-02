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


_FRAME_RE = re.compile(r"wrfout_d(\d+)_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)$")
_SEGMENT_RE = re.compile(r"seg_(\d+)(_rescue)?$")


def segment_dirs(run_dir):
    """The segment run dirs of a segmented `wtune run --out` dir (OUT/seg_<k>, plus a CFL-rescue
    re-run OUT/seg_<k>_rescue right after its seg_<k>), in segment order -- [] when run_dir holds
    wrfout_d* itself (a flat run dir) or has no seg_<int> subdirs."""
    if glob.glob(os.path.join(run_dir, "wrfout_d*")):
        return []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return []
    segs = []
    for name in names:
        m = _SEGMENT_RE.match(name)
        path = os.path.join(run_dir, name)
        if m and os.path.isdir(path):
            segs.append(((int(m.group(1)), bool(m.group(2))), path))
    return [path for _key, path in sorted(segs)]


_TIMES_CACHE = {}       # {(path, mtime_ns, size): [valid_time_str, ...] or None}


def _decode_times(var):
    """WRF's Times(Time, DateStrLen) char variable -> [time_str, ...]; also accepts a 1-D
    string variable (a converted file, or netCDF4 auto-chartostring via _Encoding)."""
    import numpy as np
    raw = np.ma.getdata(var[:])
    if raw.ndim == 2 and raw.dtype.kind == "S":
        rows = [b"".join(r.tolist()) for r in raw]
    else:
        rows = list(np.atleast_1d(raw).tolist())
    return [(r.decode("ascii", "replace") if isinstance(r, bytes) else str(r)).strip("\x00 ")
            for r in rows]


def file_times(path):
    """The valid times a wrfout file holds, from its own Times variable (WRF writes
    frames_per_outfile > 1 history times into one file) -- [time_str, ...] in file order, the
    list index being the file's own timeidx. None when the file is not readable netCDF or has
    no usable Times (the caller then falls back to the one file-name time). Reads only the
    Times variable; cached in-process per (path, mtime, size), so re-listing a run dir after a
    segment rewrote a file re-reads it, and an unchanged 400-file climate run is opened once."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    if key in _TIMES_CACHE:
        return _TIMES_CACHE[key]
    times = None
    if st.st_size > 0:
        try:
            netcdf4 = _netcdf4()
            with netcdf4.Dataset(path) as ds:
                if "Times" in ds.variables:
                    times = [t for t in _decode_times(ds.variables["Times"]) if t] or None
        except Exception:       # not netCDF / truncated / netCDF4 missing -> file-name time
            times = None
    _TIMES_CACHE[key] = times
    return times


def _file_entries(run_dir):
    """[(domain, file_name_time, path), ...] for every wrfout_dNN_TIME file, segment order
    (segment_dirs) then file-name order -- the order "later wins" is decided in."""
    out = []
    for d in segment_dirs(run_dir) or [run_dir]:
        for path in sorted(glob.glob(os.path.join(d, "wrfout_d*_*"))):
            m = _FRAME_RE.search(os.path.basename(path))
            if m:
                out.append((int(m.group(1)), m.group(2), path))
    return out


def file_index(run_dir):
    """{(domain, file_name_time): path} -- one entry per wrfout FILE, keyed by the time in its
    name (a later segment's file of the same name wins). For whole-file tools (diffwrf in
    post/compare.py) that compare files rather than individual valid times."""
    return {(dom, t): path for dom, t, path in _file_entries(run_dir)}


def frame_refs(run_dir):
    """{(domain, valid_time): (path, timeidx)} for every history time in run_dir -- or, for a
    segmented run dir (segment_dirs), in all its segments. Valid times come from each file's
    own Times variable (file_times), so a file holding frames_per_outfile > 1 times yields one
    entry per time; an unreadable file (or one without Times) contributes its file-name time
    at timeidx 0, exactly the pre-Times behaviour. A valid time present in two files (the shared
    boundary frame of two segments, a rescued segment's re-run) maps to the LATER file in
    segment-then-name order."""
    out = {}
    for dom, name_time, path in _file_entries(run_dir):
        times = file_times(path) or [name_time]
        for idx, t in enumerate(times):
            out[(dom, t)] = (path, idx)
    return out


def frame_index(run_dir):
    """{(domain, valid_time): path} -- frame_refs without the timeidx (the containing file of
    each valid time). Identical to the old file-name index for single-frame files."""
    return {k: path for k, (path, _idx) in frame_refs(run_dir).items()}


def list_frame_refs(run_dir, domain=1):
    """Sorted [(valid_time, path, timeidx), ...] for domain <domain> (see frame_refs)."""
    return sorted((t, path, idx) for (d, t), (path, idx) in frame_refs(run_dir).items()
                  if d == domain)


def list_frames(run_dir, domain=1):
    """Sorted [(valid_time, path), ...] for domain <domain> (merged across segments for a
    segmented run dir, see frame_refs). With multi-frame files several valid times share one
    path -- a caller that reads a time-varying field must use list_frame_refs/frame_ref for
    the timeidx instead of reading index 0."""
    return [(t, path) for t, path, _idx in list_frame_refs(run_dir, domain)]


def frame_ref(run_dir, domain, time_str):
    """(path, timeidx) of valid time time_str on domain <domain> (segment-aware, Times-based,
    see frame_refs); (the flat run_dir/wrfout_dNN_<time_str> path, 0) when no file holds that
    time, so a caller's own exists()/open error still names the expected file."""
    flat = os.path.join(run_dir, "wrfout_d%02d_%s" % (int(domain), time_str))
    ref = frame_refs(run_dir).get((int(domain), time_str))
    if ref is not None:
        return ref
    return flat, 0


def frame_path(run_dir, domain, time_str):
    """Path of the file holding valid time time_str (frame_ref without the timeidx)."""
    return frame_ref(run_dir, domain, time_str)[0]


class MissingFields(ExtractError):
    """A wrfout history stream lacks variables/global attributes a diagnostic needs -- the
    usual cause is an iofields file that reduced the history stream (climate runs)."""

    def __init__(self, names, what=None):
        self.names = list(names)
        super().__init__("history stream lacks %s%s (iofields-reduced output?)"
                         % (", ".join(self.names), (" for " + what) if what else ""))


def require_fields(ds, variables=(), attributes=(), what=None):
    """Raise MissingFields naming every one of `variables` (netCDF variables) and `attributes`
    (global attributes) that ds lacks -- checked up front, so a recipe reports what the
    reduced history stream is missing instead of a bare AttributeError/KeyError."""
    missing = [v for v in variables if v not in ds.variables]
    attrs = set(ds.ncattrs()) if hasattr(ds, "ncattrs") else set()
    missing += [a for a in attributes if a not in attrs]
    if missing:
        raise MissingFields(missing, what)


def is_missing_field_error(exc):
    """True for the exceptions a reduced (iofields) history stream produces when a diagnostic
    reads a variable/attribute that is not there: MissingFields, a netCDF4/wrf-python KeyError,
    wrf.getvar's ValueError "'X' is not a valid variable name", netCDF's AttributeError
    "NetCDF: Attribute not found", wrf-python's RuntimeError "'X' not found in NetCDF file"."""
    if isinstance(exc, (MissingFields, KeyError)):
        return True
    text = str(exc)
    return ("is not a valid variable name" in text or "Attribute not found" in text
            or "not found in NetCDF" in text)


def missing_reason(exc):
    """A human missing-entry reason for an exception raised while reading a wrfout frame:
    MissingFields keeps its own text; a netCDF/wrf-python KeyError/AttributeError/ValueError
    that names a variable or attribute becomes "history stream lacks <name> ..." rather than
    a raw exception repr."""
    if isinstance(exc, MissingFields):
        return str(exc)
    text = str(exc)
    m = (re.search(r"'([A-Za-z0-9_]+)' is not a valid variable name", text)   # wrf.getvar
         or re.search(r"'([A-Za-z0-9_]+)' not found", text))
    if m is None and isinstance(exc, KeyError) and exc.args:                # ds.variables[...]
        m = re.match(r"([A-Za-z0-9_]+)$", str(exc.args[0]))
    if m:
        return "history stream lacks %s (iofields-reduced output?)" % m.group(1)
    if "Attribute not found" in text:
        return ("history stream lacks a global attribute this diagnostic needs "
                "(iofields-reduced output?)")
    return "%s: %s" % (type(exc).__name__, text)


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
    surface_vars, level_vars, levels_hpa[, timeidx]) -- timeidx (default 0) is the valid
    time's index inside a multi-frame file (frame_ref). Returns {"time": ..., fields...} with each
    surface field a 2-D ndarray and each level field a dict {level_hpa: 2-D ndarray}."""
    path, time_str, surface_vars, level_vars, levels_hpa = task[:5]
    timeidx = task[5] if len(task) > 5 else 0
    import numpy as np
    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    out = {"time": time_str}
    with netcdf4.Dataset(path) as ds:
        for name in surface_vars:
            if name == "rain_rate":
                continue           # derived after all frames are gathered
            out[name] = np.asarray(_surface_field(ds, wrf_mod, name, timeidx))
        if level_vars and levels_hpa:
            pressure = wrf_mod.getvar(ds, "pressure", timeidx=timeidx)
            for name in level_vars:
                base = _level_base_field(ds, wrf_mod, name, timeidx)
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

    all_frames = list_frame_refs(run_dir, domain=domain)
    if times:
        wanted = set(times)
        all_frames = [f for f in all_frames if f[0] in wanted]
    if not all_frames:
        raise ExtractError("no wrfout_d%02d_* frames found in %s" % (domain, run_dir))

    tasks = [(path, t, needed_surface, level_vars, levels_hpa or [], idx)
             for t, path, idx in all_frames]
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

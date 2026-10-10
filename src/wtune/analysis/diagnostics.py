#!/usr/bin/env python3
"""analysis/diagnostics.py -- wrf-python derived TC diagnostic fields, written as a
CF-ish NetCDF that TempestExtremes' DetectNodes/StitchNodes can read directly.

Why wrf-python instead of the vendored formula in predictor/wrf_slp_vendored.py: the
advisor asked for an established meteorological tool, not more ad-hoc AI code. wrf.getvar
implements the same NCL/WRF-Fortran diagnostics (slp, uvmet10 destaggered + earth-rotated,
pw, td2) used across the WRF community, so a reviewer can check this module's arithmetic
against wrf-python's own documented behaviour instead of against nothing.

Grid separability (mandatory correctness check, never assumed):
  A Mercator or regular lat-lon WRF grid (MAP_PROJ 0 or 3) is a cylindrical projection
  applied uniformly along each grid axis: XLAT varies only with the south_north index,
  XLONG only with west_east. That makes the grid SEPARABLE -- it can be described with
  1-D lat(ny) and lon(nx) coordinate vectors, which is the cheap, exact TempestExtremes
  input (DetectNodes reads it directly with --latname/--lonname, no connectivity file).

  A Lambert conformal (MAP_PROJ 1) or polar stereographic (MAP_PROJ 2) grid is NOT
  separable: XLAT and XLONG both vary with both indices. Assuming separability there
  would silently mis-locate every node by grid-relative amounts that grow toward the
  domain edges -- exactly the kind of ad-hoc mistake this module exists to avoid. This
  module NUMERICALLY CHECKS separability on the actual XLAT/XLONG (never trusts MAP_PROJ
  alone, since a "Mercator" run can still be rotated/shifted enough to matter) and only
  takes the 1-D shortcut when the deviation is below SEPARABLE_TOL_DEG (1e-4 deg, ~11 m).

  For a non-separable grid this module writes the fields flattened onto an unstructured
  `ncol` dimension (1-D lat(ncol), lon(ncol), fields(time, ncol)), which is what
  analysis.tc_track.detect_and_stitch feeds to TempestExtremes' GenerateConnectivityFile
  to build a Delaunay-type point-cloud connectivity file, then to DetectNodes via
  --in_connect. This was verified empirically against tempest-extremes 2.4.2
  (conda-forge nompi build 103): GenerateConnectivityFile builds valid connectivity from
  --in_data + --latname/--lonname alone (curvilinear or scattered points, no separate
  mesh file needed), and DetectNodes/StitchNodes correctly track a moving feature through
  it -- see analysis/tc_track.py's module docstring for the exact commands and the
  synthetic-grid test that exercises this path (test_analysis_tc_track.py). If
  GenerateConnectivityFile itself fails for a given file, analysis.tc_track raises
  NotImplementedError naming the projection rather than mis-gridding silently.
"""
from __future__ import annotations

import glob as globmod
import os
import sys
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
SEPARABLE_TOL_DEG = 1e-4          # ~11 m at the equator; see module docstring

MAP_PROJ_NAMES = {
    0: "idealized (no projection)",
    1: "Lambert conformal",
    2: "polar stereographic",
    3: "Mercator",
    6: "lat-lon (incl. global)",
}

# Diagnostics this module knows how to extract. Each maps to either a single wrf.getvar
# call ("simple") or a small composite built from one ("composite"), so wrfout_to_tc_fields
# can request a subset without importing wrf-python's internals directly.
DEFAULT_VARIABLES = ("slp", "wspd10", "u10", "v10", "pw", "td2")


class DiagnosticsError(RuntimeError):
    """Raised for anything this module refuses to do silently (bad grid, missing var)."""


def _netcdf4():
    import netCDF4
    return netCDF4


def _wrf():
    import wrf
    return wrf


def expand_wrfout(patterns):
    """Globs/paths -> unique file list: each glob's hits sorted, the caller's pattern order kept
    (an explicit segment-ordered list from post.extract must not be re-sorted as strings, or
    seg_10 lands before seg_2 and the wrong file wins a valid time two segments share).
    Raises when nothing matches.

    Mirrors predictor/verify_obs.py's expand_wrfout (not imported: that module is owned
    by a different part of the codebase and this package must not depend on its private
    helpers changing shape under it -- only its documented, stable parse_hurdat2 and
    great_circle_km are reused, per the task's own instruction, from analysis.tc_track).
    """
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    files = []
    for pattern in patterns:
        pattern = str(pattern)
        hits = sorted(globmod.glob(pattern))
        if not hits and os.path.exists(pattern):
            hits = [pattern]
        files.extend(hits)
    files = list(dict.fromkeys(files))
    if not files:
        raise DiagnosticsError(f"no wrfout files matched: {list(patterns)}")
    return files


def _frame_times(ds):
    netcdf4 = _netcdf4()
    times = netcdf4.chartostring(ds.variables["Times"][:])
    return [datetime.strptime(str(t).strip(), WRF_TIME_FMT) for t in np.atleast_1d(times)]


def check_grid_separable(lat2d, lon2d, tol=SEPARABLE_TOL_DEG):
    """Numerically test whether (lat2d, lon2d) is an axis-separable grid.

    Returns (separable, max_lat_deviation_deg, max_lon_deviation_deg). "Deviation" is how
    far each row of lat / column of lon strays from its own first row/column -- zero for
    a perfectly separable (Mercator/lat-lon) grid, and growing toward the domain edges
    for Lambert/polar grids.
    """
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)
    max_lat_dev = float(np.max(np.abs(lat2d - lat2d[:, [0]])))
    max_lon_dev = float(np.max(np.abs(lon2d - lon2d[[0], :])))
    separable = max_lat_dev < tol and max_lon_dev < tol
    return separable, max_lat_dev, max_lon_dev


def _get_field(ds, wrf_mod, name, timeidx):
    """One diagnostic field, wrf-python units as documented in its product table.

    slp: hPa. wspd10/u10/v10: m/s (uvmet10 = earth-rotated 10 m wind, not grid-relative).
    pw: kg m-2 (precipitable water). td2: degC (2 m dewpoint).
    """
    if name == "slp":
        return wrf_mod.getvar(ds, "slp", timeidx=timeidx, units="hPa")
    if name in ("wspd10", "u10", "v10"):
        uvmet10 = wrf_mod.getvar(ds, "uvmet10", timeidx=timeidx, units="m s-1")
        # uvmet10 dims: (u_v, [Time,] south_north, west_east)
        u = uvmet10[0]
        v = uvmet10[1]
        if name == "u10":
            return u
        if name == "v10":
            return v
        return np.hypot(np.asarray(u), np.asarray(v))
    if name == "pw":
        return wrf_mod.getvar(ds, "pw", timeidx=timeidx)
    if name == "td2":
        return wrf_mod.getvar(ds, "td2", timeidx=timeidx, units="degC")
    raise DiagnosticsError(f"unknown diagnostic {name!r}; known: {DEFAULT_VARIABLES}")


def wrfout_to_tc_fields(wrfout_paths, out_nc, variables=DEFAULT_VARIABLES):
    """Extract per-time TC diagnostic fields from wrfout files via wrf-python and write
    them as a NetCDF TempestExtremes can read (see module docstring for the grid layout).

    wrfout_paths: glob(s)/path(s), expanded and read in time order (like
                  predictor/verify_obs.py's station_series/storm_centres).
    out_nc:       output path. Overwritten if it exists.
    variables:    subset of DEFAULT_VARIABLES to compute. "slp" is required for tracking
                  (DetectNodes --searchbymin slp) and is always included even if omitted.

    Returns a metadata dict: {out_nc, n_times, grid_type ("separable"|"ncol"),
    separable, max_lat_dev_deg, max_lon_dev_deg, map_proj, map_proj_name, ny, nx,
    source_files}.
    """
    wrf_mod = _wrf()
    netcdf4 = _netcdf4()
    variables = tuple(dict.fromkeys(("slp",) + tuple(variables)))   # dedupe, slp first
    files = expand_wrfout(wrfout_paths)

    by_time = {}            # datetime -> {var: 2-D ndarray}
    lat2d = lon2d = None
    map_proj = None

    for path in files:
        with netcdf4.Dataset(path) as ds:
            if map_proj is None:
                map_proj = int(ds.getncattr("MAP_PROJ")) if "MAP_PROJ" in ds.ncattrs() else None
            if lat2d is None:
                lat_dv = wrf_mod.getvar(ds, "lat", timeidx=0)
                lon_dv = wrf_mod.getvar(ds, "lon", timeidx=0)
                lat2d = np.asarray(lat_dv, dtype=float)
                lon2d = np.asarray(lon_dv, dtype=float)
            times = _frame_times(ds)
            nt = len(times)
            for var in variables:
                field_all = _get_field(ds, wrf_mod, var, timeidx=wrf_mod.ALL_TIMES)
                arr = np.asarray(field_all, dtype=float)
                if nt == 1 and arr.ndim == 2:
                    arr = arr[None, ...]
                if arr.shape[0] != nt:
                    raise DiagnosticsError(
                        f"{path}: {var} has {arr.shape[0]} time slices, Times has {nt}")
                for t, when in enumerate(times):
                    slot = by_time.setdefault(when, {})
                    slot[var] = arr[t]

    if not by_time:
        raise DiagnosticsError(f"no time frames extracted from {files}")
    if lat2d is None:
        raise DiagnosticsError(f"no XLAT/XLONG found in {files}")

    ny, nx = lat2d.shape
    separable, max_lat_dev, max_lon_dev = check_grid_separable(lat2d, lon2d)
    map_proj_name = MAP_PROJ_NAMES.get(map_proj, f"unknown ({map_proj!r})")
    grid_type = "separable" if separable else "ncol"

    sorted_times = sorted(by_time)
    n_times = len(sorted_times)

    out_nc = str(out_nc)
    if os.path.exists(out_nc):
        os.remove(out_nc)
    with netcdf4.Dataset(out_nc, "w") as out:
        out.createDimension("time", None)
        tvar = out.createVariable("time", "f8", ("time",))
        tvar.units = "hours since 1970-01-01 00:00:00"
        tvar.calendar = "standard"
        epoch = datetime(1970, 1, 1)
        tvar[:] = [(t - epoch).total_seconds() / 3600.0 for t in sorted_times]

        if separable:
            out.createDimension("lat", ny)
            out.createDimension("lon", nx)
            latv = out.createVariable("lat", "f8", ("lat",))
            lonv = out.createVariable("lon", "f8", ("lon",))
            latv.units = "degrees_north"
            lonv.units = "degrees_east"
            latv[:] = lat2d[:, 0]
            lonv[:] = lon2d[0, :]
            field_dims = ("time", "lat", "lon")
        else:
            ncol = ny * nx
            out.createDimension("ncol", ncol)
            latv = out.createVariable("lat", "f8", ("ncol",))
            lonv = out.createVariable("lon", "f8", ("ncol",))
            latv.units = "degrees_north"
            lonv.units = "degrees_east"
            latv[:] = lat2d.reshape(-1)
            lonv[:] = lon2d.reshape(-1)
            field_dims = ("time", "ncol")

        for var in variables:
            v = out.createVariable(var, "f8", field_dims)
            for t, when in enumerate(sorted_times):
                arr = by_time[when][var]
                v[t] = arr.reshape(-1) if not separable else arr

        out.title = "WTune TC diagnostic fields (wrf-python derived)"
        out.source_files = ";".join(os.path.abspath(f) for f in files)
        out.grid_type = grid_type
        out.map_proj = map_proj if map_proj is not None else -1
        out.map_proj_name = map_proj_name
        out.ny = ny
        out.nx = nx
        out.history = (
            f"analysis.diagnostics.wrfout_to_tc_fields; wrf-python {wrf_mod.__version__}; "
            f"{n_times} frames from {len(files)} file(s); "
            f"grid_separable={separable} (max_lat_dev_deg={max_lat_dev:.6g}, "
            f"max_lon_dev_deg={max_lon_dev:.6g}, tol={SEPARABLE_TOL_DEG:g})"
        )

    return {
        "out_nc": os.path.abspath(out_nc),
        "n_times": n_times,
        "grid_type": grid_type,
        "separable": separable,
        "max_lat_dev_deg": max_lat_dev,
        "max_lon_dev_deg": max_lon_dev,
        "map_proj": map_proj,
        "map_proj_name": map_proj_name,
        "ny": ny,
        "nx": nx,
        "source_files": [os.path.abspath(f) for f in files],
    }

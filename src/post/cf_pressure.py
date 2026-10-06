"""post/cf_pressure.py -- one wrfout frame -> a CF-1.8 NetCDF on pressure levels and a regular
lon-lat grid (`wtune post cf3d`; also a data product of `wtune post recipe`) for Met.3D and
ParaView, which cannot read WRF's staggered, terrain-following, projected grid.

wrf-python does the physics (destaggering, uvmet, tk, rh, z, interplevel) and the projection;
pyproj + scipy.ndimage.map_coordinates do the lookup. Here: the target grid, each target point's
index in WRF's projected x/y (bilinear there is exact for fields linear in x/y), the CF metadata.
No extrapolation: levels outside a column's pressure range, points outside the native grid or in
the lateral boundary zone are missing (NaN in memory, FILL_VALUE on disk).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"

PRESSURE_LEVELS_HPA = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450,
                       400, 350, 300, 250, 200, 150, 100)

# NC_FILL_FLOAT, not NaN: Met.3D replaces values EQUAL to _FillValue, which NaN never passes
# (ParaView needs "Replace Fill Value With Nan").
FILL_VALUE = 9.96921e36

_KM_PER_DEG = 111.2
_MIN_SPACING_DEG = 0.01
_COMMENT_BELOW_GROUND = ("levels below the lowest model mass level (below ground) are "
                         "_FillValue; no extrapolation")

# (name, units, standard_name or None, long_name) of every 3-D output field, in file order.
# Q* have no standard_name: WRF's are per kg of DRY air, CF's mass_fraction_* per kg of moist air.
FIELDS_3D = (
    ("air_temperature", "K", "air_temperature", "air temperature (wrf-python 'tk')"),
    ("eastward_wind", "m s-1", "eastward_wind",
     "earth-relative eastward wind (wrf-python 'uvmet', rotated from grid-relative U)"),
    ("northward_wind", "m s-1", "northward_wind",
     "earth-relative northward wind (wrf-python 'uvmet', rotated from grid-relative V)"),
    ("upward_air_velocity", "m s-1", "upward_air_velocity",
     "vertical velocity on mass levels (wrf-python 'wa')"),
    ("geopotential_height", "m", "geopotential_height", "geopotential height (wrf-python 'z')"),
    ("relative_humidity", "%", "relative_humidity", "relative humidity (wrf-python 'rh')"),
    ("QCLOUD", "kg kg-1", None, "cloud water mixing ratio"),
    ("QICE", "kg kg-1", None, "cloud ice mixing ratio"),
    ("CLOUD_TOTAL", "kg kg-1", None, "cloud water + cloud ice mixing ratio (QCLOUD+QICE)"),
)


class CFPressureError(RuntimeError):
    pass


def output_name(domain: int, time_str: str) -> str:
    """wrf_d0N_pl_YYYYMMDD_HHMM.nc -- no ':' (Met.3D/ParaView file-series patterns and Windows
    both fail on WRF's own wrfout_d01_2005-08-29_00:00:00 style)."""
    t = datetime.strptime(time_str, WRF_TIME_FMT)
    return "wrf_d%02d_pl_%s.nc" % (int(domain), t.strftime("%Y%m%d_%H%M"))


def grid_spacing_deg(dx_m: float) -> float:
    """Target lon-lat spacing ~ the native dx: round(dx_km / 111.2, 2) (12 km -> 0.11 deg).
    Floored at 0.01 deg, since a sub-km nest would otherwise round to a zero spacing."""
    return max(round(dx_m / 1000.0 / _KM_PER_DEG, 2), _MIN_SPACING_DEG)


def wrf_crs(ncfile) -> Any:
    """pyproj.CRS of the wrfout's native grid, from wrf-python's own projection object (WRF's
    6370 km sphere is part of its proj4 string)."""
    import pyproj
    import wrf
    proj = wrf.getproj(**wrf.get_proj_params(ncfile))
    return pyproj.CRS(proj.proj4())


def _to_native(crs, lon, lat):
    import numpy as np
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def native_affine(lat2d, lon2d, crs) -> tuple[float, float, float, float]:
    """(x0, sx, y0, sy) with projected x = x0 + i*sx, y = y0 + j*sy (i: west_east index, j:
    south_north index), least-squares fitted to every cell's XLAT/XLONG. Raises CFPressureError
    when the fit residual exceeds 1% of a grid spacing: the projection does not describe the
    grid (e.g. a rotated pole, MAP_PROJ=6), and regridding with it would displace every field."""
    import numpy as np
    # fitted from XLAT/XLONG: a nest's CEN_LAT/CEN_LON is not a reliable anchor across WRF versions
    x, y = _to_native(crs, lon2d, lat2d)
    ny, nx = x.shape
    jj, ii = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
    sx, x0 = np.polyfit(ii.ravel(), x.ravel(), 1)
    sy, y0 = np.polyfit(jj.ravel(), y.ravel(), 1)
    resid = max(np.nanmax(np.abs(x - (x0 + ii * sx))), np.nanmax(np.abs(y - (y0 + jj * sy))))
    spacing = min(abs(sx), abs(sy))
    if not np.isfinite(resid) or spacing == 0 or resid > 0.01 * spacing:
        raise CFPressureError(
            "native grid is not regular in the wrfout's projection (%s): max residual %.3g m vs "
            "grid spacing %.3g m -- MAP_PROJ not representable by wrf-python's proj4 (rotated "
            "pole?)" % (getattr(crs, "srs", crs), resid, spacing))
    return float(x0), float(sx), float(y0), float(sy)


def _interior(a, bdy_width):
    ny, nx = a.shape
    b = int(bdy_width)
    if ny - 2 * b < 1 or nx - 2 * b < 1:
        raise CFPressureError("domain %dx%d has no interior left after excluding %d boundary "
                              "cells" % (nx, ny, b))
    return a[b:ny - b, b:nx - b]


def _unwrap_lon(lon):
    """Longitudes of a dateline-crossing domain shifted to one continuous range (CF allows
    lon > 180); unchanged otherwise."""
    import numpy as np
    lon = np.asarray(lon, dtype=float)
    if lon.max() - lon.min() > 180.0:
        return np.where(lon < 0.0, lon + 360.0, lon)
    return lon


def target_grid(lat2d, lon2d, bdy_width: int, dlon: float) -> tuple:
    """(lat1d, lon1d): ascending, regular (spacing dlon in both directions) and snapped to
    multiples of dlon, covering the lat/lon bounding box of the domain INTERIOR. Built from
    integer counts so float drift never adds or drops an end point. Corners of the box that
    fall outside the interior are left to regrid_indices() to mark missing."""
    import numpy as np
    lat_in = _interior(np.asarray(lat2d, dtype=float), bdy_width)
    lon_in = _unwrap_lon(_interior(np.asarray(lon2d, dtype=float), bdy_width))

    def axis(lo, hi):
        k0 = int(np.ceil(lo / dlon - 1e-6))
        k1 = int(np.floor(hi / dlon + 1e-6))
        if k1 < k0:
            raise CFPressureError("domain interior is narrower than one %g deg target cell"
                                  % dlon)
        return np.round(np.arange(k0, k1 + 1) * dlon, 6)

    return axis(lat_in.min(), lat_in.max()), axis(lon_in.min(), lon_in.max())


def regrid_indices(lat1d, lon1d, crs, affine, shape, bdy_width: int) -> tuple:
    """(jj, ii), each (nlat, nlon): every target point's fractional native (south_north,
    west_east) index. NaN where that index falls outside the interior box [b, n-1-b] -- i.e.
    outside the native grid or inside the lateral boundary zone."""
    import numpy as np
    x0, sx, y0, sy = affine
    lon2, lat2 = np.meshgrid(np.asarray(lon1d, dtype=float), np.asarray(lat1d, dtype=float))
    x, y = _to_native(crs, lon2, lat2)
    ii = (x - x0) / sx
    jj = (y - y0) / sy
    ny, nx = shape
    b = int(bdy_width)
    ok = (np.isfinite(ii) & np.isfinite(jj) & (ii >= b) & (ii <= nx - 1 - b)
          & (jj >= b) & (jj <= ny - 1 - b))
    return np.where(ok, jj, np.nan), np.where(ok, ii, np.nan)


def regrid(field, jj, ii) -> Any:
    """field (..., ny, nx) -> (..., nlat, nlon) float32, bilinear (map_coordinates order=1) at
    the fractional indices jj/ii, NaN where they are NaN. A NaN among a target point's four
    native neighbours (below ground after vertical interpolation) makes it NaN too -- intended:
    no value is ever built from a partly-missing stencil."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    field = np.asarray(field, dtype=float)
    lead = field.shape[:-2]
    flat = field.reshape((-1,) + field.shape[-2:])
    ok = np.isfinite(jj) & np.isfinite(ii)
    coords = np.vstack([jj[ok], ii[ok]])
    out = np.full((flat.shape[0],) + jj.shape, np.nan, dtype=np.float32)
    for k in range(flat.shape[0]):
        out[k][ok] = map_coordinates(flat[k], coords, order=1, mode="constant", cval=np.nan)
    return out.reshape(lead + jj.shape)


def _wrf_version(title):
    """' OUTPUT FROM WRF V4.5.1 MODEL' -> 'V4.5.1'; the raw TITLE when it has no V<version>."""
    for tok in (title or "").split():
        if tok[:1] == "V" and tok[1:2].isdigit():
            return tok
    return (title or "unknown").strip()


# what read_frame needs from the history stream: an iofields-reduced stream (climate runs)
# often drops QCLOUD/W and the P_TOP variable -- checked up front so the frame is reported as
# "history stream lacks ..." instead of netCDF's bare AttributeError on P_TOP
REQUIRED_VARIABLES = ("Times", "P", "PB", "T", "PH", "PHB", "QVAPOR", "U", "V", "W", "QCLOUD",
                      "HGT", "XLAT", "XLONG")


def check_history_fields(nc) -> None:
    """Raise post.extract.MissingFields naming every REQUIRED_VARIABLES entry and P_TOP
    (variable or global attribute) that nc lacks."""
    from post.extract import MissingFields
    missing = [] if ("P_TOP" in nc.variables or "P_TOP" in nc.ncattrs()) else ["P_TOP"]
    missing += [v for v in REQUIRED_VARIABLES if v not in nc.variables]
    if missing:
        raise MissingFields(missing)


def _levels_above_top(nc, levels_hpa, timeidx: int) -> tuple[float, list[float], list[float]]:
    """(P_TOP Pa, levels above the model top, dropped levels): a level at or above P_TOP has no
    model data in any column, so it is dropped rather than written all-missing."""
    p_top = (float(nc.variables["P_TOP"][timeidx]) if "P_TOP" in nc.variables
             else float(nc.getncattr("P_TOP")))
    levels = [float(lev) for lev in levels_hpa if lev * 100.0 > p_top]
    dropped = [float(lev) for lev in levels_hpa if lev * 100.0 <= p_top]
    if not levels:
        raise CFPressureError("every requested level is above the model top (P_TOP=%g hPa)"
                              % (p_top / 100.0))
    return p_top, levels, dropped


def _interpolated_fields(nc, levels: list[float], timeidx: int) -> tuple[dict, bool]:
    """({name: (nlev, ny, nx) float32, NaN = missing}, has_qice) via wrf-python interplevel."""
    import numpy as np
    import wrf

    pressure = wrf.getvar(nc, "pressure", timeidx=timeidx, meta=False)

    def on_levels(base):
        # interplevel never extrapolates: a level outside the column's mass-level pressure
        # range comes back as `missing`; filled() also covers a masked-array return
        out = wrf.interplevel(np.asarray(base), pressure, levels, missing=np.nan, meta=False)
        out = np.ma.filled(np.ma.asarray(out, dtype=float), np.nan)
        return out.reshape((len(levels),) + out.shape[-2:]).astype(np.float32)

    uvmet = wrf.getvar(nc, "uvmet", timeidx=timeidx, units="m s-1", meta=False)
    fields = {
        "air_temperature": on_levels(wrf.getvar(nc, "tk", timeidx=timeidx, meta=False)),
        "eastward_wind": on_levels(uvmet[0]),
        "northward_wind": on_levels(uvmet[1]),
        "upward_air_velocity": on_levels(wrf.getvar(nc, "wa", timeidx=timeidx, meta=False)),
        "geopotential_height": on_levels(wrf.getvar(nc, "z", timeidx=timeidx, units="m",
                                                    meta=False)),
        "relative_humidity": on_levels(wrf.getvar(nc, "rh", timeidx=timeidx, meta=False)),
        "QCLOUD": on_levels(wrf.getvar(nc, "QCLOUD", timeidx=timeidx, meta=False)),
    }
    has_qice = "QICE" in nc.variables
    if has_qice:
        fields["QICE"] = on_levels(wrf.getvar(nc, "QICE", timeidx=timeidx, meta=False))
        fields["CLOUD_TOTAL"] = fields["QCLOUD"] + fields["QICE"]
    else:
        # warm-rain microphysics (e.g. Kessler) carries no ice: the total is QCLOUD alone
        fields["CLOUD_TOTAL"] = fields["QCLOUD"].copy()
    return fields, has_qice


def read_frame(wrfout_path: str, levels_hpa=PRESSURE_LEVELS_HPA, timeidx: int = 0) -> dict:
    """wrf-python diagnostics of time index `timeidx` of wrfout_path (a frames_per_outfile > 1 file
    holds several, see post.extract.frame_ref) on levels_hpa below the model top. Returns dict(levels,
    dropped_levels, fields, hgt, lat2d, lon2d, crs, dx, p_top, sim_start, valid_time, wrf_version,
    has_qice)."""
    import netCDF4
    import numpy as np

    with netCDF4.Dataset(wrfout_path) as nc:
        check_history_fields(nc)
        p_top, levels, dropped = _levels_above_top(nc, levels_hpa, timeidx)
        fields, has_qice = _interpolated_fields(nc, levels, timeidx)
        times = netCDF4.chartostring(nc.variables["Times"][:])
        frame = dict(
            levels=levels, dropped_levels=dropped, fields=fields,
            hgt=np.asarray(nc.variables["HGT"][timeidx], dtype=float),
            lat2d=np.asarray(nc.variables["XLAT"][timeidx], dtype=float),
            lon2d=np.asarray(nc.variables["XLONG"][timeidx], dtype=float),
            crs=wrf_crs(nc), dx=float(nc.getncattr("DX")), p_top=p_top,
            sim_start=str(nc.getncattr("SIMULATION_START_DATE")),
            valid_time=str(np.atleast_1d(times)[timeidx]),
            wrf_version=_wrf_version(getattr(nc, "TITLE", "")), has_qice=has_qice)
    return frame


def _masked(a):
    import numpy as np
    return np.ma.masked_invalid(np.asarray(a, dtype=np.float32))


def _native_history(wrfout_path):
    """The ADIOS2 BP directory a cached wrfout was converted from, else "netcdf"."""
    from post.extract import bp_origin
    return bp_origin(wrfout_path) or "netcdf"


def _write_coordinates(nc, frame: dict, lat1d, lon1d, t0: datetime, tv: datetime) -> None:
    """time, forecast_reference_time, isobaric, lat, lon variables."""
    import numpy as np
    time_units = "hours since %s" % t0.strftime("%Y-%m-%d %H:%M:%S")
    v = nc.createVariable("time", "f8", ("time",))
    v.setncatts(dict(units=time_units, calendar="standard", standard_name="time",
                     long_name="valid time", axis="T"))
    v[0] = (tv - t0).total_seconds() / 3600.0
    # the init time as its own variable, so it is explicit rather than implied by `units`
    v = nc.createVariable("forecast_reference_time", "f8", ())
    v.setncatts(dict(units=time_units, calendar="standard",
                     standard_name="forecast_reference_time",
                     long_name="simulation start (SIMULATION_START_DATE)"))
    v.assignValue(0.0)

    v = nc.createVariable("isobaric", "f4", ("isobaric",))
    v.setncatts(dict(units="hPa", long_name="Isobaric surface", standard_name="air_pressure",
                     positive="down", axis="Z"))
    v[:] = np.asarray(frame["levels"], dtype=np.float32)
    v = nc.createVariable("lat", "f8", ("lat",))
    v.setncatts(dict(units="degrees_north", standard_name="latitude", long_name="latitude",
                     axis="Y"))
    v[:] = lat1d
    v = nc.createVariable("lon", "f8", ("lon",))
    v.setncatts(dict(units="degrees_east", standard_name="longitude", long_name="longitude",
                     axis="X"))
    v[:] = lon1d


def _write_fields(nc, frame: dict, regridded: dict, hgt_ll) -> None:
    """The 3-D FIELDS_3D variables (those present in `regridded`) and surface_altitude."""
    import numpy as np
    data_kw = dict(zlib=True, complevel=4, fill_value=np.float32(FILL_VALUE))
    for name, units, std, long_name in FIELDS_3D:
        if name not in regridded:
            continue                       # QICE absent (warm-rain microphysics)
        v = nc.createVariable(name, "f4", ("time", "isobaric", "lat", "lon"), **data_kw)
        atts = dict(units=units, long_name=long_name, missing_value=np.float32(FILL_VALUE),
                    comment=_COMMENT_BELOW_GROUND)
        if std:
            atts["standard_name"] = std
        if name == "CLOUD_TOTAL" and not frame["has_qice"]:
            atts["long_name"] = ("cloud water mixing ratio (QCLOUD only; no QICE in this "
                                 "wrfout, warm-rain microphysics)")
            atts["comment"] = ("CLOUD_TOTAL = QCLOUD: this wrfout has no QICE. "
                               + _COMMENT_BELOW_GROUND)
        v.setncatts(atts)
        v[0] = _masked(regridded[name])

    v = nc.createVariable("surface_altitude", "f4", ("lat", "lon"), **data_kw)
    v.setncatts(dict(units="m", standard_name="surface_altitude",
                     long_name="model terrain height (HGT)",
                     missing_value=np.float32(FILL_VALUE)))
    v[:] = _masked(hgt_ll)


def _global_attributes(frame: dict, domain: int, bdy_width: int, dlon: float, wrfout_path: str,
                       command: str) -> dict:
    import numpy as np
    return {
        "Conventions": "CF-1.8",
        "title": "WRF d%02d pressure-level fields on a regular lon-lat grid" % int(domain),
        "source": "WRF %s" % frame["wrf_version"],
        "history": "%s: %s" % (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                               command),
        "wtune_wrfout": os.path.abspath(wrfout_path),
        # ADIOS2 BP history: wtune_wrfout is post.extract's netCDF cache copy of this BP dir
        "wtune_wrfout_native": _native_history(wrfout_path),
        "wtune_domain": np.int32(domain),
        "wtune_pressure_levels_hpa": np.asarray(frame["levels"], dtype=np.float32),
        "wtune_dropped_levels_hpa": (np.asarray(frame["dropped_levels"], dtype=np.float32)
                                     if frame["dropped_levels"] else "none"),
        "wtune_model_top_hpa": np.float32(frame["p_top"] / 100.0),
        "wtune_vertical_interpolation": (
            "wrf-python interplevel on mass-level pressure, no extrapolation; below-ground "
            "points are _FillValue"),
        "wtune_regrid_method": (
            "bilinear in native WRF projected x/y (pyproj + scipy.ndimage.map_coordinates "
            "order=1); target points outside the native grid or inside the lateral "
            "boundary zone are _FillValue"),
        "wtune_grid_spacing_deg": np.float64(dlon),
        "wtune_boundary_zone_cells": np.int32(bdy_width),
        "wtune_native_projection": frame["crs"].srs,       # the proj4 wrf-python gave
        "wtune_qice": "present" if frame["has_qice"] else "absent (CLOUD_TOTAL = QCLOUD)",
    }


def write_cf(out_path: str, frame: dict, lat1d, lon1d, regridded: dict, hgt_ll, bdy_width: int,
             dlon: float, wrfout_path: str, domain: int, command: str | None = None) -> None:
    """Write the CF-1.8 file: dims (time, isobaric, lat, lon), one time record, the FIELDS_3D
    variables, 2-D surface_altitude, wtune_* provenance as global attributes. regridded:
    {name: (nlev, nlat, nlon)}, NaN = missing; hgt_ll: (nlat, nlon)."""
    import netCDF4

    t0 = datetime.strptime(frame["sim_start"], WRF_TIME_FMT)
    tv = datetime.strptime(frame["valid_time"], WRF_TIME_FMT)
    command = command or " ".join(sys.argv)
    with netCDF4.Dataset(out_path, "w", format="NETCDF4") as nc:
        nc.createDimension("time", None)
        nc.createDimension("isobaric", len(frame["levels"]))
        nc.createDimension("lat", len(lat1d))
        nc.createDimension("lon", len(lon1d))
        _write_coordinates(nc, frame, lat1d, lon1d, t0, tv)
        _write_fields(nc, frame, regridded, hgt_ll)
        nc.setncatts(_global_attributes(frame, domain, bdy_width, dlon, wrfout_path, command))


def convert_frame(wrfout_path: str, out_path: str, domain: int, bdy_width: int,
                  levels_hpa=PRESSURE_LEVELS_HPA, command: str | None = None,
                  timeidx: int = 0) -> dict:
    """Time index `timeidx` of wrfout_path -> out_path. Returns dict(path, valid_time, levels,
    dropped_levels, nlat, nlon, dlon, has_qice, size_bytes)."""
    frame = read_frame(wrfout_path, levels_hpa=levels_hpa, timeidx=timeidx)
    affine = native_affine(frame["lat2d"], frame["lon2d"], frame["crs"])
    dlon = grid_spacing_deg(frame["dx"])
    lat1d, lon1d = target_grid(frame["lat2d"], frame["lon2d"], bdy_width, dlon)
    jj, ii = regrid_indices(lat1d, lon1d, frame["crs"], affine, frame["lat2d"].shape, bdy_width)
    regridded = {name: regrid(a, jj, ii) for name, a in frame["fields"].items()}
    hgt_ll = regrid(frame["hgt"], jj, ii)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_cf(out_path, frame, lat1d, lon1d, regridded, hgt_ll, bdy_width, dlon, wrfout_path,
             domain, command=command)
    return dict(path=out_path, valid_time=frame["valid_time"], levels=frame["levels"],
                dropped_levels=frame["dropped_levels"], nlat=len(lat1d), nlon=len(lon1d),
                dlon=dlon, has_qice=frame["has_qice"], size_bytes=os.path.getsize(out_path))


def select_frames(run_dir: str, domain: int, every=None, times=None) -> list[tuple]:
    """[(time_str, path, timeidx), ...] of one domain at the --every/--times selection (post/timeseries
    .select_times, on that domain's own frames), or the LAST frame only when neither is given.
    Shared with post/vapor_bundle.py so both data products pick the same times. [] when the
    domain has no frames."""
    from post.extract import list_frame_refs
    from post.timeseries import select_times
    # segment-aware and Times-based (a multi-frame wrfout gives one entry per valid time)
    frames = {t: (path, idx) for t, path, idx in list_frame_refs(run_dir, domain=domain)}
    all_times = sorted(frames)
    if not all_times:
        return []
    selected = (select_times(all_times, every=every, times=times) if (every or times)
                else all_times[-1:])
    return [(t,) + frames[t] for t in selected]


def export(run_dir: str, out_root: str, domains, bdy_width: int, every=None, times=None,
           command: str | None = None) -> list[dict]:
    """Writes <out_root>/cf3d/d0N/wrf_d0N_pl_YYYYMMDD_HHMM.nc for each domain in `domains` at
    the --every/--times selection (post/timeseries.select_times, on that domain's own frames),
    or the LAST frame only when neither is given. Returns one dict per frame: convert_frame()'s
    plus "domain", or dict(domain, valid_time, error) for a frame that failed -- one bad frame
    never stops the rest."""
    results = []
    for domain in domains:
        selected = select_frames(run_dir, domain, every=every, times=times)
        if not selected:
            results.append(dict(domain=domain, valid_time=None,
                                error="no wrfout_d%02d_* frames in %s" % (domain, run_dir)))
            continue
        for t, path, timeidx in selected:
            out_path = os.path.join(out_root, "cf3d", "d%02d" % domain, output_name(domain, t))
            try:
                r = convert_frame(path, out_path, domain, bdy_width, command=command,
                                  timeidx=timeidx)
                r["domain"] = domain
                results.append(r)
            except Exception as e:                      # noqa: BLE001 -- wrf-python's own
                # DiagnosticError, a truncated file, a variable missing from a reduced history
                # stream: one frame's failure is recorded, never a failed recipe report
                from post.extract import missing_reason
                results.append(dict(domain=domain, valid_time=t, error=missing_reason(e)))
    return results

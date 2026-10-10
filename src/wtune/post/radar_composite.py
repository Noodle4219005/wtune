"""post/radar_composite.py -- composite (column-maximum) reflectivity from NEXRAD Level II
volumes, using Py-ART's own gridder (`pyart.io.read_nexrad_archive`, `pyart.map.grid_from_radars`,
Barnes2 weighting): the volumes of every radar available near the valid time are gridded together
onto one Cartesian grid (2 km, 0-10 km in 1 km steps) centred on the model domain, the composite
is the maximum over height, then `cdo remapnn` puts it on the model's lat/lon verification grid
(post/met_gridstat.py curvilinear_to_latlon, same griddes) as a CF NetCDF MET can read.

Coverage: a cell counts as "no echo" (-20 dBZ) when it lies within RANGE_KM of a radar that has a
volume, and as missing (fill) beyond every radar's range -- geometry glue, Py-ART returns masked
for both. Py-ART is not in envs/wtune yet: install with `uv pip install --no-deps --target DIR
arm_pyart lazy_loader` and export WTUNE_PYART_PATH=DIR (or add arm_pyart to envs/wtune/pixi.toml).
"""
from __future__ import annotations

import math
import os
import sys

RANGE_KM = 230.0
NO_ECHO_DBZ = -20.0
FILL = 9.96921e36
DX_M = 2000.0
NZ, ZTOP_M = 11, 10000.0


class PyartUnavailable(RuntimeError):
    pass


def import_pyart():
    extra = os.environ.get("WTUNE_PYART_PATH")
    if extra and extra not in sys.path:
        sys.path.append(extra)
    try:
        import pyart
        return pyart
    except ImportError as e:
        raise PyartUnavailable(
            "Py-ART is not installed in this interpreter (%s); add `arm_pyart` to "
            "envs/wtune/pixi.toml, or `uv pip install --no-deps --target DIR arm_pyart lazy_loader` "
            "and export WTUNE_PYART_PATH=DIR" % e)


def _haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 12742.0 * np.arcsin(np.sqrt(a))


def load_sites(project_dir):
    """{ICAO: (lat, lon)} from the HOMR radar list saved by the nexrad/stage4 fetch modules
    (nexrad_sites.txt), {} when absent."""
    from wtune.post import sources
    from wtune.post.fetch.nexrad import parse_homr
    for name in ("nexrad", "stage4"):
        path = os.path.join(sources.dataset_dir(project_dir, name), "nexrad_sites.txt") if project_dir else None
        if path and os.path.isfile(path):
            with open(path) as fh:
                return {icao: (la, lo) for icao, la, lo in parse_homr(fh.read())}
    return {}


def composite_curvilinear(volume_paths, bbox, sites=None):
    """(lat2d, lon2d, composite dBZ 2-D with NaN outside every radar range, n_radars) for the
    volumes in `volume_paths` over bbox=(lat0, lat1, lon0, lon1). Pre-2008 Archive II volumes
    carry no site position (Py-ART reads latitude = longitude = 0), so the position comes from
    `sites` ({ICAO: (lat, lon)}, ICAO = first four characters of the file name); a volume that
    has neither is skipped."""
    import numpy as np
    pyart = import_pyart()
    radars = []
    for p in volume_paths:
        try:
            r = pyart.io.read_nexrad_archive(p)
        except Exception:  # noqa: BLE001 -- one unreadable volume must not lose the others
            continue
        if "reflectivity" not in r.fields:
            continue
        if abs(float(r.latitude["data"][0])) < 1e-6 and abs(float(r.longitude["data"][0])) < 1e-6:
            site = (sites or {}).get(os.path.basename(p)[:4].upper())
            if site is None:
                continue
            r.latitude["data"] = np.array([site[0]])
            r.longitude["data"] = np.array([site[1]])
        radars.append(r)
    if not radars:
        return None
    clat, clon = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
    hy = (bbox[1] - bbox[0]) / 2 * 111.2e3 + 20e3
    hx = (bbox[3] - bbox[2]) / 2 * 111.2e3 * math.cos(math.radians(clat)) + 20e3
    ny, nx = int(2 * hy / DX_M) + 1, int(2 * hx / DX_M) + 1
    grid = pyart.map.grid_from_radars(
        radars, grid_shape=(NZ, ny, nx),
        grid_limits=((0.0, ZTOP_M), (-hy, hy), (-hx, hx)), grid_origin=(clat, clon),
        fields=["reflectivity"], weighting_function="Barnes2", gridding_algo="map_gates_to_grid")
    refl = np.ma.filled(np.ma.max(grid.fields["reflectivity"]["data"], axis=0), np.nan).astype("f8")
    lon, lat = grid.get_point_longitude_latitude(level=0)
    lat, lon = np.asarray(lat, "f8"), np.asarray(lon, "f8")
    dmin = np.full(lat.shape, np.inf)
    for r in radars:
        dmin = np.minimum(dmin, _haversine_km(lat, lon, float(r.latitude["data"][0]),
                                               float(r.longitude["data"][0])))
    inside = dmin <= RANGE_KM
    out = np.where(inside & ~np.isfinite(refl), NO_ECHO_DBZ, refl)
    out = np.where(inside, out, np.nan)
    return lat, lon, out, len(radars)


def write_composite_latlon(volume_paths, model_lat, model_lon, griddes, valid, out_nc,
                           var="REFC_OBS", sites=None):
    """Composite from the volumes -> lat/lon CF NetCDF on `griddes` (cdo remapnn). Returns
    (out_nc, n_radars) or (None, 0) when no volume could be read."""
    import netCDF4
    import numpy as np
    from wtune.post import tools
    from wtune.post.met_gridstat import FILL as MET_FILL, _tag_axes
    bbox = (float(model_lat.min()), float(model_lat.max()), float(model_lon.min()),
            float(model_lon.max()))
    got = composite_curvilinear(volume_paths, bbox, sites)
    if got is None:
        return None, 0
    lat, lon, refl, n = got
    cdo = tools.find_on_path("cdo")
    stem = out_nc[:-3]
    curv = stem + ".curv.nc"
    ny, nx = lat.shape
    with netCDF4.Dataset(curv, "w") as dst:
        dst.createDimension("time", 1)
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)
        t = dst.createVariable("time", "f8", ("time",))
        t.units, t.calendar = "hours since 1970-01-01 00:00:00", "standard"
        t[:] = netCDF4.date2num(valid, t.units)
        for name, units, std, data in (("lat", "degrees_north", "latitude", lat),
                                       ("lon", "degrees_east", "longitude", lon)):
            v = dst.createVariable(name, "f8", ("y", "x"))
            v[:], v.units, v.standard_name = data, units, std
        v = dst.createVariable(var, "f4", ("time", "y", "x"), fill_value=np.float32(MET_FILL))
        v[:] = np.where(np.isfinite(refl), refl, MET_FILL)[None]
        v.units, v.coordinates = "dBZ", "lon lat"
    tools.run([cdo, "-s", "remapnn,%s" % griddes, curv, out_nc])
    os.remove(curv)
    _tag_axes(out_nc)
    return out_nc, n


def mrms_on_latlon(grib2, griddes, valid, out_nc, var="REFC_OBS"):
    """MRMS composite reflectivity GRIB2 -> lat/lon CF NetCDF on `griddes` with cdo (remapnn), then
    rewritten as a clean (time, lat, lon) file (cdo keeps a singleton level axis and a
    'day as %Y%m%d.%f' time axis MET's CF reader does not take). MRMS codes -99 = no echo (kept
    as NO_ECHO_DBZ) and -999 = no coverage (missing)."""
    import netCDF4
    import numpy as np
    from wtune.post import tools
    from wtune.post.met_gridstat import FILL as MET_FILL, _tag_axes
    cdo = tools.find_on_path("cdo")
    tmp = out_nc[:-3] + ".cdo.nc"
    tools.run([cdo, "-s", "-f", "nc4", "-setmissval,-9999", "-remapnn,%s" % griddes, grib2, tmp])
    with netCDF4.Dataset(tmp) as ds:
        name = [v for v in ds.variables if ds.variables[v].ndim >= 3][0]
        arr = np.ma.filled(np.ma.squeeze(ds.variables[name][:]), np.nan).astype("f8")
        lat = np.asarray(ds.variables["lat"][:], "f8")
        lon = np.asarray(ds.variables["lon"][:], "f8")
    os.remove(tmp)
    arr = np.where(arr <= -998, np.nan, np.where(arr <= -98, NO_ECHO_DBZ, arr))
    with netCDF4.Dataset(out_nc, "w") as dst:
        dst.createDimension("time", 1)
        dst.createDimension("lat", len(lat))
        dst.createDimension("lon", len(lon))
        t = dst.createVariable("time", "f8", ("time",))
        t.units, t.calendar = "hours since 1970-01-01 00:00:00", "standard"
        t[:] = netCDF4.date2num(valid, t.units)
        for nm, units, std, data in (("lat", "degrees_north", "latitude", lat),
                                     ("lon", "degrees_east", "longitude", lon)):
            v = dst.createVariable(nm, "f8", (nm,))
            v[:], v.units, v.standard_name = data, units, std
        v = dst.createVariable(var, "f4", ("time", "lat", "lon"), fill_value=np.float32(MET_FILL))
        v[:] = np.where(np.isfinite(arr), arr, MET_FILL)[None]
        v.units = "dBZ"
    _tag_axes(out_nc)
    return out_nc

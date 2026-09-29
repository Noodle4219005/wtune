"""post/met_gridstat.py -- MET Grid-Stat for the synoptic model-vs-ERA5 500 hPa height figure
(decision 0057 section 3). Same recipe as post/met_tc.py::regrid_for_tc_rmw: wrf-python puts the
field on a pressure level, `cdo remapbil` moves it from the curvilinear WRF grid to a regular
lat/lon grid (native point count over the native lon/lat extent) MET can read, and MET then
does the verification itself: `regrid = { to_grid = OBS; method = BILIN; width = 2; }` onto the
ERA5 0.25 deg grid, masked to a polygon around the model domain's interior (spec_bdy_width cells
excluded), CNT line (RMSE, ME = bias) in the .stat file, matched-pair DIFF in a NetCDF the
figure fills. Both inputs are CF NetCDF (Grid-Stat needs no pressure axis, unlike tc_rmw, so no
GRIB1 conversion).
"""
from __future__ import annotations

import os

from post import tools
from post.met_tc import METError, _write_config

FCST_NAME = "Z500"
OBS_NAME = "Z500"


def write_era5_z500_nc(lat, lon, z, valid, out_nc):
    """ERA5 z500 (m) on its regular 1-D lat/lon grid -> CF NetCDF for MET (OBS)."""
    return _write_regular_nc(lat, lon, z, valid, out_nc)


def _write_regular_nc(lat, lon, z, valid, out_nc):
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(out_nc, "w") as dst:
        dst.createDimension("time", 1)
        dst.createDimension("lat", len(lat))
        dst.createDimension("lon", len(lon))
        tvar = dst.createVariable("time", "f8", ("time",))
        dst.Conventions = "CF-1.6"
        tvar.units, tvar.calendar = "hours since 1970-01-01 00:00:00", "standard"
        tvar.standard_name, tvar.axis = "time", "T"   # MET's CF reader needs the axis tags
        tvar[:] = netCDF4.date2num(valid, tvar.units)
        for name, units, std, axis, data in (("lat", "degrees_north", "latitude", "Y", lat),
                                             ("lon", "degrees_east", "longitude", "X", lon)):
            var = dst.createVariable(name, "f8", (name,))
            var[:] = data
            var.units, var.standard_name, var.axis = units, std, axis
        var = dst.createVariable(OBS_NAME, "f4", ("time", "lat", "lon"),
                                 fill_value=np.float32(9.96921e36))
        var[:] = np.asarray(z, dtype="f4")[None]
        var.units, var.long_name = "m", "500 hPa geopotential height"
    return out_nc


def model_z500_on_latlon(wrfout_path, out_nc):
    """wrfout frame -> regular lat/lon CF NetCDF of z500 (m): wrf-python z on 500 hPa, then
    `cdo remapbil` onto a lat/lon grid with the native point count over the native extent
    (cdo cannot remap a raw wrfout: it aborts on the eta coordinate ZNU)."""
    import netCDF4
    import numpy as np
    import wrf
    cdo = tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or "
                             "envs/wtune/pixi.toml provides it.")
    with netCDF4.Dataset(wrfout_path) as src:
        lat = np.asarray(wrf.getvar(src, "lat"))
        lon = np.asarray(wrf.getvar(src, "lon"))
        z500 = np.asarray(wrf.interplevel(wrf.getvar(src, "z", units="m"),
                                          wrf.getvar(src, "pressure"), 500))
        valid = wrf.extract_times(src, 0)
    ny, nx = lat.shape
    stem = out_nc[:-3] if out_nc.endswith(".nc") else out_nc
    curv = stem + ".curv.nc"
    with netCDF4.Dataset(curv, "w") as dst:
        dst.createDimension("time", 1)
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)
        tvar = dst.createVariable("time", "f8", ("time",))
        tvar.units, tvar.calendar = "hours since 1970-01-01 00:00:00", "standard"
        tvar[:] = netCDF4.date2num(valid.astype("datetime64[s]").astype(object), tvar.units)
        for name, units, std, data in (("lat", "degrees_north", "latitude", lat),
                                       ("lon", "degrees_east", "longitude", lon)):
            var = dst.createVariable(name, "f4", ("y", "x"))
            var[:], var.units, var.standard_name = data, units, std
        var = dst.createVariable(FCST_NAME, "f4", ("time", "y", "x"),
                                 fill_value=np.float32(9.96921e36))
        var[:] = np.where(np.isfinite(z500), z500, 9.96921e36)[None]
        var.units, var.coordinates = "m", "lon lat"
    griddes = stem + ".griddes"
    with open(griddes, "w") as fh:
        fh.write("gridtype = lonlat\nxsize = %d\nysize = %d\n"
                 "xfirst = %.6f\nxinc = %.6f\nyfirst = %.6f\nyinc = %.6f\n"
                 % (nx, ny, lon.min(), (lon.max() - lon.min()) / (nx - 1), lat.min(),
                    (lat.max() - lat.min()) / (ny - 1)))
    tools.run([cdo, "-s", "remapbil,%s" % griddes, curv, out_nc])
    return out_nc


def write_interior_polygon(wrfout_path, bdy_width, out_path, name="INTERIOR"):
    """MET poly file (name, then one 'lat lon' vertex per line, lon east-positive) tracing the
    model domain's interior: the rectangle of grid indices [bdy_width, n - bdy_width) on each
    axis, walked cell by cell so the polygon follows the curvilinear grid's own edge."""
    import netCDF4
    import numpy as np
    from post.caseinfo import _interior_bounds
    with netCDF4.Dataset(wrfout_path) as ds:
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
    bounds = _interior_bounds(lat.shape[0], lat.shape[1], bdy_width)
    if bounds is None:
        raise METError("domain %dx%d too small for spec_bdy_width=%d to leave an interior"
                       % (lat.shape[0], lat.shape[1], bdy_width))
    j0, j1, i0, i1 = bounds
    j1, i1 = j1 - 1, i1 - 1
    ring = ([(j0, i) for i in range(i0, i1 + 1)] + [(j, i1) for j in range(j0 + 1, j1 + 1)]
            + [(j1, i) for i in range(i1 - 1, i0 - 1, -1)]
            + [(j, i0) for j in range(j1 - 1, j0, -1)])
    with open(out_path, "w") as fh:
        fh.write("%s\n" % name)
        for j, i in ring:
            fh.write("%.5f %.5f\n" % (lat[j, i], lon[j, i]))
    return out_path


def run_grid_stat(fcst_nc, obs_nc, poly_file, out_dir):
    """Runs `grid_stat FCST OBS CONFIG -outdir OUT_DIR`; returns (stat_path, cnt_txt_path, pairs_nc_path)."""
    grid_stat = tools.find_postproc_tool("grid_stat")
    os.makedirs(out_dir, exist_ok=True)
    config = _write_config(out_dir, "grid_stat.conf", FCST_NAME=FCST_NAME, OBS_NAME=OBS_NAME,
                           POLY_FILE=os.path.abspath(poly_file))
    tools.run([grid_stat, fcst_nc, obs_nc, config, "-outdir", out_dir])
    files = sorted(os.listdir(out_dir))
    stats = [f for f in files if f.endswith(".stat")]
    cnts = [f for f in files if f.endswith("_cnt.txt")]
    pairs = [f for f in files if f.endswith("_pairs.nc")]
    if not (stats and cnts and pairs):
        raise METError("grid_stat wrote no .stat/_cnt.txt/_pairs.nc into %s" % out_dir)
    return tuple(os.path.join(out_dir, x[-1]) for x in (stats, cnts, pairs))


def parse_cnt(cnt_text):
    """RMSE, ME (bias) and TOTAL (N matched points) from Grid-Stat's `_cnt.txt` (the same CNT
    line as in the .stat file, but with a full named header: a .stat header names only the
    columns common to every line type, not TOTAL/ME/RMSE). Raises METError without a CNT row."""
    from post.met_tc import parse_met_table
    for row in parse_met_table(cnt_text):
        if row.get("LINE_TYPE") == "CNT":
            return {"rmse_m": float(row["RMSE"]), "bias_m": float(row["ME"]),
                    "n": int(row["TOTAL"])}
    raise METError("no CNT line in the Grid-Stat output")


def grid_stat_z500(wrfout_path, era5_lat, era5_lon, era5_z500, valid, out_dir, bdy_width):
    """Whole chain for one frame. Returns dict(rmse_m, bias_m, n, stat_file, pairs_nc)."""
    os.makedirs(out_dir, exist_ok=True)
    obs_nc = write_era5_z500_nc(era5_lat, era5_lon, era5_z500, valid,
                                os.path.join(out_dir, "era5_z500.nc"))
    fcst_nc = model_z500_on_latlon(wrfout_path, os.path.join(out_dir, "model_z500.nc"))
    poly = write_interior_polygon(wrfout_path, bdy_width,
                                  os.path.join(out_dir, "interior.poly"))
    stat, cnt, pairs = run_grid_stat(fcst_nc, obs_nc, poly, out_dir)
    with open(cnt) as fh:
        res = parse_cnt(fh.read())
    res.update(stat_file=stat, pairs_nc=pairs)
    return res

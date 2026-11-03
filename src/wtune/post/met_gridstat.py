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

from wtune.post import tools
from wtune.post.met_tc import METError, _write_config

FCST_NAME = "Z500"
OBS_NAME = "Z500"


def write_era5_z500_nc(lat, lon, z, valid, out_nc):
    """ERA5 z500 (m) on its regular 1-D lat/lon grid -> CF NetCDF for MET (OBS)."""
    return _write_regular_nc(lat, lon, z, valid, out_nc)


def _write_regular_nc(lat, lon, z, valid, out_nc, name=OBS_NAME, units="m",
                      long_name="500 hPa geopotential height"):
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
        for coord, cunits, std, axis, data in (("lat", "degrees_north", "latitude", "Y", lat),
                                               ("lon", "degrees_east", "longitude", "X", lon)):
            var = dst.createVariable(coord, "f8", (coord,))
            var[:] = data
            var.units, var.standard_name, var.axis = cunits, std, axis
        var = dst.createVariable(name, "f4", ("time", "lat", "lon"),
                                 fill_value=np.float32(9.96921e36))
        var[:] = np.where(np.isfinite(z), z, 9.96921e36).astype("f4")[None]
        var.units, var.long_name = units, long_name
    return out_nc


def model_z500_on_latlon(wrfout_path, out_nc, timeidx=0):
    """wrfout frame -> regular lat/lon CF NetCDF of z500 (m): wrf-python z on 500 hPa, then
    `cdo remapbil` onto a lat/lon grid with the native point count over the native extent
    (cdo cannot remap a raw wrfout: it aborts on the eta coordinate ZNU)."""
    import netCDF4
    import numpy as np
    import wrf
    cdo = tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or "
                             "envs/wtune/pixi.toml provides it.")
    from wtune.post.conventions import continuous_lons
    with netCDF4.Dataset(wrfout_path) as src:
        lat = np.asarray(wrf.getvar(src, "lat", timeidx=timeidx))
        lon = continuous_lons(np.asarray(wrf.getvar(src, "lon", timeidx=timeidx)))  # across 180E
        z500 = np.asarray(wrf.interplevel(wrf.getvar(src, "z", timeidx=timeidx, units="m"),
                                          wrf.getvar(src, "pressure", timeidx=timeidx), 500))
        valid = wrf.extract_times(src, timeidx)
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
    from wtune.post.caseinfo import _interior_bounds
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


def run_grid_stat(fcst_nc, obs_nc, poly_file, out_dir, fcst_name=FCST_NAME, obs_name=OBS_NAME):
    """Runs `grid_stat FCST OBS CONFIG -outdir OUT_DIR`; returns (stat_path, cnt_txt_path, pairs_nc_path)."""
    grid_stat = tools.find_postproc_tool("grid_stat")
    os.makedirs(out_dir, exist_ok=True)
    config = _write_config(out_dir, "grid_stat.conf", FCST_NAME=fcst_name, OBS_NAME=obs_name,
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
    from wtune.post.met_tc import parse_met_table
    for row in parse_met_table(cnt_text):
        if row.get("LINE_TYPE") == "CNT":
            return {"rmse_m": float(row["RMSE"]), "bias_m": float(row["ME"]),
                    "n": int(row["TOTAL"])}
    raise METError("no CNT line in the Grid-Stat output")


def grid_stat_z500(wrfout_path, era5_lat, era5_lon, era5_z500, valid, out_dir, bdy_width,
                   timeidx=0):
    """Whole chain for one frame (time index `timeidx` of wrfout_path). Returns dict(rmse_m, bias_m, n, stat_file, pairs_nc)."""
    os.makedirs(out_dir, exist_ok=True)
    obs_nc = write_era5_z500_nc(era5_lat, era5_lon, era5_z500, valid,
                                os.path.join(out_dir, "era5_z500.nc"))
    fcst_nc = model_z500_on_latlon(wrfout_path, os.path.join(out_dir, "model_z500.nc"),
                                   timeidx=timeidx)
    poly = write_interior_polygon(wrfout_path, bdy_width,
                                  os.path.join(out_dir, "interior.poly"))
    stat, cnt, pairs = run_grid_stat(fcst_nc, obs_nc, poly, out_dir)
    with open(cnt) as fh:
        res = parse_cnt(fh.read())
    res.update(stat_file=stat, pairs_nc=pairs)
    return res


def grid_stat_field(lat, lon, model2d, e_lat, e_lon, e_field, valid, poly_file, out_dir, name,
                    units):
    """CNT of one 2-D model field (on the WRF grid lat/lon) against the same field on ERA5's
    regular grid: `cdo remapbil` onto lat/lon, MET Grid-Stat on the ERA5 grid inside poly_file.
    Returns dict(rmse, bias, n, stat_file, pairs_nc)."""
    os.makedirs(out_dir, exist_ok=True)
    fcst = curvilinear_to_latlon(lat, lon, {"F_" + name: (model2d, units, name)}, valid,
                                 os.path.join(out_dir, "model_%s.nc" % name), method="remapbil")
    obs = _write_regular_nc(e_lat, e_lon, e_field, valid, os.path.join(out_dir, "era5_%s.nc" % name),
                            name="O_" + name, units=units, long_name=name)
    stat, cnt, pairs = run_grid_stat(fcst, obs, poly_file, out_dir, fcst_name="F_" + name,
                                     obs_name="O_" + name)
    with open(cnt) as fh:
        res = parse_cnt(fh.read())
    return dict(rmse=res["rmse_m"], bias=res["bias_m"], n=res["n"], stat_file=stat, pairs_nc=pairs)


# ======================================================================================
# Model-vs-observation verification on arbitrary fields (Stage IV precipitation, radar
# composite reflectivity). Same recipe as the z500 chain above: cdo moves the curvilinear WRF
# grid to a regular lat/lon grid MET can read, MET regrids/masks/scores.
# ======================================================================================

FILL = 9.96921e36


def _cell_corners(a):
    """Cell-corner array (ny+1, nx+1) for cell-centre array a (ny, nx): the mean of the four
    surrounding centres, linearly extrapolated one half cell at the edges. cdo's conservative
    remap needs cell corners and the wrfout carries none for the mass grid."""
    import numpy as np
    p = np.pad(a, 1, mode="reflect", reflect_type="odd")
    return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])


def curvilinear_to_latlon(lat, lon, fields, valid, out_nc, method="remapcon", griddes_out=None):
    """Writes `fields` ({name: (2-D array on the WRF grid, units, long_name)}) as a regular
    lat/lon CF NetCDF at out_nc via `cdo <method>` onto a lat/lon grid with the native point
    count over the native lon/lat extent (same grid as model_z500_on_latlon). method remapcon =
    first-order conservative (precipitation), remapnn = nearest neighbour (reflectivity,
    thresholded categories), remapbil = bilinear (smooth fields). Returns out_nc."""
    import netCDF4
    import numpy as np
    from wtune.post.conventions import continuous_lons
    cdo = tools.find_on_path("cdo", install_hint="envs/wtune/pixi.toml provides it.")
    lon = continuous_lons(lon)             # across 180E the min/max box would span the globe
    ny, nx = lat.shape
    stem = out_nc[:-3] if out_nc.endswith(".nc") else out_nc
    curv = stem + ".curv.nc"
    with netCDF4.Dataset(curv, "w") as dst:
        dst.createDimension("time", 1)
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)
        dst.createDimension("nv", 4)
        tvar = dst.createVariable("time", "f8", ("time",))
        tvar.units, tvar.calendar = "hours since 1970-01-01 00:00:00", "standard"
        tvar[:] = netCDF4.date2num(valid, tvar.units)
        latc, lonc = _cell_corners(lat), _cell_corners(lon)
        for name, units, std, data, cor in (("lat", "degrees_north", "latitude", lat, latc),
                                            ("lon", "degrees_east", "longitude", lon, lonc)):
            var = dst.createVariable(name, "f8", ("y", "x"))
            var[:], var.units, var.standard_name = data, units, std
            var.bounds = name + "_bnds"
            bnd = dst.createVariable(name + "_bnds", "f8", ("y", "x", "nv"))
            bnd[:] = np.stack([cor[:-1, :-1], cor[:-1, 1:], cor[1:, 1:], cor[1:, :-1]], axis=-1)
        for name, (arr, units, long_name) in fields.items():
            var = dst.createVariable(name, "f4", ("time", "y", "x"), fill_value=np.float32(FILL))
            var[:] = np.where(np.isfinite(arr), arr, FILL)[None]
            var.units, var.long_name, var.coordinates = units, long_name, "lon lat"
    griddes = griddes_out or (stem + ".griddes")
    with open(griddes, "w") as fh:
        fh.write("gridtype = lonlat\nxsize = %d\nysize = %d\n"
                 "xfirst = %.6f\nxinc = %.6f\nyfirst = %.6f\nyinc = %.6f\n"
                 % (nx, ny, lon.min(), (lon.max() - lon.min()) / (nx - 1), lat.min(),
                    (lat.max() - lat.min()) / (ny - 1)))
    tools.run([cdo, "-s", "%s,%s" % (method, griddes), curv, out_nc])
    os.remove(curv)
    _tag_axes(out_nc)
    return out_nc


def _tag_axes(nc_path):
    """MET's CF reader needs standard_name/axis tags on lat/lon/time (cdo output has most)."""
    import netCDF4
    with netCDF4.Dataset(nc_path, "a") as ds:
        for name, axis, std in (("lat", "Y", "latitude"), ("lon", "X", "longitude"),
                                ("time", "T", "time")):
            if name in ds.variables:
                ds.variables[name].axis = axis
                ds.variables[name].standard_name = std


def run_verification(fcst_file, obs_file, poly_file, out_dir, fcst_name, obs_name, thresholds,
                     fcst_level="(0,*,*)", obs_level="(0,*,*)", widths=(1, 3, 5, 9),
                     to_grid="OBS", regrid_method="BUDGET", model="WRF", obtype="OBS"):
    """Runs Grid-Stat for one field pair with the verif_grid_stat template: categorical
    (FHO/CTC/CTS) at `thresholds` (each a MET threshold string like '>=25'), NBRCNT (FSS) at
    the neighbourhood `widths`, CNT, masked to poly_file, matched-pair NetCDF with raw FCST/OBS.
    Returns dict(out_dir, stat, cts, nbrcnt, cnt, pairs_nc, ctc)."""
    grid_stat = tools.find_postproc_tool("grid_stat")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):                    # a re-run must not read stale output
        if f.startswith("grid_stat_"):
            os.remove(os.path.join(out_dir, f))
    config = _write_config(out_dir, "verif_grid_stat.conf", MODEL=model, OBTYPE=obtype,
                           FCST_NAME=fcst_name, FCST_LEVEL=fcst_level, OBS_NAME=obs_name,
                           OBS_LEVEL=obs_level, CAT_THRESH=", ".join(thresholds),
                           TO_GRID=to_grid, REGRID_METHOD=regrid_method,
                           POLY_FILE=os.path.abspath(poly_file),
                           NBR_WIDTHS=", ".join(str(w) for w in widths))
    tools.run([grid_stat, fcst_file, obs_file, config, "-outdir", out_dir])
    files = sorted(os.listdir(out_dir))

    def one(suffix):
        hit = [f for f in files if f.endswith(suffix)]
        return os.path.join(out_dir, hit[0]) if hit else None
    res = dict(out_dir=out_dir, stat=one(".stat"), cts=one("_cts.txt"), ctc=one("_ctc.txt"),
               nbrcnt=one("_nbrcnt.txt"), cnt=one("_cnt.txt"), pairs_nc=one("_pairs.nc"))
    if not (res["stat"] and res["cts"] and res["nbrcnt"] and res["pairs_nc"]):
        raise METError("grid_stat wrote no stat/cts/nbrcnt/pairs into %s (%s)"
                       % (out_dir, ", ".join(files)))
    return res


def _rows(path, line_type):
    from wtune.post.met_tc import parse_met_table
    with open(path) as fh:
        return [r for r in parse_met_table(fh.read()) if r.get("LINE_TYPE") == line_type]


def _thresh_value(text):
    """'>=25' / 'ge25' style MET threshold string -> float (the number after the operator)."""
    import re
    m = re.search(r"(-?\d+\.?\d*)", text)
    return float(m.group(1)) if m else float("nan")


def parse_cts_table(cts_path):
    """[{thresh, pody (POD), far, csi, fbias, total}] from Grid-Stat's _cts.txt, one per
    categorical threshold (values NaN where MET wrote NA, e.g. no observed event)."""
    out = []
    for r in _rows(cts_path, "CTS"):
        def num(k):
            try:
                return float(r[k])
            except (KeyError, ValueError):
                return float("nan")
        out.append(dict(thresh=_thresh_value(r["FCST_THRESH"]), pod=num("PODY"), far=num("FAR"),
                        csi=num("CSI"), fbias=num("FBIAS"), total=int(float(r.get("TOTAL", "nan")))))
    return sorted(out, key=lambda d: d["thresh"])


def parse_ctc_counts(ctc_path):
    """[{thresh, fy_oy, fy_on, fn_oy, fn_on}] per threshold from _ctc.txt (for aggregating
    several times by summing the contingency-table counts)."""
    out = []
    for r in _rows(ctc_path, "CTC"):
        out.append(dict(thresh=_thresh_value(r["FCST_THRESH"]), fy_oy=int(float(r["FY_OY"])),
                        fy_on=int(float(r["FY_ON"])), fn_oy=int(float(r["FN_OY"])),
                        fn_on=int(float(r["FN_ON"]))))
    return sorted(out, key=lambda d: d["thresh"])


def parse_nbrcnt_table(nbrcnt_path):
    """[{thresh, width, fss, fbs, f_rate, o_rate}]: width = neighbourhood width in grid points
    (INTERP_PNTS is width**2), per event threshold."""
    import math
    out = []
    for r in _rows(nbrcnt_path, "NBRCNT"):
        def num(k):
            try:
                return float(r[k])
            except (KeyError, ValueError):
                return float("nan")
        pts = num("INTERP_PNTS")
        out.append(dict(thresh=_thresh_value(r["FCST_THRESH"]),
                        width=int(round(math.sqrt(pts))) if pts == pts else 0, fss=num("FSS"),
                        fbs=num("FBS"), f_rate=num("F_RATE"), o_rate=num("O_RATE")))
    return sorted(out, key=lambda d: (d["thresh"], d["width"]))


def read_pairs_fields(pairs_nc):
    """(lat2d, lon2d, fcst2d, obs2d, fcst_name, obs_name) from a Grid-Stat matched-pair file's
    raw FCST_*/OBS_* variables (masked cells NaN)."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(pairs_nc) as ds:
        names = list(ds.variables)
        fn = [n for n in names if n.startswith("FCST_") and "_MASK_" not in n
              and "DIFF" not in n and "NBRHD" not in n]
        on = [n for n in names if n.startswith("OBS_") and "_MASK_" not in n]
        if not fn or not on:
            raise METError("no FCST_*/OBS_* raw fields in %s (variables: %s)"
                           % (pairs_nc, ", ".join(names)))
        lat = np.ma.filled(ds.variables["lat"][:], np.nan).astype("f8")
        lon = np.ma.filled(ds.variables["lon"][:], np.nan).astype("f8")
        fc = np.ma.filled(ds.variables[fn[0]][:], np.nan).astype("f8")
        ob = np.ma.filled(ds.variables[on[0]][:], np.nan).astype("f8")
    if lat.ndim == 1:                                  # regular lat/lon verification grid
        lon, lat = np.meshgrid(lon, lat)
    return lat, lon, fc, ob, fn[0], on[0]


def coverage_polygon(sites, radius_km, interior_poly, out_path, name="RADAR_COVERAGE"):
    """MET poly file = the interior polygon intersected with the union of `radius_km` circles
    around the radar `sites` [(icao, lat, lon)]. Stage IV holds 0 (not missing) beyond radar
    range, so verifying over open water would score the model against fake dry cells. Geometry
    glue (shapely, planar lon/lat with a cos(lat) longitude scale); if the result has several
    parts the largest is kept. Returns (path, n_parts_dropped) or (None, 0) when no site reaches
    the interior."""
    import math
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    with open(interior_poly) as fh:
        lines = [ln.split() for ln in fh.read().splitlines()[1:] if ln.strip()]
    interior = Polygon([(float(lo), float(la)) for la, lo in lines])
    circles = []
    for _icao, la, lo in sites:
        dlat = radius_km / 111.2
        dlon = dlat / max(math.cos(math.radians(la)), 0.2)
        circles.append(Polygon([(lo + dlon * math.cos(a * math.pi / 36), la + dlat * math.sin(a * math.pi / 36))
                                for a in range(72)]))
    if not circles:
        return None, 0
    geom = interior.intersection(unary_union(circles))
    parts = list(getattr(geom, "geoms", [geom]))
    parts = [g for g in parts if g.geom_type == "Polygon" and g.area > 0]
    if not parts:
        return None, 0
    parts.sort(key=lambda g: -g.area)
    with open(out_path, "w") as fh:
        fh.write("%s\n" % name)
        for lo, la in list(parts[0].exterior.coords)[:-1]:
            fh.write("%.5f %.5f\n" % (la, lo))
    return out_path, len(parts) - 1

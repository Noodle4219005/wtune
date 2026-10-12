"""post/met_tc.py -- wraps MET's tc_pairs/tc_stat (track/intensity/wind-radii error against
a b-deck) and tc_rmw (storm-relative azimuthal-mean radius-height structure), replacing the
self-written track-error/intensity/azimuthal-profile/RMW code post/tc_metrics.py used to
carry (decision 0056, 2026-09-29 update). Config files are generated from the templates in
post/met/ (${...} string substitution -- MET's own config language, not Python format
strings, so a literal ${...} in a value would need escaping; none of ours do).

MET is not installed yet (bin/install-postproc.sh, WTUNE_POSTPROC_VIEW) at the time this
was written, so none of the exact tc_pairs/tc_stat/tc_rmw CLI flags or config keys below
have been run against a real binary -- see each .conf.tmpl's header comment and this
module's functions for the "NOT YET VERIFIED" notes. What IS verified: MET's .tcst/.stat
output convention (a header line of column names, then one whitespace-padded row per
record) is stable across every MET tool and documented in the MET User's Guide -- parse_met_table()
relies only on that, not on exact column names, so it degrades gracefully (returns whatever
columns the installed MET version actually wrote) rather than breaking on a version skew.
"""
from __future__ import annotations

import os
import string

from wtune.post import tools

_MET_DIR = os.path.dirname(__file__) + "/met"


class METError(RuntimeError):
    pass


def _render_template(name, **subs):
    with open(os.path.join(_MET_DIR, name)) as fh:
        template = string.Template(fh.read())
    return template.substitute(**subs)


def _write_config(out_dir, name, **subs):
    text = _render_template(name + ".tmpl", **subs)
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def parse_met_table(text):
    """A MET .tcst/.stat table (header line of column names, then whitespace-padded rows)
    -> list of dicts. Skips blank lines and MET's occasional leading comment lines (those
    starting with '#', seen in some MET tool banners)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    header = lines[0].split()
    rows = []
    for line in lines[1:]:
        values = line.split()
        rows.append(dict(zip(header, values)))
    return rows


def run_tc_pairs(adeck, bdeck, out_dir, model, dland_file=""):
    """Runs `tc_pairs -adeck ADECK -bdeck BDECK -config CONFIG -out OUT_PREFIX`. Returns
    the .tcst path. NOT YET VERIFIED -- see module docstring."""
    tc_pairs = tools.find_postproc_tool("tc_pairs")
    os.makedirs(out_dir, exist_ok=True)
    config = _write_config(out_dir, "tc_pairs.conf", MODEL=model, DLAND_FILE=dland_file)
    out_prefix = os.path.join(out_dir, "tc_pairs")
    tools.run([tc_pairs, "-adeck", adeck, "-bdeck", bdeck, "-config", config,
              "-out", out_prefix])
    tcst_path = out_prefix + ".tcst"
    if not os.path.exists(tcst_path):
        raise METError("tc_pairs did not produce %s" % tcst_path)
    return tcst_path


def run_tc_stat(tcst_path, out_dir):
    """Runs `tc_stat -lookin TCST -config CONFIG`, producing track/intensity error
    (tc_stat_track.stat) and wind-radii error (tc_stat_wind.stat) summary tables. Returns
    (track_stat_path, wind_stat_path). NOT YET VERIFIED -- see module docstring."""
    tc_stat = tools.find_postproc_tool("tc_stat")
    out_stat = os.path.join(out_dir, "tc_stat_track.stat")
    out_stat_wind = os.path.join(out_dir, "tc_stat_wind.stat")
    config = _write_config(out_dir, "tc_stat.conf", OUT_STAT=out_stat,
                           OUT_STAT_WIND=out_stat_wind)
    tools.run([tc_stat, "-lookin", tcst_path, "-config", config])
    for p in (out_stat, out_stat_wind):
        if not os.path.exists(p):
            raise METError("tc_stat did not produce %s" % p)
    return out_stat, out_stat_wind


def run_tc_rmw(data_grb, deck, out_nc, levels=None, valid_hours=(0, 6, 12, 18), n_range=60,
               n_azimuth=180, delta_range_km=10.0):
    """Runs `tc_rmw -data DATA_GRB -deck DECK -config CONFIG -out OUT_NC`: 10 m wind speed and
    u/v on `levels` hPa, which tc_rmw turns into tangential/radial winds (VT/VR) on its
    storm-centred range-azimuth grid for every valid time of the deck that the data covers.
    data_grb comes from regrid_for_tc_rmw() below (GRIB1: this MET build has no GRIB2, and
    tc_rmw only keeps a pressure axis for GRIB-style P levels, not for CF NetCDF selections)."""
    tc_rmw = tools.find_postproc_tool("tc_rmw")
    out_dir = os.path.dirname(os.path.abspath(out_nc))
    os.makedirs(out_dir, exist_ok=True)
    levels = PRESSURE_LEVELS_HPA if levels is None else levels
    plev = ", ".join('"P%d"' % p for p in levels)
    fields = ['{ name = "UGRD"; level = [%s]; }' % plev,
              '{ name = "VGRD"; level = [%s]; }' % plev]
    config = _write_config(out_dir, "tc_rmw.conf", FIELDS=", ".join(fields),
                           VALID_HOURS=", ".join('"%02d0000"' % h for h in valid_hours),
                           N_RANGE=n_range,
                           N_AZIMUTH=n_azimuth, DELTA_RANGE_KM=delta_range_km)
    tools.run([tc_rmw, "-data", data_grb, "-deck", deck, "-config", config, "-out", out_nc])
    if not os.path.exists(out_nc):
        raise METError("tc_rmw did not produce %s" % out_nc)
    return out_nc


def storm_relative_winds(u, v, azimuth_deg):
    """(VT, VR) from u/v that tc_rmw sampled on its range-azimuth grid, using the azimuth
    convention tc_rmw labels that grid with ("degrees_clockwise_from_east"): VT positive
    counterclockwise (cyclonic in the north), VR positive outward. The one self-computed
    step in this module, because tc_rmw's own VT/VR are not usable here: with MET 12.1.1 on
    scc its sampled u/v form a correct cyclone on that clockwise grid, but its VT/VR match a
    counterclockwise decomposition (Katrina, 850 hPa, 50 km: at azimuth 80 it reports
    VT -38.1 / VR +28.4 where u=42.4, v=21.4 give VT +45.5 / VR -13.7), so their azimuthal
    means cancel to ~0. `azimuth_deg` broadcasts against the last axis of u and v."""
    import numpy as np
    az = np.radians(azimuth_deg)
    return u * np.sin(az) + v * np.cos(az), u * np.cos(az) - v * np.sin(az)


# GRIB1 parameter codes, NCEP table 2: UGRD 33, VGRD 34
_GRIB1_PARAMS = (("u", "33.2"), ("v", "34.2"))


def regrid_for_tc_rmw(wrfout_paths, out_grb):
    """wrfout frames (each a path, or (path, timeidx) for a valid time inside a multi-frame
    wrfout -- post.extract.frame_ref) -> one multi-time GRIB1 file on a regular lat/lon grid for tc_rmw:
    per frame, wrf-python fields on pressure levels (cdo cannot remap a raw wrfout: it
    aborts on the eta coordinate ZNU) -> `cdo remapbil`; then `cdo mergetime` and
    `cdo -f grb setparam` to NCEP GRIB1 codes. The target grid keeps the native point count
    over the native lon/lat extent (from `cdo sinfon`), neither inventing nor discarding
    resolution."""
    cdo = tools.find_on_path("cdo", install_hint="conda-forge 'cdo' or "
                             "envs/wtune/pixi.toml provides it.")
    refs = [(p, 0) if isinstance(p, (str, os.PathLike)) else tuple(p) for p in wrfout_paths]
    wrfout_paths = [p for p, _ti in refs]
    grid = tools.parse_cdo_sinfon_grid(tools.cdo_sinfon(wrfout_paths[0]))
    if grid is None or grid["lon_range"] is None or grid["lat_range"] is None:
        raise METError("could not read a lon/lat extent from `cdo sinfon %s` to build the "
                       "tc_rmw regrid target." % wrfout_paths[0])
    lon0, lon1 = grid["lon_range"]
    lat0, lat1 = grid["lat_range"]
    nlon, nlat = grid["nx"], grid["ny"]
    griddes_path = out_grb + ".griddes"
    with open(griddes_path, "w") as fh:
        fh.write("gridtype = lonlat\nxsize = %d\nysize = %d\n"
                 "xfirst = %.6f\nxinc = %.6f\nyfirst = %.6f\nyinc = %.6f\n"
                 % (nlon, nlat, lon0, (lon1 - lon0) / (nlon - 1), lat0,
                    (lat1 - lat0) / (nlat - 1)))
    frames = []
    for i, (path, ti) in enumerate(refs):
        fields_nc, remapped = "%s.f%03d.nc" % (out_grb, i), "%s.r%03d.nc" % (out_grb, i)
        _wrf_fields_on_pressure_levels(path, fields_nc, timeidx=ti)
        tools.run([cdo, "-s", "remapbil,%s" % griddes_path, fields_nc, remapped])
        frames.append(remapped)
    merged = out_grb + ".merged.nc"
    tools.run([cdo, "-s", "mergetime"] + frames + [merged])
    chain = []
    for name, code in _GRIB1_PARAMS:
        chain += ["-setparam,%s" % code, "-selname,%s" % name, merged]
    tools.run([cdo, "-s", "-f", "grb", "merge"] + chain + [out_grb])
    return out_grb


# every 50 hPa, 1000-100 (19 levels): the 7 standard levels this used to carry drew the RMW
# radius-height figure as wide horizontal bands
PRESSURE_LEVELS_HPA = tuple(range(1000, 99, -50))


def _wrf_fields_on_pressure_levels(wrfout_path, out_nc, levels=PRESSURE_LEVELS_HPA, timeidx=0):
    """wrf-python diagnostics tc_rmw needs, written as a small NetCDF CDO can remap: 10 m wind
    speed and u/v (earth-relative) on `levels` hPa, each with `coordinates = "lon lat"`
    pointing at 2-D lat/lon, so CDO treats the grid as curvilinear."""
    import netCDF4
    import numpy as np
    import wrf
    with netCDF4.Dataset(wrfout_path) as src:
        lat = np.asarray(wrf.getvar(src, "lat", timeidx=timeidx))
        lon = np.asarray(wrf.getvar(src, "lon", timeidx=timeidx))
        wspd10 = np.asarray(wrf.getvar(src, "wspd_wdir10", timeidx=timeidx)[0])
        p = wrf.getvar(src, "pressure", timeidx=timeidx)
        # uvmet = rotated to earth coordinates; ua/va are grid-relative (on a Lambert grid
        # tc_rmw would then read tangential wind as radial)
        uvmet = wrf.getvar(src, "uvmet", timeidx=timeidx, units="m s-1")
        u = np.asarray(wrf.interplevel(uvmet[0], p, list(levels)))
        v = np.asarray(wrf.interplevel(uvmet[1], p, list(levels)))
        valid = wrf.extract_times(src, timeidx)
    ny, nx = lat.shape
    with netCDF4.Dataset(out_nc, "w") as dst:
        dst.createDimension("time", None)
        tvar = dst.createVariable("time", "f8", ("time",))
        tvar.units, tvar.calendar = "hours since 1970-01-01 00:00:00", "standard"
        tvar[:] = netCDF4.date2num(valid.astype("datetime64[s]").astype(object), tvar.units)
        dst.createDimension("lev", len(levels))
        dst.createDimension("y", ny)
        dst.createDimension("x", nx)
        lev = dst.createVariable("lev", "f4", ("lev",))
        lev[:] = levels
        lev.units, lev.positive, lev.standard_name = "hPa", "down", "air_pressure"
        for name, units, std, data in (("lat", "degrees_north", "latitude", lat),
                                       ("lon", "degrees_east", "longitude", lon)):
            var = dst.createVariable(name, "f4", ("y", "x"))
            var[:], var.units, var.standard_name = data, units, std
        for name, dims, data, units in (("wind_speed", ("time", "y", "x"), wspd10[None], "m s-1"),
                                        ("u", ("time", "lev", "y", "x"), u[None], "m s-1"),
                                        ("v", ("time", "lev", "y", "x"), v[None], "m s-1")):
            var = dst.createVariable(name, "f4", dims, fill_value=np.float32(9.96921e36))
            var[:] = np.where(np.isfinite(data), data, 9.96921e36)
            var.units, var.coordinates = units, "lon lat"

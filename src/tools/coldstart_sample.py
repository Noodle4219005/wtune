#!/usr/bin/env python3
"""tools/coldstart_sample.py -- unknown-case wall-time prediction by cold-start sampling windows.

Given a WRF case that has only a namelist.input (+ a source of map projection) and NO
wrfinput/wrfbdy/restart files for the run being predicted, this tool:
  1. works out what is missing and where to get it (GFS via tools/metdata.py),
  2. picks K "characteristic point" sampling times inside the run,
  3. fetches only the GRIB times those windows need,
  4. stages ONE sbatch that runs geogrid -> ungrib -> metgrid -> real.exe (one per window) ->
     wrf.exe (one short cold-start window per sampling time, all with the configuration being
     predicted),
  5. scores the windows and prints a whole-run wall-time prediction.

A window is an independent cold start from GFS state at time T_k (NOT a restart -- an unknown
case has no restart file). Measured on the cluster (conus12, 3 times x paired with true
restarts): cold-start cost equals restart cost within 0.35% once the first ~1 simulated hour is
discarded; with only 37 simulated minutes it was biased up to 3.1%. Hence spin-up discard in
`score`.

This is a GENERALISATION of the ad-hoc, conus12-hardcoded recipe in .route1-ref/ (proven on the
cluster once for one case) -- see that directory's SPEC.md for the full design. Notably reused,
not reimplemented, from that recipe: the packed-line-safe namelist set_list/count_key editing
approach (gate2_stage_window.py / gate2_stage_real.py) and the "never write through a
pre-existing symlink" rule it discovered the hard way.

CLI: python3 tools/coldstart_sample.py <inspect|plan|fetch|stage|score|fix-wps-namelist|
fix-real-namelist|fix-wrf-namelist> ...  See each subcommand's --help.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import glob
import os
import re
import shlex
import shutil
import statistics
import subprocess
import threading
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wtune_predict as predict   # repo root -- delegates run-length resolution to
                                   # wtune_runlength.resolve_run_seconds internally
                                   # (stdlib only, per its own docstring -- no numpy)

DEFAULT_CALIBRATION = os.path.join(
    os.path.dirname(_REPO_ROOT), "machine", "calibration", "scc.json")  # machine/ stays at repo root
_WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
_SELF_PATH = os.path.abspath(__file__)
# K5 (2026-09-24): state_correction_model.py lives next to this file -- the ML-apply sbatch step
# invokes it with the grib-env python (never /usr/bin/python3: it needs numpy+netCDF4).
_SCM_PATH = os.path.join(os.path.dirname(_SELF_PATH), "state_correction_model.py")


class ColdstartError(Exception):
    """A user-facing input/modeling error (bad namelist, missing projection, etc.)."""


# tools/metdata.py and tools/grib_state.py both `import numpy as np` unconditionally at module
# level, and scc's compute-node /usr/bin/python3 has no numpy (2026-09 finding). Every path that
# runs on a compute node -- fix-real-namelist, fix-wrf-namelist, fix-wps-namelist, stage, score
# -- must not trigger these imports at all (wtune_predict.py is stdlib-only and stays a normal
# eager import above). Only `inspect`, `plan --state`, and `fetch` (all login-node/offline-doc
# operations) ever need metdata/grib_state, and they get them lazily through these two helpers.
_md = None
_gs = None


def _metdata():
    global _md
    if _md is None:
        try:
            import metdata as _md_mod
        except ImportError as exc:
            raise ColdstartError(
                "tools/metdata.py could not be imported (needs numpy): %s" % exc)
        _md = _md_mod
    return _md


def _sh_quote(s):
    return shlex.quote(s)


def _grib_state():
    global _gs
    if _gs is None:
        try:
            import grib_state as _gs_mod
        except ImportError as exc:
            raise ColdstartError(
                "tools/grib_state.py could not be imported (needs numpy): %s" % exc)
        _gs = _gs_mod
    return _gs


# ---------------------------------------------------------------------------
# namelist.input helpers (thin wrappers over wtune_predict -- do not re-parse)
# ---------------------------------------------------------------------------

def load_namelist(path):
    if not os.path.exists(path):
        raise ColdstartError("no namelist.input at %s" % path)
    return predict.parse_namelist(path)


def run_span(nl):
    """(start_dt, end_dt, run_seconds) for the case's own d01 run length."""
    start_dt = predict.get_time_control_datetime(nl, "start")
    if start_dt is None:
        raise ColdstartError("namelist.input &time_control has no start_year/month/day")
    run_s = predict.get_run_seconds(nl)
    end_dt = start_dt + datetime.timedelta(seconds=run_s)
    return start_dt, end_dt, run_s


# ---------------------------------------------------------------------------
# Projection completion (namelist.wps -- see Inputs/Projection in SPEC.md)
# ---------------------------------------------------------------------------

MAP_PROJ_NAMES = {1: "lambert", 2: "polar", 3: "mercator", 6: "lat-lon"}


def find_ncdump(explicit=None):
    if explicit:
        return explicit
    exe = shutil.which("ncdump")
    if exe:
        return exe
    netcdf_dir = os.environ.get("NETCDF")
    if netcdf_dir:
        cand = os.path.join(netcdf_dir, "bin", "ncdump")
        if os.path.exists(cand):
            return cand
    return None


def read_wrfinput_global_attrs(wrfinput_path, ncdump_exe):
    """ncdump -h TEXT parse of the '// global attributes:' block -> {ATTR: float|str}. Same
    approach as .route1-ref/make_namelist_wps.py's read_global_attrs (kept here rather than
    imported: that script lives in the untracked, do-not-import .route1-ref/ reference dir)."""
    out = subprocess.run([ncdump_exe, "-h", wrfinput_path], check=True,
                          capture_output=True, text=True).stdout
    m = re.search(r"^// global attributes:\s*$(.*)^\}\s*$", out, re.MULTILINE | re.DOTALL)
    if not m:
        raise ColdstartError("could not locate '// global attributes:' block in ncdump -h of %s"
                              % wrfinput_path)
    attrs = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        mm = re.match(r":([A-Za-z0-9_-]+)\s*=\s*(.+?)\s*;\s*$", line)
        if not mm:
            continue
        name, val = mm.group(1), mm.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            attrs[name] = val[1:-1]
        else:
            val = val.rstrip("fF")
            try:
                attrs[name] = float(val)
            except ValueError:
                attrs[name] = val
    return attrs


def geo_from_wrfinput_attrs(attrs, nl):
    """d01 projection fields from a wrfinput_d01 global-attribute dict (see
    read_wrfinput_global_attrs / make_namelist_wps.py). Cross-checks e_we/e_sn/dx against
    namelist.input; mismatch is an error. Deliberately DROPS make_namelist_wps.py's
    MOAD_CEN_LAT==CEN_LAT refusal: for d01, MOAD_CEN_LAT==CEN_LAT always (d01 IS the MOAD), so
    that check never actually distinguished a nested case from an unnested one -- nests are
    described by namelist.input's own per-domain lists (domain_list_fields), not by wrfinput.
    """
    if "MAP_PROJ" not in attrs:
        raise ColdstartError("wrfinput global attributes have no MAP_PROJ")
    map_proj_code = int(attrs["MAP_PROJ"])
    if map_proj_code not in MAP_PROJ_NAMES:
        raise ColdstartError("unhandled MAP_PROJ=%r (only 1/2/3/6 mapped)" % map_proj_code)

    e_we_attr = int(attrs["WEST-EAST_GRID_DIMENSION"])
    e_sn_attr = int(attrs["SOUTH-NORTH_GRID_DIMENSION"])
    nl_e_we = predict.g_dom(nl, "domains", "e_we", 0)
    nl_e_sn = predict.g_dom(nl, "domains", "e_sn", 0)
    if nl_e_we is not None and int(nl_e_we) != e_we_attr:
        raise ColdstartError("wrfinput_d01 e_we=%d != namelist.input e_we=%d" % (e_we_attr, nl_e_we))
    if nl_e_sn is not None and int(nl_e_sn) != e_sn_attr:
        raise ColdstartError("wrfinput_d01 e_sn=%d != namelist.input e_sn=%d" % (e_sn_attr, nl_e_sn))

    dx = float(attrs["DX"])
    dy = float(attrs["DY"])
    nl_dx = predict.g_dom(nl, "domains", "dx", 0)
    if nl_dx is not None and abs(dx - float(nl_dx)) > 1e-6:
        raise ColdstartError("wrfinput_d01 DX=%r != namelist.input dx=%r" % (dx, nl_dx))

    return {
        "map_proj": MAP_PROJ_NAMES[map_proj_code],
        "truelat1": attrs.get("TRUELAT1"),
        "truelat2": attrs.get("TRUELAT2"),
        "stand_lon": attrs.get("STAND_LON"),
        "pole_lat": attrs.get("POLE_LAT"),
        "pole_lon": attrs.get("POLE_LON"),
        "ref_lat": float(attrs["CEN_LAT"]),
        "ref_lon": float(attrs["CEN_LON"]),
        "e_we": e_we_attr, "e_sn": e_sn_attr, "dx": dx, "dy": dy,
        "geog_data_res": "default",
        "source": "wrfinput",
    }


def geo_from_namelist_wps(nlw, nl):
    """d01 projection fields from an existing namelist.wps &geogrid group (already parsed via
    wtune_predict.parse_namelist)."""
    if "geogrid" not in nlw or "share" not in nlw:
        raise ColdstartError("namelist.wps has no &geogrid/&share group")
    map_proj = predict.g(nlw, "geogrid", "map_proj", None)
    if map_proj is None:
        raise ColdstartError("namelist.wps &geogrid has no map_proj")
    e_we = predict.g_dom(nlw, "geogrid", "e_we", 0)
    e_sn = predict.g_dom(nlw, "geogrid", "e_sn", 0)
    dx = predict.g(nlw, "geogrid", "dx", None)
    dy = predict.g(nlw, "geogrid", "dy", None)
    nl_e_we = predict.g_dom(nl, "domains", "e_we", 0)
    nl_e_sn = predict.g_dom(nl, "domains", "e_sn", 0)
    if nl_e_we is not None and e_we is not None and int(e_we) != int(nl_e_we):
        raise ColdstartError("namelist.wps e_we=%s != namelist.input e_we=%s" % (e_we, nl_e_we))
    if nl_e_sn is not None and e_sn is not None and int(e_sn) != int(nl_e_sn):
        raise ColdstartError("namelist.wps e_sn=%s != namelist.input e_sn=%s" % (e_sn, nl_e_sn))
    nl_dx = predict.g_dom(nl, "domains", "dx", 0)
    if nl_dx is not None and dx is not None and abs(float(dx) - float(nl_dx)) > 1e-6:
        raise ColdstartError("namelist.wps dx=%s != namelist.input dx=%s" % (dx, nl_dx))
    return {
        "map_proj": str(map_proj),
        "truelat1": predict.g(nlw, "geogrid", "truelat1", None),
        "truelat2": predict.g(nlw, "geogrid", "truelat2", None),
        "stand_lon": predict.g(nlw, "geogrid", "stand_lon", None),
        "pole_lat": predict.g(nlw, "geogrid", "pole_lat", None),
        "pole_lon": predict.g(nlw, "geogrid", "pole_lon", None),
        "ref_lat": float(predict.g(nlw, "geogrid", "ref_lat")),
        "ref_lon": float(predict.g(nlw, "geogrid", "ref_lon")),
        "e_we": int(e_we) if e_we is not None else None,
        "e_sn": int(e_sn) if e_sn is not None else None,
        "dx": float(dx) if dx is not None else None,
        "dy": float(dy) if dy is not None else None,
        "geog_data_res": predict.g(nlw, "geogrid", "geog_data_res", "default"),
        "source": "namelist.wps",
    }


def resolve_projection(nl, namelist_wps_path=None, geo_from_wrfinput_path=None, ncdump=None):
    """Projection source, in the order SPEC.md's Inputs section gives:
      1. --namelist-wps / D/namelist.wps
      2. --geo-from-wrfinput / D/wrfinput_d01
      3. otherwise blocked (raises ColdstartError).
    Returns geo_from_wrfinput_attrs/geo_from_namelist_wps's dict."""
    if namelist_wps_path and os.path.exists(namelist_wps_path):
        nlw = predict.parse_namelist(namelist_wps_path)
        return geo_from_namelist_wps(nlw, nl)
    if geo_from_wrfinput_path and os.path.exists(geo_from_wrfinput_path):
        ncdump_exe = find_ncdump(ncdump)
        if not ncdump_exe:
            raise ColdstartError(
                "no ncdump on PATH and $NETCDF/bin/ncdump does not exist -- cannot read %s "
                "(source ~/wtune-scc/env.sh, or pass --ncdump)" % geo_from_wrfinput_path)
        attrs = read_wrfinput_global_attrs(geo_from_wrfinput_path, ncdump_exe)
        return geo_from_wrfinput_attrs(attrs, nl)
    raise ColdstartError(
        "no projection source: need --namelist-wps/D/namelist.wps or "
        "--geo-from-wrfinput/D/wrfinput_d01 (namelist.input alone has no map projection info)")


def domain_list_fields(nl, max_dom):
    """Per-domain &geogrid lists (parent_id, parent_grid_ratio, i_parent_start, j_parent_start,
    e_we, e_sn) for namelist.wps. Domain 1 always uses geogrid's own self-parent convention
    (1,1,1,1) regardless of what namelist.input's WRF-side &domains says for d01 (WRF and WPS
    disagree on this by convention: WRF's own d01 parent_id/i_parent_start/j_parent_start are
    often 0, geogrid always wants 1 -- see .route1-ref/make_namelist_wps.py). Domain 2+ (a real
    nest) copies parent_id/parent_grid_ratio/i_parent_start/j_parent_start verbatim from
    namelist.input, since those WRF-side numbers for a nest ARE the same numbers WPS/geogrid
    originally placed it with. e_we/e_sn are always copied from namelist.input for every domain.
    """
    parent_id, parent_grid_ratio = [], []
    i_parent_start, j_parent_start = [], []
    e_we, e_sn = [], []
    for d in range(max_dom):
        if d == 0:
            parent_id.append(1)
            parent_grid_ratio.append(1)
            i_parent_start.append(1)
            j_parent_start.append(1)
        else:
            parent_id.append(int(predict.g_dom(nl, "domains", "parent_id", d, 1)))
            parent_grid_ratio.append(int(predict.g_dom(nl, "domains", "parent_grid_ratio", d, 1)))
            i_parent_start.append(int(predict.g_dom(nl, "domains", "i_parent_start", d, 1)))
            j_parent_start.append(int(predict.g_dom(nl, "domains", "j_parent_start", d, 1)))
        ew = predict.g_dom(nl, "domains", "e_we", d)
        es = predict.g_dom(nl, "domains", "e_sn", d)
        if ew is None or es is None:
            raise ColdstartError("domain %d: missing e_we/e_sn in &domains" % (d + 1))
        e_we.append(int(ew))
        e_sn.append(int(es))
    return {"parent_id": parent_id, "parent_grid_ratio": parent_grid_ratio,
            "i_parent_start": i_parent_start, "j_parent_start": j_parent_start,
            "e_we": e_we, "e_sn": e_sn}


def geog_data_path_default():
    return os.environ.get("WPS_GEOG") or "/nfs/data/wrf/WPS_GEOG/"


# ---------------------------------------------------------------------------
# Capability gaps: what GFS (Vtable.GFS) and geog_data_path can actually supply.
#
# Verified on scc, 2026-09-24: Vtable.GFS extracts TT UU VV RH HGT PSFC PMSL SKINTEMP SEAICE
# LANDSEA SNOW SNOWH SOILHGT, 4 soil layers ST/SM (000010/010040/040100/100200), and
# tropopause/max-wind fields -- NO hydrometeors, NO number concentrations, NO aerosols.
# /nfs/data/wrf/WPS_GEOG on scc has only: albedo_modis, greenfrac_fpar_modis, lai_modis_10m,
# lai_modis_30s, maxsnowalb_modis, modis_landuse_20class_30s_with_lakes, orogwd_{10m,1deg,20m,
# 2deg,30m}, soiltemp_1deg, soiltype_{bot,top}_30s, topo_gmted2010_30s, varsso{,_10m,_2m,_5m,30m}
# -- e.g. no lake_depth, no urban parameter tables. These two lists are NOT hardcoded into the
# check below (geog_directory_listing reads the real directory so this also works at a site
# whose geog mirror carries a different, possibly richer, dataset set); only the Vtable.GFS
# field list (WRF's GFS Vtable is effectively fixed across sites) is a fact used to justify
# which physics options this function treats as fundamentally unsupported by cold-start-from-GFS.
# ---------------------------------------------------------------------------

def geog_directory_listing(geog_dir):
    """Lower-cased entry names directly under geog_dir, or None if the directory can't be
    listed (wrong site, no filesystem access, ...) -- callers MUST treat None as 'unknown',
    never as 'nothing present'."""
    try:
        return [n.lower() for n in os.listdir(geog_dir)]
    except OSError:
        return None


def geog_has_dataset(names, *substrings):
    """True/False iff `names` (geog_directory_listing's result) does/doesn't contain an entry
    with every one of `substrings`; None (unknown) when names itself is None."""
    if names is None:
        return None
    return any(all(s in n for s in substrings) for n in names)


# name-fragment -> WRF NUM_LAND_CAT for every landuse table geogrid/real.exe ship with (WRF's
# own LANDUSE.TBL section names) -- used only to explain a num_land_cat mismatch, never to block.
_LANDUSE_NUM_CAT_HINTS = (
    (("modis", "lake"), 21),   # MODIS 20-category + a separate lake category (LANDUSE.TBL MODIS_LAKE)
    (("modis",), 20),
    (("usgs",), 24),
    (("nlcd", "2011"), 50),
    (("nlcd",), 44),
)


def detect_geog_landuse(names):
    """(dataset_name, num_land_cat) for the landuse dataset actually present in `names`
    (geog_directory_listing's result), or (None, None) when none is found / names is None /
    the dataset's category count isn't in the recognised-name table above."""
    if not names:
        return None, None
    candidates = sorted(n for n in names if "landuse" in n)
    if not candidates:
        return None, None
    name = candidates[0]
    for keys, cat in _LANDUSE_NUM_CAT_HINTS:
        if all(k in name for k in keys):
            return name, cat
    return name, None


def _fmt_num(x):
    if x is None:
        return None
    xf = float(x)
    if xf.is_integer():
        return str(int(xf))
    return repr(xf)


WPS_STATIC_DIR_ENV = "WTUNE_WPS_STATIC_DIR"


def wps_constants_for(nl, static_dir=None):
    """Absolute paths for &metgrid constants_name that the case's physics needs but no GFS-family
    analysis carries. Thompson aerosol-aware input (use_aero_icbc / wif_input_opt, per
    https://www2.mmm.ucar.edu/wrf/users/physics/mp28_updated.html): wif_input_opt=2 (WRF 4.4+,
    with black carbon) needs QNWFA_QNIFA_QNBCA_SIGMA_MONTHLY.dat, otherwise
    QNWFA_QNIFA_SIGMA_MONTHLY.dat. lafe (scc job 1009) died in real.exe with 'mp_physics=28 and
    use_aero_icbc=.true. but aerosol climatology field(s) missing'. Raises when a needed file is
    not in static_dir ($WTUNE_WPS_STATIC_DIR) -- at plan time, not an hour later in real.exe."""
    icbc = str(predict.g(nl, "physics", "use_aero_icbc", "") or "").strip(".").lower() in ("true", "t")
    wif = int(predict.g(nl, "domains", "wif_input_opt", 0) or 0)
    if not icbc and wif == 0:
        return []
    name = ("QNWFA_QNIFA_QNBCA_SIGMA_MONTHLY.dat" if wif == 2
            else "QNWFA_QNIFA_SIGMA_MONTHLY.dat")
    static_dir = static_dir or os.environ.get(WPS_STATIC_DIR_ENV)
    path = os.path.join(static_dir, name) if static_dir else None
    if not path or not os.path.isfile(path):
        raise ColdstartError(
            "case needs %s (use_aero_icbc=%s wif_input_opt=%d) for metgrid constants_name; put it "
            "in $%s (got %r)" % (name, icbc, wif, WPS_STATIC_DIR_ENV, static_dir))
    return [os.path.abspath(path)]


def build_namelist_wps_fields(nl, geo_d01, start_dt, end_dt, interval_seconds, geog_data_path=None,
                              prefix="GFS"):
    """Pure builder: the exact set of values namelist.wps will carry, for both rendering and
    direct test assertion (avoids re-parsing rendered text in tests). See SPEC.md Inputs:
    'namelist.wps: max_dom and per-domain lists from namelist.input; start/end = min/max of the
    needed GRIB times; interval_seconds from namelist.input.' `prefix` (&ungrib prefix / &metgrid
    fg_name) is now source-specific -- 'GFS' for any GFS-family source (nomads/aws/rda/NCEI g3/
    g4), 'FNL' for d083002, 'CFSR' for d093000 (see metdata.ForcingSource.prefix)."""
    max_dom = predict.get_max_dom(nl)
    dom = domain_list_fields(nl, max_dom)
    return {
        "max_dom": max_dom,
        "prefix": prefix,
        "map_proj": geo_d01["map_proj"],
        "truelat1": geo_d01.get("truelat1"),
        "truelat2": geo_d01.get("truelat2"),
        "stand_lon": geo_d01.get("stand_lon"),
        "pole_lat": geo_d01.get("pole_lat"),
        "pole_lon": geo_d01.get("pole_lon"),
        "ref_lat": geo_d01["ref_lat"],
        "ref_lon": geo_d01["ref_lon"],
        "dx": geo_d01["dx"],
        "dy": geo_d01["dy"],
        "geog_data_res": [geo_d01.get("geog_data_res") or "default"] * max_dom,
        "geog_data_path": geog_data_path or geog_data_path_default(),
        "parent_id": dom["parent_id"], "parent_grid_ratio": dom["parent_grid_ratio"],
        "i_parent_start": dom["i_parent_start"], "j_parent_start": dom["j_parent_start"],
        "e_we": dom["e_we"], "e_sn": dom["e_sn"],
        "interval_seconds": int(interval_seconds),
        "start_date": start_dt.strftime(_WRF_TIME_FMT),
        "end_date": end_dt.strftime(_WRF_TIME_FMT),
        "constants_name": wps_constants_for(nl),
    }


def _wps_list(values, quote=False):
    def one(v):
        return ("'%s'" % v) if quote else str(v)
    return ", ".join(one(v) for v in values) + ","


def render_namelist_wps(fields):
    """WPS namelist.wps text from build_namelist_wps_fields's dict."""
    max_dom = fields["max_dom"]
    lines = []
    lines.append("&share")
    lines.append(" wrf_core = 'ARW',")
    lines.append(" max_dom = %d," % max_dom)
    lines.append(" start_date = %s" % _wps_list([fields["start_date"]] * max_dom, quote=True))
    lines.append(" end_date   = %s" % _wps_list([fields["end_date"]] * max_dom, quote=True))
    lines.append(" interval_seconds = %d," % fields["interval_seconds"])
    lines.append(" io_form_geogrid = 2,")
    lines.append("/")
    lines.append("")
    lines.append("&geogrid")
    lines.append(" parent_id         = %s" % _wps_list(fields["parent_id"]))
    lines.append(" parent_grid_ratio = %s" % _wps_list(fields["parent_grid_ratio"]))
    lines.append(" i_parent_start    = %s" % _wps_list(fields["i_parent_start"]))
    lines.append(" j_parent_start    = %s" % _wps_list(fields["j_parent_start"]))
    lines.append(" e_we              = %s" % _wps_list(fields["e_we"]))
    lines.append(" e_sn              = %s" % _wps_list(fields["e_sn"]))
    lines.append(" geog_data_res     = %s" % _wps_list(fields["geog_data_res"], quote=True))
    lines.append(" dx                = %s," % _fmt_num(fields["dx"]))
    lines.append(" dy                = %s," % _fmt_num(fields["dy"]))
    lines.append(" map_proj          = '%s'," % fields["map_proj"])
    if fields.get("truelat1") is not None:
        lines.append(" truelat1          = %s," % _fmt_num(fields["truelat1"]))
    if fields.get("truelat2") is not None and fields["map_proj"] == "lambert":
        lines.append(" truelat2          = %s," % _fmt_num(fields["truelat2"]))
    if fields.get("stand_lon") is not None:
        lines.append(" stand_lon         = %s," % _fmt_num(fields["stand_lon"]))
    if fields.get("pole_lat") is not None:
        lines.append(" pole_lat          = %s," % _fmt_num(fields["pole_lat"]))
    if fields.get("pole_lon") is not None:
        lines.append(" pole_lon          = %s," % _fmt_num(fields["pole_lon"]))
    lines.append(" ref_lat           = %s," % _fmt_num(fields["ref_lat"]))
    lines.append(" ref_lon           = %s," % _fmt_num(fields["ref_lon"]))
    lines.append(" geog_data_path    = '%s'" % fields["geog_data_path"])
    lines.append("/")
    lines.append("")
    prefix = fields.get("prefix") or "GFS"
    lines.append("&ungrib")
    lines.append(" out_format = 'WPS',")
    lines.append(" prefix = '%s'," % prefix)
    lines.append("/")
    lines.append("")
    lines.append("&metgrid")
    lines.append(" fg_name = '%s'," % prefix)
    if fields.get("constants_name"):
        lines.append(" constants_name = %s" % _wps_list(fields["constants_name"], quote=True))
    lines.append(" io_form_metgrid = 2,")
    lines.append("/")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def inspect_case(case_dir, namelist=None, namelist_wps=None, geo_from_wrfinput=None,
                  grib_dir=None, ncdump=None, now=None, geog_path=None):
    """Returns (rows, blockers, nl) -- rows: [(item, present, how, note)], blockers: [str].
    Exit-code contract (blocked -> 3, else 0) lives in main()."""
    namelist_path = namelist or os.path.join(case_dir, "namelist.input")
    rows = []
    blockers = []

    if not os.path.exists(namelist_path):
        blockers.append("no namelist.input at %s" % namelist_path)
        return rows, blockers, None
    nl = load_namelist(namelist_path)
    rows.append(("namelist.input", True, namelist_path, ""))

    max_dom = predict.get_max_dom(nl)
    try:
        start_dt, end_dt, run_s = run_span(nl)
        rows.append(("run length", True, "%.0fs" % run_s, ""))
    except ColdstartError as exc:
        blockers.append(str(exc))
        start_dt = end_dt = None

    # Projection source
    nlw_path = namelist_wps or os.path.join(case_dir, "namelist.wps")
    wi_path = geo_from_wrfinput or os.path.join(case_dir, "wrfinput_d01")
    try:
        geo = resolve_projection(nl, nlw_path if os.path.exists(nlw_path) else None,
                                  wi_path if os.path.exists(wi_path) else None, ncdump=ncdump)
        rows.append(("projection", True, geo["source"], "map_proj=%s" % geo["map_proj"]))
    except ColdstartError as exc:
        blockers.append(str(exc))
        geo = None

    # ifire (WRF-SFIRE)
    ifire = 0
    for d in range(max_dom):
        v = predict.g_dom(nl, "physics", "ifire", d, 0) or 0
        ifire = max(ifire, int(v))
    if ifire > 0:
        blockers.append("ifire=%d in &physics (WRF-SFIRE needs fire geog data; unsupported)" % ifire)
    rows.append(("ifire", ifire == 0, str(ifire), "" if ifire == 0 else "WRF-SFIRE unsupported"))

    # Capability gaps: what GFS (Vtable.GFS) + geog_data_path can actually supply (see
    # geog_directory_listing / _LANDUSE_NUM_CAT_HINTS docstrings for the verified facts this is
    # built on). Blockers: sf_lake_physics with no lake_depth dataset; sf_urban_physics in
    # (2, 3) (BEP/BEM need urban parameter data no geog mirror carries); chem_opt>0 (WRF-Chem
    # emissions); obs_nudge_opt>0 (observation files). Warnings ("filled differently: ..."):
    # completable, but real.exe/wrf.exe will not do quite what the case's own namelist implies.
    geog_dir = geog_path or geog_data_path_default()
    geog_names = geog_directory_listing(geog_dir)

    sf_lake_physics = int(predict.g_dom(nl, "physics", "sf_lake_physics", 0, 0) or 0)
    sf_urban_physics = int(predict.g_dom(nl, "physics", "sf_urban_physics", 0, 0) or 0)
    chem_opt = int(predict.g_dom(nl, "chem", "chem_opt", 0, 0) or 0)
    obs_nudge_opt = int(predict.g_dom(nl, "fdda", "obs_nudge_opt", 0, 0) or 0)
    mp_physics = int(predict.g_dom(nl, "physics", "mp_physics", 0, 0) or 0)
    sf_surface_physics = int(predict.g_dom(nl, "physics", "sf_surface_physics", 0, 0) or 0)
    num_soil_layers = predict.g_dom(nl, "physics", "num_soil_layers", 0, None)
    grid_fdda = int(predict.g_dom(nl, "fdda", "grid_fdda", 0, 0) or 0)
    num_land_cat_nl = predict.g(nl, "domains", "num_land_cat", None)

    if sf_lake_physics > 0:
        has_lake = geog_has_dataset(geog_names, "lake_depth")
        if has_lake is False:
            blockers.append("sf_lake_physics=%d but no lake_depth dataset under %s"
                            % (sf_lake_physics, geog_dir))
        elif has_lake is None:
            rows.append(("geog: lake_depth", False, "skipped",
                        "could not list %s to check for a lake_depth dataset" % geog_dir))
        else:
            rows.append(("geog: lake_depth", True, "present", ""))

    if sf_urban_physics in (2, 3):
        blockers.append("sf_urban_physics=%d (BEP/BEM) needs urban parameter data not in geog"
                        % sf_urban_physics)

    if chem_opt > 0:
        # WRF-Chem is "grid/rank only" for WTune (user decision 2026-09-29): chem_opt/chemdt/photdt/
        # emissions are left exactly as given, so chem_opt>0 alone no longer blocks. What still
        # blocks is a real crash risk: wrf.exe aborts opening emission/chem-input files that the
        # case does not carry (real.exe + GFS cannot produce them).
        emiss_opt = int(predict.g_dom(nl, "chem", "emiss_inpt_opt", 0, 1) or 0)   # Registry default 1
        io_emiss = int(predict.g_dom(nl, "chem", "io_style_emissions", 0, 0) or 0)
        chem_in_opt = int(predict.g_dom(nl, "chem", "chem_in_opt", 0, 0) or 0)
        have = lambda pat: bool(glob.glob(os.path.join(case_dir, pat)))
        missing = []
        if (emiss_opt > 0 or io_emiss > 0) and not have("wrfchemi_d*"):
            missing.append("emission files wrfchemi_d* (emiss_inpt_opt=%d, io_style_emissions=%d)"
                           % (emiss_opt, io_emiss))
        if chem_in_opt == 1 and not have("wrf_chem_input*") and not have("wrfchem_input*"):
            missing.append("chemistry initial-condition file (chem_in_opt=1)")
        if missing:
            blockers.append("chem_opt=%d (WRF-Chem) needs input files this case does not carry: %s "
                            "-- wrf.exe would abort at start-up (real crash risk); WTune cannot "
                            "generate them" % (chem_opt, "; ".join(missing)))
        else:
            rows.append(("WRF-Chem", True, "chem_opt=%d" % chem_opt,
                         "grid/rank tuning only: chem settings untouched, time prediction UNCALIBRATED"))

    if obs_nudge_opt > 0:
        blockers.append("obs_nudge_opt=%d needs observation files (unsupported)" % obs_nudge_opt)

    import wtune_extensions
    for _d in wtune_extensions.detect(nl, case_dir):
        if _d["id"] in ("hydro", "movenest", "wrfda") and _d["active"]:
            rows.append((_d["label"], True, "; ".join(_d["evidence"]),
                         "grid/rank tuning only: extension settings untouched"))

    capability_warnings = [
        "filled differently: hydrometeors and number concentrations start at zero (GFS via "
        "Vtable.GFS carries no hydrometeor/number-concentration fields) -- this is why "
        "cold-start windows discard the first ~1 simulated hour as spin-up",
    ]
    if mp_physics == 28:
        capability_warnings.append(
            "filled differently: mp_physics=28 (aerosol-aware Thompson) but GFS has no aerosol "
            "fields and no climatology file is wired in")
    landuse_name, landuse_cat = detect_geog_landuse(geog_names)
    if (landuse_cat is not None and num_land_cat_nl is not None
            and int(num_land_cat_nl) != landuse_cat):
        capability_warnings.append(
            "filled differently: case specifies num_land_cat=%s but the only landuse dataset "
            "present (%s) has %d categories -- real.exe will run with %d"
            % (num_land_cat_nl, landuse_name, landuse_cat, landuse_cat))
    if sf_surface_physics == 3 and num_soil_layers is not None and int(num_soil_layers) != 4:
        capability_warnings.append(
            "filled differently: sf_surface_physics=3 (RUC) with num_soil_layers=%s -- real.exe "
            "vertically interpolates GFS's 4 soil layers" % num_soil_layers)
    if grid_fdda > 0:
        capability_warnings.append(
            "filled differently: grid_fdda=%d -- analysis nudging files are only generated for "
            "the sampled windows' own span, not the whole run" % grid_fdda)
    for w in capability_warnings:
        rows.append(("capability", True, "warning", w))

    # Nest start-time consistency
    if max_dom > 1:
        d01_start = tuple(predict.g_dom(nl, "time_control", "start_" + k, 0)
                           for k in ("year", "month", "day", "hour", "minute", "second"))
        for d in range(1, max_dom):
            dn_start = tuple(predict.g_dom(nl, "time_control", "start_" + k, d)
                              for k in ("year", "month", "day", "hour", "minute", "second"))
            if dn_start != d01_start:
                blockers.append(
                    "domain %d start time %s differs from d01's %s (unsupported)"
                    % (d + 1, dn_start, d01_start))
        rows.append(("nest start times", not any("differs from d01" in b for b in blockers),
                     "max_dom=%d" % max_dom, ""))

    # Forcing chain (metdata.resolve_forcing_chain -- date rules only, no network access; rows
    # say 'not probed', matching SPEC item F). Lazily imports metdata (needs numpy) -- degrade
    # with a clear row/message rather than crash when numpy isn't installed on whatever python3
    # is running `inspect` (see _metdata()'s docstring). 2026-09-24 FIX: a run before
    # 2015-01-15 is NO LONGER a blocker by itself -- the OLD single-source rule
    # (select_source: GFS 0.25deg or nothing) wrongly reported every such case as missing
    # weather data, even though NCEI's GFS analysis archive (0.5/1deg, from 2004), NCEP's FNL
    # (1deg, from 1999-07-30), and CFSR reanalysis (0.5deg, 1979-2010) all cover it. The ONLY
    # remaining blocker is a genuinely empty chain (older than 1979, or a date gap none of the
    # bands cover) with no --grib-dir. A coarser-than-0.25deg forcing, or a reanalysis fallback,
    # is a WARNING (extra prediction uncertainty), never a blocker -- and WRF's own dx being much
    # finer than the forcing resolution is never checked here at all (that's normal downscaling,
    # not a capability gap).
    if start_dt is not None:
        try:
            md_mod = _metdata()
        except ColdstartError as exc:
            rows.append(("forcing chain", False, "skipped", str(exc)))
        else:
            for label, t in (("run start", start_dt), ("run end", end_dt)):
                chain = md_mod.resolve_forcing_chain(t, now=now, grib_dir=grib_dir)
                if not chain:
                    blockers.append(
                        "%s %s: no forcing source covers this date (no --grib-dir given)"
                        % (label, t.strftime(_WRF_TIME_FMT)))
                    rows.append(("forcing chain (%s)" % label, False, "NONE",
                                "no candidate source and no --grib-dir"))
                    continue
                primary = chain[0]
                how = " -> ".join(s.dataset for s in chain)
                rows.append((
                    "forcing chain (%s)" % label, True, how,
                    "not probed (date rules only): %s %sdeg %s interval=%sh %s"
                    % (primary.dataset, primary.resolution_deg, primary.format,
                       primary.forcing_interval_h, primary.source_kind)))
                if primary.resolution_deg is not None and primary.resolution_deg > 0.25:
                    rows.append(("capability", True, "warning",
                                "filled differently: forcing resolution %sdeg (coarser than "
                                "0.25) at %s -- extra prediction uncertainty"
                                % (primary.resolution_deg, t.strftime(_WRF_TIME_FMT))))
                if primary.source_kind == "reanalysis":
                    rows.append(("capability", True, "warning",
                                "filled differently: reanalysis_fallback at %s (%s) -- CFSR is "
                                "a reanalysis, not an operational analysis/forecast"
                                % (t.strftime(_WRF_TIME_FMT), primary.dataset)))

    return rows, blockers, nl


def print_inspect(rows, blockers):
    print("%-28s %-6s %-18s %s" % ("item", "ok", "how", "note"))
    for item, present, how, note in rows:
        print("%-28s %-6s %-18s %s" % (item, "yes" if present else "no", how, note))
    if blockers:
        print()
        print("BLOCKED:")
        for b in blockers:
            print("  - %s" % b)


# ---------------------------------------------------------------------------
# Domain sample lattice (for ranking day/night candidates -- SPEC.md Window
# selection point 3: "approximated from ref_lat/ref_lon and the d01 extent")
# ---------------------------------------------------------------------------

def domain_lattice_from_geom(ref_lat, ref_lon, dx_m, e_we, dy_m=None, e_sn=None):
    """5x5 sample lattice (wtune_predict.sample_lattice_5x5) approximated from the d01 center
    and extent: half-width = dx*(e_we-1)/2 converted from meters to km to degrees (1 deg lat ~=
    111 km; lon degrees scaled by 1/cos(lat) for the same km extent)."""
    dy_m = dx_m if dy_m is None else dy_m
    e_sn = e_we if e_sn is None else e_sn
    half_width_km = dx_m * (e_we - 1) / 2.0 / 1000.0
    half_height_km = dy_m * (e_sn - 1) / 2.0 / 1000.0
    dlat = half_height_km / 111.0
    coslat = math.cos(math.radians(ref_lat))
    dlon = half_width_km / (111.0 * coslat) if abs(coslat) > 1e-9 else half_width_km / 111.0
    sw = (ref_lat - dlat, ref_lon - dlon)
    nw = (ref_lat + dlat, ref_lon - dlon)
    ne = (ref_lat + dlat, ref_lon + dlon)
    se = (ref_lat - dlat, ref_lon + dlon)
    center = (ref_lat, ref_lon)
    return predict.sample_lattice_5x5(sw, nw, ne, se, center)


def measured_window_mean_sunlit(sample_points, t_start, t_end, n_samples=5):
    """Mean of wtune_predict.sunlit_fraction over n_samples evenly spaced times in
    [t_start, t_end] -- "approximate is fine for ranking candidates" (SPEC.md). None when
    sample_points is empty/None."""
    if not sample_points:
        return None
    if n_samples <= 1:
        times = [t_start]
    else:
        step = (t_end - t_start) / (n_samples - 1)
        times = [t_start + step * i for i in range(n_samples)]
    fracs = [predict.sunlit_fraction(sample_points, t) for t in times]
    fracs = [f for f in fracs if f is not None]
    return sum(fracs) / len(fracs) if fracs else None


# ---------------------------------------------------------------------------
# Window selection (plan)
# ---------------------------------------------------------------------------

def compute_window_seconds(spinup_min, dt_seconds, radt_minutes, measure_steps=None):
    """W = spinup_min*60 + measure_steps*dt, measure_steps = max(30, 2*stepra) by default, stepra =
    wtune_predict.steps_per_radiation_call(radt, dt) -- so every window contains >= 2
    post-spin-up radiation steps. An explicit `measure_steps` (>= stepra + 1, so at least one
    radiation step is measured) shortens the window; scoring then locates radiation steps by
    their known phase instead of detecting it. Returns (W_seconds, stepra, measure_steps)."""
    stepra = predict.steps_per_radiation_call(radt_minutes, dt_seconds)
    if measure_steps is None:
        measure_steps = max(30, 2 * stepra)
    elif int(measure_steps) < stepra + 1:
        raise ColdstartError("measure_steps %d < stepra+1 (%d): no radiation step would be measured"
                             % (measure_steps, stepra + 1))
    W = spinup_min * 60.0 + measure_steps * dt_seconds
    return W, stepra, measure_steps


def candidate_times(t0, run_end_dt, interval_seconds, W_seconds):
    """GRIB-aligned candidates t0 + j*interval_seconds with t + W <= run_end_dt."""
    out = []
    j = 0
    while True:
        t = t0 + datetime.timedelta(seconds=interval_seconds * j)
        if t + datetime.timedelta(seconds=W_seconds) > run_end_dt:
            break
        out.append(t)
        j += 1
    return out


def _farthest_from(candidates, picked):
    best, best_dist = None, -1.0
    for c in candidates:
        if c in picked:
            continue
        dist = min(abs((c - p).total_seconds()) for p in picked) if picked else 0.0
        if dist > best_dist:
            best, best_dist = c, dist
    return best


def _select_by_cost(candidates, cost_lookup, K):
    """Cost-indicator mode of select_windows (decision 0060). t0 always; the argmax-c candidate
    ("peak") always; then the candidates are ranked by c and cut into min(K, n) equal-count quantile
    strata, and every stratum not already holding a pick (t0 / peak count for their own stratum)
    contributes the member whose c is nearest the stratum's median c, highest-c stratum first. Left-
    over slots go to the farthest-in-time candidate, as in the time mode. Reasons:
    "t0", "peak", "stratum j/K (c=...)". Returns None when no candidate has a c value."""
    scored = [c for c in candidates if cost_lookup.get(c) is not None]
    if not scored:
        return None
    picks, picked = [], set()

    def add(t, reason):
        if t not in picked and len(picks) < K:
            picks.append((t, reason))
            picked.add(t)

    peak = max(scored, key=lambda c: (cost_lookup[c], -candidates.index(c)))
    add(candidates[0], "t0+peak" if peak == candidates[0] else "t0")
    add(peak, "peak")
    ordered = sorted(scored, key=lambda c: (cost_lookup[c], c))
    n, Ks = len(ordered), min(K, len(ordered))
    groups = [ordered[j * n // Ks:(j + 1) * n // Ks] for j in range(Ks)]
    for j in range(Ks - 1, -1, -1):
        g = groups[j]
        if any(p in g for p in picked):
            continue
        vals = sorted(cost_lookup[c] for c in g)
        h = len(vals) // 2
        med = vals[h] if len(vals) % 2 else 0.5 * (vals[h - 1] + vals[h])
        best = min(g, key=lambda c: (abs(cost_lookup[c] - med), c))
        add(best, "stratum %d/%d (c=%.2f)" % (j + 1, Ks, cost_lookup[best]))
    while len(picks) < K:
        nxt = _farthest_from(candidates, picked)
        if nxt is None:
            break
        add(nxt, "farthest")
    return picks


def select_windows(candidates, spinup_seconds, W_seconds, K, sample_points=None,
                    precip_lookup=None, sunlit_shift_s=0.0, cost_lookup=None):
    """Picks (deduplicated to distinct candidates, capped at K), each with a reason:
      1. t0 -- always.
      2. night -- lowest mean sunlit fraction over the measured part.
      3. day -- highest mean sunlit fraction.
      4. late -- the last candidate.
      5. precip_peak (only when precip_lookup is given) -- candidate with max domain precip
         fraction, precip_lookup: {candidate_dt: precip_fraction}.
    Remaining slots up to K filled by the candidate farthest in time from all picked ones.
    `sunlit_shift_s` (R6-D, date-shifted sampling): candidates are on the ANALOG timeline (the
    weather actually being sampled/run), but night/day ranking must use the STATED timeline's
    own sun position (the sun the PREDICTED run will actually see) -- so the sunlit-fraction
    evaluation subtracts this shift (candidate - shift_s) before scoring, while every OTHER use
    of `candidates` (t0/late/precip_peak/farthest, and the returned pick times themselves) stays
    on the analog timeline unchanged. 0.0 (default) reproduces the pre-R6-D behaviour exactly.
    `cost_lookup` ({candidate_dt: c}, decision 0060; None = the behaviour above, unchanged): pick by
    weather cost proxy instead -- see _select_by_cost; falls through to the time mode when no
    candidate has a c value.
    Returns [(dt, reason), ...] in pick order (analog-timeline datetimes)."""
    if not candidates:
        return []
    if cost_lookup:
        by_cost = _select_by_cost(candidates, cost_lookup, K)
        if by_cost is not None:
            return by_cost
    picks = []
    picked_set = set()
    sunlit_shift = datetime.timedelta(seconds=sunlit_shift_s)

    def add(t, reason):
        if t is not None and t not in picked_set and len(picks) < K:
            picks.append((t, reason))
            picked_set.add(t)

    add(candidates[0], "t0")

    if sample_points and len(picks) < K:
        scored = []
        for c in candidates:
            c_sun = c - sunlit_shift
            frac = measured_window_mean_sunlit(sample_points, c_sun + datetime.timedelta(seconds=spinup_seconds),
                                                c_sun + datetime.timedelta(seconds=W_seconds))
            if frac is not None:
                scored.append((c, frac))
        if scored:
            night_c = min(scored, key=lambda cf: cf[1])[0]
            day_c = max(scored, key=lambda cf: cf[1])[0]
            add(night_c, "night")
            add(day_c, "day")

    add(candidates[-1], "late")

    if precip_lookup and len(picks) < K:
        scored = [(c, precip_lookup[c]) for c in candidates if c in precip_lookup]
        if scored:
            peak_c = max(scored, key=lambda cf: cf[1])[0]
            add(peak_c, "precip_peak")

    while len(picks) < K:
        nxt = _farthest_from(candidates, picked_set)
        if nxt is None:
            break
        add(nxt, "farthest")

    return picks


def next_grib_time_ge(t0, interval_seconds, target):
    """Smallest t0 + j*interval_seconds (j>=0) that is >= target -- used for a window's
    real.exe boundary end time (SPEC.md: 'end = T_k_end (next GRIB time >= T_k + window
    length)')."""
    if target <= t0:
        return t0
    j = math.ceil((target - t0).total_seconds() / interval_seconds)
    return t0 + datetime.timedelta(seconds=interval_seconds * j)


def align_to_interval(t, interval_h):
    """Smallest UTC time >= t that is an integer multiple of interval_h hours since the UNIX
    epoch (equivalently, since midnight -- every forcing_interval_h this tool uses, 3 or 6,
    divides 24 evenly, so epoch-alignment and midnight-alignment agree). 2026-09-24: window
    starts must be VALID TIMES OF THE FORCING (e.g. FNL's 00/06/12/18Z), not
    t0 + j*case_interval_seconds -- a case whose own interval_seconds doesn't match its actual
    forcing source silently produced windows metgrid could never satisfy."""
    interval_s = interval_h * 3600
    epoch = datetime.datetime(1970, 1, 1)
    secs = (t - epoch).total_seconds()
    aligned = math.ceil(secs / interval_s) * interval_s
    return epoch + datetime.timedelta(seconds=aligned)


def window_valid_times(T, T_end, interval_seconds):
    """[T, T+interval, ..., T_end] inclusive -- the WPS/ungrib valid times a window of
    [T, T_end] at `interval_seconds` cadence needs."""
    out = []
    t = T
    while t <= T_end:
        out.append(t)
        t += datetime.timedelta(seconds=interval_seconds)
    return out


def expected_wps_intermediate_files(prefix, T, T_end, interval_seconds):
    """['<prefix>:YYYY-MM-DD_HH', ...] -- the exact ungrib intermediate-file names a window
    [T, T_end] must produce at `interval_seconds` cadence. 2026-09-24: run.sbatch used to only
    check 'ungrib produced >=1 GFS:* file'; that passes even when ungrib silently skipped a
    fetched-but-out-of-range time, so real.exe/metgrid would read a truncated boundary set
    without the run ever failing at the step that actually caused it."""
    return ["%s:%s" % (prefix, t.strftime("%Y-%m-%d_%H")) for t in window_valid_times(T, T_end, interval_seconds)]


def expected_met_em_files(max_dom, T, T_end, interval_seconds):
    """['met_em.d01.<time>.nc', ...] a window [T, T_end] must produce from metgrid: d01 at
    EVERY valid time in the window (it has the time-varying lateral boundary); a nest d02..dN
    only at the window's own start T (nests are initialized once, not boundary-forced -- see
    render_run_sbatch's wrf-dir linking comment)."""
    out = ["met_em.d01.%s.nc" % t.strftime("%Y-%m-%d_%H:%M:%S")
           for t in window_valid_times(T, T_end, interval_seconds)]
    for d in range(2, max_dom + 1):
        out.append("met_em.d%02d.%s.nc" % (d, T.strftime("%Y-%m-%d_%H:%M:%S")))
    return out


# ---------------------------------------------------------------------------
# Forcing source fallback chain orchestration (thin wrapper over metdata.ForcingSource /
# resolve_forcing_chain / probe_forcing_entries -- see that module for the verified URL patterns
# and date bands). coldstart_sample.py owns WHICH window uses WHICH source and cadence
# realignment; metdata.py owns URL construction and the HTTP probe.
# ---------------------------------------------------------------------------

ERA5_DATASET = "era5_arco"


def _era5_source():
    """ERA5 from the public ARCO store, written straight to WPS intermediate files (no GRIB, no
    ungrib) by tools/era5_intermediate.py -- hourly, with cloud water/ice (and rain/snow from the
    137-level store when WTUNE_ERA5_RAIN_SNOW=1), so a rebuild segment starts with clouds
    (decision 0054 phase 3). Only reachable through an explicit forcing pin."""
    md_mod = _metdata()
    return md_mod.ForcingSource("arco-gcs", ERA5_DATASET, "era5-0p25", 0.25, "wps-intermediate",
                                1, "reanalysis", "none", "ERA5",
                                datetime.datetime(1940, 1, 1), None, lambda dt, host=None: [])


def _forcing_sources_by_dataset():
    md_mod = _metdata()
    srcs = {s.dataset: s for s in (md_mod.SRC_NOMADS_WPS, md_mod.SRC_AWS_WPS,
                                   md_mod.SRC_RDA_D084001, md_mod.SRC_NCEI_G4, md_mod.SRC_NCEI_G3,
                                   md_mod.SRC_FNL, md_mod.SRC_CFSR)}
    srcs[ERA5_DATASET] = _era5_source()
    return srcs


_NETCDF_LOCK = threading.Lock()


def _d01_bbox_from_case(case_dir):
    """(lat_min, lat_max, lon_min, lon_max) of the case's own wrfinput_d01 grid. Serialized:
    wtune_tune.run_rebuild_points fetches points from a thread pool, and netCDF-C is not
    thread-safe -- concurrent opens corrupted its ID table and a later close in the same process
    failed with "NetCDF: Not a valid ID" (dorian2019 short-window run, 2026-09-27)."""
    import netCDF4
    with _NETCDF_LOCK, netCDF4.Dataset(os.path.join(case_dir, "wrfinput_d01")) as d:
        lat, lon = d["XLAT"][0], d["XLONG"][0]
        return (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))


def _fetch_era5_intermediate(plan, plan_dir, runner=None):
    """fetch for the ERA5 pin: write ERA5:<time> intermediate files for every hourly valid time
    of every window into PLAN/wps via the envs/ai interpreter (xarray/zarr/gcsfs live there)."""
    src = _era5_source()
    times = sorted({t for w in plan["windows"]
                    for t in window_valid_times(datetime.datetime.strptime(w["t"], _WRF_TIME_FMT),
                                                datetime.datetime.strptime(w["t_end"], _WRF_TIME_FMT),
                                                3600)})
    bbox = _d01_bbox_from_case(plan["case_dir"])
    wps_dir = os.path.join(plan_dir, "wps")
    os.makedirs(wps_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ai_python = os.environ.get("WTUNE_AI_PYTHON",
                               os.path.join(repo, "envs", "ai", ".pixi", "envs", "default", "bin", "python"))
    cmd = [ai_python, os.path.join(repo, "src", "tools", "era5_intermediate.py"),
           "--bbox", ",".join("%.3f" % v for v in bbox),
           "--times", ",".join(t.strftime("%Y-%m-%d_%H") for t in times),
           "--out", wps_dir, "--prefix", src.prefix]
    if os.environ.get("WTUNE_ERA5_RAIN_SNOW") == "1":
        cmd.append("--rain-snow")
    (runner or (lambda c: subprocess.run(c, check=True)))(cmd)
    names = ["%s:%s" % (src.prefix, t.strftime("%Y-%m-%d_%H")) for t in times]
    missing = [n for n in names if not os.path.exists(os.path.join(wps_dir, n))]
    if missing:
        raise ColdstartError("fetch: ERA5 intermediate files missing after generation: %s"
                             % ", ".join(missing))
    total_bytes = sum(os.path.getsize(os.path.join(wps_dir, n)) for n in names)
    windows_out = [{"win": k, "chosen": src.to_dict(), "candidates_tried": []}
                   for k in range(len(plan["windows"]))]
    forcing = {"windows": windows_out, "files": names, "intermediate": True}
    with open(os.path.join(plan_dir, "forcing.json"), "w") as f:
        json.dump(forcing, f, indent=2)
    open(os.path.join(plan_dir, "grib_manifest.txt"), "w").close()
    print("fetched %d bytes of ERA5 intermediate files (%d valid times)" % (total_bytes, len(names)))
    return total_bytes, forcing


def resolve_forcing_chain_for(valid_dt, grib_dir=None, forcing_pin=None, now=None):
    """metdata.resolve_forcing_chain(valid_dt), or a single-source chain pinned to
    `forcing_pin` (SPEC 2026-09-24 item D's `--forcing <dataset>` override, used to re-run `plan`
    against a specific dataset after `fetch` reports a cadence-mismatched fallback)."""
    if forcing_pin:
        src = _forcing_sources_by_dataset().get(forcing_pin)
        if src is None:
            raise ColdstartError(
                "--forcing %r is not a known dataset (know: %s)"
                % (forcing_pin, sorted(_forcing_sources_by_dataset())))
        return [src]
    md_mod = _metdata()
    return md_mod.resolve_forcing_chain(valid_dt, now=now, grib_dir=grib_dir)


def choose_forcing_for_window(T, T_end, chain, probe_fn=None):
    """Walk `chain` (resolve_forcing_chain_for's result) in order; the first candidate whose
    probe finds every needed file for EVERY valid time in [T, T_end] (at that candidate's OWN
    forcing_interval_h) wins -- SPEC item B: 'a window uses ONE source that covers ALL its valid
    times.' probe_fn(entries) -> (all_found, results); defaults to metdata.probe_forcing_entries
    (real HTTP, via check_url) -- offline tests inject a mock here, never touch the network.
    Returns (chosen_source_or_None, candidates_tried: [{'dataset', 'ok', 'n_files', 'missing',
    'first_missing': {'url', 'final_status'} or None}]) -- `first_missing` (T4b, 2026-09-24) is
    the FIRST url that was still missing after every retry, for a clear failure message ('fetch:
    ... first missing: <url> (final status <n>)') rather than just a filename list."""
    md_mod = _metdata()
    probe_fn = probe_fn or md_mod.probe_forcing_entries
    tried = []
    for src in chain:
        valid_times = window_valid_times(T, T_end, src.forcing_interval_h * 3600)
        entries = []
        for vt in valid_times:
            entries.extend(src.urls_for(vt))
        ok, results = probe_fn(entries)
        first_missing = None
        for r in results:
            if r["chosen_url"] is None:
                checked0 = r["checked"][0] if r.get("checked") else {}
                attempts = checked0.get("attempts") or [{"status": checked0.get("status")}]
                first_missing = {"url": checked0.get("url"), "final_status": attempts[-1].get("status"),
                                 "detail": attempts[-1].get("detail") or checked0.get("detail")}
                break
        tried.append({"dataset": src.dataset, "ok": ok, "n_files": len(entries),
                      "missing": [r["filename"] for r in results if r["chosen_url"] is None],
                      "first_missing": first_missing})
        if ok:
            return src, tried
    return None, tried


def try_precip_lookup(case_dir, nl, candidates, W_seconds, grib_dir=None, source="auto",
                      cache_dir=None):
    """--state precip_peak candidate lookup via tools/grib_state.py's trajectory (import, not a
    subprocess). Returns (lookup_dict_or_None, skip_reason_or_None). Best-effort: any failure
    (missing wrfinput for the target grid, no network, missing deps) is caught and reported as a
    skip reason, never raised -- SPEC.md: 'if grib_state needs a wrfinput for its target grid and
    none exists, print that --state was skipped and why.'"""
    case = {
        "start_year": predict.g(nl, "time_control", "start_year"),
        "start_month": predict.g(nl, "time_control", "start_month"),
        "start_day": predict.g(nl, "time_control", "start_day"),
        "start_hour": predict.g(nl, "time_control", "start_hour", 0),
        "run_total_hours": (candidates[-1] - candidates[0]).total_seconds() / 3600.0 + W_seconds / 3600.0
                            if candidates else 0.0,
    }
    try:
        gs_mod = _grib_state()
        md_mod = _metdata()
    except ColdstartError as exc:
        return None, "--state skipped: %s" % exc
    wi_path = os.path.join(case_dir, "wrfinput_d01") if case_dir else None
    try:
        lat2d, lon2d, _mode = gs_mod.target_grid(case, wi_path)
    except Exception as exc:
        return None, "--state skipped: no target grid (%s)" % exc
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "coldstart_sample_state_cache")
    lookup = {}
    try:
        for c in candidates:
            # precip fraction needs a real GRIB fetch (apcp); reuse grib_state's own per-time
            # fetch path via trajectory() would refetch every field -- instead ask metdata
            # directly for just apcp at this one time, mirroring trajectory()'s own call.
            cycle_dt, fhour = md_mod.gfs_cycle_for(c)
            src_choice, reason = md_mod.select_source(c, override=source, grib_dir=grib_dir)
            if src_choice is None:
                return None, "--state skipped: %s" % reason
            apcp_cycle, apcp_fhour, apcp_window, _w = md_mod.resolve_apcp_request(cycle_dt, fhour)
            slat, slon, sval = md_mod.fetch_and_decode_state_field(
                "apcp", apcp_cycle, apcp_fhour, src_choice, cache_dir, grib_dir=grib_dir)
            slat, slon, sval = md_mod.crop_bbox(slat, slon, sval, float(lat2d.min()), float(lat2d.max()),
                                                 float(lon2d.min()), float(lon2d.max()))
            grid = gs_mod.regrid_nearest_regular(slat, slon, sval, lat2d, lon2d)
            rate = gs_mod.precip_rate_from_apcp_window(grid, apcp_window)
            lookup[c] = gs_mod.precip_frac(rate)
        return lookup, None
    except Exception as exc:
        return None, "--state skipped: %s" % exc


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def build_plan(case_dir, out_dir, k=4, nodes=1, ranks=72, omp=1, nx=None, ny=None,
               spinup_min=60, namelist=None, namelist_wps=None, geo_from_wrfinput=None,
               grib_dir=None, ncdump=None, use_state=False, now=None, geog_path=None,
               forcing=None, date_shift_to=None, analog_path=None, times=None,
               real_only=False, ml_model=None, numtiles=1, measure_steps=None, io_probe=False):
    """Writes OUT_DIR/plan.json + namelist.wps. `io_probe` (default False: plan.json unchanged)
    adds plan['io_probe'] (plan_io_probe), applied to the LAST window -- see apply_io_probe."""
    rows, blockers, nl = inspect_case(case_dir, namelist=namelist, namelist_wps=namelist_wps,
                                       geo_from_wrfinput=geo_from_wrfinput, grib_dir=grib_dir,
                                       ncdump=ncdump, now=now, geog_path=geog_path)
    if blockers:
        raise ColdstartError("plan: case is blocked:\n  - " + "\n  - ".join(blockers))

    max_dom = predict.get_max_dom(nl)
    dt = predict.get_timestep_seconds(nl)
    radt = predict.g_dom(nl, "physics", "radt", 0)
    if not radt:
        raise ColdstartError("&physics radt (domain 1) is required for window sizing")
    start_dt, end_dt, run_s = run_span(nl)   # the case's own STATED span -- unshifted
    stated_start = start_dt
    case_interval_seconds = predict.g(nl, "time_control", "interval_seconds", None)
    case_interval_seconds = int(case_interval_seconds) if case_interval_seconds else None

    # R6-D date-shifted sampling (analog): resolve an ANALOG start, either pinned directly
    # (--date-shift-to) or from a prior `analog` search's own output (--analog OUT/analog.json).
    # Everything that decides WHICH weather to fetch/run (candidate generation, forcing chain
    # resolution, cadence alignment) runs on the analog timeline below -- that is the weather
    # actually being sampled. run_start/run_end (what `score`'s WHOLE-RUN prediction targets --
    # the STATED/future run) stay on the stated timeline, unshifted.
    if date_shift_to and analog_path:
        raise ColdstartError("plan: --date-shift-to and --analog are mutually exclusive")
    analog_start = None
    analog_quality = None
    if date_shift_to:
        analog_start = datetime.datetime.strptime(date_shift_to, _WRF_TIME_FMT)
        analog_quality = "manual"   # a direct pin carries no analog-search quality score
    elif analog_path:
        with open(analog_path) as f:
            analog_data = json.load(f)
        chosen = analog_data.get("chosen") or {}
        if not chosen.get("date"):
            raise ColdstartError("plan: --analog %s has no 'chosen' analog date" % analog_path)
        analog_start = datetime.datetime.strptime(chosen["date"], _WRF_TIME_FMT)
        analog_quality = analog_data.get("quality")
    date_shift_s = (analog_start - stated_start).total_seconds() if analog_start else 0.0
    window_anchor_start = analog_start or stated_start
    window_anchor_end = window_anchor_start + datetime.timedelta(seconds=run_s)
    if analog_start:
        print("date-shifted sampling: stated=%s analog=%s shift_s=%+.0f (quality=%s)"
              % (stated_start.strftime(_WRF_TIME_FMT), analog_start.strftime(_WRF_TIME_FMT),
                 date_shift_s, analog_quality))

    # Forcing cadence: window starts must be VALID TIMES OF THE FORCING (2026-09-24), driven by
    # the FIRST candidate of the resolved chain (--forcing pins it explicitly) -- NOT the case's
    # own interval_seconds, which may not even match what's actually fetchable (see `fetch`'s own
    # cadence-mismatch check when a coarser fallback source gets chosen at download time).
    chain = resolve_forcing_chain_for(window_anchor_start, grib_dir=grib_dir, forcing_pin=forcing,
                                      now=now)
    if not chain:
        raise ColdstartError(
            "plan: no forcing source covers %s (and no --grib-dir given)"
            % window_anchor_start.strftime(_WRF_TIME_FMT))
    primary_source = chain[0]
    forcing_interval_h = primary_source.forcing_interval_h
    window_interval_seconds = forcing_interval_h * 3600

    t0_grid = align_to_interval(window_anchor_start, forcing_interval_h)
    shift_s_t0 = (t0_grid - window_anchor_start).total_seconds()
    if case_interval_seconds and case_interval_seconds != window_interval_seconds:
        print("interval_seconds: case=%ds window=%ds (forcing=%s, %dh cadence)"
              % (case_interval_seconds, window_interval_seconds, primary_source.dataset,
                 forcing_interval_h))
    if shift_s_t0:
        print("t0 shifted %+ds to the forcing's own grid (%s -> %s)"
              % (shift_s_t0, window_anchor_start.strftime(_WRF_TIME_FMT),
                 t0_grid.strftime(_WRF_TIME_FMT)))

    W, stepra, measure_steps = compute_window_seconds(spinup_min, dt, radt,
                                                      measure_steps=measure_steps)
    if io_probe and str(predict.g(nl, "physics", "nwp_diagnostics", 0)).strip() == "1":
        # nwp_diagnostics = 1 forbids history_interval(1) == 0, so the probe must use the plain
        # minutes keys, which needs a whole-minute window (elevconv 2026-09-28: a 27.5 min window
        # used history_interval_s and summed with the huge plain value -> no frame at all).
        while abs(W % 60.0) > 1e-6 and abs(W % 60.0 - 60.0) > 1e-6:
            measure_steps += 1
            W += dt

    nlw_path = namelist_wps or os.path.join(case_dir, "namelist.wps")
    wi_path = geo_from_wrfinput or os.path.join(case_dir, "wrfinput_d01")
    geo = resolve_projection(nl, nlw_path if os.path.exists(nlw_path) else None,
                              wi_path if os.path.exists(wi_path) else None, ncdump=ncdump)

    if times:
        # T1 (2026-09-24): explicit pinned window starts REPLACE characteristic-point selection
        # entirely -- --k is meaningless here (there is no "cap the picks" step) so it's ignored,
        # loudly, rather than silently. Every pinned time must already be a valid time of the
        # forcing grid (same grid candidate_times would have walked) and its window must fit in
        # the run, or this is a plan-time error naming the exact offending time -- a bad pin
        # should never silently produce a window ungrib/metgrid can't satisfy.
        if k != 4:
            print("--times given: --k=%d is ignored (using exactly the %d pinned time(s))"
                  % (k, len(times)))
        else:
            print("--times given: --k is ignored (using exactly the %d pinned time(s))" % len(times))
        picks = []
        for t in times:
            aligned = align_to_interval(t, forcing_interval_h)
            if aligned != t:
                raise ColdstartError(
                    "plan: --times %s is not on the forcing grid (%s, %dh cadence -- the next "
                    "grid time is %s)" % (t.strftime(_WRF_TIME_FMT), primary_source.dataset,
                                          forcing_interval_h, aligned.strftime(_WRF_TIME_FMT)))
            if t + datetime.timedelta(seconds=W) > window_anchor_end:
                raise ColdstartError(
                    "plan: --times %s + W(%.0fs) does not fit in the run (run ends %s)"
                    % (t.strftime(_WRF_TIME_FMT), W, window_anchor_end.strftime(_WRF_TIME_FMT)))
            picks.append((t, "pinned"))
    else:
        candidates = candidate_times(t0_grid, window_anchor_end, window_interval_seconds, W)
        if not candidates:
            raise ColdstartError(
                "no candidate window fits: run is %.0fs but one window needs W=%.0fs "
                "(spinup_min=%d, measure_steps=%d, dt=%.1f)" % (run_s, W, spinup_min, measure_steps, dt))

        sample_points = domain_lattice_from_geom(geo["ref_lat"], geo["ref_lon"], geo["dx"],
                                                  geo["e_we"], dy_m=geo["dy"], e_sn=geo["e_sn"])

        precip_lookup, state_skip_reason = (None, None)
        if use_state:
            precip_lookup, state_skip_reason = try_precip_lookup(case_dir, nl, candidates, W,
                                                                   grib_dir=grib_dir)
            if state_skip_reason:
                print(state_skip_reason)

        # night/day ranking uses the STATED timeline's own sun position (the sun the PREDICTED
        # run will actually see), even though the windows themselves run at the analog times.
        picks = select_windows(candidates, spinup_min * 60.0, W, k, sample_points=sample_points,
                                precip_lookup=precip_lookup, sunlit_shift_s=date_shift_s)

    windows = []
    grib_times = set()
    for t, reason in picks:
        t_end = next_grib_time_ge(t0_grid, window_interval_seconds, t + datetime.timedelta(seconds=W))
        # shift_s: only t0 has a case-defined "intended" time to shift away from (the case's own
        # literal start_dt); every other pick was chosen directly ON the forcing grid to begin
        # with, so it has no unaligned counterpart to report a shift against.
        shift_s = shift_s_t0 if reason == "t0" else 0.0
        windows.append({"t": t.strftime(_WRF_TIME_FMT), "t_end": t_end.strftime(_WRF_TIME_FMT),
                        "reason": reason, "shift_s": shift_s})
        if shift_s:
            print("  %s (%s): shift_s=%+.0f" % (t.strftime(_WRF_TIME_FMT), reason, shift_s))
        grib_times.add(t)
        grib_times.add(t_end)
    grib_times = sorted(grib_times)

    wps_fields = build_namelist_wps_fields(
        nl, geo, min(grib_times), max(grib_times), window_interval_seconds,
        prefix=primary_source.prefix)

    config = {"ranks": ranks, "omp": omp, "nx": nx or 1, "ny": ny or ranks // max(omp, 1),
              "numtiles": int(numtiles or 1)}

    # T12 (2026-09-24, job 965): read ZNW/P_TOP HERE (login node, netCDF4/numpy available) --
    # fix-real-namelist runs inside the batch job under /usr/bin/python3 (stdlib only) and can no
    # longer do this itself. A failed read is recorded loudly (vertical_grid: null + a warning IN
    # plan.json), not silently skipped at batch time. matthew-style dzbot/dzstretch FATALs hit
    # coldstart's own real.exe windows the same way case_rebuild's rebuild does.
    vertical_grid, vertical_grid_warning = read_vertical_grid_or_warn(
        wi_path if os.path.exists(wi_path) else None)
    if vertical_grid_warning:
        print("WARNING: %s" % vertical_grid_warning)

    plan = {
        "case_dir": os.path.abspath(case_dir),
        "namelist_input": os.path.abspath(namelist or os.path.join(case_dir, "namelist.input")),
        "max_dom": max_dom, "dt_s": dt, "radt_min": radt, "stepra": stepra,
        "measure_steps": measure_steps, "spinup_min": spinup_min, "W_seconds": W,
        "case_interval_seconds": case_interval_seconds,
        "window_interval_seconds": window_interval_seconds,
        "forcing_pin": forcing,
        "forcing_chain_primary": primary_source.to_dict(),
        "run_start": start_dt.strftime(_WRF_TIME_FMT), "run_end": end_dt.strftime(_WRF_TIME_FMT),
        "stated_start": stated_start.strftime(_WRF_TIME_FMT),
        "analog_start": analog_start.strftime(_WRF_TIME_FMT) if analog_start else None,
        "shift_s": date_shift_s, "analog_quality": analog_quality,
        "windows": windows, "grib_times": [t.strftime(_WRF_TIME_FMT) for t in grib_times],
        "geo": {k2: v2 for k2, v2 in geo.items()},
        "namelist_wps_fields": wps_fields,
        "nodes": nodes, "config": config, "real_only": bool(real_only),
        "vertical_grid": vertical_grid, "vertical_grid_warning": vertical_grid_warning,
        # K5 (2026-09-24): ML state completion, default OFF -- None unless --ml-model was given.
        "ml_model": os.path.abspath(ml_model) if ml_model else None,
    }
    if io_probe:
        probe = plan_io_probe(nl, case_dir, run_s)
        probe["window"] = len(windows) - 1
        plan["io_probe"] = probe
        for w in probe["warnings"]:
            print("WARNING: io probe: %s" % w)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    with open(os.path.join(out_dir, "namelist.wps"), "w") as f:
        f.write(render_namelist_wps(wps_fields))
    return plan


def print_plan(plan):
    print("W=%.0fs (spinup_min=%d measure_steps=%d stepra=%d dt=%.2fs)"
          % (plan["W_seconds"], plan["spinup_min"], plan["measure_steps"], plan["stepra"],
             plan["dt_s"]))
    for w in plan["windows"]:
        print("  %s .. %s  (%s)" % (w["t"], w["t_end"], w["reason"]))
    print("GRIB times needed: %d" % len(plan["grib_times"]))


# ---------------------------------------------------------------------------
# fetch -- probe -> choose -> download, per window, walking each window's own forcing chain
# (do NOT use metdata.resolve_wps_times, which reads interval_seconds only from
# case_dir/namelist.wps; SPEC.md is explicit about this -- windows already carry their own
# explicit [t, t_end]). Writes PLAN/forcing.json recording the choice (SPEC item C).
# ---------------------------------------------------------------------------

def _extract_cfsr_members(tar_path, valid_times, out_dir):
    """Extract the tar members needed for `valid_times` from a CFSR pgbh06/flxf06 tar.
    CFSR's outer tar URL/span pattern was VERIFIED live (see metdata.cfsr_tar_urls); the
    INTERNAL member naming was not independently confirmed beyond the standard NCEP
    pgbh06.gdas.<YYYYMMDDHH>.grb2-style convention this pattern-matches on -- best-effort, not
    a confirmed format."""
    import tarfile
    wanted = set(t.strftime("%Y%m%d%H") for t in valid_times)
    extracted = []
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if any(w in member.name for w in wanted):
                tf.extract(member, path=out_dir)
                extracted.append(os.path.join(out_dir, member.name))
    return extracted


def _download_forcing_for_window(source, T, T_end, cache_dir, keep_archives=False,
                                 download_fn=None, attempts=3, sleep_fn=None):
    """Downloads whatever `source` needs to cover [T, T_end] into cache_dir. Whole-file per
    valid time for every source except CFSR (d093000), which downloads the covering 5-day
    tar(s) once and extracts only the needed members (tarfile, stdlib), deleting the tar
    afterwards unless keep_archives. Returns the SET of absolute file paths this window ended up
    needing present on disk (not a byte count -- 2026-09-24: an RDA whole-file source answers
    EVERY one of its ~150 WPS_FIELD_SPECS-style needs with the SAME physical path, and two
    windows can independently need the same valid time's same file; summing bytes per-call, or
    even per-window, double- (or 150x-) counts it. The caller sums sizes once over the UNION of
    every window's returned set)."""
    md_mod = _metdata()
    download_fn = download_fn or md_mod.download_full
    valid_times = window_valid_times(T, T_end, source.forcing_interval_h * 3600)
    paths = set()
    if source.dataset == "d093000":
        spans = sorted(set(md_mod.cfsr_tar_span(t) for t in valid_times))
        for start, _end in spans:
            span_dt = datetime.datetime(start.year, start.month, start.day)
            pgb_url, flx_url = md_mod.cfsr_tar_urls(span_dt)
            for url in (pgb_url, flx_url):
                tar_path = os.path.join(cache_dir, os.path.basename(url))
                if not os.path.exists(tar_path):
                    download_fn(url, tar_path)
                if not os.path.exists(tar_path):
                    continue
                for member_path in _extract_cfsr_members(tar_path, valid_times, cache_dir):
                    if os.path.exists(member_path):
                        paths.add(os.path.abspath(member_path))
                if keep_archives:
                    paths.add(os.path.abspath(tar_path))
                else:
                    os.remove(tar_path)
        return paths
    # yagi2024 rebuild (scc job 1002): one transient RDA failure on fnl_2024090312 was swallowed
    # here, fetch still "succeeded" with 20 of 21 files, and ungrib died later on the missing
    # span start. Retry each URL with backoff, and fail fetch itself when any file is absent.
    import time
    sleep_fn = sleep_fn or time.sleep
    missing = []
    for vt in valid_times:
        for filename, urls in source.urls_for(vt):
            dest = os.path.join(cache_dir, filename)
            for attempt in range(attempts):
                if os.path.exists(dest):
                    break
                if attempt:
                    sleep_fn(5 * 2 ** (attempt - 1))
                for url in urls:
                    try:
                        download_fn(url, dest)
                    except Exception:
                        continue
                    if os.path.exists(dest):
                        break
            if os.path.exists(dest):
                paths.add(os.path.abspath(dest))
            else:
                missing.append(filename)
    if missing:
        raise ColdstartError("fetch: %s: %d file(s) still missing after %d attempts: %s"
                             % (source.dataset, len(missing), attempts, ", ".join(missing)))
    return paths


def resolve_and_fetch_forcing(plan, plan_dir, cache_dir, grib_dir=None, keep_archives=False,
                              probe_fn=None, download_fn=None):
    """Per window: walk resolve_forcing_chain_for(T) (or the plan's --forcing pin), pick the
    first candidate whose probe finds every needed file (choose_forcing_for_window -- SPEC item
    B), download it, and record the choice. Raises ColdstartError naming the actually-available
    dataset when a window's chosen source's cadence doesn't match the plan's own cadence (SPEC
    item D: 'fetch must re-align: fail clearly ... re-run plan with --forcing <dataset>') --
    silently proceeding would stage a namelist.wps/real.exe interval that doesn't match what was
    actually fetched. Writes PLAN_DIR/forcing.json. Returns (total_bytes, forcing_dict)."""
    os.makedirs(cache_dir, exist_ok=True)
    forcing_pin = plan.get("forcing_pin")
    if forcing_pin == ERA5_DATASET:
        return _fetch_era5_intermediate(plan, plan_dir)
    window_interval_s = plan.get("window_interval_seconds")
    windows_out = []
    all_paths = set()   # dedup across windows -- see _download_forcing_for_window's docstring
    for k, w in enumerate(plan["windows"]):
        T = datetime.datetime.strptime(w["t"], _WRF_TIME_FMT)
        T_end = datetime.datetime.strptime(w["t_end"], _WRF_TIME_FMT)
        chain = resolve_forcing_chain_for(T, grib_dir=grib_dir, forcing_pin=forcing_pin)
        chosen, tried = choose_forcing_for_window(T, T_end, chain, probe_fn=probe_fn)
        if chosen is None:
            # T4b (2026-09-24): always write the probe record, even on failure -- otherwise a
            # failed `fetch` leaves no trace of which URLs were actually probed.
            windows_out.append({"win": k, "chosen": None, "candidates_tried": tried})
            forcing = {"status": "failed", "windows": windows_out}
            with open(os.path.join(plan_dir, "forcing.json"), "w") as f:
                json.dump(forcing, f, indent=2)
            first_missing = None
            for t in tried:
                if t.get("first_missing"):
                    first_missing = t["first_missing"]
                    break
            # status None means the probe itself raised (e.g. job 1231's driver ran in an env
            # without `requests`); show the exception instead of implying the file is absent.
            detail = ("%s (final status %s%s)" % (
                first_missing["url"], first_missing["final_status"],
                ("; probe error: %s" % first_missing["detail"])
                if first_missing.get("final_status") is None and first_missing.get("detail") else "")
                      if first_missing and first_missing.get("url") else "none")
            raise ColdstartError(
                "fetch: window %d [%s..%s]: no forcing source in the chain has every needed "
                "file (tried: %s; first missing: %s)"
                % (k, w["t"], w["t_end"], ", ".join(t["dataset"] for t in tried) or "none", detail))
        if (window_interval_s is not None
                and chosen.forcing_interval_h * 3600 != window_interval_s):
            primary_dataset = plan.get("forcing_chain_primary", {}).get("dataset", "?")
            raise ColdstartError(
                "fetch: window %d fell back to %s (%dh cadence), which does not match the "
                "plan's own cadence (%ds, planned for %s) -- re-run plan with --forcing %s"
                % (k, chosen.dataset, chosen.forcing_interval_h, window_interval_s,
                   primary_dataset, chosen.dataset))
        win_paths = _download_forcing_for_window(chosen, T, T_end, cache_dir,
                                                 keep_archives=keep_archives,
                                                 download_fn=download_fn)
        all_paths |= win_paths
        windows_out.append({"win": k, "chosen": chosen.to_dict(), "candidates_tried": tried})
        print("window %d: %s (%s, %dh cadence)" % (k, chosen.dataset, chosen.source_kind,
                                                    chosen.forcing_interval_h))

    # Sum each distinct path's size exactly ONCE, over the union of every window's needs --
    # never per-window, never per-message (see _download_forcing_for_window's docstring: an RDA
    # whole-file source answering multiple message-shaped needs, or two windows sharing a valid
    # time, must not multiply the reported total).
    total_bytes = sum(os.path.getsize(p) for p in all_paths if os.path.exists(p))
    forcing = {"windows": windows_out, "files": sorted(os.path.basename(p) for p in all_paths)}
    with open(os.path.join(plan_dir, "forcing.json"), "w") as f:
        json.dump(forcing, f, indent=2)
    # T9 (2026-09-24): run.sbatch's link_grib.csh step must link exactly what fetch wrote -- FNL
    # (d083002) is cached as .grib and ncei_g3/g4 as .grb, but the step used to glob only
    # *.grib2 and FATAL'd 'no *.grib2 files' even though the grib/ dir was fully populated. Write
    # the exact basenames fetch wrote as a plain-text manifest (one per line) so the script can
    # link that list instead of assuming an extension.
    with open(os.path.join(plan_dir, "grib_manifest.txt"), "w") as f:
        for name in forcing["files"]:
            f.write(name + "\n")
    print("fetched %d bytes total across %d windows (%d distinct files)"
          % (total_bytes, len(windows_out), len(all_paths)))
    return total_bytes, forcing


# ---------------------------------------------------------------------------
# Namelist editing -- packed-line-safe set_list/count_key, reused verbatim (mechanism, not
# hardcodes) from .route1-ref/gate2_stage_window.py and gate2_stage_real.py.
# ---------------------------------------------------------------------------

def _set_list_rhs(text, key, rhs, new_key_group="time_control"):
    """Shared core of set_list/set_list_values: replace `key`'s existing value list with the
    literal `rhs` text (already comma-joined, trailing comma included), packed-line-safe
    (gate2_stage_window.py's approach). When the key doesn't exist at all, appends it to the
    FIRST `&new_key_group` group found (default &time_control, matching the original
    set_list's own fallback) -- case_rebuild.py's complete_namelist passes a different group for
    keys that don't belong in &time_control (e.g. physics_suite -> &physics); if that group
    doesn't exist at all either, a new one is appended holding just this key."""
    keyre = r"(?<![A-Za-z0-9_])(%s\s*=\s*)" % re.escape(key)
    pat = re.compile(keyre + r"((?:[^,=\n/!]*,)+)(?=\s*(?:[A-Za-z_][A-Za-z0-9_%]*\s*=|$|/|!))",
                      re.I | re.M)
    new, n = pat.subn(lambda m: m.group(1) + rhs, text, count=1)
    if n:
        return new
    # a value list whose LAST entry has no trailing comma ("2024, 2024,2024" -- yagi2024's own
    # namelist); the first pattern needs a comma after every entry, and falling through to the
    # insert below used to add a SECOND start_year line (fix-span-wrf-namelist: 'keys not present
    # exactly once')
    pat2 = re.compile(keyre + r"([^=\n/!]*?)\s*(?=[A-Za-z_][A-Za-z0-9_%]*\s*=|$|/|!)", re.I | re.M)
    new, n = pat2.subn(lambda m: m.group(1) + rhs, text, count=1)
    if n:
        return new
    if count_key(text, key):
        raise ColdstartError("set_list: %s exists but its value list could not be parsed" % key)
    group_re = re.compile(r"(&%s[^\n]*\n)" % re.escape(new_key_group), re.I)
    new, n = group_re.subn(r"\g<1> %s = %s\n" % (key, rhs), text, count=1)
    if n:
        return new
    return text.rstrip("\n") + "\n\n&%s\n %s = %s\n/\n" % (new_key_group, key, rhs)


def set_list(text, key, value, ndom, new_key_group="time_control"):
    # T11 (2026-09-24): new_key_group now threaded through (was silently dropped -- every caller
    # of set_list that needed a NEWLY-inserted key to land outside &time_control had no way to
    # say so, e.g. case_rebuild.complete_namelist's p_top_requested fill landed in &time_control
    # and made real.exe FATAL 'Maybe here?: p_top_requested =').
    return _set_list_rhs(text, key, ", ".join([str(value)] * ndom) + ",",
                         new_key_group=new_key_group)


def set_list_values(text, key, values, new_key_group="time_control"):
    """Like set_list, but for explicit PER-DOMAIN values that differ (e.g.
    parent_time_step_ratio = parent_grid_ratio per nest) rather than one value broadcast to
    every domain."""
    return _set_list_rhs(text, key, ", ".join(str(v) for v in values) + ",",
                         new_key_group=new_key_group)


def count_key(text, key):
    body = "\n".join(l.split("!")[0] for l in text.splitlines())
    return len(re.findall(r"(?<![A-Za-z0-9_])%s\s*=" % re.escape(key), body, re.I))


def get_first(text, key, default=None):
    m = re.search(r"(?<![A-Za-z0-9_])%s\s*=\s*([^,\n/]+)" % re.escape(key), text, re.I)
    return m.group(1).strip() if m else default


def remove_namelist_key(text, key):
    """Deletes every line whose (comment-stripped) body assigns `key` -- used to strip a
    namelist key that CONFLICTS with a value being written explicitly (T8, 2026-09-24:
    dzbot/dzstretch_s/dzstretch_u/auto_levels_opt all conflict with an explicit eta_levels
    list). Line-based, matching count_key's own comment-stripping convention -- these are scalar
    keys that are one-per-line in every real WRF namelist this tool has been run against; a key
    packed onto the same line as a neighbor is not handled (disclosed limitation, same as the
    rest of this module's line-oriented namelist editing). Returns (new_text, n_removed). Shared
    by BOTH coldstart_sample (T12) and case_rebuild.complete_namelist (T8) -- moved here
    (2026-09-24) since case_rebuild imports coldstart_sample, not the reverse."""
    keyre = re.compile(r"(?<![A-Za-z0-9_])%s\s*=" % re.escape(key), re.I)
    kept = []
    n = 0
    for ln in text.split("\n"):
        if keyre.search(ln.split("!")[0]):
            n += 1
            continue
        kept.append(ln)
    return "\n".join(kept), n


def namelist_values_equal(old, new):
    """True if two namelist scalar strings represent the same value -- numeric comparison when
    both sides parse as a number ('01' vs '1', '5000' vs '5000.0'), exact string equality
    otherwise (needed for .true./.false./quoted strings, where '1'=='1.0' would be wrong). T11
    (2026-09-24): complete_namelist's sync() used to compare raw strings, so a spelling-only
    difference was logged as a real change."""
    if old == new:
        return True
    try:
        return float(old) == float(new)
    except (TypeError, ValueError):
        return False


def vertical_grid_from_wrfinput(path):
    """{'eta_levels': [floats, top-down: 1.0 (surface) -> 0.0 (model top)], 'p_top_requested':
    float (Pa)} read directly from the case's own wrfinput_d01's ZNW/P_TOP. T8 (2026-09-24,
    matthew): matthew's OWN namelist (dzbot=30, dzstretch_s=1.11, dzstretch_u=1.10, e_vert=45,
    p_top_requested=5000) makes real.exe FATAL ('You need one of four things: more eta levels /
    lower p_top / dzbot / dzstretch') even though the ORIGINAL wrfinput has exactly 45 ZNW
    levels -- the namelist's own dzbot/dzstretch recipe does not reproduce the vertical grid the
    file was actually built with. Writing eta_levels/p_top_requested straight from ZNW/P_TOP
    reproduces it exactly instead of re-deriving it. ZNW already stores levels in the SAME
    top-down order WRF's own eta_levels namelist list expects (1.0 -> 0.0) -- no reversal.
    Rounded to 6 decimals (ZNW is float32 in the file; full float64 repr of a float32 value is
    binary noise, not a meaningful digit).

    T12 (2026-09-24, job 965): this needs numpy/netCDF4, so it may ONLY be called from a
    login-node/PLAN-time code path (build_plan / case_rebuild.cmd_rebuild_plan) -- the batch job
    runs fix-*-namelist under /usr/bin/python3 (stdlib only, no numpy), where calling this
    directly silently no-ops the whole vertical-grid-fidelity feature (that WAS the job 965 bug:
    'WARNING: vertical grid fidelity (T8) skipped -- No module named numpy'). PLAN time reads
    this ONCE and stores the plain-JSON result in plan.json['vertical_grid']; batch time reads
    THAT (see apply_vertical_grid_override / case_rebuild.complete_namelist's vertical_grid
    param) and never imports numpy/netCDF4 at all."""
    import numpy as np
    import netCDF4
    with _NETCDF_LOCK:   # build_plan runs per point in run_rebuild_points' thread pool
        return _vertical_grid_from_dataset(netCDF4, np, path)


def _vertical_grid_from_dataset(netCDF4, np, path):
    ds = netCDF4.Dataset(path)
    try:
        if "ZNW" not in ds.variables:
            raise ColdstartError("vertical_grid_from_wrfinput: %s has no ZNW" % path)
        if "P_TOP" not in ds.variables:
            raise ColdstartError("vertical_grid_from_wrfinput: %s has no P_TOP" % path)
        znw = np.asarray(ds.variables["ZNW"][0], dtype=float)
        eta_levels = [round(float(v), 6) for v in znw]
        p_top = float(np.asarray(ds.variables["P_TOP"][:]).reshape(-1)[0])
        return {"eta_levels": eta_levels, "p_top_requested": p_top}
    finally:
        ds.close()


def read_vertical_grid_or_warn(wi_path):
    """(vertical_grid_dict_or_None, warning_string_or_None) -- the PLAN-time (login node) wrapper
    around vertical_grid_from_wrfinput that both build_plan and case_rebuild.cmd_rebuild_plan
    call. wi_path falsy/nonexistent -> (None, None): no wrfinput at all is not a failure, just
    nothing to reproduce. wi_path exists but the read fails (no netCDF4, corrupt file, no
    ZNW/P_TOP) -> (None, '<message>') -- T12 (2026-09-24): the caller MUST store this warning
    loudly in plan.json (not just print it), since a plan-time failure that only shows up as a
    buried WARNING in a batch job's run log (job 965) is exactly the failure mode this exists to
    prevent. The returned dict uses plan.json's own field names (p_top_pa, not
    p_top_requested) -- see apply_vertical_grid_override's docstring for the rename."""
    if not wi_path or not os.path.exists(wi_path):
        return None, None
    try:
        vg = vertical_grid_from_wrfinput(wi_path)
        return {"eta_levels": vg["eta_levels"], "p_top_pa": vg["p_top_requested"]}, None
    except Exception as exc:
        return None, ("vertical grid fidelity (T8/T12): could not read ZNW/P_TOP from %s -- %s"
                      % (wi_path, exc))


def apply_vertical_grid_override(txt, vertical_grid, max_dom):
    """Applies vertical-grid-fidelity (T8/T12, 2026-09-24) to a coldstart real-namelist text,
    using a vertical_grid dict already read at PLAN time (plan.json's 'vertical_grid' --
    {'eta_levels': [...], 'p_top_pa': ...}, see read_vertical_grid_or_warn) -- stdlib only, no
    numpy/netCDF4 import at batch time. eta_levels/p_top_requested/e_vert are set EXACTLY from
    the case's own wrfinput ZNW/P_TOP (ALWAYS overridden, not fill-if-missing); dzbot/
    dzstretch_s/dzstretch_u/auto_levels_opt are removed (they conflict with an explicit
    eta_levels list). Applied AFTER build_real_namelist_text/assert_no_key_drift, as a separate
    step -- build_real_namelist_text's 'only touch present keys, a key present in the input must
    still be present in the output' invariant is not asked to also tolerate the dz-key removal
    this specifically requires. All three keys land in &domains (verified against
    wtune_catalog_registry.json in T11 -- e_vert/eta_levels/p_top_requested all resolve to
    'domains'; hardcoded here rather than via case_rebuild's registry_group_for_key/
    wtune_catalog_registry.json lookup, since coldstart_sample must not depend on
    case_rebuild.py, and this key set's group never varies). Returns (new_text, changes) -- the
    same {key,old,new,reason} shape case_rebuild.complete_namelist uses. vertical_grid=None (no
    wrfinput, or the PLAN-time read failed) -> (txt, []) unchanged, same as before T12."""
    if vertical_grid is None:
        return txt, []
    changes = []
    for key in ("dzbot", "dzstretch_s", "dzstretch_u", "auto_levels_opt"):
        txt, n_removed = remove_namelist_key(txt, key)
        if n_removed:
            changes.append({"key": key, "old": "present", "new": None,
                            "reason": "conflicts with explicit eta_levels from wrfinput ZNW"})
    eta_levels = list(vertical_grid["eta_levels"])
    old_e_vert = get_first(txt, "e_vert")
    new_e_vert = str(len(eta_levels))
    if old_e_vert is None or not namelist_values_equal(old_e_vert.strip(), new_e_vert):
        txt = set_list(txt, "e_vert", len(eta_levels), max_dom, new_key_group="domains")
        changes.append({"key": "e_vert", "old": old_e_vert, "new": new_e_vert,
                        "reason": "= len(wrfinput ZNW)"})
    txt = set_list_values(txt, "eta_levels", eta_levels, new_key_group="domains")
    changes.append({"key": "eta_levels", "old": None, "new": str(eta_levels),
                    "reason": "= wrfinput ZNW (exact vertical grid reproduction)"})
    old_ptop = get_first(txt, "p_top_requested")
    new_ptop = str(vertical_grid["p_top_pa"])
    if old_ptop is None or not namelist_values_equal(old_ptop.strip(), new_ptop):
        txt = set_list(txt, "p_top_requested", vertical_grid["p_top_pa"], 1,
                       new_key_group="domains")
        changes.append({"key": "p_top_requested", "old": old_ptop, "new": new_ptop,
                        "reason": "= wrfinput P_TOP"})
    return txt, changes


def assert_no_key_drift(text0, txt, out_label, required_present_once):
    bad = {k: count_key(txt, k) for k in required_present_once if count_key(txt, k) != 1}
    if bad:
        raise ColdstartError("%s: keys not present exactly once: %s" % (out_label, bad))
    lost = [k for k in re.findall(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*=", text0)
            if count_key(text0, k) and not count_key(txt, k)]
    if lost:
        raise ColdstartError("%s: edit dropped keys %s" % (out_label, sorted(set(lost))))


def build_real_namelist_text(case_namelist_text, T, T_end, num_metgrid_levels,
                              num_metgrid_soil_levels, num_land_cat, interval_seconds=None,
                              nproc_x=-1, nproc_y=-1):
    """SPEC.md Inputs: 'real.exe namelist per window: the case's namelist.input with (all
    domains) start = T_k, end = T_k_end, run_* = 0 except run_seconds/run_hours matching;
    num_metgrid_levels/num_metgrid_soil_levels/num_land_cat from met_em.' `interval_seconds`,
    when given, OVERRIDES the case's own (2026-09-24: it must equal the window's ACTUAL forcing
    cadence -- forcing_interval_h*3600 -- not the case's, since a fallback source (FNL/CFSR, 6h)
    can be coarser than what the case's own namelist.input assumed; only touched when the key
    exists, same 'only touch present keys' rule as history_interval*). `nproc_x`/`nproc_y`
    default to -1 (real.exe auto-decomposes) but 2026-09-24 cluster findings (sc26p1/severe2023:
    real.exe HANGS at some decompositions -- 3x3, 5x5 -- with no odd/square pattern identified,
    mechanism unknown) mean run.sbatch now always passes the PREDICTED config's own nx/ny (the
    decomposition wrf.exe will use anyway) -- or 1/1 (see cmd_fix_real_namelist's --serial) on
    the decomposed attempt's timeout/failure fallback."""
    text0 = case_namelist_text
    ndom = int(get_first(text0, "max_dom", "1"))
    txt = text0
    for k, v in (("year", T.year), ("month", "%02d" % T.month), ("day", "%02d" % T.day),
                 ("hour", "%02d" % T.hour), ("minute", "%02d" % T.minute), ("second", "%02d" % T.second)):
        # minute/second are written even when the case omits them: WRF defaults a missing
        # end_minute to 0, so maria1km (scc job 1006) nest d02 stopped at 10:00:00 while d01
        # ran on to 10:24:10 via run_seconds, and the post-spin-up window timed d01 alone
        if count_key(text0, "start_" + k) or k in ("minute", "second"):
            txt = set_list(txt, "start_" + k, v, ndom)
    for k, v in (("year", T_end.year), ("month", "%02d" % T_end.month), ("day", "%02d" % T_end.day),
                 ("hour", "%02d" % T_end.hour), ("minute", "%02d" % T_end.minute),
                 ("second", "%02d" % T_end.second)):
        if count_key(text0, "end_" + k) or k in ("minute", "second"):
            txt = set_list(txt, "end_" + k, v, ndom)
    win_seconds = int(round((T_end - T).total_seconds()))
    for k, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0), ("run_seconds", win_seconds)):
        txt = set_list(txt, k, v, 1)
    if count_key(text0, "num_metgrid_levels"):
        txt = set_list(txt, "num_metgrid_levels", int(num_metgrid_levels), 1)
    if count_key(text0, "num_metgrid_soil_levels") and num_metgrid_soil_levels is not None:
        txt = set_list(txt, "num_metgrid_soil_levels", int(num_metgrid_soil_levels), 1)
    if count_key(text0, "num_land_cat") and num_land_cat is not None:
        txt = set_list(txt, "num_land_cat", int(num_land_cat), 1)
    if interval_seconds is not None and count_key(text0, "interval_seconds"):
        txt = set_list(txt, "interval_seconds", int(interval_seconds), 1)
    if count_key(text0, "nproc_x"):
        txt = set_list(txt, "nproc_x", nproc_x, 1)
    if count_key(text0, "nproc_y"):
        txt = set_list(txt, "nproc_y", nproc_y, 1)

    required = (["start_" + k for k in ("year", "month", "day", "hour") if count_key(text0, "start_" + k)]
                + ["run_days", "run_hours", "run_minutes", "run_seconds"]
                + (["interval_seconds"] if interval_seconds is not None
                   and count_key(text0, "interval_seconds") else []))
    assert_no_key_drift(text0, txt, "real namelist T=%s" % T.strftime(_WRF_TIME_FMT), required)
    return txt


_HISTORY_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])(history_interval\w*)\s*=", re.I | re.M)


HUGE_HISTORY_MIN = 999999


def comment_out_of_group_assignments(text):
    """Comment out `key = value` lines that sit OUTSIDE every &group ... / block. Fortran's
    namelist read ignores them, but they break key-count checks (elevconv: a second
    iofields_filename between two groups -> 'keys not present exactly once', 2026-09-28).
    Semantics-preserving; lines inside groups are untouched."""
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        code = line.split("!")[0].strip()
        if code.startswith("&"):
            inside = True
            out.append(line)
            continue
        if inside and code.startswith("/"):
            inside = False
            out.append(line)
            continue
        if not inside and "=" in code:
            out.append("! (outside any namelist group; ignored by Fortran) " + line)
            continue
        out.append(line)
    return "".join(out)


def build_wrf_window_namelist_text(real_namelist_text, T, win_steps, dt_seconds,
                                    nproc_x, nproc_y, numtiles, io_probe=None):
    """SPEC.md Inputs: 'wrf.exe namelist per window: the real namelist plus: restart=.false.;
    history and restart output disabled (restart_interval and every history_interval* key that
    EXISTS set to a huge value -- only touch keys already present); run length = window steps x
    time_step exactly; nproc_x/nproc_y/numtiles from the config being predicted. Use key-count
    assertions exactly like gate2_stage_window.py.' `io_probe` (plan.json's 'io_probe' block,
    only for the probe window; None = the above, unchanged) re-enables ONE history frame per
    history-writing domain and, when the case writes restarts, one restart, both at the window
    end -- see apply_io_probe."""
    text0 = comment_out_of_group_assignments(real_namelist_text)
    ndom = int(get_first(text0, "max_dom", "1"))
    win_seconds = int(round(win_steps * dt_seconds))
    end = T + datetime.timedelta(seconds=win_seconds)

    txt = text0
    for k, v in (("year", T.year), ("month", "%02d" % T.month), ("day", "%02d" % T.day),
                 ("hour", "%02d" % T.hour), ("minute", "%02d" % T.minute), ("second", "%02d" % T.second)):
        if count_key(text0, "start_" + k) or k in ("minute", "second"):
            txt = set_list(txt, "start_" + k, v, ndom)
    for k, v in (("year", end.year), ("month", "%02d" % end.month), ("day", "%02d" % end.day),
                 ("hour", "%02d" % end.hour), ("minute", "%02d" % end.minute),
                 ("second", "%02d" % end.second)):
        if count_key(text0, "end_" + k) or k in ("minute", "second"):
            txt = set_list(txt, "end_" + k, v, ndom)
    for k, v in (("run_days", 0), ("run_hours", 0), ("run_minutes", 0), ("run_seconds", win_seconds)):
        txt = set_list(txt, k, v, 1)
    if count_key(text0, "restart"):
        txt = set_list(txt, "restart", ".false.", 1)
    if count_key(text0, "restart_interval"):
        txt = set_list(txt, "restart_interval", 999999, 1)
    history_keys = sorted(set(m.group(1).lower() for m in _HISTORY_KEY_RE.finditer(text0)))
    # 0 = history off. A huge interval still writes the t0 frame (job 1238: 68 s for 2.3 GB).
    # nwp_diagnostics = 1 makes check_a_mundo FATAL on history_interval(1) == 0 ('nwp_diagnostics
    # requires the use of "history_interval"', elevconv 2026-09-28): there, keep a huge plain
    # history_interval and push history_begin past the window instead (no frame at all).
    nwp_diag = str(get_first(text0, "nwp_diagnostics", "0")).strip() == "1"
    for hk in history_keys:
        txt = set_list(txt, hk, HUGE_HISTORY_MIN if (nwp_diag and hk == "history_interval") else 0,
                       ndom)
    if nwp_diag:
        if "history_interval" not in history_keys:
            txt = set_list(txt, "history_interval", HUGE_HISTORY_MIN, ndom,
                           new_key_group="time_control")
            history_keys = sorted(history_keys + ["history_interval"])
        for bk in ("history_begin_d", "history_begin_h", "history_begin_m", "history_begin_s"):
            if count_key(text0, bk):
                txt = set_list(txt, bk, 0, ndom)
        txt = set_list(txt, "history_begin", HUGE_HISTORY_MIN, ndom,
                       new_key_group=None if count_key(text0, "history_begin") else "time_control")
    # The decomposition is what is being measured, so it is written even when the case's own
    # namelist omits it (conus25 has no nproc_x/nproc_y/numtiles: C3 job 1237's rebuild window
    # silently ran WRF's default decomposition, +21% per step vs the G0014 truth).
    for key, val, default in (("nproc_x", nproc_x, -1), ("nproc_y", nproc_y, -1),
                              ("numtiles", numtiles, 1)):
        if count_key(text0, key):
            txt = set_list(txt, key, val, 1)
        elif int(val) != default:
            txt = set_list(txt, key, val, 1, new_key_group="domains")

    required = (["start_" + k for k in ("year", "month", "day", "hour") if count_key(text0, "start_" + k)]
                + ["run_days", "run_hours", "run_minutes", "run_seconds"]
                + (["restart"] if count_key(text0, "restart") else [])
                + (["restart_interval"] if count_key(text0, "restart_interval") else [])
                + history_keys)
    if io_probe is not None:
        txt, probe_keys = apply_io_probe(text0, txt, ndom, win_seconds, io_probe)
        required += [k for k in probe_keys if k not in required]
    assert_no_key_drift(text0, txt, "wrf window namelist T=%s" % T.strftime(_WRF_TIME_FMT), required)
    return txt


# ---------------------------------------------------------------------------
# I/O probe (opt-in: build_plan(io_probe=True)). The LAST window writes exactly one history frame
# per history-writing domain (+ one restart when the case's own run writes restarts) at its END,
# with the case's own io_form_history/io_form_restart/history_outname/frames_per_outfile/
# iofields_filename; WRF times each write itself ('Timing for Writing <file> for domain N: <s>
# elapsed seconds'). WRF semantics (Registry/registry.io_boilerplate, inc/set_timekeeping_
# alarms.inc, share/set_timekeeping.F, frame/module_integrate.F): history_interval[_d|_h|_m|_s]
# and history_begin[_d|_h|_m|_s] are SUMMED per domain (the _m key falls back to the plain one
# when 0); with begin > 0 the history ringer is not switched on at t=0, so history_begin = W
# writes only the end frame. The end-of-run history and restart writes happen in
# med_last_solve_io, AFTER the last 'Timing for main' line, so no measured step absorbs them.
# restart_interval[_d|_h|_m|_s] is an interval-only alarm (no t=0 write).
# ---------------------------------------------------------------------------

IO_PROBE_TIMEOUT_EXTRA_MIN = 15   # stage_plan: allowance for the end-of-window writes (restart
                                  # writes of ~300 s were measured on scc, marshall)
_HISTORY_BEGIN_END_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])(history_(?:begin|end)\w*)\s*=",
                                       re.I | re.M)
_RESTART_INTERVAL_KEY_RE = re.compile(r"(?<![A-Za-z0-9_])(restart_interval\w*)\s*=", re.I | re.M)


def wrf_interval_minutes(nl, base, dom_index=0):
    """WRF's own sum for `base`[_d|_h|_m|_s] of parsed namelist `nl`, in minutes: d*1440 + h*60 +
    (m, or the plain `base` key when m is 0) + s/60. 0 = off/absent."""
    def v(key):
        return float(predict.g_dom(nl, "time_control", key, dom_index, 0) or 0)
    return (v(base + "_d") * 1440.0 + v(base + "_h") * 60.0 + (v(base + "_m") or v(base))
            + v(base + "_s") / 60.0)


def case_writes_restart(nl, run_s):
    """True when the case's own run writes at least one restart (restart_interval <= run length)."""
    ri = wrf_interval_minutes(nl, "restart_interval")
    return ri > 0 and math.floor(run_s / 60.0 / ri + 1e-9) >= 1


def history_file_prefix(nl):
    """Basename prefix of the case's history files (history_outname up to its first '<'; WRF's
    default 'wrfout_d<domain>_<date>' -> 'wrfout_d') -- what parse_rsl_write_seconds uses to tell
    history writes from auxhist ones."""
    name = predict.g(nl, "time_control", "history_outname", None) or "wrfout_d<domain>_<date>"
    return os.path.basename(str(name).strip()).split("<")[0]


def case_iofields_links(plan, plan_dir):
    """Symlink the case's existing iofields_filename files into every win_k/wrf dir of the plan;
    returns [(k, dst), ...]. Missing files are left alone (the window then hangs at init unless
    the watcher in render_run_sbatch stops it)."""
    case_dir = plan.get("case_dir")
    nl_path = plan.get("namelist_input") or (os.path.join(case_dir, "namelist.input")
                                             if case_dir else None)
    if not case_dir or not nl_path or not os.path.exists(nl_path):
        return []
    nl = predict.parse_namelist(nl_path)
    names = set()
    for d in range(predict.get_max_dom(nl)):
        fname = str(predict.g_dom(nl, "time_control", "iofields_filename", d) or "").strip()
        if fname and fname != "NONE_SPECIFIED" and not os.path.isabs(fname):
            names.add(fname)
    out = []
    for k in range(len(plan.get("windows") or [])):
        wrf_dir = os.path.join(plan_dir, "win_%d" % k, "wrf")
        for fname in sorted(names):
            src = os.path.join(case_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(wrf_dir, fname)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                _symlink_one(os.path.abspath(src), dst)
                out.append((k, dst))
    return out


def plan_io_probe(nl, case_dir, run_s):
    """plan.json's 'io_probe' block for parsed case namelist `nl`: per-domain history on/off
    (io_form_history != 0 and a history interval > 0), whether a restart is probed
    (case_writes_restart), per-domain iofields files (None where the case sets none or the file is
    missing -- the missing one is a WARNING and the probe window gets 'NONE_SPECIFIED', WRF's own
    no-file value, because WRF hangs on a missing iofields file), and 'auxhist not probed' when
    any io_form_auxhistN is set."""
    max_dom = predict.get_max_dom(nl)
    io_form = predict.g(nl, "time_control", "io_form_history", 2)
    history = [bool(io_form) and wrf_interval_minutes(nl, "history_interval", d) > 0
               for d in range(max_dom)]
    warnings, notes = [], []
    iofields = None
    if nl.get("time_control", {}).get("iofields_filename"):
        iofields = []
        for d in range(max_dom):
            fname = str(predict.g_dom(nl, "time_control", "iofields_filename", d) or "").strip()
            if not fname or fname == "NONE_SPECIFIED":
                iofields.append(None)
                continue
            path = fname if os.path.isabs(fname) else os.path.join(case_dir, fname)
            if os.path.isfile(path):
                iofields.append({"name": fname, "path": os.path.abspath(path)})
            else:
                iofields.append(None)
                warnings.append("iofields_filename d%02d=%s not found at %s: not used by the I/O "
                                "probe (WRF hangs on a missing iofields file) -- the probed frame "
                                "carries the default history field list" % (d + 1, fname, path))
    aux = [n for n in range(1, 25) if predict.g(nl, "time_control", "io_form_auxhist%d" % n, 0)]
    if aux:
        notes.append("auxhist not probed (io_form_auxhist%s set)" % ",".join(str(n) for n in aux))
    return {"history": history, "restart": case_writes_restart(nl, run_s),
            "io_form_history": io_form,
            "io_form_restart": predict.g(nl, "time_control", "io_form_restart", io_form),
            "history_prefix": history_file_prefix(nl), "iofields": iofields,
            "warnings": warnings, "notes": notes}


def apply_io_probe(text0, txt, ndom, win_seconds, probe):
    """The probe edits on top of build_wrf_window_namelist_text's output `txt` (history/restart
    already off): history_interval = history_begin = W per history-writing domain (0 elsewhere),
    in minutes, or the _s keys when W is not whole minutes; every other history_begin*/end* key
    the case sets is zeroed (WRF sums them); iofields_filename rewritten per plan_io_probe; when
    probe['restart'], restart_interval = W (other restart_interval_* keys zeroed). Returns
    (txt, keys that must be present exactly once)."""
    suffix, val = ("", win_seconds // 60) if win_seconds % 60 == 0 else ("_s", win_seconds)
    nwp_diag = str(get_first(text0, "nwp_diagnostics", "0")).strip() == "1"
    if nwp_diag and suffix:
        raise ColdstartError("io probe: nwp_diagnostics = 1 needs a whole-minute window (have %d s)"
                             % win_seconds)
    hist = list(probe.get("history") or [])
    hist = (hist + [hist[-1] if hist else False] * ndom)[:ndom]
    keys = []
    if any(hist):
        on = [val if h else 0 for h in hist]
        h_int, h_beg = "history_interval" + suffix, "history_begin" + suffix
        for k in sorted(set(m.group(1).lower() for m in _HISTORY_BEGIN_END_KEY_RE.finditer(text0))):
            if k != h_beg:
                txt = set_list(txt, k, 0, ndom)
        txt = set_list_values(txt, h_int, on)
        txt = set_list_values(txt, h_beg, on)
        keys += [h_int, h_beg]
        if suffix:
            # WRF sums the plain (minutes) key and its _s variant: zero the plain ones the base
            # window may have set (e.g. the huge nwp_diagnostics-safe values).
            for plain in ("history_interval", "history_begin"):
                if count_key(txt, plain):
                    txt = set_list(txt, plain, 0, ndom)
        iof = probe.get("iofields")
        if iof is not None and count_key(text0, "iofields_filename"):
            iof = (list(iof) + [iof[-1] if iof else None] * ndom)[:ndom]
            txt = set_list_values(txt, "iofields_filename",
                                  ["'%s'" % (f["name"] if f else "NONE_SPECIFIED") for f in iof])
            keys.append("iofields_filename")
    if probe.get("restart"):
        r_int = "restart_interval" + suffix
        for k in sorted(set(m.group(1).lower() for m in _RESTART_INTERVAL_KEY_RE.finditer(text0))):
            if k != r_int:
                txt = set_list(txt, k, 0, 1)
        txt = set_list(txt, r_int, val, 1)
        keys.append(r_int)
    return txt, keys


_RSL_WRITE_RE = re.compile(r"Timing for Writing (.+?) for domain\s+(\d+):\s+([\d.]+)\s+elapsed seconds")
_RSL_MAIN_RE = re.compile(r"Timing for main: time (\S+) on domain\s+(\d+):\s+([\d.]+)\s+elapsed seconds")


def parse_rsl_write_seconds(path, history_prefix="wrfout_d"):
    """{'history': {dom: [s, ...]}, 'restart': {dom: [s, ...]}, 'other': [(name, dom, s), ...]}
    from WRF's 'Timing for Writing <name> for domain N: <s> elapsed seconds' lines, in file order.
    A history write is one whose file basename starts with `history_prefix` (history_file_
    prefix); 'restart' is WRF's own literal name for restart writes; anything else (auxhist,
    filter output) goes to 'other'."""
    out = {"history": {}, "restart": {}, "other": []}
    with open(path, errors="replace") as f:
        for line in f:
            m = _RSL_WRITE_RE.search(line)
            if not m:
                continue
            name, dom, sec = m.group(1).strip(), int(m.group(2)), float(m.group(3))
            base = os.path.basename(name)
            if name == "restart":
                out["restart"].setdefault(dom, []).append(sec)
            elif base.startswith(history_prefix) and (history_prefix or not base.startswith("auxhist")):
                out["history"].setdefault(dom, []).append(sec)
            else:
                out["other"].append((name, dom, sec))
    return out


def rsl_write_absorbing_d01_times(path):
    """Set of d01 step sim-time strings whose 'Timing for main' line FOLLOWS a 'Timing for
    Writing' line (any stream, any domain) since the previous d01 step line: WRF's step timer
    starts before med_before_solve_io, so that step's elapsed seconds include the write."""
    out = set()
    pending = False
    with open(path, errors="replace") as f:
        for line in f:
            if _RSL_WRITE_RE.search(line):
                pending = True
                continue
            m = _RSL_MAIN_RE.search(line)
            if m and int(m.group(2)) == 1:
                if pending:
                    out.add(m.group(1))
                pending = False
    return out


# ---------------------------------------------------------------------------
# fix-wps-namelist -- invoked from run.sbatch, once per window, BEFORE ungrib/metgrid.
#
# plan.json's own PLAN_DIR/namelist.wps spans the WHOLE run's min/max grib_times (needed for the
# one-time geogrid.exe step, and as a template). But `fetch` only downloads the windows' own
# GRIB times, not every interval-aligned time across the full run -- so running ungrib/metgrid
# ONCE against that whole-run-spanning namelist.wps makes metgrid walk every interval in
# [run_start, run_end] and die on the first time nobody fetched (sc26p1: 72h run, 6h interval, 4
# windows fetched only 8 of the 13 times metgrid would have wanted). Fix: ungrib+metgrid run
# once PER WINDOW, each time with namelist.wps &share start_date/end_date narrowed to exactly
# that window's own [t, t_end] -- the only times `fetch` guaranteed exist on disk.
# ---------------------------------------------------------------------------

def _load_window_forcing(plan_dir, win):
    """PLAN/forcing.json's 'chosen' ForcingSource dict for window `win` (written by `fetch` --
    see resolve_and_fetch_forcing), or None when forcing.json doesn't exist / has no entry for
    this window (e.g. every offline test that never ran `fetch`, or a plan staged before this
    mechanism existed) -- callers fall back to plan.json's own plan-time defaults ('GFS'/
    Vtable.GFS / plan['window_interval_seconds']), preserving old behaviour exactly."""
    path = os.path.join(plan_dir, "forcing.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        forcing = json.load(f)
    for w in forcing.get("windows", []):
        if w.get("win") == win:
            return w.get("chosen")
    return None


def wps_fields_for_window(base_fields, T, T_end, prefix=None, interval_seconds=None):
    """base_fields (build_namelist_wps_fields's dict, or plan['namelist_wps_fields']) with
    start_date/end_date replaced by this window's own [T, T_end] (all max_dom entries --
    render_namelist_wps already broadcasts a single start_date/end_date string to every domain).
    `prefix`/`interval_seconds`, when given, override base_fields' own (the window's ACTUALLY
    chosen forcing source, from forcing.json, can differ from the plan-time expected one -- see
    cmd_fix_wps_namelist). Pure function; everything else (projection, per-domain geometry) is
    unchanged."""
    fields = dict(base_fields)
    fields["start_date"] = T.strftime(_WRF_TIME_FMT)
    fields["end_date"] = T_end.strftime(_WRF_TIME_FMT)
    if prefix is not None:
        fields["prefix"] = prefix
    if interval_seconds is not None:
        fields["interval_seconds"] = int(interval_seconds)
    return fields


def cmd_fix_wps_namelist(plan_dir, win):
    with open(os.path.join(plan_dir, "plan.json")) as f:
        plan = json.load(f)
    w = plan["windows"][win]
    T = datetime.datetime.strptime(w["t"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(w["t_end"], _WRF_TIME_FMT)
    win_forcing = _load_window_forcing(plan_dir, win)
    prefix = win_forcing["prefix"] if win_forcing else None
    interval_s = win_forcing["forcing_interval_h"] * 3600 if win_forcing else \
        plan.get("window_interval_seconds")
    fields = wps_fields_for_window(plan["namelist_wps_fields"], T, T_end, prefix=prefix,
                                    interval_seconds=interval_s)
    text = render_namelist_wps(fields)
    out_path = os.path.join(plan_dir, "wps", "namelist.wps")
    if os.path.lexists(out_path):
        os.remove(out_path)
    with open(out_path, "w") as f:
        f.write(text)
    print("wrote %s (window %d: %s .. %s)" % (out_path, win, fields["start_date"], fields["end_date"]))
    return fields


# ---------------------------------------------------------------------------
# fix-real-namelist / fix-wrf-namelist -- invoked from run.sbatch, after met_em / real.exe exist
# (SPEC.md Inputs: 'these are written by a small helper subcommand ... because the values only
# exist once met_em exists').
# ---------------------------------------------------------------------------

def _read_met_em_dims(met_em_path, ncdump_exe):
    """num_metgrid_levels (a DIMENSION), NUM_METGRID_SOIL_LEVELS and NUM_LAND_CAT (global
    attributes) from a met_em file -- SPEC.md Inputs / .route1-ref/job1_wps_real.sbatch's own
    comment on which of the two is a dimension vs. an attribute."""
    out = subprocess.run([ncdump_exe, "-h", met_em_path], check=True,
                          capture_output=True, text=True).stdout
    m = re.search(r"^\s*num_metgrid_levels\s*=\s*(\d+)", out, re.M)
    if not m:
        raise ColdstartError("could not read num_metgrid_levels dimension from %s" % met_em_path)
    nlev = int(m.group(1))
    soil = None
    m2 = re.search(r":NUM_METGRID_SOIL_LEVELS\s*=\s*(\d+)", out)
    if m2:
        soil = int(m2.group(1))
    land = None
    m3 = re.search(r":NUM_LAND_CAT\s*=\s*(\d+)", out)
    if m3:
        land = int(m3.group(1))
    return nlev, soil, land


def cmd_fix_real_namelist(plan_dir, win, ncdump=None, serial=False):
    with open(os.path.join(plan_dir, "plan.json")) as f:
        plan = json.load(f)
    real_dir = os.path.join(plan_dir, "win_%d" % win, "real")
    met_ems = sorted(p for p in os.listdir(real_dir) if p.startswith("met_em.d01."))
    if not met_ems:
        raise ColdstartError("fix-real-namelist: no met_em.d01.* in %s" % real_dir)
    ncdump_exe = find_ncdump(ncdump)
    if not ncdump_exe:
        raise ColdstartError("fix-real-namelist: no ncdump available")
    nlev, soil, land = _read_met_em_dims(os.path.join(real_dir, met_ems[0]), ncdump_exe)
    w = plan["windows"][win]
    T = datetime.datetime.strptime(w["t"], _WRF_TIME_FMT)
    T_end = datetime.datetime.strptime(w["t_end"], _WRF_TIME_FMT)
    case_text = open(plan["namelist_input"]).read()
    # num_land_cat: real.exe REQUIRES this to match the geog/met_em landuse dataset's own
    # category count, so the overwrite below is correct and stays -- but it silently changes
    # what the case's own namelist.input specified (see inspect's matching "filled differently"
    # warning), so print the old->new value rather than overwriting mutely.
    old_land_cat = get_first(case_text, "num_land_cat")
    if land is not None and old_land_cat is not None and str(land) != old_land_cat.strip():
        print("num_land_cat: %s -> %d (from met_em NUM_LAND_CAT)" % (old_land_cat.strip(), land))
    win_forcing = _load_window_forcing(plan_dir, win)
    interval_s = win_forcing["forcing_interval_h"] * 3600 if win_forcing else \
        plan.get("window_interval_seconds")
    old_interval = get_first(case_text, "interval_seconds")
    if interval_s is not None and old_interval is not None and str(int(interval_s)) != old_interval.strip():
        print("interval_seconds: %s -> %d (case interval vs. window forcing cadence)"
              % (old_interval.strip(), interval_s))
    # 2026-09-24 cluster findings: real.exe HANGS at some decompositions (3x3, 5x5 -- no
    # odd/square rule identified) but completes fine at np=1. `--serial` is the fallback
    # run.sbatch retries with after a `timeout 600` decomposed attempt fails/hangs.
    if serial:
        nproc_x, nproc_y = 1, 1
    else:
        config = plan.get("config", {})
        nproc_x = int(config.get("nx", -1) or -1)
        nproc_y = int(config.get("ny", -1) or -1)
    txt = build_real_namelist_text(case_text, T, T_end, nlev, soil, land, interval_seconds=interval_s,
                                    nproc_x=nproc_x, nproc_y=nproc_y)
    # T12 (2026-09-24, job 965): vertical_grid was already read at PLAN time (login node,
    # netCDF4) and stored verbatim in plan.json -- this subcommand runs inside the batch job
    # under /usr/bin/python3 (stdlib only) and must never import numpy/netCDF4 itself. Applied
    # AFTER build_real_namelist_text/assert_no_key_drift, as its own step (see
    # apply_vertical_grid_override's docstring for why).
    vertical_grid = plan.get("vertical_grid")
    if vertical_grid is None and plan.get("vertical_grid_warning"):
        print("WARNING: %s (recorded at plan time)" % plan["vertical_grid_warning"])
    txt, vg_changes = apply_vertical_grid_override(txt, vertical_grid, plan.get("max_dom", 1))
    out_path = os.path.join(real_dir, "namelist.input")
    if os.path.lexists(out_path):
        os.remove(out_path)
    with open(out_path, "w") as f:
        f.write(txt)
    if vg_changes:
        print("vertical grid fidelity: %s" % ", ".join(c["key"] for c in vg_changes))
    print("wrote %s (num_metgrid_levels=%d, nproc_x=%d nproc_y=%d%s)"
          % (out_path, nlev, nproc_x, nproc_y, " serial" if serial else ""))


def cmd_fix_wrf_namelist(plan_dir, win, nproc_x, nproc_y, numtiles, ncdump=None):
    with open(os.path.join(plan_dir, "plan.json")) as f:
        plan = json.load(f)
    real_dir = os.path.join(plan_dir, "win_%d" % win, "real")
    wrf_dir = os.path.join(plan_dir, "win_%d" % win, "wrf")
    real_nl_path = os.path.join(real_dir, "namelist.input")
    if not os.path.exists(real_nl_path):
        raise ColdstartError("fix-wrf-namelist: %s does not exist yet (run fix-real-namelist "
                              "first)" % real_nl_path)
    real_text = open(real_nl_path).read()
    w = plan["windows"][win]
    T = datetime.datetime.strptime(w["t"], _WRF_TIME_FMT)
    win_steps = plan["measure_steps"] + int(round(plan["spinup_min"] * 60.0 / plan["dt_s"]))
    probe = plan.get("io_probe")
    txt = build_wrf_window_namelist_text(real_text, T, win_steps, plan["dt_s"], nproc_x, nproc_y,
                                          numtiles,
                                          io_probe=probe if probe and probe.get("window") == win else None)
    out_path = os.path.join(wrf_dir, "namelist.input")
    if os.path.lexists(out_path):
        os.remove(out_path)
    with open(out_path, "w") as f:
        f.write(txt)
    print("wrote %s (win_steps=%d)" % (out_path, win_steps))


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------

def _symlink_tree(src_dir, dst_dir, skip=frozenset(("namelist.input", "namelist.output"))):
    """Symlink every file/link in src_dir into dst_dir, skipping `skip` (gate2_stage_window.py's
    rule: never let a case's own WRF_TREE/run/namelist.input get symlinked in first, or a later
    open(path,'w') writes THROUGH the symlink into the shared tables dir)."""
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.isdir(src_dir):
        return
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src = os.path.join(src_dir, name)
        if os.path.isfile(src) or os.path.islink(src):
            dst = os.path.join(dst_dir, name)
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.realpath(src), dst)


def _symlink_one(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def _symlink_dir(src_dir, dst):
    """ln -sfn semantics for a directory symlink: replaces an existing SYMLINK at dst, but
    refuses to touch a real (non-symlink) file/directory already there -- 'never write through
    it' (gate2_stage_window.py's rule, generalised to directories: geogrid.exe/metgrid.exe open
    ./geogrid/GEOGRID.TBL and ./metgrid/METGRID.TBL relative to cwd by default, and fail with
    'Could not open GEOGRID.TBL'/'... METGRID.TBL' -- MPI_Abort(errorcode=0), so mpirun's own rc
    is 0 -- when those directories aren't there; jobs 941/942, 2026-09-24)."""
    if os.path.islink(dst):
        os.remove(dst)
    elif os.path.exists(dst):
        raise ColdstartError("refusing to replace non-symlink %s with a directory symlink" % dst)
    os.symlink(os.path.realpath(src_dir), dst, target_is_directory=True)


# T13 (2026-09-24, V5 sc26p2, job real.exe FATAL 'ERROR in psfc: flag_psfc=1, flag_soilhgt=1,
# flag_slp=0'): the literal replacement line, taken verbatim from WPS 4.6 Vtable.GFS's own
# commented-out PRMSL alternative (its comment: older GFS files carry only PRMSL, not MSLET --
# use this line instead of the active MSLET one). GRIB1 param 2 / GRIB2 0-3-1 (discipline 0,
# category 3, parameter 1) vs the active line's MSLET (GRIB1 param 130 / GRIB2 0-3-192).
VTABLE_GFS_PRMSL_LINE = ("  2 | 102  |   0  |      | PMSL     | Pa      | "
                         "Sea-level Pressure                      |  0  |  3  |  1  | 101 |")


def generate_vtable_gfs_prmsl(vtable_gfs_text):
    """PLAN/wps/Vtable.GFS_PRMSL's content (T13, 2026-09-24), generated from a REAL Vtable.GFS
    (never hand-written): replaces the one ACTIVE (non-comment) PMSL line -- which maps from
    MSLET (GRIB1 param 130 / GRIB2 0-3-192) -- with VTABLE_GFS_PRMSL_LINE (GRIB1 param 2 / GRIB2
    0-3-1, PRMSL), leaving every other line untouched (same table, same formatting, same
    comments -- just this one line swapped). This is needed because an older-era GRIB1 analysis
    file (NCEI g3, verified on a real 2005 file) carries only PRMSL, not MSLET, and Vtable.GFS's
    active MSLET-only mapping made real.exe FATAL 'ERROR in psfc: flag_psfc=1, flag_soilhgt=1,
    flag_slp=0' (PMSL never got filled at all). Raises ColdstartError if the input text doesn't
    have exactly one active PMSL line (the transform doesn't know what to do with 0 or >1 --
    that means the source table isn't shaped the way this transform assumes)."""
    lines = vtable_gfs_text.split("\n")
    out = []
    replaced = 0
    pmsl_re = re.compile(r"\|\s*PMSL\s*\|")
    for ln in lines:
        stripped = ln.strip()
        is_comment_or_blank = (not stripped) or stripped.startswith("#") or stripped.startswith("!")
        if not is_comment_or_blank and pmsl_re.search(ln):
            out.append(VTABLE_GFS_PRMSL_LINE)
            replaced += 1
        else:
            out.append(ln)
    if replaced != 1:
        raise ColdstartError(
            "generate_vtable_gfs_prmsl: expected exactly 1 active PMSL line in the source "
            "Vtable.GFS, found %d -- the source table is not shaped as expected" % replaced)
    return "\n".join(out)


def stage_plan(plan_dir, site="scc", per_window_min=15):
    with open(os.path.join(plan_dir, "plan.json")) as f:
        plan = json.load(f)

    wps_dir_env = os.environ.get("WPS_DIR")
    wrf_tree = os.environ.get("WRF_TREE")
    wps_out = os.path.join(plan_dir, "wps")
    os.makedirs(wps_out, exist_ok=True)
    if wps_dir_env:
        for exe in ("geogrid.exe", "ungrib.exe", "metgrid.exe", "link_grib.csh"):
            src = os.path.join(wps_dir_env, exe)
            if os.path.exists(src):
                _symlink_one(src, os.path.join(wps_out, exe))
        # geogrid.exe/metgrid.exe each need their own TBL directory (see _symlink_dir docstring).
        for subdir in ("geogrid", "metgrid"):
            src_dir = os.path.join(wps_dir_env, subdir)
            if os.path.isdir(src_dir):
                _symlink_dir(src_dir, os.path.join(wps_out, subdir))
            else:
                print("WARNING: $WPS_DIR/%s not found -- geogrid/metgrid will fail to open "
                      "its own TBL file" % subdir)
        # Both Vtable.GFS (GFS-family) and Vtable.CFSR are staged by their real names, since a
        # window's own forcing choice (per forcing.json) can differ -- run.sbatch relinks the
        # generic 'Vtable' ungrib reads per window (see render_run_sbatch).
        for vt in ("Vtable.GFS", "Vtable.CFSR"):
            vt_src = os.path.join(wps_dir_env, "ungrib", "Variable_Tables", vt)
            if os.path.exists(vt_src):
                _symlink_one(vt_src, os.path.join(wps_out, vt))
        # T13 (2026-09-24): Vtable.GFS_PRMSL (NCEI g3/g4 + FNL -- see metdata.SRC_NCEI_G3/G4/
        # SRC_FNL) is GENERATED, not symlinked -- it does not exist in $WPS_DIR at all, it is a
        # one-line edit of the real Vtable.GFS (see generate_vtable_gfs_prmsl). Never write
        # through a stale symlink from a prior stage: remove whatever's there first.
        gfs_vt_src = os.path.join(wps_dir_env, "ungrib", "Variable_Tables", "Vtable.GFS")
        if os.path.exists(gfs_vt_src):
            prmsl_path = os.path.join(wps_out, "Vtable.GFS_PRMSL")
            try:
                with open(gfs_vt_src) as f:
                    prmsl_text = generate_vtable_gfs_prmsl(f.read())
                if os.path.lexists(prmsl_path):
                    os.remove(prmsl_path)
                with open(prmsl_path, "w") as f:
                    f.write(prmsl_text)
            except ColdstartError as exc:
                print("WARNING: Vtable.GFS_PRMSL generation skipped -- %s" % exc)
        default_vt = os.path.join(wps_out, "Vtable.GFS")
        if os.path.exists(default_vt):
            _symlink_one(default_vt, os.path.join(wps_out, "Vtable"))
    else:
        print("WARNING: $WPS_DIR not set -- wps/ exe/table symlinks skipped "
              "(namelist.wps was still written)")
    nlw_src = os.path.join(plan_dir, "namelist.wps")
    nlw_dst = os.path.join(wps_out, "namelist.wps")
    if os.path.exists(nlw_src) and os.path.abspath(nlw_src) != os.path.abspath(nlw_dst):
        shutil.copyfile(nlw_src, nlw_dst)

    for k in range(len(plan["windows"])):
        real_dir = os.path.join(plan_dir, "win_%d" % k, "real")
        wrf_dir = os.path.join(plan_dir, "win_%d" % k, "wrf")
        if wrf_tree:
            _symlink_tree(os.path.join(wrf_tree, "run"), real_dir)
            _symlink_tree(os.path.join(wrf_tree, "run"), wrf_dir)
            real_exe = os.path.join(wrf_tree, "main", "real.exe")
            wrf_exe = os.path.join(wrf_tree, "main", "wrf.exe")
            if os.path.exists(real_exe):
                _symlink_one(real_exe, os.path.join(real_dir, "real.exe"))
            if os.path.exists(wrf_exe):
                _symlink_one(wrf_exe, os.path.join(wrf_dir, "wrf.exe"))
        else:
            os.makedirs(real_dir, exist_ok=True)
            os.makedirs(wrf_dir, exist_ok=True)

    # Every window reads the case's own iofields file(s) from its run directory: WRF hangs at
    # init when one is missing ('Problem opening storm_diags_d01.txt', elevconv 2026-09-28: every
    # non-probe window idled 34 min until the timeout).
    case_iofields_links(plan, plan_dir)
    probe = plan.get("io_probe")
    if probe:
        # the probe window reads the case's own iofields file(s) from its run directory
        wrf_dir = os.path.join(plan_dir, "win_%d" % probe["window"], "wrf")
        for f in probe.get("iofields") or []:
            if f and not os.path.isabs(f["name"]):
                dst = os.path.join(wrf_dir, f["name"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                _symlink_one(f["path"], dst)
        per_window_min = per_window_min + IO_PROBE_TIMEOUT_EXTRA_MIN
    plan["_plan_dir"] = os.path.abspath(plan_dir)
    per_window_forcing = _per_window_forcing_from_forcing_json(plan_dir, plan)
    sbatch_text = render_run_sbatch(plan, site=site, per_window_forcing=per_window_forcing,
                                    per_window_estimate_min=per_window_min)
    sbatch_path = os.path.join(plan_dir, "run.sbatch")
    with open(sbatch_path, "w") as f:
        f.write(sbatch_text)
    os.chmod(sbatch_path, 0o755)
    if not wrf_tree:
        print("WARNING: $WRF_TREE not set -- win_k/{real,wrf}/ table symlinks skipped")
    print("staged %s (%d windows)" % (plan_dir, len(plan["windows"])))
    return sbatch_path


def _default_per_window_forcing(plan, n_windows):
    """[{'prefix','vtable','forcing_interval_h'}, ...] used when PLAN/forcing.json doesn't
    exist yet (stage run before fetch, or an old-style plan/test with no forcing chain at all)
    -- 'GFS'/Vtable.GFS at whatever cadence the plan itself recorded, preserving old
    single-source behaviour exactly."""
    prefix = plan.get("namelist_wps_fields", {}).get("prefix") or \
        plan.get("forcing_chain_primary", {}).get("prefix") or "GFS"
    vtable = plan.get("forcing_chain_primary", {}).get("vtable") or "Vtable.GFS"
    interval_s = plan.get("window_interval_seconds") or plan.get("interval_seconds") or 10800
    return [{"prefix": prefix, "vtable": vtable, "forcing_interval_h": interval_s // 3600}
            for _ in range(n_windows)]


def _per_window_forcing_from_forcing_json(plan_dir, plan):
    """[{'prefix','vtable','forcing_interval_h'}, ...] aligned with plan['windows'] indices, from
    PLAN/forcing.json's per-window 'chosen' record (written by `fetch` -- see
    resolve_and_fetch_forcing) when it exists; a window `fetch` never ran (or forcing.json is
    altogether absent -- e.g. staging before fetching) falls back to _default_per_window_forcing
    for THAT window only."""
    n_windows = len(plan["windows"])
    defaults = _default_per_window_forcing(plan, n_windows)
    forcing_path = os.path.join(plan_dir, "forcing.json")
    if not os.path.exists(forcing_path):
        return defaults
    with open(forcing_path) as f:
        forcing = json.load(f)
    by_win = {w["win"]: w["chosen"] for w in forcing.get("windows", []) if "chosen" in w}
    out = []
    for k in range(n_windows):
        chosen = by_win.get(k)
        if chosen:
            out.append({"prefix": chosen["prefix"], "vtable": chosen["vtable"],
                        "forcing_interval_h": chosen["forcing_interval_h"]})
        else:
            out.append(defaults[k])
    return out


def render_signal_trap_and_hygiene_preamble(nodes, extra_lines_after_alloc_nodes=None,
                                            record_label="WINDOW"):
    """ALLOC_NODES + run_hygiene() + the USR1/TERM/INT signal-trap machinery (kill any in-flight
    exe, reap orphans, record an INTERRUPTED line for the caller's current unit if it tracks one,
    exit 3), ending with 'run_hygiene reap+census'. Shared verbatim by BOTH render_run_sbatch and
    case_rebuild.render_rebuild_sbatch (T5, 2026-09-24) so the two renderers cannot drift out of
    sync on this again -- case_rebuild's own hand-copy of this block omitted ALLOC_NODES
    entirely, so site_hygiene's `mpirun -np "$ALLOC_NODES"` failed 'ALLOC_NODES: unbound
    variable' under `set -u` on scc (jobs 954/955).

    Call this AFTER `site_env` (site_hygiene/ALLOC_NODES need the sourced site file) and after
    the caller's own $STATUS assignment is ready to be included via
    extra_lines_after_alloc_nodes (on_signal only dereferences $STATUS when the caller has set
    CUR_WINDOW to a non-empty id around its own exe launches -- the guard `[ -n "$CUR_WINDOW" ]`
    means $STATUS is never touched at all for a caller that never sets CUR_WINDOW, so it is safe
    to omit extra_lines_after_alloc_nodes/never set CUR_WINDOW). WRF_PID is the caller's own
    responsibility: set it (via `... &`; `WRF_PID=$!`; `wait "$WRF_PID"`) around any exe launch
    that should be killable by PID rather than only by `pkill -x <name>` (kill_and_reap always
    pkills real.exe/wrf.exe by name too, so a foreground/non-backgrounded launch -- e.g.
    `timeout 600 mpirun ... ./real.exe` -- is still interruptible even without WRF_PID)."""
    lines = []
    lines.append("ALLOC_NODES=${SLURM_JOB_NUM_NODES:-%d}" % nodes)
    for extra in (extra_lines_after_alloc_nodes or []):
        lines.append(extra)
    lines.append("")
    lines.append("run_hygiene() {")
    lines.append("  site_hygiene \"$1\"")
    lines.append("  timeout 90 \"${SITE_HYGIENE_CMD[@]}\" 2>&1 || true")
    lines.append("}")
    lines.append("")
    # WRF_PID/CUR_WINDOW are set by the caller's own exe-launch code (a normal function call/
    # inline code in this same shell, not a subshell, so the globals it sets are visible to the
    # signal trap below).
    lines.append("WRF_PID=\"\"")
    lines.append("CUR_WINDOW=\"\"")
    lines.append("kill_and_reap() {")
    lines.append("  if [ -n \"$WRF_PID\" ] && kill -0 \"$WRF_PID\" 2>/dev/null; then")
    lines.append("    kill -TERM \"$WRF_PID\" 2>/dev/null || true; sleep 3")
    lines.append("    kill -0 \"$WRF_PID\" 2>/dev/null && kill -KILL \"$WRF_PID\" 2>/dev/null || true")
    lines.append("  fi")
    lines.append("  pkill -u \"$USER\" -x real.exe 2>/dev/null; pkill -u \"$USER\" -x wrf.exe 2>/dev/null")
    lines.append("  run_hygiene reap")
    lines.append("}")
    lines.append("")
    # SIGNAL_TRAP_DONE: a signal (USR1/TERM/INT) runs kill_and_reap + records INTERRUPTED, then
    # exits 3 -- the subsequent EXIT trap must not repeat the reap (verified pattern from
    # campaign/strict_campaign.sbatch's NODE_HYGIENE_TRAP_DONE).
    lines.append("SIGNAL_TRAP_DONE=0")
    lines.append("on_signal() {")
    lines.append("  local sig=\"$1\"")
    lines.append("  [ \"$SIGNAL_TRAP_DONE\" = 1 ] && return 0")
    lines.append("  SIGNAL_TRAP_DONE=1")
    lines.append("  kill_and_reap")
    lines.append("  if [ -n \"$CUR_WINDOW\" ]; then")
    lines.append("    echo \"%s $CUR_WINDOW rc=1 ok=0 INTERRUPTED sig=$sig\" >> \"$STATUS\""
                  % record_label)
    lines.append("  fi")
    lines.append("  exit 3")
    lines.append("}")
    lines.append("on_exit() {")
    lines.append("  [ \"$SIGNAL_TRAP_DONE\" = 1 ] && return 0")
    lines.append("  run_hygiene reap")
    lines.append("}")
    lines.append("trap 'on_signal USR1' USR1")
    lines.append("trap 'on_signal TERM' TERM")
    lines.append("trap 'on_signal INT' INT")
    lines.append("trap on_exit EXIT")
    lines.append("")
    lines.append("run_hygiene reap+census")
    lines.append("")
    return lines


# metgrid.exe stays on one node (-np ppn): running it on the whole allocation was measured SLOWER
# on scc (dorian2019, 4 nodes: 70 s at 72 ranks -> 327 s at 288 ranks, met_em bit-identical;
# 2026-09-28) -- it is I/O/communication bound at these grid sizes, not compute bound.


def render_run_sbatch(plan, site="scc", per_window_estimate_min=15, wps_estimate_min=20,
                      per_window_forcing=None):
    """SPEC.md 'run.sbatch (site scc)': #SBATCH --exclude=head, -N nodes --exclusive,
    --signal=B:USR1@90, generous --time computed from a stated per-window estimate; site_env /
    site_hygiene / site_launch reused from campaign/sites/scc.sh (NOT reimplemented); geogrid
    (once) -> link_grib.csh (once) -> PER WINDOW: fix-wps-namelist (namelist.wps start/end =
    that window's own [t, t_end], prefix/interval = that window's ACTUALLY chosen forcing source
    -- ungrib/metgrid must NOT span the whole run, or metgrid walks every interval in
    [run start, run end] and dies on the first GRIB time that was never fetched, since `fetch`
    only downloads the windows' own times), relink Vtable to that source's own, ungrib, verify
    EVERY expected '<prefix>:YYYY-MM-DD_HH' intermediate file exists (not just >=1 -- a partial
    ungrib used to pass silently), metgrid, fix-real-namelist, real.exe, check
    wrfinput_d0*/wrfbdy_d01, fix-wrf-namelist, wrf.exe (background + wait so the USR1 trap fires
    -- job 740 postmortem in campaign/strict_campaign.sbatch). A window that fails at any step is
    recorded ok=0 in windows_status.txt and does NOT abort the remaining windows (only the
    one-time geogrid/link_grib preamble aborts the whole job). No LD_PRELOAD, no wrapper around
    wrf.exe -- hard user constraint. `per_window_forcing` (stage_plan reads it from
    PLAN/forcing.json when `fetch` already ran; else _default_per_window_forcing keeps the old
    single-source 'GFS'/Vtable.GFS behaviour)."""
    n_windows = len(plan["windows"])
    per_window_forcing = per_window_forcing or _default_per_window_forcing(plan, n_windows)
    nodes = plan["nodes"]
    ranks = plan["config"]["ranks"]
    omp = plan["config"]["omp"]
    nx = plan["config"]["nx"]
    ny = plan["config"]["ny"]
    numtiles = int(plan["config"].get("numtiles", 1) or 1)
    # T7 (2026-09-24, jobs on scc): ranks-per-node = ceil(ranks/nodes), not floor -- floor
    # under-requests Slurm slots (90 ranks/4 nodes: floor=22 -> --ntasks-per-node=22 -> only 88
    # slots allocated, but real.exe's own mpirun -np 90 --map-by ppr:23:node (and site_launch's
    # self-correcting ppr for wrf.exe) both need 23/node -> mpirun rc=213 'not enough slots').
    # CORES_PER_NODE is scc's (campaign/sites/scc.sh); assert instead of silently over-packing.
    ppn = -(-ranks // max(nodes, 1))
    assert ppn <= 72, ("ppn=%d exceeds CORES_PER_NODE=72 (ranks=%d nodes=%d)"
                       % (ppn, ranks, nodes))
    real_tpn = ppn
    total_min = wps_estimate_min + n_windows * per_window_estimate_min
    # generous: 2x the stated estimate, minimum 60 minutes
    time_min = max(60, total_min * 2)
    hh = time_min // 60
    mm = time_min % 60
    time_str = "%02d:%02d:00" % (hh, mm)
    wrf_timeout_s = per_window_estimate_min * 60

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("#SBATCH --job-name=coldstart-sample")
    lines.append("#SBATCH --partition=compute")
    # WTUNE_SBATCH_EXCLUDE: extra nodes to avoid (comma list), e.g. one carrying non-Slurm load
    # (scc work5, 2026-09-27: a root VS Code server stole a core and slowed wrf.exe 5x).
    # merged with the controller's own node-health pre-flight failures (scc_sweep.excluded_nodes)
    try:
        import scc_sweep
        excluded = scc_sweep.excluded_nodes()
    except ImportError:
        excluded = ["head"] + [n.strip() for n in os.environ.get("WTUNE_SBATCH_EXCLUDE", "").split(",")
                               if n.strip()]
    lines.append("#SBATCH --exclude=%s" % ",".join(excluded))
    lines.append("#SBATCH -N %d --exclusive --ntasks-per-node=%d" % (nodes, ppn))
    lines.append("#SBATCH --time=%s" % time_str)
    lines.append("#SBATCH --signal=B:USR1@90")
    plan_dir_abs = plan.get("_plan_dir")
    if plan_dir_abs:
        lines.append("#SBATCH --output=%s/run-%%j.out" % os.path.abspath(plan_dir_abs))
    lines.append("set -uo pipefail")
    lines.append("")
    lines.append("# estimated %d min (wps ~%dmin + %d windows x ~%dmin/window); --time is 2x that"
                  % (total_min, wps_estimate_min, n_windows, per_window_estimate_min))
    lines.append("")
    # Slurm runs a COPY of the batch script from its spool directory, so BASH_SOURCE's dirname is
    # not the plan directory: the path has to be written in at stage time.
    if plan_dir_abs:
        lines.append("PLAN=%s" % shlex.quote(os.path.abspath(plan_dir_abs)))
    else:
        lines.append("PLAN=$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)")
    # TOOL: absolute path of THIS script, written at stage time -- "$PLAN/../coldstart_sample.py"
    # does not exist (PLAN_DIR is not a subdirectory of tools/); every fix-*-namelist call below
    # must invoke the actual installed location.
    lines.append("TOOL=%s" % _sh_quote(_SELF_PATH))
    ml_model = plan.get("ml_model")
    if ml_model:
        # K5 (2026-09-24): ML state completion, default OFF -- only emitted when the plan was
        # built with --ml-model. TOOL_SCM/GRIB_PY are ONLY used inside run_window()'s ML-apply
        # step below; every other fix-*-namelist call keeps using plain "python3 $TOOL" (stdlib
        # only, /usr/bin/python3 is fine for those).
        lines.append("TOOL_SCM=%s" % _sh_quote(_SCM_PATH))
        lines.append("ML_MODEL=%s" % _sh_quote(os.path.abspath(ml_model)))
        lines.append("GRIB_PY=${GRIB_PYTHON:-$HOME/wtune/envs/grib/.pixi/envs/default/bin/python}")
    lines.append("SITE=%s" % site)
    lines.append("CAMPAIGN=${CAMPAIGN:-$HOME/wtune/campaign}")
    lines.append("source \"$CAMPAIGN/sites/$SITE.sh\"   # site_env / site_launch / site_hygiene -- "
                  "reused, not reimplemented")
    lines.append("source ~/wtune-scc/env.sh")
    lines.append("export PATH=$NETCDF/bin:$PATH")
    lines.append("ulimit -s unlimited")
    lines.append("site_env")
    lines.extend(render_signal_trap_and_hygiene_preamble(
        nodes,
        extra_lines_after_alloc_nodes=["STATUS=\"$PLAN/windows_status.txt\"",
                                      ": > \"$STATUS\""],
        record_label="WINDOW"))
    max_dom = plan.get("max_dom", 1)
    geo_em_expected = ["geo_em.d%02d.nc" % d for d in range(1, max_dom + 1)]
    lines.append("cd \"$PLAN/wps\"")
    # geo_em depends only on the &geogrid/&share domain definition and the static data, never on
    # the date: cache it per domain (rebuild job 1238: geogrid was 62 of 845 s, repeated per point).
    lines.append("GEO_CACHE=\"${WTUNE_GEO_CACHE:-$HOME/wtune-runs/geo_cache}/$(sed -n '/&share/,/^\\//p;/&geogrid/,/^\\//p' namelist.wps | grep -vE 'start_date|end_date|interval_seconds' | sha1sum | cut -c1-16)\"")
    lines.append("GEO_CACHED=1; for f in %s; do [ -s \"$GEO_CACHE/$f\" ] || GEO_CACHED=0; done"
                 % " ".join(geo_em_expected))
    lines.append("if [ $GEO_CACHED = 1 ]; then")
    lines.append("  echo \"=== geogrid: cached $GEO_CACHE ($(date -u +%FT%TZ)) ===\"")
    lines.append("  for f in %s; do ln -sf \"$GEO_CACHE/$f\" .; done" % " ".join(geo_em_expected))
    lines.append("  echo 'Successful completion of geogrid (cached)' > geogrid.log.cached; rc=0")
    lines.append("else")
    lines.append("echo \"=== geogrid.exe ($(date -u +%FT%TZ)) ===\"")
    lines.append("mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./geogrid.exe" % (ppn, ppn))
    lines.append("rc=$?; pkill -u \"$USER\" -x geogrid.exe 2>/dev/null")
    lines.append("fi")
    # geogrid.exe MPI_Abort()s with errorcode 0 on a missing GEOGRID.TBL (jobs 941/942,
    # 2026-09-24), so mpirun's own rc alone does NOT catch that failure -- also require every
    # geo_em.d0N.nc (N=1..max_dom) to exist and the log to say 'Successful completion'.
    lines.append("GEO_EM_EXPECTED=(%s)" % " ".join(shlex.quote(f) for f in geo_em_expected))
    lines.append("GEO_EM_MISSING=()")
    lines.append("for f in \"${GEO_EM_EXPECTED[@]}\"; do [ -f \"$f\" ] || GEO_EM_MISSING+=(\"$f\"); done")
    lines.append("if [ $rc -ne 0 ] || [ ${#GEO_EM_MISSING[@]} -gt 0 ] "
                  "|| ! grep -q 'Successful completion' geogrid.log* 2>/dev/null; then")
    lines.append("  echo \"FATAL geogrid rc=$rc missing=${GEO_EM_MISSING[*]:-none}\"; exit 1")
    lines.append("fi")
    lines.append("if [ $GEO_CACHED = 0 ]; then mkdir -p \"$GEO_CACHE\" && cp -f %s \"$GEO_CACHE/\" 2>/dev/null; fi"
                 % " ".join(geo_em_expected))
    lines.append("")
    all_intermediate = bool(per_window_forcing) and all(
        f.get("vtable") == "none" for f in per_window_forcing)
    if all_intermediate:
        lines.append("echo \"=== link_grib.csh: skipped (intermediate forcing) ===\"")
    lines.append("if [ %d = 0 ]; then" % int(all_intermediate))
    lines.append("echo \"=== link_grib.csh ($(date -u +%FT%TZ)) ===\"")
    lines.append("rm -f GRIBFILE.*")
    # T9 (2026-09-24): link exactly what `fetch` wrote (grib_manifest.txt, written by
    # resolve_and_fetch_forcing) rather than assuming an extension -- FNL (d083002) is cached as
    # .grib and ncei_g3/g4 as .grb, not .grib2, and a *.grib2-only glob FATAL'd 'no *.grib2
    # files' even with $PLAN/grib fully populated. Fall back to a widened glob only if the
    # manifest is missing (e.g. grib/ was staged by hand, not via `fetch`).
    lines.append("if [ -s \"$PLAN/grib_manifest.txt\" ]; then")
    lines.append("  mapfile -t GRIB_FILES < <(sed \"s#^#$PLAN/grib/#\" \"$PLAN/grib_manifest.txt\")")
    lines.append("else")
    lines.append("  shopt -s nullglob")
    lines.append("  GRIB_FILES=(\"$PLAN\"/grib/*.grib2 \"$PLAN\"/grib/*.grib \"$PLAN\"/grib/*.grb)")
    lines.append("  shopt -u nullglob")
    lines.append("fi")
    lines.append("if [ ${#GRIB_FILES[@]} -eq 0 ]; then")
    lines.append("  echo \"FATAL: no forcing files in $PLAN/grib -- run "
                  "'coldstart_sample.py fetch --plan $PLAN' first\"; exit 1")
    lines.append("fi")
    lines.append("./link_grib.csh \"${GRIB_FILES[@]}\"")
    lines.append("[ $? -ne 0 ] && { echo \"FATAL link_grib.csh failed\"; exit 1; }")
    lines.append("fi")
    lines.append("")

    lines.append("run_window() {")
    lines.append("  local k=\"$1\" prefix=\"$2\" vtable=\"$3\"; shift 3")
    lines.append("  local n_ungrib=\"$1\"; shift")
    lines.append("  local expected=(); for ((i = 0; i < n_ungrib; i++)); do expected+=(\"$1\"); shift; done")
    lines.append("  local n_metem=\"$1\"; shift")
    lines.append("  local expected_met=(); for ((i = 0; i < n_metem; i++)); do expected_met+=(\"$1\"); shift; done")
    lines.append("  local rc missing=() missing_met=() real_path")
    lines.append("  CUR_WINDOW=\"$k\"")
    lines.append("  LAST_REAL_PATH=\"\"")
    # T2 (2026-09-24): --real-only never runs this far into a "wrf" branch at all, so
    # LAST_WRF_STATUS's default of "skipped" is what ends up in windows_status.txt for that
    # case; the non-real-only wrf.exe branch below overwrites it to "ok" on success.
    lines.append("  LAST_WRF_STATUS=\"skipped\"")
    # K5 (2026-09-24): "off" unless --ml-model was given (ML_MODEL unset -> the block below
    # never runs at all); "skipped" if it ran but nothing was improved/it failed; "applied" only
    # on a genuine swap-in.
    lines.append("  LAST_ML_STATUS=\"off\"")
    lines.append("  LAST_ML_TARGETS=\"\"")
    # OMP_NUM_THREADS=1 for WPS + real.exe (unset for the rest of the window until the wrf.exe
    # launch below sets it back to the predicted config's own omp) -- otherwise these steps are
    # not timed/run under the same threading conditions as the ground-truth runner
    # (predict_validate/truth_n.sbatch), which is the whole point of a cold-start measurement.
    lines.append("  export OMP_NUM_THREADS=1")
    lines.append("  cd \"$PLAN/wps\" || return 1")
    lines.append("  echo \"=== window $k: wps namelist + ungrib (prefix=$prefix vtable=$vtable) "
                  "($(date -u +%FT%TZ)) ===\"")
    lines.append("  python3 \"$TOOL\" fix-wps-namelist --plan \"$PLAN\" --win \"$k\" || return 1")
    # Vtable.GFS/Vtable.CFSR were symlinked into $PLAN/wps by their real names at stage time
    # (from $WPS_DIR/ungrib/Variable_Tables); relink the generic 'Vtable' ungrib itself reads to
    # whichever this window's chosen forcing source actually needs.
    # vtable "none": the intermediate files were written directly at fetch time (ERA5 pin).
    lines.append("  if [ \"$vtable\" = none ]; then rc=0")
    lines.append("  else")
    lines.append("  if [ -e \"$vtable\" ]; then ln -sf \"$vtable\" Vtable")
    lines.append("  else echo \"window $k: $vtable not staged in $PLAN/wps (is \\$WPS_DIR set?)\" >&2; return 1; fi")
    lines.append("  rm -f \"${prefix}:\"*")
    lines.append("  ./ungrib.exe")
    lines.append("  rc=$?; pkill -u \"$USER\" -x ungrib.exe 2>/dev/null")
    lines.append("  fi")
    lines.append("  for f in \"${expected[@]}\"; do [ -f \"$f\" ] || missing+=(\"$f\"); done")
    lines.append("  if [ $rc -ne 0 ] || [ ${#missing[@]} -gt 0 ]; then")
    lines.append("    echo \"window $k: ungrib rc=$rc missing=${missing[*]:-none}\" >&2; return 1")
    lines.append("  fi")
    lines.append("")
    lines.append("  echo \"=== window $k: metgrid.exe ($(date -u +%FT%TZ)) ===\"")
    lines.append("  mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./metgrid.exe" % (ppn, ppn))
    lines.append("  rc=$?; pkill -u \"$USER\" -x metgrid.exe 2>/dev/null")
    # metgrid.exe also MPI_Abort()s with errorcode 0 on a missing METGRID.TBL -- require every
    # expected met_em.d0N.<time>.nc AND the log to say 'Successful completion' (jobs 941/942).
    lines.append("  for f in \"${expected_met[@]}\"; do [ -f \"$f\" ] || missing_met+=(\"$f\"); done")
    lines.append("  if [ $rc -ne 0 ] || [ ${#missing_met[@]} -gt 0 ] "
                  "|| ! grep -q 'Successful completion' metgrid.log* 2>/dev/null; then")
    lines.append("    echo \"window $k: metgrid rc=$rc missing=${missing_met[*]:-none}\" >&2; return 1")
    lines.append("  fi")
    lines.append("")
    lines.append("  echo \"=== window $k: real.exe ($(date -u +%FT%TZ)) ===\"")
    lines.append("  cd \"$PLAN/win_${k}/real\" || return 1")
    lines.append("  rm -f met_em.d0*.*.nc")
    # nests: link met_em for EVERY domain (d02..dN carry at least their initialization time t;
    # d01 carries every time in [t, t_end] -- both come from this window's own namelist.wps).
    lines.append("  ln -sf \"$PLAN\"/wps/met_em.d0*.*.nc .")
    # metgrid always writes 'YYYY-MM-DD_HH:MM:SS' names, but real.exe under the case's own
    # nocolons=.true. (lafe) opens 'YYYY-MM-DD_HH_MM_SS' (scc job 983: every window FATAL
    # 'error opening met_em.d01.2017-08-23_06_00_00.nc'); link both spellings, namelist untouched
    lines.append("  for f in met_em.d0*.*.nc; do [ \"$f\" = \"${f//:/_}\" ] || ln -sf \"$f\" \"${f//:/_}\"; done")
    lines.append("  python3 \"$TOOL\" fix-real-namelist --plan \"$PLAN\" --win \"$k\" || return 1")
    # 2026-09-24 cluster findings (sc26p1/severe2023): real.exe HANGS (100% CPU, no rsl
    # progress) at SOME decompositions (3x3, 5x5 observed) with no odd/square pattern -- do not
    # trust nx/ny alone. Try the predicted config's own decomposition under `timeout 600`
    # first (this is the decomposition wrf.exe will use anyway, so it's the useful measurement
    # when it doesn't hang); on rc=124 or missing wrfinput_d01/wrfbdy_d01, kill any survivors,
    # rewrite the namelist nproc_x=nproc_y=1 (--serial), and retry under `timeout 1800`.
    lines.append("  timeout 600 mpirun -np %d --map-by ppr:%d:node:PE=1 --bind-to core ./real.exe"
                  % (ranks, real_tpn))
    lines.append("  rc=$?; pkill -u \"$USER\" -x real.exe 2>/dev/null")
    lines.append("  real_path=decomp")
    lines.append("  if [ $rc -eq 124 ] || [ ! -f wrfinput_d01 ] || [ ! -f wrfbdy_d01 ]; then")
    lines.append("    echo \"window $k: real.exe decomposed attempt rc=$rc -- retrying serial "
                  "(nproc_x=nproc_y=1)\" >&2")
    lines.append("    python3 \"$TOOL\" fix-real-namelist --plan \"$PLAN\" --win \"$k\" --serial || return 1")
    lines.append("    timeout 1800 mpirun -np 1 --map-by ppr:1:node:PE=1 --bind-to core ./real.exe")
    lines.append("    rc=$?; pkill -u \"$USER\" -x real.exe 2>/dev/null")
    lines.append("    real_path=serial")
    lines.append("  fi")
    lines.append("  LAST_REAL_PATH=\"$real_path\"")
    lines.append("  [ $rc -ne 0 ] && return 1")
    lines.append("  [ -f wrfinput_d01 ] && [ -f wrfbdy_d01 ] || return 1")
    lines.append("")
    if ml_model:
        # K5 (2026-09-24): ML state completion on the fresh wrfinput_d01, BEFORE wrf.exe sees it
        # -- emitted ONLY when the plan was built with --ml-model (Default: OFF means no trace
        # of this step in the rendered script at all, not just a runtime no-op guard). Never
        # /usr/bin/python3 here: this needs numpy+netCDF4 (GRIB_PY, from the PLAN-time
        # preamble). Swap in wrfinput_d01.ml ONLY on a genuine apply (rc=0 AND the file exists --
        # --targets-from-report legitimately writes nothing when no target improved, which is
        # NOT a failure); otherwise keep the original untouched.
        lines.append("  ML_OUT=$(\"$GRIB_PY\" \"$TOOL_SCM\" apply --model \"$ML_MODEL\" "
                      "--targets-from-report --wrfinput wrfinput_d01 --out wrfinput_d01.ml 2>&1)")
        lines.append("  ml_rc=$?")
        lines.append("  LAST_ML_TARGETS=$(printf '%s\\n' \"$ML_OUT\" | "
                      "grep '^APPLIED_TARGETS=' | tail -1 | sed 's/^APPLIED_TARGETS=//')")
        lines.append("  if [ $ml_rc -eq 0 ] && [ -f wrfinput_d01.ml ]; then")
        lines.append("    mv wrfinput_d01 wrfinput_d01.noml && mv wrfinput_d01.ml wrfinput_d01")
        lines.append("    LAST_ML_STATUS=\"applied\"")
        lines.append("  else")
        lines.append("    rm -f wrfinput_d01.ml")
        lines.append("    LAST_ML_STATUS=\"skipped\"")
        lines.append("    printf '%s\\n' \"$ML_OUT\" >&2")
        lines.append("  fi")
        lines.append("")
    real_only = bool(plan.get("real_only"))
    if real_only:
        # T2 (2026-09-24): --real-only stops each window here -- no fix-wrf-namelist, no
        # wrf.exe. LAST_WRF_STATUS stays "skipped" (set at the top of run_window()) so
        # windows_status.txt records it explicitly rather than just omitting the field.
        lines.append("  return 0")
    else:
        lines.append("  echo \"=== window $k: wrf.exe ($(date -u +%FT%TZ)) ===\"")
        lines.append("  cd \"$PLAN/win_${k}/wrf\" || return 1")
        # nests: every wrfinput_d0N (d01..dN) + wrfbdy_d01 (only d01 has a time-varying lateral
        # boundary); wrflowinp_d0N only exists when the case uses SST/lake update input -- link
        # it only if present, never a hard requirement.
        lines.append("  ln -sf ../real/wrfinput_d0* .")
        lines.append("  ln -sf ../real/wrfbdy_d01 .")
        lines.append("  shopt -s nullglob; WRFLOWINP=(../real/wrflowinp_d0*); shopt -u nullglob")
        lines.append("  [ ${#WRFLOWINP[@]} -gt 0 ] && ln -sf ../real/wrflowinp_d0* .")
        # Thompson computes qr_acr_q*/freezeH2O tables at init when absent (job 1238: ~70 s of
        # the first step). They depend only on the WRF build: cache per build.
        lines.append("  TBL_CACHE=\"${WTUNE_TABLE_CACHE:-$HOME/wtune-runs/thompson_tables}/$(readlink -f \"$WRF_TREE/main/wrf.exe\" | sha1sum | cut -c1-16)\"")
        lines.append("  shopt -s nullglob; for f in \"$TBL_CACHE\"/qr_acr_q*.dat \"$TBL_CACHE\"/freezeH2O.dat; do "
                     "[ -e \"$(basename \"$f\")\" ] || ln -s \"$f\" .; done; shopt -u nullglob")
        lines.append("  python3 \"$TOOL\" fix-wrf-namelist --plan \"$PLAN\" --win \"$k\" "
                      "--nproc-x %d --nproc-y %d --numtiles %d || return 1" % (nx, ny, numtiles))
        # OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND for the ACTUAL timed run must match the
        # ground-truth runner (predict_validate/truth_n.sbatch) exactly, or the window isn't
        # measuring the same conditions the prediction will be compared against.
        lines.append("  export OMP_NUM_THREADS=%d" % omp)
        lines.append("  unset OMP_PLACES OMP_PROC_BIND")
        lines.append("  site_launch %d %d %d block" % (ranks, ppn, omp))
        lines.append("  timeout %d \"${SITE_LAUNCH_CMD[@]}\" > wrf.out 2>&1 &" % wrf_timeout_s)
        lines.append("  WRF_PID=$!")
        # A WRF FATAL (MPI_Abort) can leave mpirun hanging until the window timeout (elevconv
        # 2026-09-28: 'namelist settings are wrong' at init, then 40 min idle). Stop the launch
        # 30 s after any rank prints FATAL CALLED.
        # Also a silent init hang: 'Problem opening <file>' (e.g. a missing iofields file) and no
        # 'Timing for main' line within 120 s of it.
        lines.append("  ( po=0; while kill -0 \"$WRF_PID\" 2>/dev/null; do "
                     "if grep -lqs 'FATAL CALLED' rsl.error.* 2>/dev/null; then sleep 30; "
                     "kill -TERM \"$WRF_PID\" 2>/dev/null; sleep 20; kill -KILL \"$WRF_PID\" 2>/dev/null; "
                     "echo \"window $k: WRF FATAL CALLED -- launch stopped\" >&2; break; fi; "
                     "if grep -qs 'Problem opening' rsl.error.0000 && ! grep -qs 'Timing for main' rsl.error.0000; then "
                     "po=$((po+10)); if [ $po -ge 120 ]; then kill -TERM \"$WRF_PID\" 2>/dev/null; sleep 20; "
                     "kill -KILL \"$WRF_PID\" 2>/dev/null; "
                     "echo \"window $k: WRF stuck at init after 'Problem opening' -- launch stopped\" >&2; break; fi; fi; "
                     "sleep 10; done ) &")
        lines.append("  FATAL_WATCH_PID=$!")
        lines.append("  wait \"$WRF_PID\"")
        lines.append("  rc=$?; WRF_PID=\"\"")
        lines.append("  kill \"$FATAL_WATCH_PID\" 2>/dev/null; wait \"$FATAL_WATCH_PID\" 2>/dev/null")
        lines.append("  pkill -u \"$USER\" -x wrf.exe 2>/dev/null")
        lines.append("  [ $rc -ne 0 ] && return 1")
        lines.append("  shopt -s nullglob; TBLS=(qr_acr_q*.dat freezeH2O.dat); shopt -u nullglob")
        lines.append("  for f in \"${TBLS[@]}\"; do [ -L \"$f\" ] || { mkdir -p \"$TBL_CACHE\" && "
                     "[ -e \"$TBL_CACHE/$f\" ] || cp -f \"$f\" \"$TBL_CACHE/\"; }; done")
        lines.append("  LAST_WRF_STATUS=\"ok\"")
        lines.append("  return 0")
    lines.append("}")
    lines.append("")

    for k in range(n_windows):
        w = plan["windows"][k]
        pwf = per_window_forcing[k]
        prefix = pwf["prefix"]
        vtable = pwf["vtable"]
        interval_s = pwf["forcing_interval_h"] * 3600
        T = datetime.datetime.strptime(w["t"], _WRF_TIME_FMT)
        T_end = datetime.datetime.strptime(w.get("t_end", w["t"]), _WRF_TIME_FMT)
        expected = expected_wps_intermediate_files(prefix, T, T_end, interval_s)
        expected_met = expected_met_em_files(max_dom, T, T_end, interval_s)
        argv = ([str(k), prefix, vtable, str(len(expected))] + expected
                + [str(len(expected_met))] + expected_met)
        args = " ".join(shlex.quote(x) for x in argv)
        lines.append("if run_window %s; then" % args)
        lines.append("  echo \"WINDOW %d rc=0 ok=1 real=$LAST_REAL_PATH wrf=$LAST_WRF_STATUS "
                      "ml=$LAST_ML_STATUS ml_targets=$LAST_ML_TARGETS\" >> \"$STATUS\"" % k)
        lines.append("else")
        lines.append("  echo \"WINDOW %d rc=1 ok=0 real=$LAST_REAL_PATH wrf=$LAST_WRF_STATUS "
                      "ml=$LAST_ML_STATUS ml_targets=$LAST_ML_TARGETS\" >> \"$STATUS\"" % k)
        lines.append("fi")
        lines.append("CUR_WINDOW=\"\"")
        lines.append("")

    lines.append("run_hygiene reap+census")
    lines.append("echo \"=== DONE ($(date -u +%FT%TZ)) ===\"")
    lines.append("cat \"$STATUS\"")
    text = "\n".join(lines) + "\n"
    assert "LD_PRELOAD" not in text
    assert "--wrap" not in text
    return text


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def interpolate_step_costs(step_times_s, anchors):
    """anchors: [(t_seconds, s), ...] (window measured-part midpoints). step_times_s: seconds
    from the same origin, one per predicted d01 step. Returns [s(t), ...] -- piecewise-linear
    interpolation between anchors, held flat outside [min(anchor t), max(anchor t)]."""
    if not anchors:
        raise ColdstartError("interpolate_step_costs: no anchors")
    anchors = sorted(anchors, key=lambda a: a[0])
    ts = [a[0] for a in anchors]
    ss = [a[1] for a in anchors]
    out = []
    for t in step_times_s:
        if t <= ts[0]:
            out.append(ss[0])
            continue
        if t >= ts[-1]:
            out.append(ss[-1])
            continue
        i = 0
        while i + 1 < len(ts) and ts[i + 1] < t:
            i += 1
        t0, t1 = ts[i], ts[i + 1]
        s0, s1 = ss[i], ss[i + 1]
        frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        out.append(s0 + frac * (s1 - s0))
    return out


_WINDOW_STATUS_RE = re.compile(r"^WINDOW\s+(\d+)\s+rc=(-?\d+)\s+ok=([01])")


def read_windows_status(plan_dir):
    """{window_index: ok_bool} from PLAN_DIR/windows_status.txt (written by run.sbatch's
    run_window(), one 'WINDOW k rc=.. ok=0|1[ ...]' line per window -- a trailing
    'INTERRUPTED sig=...' from the USR1/TERM/INT trap is tolerated, only the leading fields are
    parsed). {} (nothing filtered) when the file doesn't exist -- e.g. every offline test that
    calls score_windows_from_rsl directly against synthetic rsl files it wrote itself, with no
    run.sbatch ever having run."""
    path = os.path.join(plan_dir, "windows_status.txt")
    status = {}
    if not os.path.exists(path):
        return status
    for line in open(path):
        m = _WINDOW_STATUS_RE.match(line.strip())
        if m:
            status[int(m.group(1))] = (m.group(3) == "1")
    return status


def score_windows_from_rsl(plan, sample_points, calibration):
    """Per-window s_k + pooled radiation samples (SPEC.md Scoring, first paragraph). Windows
    marked ok=0 in windows_status.txt (a failed real.exe/wrf.exe, or one interrupted by the USR1
    trap -- see render_run_sbatch) are skipped, not treated as a hard error; it's an error only
    when EVERY window is unusable. Returns (windows_out, pooled_radiation_samples,
    run_start_dt_seconds_anchors)."""
    dt = plan["dt_s"]
    radt = plan["radt_min"]
    stepra = plan["stepra"]
    spinup_seconds = plan["spinup_min"] * 60.0
    run_start_dt = datetime.datetime.strptime(plan["run_start"], _WRF_TIME_FMT)
    status = read_windows_status(plan["_plan_dir"])

    windows_out = []
    pooled_radiation_samples = []
    anchors = []
    skipped = []
    for k, w in enumerate(plan["windows"]):
        if not status.get(k, True):
            skipped.append(k)
            continue
        T = datetime.datetime.strptime(w["t"], _WRF_TIME_FMT)
        rsl_path = plan.get("_rsl_path_override", {}).get(k) or \
            os.path.join(plan["_plan_dir"], "win_%d" % k, "wrf", "rsl.error.0000")
        steps = predict.parse_rsl_d01_step_seconds(rsl_path)
        cutoff = T + datetime.timedelta(seconds=spinup_seconds)
        kept = [(sd, sec) for sd, sec in steps if sd is not None and sd >= cutoff]
        if not kept:
            raise ColdstartError("window %d: no post-spinup steps found in %s (spinup=%.0fs)"
                                  % (k, rsl_path, spinup_seconds))
        # A step whose timer absorbed a history/restart/auxhist write is not a compute sample
        # (rsl_write_absorbing_d01_times). It keeps its index for radiation-phase detection, with
        # its wall replaced by the median of the other kept steps, and is then left out of both
        # the normal and the radiation samples.
        absorbed_times = rsl_write_absorbing_d01_times(rsl_path) if os.path.exists(rsl_path) else set()
        absorbed = [sd.strftime(_WRF_TIME_FMT) in absorbed_times for sd, _ in kept]
        wall = [sec for _, sec in kept]
        if any(absorbed):
            clean = [sec for i, sec in enumerate(wall) if not absorbed[i]]
            if not clean:
                raise ColdstartError("window %d: every post-spinup step absorbed a write in %s"
                                      % (k, rsl_path))
            fill = statistics.median(clean)
            wall = [fill if absorbed[i] else sec for i, sec in enumerate(wall)]
        if plan.get("measure_steps", 2 * stepra) < 2 * stepra:
            # An explicitly shortened window (compute_window_seconds' measure_steps) has too few
            # samples for classify_steps_detected's phase detection. Each window is a fresh run
            # starting at T, so its radiation steps are known: local step n (n*dt after T) with
            # n % stepra == 1.
            flags = [int(round((sd - T).total_seconds() / dt)) % stepra == 1 for sd, _ in kept]
        else:
            flags = predict.classify_steps_detected(wall, stepra)
        normal = [sec for i, (_sd, sec) in enumerate(kept) if not flags[i] and not absorbed[i]]
        if not normal:
            raise ColdstartError("window %d: no normal (non-radiation) steps after spin-up" % k)
        s_k = statistics.median(normal)
        n_rad = 0
        rad_ratios = []   # this window's OWN (sec/s_k) samples -- see r_local below
        for i, (sd, sec) in enumerate(kept):
            if flags[i] and not absorbed[i]:
                n_rad += 1
                rad_ratios.append(sec / s_k)
                frac = predict.sunlit_fraction(sample_points, sd) if sample_points else None
                if frac is not None:
                    pooled_radiation_samples.append((frac, sec / s_k))
        # r_local (2026-09-26, wtune probe multi-point "rebuild" source wiring, additive): this
        # window's OWN radiation-step ratio, median of rad_ratios (None if n_rad==0) -- unlike
        # pooled_radiation_samples (flat across every window, used for the whole-plan day/night
        # fit), this is per-window, for a caller that wants ONE (s_k, r) pair per window/point
        # (e.g. wtune_probe_plan.PointMeasurement) instead of the pooled fit.
        r_local = statistics.median(rad_ratios) if rad_ratios else None
        mid_dt = T + datetime.timedelta(seconds=spinup_seconds + (plan["W_seconds"] - spinup_seconds) / 2.0)
        # R6-D date-shifted sampling: T (and so mid_dt) is on the ANALOG timeline the window
        # actually ran at; map it back to the STATED timeline (what the whole-run prediction's
        # own step grid, anchored at run_start_dt=stated_start, is indexed on) before placing
        # the anchor. plan.get("shift_s") is 0.0 for an unshifted plan, so this is a no-op then.
        mid_dt_stated = mid_dt - datetime.timedelta(seconds=plan.get("shift_s", 0.0) or 0.0)
        anchors.append(((mid_dt_stated - run_start_dt).total_seconds(), s_k))
        forcing = _load_window_forcing(plan["_plan_dir"], k) or plan.get("forcing_chain_primary")
        windows_out.append({"t": w["t"], "reason": w["reason"], "s_k": s_k,
                            "n_normal": len(normal), "n_rad": n_rad, "r_local": r_local,
                            "shift_s": w.get("shift_s", 0.0), "forcing": forcing})
    if not windows_out:
        raise ColdstartError(
            "score: no usable windows (all %d marked ok=0 in windows_status.txt)"
            % len(plan["windows"]))
    if skipped:
        print("skipping windows (ok=0 in windows_status.txt): %s" % skipped)
    return windows_out, pooled_radiation_samples, anchors


def resolve_r_rad(pooled_radiation_samples, calibration):
    default_r_rad = calibration.get("radiation", {}).get("r_rad_default", 5.0)
    day_night_ratio = calibration.get("radiation", {}).get(
        "r_rad_day_night_ratio", predict.DEFAULT_R_RAD_DAY_NIGHT_RATIO)
    r_night, r_day, source = predict.fit_day_night_r_rad(pooled_radiation_samples, day_night_ratio)
    if r_night is None:
        cal_rad = calibration.get("radiation", {})
        if cal_rad.get("r_rad_default_night") and cal_rad.get("r_rad_default_day"):
            r_night = float(cal_rad["r_rad_default_night"])
            r_day = float(cal_rad["r_rad_default_day"])
            source = "default_calibrated"
        else:
            r_night = r_day = default_r_rad
            source = "default_unmeasured"
    return r_night, r_day, source


def predicted_compute_total(steps_n, dt, stepra, run_start_dt, anchors, sample_points,
                            r_night, r_day):
    """SPEC.md Scoring, 'Whole run': s(t_i) = piecewise-linear interpolation of s_k at each
    window's measured-part midpoint; compute = sum_i s(t_i) + sum over schedule radiation steps
    (classify_steps) of (r(t_i)-1)*s(t_i), r(t)=r_night+(r_day-r_night)*sunlit_fraction(t)."""
    step_times_s = [i * dt for i in range(1, steps_n + 1)]
    s_values = interpolate_step_costs(step_times_s, anchors)
    flags = predict.classify_steps(steps_n, stepra)
    rad_extra = 0.0
    for i in range(steps_n):
        if not flags[i]:
            continue
        t_i = run_start_dt + datetime.timedelta(seconds=step_times_s[i])
        frac = predict.sunlit_fraction(sample_points, t_i) if sample_points else None
        r_t = (r_night + (r_day - r_night) * frac) if frac is not None else (r_night + r_day) / 2.0
        rad_extra += (r_t - 1.0) * s_values[i]
    return sum(s_values) + rad_extra, s_values


def score_plan(plan_dir, calibration_path=DEFAULT_CALIBRATION, json_out=None, include_io=False):
    with open(os.path.join(plan_dir, "plan.json")) as f:
        plan = json.load(f)
    if plan.get("real_only"):
        raise ColdstartError(
            "score: this plan was built with --real-only (real.exe boundary/init check only, "
            "no wrf.exe) -- there is no rsl.error.0000 step-timing data to score. Re-run `plan` "
            "without --real-only to get a wall-time prediction.")
    plan["_plan_dir"] = plan_dir
    nl = load_namelist(plan["namelist_input"])
    calibration = predict.load_calibration(calibration_path)

    sample_points, geom_source = predict.read_wrfinput_sample_points(
        os.path.join(plan_dir, "win_0", "real"))
    if not sample_points:
        geo = plan["geo"]
        sample_points = domain_lattice_from_geom(geo["ref_lat"], geo["ref_lon"], geo["dx"],
                                                  geo["e_we"], dy_m=geo["dy"], e_sn=geo["e_sn"])
        geom_source = "plan_geom"

    windows_out, pooled_radiation_samples, anchors = score_windows_from_rsl(
        plan, sample_points, calibration)
    r_night, r_day, r_source = resolve_r_rad(pooled_radiation_samples, calibration)

    run_start_dt = datetime.datetime.strptime(plan["run_start"], _WRF_TIME_FMT)
    steps_n, run_s, dt_full, steps_note = predict.get_d01_steps(nl)
    compute_total, s_values = predicted_compute_total(
        steps_n, dt_full, plan["stepra"], run_start_dt, anchors, sample_points, r_night, r_day)

    s_ks = [w["s_k"] for w in windows_out]
    median_s = statistics.median(s_ks)
    flat_total = median_s * steps_n

    config = plan["config"]
    # case_dir = PLAN/win_0/real, NOT plan["case_dir"]: in an unknown case the original case_dir
    # has no wrfinput_d01 at all (that's the whole premise), and in the held-out validation test
    # (a case that DOES happen to have one) reading it back here would be leakage -- build_prediction
    # only uses case_dir for read_wrfinput_sample_points/predict_init, and win_0/real's own
    # real.exe-generated wrfinput_d01 is the right (and only legitimately available) substitute.
    win0_real_dir = os.path.join(plan_dir, "win_0", "real")
    pred = predict.build_prediction(plan["namelist_input"], win0_real_dir, plan["nodes"],
                                    config, ledger_paths=[], calibration_path=calibration_path,
                                    io_form_override=None, io_target=None, rsl_paths=None)
    io = pred["io"]
    io_range = pred["io_range"]
    init = pred["init"]

    drift_pct = calibration.get("drift_band_pct", 5.0)
    spread_frac = (max(s_ks) - min(s_ks)) / median_s / 2.0 if median_s else 0.0
    drift_frac = max(drift_pct / 100.0, spread_frac)
    # R6-D: date-shifted sampling measures the ANALOG date's weather, not the stated date's --
    # widen the compute band by how much the analog might diverge from what the stated date will
    # actually do (good/fair/poor/no-target/manual -- see build_plan's analog_quality). These
    # bounds are the drift MEASURED on scc (-9%..+6.5%), rounded to +-5/10/15%, not a guess.
    analog_quality = plan.get("analog_quality")
    range_note = None
    if analog_quality:
        analog_drift_pct = {"good": 5.0, "fair": 10.0}.get(analog_quality, 15.0)
        if analog_drift_pct / 100.0 > drift_frac:
            drift_frac = analog_drift_pct / 100.0
        range_note = ("analog_quality=%s widened the compute drift band to +-%.0f%% (measured "
                      "on scc: -9%%..+6.5%%, rounded)" % (analog_quality, analog_drift_pct))
    compute_drift_abs = compute_total * drift_frac
    # I/O estimated but not in TOTAL by default -- same rule as wtune_predict.build_prediction
    io_w = 1.0 if include_io else 0.0
    total_s = compute_total + io_w * io["total_s"] + init["total_s"]
    if io_range is not None:
        low = (max(0.0, compute_total - compute_drift_abs) + io_w * io_range["fast_total_s"]
               + init["total_s"])
        high = compute_total + compute_drift_abs + io_w * io_range["slow_total_s"] + init["total_s"]
    else:
        io_noise = io_w * io.get("noise_abs_s", 0.0)
        band = math.sqrt(compute_drift_abs ** 2 + io_noise ** 2)
        low = max(0.0, total_s - band)
        high = total_s + band

    result = {
        "windows": windows_out, "r_night": r_night, "r_day": r_day, "r_source": r_source,
        "geometry_source": geom_source, "n_radiation_samples": len(pooled_radiation_samples),
        "compute_s": compute_total, "flat_alternative_s": flat_total,
        "io": io, "io_included": include_io, "init": init, "total_s": total_s,
        "range_low_s": low, "range_high_s": high, "drift_frac": drift_frac,
        "steps": steps_n, "analog_quality": analog_quality, "range_note": range_note,
    }
    if json_out:
        with open(json_out, "w") as f:
            json.dump(result, f, indent=2)
    return result


def print_score(result):
    print("windows:")
    for w in result["windows"]:
        forcing_dataset = (w.get("forcing") or {}).get("dataset", "?")
        print("  %-20s %-10s s_k=%.5f n_normal=%d n_rad=%d forcing=%s shift_s=%+.0f"
              % (w["t"], w["reason"], w["s_k"], w["n_normal"], w["n_rad"], forcing_dataset,
                 w.get("shift_s", 0.0)))
    print("r_night=%.4f r_day=%.4f (source=%s, geometry=%s)"
          % (result["r_night"], result["r_day"], result["r_source"], result["geometry_source"]))
    print("compute=%.1fs  flat_alternative=%.1fs  io=%.1fs%s  init=%.1fs"
          % (result["compute_s"], result["flat_alternative_s"], result["io"]["total_s"],
             "" if result.get("io_included", True) else " (not in TOTAL)",
             result["init"]["total_s"]))
    print("TOTAL=%.1fs  range=[%.1f, %.1f]s" % (result["total_s"], result["range_low_s"],
                                                 result["range_high_s"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(prog="coldstart_sample.py", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("inspect")
    sp.add_argument("--case-dir", required=True)
    sp.add_argument("--namelist")
    sp.add_argument("--namelist-wps")
    sp.add_argument("--geo-from-wrfinput")
    sp.add_argument("--grib-dir")
    sp.add_argument("--ncdump")
    sp.add_argument("--geog-path", help="default: $WPS_GEOG or /nfs/data/wrf/WPS_GEOG/")

    sp = sub.add_parser("plan")
    sp.add_argument("--case-dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--k", type=int, default=4)
    sp.add_argument("--nodes", type=int, default=1)
    sp.add_argument("--ranks", type=int, default=72)
    sp.add_argument("--omp", type=int, default=1)
    sp.add_argument("--nx", type=int)
    sp.add_argument("--ny", type=int)
    sp.add_argument("--spinup-min", type=int, default=60)
    sp.add_argument("--state", action="store_true")
    sp.add_argument("--namelist")
    sp.add_argument("--namelist-wps")
    sp.add_argument("--geo-from-wrfinput")
    sp.add_argument("--grib-dir")
    sp.add_argument("--ncdump")
    sp.add_argument("--geog-path", help="default: $WPS_GEOG or /nfs/data/wrf/WPS_GEOG/")
    sp.add_argument("--forcing", help="pin the forcing chain to one dataset (e.g. d083002, "
                    "ncei_g4) instead of resolving it from the run's own start date -- used to "
                    "re-run plan after `fetch` falls back to a coarser-cadence source")
    date_shift_group = sp.add_mutually_exclusive_group()
    date_shift_group.add_argument("--date-shift-to", metavar="YYYY-MM-DD_HH:MM:SS",
                    help="R6-D: place/run windows at this analog date instead of the case's own "
                    "stated start (all times shifted by shift_s = analog - stated)")
    date_shift_group.add_argument("--analog", dest="analog_path", metavar="ANALOG_JSON",
                    help="R6-D: same as --date-shift-to, but reads the analog date + quality "
                    "from a prior `case_rebuild.py analog` run's OUT/analog.json")
    sp.add_argument("--times", metavar="T1,T2,...",
                    help="T1 (2026-09-24): use exactly these window starts (comma-separated "
                    "YYYY-MM-DD_HH:MM:SS, each already on the forcing grid) instead of "
                    "characteristic-point selection; reason='pinned'. --k is ignored.")
    sp.add_argument("--real-only", action="store_true",
                    help="T2 (2026-09-24): each window stops after real.exe succeeds -- no "
                    "wrf.exe, no cost prediction. `score` refuses a --real-only plan.")
    sp.add_argument("--ml-model", metavar="REPORT.json", default=None,
                    help="K5 (2026-09-24): after each window's real.exe, apply "
                    "state_correction_model.py's ML state correction (--targets-from-report) "
                    "to the fresh wrfinput_d01 before wrf.exe runs. Default: off.")

    sp = sub.add_parser("fetch")
    sp.add_argument("--plan", required=True)
    # default: PLAN/grib -- run.sbatch's link_grib.csh step links exactly "$PLAN"/grib/*.grib2
    sp.add_argument("--cache-dir", default=None)
    sp.add_argument("--grib-dir", help="local GRIB directory fallback (no online source covers "
                    "the date at all)")
    sp.add_argument("--keep-archives", action="store_true",
                    help="keep CFSR tar files after extracting instead of deleting them")

    sp = sub.add_parser("stage")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--site", default="scc")
    # also the wrf.exe per-window `timeout` (seconds = 60x): maria1km (job 982, 1 km d02, 202
    # d01 steps at ~9.3 s/step) was killed at exactly 899 s by the old fixed 15 min
    sp.add_argument("--per-window-min", type=int, default=15)

    sp = sub.add_parser("score")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--json")
    sp.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    sp.add_argument("--include-io", action="store_true",
                    help="add the history I/O estimate to TOTAL (off by default)")

    sp = sub.add_parser("fix-wps-namelist")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--win", type=int, required=True)

    sp = sub.add_parser("fix-real-namelist")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--win", type=int, required=True)
    sp.add_argument("--ncdump")
    sp.add_argument("--serial", action="store_true",
                    help="force nproc_x=nproc_y=1 (real.exe hang-at-some-decompositions "
                    "fallback -- run.sbatch's own retry path)")

    sp = sub.add_parser("fix-wrf-namelist")
    sp.add_argument("--plan", required=True)
    sp.add_argument("--win", type=int, required=True)
    sp.add_argument("--nproc-x", type=int, required=True)
    sp.add_argument("--nproc-y", type=int, required=True)
    sp.add_argument("--numtiles", type=int, default=1)
    sp.add_argument("--ncdump")

    args = ap.parse_args(argv)

    try:
        if args.cmd == "inspect":
            rows, blockers, _nl = inspect_case(
                args.case_dir, namelist=args.namelist, namelist_wps=args.namelist_wps,
                geo_from_wrfinput=args.geo_from_wrfinput, grib_dir=args.grib_dir,
                ncdump=args.ncdump, geog_path=args.geog_path)
            print_inspect(rows, blockers)
            return 3 if blockers else 0

        if args.cmd == "plan":
            rows, blockers, _nl = inspect_case(
                args.case_dir, namelist=args.namelist, namelist_wps=args.namelist_wps,
                geo_from_wrfinput=args.geo_from_wrfinput, grib_dir=args.grib_dir,
                ncdump=args.ncdump, geog_path=args.geog_path)
            if blockers:
                print_inspect(rows, blockers)
                return 3
            times = None
            if args.times:
                times = [datetime.datetime.strptime(t.strip(), _WRF_TIME_FMT)
                         for t in args.times.split(",") if t.strip()]
            plan = build_plan(args.case_dir, args.out, k=args.k, nodes=args.nodes,
                              ranks=args.ranks, omp=args.omp, nx=args.nx, ny=args.ny,
                              spinup_min=args.spinup_min, namelist=args.namelist,
                              namelist_wps=args.namelist_wps,
                              geo_from_wrfinput=args.geo_from_wrfinput, grib_dir=args.grib_dir,
                              ncdump=args.ncdump, use_state=args.state, geog_path=args.geog_path,
                              forcing=args.forcing, date_shift_to=args.date_shift_to,
                              analog_path=args.analog_path, times=times,
                              real_only=args.real_only, ml_model=args.ml_model)
            print_plan(plan)
            return 0

        if args.cmd == "fetch":
            with open(os.path.join(args.plan, "plan.json")) as f:
                plan = json.load(f)
            cache_dir = args.cache_dir or os.path.join(args.plan, "grib")
            resolve_and_fetch_forcing(plan, args.plan, cache_dir, grib_dir=args.grib_dir,
                                      keep_archives=args.keep_archives)
            return 0

        if args.cmd == "stage":
            stage_plan(args.plan, site=args.site, per_window_min=args.per_window_min)
            return 0

        if args.cmd == "score":
            result = score_plan(args.plan, calibration_path=args.calibration, json_out=args.json,
                                include_io=args.include_io)
            print_score(result)
            return 0

        if args.cmd == "fix-wps-namelist":
            cmd_fix_wps_namelist(args.plan, args.win)
            return 0

        if args.cmd == "fix-real-namelist":
            cmd_fix_real_namelist(args.plan, args.win, ncdump=args.ncdump, serial=args.serial)
            return 0

        if args.cmd == "fix-wrf-namelist":
            cmd_fix_wrf_namelist(args.plan, args.win, args.nproc_x, args.nproc_y, args.numtiles,
                                  ncdump=args.ncdump)
            return 0
    except ColdstartError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

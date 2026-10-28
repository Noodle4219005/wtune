#!/usr/bin/env python3
"""analysis/tc_track.py -- TC detection/tracking via TempestExtremes (DetectNodes +
StitchNodes), replacing predictor/verify_obs.py storm_centres()'s per-frame global
argmin(SLP) + domain-max(wind) with a real tracker that has closed-contour criteria and
track continuity.

TempestExtremes CLI facts this module depends on (verified against tempest-extremes
2.4.2, conda-forge nompi build 103 -- `DetectNodes`/`StitchNodes` have no --help/--version
flag; passing an unrecognized flag dumps the usage list, which is how these were found):

  * Boolean flags (`--regional`, `--out_header`, ...) are PRESENCE flags: `--regional`
    means true, omitting it means false. Passing `--regional true` is a parse ERROR
    ("Invalid argument \"true\""), not a way to set it false. Never pass a boolean value.
  * A regular WRF domain is a LIMITED-AREA grid, so `--regional` is always passed -- a
    global RLL grid check trips on domains with (maxlon - minlon) > ~180 deg otherwise
    ("Grid element longitudinal extent too large").
  * A separable grid (1-D lat/lon coordinate vectors) is read directly by DetectNodes
    with --latname/--lonname -- no connectivity file needed.
  * A curvilinear/non-separable grid (2-D lat/lon, e.g. Lambert conformal) is flattened
    row-major onto `ncol` (analysis.diagnostics) and read with `--in_connect`, a connectivity
    file of the (ny, nx) index grid's edge neighbours (write_grid_connectivity). Passing the
    2-D (y, x) field shape to `--in_connect` fails with "Variable ... has inconsistent
    auxiliary dimensions (3) than specified on command line (2)". StitchNodes ALSO needs
    `--in_connect` for this grid type, or it rejects DetectNodes' 1-column (ncol index)
    node file ("first 2 columns are not integer indices ...").
    `GenerateConnectivityFile --in_data` does NOT triangulate a point cloud (2.4.2): it
    writes every node with 0 neighbours, DetectNodes then treats every cell as a local
    minimum (mergedist leaves the deepest one), and `wspd10,max,R` reads the centre cell
    only -- haitang2017 d03 vmax 1.9 m/s against 21.8 m/s within 2 deg.

DetectNodes parameters (TCTrackParams below) and their justification:
  --searchbymin slp                 storm centre = local SLP minimum (not global argmin;
                                     DetectNodes finds ALL local minima meeting the
                                     criteria below, StitchNodes then picks the
                                     continuous one).
  --closedcontourcmd "slp,DELTA,DIST,0"
                                     candidate must be enclosed by a DELTA hPa SLP rise
                                     within DIST degrees (published TC-detector default:
                                     2 hPa / 5.5 deg, e.g. Ullrich & Zarzycki 2017's
                                     "PSL,200.0,5.5,0" in Pa == 2.0 hPa here since this
                                     module's fields are in hPa) -- this is what rejects a
                                     shallow/noise minimum that a plain argmin would take.
  --mergedist 6.0                   candidates within 6 deg of each other at the same time
                                     are merged into one (avoids double-counting a broad,
                                     multi-cell minimum as two storms).
  --outputcmd "slp,min,0;wspd10,max,WIND_RADIUS"
                                     per node: minimum SLP AT the node (dist 0) and the
                                     MAXIMUM 10 m wind within WIND_RADIUS degrees (default
                                     2.0 deg =~ 220 km great-circle) of the node -- an
                                     eyewall/near-core wind, not predictor/verify_obs.py's
                                     domain-wide max which can pick up an unrelated gust
                                     anywhere on the grid.

StitchNodes parameters and their justification:
  --range STITCH_RANGE (default 8.0 deg)
                                     a node may join a track only if within this distance
                                     of the track's position at the previous time -- this
                                     is what gives track CONTINUITY that argmin(SLP) has
                                     none of (see the module docstring test: a deeper but
                                     spurious/disconnected low does not hijack the track).
  --mintime MINTIME_H (default 24h) reject candidate tracks shorter than this -- drops
                                     one-off noise detections that never persisted.
  --maxgap MAXGAP_H (default 24h)   allow the track to miss up to this many hours of
                                     detections (e.g. a frame where the closed-contour
                                     test failed) without breaking the track.
  --in_fmt "lon,lat,slp,wind"       must list the outputcmd fields in the same order they
                                     were written, after the lon/lat DetectNodes always
                                     emits first.

Every TempestExtremes command actually run is recorded verbatim (argv list) in the
provenance returned by detect_and_stitch, per the task's provenance requirement.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reused per the task's own instruction: parse_hurdat2 (HURDAT2 best-track parser) and
# great_circle_km (haversine distance) are predictor/verify_obs.py's, not reimplemented.
from wtune.predictor import verify_obs  # noqa: E402

WRF_TIME_FMT = verify_obs.WRF_TIME_FMT
great_circle_km = verify_obs.great_circle_km
parse_hurdat2 = verify_obs.parse_hurdat2


class TCTrackError(RuntimeError):
    """Raised when TempestExtremes itself fails or its output cannot be parsed."""


@dataclass
class TCTrackParams:
    """DetectNodes/StitchNodes parameters. See module docstring for justification."""
    closedcontour_delta_hpa: float = 2.0     # SLP rise (hPa) required within closedcontour_dist_deg
    closedcontour_dist_deg: float = 5.5      # search radius (deg) for the closed-contour test
    mergedist_deg: float = 6.0               # merge candidates within this distance (deg)
    wind_radius_deg: float = 2.0             # radius (deg, ~220 km) for near-core max wind
    stitch_range_deg: float = 8.0            # max deg a track may move between 6-hourly detections
    stitch_range_min_deg: float = 2.0        # floor after scaling to the output interval
    mintime_hours: float = 24.0              # minimum track lifetime to keep
    maxgap_hours: float = 24.0               # maximum allowed gap in a track
    regional: bool = True                    # WRF output is always a limited-area domain


def _find_binary(name):
    """PATH, then the repo's pixi env bin and postproc view bin (post.tools.find_on_path), so
    TempestExtremes is found without bin/wtune putting the pixi env on PATH."""
    from wtune.post import tools
    try:
        return tools.find_on_path(name)
    except tools.ToolNotFoundError as e:
        raise TCTrackError(
            f"{e} Install tempest-extremes (envs/wtune/pixi.toml) "
            f"and run inside its environment, e.g. "
            f"`pixi run --manifest-path envs/wtune/pixi.toml {name}`.") from None


def _tempest_extremes_version(binary_path):
    """Best-effort version string from the conda package metadata (no --version flag
    exists on this build; passing one is a parse error, not a query)."""
    prefix = os.path.dirname(os.path.dirname(binary_path))
    meta_dir = os.path.join(prefix, "conda-meta")
    if os.path.isdir(meta_dir):
        for name in os.listdir(meta_dir):
            if name.startswith("tempest-extremes-") and name.endswith(".json"):
                return name[len("tempest-extremes-"):-len(".json")]
    return "unknown"


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _detect_workers(n_times):
    """How many DetectNodes to run at once: one per CPU this process may use, at most one
    per time step (the conda-forge build has no MPI; one process took 56 min on dorian)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(cpus, n_times))


def _split_by_time(fields_nc, workdir, n):
    """fields_nc cut into n files of consecutive time steps -> their paths, in time order."""
    netcdf4 = __import__("netCDF4")
    paths = []
    with netcdf4.Dataset(fields_nc) as src:
        bounds = np.linspace(0, len(src.dimensions["time"]), n + 1).astype(int)
        for k in range(n):
            a, b = int(bounds[k]), int(bounds[k + 1])
            path = os.path.join(workdir, "fields_part%03d.nc" % k)
            with netcdf4.Dataset(path, "w") as dst:
                dst.setncatts({k_: src.getncattr(k_) for k_ in src.ncattrs()})
                for name, dim in src.dimensions.items():
                    dst.createDimension(name, (b - a) if name == "time" else len(dim))
                for name, var in src.variables.items():
                    attrs = {k_: var.getncattr(k_) for k_ in var.ncattrs()}
                    out = dst.createVariable(name, var.dtype, var.dimensions,
                                             fill_value=attrs.pop("_FillValue", None))
                    out.setncatts(attrs)
                    if "time" in var.dimensions:
                        index = tuple(slice(a, b) if d == "time" else slice(None)
                                      for d in var.dimensions)
                        out[:] = var[index]
                    else:
                        out[:] = var[:]
            paths.append(path)
    return paths


def _detect_in_parts(detect_args, fields_nc, workdir, nodes_path, n):
    """DetectNodes on n time slices at once, merged into nodes_path. DetectNodes treats each
    time step on its own, so the merged file is the one a single run would write.
    Returns the first slice's command (provenance)."""
    parts = _split_by_time(fields_nc, workdir, n)
    cmds, procs = [], []
    for p in parts:
        stem = p[:-len(".nc")]
        cmd = [detect_args[0], "--in_data", p, "--out", stem + ".nodes"] + detect_args[1:]
        with open(stem + ".log", "w") as log:
            procs.append(subprocess.Popen(cmd, cwd=workdir, stdout=log,
                                          stderr=subprocess.STDOUT, text=True))
        cmds.append(cmd)
    codes = [p.wait() for p in procs]
    for cmd, rc in zip(cmds, codes):
        if rc != 0:
            with open(cmd[4][:-len(".nodes")] + ".log") as f:
                raise TCTrackError(f"DetectNodes failed (exit {rc}):\n{f.read()[-2000:]}\n"
                                   f"cmd: {cmd}")
    with open(nodes_path, "w") as out:
        for k, cmd in enumerate(cmds):
            with open(cmd[4]) as f:
                out.writelines(line for line in f if k == 0 or not line.startswith("#"))
    for p in parts:
        for path in (p, p[:-len(".nc")] + ".nodes", p[:-len(".nc")] + ".log"):
            os.remove(path)
    return cmds[0]


def _hours_spec(hours):
    """TempestExtremes' Time-string parser rejects a decimal point ("12.0h": "Cannot mix
    Free and Time format") -- it wants a bare integer count. Convert to whole hours when
    possible, else whole minutes."""
    if float(hours).is_integer():
        return f"{int(hours)}h"
    return f"{int(round(hours * 60))}m"


def _output_interval_hours(ds):
    """Median spacing of the fields file's time axis in hours, or None (one frame / no units)."""
    if "time" not in ds.variables or len(ds.variables["time"]) < 2:
        return None
    tvar = ds.variables["time"]
    unit = getattr(tvar, "units", "").split(" since ")[0].strip().lower()
    per_hour = {"seconds": 3600.0, "minutes": 60.0, "hours": 1.0, "days": 1.0 / 24.0}.get(unit)
    if per_hour is None:
        return None
    import numpy as np
    return float(np.median(np.diff(np.asarray(tvar[:], dtype=float)))) / per_hour


def stitch_range_for_interval(params, interval_h):
    """StitchNodes --range for this output interval. stitch_range_deg is the 6-hourly
    convention; at hourly output a fixed 8 deg lets a second low up to 8 deg away claim the
    first detection (SC26-P2: a 1007 hPa low 653 km from Katrina took t0, the real centre was
    orphaned). Scale linearly with the interval, floored at stitch_range_min_deg, never above
    stitch_range_deg."""
    if not interval_h or interval_h <= 0:
        return params.stitch_range_deg
    scaled = params.stitch_range_deg * interval_h / 6.0
    return min(params.stitch_range_deg, max(params.stitch_range_min_deg, scaled))


def write_grid_connectivity(fields_nc, out_path):
    """Edge-neighbour (4-point) connectivity of the (ny, nx) WRF grid that analysis.diagnostics
    flattened row-major onto `ncol`, in TempestExtremes' connectivity-file format (1-based
    neighbour indices, as `GenerateConnectivityFile --in_mesh` writes them for an RLL mesh).
    `GenerateConnectivityFile --in_data` is not used: on a point cloud it writes every node
    with 0 neighbours, so DetectNodes takes every cell as a local minimum and `wspd10,max,R`
    returns the centre cell alone. Area is written as 1 (DetectNodes' min/max/closed-contour
    operators do not read it)."""
    import numpy as np
    netcdf4 = __import__("netCDF4")
    with netcdf4.Dataset(fields_nc) as ds:
        ny, nx = int(ds.ny), int(ds.nx)
        lat = np.asarray(ds.variables["lat"][:], dtype=float)
        lon = np.asarray(ds.variables["lon"][:], dtype=float)
    if lat.size != ny * nx:
        raise TCTrackError(f"{fields_nc}: ncol {lat.size} != ny*nx {ny}*{nx}")
    with open(out_path, "w") as f:
        f.write("#TempestGridConnectivityFileV2.0\n1,%d\n" % (ny * nx))
        for j in range(ny):
            for i in range(nx):
                k = j * nx + i
                nb = [k - nx] * (j > 0) + [k - 1] * (i > 0) + [k + 1] * (i < nx - 1) \
                    + [k + nx] * (j < ny - 1)
                f.write("%.14e,%.14e,1.0,%d,%s\n"
                        % (lon[k], lat[k], len(nb), ",".join(str(n + 1) for n in nb)))
    return out_path


def detect_and_stitch(fields_nc, workdir, params=None):
    """Run TempestExtremes DetectNodes + StitchNodes on a fields_nc written by
    analysis.diagnostics.wrfout_to_tc_fields.

    Returns (tracks, provenance):
      tracks:     list of tracks; each track is a time-sorted list of dicts
                  {time (datetime), lat, lon, pmin_hpa, vmax_ms}.
      provenance: {"tool": "TempestExtremes", "version": ..., "detect_cmd": [...],
                   "stitch_cmd": [...], "connectivity_cmd": [...] or None,
                   "nodes_file": ..., "track_file": ..., "params": {...}}.

    A non-separable (`ncol`) grid gets its connectivity from write_grid_connectivity.
    """
    params = params or TCTrackParams()
    workdir = os.path.abspath(str(workdir))     # _run() executes with cwd=workdir: relative paths would double
    os.makedirs(workdir, exist_ok=True)
    fields_nc = os.path.abspath(str(fields_nc))

    netcdf4 = __import__("netCDF4")
    with netcdf4.Dataset(fields_nc) as ds:
        grid_type = getattr(ds, "grid_type", "separable")
        map_proj_name = getattr(ds, "map_proj_name", "unknown")
        interval_h = _output_interval_hours(ds)
        n_times = len(ds.dimensions["time"])
    stitch_range = stitch_range_for_interval(params, interval_h)

    detectnodes = _find_binary("DetectNodes")
    stitchnodes = _find_binary("StitchNodes")
    version = _tempest_extremes_version(detectnodes)

    connect_cmd = None
    in_connect_args = []
    if grid_type == "ncol":
        connect_path = os.path.join(workdir, "connect.dat")
        write_grid_connectivity(fields_nc, connect_path)
        connect_cmd = ["write_grid_connectivity", fields_nc, connect_path]
        in_connect_args = ["--in_connect", connect_path]

    nodes_path = os.path.join(workdir, "nodes.dat")
    closedcontourcmd = (f"slp,{params.closedcontour_delta_hpa},"
                        f"{params.closedcontour_dist_deg},0")
    outputcmd = f"slp,min,0;wspd10,max,{params.wind_radius_deg}"
    detect_args = [detectnodes,
                   "--searchbymin", "slp", "--closedcontourcmd", closedcontourcmd,
                   "--mergedist", str(params.mergedist_deg), "--outputcmd", outputcmd,
                   "--latname", "lat", "--lonname", "lon", "--out_header"] + in_connect_args
    if params.regional:
        detect_args.append("--regional")
    workers = _detect_workers(n_times)
    if workers > 1:
        detect_cmd = _detect_in_parts(detect_args, fields_nc, workdir, nodes_path, workers)
    else:
        detect_cmd = [detectnodes, "--in_data", fields_nc, "--out", nodes_path] + detect_args[1:]
        rc, out, err = _run(detect_cmd, workdir)
        if rc != 0:
            raise TCTrackError(f"DetectNodes failed (exit {rc}):\n{err}\ncmd: {detect_cmd}")

    track_path = os.path.join(workdir, "track.csv")
    stitch_cmd = [stitchnodes, "--in", nodes_path, "--out", track_path,
                 "--in_fmt", "lon,lat,slp,wind",
                 "--range", "%g" % stitch_range,
                 "--mintime", _hours_spec(params.mintime_hours),
                 "--maxgap", _hours_spec(params.maxgap_hours),
                 "--out_file_format", "csv", "--caltype", "standard"] + in_connect_args
    rc, out, err = _run(stitch_cmd, workdir)
    if rc != 0:
        raise TCTrackError(f"StitchNodes failed (exit {rc}):\n{err}\ncmd: {stitch_cmd}")

    tracks = _parse_stitchnodes_csv(track_path)

    provenance = {
        "tool": "TempestExtremes",
        "version": version,
        "connectivity_cmd": connect_cmd,
        "detect_cmd": detect_cmd,
        "detect_parts": workers,
        "stitch_cmd": stitch_cmd,
        "nodes_file": nodes_path,
        "track_file": track_path,
        "grid_type": grid_type,
        "map_proj_name": map_proj_name,
        "params": asdict(params),
        "output_interval_hours": interval_h,
        "stitch_range_effective_deg": stitch_range,
    }
    return tracks, provenance


def _parse_stitchnodes_csv(track_path):
    """StitchNodes --out_file_format csv output -> list of tracks (time-sorted dicts)."""
    if not os.path.exists(track_path):
        raise TCTrackError(f"StitchNodes did not write {track_path}")
    with open(track_path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    idx = {name: k for k, name in enumerate(header)}
    required = ("track_id", "year", "month", "day", "hour", "lon", "lat", "slp", "wind")
    missing = [r for r in required if r not in idx]
    if missing:
        raise TCTrackError(
            f"{track_path}: expected columns {required}, got {header} (missing {missing})")

    by_track = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        row = [c.strip() for c in row]
        tid = row[idx["track_id"]]
        when = datetime(int(row[idx["year"]]), int(row[idx["month"]]), int(row[idx["day"]]),
                        int(float(row[idx["hour"]])))
        by_track.setdefault(tid, []).append({
            "time": when,
            "lat": float(row[idx["lat"]]),
            "lon": float(row[idx["lon"]]),
            "pmin_hpa": float(row[idx["slp"]]),
            "vmax_ms": float(row[idx["wind"]]),
        })
    return [sorted(pts, key=lambda p: p["time"]) for pts in by_track.values()]


def select_track(tracks, first_guess_lat, first_guess_lon, t0=None):
    """Pick the stitched track nearest a first-guess position.

    t0: a datetime to compare at (defaults to each track's own first point). Distance is
    great-circle km from (first_guess_lat, first_guess_lon) to the track's point closest
    in time to t0 (or its first point if t0 is None).
    Returns the chosen track (list of dicts), or None if `tracks` is empty.
    """
    if not tracks:
        return None
    best_track, best_dist = None, None
    for track in tracks:
        if not track:
            continue
        if t0 is None:
            pt = track[0]
        else:
            pt = min(track, key=lambda p: abs((p["time"] - t0).total_seconds()))
        dist = float(great_circle_km(first_guess_lat, first_guess_lon, pt["lat"], pt["lon"]))
        if best_dist is None or dist < best_dist:
            best_track, best_dist = track, dist
    return best_track


def compare_best_track(track, besttrack_rows, *, wrfout_paths=None, ocean_only=True,
                       synoptic_only=True, time_tol_s=60.0):
    """Per best-track time: tracker error vs. the naive method vs. HURDAT2, side by side.

    track:           one stitched track (list of {time, lat, lon, pmin_hpa, vmax_ms}),
                     e.g. from select_track().
    besttrack_rows:  parse_hurdat2() output for the storm.
    wrfout_paths:    when given, also computes predictor/verify_obs.py's OLD naive method
                     (storm_centres(): global ocean argmin SLP + domain-max wind) on the
                     SAME wrfout, so the difference between methods is quantified on
                     identical frames. When None, the naive_* columns are omitted.

    Returns {"rows": [...], "tracker": {agg stats}, "naive": {agg stats} or None,
             "vmax_caveat": "..."} .
    Each row: {time, lead_h, obs_lat, obs_lon,
               tracker_lat, tracker_lon, tracker_track_error_km,
               tracker_pmin_hpa, tracker_pmin_error_hpa,
               tracker_vmax_ms, tracker_vmax_error_ms,
               naive_lat, naive_lon, naive_track_error_km,
               naive_pmin_hpa, naive_pmin_error_hpa,
               naive_vmax_ms, naive_vmax_error_ms}  (naive_* is None without wrfout_paths).
    """
    from datetime import timedelta

    if not track:
        raise TCTrackError("compare_best_track: empty track")
    track = sorted(track, key=lambda p: p["time"])
    track_times = [p["time"] for p in track]
    lo, hi = track_times[0], track_times[-1]
    tol = timedelta(seconds=time_tol_s)

    candidates = [p for p in besttrack_rows if lo - tol <= p["time"] <= hi + tol
                 and (not synoptic_only or (p["time"].minute == 0 and p["time"].hour % 6 == 0))]

    naive_centres = None
    if wrfout_paths is not None:
        naive_centres = verify_obs.storm_centres(wrfout_paths, ocean_only=ocean_only)
        naive_by_time = {c["time"]: c for c in naive_centres}
    else:
        naive_by_time = {}

    def _nearest(points, when):
        if not points:
            return None
        p = min(points, key=lambda q: abs((q["time"] - when).total_seconds()))
        return p if abs((p["time"] - when).total_seconds()) <= time_tol_s else None

    rows = []
    for obs in candidates:
        trk_pt = _nearest(track, obs["time"])
        if trk_pt is None:
            continue
        row = {
            "time": obs["time"].strftime(WRF_TIME_FMT),
            "lead_h": (obs["time"] - lo).total_seconds() / 3600.0,
            "obs_lat": obs["lat"], "obs_lon": obs["lon"],
            "obs_pmin_hpa": obs["pmin_hpa"], "obs_vmax_ms": obs["vmax_ms"],
            "tracker_lat": trk_pt["lat"], "tracker_lon": trk_pt["lon"],
            "tracker_track_error_km": float(great_circle_km(
                obs["lat"], obs["lon"], trk_pt["lat"], trk_pt["lon"])),
            "tracker_pmin_hpa": trk_pt["pmin_hpa"],
            "tracker_pmin_error_hpa": (None if obs["pmin_hpa"] is None
                                       else trk_pt["pmin_hpa"] - obs["pmin_hpa"]),
            "tracker_vmax_ms": trk_pt["vmax_ms"],
            "tracker_vmax_error_ms": (None if obs["vmax_ms"] is None or trk_pt["vmax_ms"] is None
                                      else trk_pt["vmax_ms"] - obs["vmax_ms"]),
        }
        if naive_by_time:
            nv_pt = naive_by_time.get(min(naive_by_time, key=lambda t: abs(
                (t - obs["time"]).total_seconds()))) if naive_by_time else None
            if nv_pt is not None and abs((nv_pt["time"] - obs["time"]).total_seconds()) > time_tol_s:
                nv_pt = None
            row.update({
                "naive_lat": None if nv_pt is None else nv_pt["lat"],
                "naive_lon": None if nv_pt is None else nv_pt["lon"],
                "naive_track_error_km": None if nv_pt is None else float(great_circle_km(
                    obs["lat"], obs["lon"], nv_pt["lat"], nv_pt["lon"])),
                "naive_pmin_hpa": None if nv_pt is None else nv_pt["pmin_hpa"],
                "naive_pmin_error_hpa": (None if nv_pt is None or obs["pmin_hpa"] is None
                                        else nv_pt["pmin_hpa"] - obs["pmin_hpa"]),
                "naive_vmax_ms": None if nv_pt is None else nv_pt["vmax_ms"],
                "naive_vmax_error_ms": (None if nv_pt is None or obs["vmax_ms"] is None
                                       or nv_pt["vmax_ms"] is None
                                       else nv_pt["vmax_ms"] - obs["vmax_ms"]),
            })
        else:
            row.update({k: None for k in (
                "naive_lat", "naive_lon", "naive_track_error_km", "naive_pmin_hpa",
                "naive_pmin_error_hpa", "naive_vmax_ms", "naive_vmax_error_ms")})
        rows.append(row)
    rows.sort(key=lambda r: r["lead_h"])

    def _agg(prefix, key):
        vals = [r[f"{prefix}_{key}"] for r in rows if r.get(f"{prefix}_{key}") is not None]
        if not vals:
            return {"n": 0, "mean": None, "abs_mean": None, "max_abs": None, "rmse": None}
        arr = np.asarray(vals, dtype=float)
        return {"n": int(arr.size), "mean": float(arr.mean()),
                "abs_mean": float(np.abs(arr).mean()), "max_abs": float(np.abs(arr).max()),
                "rmse": float(np.sqrt(np.mean(arr ** 2)))}

    def _deepening(pts, key):
        vals = [p[key] for p in pts if p.get(key) is not None] if pts else []
        return None if len(vals) < 2 else vals[0] - vals[-1]

    obs_deepening = _deepening(sorted(candidates, key=lambda p: p["time"]), "pmin_hpa")
    tracker_deepening = _deepening(track, "pmin_hpa")
    naive_deepening = (_deepening(sorted(naive_centres, key=lambda p: p["time"]), "pmin_hpa")
                       if naive_centres else None)

    result = {
        "rows": rows,
        "matched": len(rows),
        "best_track_points_in_window": len(candidates),
        "tracker": {
            "track_error_km": _agg("tracker", "track_error_km"),
            "pmin_error_hpa": _agg("tracker", "pmin_error_hpa"),
            "vmax_error_ms": _agg("tracker", "vmax_error_ms"),
            "deepening_hpa": tracker_deepening,
        },
        "naive": None if not naive_centres else {
            "track_error_km": _agg("naive", "track_error_km"),
            "pmin_error_hpa": _agg("naive", "pmin_error_hpa"),
            "vmax_error_ms": _agg("naive", "vmax_error_ms"),
            "deepening_hpa": naive_deepening,
        },
        "obs_deepening_hpa": obs_deepening,
        "vmax_caveat": (
            "vmax is max 10 m wind within wind_radius_deg of the tracked centre, "
            "instantaneous hourly model output vs. HURDAT2's 1-minute sustained wind -- "
            "not identical quantities; do not read a wind error here as a pure model bias."
        ),
    }
    return result

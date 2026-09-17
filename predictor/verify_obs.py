#!/usr/bin/env python3
"""predictor/verify_obs.py -- score a WRF run against REAL OBSERVATIONS.

Companion to fidelity_gate.py, NOT a replacement for it. The two answer different
questions and must both exist:

  fidelity_gate.py  CONSISTENCY. "Did this config move the solution away from a
                    reference RUN, by more than merely re-decomposing moves it?"
                    The judge is another model run. Every case can be judged this way.

  verify_obs.py     ACCURACY against reality. "Is this config closer to the real
                    atmosphere?" The judge is observation data. Only the cases that
                    have real observations can be judged this way -- currently the two
                    SC26 practice problems.

Why the distinction matters: the problem statement's own settings were never optimised,
so a FASTER config can legitimately be MORE accurate here. fidelity_gate would score
that as drift away from the reference; verify_obs scores it as an improvement. Never
substitute one for the other.

Two modes:

  station  SC26-P1. The problem statement specifies the metric itself: RMSE and mean
           bias of 2 m temperature against hourly observations at South Bend airport
           (41.7072 N, -86.3163 E), 2019-01-28 00Z -> 01-31 23Z, 96 points, degrees C.
           Nearest grid point, K -> C, exact time alignment.

  track    SC26-P2. Ground truth is the NHC HURDAT2 best track (AL122005, Katrina),
           6-hourly. Storm centre per wrfout frame = minimum sea-level pressure;
           reports track error (km, great-circle), central-pressure error (hPa) and
           max-wind error (m/s), per lead time and as a mean.

Self-containment: this module reads observation files FROM DISK and never touches the
network. The HURDAT2 file must be fetched once (see --hurdat2 below); when it is absent
the tool says exactly which file is missing and where it must go, rather than reaching
for the network mid-run. The sea-level-pressure formula is vendored in
predictor/wrf_slp_vendored.py (provenance header there), not imported from another repo.

CLI:
    python3 predictor/verify_obs.py station \\
        --wrfout 'run/wrfout_d02_*' --obs obs.csv \\
        --lat 41.7072 --lon -86.3163 [--obs-time-column TIME] [--obs-value-column T2]
        [--obs-units C|K|F] [--min-match-frac 0.9] [--json out.json]

    python3 predictor/verify_obs.py track \\
        --wrfout 'run/wrfout_d01_*' \\
        --hurdat2 /home/u5727520/wrf-bench/atlas/sc26p2/hurdat2-1851-2024-040425.txt \\
        --storm-id AL122005 [--ocean-only] [--min-match-frac 0.9] [--json out.json]

    python3 predictor/verify_obs.py case --cases campaign/cases.json --case sc26p2 \\
        --wrfout 'run/wrfout_d01_*' [--json out.json]

As a library (this is how predictor/recommend.py's accuracy objective uses it):

    import verify_obs
    spec = verify_obs.observations_spec(case_entry)   # None when the case declares none
    if spec is not None:
        res = verify_obs.verify(spec, wrfout_paths)
        err = verify_obs.accuracy_error(res)          # LOWER IS BETTER, always

`accuracy_error` is deliberately a single scalar in the metric the problem statement
itself scores (RMSE in C for station, mean track error in km for track), so the caller
does not have to know which mode ran. When the case declares no observations,
`observations_spec` returns None and the caller falls back to model-vs-model fidelity.

Caveats that belong in any report built on this tool:

  * The model's "max 10 m wind" is a DOMAIN MAXIMUM on a coarse grid; HURDAT2's vmax is
    a 1-minute sustained wind near the centre. They are not the same quantity and a
    12 km grid necessarily underestimates. A large negative wind error is not by itself
    a model error. The number is reported because the problem asks for intensity, not
    because it is a clean verification metric.
  * Central-pressure bias is dominated by the INITIAL CONDITION for a case like Katrina
    (neither driving dataset carries a 959 hPa vortex). The discriminating quantities
    are the deepening RATE and the track, both of which this tool reports.
  * Minimum SLP must not be taken as min(PSFC): PSFC carries terrain, so high ground
    outscores the storm centre. Use the sea-level reduction, and (--ocean-only) search
    only where LANDMASK == 0.
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import math
import os
import sys
from datetime import datetime, timedelta

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_CAMPAIGN_DIR = os.path.join(_REPO_ROOT, "campaign")
for _p in (_HERE, _REPO_ROOT, _CAMPAIGN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wrf_slp_vendored               # predictor/wrf_slp_vendored.py: slp_from_frame()


def _netcdf4():
    """netCDF4 is imported lazily, on first wrfout access.

    Deliberate: the pixi env that runs `pytest campaign predictor -q`
    (wtune-predictor-env) does NOT ship netCDF4 -- only wtune-pixi-env-20260830 does.
    Importing it at module scope would make `import verify_obs` fail there, which would
    take recommend.py down with it (recommend.py imports this module) and break the
    whole predictor suite. Everything except the two wrfout readers works without it, so
    the parsers, the metrics and the refusal logic stay testable in the predictor env.
    """
    try:
        import netCDF4
    except ImportError as exc:                                  # pragma: no cover
        raise VerifyObsError(
            "netCDF4 is required to read wrfout files but is not importable. Run this "
            "under the WRF pixi env, e.g.\n"
            "  pixi run --manifest-path "
            "/home/u5727520/wrf-stack/wtune-pixi-env-20260830/pixi.toml python "
            "predictor/verify_obs.py ...") from exc
    return netCDF4

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
KT_TO_MS = 0.514444
EARTH_RADIUS_KM = 6371.0088          # IUGG mean radius
HURDAT2_MISSING = -999

OBSERVATION_KINDS = ("station", "track")

# The single number `accuracy_error()` returns per kind: the metric the SC26 problem
# statement itself scores. LOWER IS BETTER for both.
PRIMARY_METRIC = {"station": "rmse_c", "track": "track_error_km_mean"}

# Where a missing observation file is expected to live, so the error message can say it.
EXPECTED_OBS_HINT = {
    "station": (
        "SC26-P1's observation CSV was a mail attachment "
        "(south_bend_temperature_Jan28_31_2019_observed.csv, from Laura Huber's "
        "2026-09-10 SC26 practice-task mail). Do NOT substitute another observation "
        "source -- the problem statement scores against that file."
    ),
    "track": (
        "HURDAT2 best track. Fetch once on a login node:\n"
        "  curl -sSLo /home/u5727520/wrf-bench/atlas/sc26p2/"
        "hurdat2-1851-2024-040425.txt \\\n"
        "    https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
    ),
}


class VerifyObsError(RuntimeError):
    """Base class: verify_obs refuses rather than silently degrading."""


class MissingObservationFile(VerifyObsError):
    pass


class InsufficientMatches(VerifyObsError):
    """Too few observation times matched a model frame. Never silently dropped."""


# --------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------

def great_circle_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (haversine, spherical Earth, R = 6371.0088 km).

    Scalars or numpy arrays. Haversine rather than the spherical law of cosines because
    the latter loses all precision for the short separations that matter here (a 12 km
    grid cell is 1e-3 rad).
    """
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_index(lat2d, lon2d, lat, lon):
    """Nearest grid point to (lat, lon). Returns (j, i, grid_lat, grid_lon, dist_km).

    Selection uses a cos(lat)-weighted planar distance (cheap, and exact enough to pick a
    cell); the distance REPORTED back is the true great-circle distance, so the caller can
    see how far the chosen cell really is from the station.
    """
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)
    d = (lat2d - lat) ** 2 + ((lon2d - lon) * math.cos(math.radians(lat))) ** 2
    j, i = np.unravel_index(int(np.argmin(d)), d.shape)
    glat, glon = float(lat2d[j, i]), float(lon2d[j, i])
    return int(j), int(i), glat, glon, float(great_circle_km(lat, lon, glat, glon))


# --------------------------------------------------------------------------------------
# HURDAT2
# --------------------------------------------------------------------------------------

def _hurdat2_coord(token):
    """'23.1N' / '75.1W' -> signed degrees."""
    token = token.strip()
    hemi, value = token[-1].upper(), float(token[:-1])
    if hemi in ("S", "W"):
        value = -value
    elif hemi not in ("N", "E"):
        raise VerifyObsError(f"HURDAT2: bad hemisphere in coordinate {token!r}")
    return value


def parse_hurdat2(path, storm_id):
    """Parse one storm's block out of a HURDAT2 Atlantic/Pacific best-track file.

    Format: a header line `AL122005,            KATRINA,     34,` followed by that many
    data lines `YYYYMMDD, HHMM, R, SS, LAT, LON, VMAX_KT, PMIN_HPA, <34 more radii...>`.
    -999 means missing and becomes None (never 0, never silently kept as -999).

    Returns a list of dicts sorted by time, each with:
        time (datetime), lat, lon (signed degrees), vmax_ms, vmax_kt, pmin_hpa (or None),
        status (2-char), record_id ('L' for landfall, '' normally).
    """
    if not os.path.exists(path):
        raise MissingObservationFile(
            f"HURDAT2 file not found: {path}\n{EXPECTED_OBS_HINT['track']}")
    storm_id = storm_id.strip().upper()
    points, want, seen_ids = [], 0, []
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            fields = [f.strip() for f in line.split(",")]
            is_header = len(fields) >= 3 and fields[0][:2].isalpha() and fields[0][2:6].isdigit()
            if is_header:
                seen_ids.append(fields[0].upper())
                if want > 0:
                    break                      # our block ended and the next one began
                if fields[0].upper() == storm_id:
                    want = int(fields[2])
                continue
            if want <= 0:
                continue
            if len(fields) < 8:
                raise VerifyObsError(f"HURDAT2: short data line in {storm_id}: {line!r}")
            when = datetime.strptime(fields[0] + fields[1], "%Y%m%d%H%M")
            pmin = int(fields[7])
            vmax = int(fields[6])
            points.append({
                "time": when,
                "lat": _hurdat2_coord(fields[4]),
                "lon": _hurdat2_coord(fields[5]),
                "vmax_kt": None if vmax == HURDAT2_MISSING else float(vmax),
                "vmax_ms": None if vmax == HURDAT2_MISSING else float(vmax) * KT_TO_MS,
                "pmin_hpa": None if pmin == HURDAT2_MISSING else float(pmin),
                "status": fields[3],
                "record_id": fields[2],
            })
    if want == 0:
        raise VerifyObsError(
            f"HURDAT2: storm id {storm_id!r} not found in {path} "
            f"({len(seen_ids)} storms in file). Ids look like 'AL122005'.")
    if len(points) != want:
        raise VerifyObsError(
            f"HURDAT2: {storm_id} header declares {want} records but {len(points)} parsed")
    points.sort(key=lambda p: p["time"])
    return points


# --------------------------------------------------------------------------------------
# station observation CSV
# --------------------------------------------------------------------------------------

_TIME_COLUMN_CANDIDATES = ("time", "times", "valid", "valid_time", "datetime", "date_time",
                           "date", "timestamp", "utc", "date_utc", "obs_time")
_VALUE_COLUMN_CANDIDATES = ("t2", "t2m", "temp", "temperature", "tmpc", "tmpf", "air_temp",
                            "temperature_c", "obs", "observed", "value", "t2_c", "tair")
_TIME_FORMATS = ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                 "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
                 "%Y-%m-%d %H%M", "%Y%m%d%H%M", "%Y%m%d %H%M", "%Y/%m/%d %H:%M",
                 "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d_%H")


def _pick_column(header, explicit, candidates, what, path):
    if explicit:
        if explicit not in header:
            raise VerifyObsError(
                f"observation CSV {path}: no column {explicit!r}. Columns: {header}")
        return explicit
    norm = {c.strip().lower().replace(" ", "_"): c for c in header}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    raise VerifyObsError(
        f"observation CSV {path}: cannot identify the {what} column automatically. "
        f"Columns present: {header}. Pass it explicitly "
        f"(--obs-{'time' if what == 'time' else 'value'}-column, or the "
        f"'{'time_column' if what == 'time' else 'value_column'}' key in the case's "
        f"observations block).")


def _parse_obs_time(text, fmt):
    text = text.strip()
    if fmt:
        return datetime.strptime(text, fmt)
    for candidate in _TIME_FORMATS:
        try:
            return datetime.strptime(text, candidate)
        except ValueError:
            continue
    raise VerifyObsError(
        f"observation CSV: cannot parse timestamp {text!r}. Pass --obs-time-format "
        f"(strptime, e.g. '%Y-%m-%d %H:%M') or a 'time_format' key in the case's "
        f"observations block.")


def _to_celsius(value, units):
    units = (units or "C").strip().upper()
    if units in ("C", "DEGC", "CELSIUS"):
        return value
    if units == "K":
        return value - 273.15
    if units in ("F", "DEGF", "FAHRENHEIT"):
        return (value - 32.0) * 5.0 / 9.0
    raise VerifyObsError(f"unknown observation units {units!r}; use C, K or F")


def read_station_obs(path, time_column=None, value_column=None, time_format=None, units="C"):
    """Read an hourly station CSV. Returns a list of (datetime, value_in_C), time-sorted.

    Blank values are dropped (a gap in the record is not a match failure); everything else
    that fails to parse raises, because a silently-dropped bad row moves the RMSE.
    """
    if not os.path.exists(path):
        raise MissingObservationFile(
            f"station observation CSV not found: {path}\n{EXPECTED_OBS_HINT['station']}")
    with open(path, "r", newline="", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise VerifyObsError(f"observation CSV {path} has no data rows")
    header = [h for h in rows[0].keys() if h is not None]
    tcol = _pick_column(header, time_column, _TIME_COLUMN_CANDIDATES, "time", path)
    vcol = _pick_column(header, value_column, _VALUE_COLUMN_CANDIDATES, "value", path)
    out = []
    for n, row in enumerate(rows, start=2):
        raw = (row.get(vcol) or "").strip()
        if raw == "" or raw.upper() in ("NA", "NAN", "M", "MISSING"):
            continue
        try:
            value = float(raw)
        except ValueError as exc:
            raise VerifyObsError(
                f"observation CSV {path} line {n}: value {raw!r} is not a number") from exc
        out.append((_parse_obs_time(row[tcol], time_format), _to_celsius(value, units)))
    if not out:
        raise VerifyObsError(f"observation CSV {path}: every row had a missing value")
    out.sort(key=lambda r: r[0])
    return out


# --------------------------------------------------------------------------------------
# wrfout access
# --------------------------------------------------------------------------------------

def expand_wrfout(patterns):
    """Globs/paths -> sorted unique file list. Raises when nothing matches."""
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    files = []
    for pattern in patterns:
        pattern = str(pattern)
        hits = sorted(globmod.glob(pattern))
        if not hits and os.path.exists(pattern):
            hits = [pattern]
        files.extend(hits)
    files = sorted(set(files))
    if not files:
        raise VerifyObsError(f"no wrfout files matched: {list(patterns)}")
    return files


def _frame_times(ds):
    times = _netcdf4().chartostring(ds.variables["Times"][:])
    return [datetime.strptime(str(t).strip(), WRF_TIME_FMT) for t in np.atleast_1d(times)]


def station_series(paths, lat, lon, variable="T2"):
    """Nearest-grid-point time series of `variable` (model units). Returns (series, meta).

    series: list of (datetime, value) sorted by time, duplicates by time dropped
            (a restarted run can write the same valid time twice).
    meta:   which cell was picked and how far it is from the requested point.
    """
    files = expand_wrfout(paths)
    netcdf4 = _netcdf4()
    picked, by_time = None, {}
    for path in files:
        with netcdf4.Dataset(path) as ds:
            if variable not in ds.variables:
                raise VerifyObsError(f"{path}: no variable {variable!r}")
            if picked is None:
                lat2d = np.asarray(ds.variables["XLAT"][0])
                lon2d = np.asarray(ds.variables["XLONG"][0])
                j, i, glat, glon, dist = nearest_index(lat2d, lon2d, lat, lon)
                picked = {"j": j, "i": i, "grid_lat": glat, "grid_lon": glon,
                          "distance_km": dist, "ny": int(lat2d.shape[0]),
                          "nx": int(lat2d.shape[1])}
            var = ds.variables[variable]
            for t, when in enumerate(_frame_times(ds)):
                value = var[t, picked["j"], picked["i"]] if var.ndim == 3 else \
                    var[t, 0, picked["j"], picked["i"]]
                by_time[when] = float(value)
    picked["files"] = len(files)
    picked["frames"] = len(by_time)
    return sorted(by_time.items()), picked


def storm_centres(paths, ocean_only=True):
    """Per frame: (time, centre lat/lon, min SLP hPa, max 10 m wind m/s).

    Centre = the minimum of the sea-level-reduced pressure (predictor/wrf_slp_vendored.py),
    NOT min(PSFC): PSFC carries terrain and high ground beats a storm centre. With
    ocean_only the search is restricted to LANDMASK == 0, which is what makes a Gulf
    hurricane win over the continental interior.

    Max wind is the DOMAIN maximum of sqrt(U10^2 + V10^2) -- see the module docstring for
    why that is not the same quantity as HURDAT2's vmax.
    """
    files = expand_wrfout(paths)
    netcdf4 = _netcdf4()
    by_time = {}
    for path in files:
        with netcdf4.Dataset(path) as ds:
            for name in ("P", "PB", "T", "QVAPOR", "PH", "PHB", "XLAT", "XLONG"):
                if name not in ds.variables:
                    raise VerifyObsError(
                        f"{path}: no variable {name!r} -- track mode needs the 3-D state "
                        f"(P, PB, T, QVAPOR, PH, PHB) to reduce pressure to sea level")
            lat2d = np.asarray(ds.variables["XLAT"][0])
            lon2d = np.asarray(ds.variables["XLONG"][0])
            mask = None
            if ocean_only and "LANDMASK" in ds.variables:
                mask = np.asarray(ds.variables["LANDMASK"][0]) < 0.5
                if not mask.any():
                    raise VerifyObsError(
                        f"{path}: --ocean-only but LANDMASK has no water point")
            for t, when in enumerate(_frame_times(ds)):
                slp = wrf_slp_vendored.slp_from_frame(
                    np.asarray(ds.variables["P"][t]), np.asarray(ds.variables["PB"][t]),
                    np.asarray(ds.variables["T"][t]), np.asarray(ds.variables["QVAPOR"][t]),
                    np.asarray(ds.variables["PH"][t]), np.asarray(ds.variables["PHB"][t]))
                search = np.where(mask, slp, np.inf) if mask is not None else slp
                j, i = np.unravel_index(int(np.argmin(search)), search.shape)
                wind = None
                if "U10" in ds.variables and "V10" in ds.variables:
                    u10 = np.asarray(ds.variables["U10"][t])
                    v10 = np.asarray(ds.variables["V10"][t])
                    wind = float(np.max(np.sqrt(u10 ** 2 + v10 ** 2)))
                by_time[when] = {
                    "time": when, "lat": float(lat2d[j, i]), "lon": float(lon2d[j, i]),
                    "pmin_hpa": float(slp[j, i]), "vmax_ms": wind,
                    "j": int(j), "i": int(i),
                }
    return [by_time[k] for k in sorted(by_time)]


# --------------------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------------------

def match_times(obs_times, model_times, tol_s):
    """Pair each observation time with the closest model time within tol_s.

    Returns [(obs_index, model_index, delta_seconds)]. Each model time is used at most
    once; unmatched observation times are simply absent from the result -- the CALLER
    decides whether the match count is acceptable (see _require_matches). Nothing here
    interpolates: a nearest-in-time pairing inside a tolerance is honest about what was
    compared, an interpolation would invent an observation.
    """
    used, pairs = set(), []
    model_sorted = sorted(range(len(model_times)), key=lambda k: model_times[k])
    for oi, otime in enumerate(obs_times):
        best, best_delta = None, None
        for mi in model_sorted:
            if mi in used:
                continue
            delta = abs((model_times[mi] - otime).total_seconds())
            if delta <= tol_s and (best_delta is None or delta < best_delta):
                best, best_delta = mi, delta
        if best is not None:
            used.add(best)
            pairs.append((oi, best, best_delta))
    return pairs


def _require_matches(n_matched, n_candidate, min_frac, what):
    frac = (n_matched / n_candidate) if n_candidate else 0.0
    if n_candidate == 0:
        raise InsufficientMatches(
            f"no {what} fall inside the model's output window at all -- nothing to verify")
    if frac < min_frac:
        raise InsufficientMatches(
            f"only {n_matched} of {n_candidate} {what} matched a model frame "
            f"({frac:.1%} < required {min_frac:.1%}). REFUSING to report a metric from a "
            f"partial match: a metric computed over a silently-shrunk sample is not the "
            f"metric the problem statement asks for. Check the run's history_interval, "
            f"its start/end times, and --time-tol-s.")
    return frac


def _stats(model_values, obs_values):
    model = np.asarray(model_values, dtype=float)
    obs = np.asarray(obs_values, dtype=float)
    diff = model - obs
    out = {
        "n": int(model.size),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "model_mean": float(np.mean(model)),
        "obs_mean": float(np.mean(obs)),
        "obs_std": float(np.std(obs)),
    }
    if model.size >= 2 and np.std(model) > 0 and np.std(obs) > 0:
        out["correlation"] = float(np.corrcoef(model, obs)[0, 1])
    else:
        out["correlation"] = None      # undefined, not 0 -- a constant series has no r
    return out


# --------------------------------------------------------------------------------------
# verification modes
# --------------------------------------------------------------------------------------

def verify_station(wrfout, obs_path, lat, lon, *, variable="T2", time_column=None,
                   value_column=None, time_format=None, obs_units="C",
                   min_match_frac=0.9, time_tol_s=60.0, station_name=None):
    """SC26-P1 metric: RMSE / mean bias / correlation of 2 m temperature in degrees C."""
    obs = read_station_obs(obs_path, time_column, value_column, time_format, obs_units)
    series, meta = station_series(wrfout, lat, lon, variable=variable)
    if not series:
        raise VerifyObsError("no model frames found")
    model_times = [t for t, _ in series]
    lo, hi = model_times[0], model_times[-1]
    tol = timedelta(seconds=time_tol_s)
    in_window = [k for k, (t, _) in enumerate(obs) if lo - tol <= t <= hi + tol]
    pairs = match_times([obs[k][0] for k in in_window], model_times, time_tol_s)
    frac = _require_matches(len(pairs), len(in_window), min_match_frac,
                            "observation times inside the model window")

    rows = []
    for oi, mi, delta in pairs:
        otime, ovalue = obs[in_window[oi]]
        mvalue_c = series[mi][1] - 273.15          # T2 is K; the metric is C
        rows.append({"time": otime.strftime(WRF_TIME_FMT),
                     "model_time": model_times[mi].strftime(WRF_TIME_FMT),
                     "dt_s": float(delta), "obs_c": ovalue, "model_c": mvalue_c,
                     "error_c": mvalue_c - ovalue})
    stats = _stats([r["model_c"] for r in rows], [r["obs_c"] for r in rows])
    return {
        "kind": "station",
        "station": {"name": station_name, "lat": lat, "lon": lon, **meta},
        "obs_file": os.path.abspath(obs_path),
        "obs_points_total": len(obs),
        "obs_points_in_window": len(in_window),
        "matched": len(pairs),
        "matched_fraction": frac,
        "min_match_frac": min_match_frac,
        "window": [lo.strftime(WRF_TIME_FMT), hi.strftime(WRF_TIME_FMT)],
        "rmse_c": stats["rmse"],
        "bias_c": stats["bias"],
        "mae_c": stats["mae"],
        "correlation": stats["correlation"],
        "model_mean_c": stats["model_mean"],
        "obs_mean_c": stats["obs_mean"],
        "obs_std_c": stats["obs_std"],
        "series": rows,
    }


def verify_track(wrfout, hurdat2_path, storm_id, *, ocean_only=True, min_match_frac=0.9,
                 time_tol_s=60.0, synoptic_only=True):
    """SC26-P2 metric: track error (km), central-pressure error (hPa), max-wind error (m/s).

    synoptic_only drops HURDAT2's off-hour landfall records (record_id 'L' at e.g. 2230Z):
    with history_interval=60 the model has no frame at those times, and counting them as
    unmatched would trip the refusal for a reason that is not a defect of the run.
    """
    best = parse_hurdat2(hurdat2_path, storm_id)
    centres = storm_centres(wrfout, ocean_only=ocean_only)
    if not centres:
        raise VerifyObsError("no model frames found")
    model_times = [c["time"] for c in centres]
    lo, hi = model_times[0], model_times[-1]
    tol = timedelta(seconds=time_tol_s)
    candidates = [p for p in best if lo - tol <= p["time"] <= hi + tol
                  and (not synoptic_only or (p["time"].minute == 0 and p["time"].hour % 6 == 0))]
    pairs = match_times([p["time"] for p in candidates], model_times, time_tol_s)
    frac = _require_matches(len(pairs), len(candidates), min_match_frac,
                            "best-track times inside the model window")

    rows = []
    for oi, mi, delta in pairs:
        obs_pt, model_pt = candidates[oi], centres[mi]
        lead_h = (model_pt["time"] - lo).total_seconds() / 3600.0
        row = {
            "lead_h": lead_h,
            "time": obs_pt["time"].strftime(WRF_TIME_FMT),
            "dt_s": float(delta),
            "obs_lat": obs_pt["lat"], "obs_lon": obs_pt["lon"],
            "model_lat": model_pt["lat"], "model_lon": model_pt["lon"],
            "track_error_km": float(great_circle_km(
                obs_pt["lat"], obs_pt["lon"], model_pt["lat"], model_pt["lon"])),
            "obs_pmin_hpa": obs_pt["pmin_hpa"],
            "model_pmin_hpa": model_pt["pmin_hpa"],
            "pmin_error_hpa": (None if obs_pt["pmin_hpa"] is None
                               else model_pt["pmin_hpa"] - obs_pt["pmin_hpa"]),
            "obs_vmax_ms": obs_pt["vmax_ms"],
            "model_vmax_ms": model_pt["vmax_ms"],
            "vmax_error_ms": (None if obs_pt["vmax_ms"] is None or model_pt["vmax_ms"] is None
                              else model_pt["vmax_ms"] - obs_pt["vmax_ms"]),
        }
        rows.append(row)
    rows.sort(key=lambda r: r["lead_h"])

    def _agg(key):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return {"n": 0, "mean": None, "abs_mean": None, "max_abs": None, "rmse": None}
        arr = np.asarray(vals, dtype=float)
        return {"n": int(arr.size), "mean": float(arr.mean()),
                "abs_mean": float(np.abs(arr).mean()), "max_abs": float(np.abs(arr).max()),
                "rmse": float(np.sqrt(np.mean(arr ** 2)))}

    # Deepening over the verified window: the quantity that actually discriminates
    # configurations here, because the absolute pmin bias is set by the initial condition.
    def _deepening(first, last):
        return None if first is None or last is None else first - last

    obs_p = [r["obs_pmin_hpa"] for r in rows if r["obs_pmin_hpa"] is not None]
    mod_p = [r["model_pmin_hpa"] for r in rows if r["model_pmin_hpa"] is not None]
    jumps = [float(great_circle_km(centres[k - 1]["lat"], centres[k - 1]["lon"],
                                   centres[k]["lat"], centres[k]["lon"]))
             for k in range(1, len(centres))]

    track = _agg("track_error_km")
    return {
        "kind": "track",
        "storm_id": storm_id.upper(),
        "obs_file": os.path.abspath(hurdat2_path),
        "ocean_only": ocean_only,
        "best_track_points_total": len(best),
        "best_track_points_in_window": len(candidates),
        "matched": len(rows),
        "matched_fraction": frac,
        "min_match_frac": min_match_frac,
        "window": [lo.strftime(WRF_TIME_FMT), hi.strftime(WRF_TIME_FMT)],
        "model_frames": len(centres),
        "track_error_km_mean": track["mean"],
        "track_error_km_max": track["max_abs"],
        "pmin_error_hpa": _agg("pmin_error_hpa"),
        "vmax_error_ms": _agg("vmax_error_ms"),
        "model_pmin_min_hpa": min(mod_p) if mod_p else None,
        "obs_pmin_min_hpa": min(obs_p) if obs_p else None,
        "model_deepening_hpa": _deepening(mod_p[0], mod_p[-1]) if len(mod_p) >= 2 else None,
        "obs_deepening_hpa": _deepening(obs_p[0], obs_p[-1]) if len(obs_p) >= 2 else None,
        "model_vmax_max_ms": max([r["model_vmax_ms"] for r in rows
                                  if r["model_vmax_ms"] is not None], default=None),
        "max_centre_jump_km": max(jumps) if jumps else None,
        "series": rows,
    }


# --------------------------------------------------------------------------------------
# case-driven entry points (this is what recommend.py calls)
# --------------------------------------------------------------------------------------

def observations_spec(case_entry):
    """Return the case's `observations` block, validated, or None when it declares none.

    A case with no observations is NOT an error: the caller falls back to the existing
    model-vs-model fidelity. A case with a MALFORMED observations block IS an error --
    silently falling back would hide a typo'd path as "this case has no observations".
    """
    if not isinstance(case_entry, dict):
        raise VerifyObsError(f"case entry must be a dict, got {type(case_entry).__name__}")
    spec = case_entry.get("observations")
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise VerifyObsError("case 'observations' must be an object")
    kind = spec.get("kind")
    if kind not in OBSERVATION_KINDS:
        raise VerifyObsError(
            f"case 'observations.kind' must be one of {OBSERVATION_KINDS}, got {kind!r}")
    if not spec.get("path"):
        raise VerifyObsError("case 'observations.path' is required")
    if kind == "station":
        station = spec.get("station") or {}
        if "lat" not in station or "lon" not in station:
            raise VerifyObsError(
                "case 'observations.station' needs 'lat' and 'lon' for kind='station'")
    elif kind == "track" and not spec.get("storm_id"):
        raise VerifyObsError("case 'observations.storm_id' is required for kind='track'")
    return spec


def verify(spec, wrfout, *, min_match_frac=None, time_tol_s=None):
    """Run whichever mode the spec declares. `spec` is a case's observations block."""
    kind = spec["kind"]
    # A case that names a domain means it: SC26-P1's d01 (9 km) and d02 (3 km) give different
    # RMSEs at South Bend, and scoring the wrong stream would look like a config effect.
    domain = spec.get("domain")
    if domain:
        files = expand_wrfout(wrfout)
        if not any(f"_{domain}_" in os.path.basename(f) for f in files):
            raise VerifyObsError(
                f"this case's observations block scores domain {domain!r}, but none of the "
                f"{len(files)} wrfout file(s) given is a {domain} history stream "
                f"(first: {os.path.basename(files[0])}). Pass the {domain} glob.")
    frac = spec.get("min_match_frac", 0.9) if min_match_frac is None else min_match_frac
    tol = spec.get("time_tol_s", 60.0) if time_tol_s is None else time_tol_s
    if kind == "station":
        station = spec["station"]
        return verify_station(
            wrfout, spec["path"], float(station["lat"]), float(station["lon"]),
            variable=spec.get("variable", "T2"),
            time_column=spec.get("time_column"), value_column=spec.get("value_column"),
            time_format=spec.get("time_format"), obs_units=spec.get("units", "C"),
            min_match_frac=frac, time_tol_s=tol, station_name=station.get("name"))
    return verify_track(
        wrfout, spec["path"], spec["storm_id"],
        ocean_only=bool(spec.get("ocean_only", True)),
        min_match_frac=frac, time_tol_s=tol,
        synoptic_only=bool(spec.get("synoptic_only", True)))


def accuracy_error(result):
    """The single scalar the accuracy objective ranks on. LOWER IS BETTER, always.

    station -> RMSE of 2 m temperature in C   (the metric SC26-P1 specifies)
    track   -> mean great-circle track error in km (the metric SC26-P2 is scored on)
    """
    metric = PRIMARY_METRIC[result["kind"]]
    value = result[metric]
    if value is None or not math.isfinite(float(value)):
        raise VerifyObsError(f"{metric} is not finite -- refusing to rank on it")
    return float(value)


def accuracy_error_source(result):
    """Provenance string for recommend.py's `fidelity_error_source` slot."""
    metric = PRIMARY_METRIC[result["kind"]]
    return (f"measured vs observations ({result['kind']}, {metric}, "
            f"{result['matched']} matched times, {os.path.basename(result['obs_file'])})")


def score_case(case_entry, wrfout, *, min_match_frac=None, time_tol_s=None):
    """One call for recommend.py: (error, source, result), or (None, None, None).

    Returns (None, None, None) exactly when the case declares no `observations` block --
    the caller then falls back to the existing model-vs-model fidelity. A malformed block
    or an unreadable observation file RAISES: silently falling back would turn a typo'd
    path into "this case has no ground truth", which is the failure this tool exists to
    prevent.
    """
    spec = observations_spec(case_entry)
    if spec is None:
        return None, None, None
    result = verify(spec, wrfout, min_match_frac=min_match_frac, time_tol_s=time_tol_s)
    return accuracy_error(result), accuracy_error_source(result), result


def load_case(cases_path, case_name):
    with open(cases_path, "r") as fh:
        doc = json.load(fh)
    cases = doc.get("cases", doc)
    if case_name not in cases:
        raise VerifyObsError(f"case {case_name!r} not in {cases_path}")
    return cases[case_name]


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------

def _fmt(value, spec="{:.3f}"):
    return "--" if value is None else spec.format(value)


def format_report(result):
    out = []
    if result["kind"] == "station":
        st = result["station"]
        out.append(f"station verification vs {result['obs_file']}")
        out.append(f"  station        {st.get('name') or '(unnamed)'} "
                   f"at {st['lat']:.4f} N, {st['lon']:.4f} E")
        out.append(f"  nearest cell   (j={st['j']}, i={st['i']}) at "
                   f"{st['grid_lat']:.4f} N, {st['grid_lon']:.4f} E "
                   f"-- {st['distance_km']:.2f} km from the station")
        out.append(f"  window         {result['window'][0]} .. {result['window'][1]}")
        out.append(f"  matched        {result['matched']} of "
                   f"{result['obs_points_in_window']} obs in window "
                   f"({result['matched_fraction']:.1%}; "
                   f"{result['obs_points_total']} in the file)")
        out.append("")
        out.append(f"  RMSE           {result['rmse_c']:.3f} C")
        out.append(f"  mean bias      {result['bias_c']:+.3f} C  (model - obs)")
        out.append(f"  MAE            {result['mae_c']:.3f} C")
        out.append(f"  correlation    {_fmt(result['correlation'], '{:.4f}')}")
        out.append(f"  means          model {result['model_mean_c']:.3f} C / "
                   f"obs {result['obs_mean_c']:.3f} C (obs std {result['obs_std_c']:.3f})")
        return "\n".join(out)

    out.append(f"track verification vs {result['obs_file']} ({result['storm_id']})")
    out.append(f"  window         {result['window'][0]} .. {result['window'][1]} "
               f"({result['model_frames']} model frames)")
    out.append(f"  matched        {result['matched']} of "
               f"{result['best_track_points_in_window']} best-track times in window "
               f"({result['matched_fraction']:.1%})")
    out.append(f"  centre search  minimum sea-level pressure"
               f"{', ocean points only' if result['ocean_only'] else ''}; "
               f"largest frame-to-frame centre jump "
               f"{_fmt(result['max_centre_jump_km'], '{:.0f}')} km")
    out.append("")
    out.append("  lead   valid time            track err   pmin model/obs      "
               "pmin err   vmax model/obs     vmax err")
    out.append("  " + "-" * 104)
    for r in result["series"]:
        out.append(
            f"  {r['lead_h']:5.0f}h {r['time']}  "
            f"{r['track_error_km']:8.1f} km  "
            f"{_fmt(r['model_pmin_hpa'], '{:7.1f}')}/{_fmt(r['obs_pmin_hpa'], '{:6.1f}')} hPa  "
            f"{_fmt(r['pmin_error_hpa'], '{:+7.1f}')}  "
            f"{_fmt(r['model_vmax_ms'], '{:6.1f}')}/{_fmt(r['obs_vmax_ms'], '{:5.1f}')} m/s  "
            f"{_fmt(r['vmax_error_ms'], '{:+7.1f}')}")
    out.append("  " + "-" * 104)
    track_mean = result["track_error_km_mean"]
    out.append(f"  mean track error     {_fmt(track_mean, '{:.1f}')} km "
               f"(max {_fmt(result['track_error_km_max'], '{:.1f}')} km)")
    pm, vm = result["pmin_error_hpa"], result["vmax_error_ms"]
    out.append(f"  mean pmin error      {_fmt(pm['mean'], '{:+.1f}')} hPa "
               f"(|mean| {_fmt(pm['abs_mean'], '{:.1f}')}, RMSE {_fmt(pm['rmse'], '{:.1f}')})")
    out.append(f"  mean vmax error      {_fmt(vm['mean'], '{:+.1f}')} m/s "
               f"(|mean| {_fmt(vm['abs_mean'], '{:.1f}')}, RMSE {_fmt(vm['rmse'], '{:.1f}')})")
    out.append(f"  deepening            model {_fmt(result['model_deepening_hpa'], '{:.1f}')} hPa"
               f" vs best track {_fmt(result['obs_deepening_hpa'], '{:.1f}')} hPa")
    out.append(f"  minimum pressure     model {_fmt(result['model_pmin_min_hpa'], '{:.1f}')} hPa"
               f" vs best track {_fmt(result['obs_pmin_min_hpa'], '{:.1f}')} hPa")
    out.append("")
    out.append("  NOTE model vmax is a DOMAIN maximum of sqrt(U10^2+V10^2); HURDAT2 vmax is a")
    out.append("       1-minute sustained wind near the centre. A coarse grid necessarily")
    out.append("       underestimates it -- do not read the wind error as a model error.")
    out.append("  NOTE the absolute pmin bias is dominated by the initial condition. Rank")
    out.append("       configurations on the deepening rate and the track instead.")
    return "\n".join(out)


def write_csv(result, path):
    """The per-time series, so a reported number keeps a machine-readable dataset."""
    rows = result["series"]
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="verify_obs.py",
        description="Score a WRF run against real observations (station 2 m temperature, "
                    "or NHC HURDAT2 hurricane best track). Companion to fidelity_gate.py, "
                    "which scores against another model run instead.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(parser):
        parser.add_argument("--wrfout", nargs="+", required=True,
                            help="wrfout file(s) or glob(s), e.g. 'run/wrfout_d01_*'")
        parser.add_argument("--min-match-frac", type=float, default=0.9,
                            help="refuse to report a metric below this matched fraction "
                                 "(default 0.9)")
        parser.add_argument("--time-tol-s", type=float, default=60.0,
                            help="observation/model time pairing tolerance, seconds "
                                 "(default 60)")
        parser.add_argument("--json", dest="json_out", default=None,
                            help="write the full result (including the per-time series) here")
        parser.add_argument("--csv", dest="csv_out", default=None,
                            help="write the per-time series here as CSV")

    p = sub.add_parser("station", help="SC26-P1: RMSE / mean bias of 2 m temperature")
    common(p)
    p.add_argument("--obs", required=True, help="hourly observation CSV")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--name", default=None, help="station name, for the report only")
    p.add_argument("--variable", default="T2")
    p.add_argument("--obs-time-column", default=None)
    p.add_argument("--obs-value-column", default=None)
    p.add_argument("--obs-time-format", default=None, help="strptime format, e.g. '%%Y-%%m-%%d %%H:%%M'")
    p.add_argument("--obs-units", default="C", choices=["C", "K", "F"])

    p = sub.add_parser("track", help="SC26-P2: track / central pressure / max wind vs HURDAT2")
    common(p)
    p.add_argument("--hurdat2", required=True, help="HURDAT2 best-track text file on disk")
    p.add_argument("--storm-id", required=True, help="e.g. AL122005")
    p.add_argument("--all-records", action="store_true",
                   help="also verify off-hour landfall records (default: 6-hourly only)")
    p.add_argument("--include-land", action="store_true",
                   help="search the whole domain for the pressure minimum, not just "
                        "LANDMASK == 0 (default is ocean only)")

    p = sub.add_parser("case", help="run whichever mode campaign/cases.json declares")
    common(p)
    p.add_argument("--cases", default=None,
                   help="path to cases.json (default: campaign/cases.json next to this repo)")
    p.add_argument("--case", required=True, help="case name, e.g. sc26p2")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.mode == "station":
            result = verify_station(
                args.wrfout, args.obs, args.lat, args.lon, variable=args.variable,
                time_column=args.obs_time_column, value_column=args.obs_value_column,
                time_format=args.obs_time_format, obs_units=args.obs_units,
                min_match_frac=args.min_match_frac, time_tol_s=args.time_tol_s,
                station_name=args.name)
        elif args.mode == "track":
            result = verify_track(
                args.wrfout, args.hurdat2, args.storm_id,
                ocean_only=not args.include_land, min_match_frac=args.min_match_frac,
                time_tol_s=args.time_tol_s, synoptic_only=not args.all_records)
        else:
            cases_path = args.cases or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "campaign", "cases.json")
            spec = observations_spec(load_case(cases_path, args.case))
            if spec is None:
                print(f"case {args.case!r} declares no 'observations' block: there is no "
                      f"observation ground truth for it. Use fidelity_gate.py (model vs "
                      f"model) for this case.", file=sys.stderr)
                return 2
            result = verify(spec, args.wrfout, min_match_frac=args.min_match_frac,
                            time_tol_s=args.time_tol_s)
    except MissingObservationFile as exc:
        print(f"verify_obs: MISSING OBSERVATION FILE\n{exc}", file=sys.stderr)
        return 3
    except InsufficientMatches as exc:
        print(f"verify_obs: REFUSED\n{exc}", file=sys.stderr)
        return 4
    except VerifyObsError as exc:
        print(f"verify_obs: {exc}", file=sys.stderr)
        return 1

    print(format_report(result))
    print(f"\naccuracy_error ({PRIMARY_METRIC[result['kind']]}, lower is better): "
          f"{accuracy_error(result):.4f}")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
        print(f"wrote {args.json_out}", file=sys.stderr)
    if args.csv_out:
        write_csv(result, args.csv_out)
        print(f"wrote {args.csv_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

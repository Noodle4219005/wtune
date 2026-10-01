"""post/tc.py -- `wtune post tc RUN_DIR [RUN_DIR ...] --hurdat2 F --storm ID --out DIR`:
per-run TC track via TempestExtremes (analysis.tc_track, reused as-is -- Vmax/MSLP per
point come straight from its own track, no self-written re-derivation), model quadrant
34/50/64 kt wind radii (post/tc_metrics.py's one surviving piece), written out as an ATCF
a-deck (post/atcf.py) and verified against an ATCF b-deck (downloaded from the NHC archive,
a local --bdeck file, or HURDAT2 as a last-resort fallback converted through the same ATCF
writer) with MET tc_pairs/tc_stat (track/intensity/wind-radii error) and tc_rmw
(azimuthal-mean radius-height structure) (post/met_tc.py) -- decision 0056, 2026-09-29
update: wrap established tools, delete the self-written track-error/intensity/azimuthal-
profile/RMW code the previous version of this module carried.

Finest-domain tracking (2026-09-29): the parent-domain (d01) track is only the seed. At every
output time the storm is located on the finest domain whose INTERIOR (spec_bdy_width cells
excluded, post/caseinfo.py) contains it -- containment tested by grid index (wrf.ll_to_xy),
not a lat/lon bbox -- and falls back to the parent domain otherwise; the per-time domain goes
to track_domains_<label>.csv (the ATCF a-deck has no field for it). MSLP/Vmax in the a-deck,
and therefore in MET tc_pairs, are that domain's own values.

wrf-python/netCDF4/matplotlib are imported lazily, inside functions only. MET is a
separately-installed tool (bin/install-postproc.sh, WTUNE_POSTPROC_VIEW): if it is missing,
the a-deck/b-deck/track-map part of this report still runs (it only needs TempestExtremes
and wrf-python), and the MET-dependent error/RMW figures are skipped with a warning in the
returned summary rather than failing the whole report -- see run_tc_report()'s docstring.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

from post import atcf, met_tc, tc_metrics as tm, tools

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"
KT_TO_MS = 0.514444
NMI_TO_KM = 1.852
HURDAT2_MISSING = -999


class TCError(RuntimeError):
    pass


def _netcdf4():
    import netCDF4
    return netCDF4


def _wrf():
    import wrf
    return wrf


# --------------------------------------------------------------------------------------
# observations: HURDAT2 (fallback b-deck source) + optional EBTRK (RMW only)
# --------------------------------------------------------------------------------------

def _hurdat2_coord(token):
    token = token.strip()
    hemi, value = token[-1].upper(), float(token[:-1])
    return -value if hemi in ("S", "W") else value


def parse_hurdat2_full(path, storm_id):
    """HURDAT2 best track including the 34/50/64 kt quadrant radii and RMW (2021+ column
    layout: ..., pmin, 34ktNE,SE,SW,NW, 50kt..., 64kt..., RMW). -999 -> None per field.

    Returns a list of dicts sorted by time: {time, lat, lon, vmax_ms, pmin_hpa,
    r34_ne_km..r64_nw_km, rmw_km (None if this HURDAT2 file predates the RMW column)}."""
    if not os.path.exists(path):
        raise TCError("HURDAT2 file not found: %s" % path)
    storm_id = storm_id.strip().upper()
    points, want = [], 0
    with open(path, errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            fields = [f.strip() for f in line.split(",")]
            is_header = len(fields) >= 3 and fields[0][:2].isalpha() and fields[0][2:6].isdigit()
            if is_header:
                if want > 0:
                    break
                if fields[0].upper() == storm_id:
                    want = int(fields[2])
                continue
            if want <= 0:
                continue
            when = datetime.strptime(fields[0] + fields[1], "%Y%m%d%H%M")

            def _val(idx):
                if idx >= len(fields) or fields[idx] == "":
                    return None
                v = float(fields[idx])
                return None if v == HURDAT2_MISSING else v

            vmax_kt = _val(6)
            row = dict(time=when, lat=_hurdat2_coord(fields[4]), lon=_hurdat2_coord(fields[5]),
                      vmax_ms=None if vmax_kt is None else vmax_kt * KT_TO_MS,
                      pmin_hpa=_val(7))
            quad_idx = {"r34": 8, "r50": 12, "r64": 16}
            for name, base in quad_idx.items():
                for k, qn in enumerate(tm.QUADRANT_NAMES):
                    nmi = _val(base + k)
                    row["%s_%s_km" % (name, qn)] = None if nmi is None else nmi * NMI_TO_KM
            rmw_nmi = _val(20)
            row["rmw_km"] = None if rmw_nmi is None else rmw_nmi * NMI_TO_KM
            points.append(row)
    if want == 0:
        raise TCError("HURDAT2: storm id %r not found in %s" % (storm_id, path))
    points.sort(key=lambda p: p["time"])
    return points


def parse_ebtrk_rmw(path, storm_id):
    """Extended Best Track RMW only (nmi -> km). Row layout (whitespace-separated):
    storm-id, name, MMDDHH, year, lat, lon, vmax, pmin, rmw, eye, poci, roci, ... -- R34
    quadrants in this format can be glued digits with no separator and are deliberately
    NOT parsed here; only column index 8 (rmw, nmi) is read.

    Returns [{"time": datetime, "rmw_km": float}, ...] sorted by time.
    """
    if not os.path.exists(path):
        raise TCError("EBTRK file not found: %s" % path)
    storm_id = storm_id.strip().upper()
    out = []
    with open(path, errors="replace") as fh:
        for raw in fh:
            tokens = raw.split()
            if len(tokens) < 9:
                continue
            if tokens[0].upper() != storm_id:
                continue
            mmddhh, year = tokens[2], tokens[3]
            if len(mmddhh) != 6 or not mmddhh.isdigit():
                continue
            try:
                when = datetime.strptime(year + mmddhh, "%Y%m%d%H")
            except ValueError:
                continue
            try:
                rmw_nmi = float(tokens[8])
            except ValueError:
                continue
            if rmw_nmi < 0:
                continue
            out.append(dict(time=when, rmw_km=rmw_nmi * NMI_TO_KM))
    out.sort(key=lambda p: p["time"])
    return out


def build_obs(hurdat2_path, storm_id, ebtrk_path=None):
    """{"hurdat2": [...], "ebtrk_rmw": [...] or None} -- see module docstring. HURDAT2 is
    kept only as the b-deck fallback source (resolve_bdeck()) and for the EBTRK-adjacent
    RMW column this repo has no ATCF-equivalent for; it is no longer the primary
    observation MET compares against once a real b-deck is available."""
    obs = dict(hurdat2=[dict(r, time=r["time"].strftime(WRF_TIME_FMT))
                        for r in parse_hurdat2_full(hurdat2_path, storm_id)])
    if ebtrk_path:
        obs["ebtrk_rmw"] = [dict(r, time=r["time"].strftime(WRF_TIME_FMT))
                            for r in parse_ebtrk_rmw(ebtrk_path, storm_id)]
    else:
        obs["ebtrk_rmw"] = None
    return obs


def storm_id_parts(storm_id):
    """"AL122005" -> ("AL", "12", 2005) -- basin, cyclone number, year."""
    storm_id = storm_id.strip().upper()
    if len(storm_id) != 8 or not storm_id[2:4].isdigit() or not storm_id[4:].isdigit():
        raise TCError("--storm must look like AL122005 (basin+cyclone+year), got %r" % storm_id)
    return storm_id[:2], storm_id[2:4], int(storm_id[4:])


def resolve_bdeck(storm_id, out_dir, bdeck_path=None, hurdat2_points=None):
    """b-deck ATCF file for MET tc_pairs, in priority order: --bdeck FILE, the NHC ATCF
    archive download (atcf.fetch_bdeck), HURDAT2 converted through the same ATCF writer
    (last-resort fallback -- see module docstring). Returns the resolved path."""
    if bdeck_path:
        if not os.path.exists(bdeck_path):
            raise TCError("--bdeck file not found: %s" % bdeck_path)
        return bdeck_path
    basin, cy, year = storm_id_parts(storm_id)
    dest = os.path.join(out_dir, "b%s%s%d.dat" % (basin.lower(), cy, year))
    try:
        return atcf.fetch_bdeck(year, basin, cy, dest)
    except atcf.ATCFError as e:
        if hurdat2_points is None:
            raise TCError("b-deck download failed (%s) and no HURDAT2 fallback was given" % e)
        print("post tc: b-deck download failed (%s) -- falling back to HURDAT2." % e)
        rows = atcf.hurdat2_to_bdeck_rows(hurdat2_points)
        return atcf.write_lines(rows, dest, basin=basin, cy=cy, tech="BEST")


# --------------------------------------------------------------------------------------
# per-run track + model quadrant wind radii
# --------------------------------------------------------------------------------------

def track_dict(chosen_track):
    """analysis.tc_track chosen track (list of {time (datetime), lat, lon, pmin_hpa,
    vmax_ms}) -> {time_str: point_dict}."""
    return {pt["time"].strftime(WRF_TIME_FMT): pt for pt in chosen_track}


# Tracking on a smoothed SLP field. In a large eye (Katrina m1, d02 4 km) the raw SLP minimum
# hops between mesovortices tens of km apart from one hour to the next, several times the
# storm's own motion, and TempestExtremes follows each hop. The centre is therefore searched on
# CDO-smoothed SLP (same PHYSICAL radius on every domain, so d01 and the nests are comparable),
# while intensity is read from the UNSMOOTHED fields around that centre.
SMOOTH_RADIUS_KM = 30.0        # cdo smooth radius (lead-set 25-40 km; ~7 cells at 4 km, ~2.5 at 12 km)
MSLP_RADIUS_KM = 50.0          # unsmoothed MSLP = min within this radius of the smoothed centre
# Vmax is TempestExtremes' own unsmoothed wspd10 max within TCTrackParams.wind_radius_deg (2 deg).


def _domain_dx_km(run_dir, domain):
    import glob
    hits = sorted(glob.glob(os.path.join(run_dir, "wrfout_d%02d_*" % domain)))
    if not hits:
        raise TCError("no wrfout_d%02d_* files in %s" % (domain, run_dir))
    with _netcdf4().Dataset(hits[0]) as ds:
        return float(ds.getncattr("DX")) / 1000.0


def smooth_slp_field(fields_nc, out_nc, dx_km, radius_km=SMOOTH_RADIUS_KM):
    """fields_nc (analysis.diagnostics.wrfout_to_tc_fields output) copied to out_nc with its `slp`
    replaced by `cdo smooth,radius=<radius_km>km` of it -- CDO's own smoother, no self-written
    filter. maxpoints is sized to the number of grid points inside the radius (CDO's default 81
    would silently shrink the radius on a 4 km grid). Every other variable is untouched, so
    TempestExtremes' wind output stays raw. Raises tools.ToolError/ToolNotFoundError."""
    import math
    import shutil
    netcdf4 = _netcdf4()
    cdo = tools.find_on_path("cdo", install_hint="envs/wtune/pixi.toml provides it.")
    maxpoints = int(1.3 * math.pi * (radius_km / dx_km) ** 2) + 9
    slp_raw, slp_s = out_nc + ".slp_raw.nc", out_nc + ".slp_smooth.nc"
    tools.run([cdo, "-s", "selname,slp", fields_nc, slp_raw])
    tools.run([cdo, "-s", "smooth,radius=%gkm,maxpoints=%d" % (radius_km, maxpoints), slp_raw, slp_s])
    shutil.copyfile(fields_nc, out_nc)
    with netcdf4.Dataset(slp_s) as src, netcdf4.Dataset(out_nc, "a") as dst:
        dst.variables["slp"][:] = src.variables["slp"][:]
    for p in (slp_raw, slp_s):
        os.remove(p)
    return out_nc


def _refine_mslp(tracks, fields_nc, radius_km=MSLP_RADIUS_KM):
    """Replace each point's pmin_hpa (smoothed SLP at the centre) by the minimum of the
    UNSMOOTHED slp in fields_nc within radius_km of the (smoothed) centre. Position unchanged."""
    import numpy as np
    netcdf4 = _netcdf4()
    with netcdf4.Dataset(fields_nc) as ds:
        epoch = datetime(1970, 1, 1)
        hours = np.asarray(ds.variables["time"][:], dtype=float)
        idx = {round(h): k for k, h in enumerate(hours)}
        lat, lon = np.asarray(ds.variables["lat"][:]), np.asarray(ds.variables["lon"][:])
        slp = ds.variables["slp"]
        for tr in tracks:
            for pt in tr:
                k = idx.get(round((pt["time"] - epoch).total_seconds() / 3600.0))
                if k is None:
                    continue
                field = np.asarray(slp[k])
                if field.ndim == 2:
                    la, lo = np.meshgrid(lat, lon, indexing="ij")
                else:
                    la, lo = lat, lon
                near = tm.dist_km(pt["lat"], pt["lon"], la, lo) <= radius_km
                if near.any():
                    pt["pmin_hpa"] = float(field[near].min())
    return tracks


def _tracks_for_domain(run_dir, domain, work_dir, params=None, smooth_km=SMOOTH_RADIUS_KM):
    """analysis.diagnostics + analysis.tc_track (TempestExtremes) on one domain's wrfout files.
    Position: TempestExtremes on CDO-smoothed SLP (smooth_km; None/0 = raw, also the fallback when
    CDO cannot smooth this grid). Intensity: unsmoothed (_refine_mslp; wspd10 was never smoothed).
    Returns ALL stitched tracks (each a time-sorted list of {time, lat, lon, pmin_hpa, vmax_ms})."""
    from analysis import diagnostics as diag_mod
    from analysis import tc_track

    os.makedirs(work_dir, exist_ok=True)
    fields_nc = os.path.join(work_dir, "fields.nc")
    diag_mod.wrfout_to_tc_fields(os.path.join(run_dir, "wrfout_d%02d_*" % domain), fields_nc)
    track_nc = fields_nc
    if smooth_km:
        try:
            track_nc = smooth_slp_field(fields_nc, os.path.join(work_dir, "fields_smooth.nc"),
                                        _domain_dx_km(run_dir, domain), smooth_km)
        except (tools.ToolError, tools.ToolNotFoundError, KeyError, OSError) as e:
            print("post tc: d%02d SLP smoothing unavailable (%s) -- tracking the raw minimum"
                  % (domain, str(e).splitlines()[0] if str(e) else type(e).__name__))
    tracks, _prov = tc_track.detect_and_stitch(track_nc, work_dir, params=params)
    if track_nc != fields_nc:
        _refine_mslp(tracks, fields_nc)
    return tracks


def track_this_run(run_dir, first_guess_lat, first_guess_lon, work_dir, domain=1,
                   smooth_km=SMOOTH_RADIUS_KM):
    """Runs analysis.diagnostics + analysis.tc_track end to end on this run's `domain` files
    (default d01). Returns the chosen track (list of {time, lat, lon, pmin_hpa, vmax_ms})."""
    from analysis import tc_track

    tracks = _tracks_for_domain(run_dir, domain, work_dir, smooth_km=smooth_km)
    if not tracks:
        raise TCError("no track found in %s" % run_dir)
    chosen = tc_track.select_track(tracks, first_guess_lat, first_guess_lon)
    if chosen is None:
        raise TCError("select_track picked nothing for %s" % run_dir)
    return chosen


# --------------------------------------------------------------------------------------
# finest-domain tracking: d01 track = seed, every nest re-tracked, per-time finest interior
# --------------------------------------------------------------------------------------

FINE_TOL_KM = 150.0          # a nest fix farther than this from the coarser track is a different minimum
GRID_LIMITED_DX_KM = 5.0     # coarser than this: position/intensity are grid-limited (user hint b)


def run_domains(run_dir, max_dom=None):
    """{domain: {"frames": {time_str: path}, "dx_km", "ny", "nx", "moving"}} for every
    wrfout_dNN_* domain in run_dir (<= max_dom when given). A nest whose XLAT[0,0] differs
    between its first and last frame is flagged moving (a static-grid tracker cannot follow it)."""
    import glob
    import re
    netcdf4 = _netcdf4()
    doms = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "wrfout_d*_*"))):
        m = re.match(r"wrfout_d(\d+)_(.+)$", os.path.basename(p))
        if m and (max_dom is None or int(m.group(1)) <= max_dom):
            doms.setdefault(int(m.group(1)), {"frames": {}})["frames"][m.group(2)] = p
    for d, info in doms.items():
        times = sorted(info["frames"])
        first, last = info["frames"][times[0]], info["frames"][times[-1]]
        with netcdf4.Dataset(first) as ds:
            info["dx_km"] = float(ds.getncattr("DX")) / 1000.0
            info["ny"] = len(ds.dimensions["south_north"])
            info["nx"] = len(ds.dimensions["west_east"])
            lat00 = float(ds.variables["XLAT"][0, 0, 0])
            lon00 = float(ds.variables["XLONG"][0, 0, 0])
        moving = False
        if last != first:
            with netcdf4.Dataset(last) as ds:
                moving = (abs(float(ds.variables["XLAT"][0, 0, 0]) - lat00) > 1e-3
                          or abs(float(ds.variables["XLONG"][0, 0, 0]) - lon00) > 1e-3)
        info["moving"] = moving
    return doms


def in_domain_interior(frame_path, lat, lon, bdy_width):
    """True iff (lat, lon) maps (wrf.ll_to_xy, this frame's own projection) to a grid cell that
    is at least bdy_width cells from every edge of the domain -- by grid index, not a bbox."""
    from post.caseinfo import _interior_bounds
    netcdf4 = _netcdf4()
    with netcdf4.Dataset(frame_path) as ds:
        nx, ny = len(ds.dimensions["west_east"]), len(ds.dimensions["south_north"])
        xy = _wrf().ll_to_xy(ds, float(lat), float(lon), meta=False, as_int=True)
    x, y = int(xy[0]), int(xy[1])
    bounds = _interior_bounds(ny, nx, bdy_width)
    if bounds is None:
        return False
    j0, j1, i0, i1 = bounds
    return i0 <= x < i1 and j0 <= y < j1


def _interp_position(points, when):
    """(lat, lon) of a time-sorted point list at `when` (linear in time; nearest end point when
    outside the list by <= 3 h, else None)."""
    if not points:
        return None
    if when <= points[0]["time"]:
        p = points[0]
        return (p["lat"], p["lon"]) if (p["time"] - when).total_seconds() <= 3 * 3600 else None
    if when >= points[-1]["time"]:
        p = points[-1]
        return (p["lat"], p["lon"]) if (when - p["time"]).total_seconds() <= 3 * 3600 else None
    for a, b in zip(points, points[1:]):
        if a["time"] <= when <= b["time"]:
            span = (b["time"] - a["time"]).total_seconds()
            w = 0.0 if span == 0 else (when - a["time"]).total_seconds() / span
            return (a["lat"] + w * (b["lat"] - a["lat"]), a["lon"] + w * (b["lon"] - a["lon"]))
    return None


def _pick_seeded_track(tracks, ref_points, tol_km):
    """Of the stitched tracks found on a nest, the one that agrees longest with the coarser
    track (fixes within tol_km of it at the same time); ties -> the longer track. None if no
    fix of any track is within tol_km. This is the parent-domain seed: a nest-only minimum
    that never coincides with the parent's storm cannot be picked."""
    best, best_key = None, (0, 0)
    for tr in tracks:
        n_ok = 0
        for pt in tr:
            pos = _interp_position(ref_points, pt["time"])
            if pos is not None and float(tm.dist_km(pos[0], pos[1], pt["lat"], pt["lon"])) <= tol_km:
                n_ok += 1
        if n_ok and (n_ok, len(tr)) > best_key:
            best, best_key = tr, (n_ok, len(tr))
    return best


def finest_domain_track(run_dir, parent_track, doms, parent_dom, work_dir, bdy_width,
                        tol_km=FINE_TOL_KM, smooth_km=SMOOTH_RADIUS_KM):
    """Per parent-track time, the storm on the finest domain whose interior contains it.

    parent_track: the coarsest domain's stitched track (the seed). doms: run_domains().
    Every nest (coarse -> fine) is tracked with TempestExtremes, seeded by the coarser result
    (_pick_seeded_track), and a nest fix replaces the coarser one only if (1) it is within
    tol_km of the coarser position at that time, (2) that domain has a frame at that time and
    (3) the fix lies in the frame's interior (in_domain_interior). Otherwise the coarser fix
    stays.

    Returns (records, notes_info): records = one dict per parent-track time {time, lat, lon,
    pmin_hpa, vmax_ms, domain, finest_available, reason} where reason is "" when the finest
    available domain was used, else why not; notes_info = {"skipped": {dom: reason}}."""
    from analysis import tc_track  # noqa: F401  (import error surfaces here, not mid-loop)

    merged = {pt["time"]: dict(pt, domain=parent_dom, reason="") for pt in parent_track}
    skipped = {}
    for dom in sorted(d for d in doms if d > parent_dom):
        info = doms[dom]
        times = sorted(datetime.strptime(t, WRF_TIME_FMT) for t in info["frames"])
        if info["moving"]:
            skipped[dom] = "moving nest (XLAT/XLONG change between frames): not tracked"
            continue
        if len(times) < 3:
            skipped[dom] = "fewer than 3 output frames"
            continue
        span_h = (times[-1] - times[0]).total_seconds() / 3600.0
        from analysis.tc_track import TCTrackParams
        params = TCTrackParams(mintime_hours=max(1.0, min(24.0, span_h / 2.0)))
        try:
            tracks = _tracks_for_domain(run_dir, dom, os.path.join(work_dir, "d%02d" % dom),
                                        params=params, smooth_km=smooth_km)
        except Exception as e:      # nest tracker failure must not lose the parent track
            skipped[dom] = "tracking failed: %s" % str(e).splitlines()[0]
            continue
        ref = sorted(merged.values(), key=lambda p: p["time"])
        chosen = _pick_seeded_track(tracks, ref, tol_km)
        by_time = {pt["time"]: pt for pt in (chosen or [])}
        for when, cur in merged.items():
            ts = when.strftime(WRF_TIME_FMT)
            frame = info["frames"].get(ts)
            if frame is None:
                cur.setdefault("_reasons", {})[dom] = "no d%02d frame at this time" % dom
                continue
            pt = by_time.get(when)
            if pt is None:
                cur.setdefault("_reasons", {})[dom] = "no d%02d storm fix" % dom
                continue
            if float(tm.dist_km(cur["lat"], cur["lon"], pt["lat"], pt["lon"])) > tol_km:
                cur.setdefault("_reasons", {})[dom] = "d%02d minimum not the parent storm" % dom
                continue
            if not in_domain_interior(frame, pt["lat"], pt["lon"], bdy_width):
                cur.setdefault("_reasons", {})[dom] = "storm outside d%02d interior" % dom
                continue
            merged[when] = dict(pt, domain=dom, reason="", _reasons=cur.get("_reasons", {}))
    records = []
    for when in sorted(merged):
        rec = merged[when]
        reasons = rec.pop("_reasons", {})
        avail = [d for d in doms if d >= parent_dom
                 and when.strftime(WRF_TIME_FMT) in doms[d]["frames"] and d not in skipped]
        rec["finest_available"] = max(avail) if avail else parent_dom
        if rec["domain"] < rec["finest_available"]:
            rec["reason"] = reasons.get(rec["finest_available"], "finer domain not used")
        records.append(rec)
    return records, dict(skipped=skipped)


def _bdeck_fixes(bdeck_path):
    """[(valid time, lat, lon)] from an ATCF b-deck, one per time (the file repeats a time once per
    wind-radius threshold). [] when there is no b-deck."""
    from datetime import timedelta
    if not bdeck_path or not os.path.exists(bdeck_path):
        return []
    by_time = {}
    for r in atcf.parse_file(bdeck_path):
        by_time.setdefault(r["time"] + timedelta(hours=r["tau_h"] or 0), (r["lat"], r["lon"]))
    return [(t, ll[0], ll[1]) for t, ll in sorted(by_time.items())]


def hourly_step_stats(points, spinup_h=6.0, bdeck_points=None):
    """Track smoothness after the spin-up (fixes at least spinup_h after the first), pre-registered
    2026-09-29 for the smoothed-SLP tracking:

      median_km_h       median great-circle step per hour
      bt_speed_km_h     best-track mean translation speed over the same period (path length /
                        elapsed time of the 6-hourly b-deck fixes inside it); None without b-deck
      speed_ratio       median_km_h / bt_speed_km_h        (criterion a: <= 1.5)
      tortuosity        path length / net displacement     (criterion b: <= 1.3)
      max_km_h, max_at, ratio (max / median)               (informational)
      n                 number of steps
    None with fewer than 2 fixes after the spin-up."""
    import numpy as np
    pts = sorted(points, key=lambda p: p["time"])
    if len(pts) < 2:
        return None
    t0 = pts[0]["time"]
    kept = [p for p in pts if (p["time"] - t0).total_seconds() >= spinup_h * 3600]
    if len(kept) < 2:
        return None
    steps, path = [], 0.0
    for a, b in zip(kept, kept[1:]):
        dt = (b["time"] - a["time"]).total_seconds() / 3600.0
        if dt <= 0:
            continue
        km = float(tm.dist_km(a["lat"], a["lon"], b["lat"], b["lon"]))
        path += km
        steps.append((km / dt, b["time"]))
    if not steps:
        return None
    vals = np.array([v for v, _ in steps])
    med = float(np.median(vals))
    k = int(np.argmax(vals))
    net = float(tm.dist_km(kept[0]["lat"], kept[0]["lon"], kept[-1]["lat"], kept[-1]["lon"]))
    bt = [f for f in (bdeck_points or []) if kept[0]["time"] <= f[0] <= kept[-1]["time"]]
    bt_speed = None
    if len(bt) >= 2 and bt[-1][0] > bt[0][0]:
        bt_path = sum(float(tm.dist_km(a[1], a[2], b[1], b[2])) for a, b in zip(bt, bt[1:]))
        bt_speed = bt_path / ((bt[-1][0] - bt[0][0]).total_seconds() / 3600.0)
    return dict(n=len(vals), median_km_h=med, max_km_h=float(vals[k]), max_at=steps[k][1],
                ratio=float(vals[k] / med) if med > 0 else float("inf"),
                tortuosity=path / net if net > 0 else float("inf"),
                bt_speed_km_h=bt_speed,
                speed_ratio=(med / bt_speed) if bt_speed else None)


def format_track_steps(label, st):
    """One printable line for hourly_step_stats() (used by `wtune post tc` and `wtune post recipe`)."""
    if not st:
        return "track steps %s: not enough fixes after the 6 h spin-up" % label
    bt = ("%.1f km/h best-track speed, median/best-track %.2f (<=1.5)"
          % (st["bt_speed_km_h"], st["speed_ratio"]) if st["bt_speed_km_h"]
          else "no best-track speed (no b-deck)")
    return ("track steps %s (after 6 h spin-up, n=%d): median %.1f km/h, %s; tortuosity %.2f (<=1.3); "
            "max %.1f km/h at %s, max/median %.2f (info)"
            % (label, st["n"], st["median_km_h"], bt, st["tortuosity"], st["max_km_h"],
               st["max_at"].strftime("%m-%d %H:%MZ"), st["ratio"]))


def _ranges(times, step_h):
    """Compress a sorted datetime list into 'YYYY-MM-DD HH:MM-HH:MM' style ranges."""
    if not times:
        return ""
    out, start, prev = [], times[0], times[0]
    for t in times[1:] + [None]:
        if t is not None and (t - prev).total_seconds() <= step_h * 3600 * 1.5:
            prev = t
            continue
        out.append(start.strftime("%m-%d %H:%MZ") if start == prev else
                   "%s..%s" % (start.strftime("%m-%d %H:%MZ"), prev.strftime("%m-%d %H:%MZ")))
        if t is not None:
            start = prev = t
    return ", ".join(out)


def tracking_notes(label, records, doms, skipped=None):
    """User-facing hints about the tracking grid, as a list of strings (empty when the storm was
    followed on the finest domain throughout and that grid is fine enough).

    (a) fallback to a coarser domain at some times (which times, and why);
    (b) finest grid coarser than GRID_LIMITED_DX_KM -- position and intensity grid-limited
        (worded differently when the case has no nest at all);
    (c) storm left a nest's interior after having been in it."""
    notes = []
    if not records:
        return notes
    step_h = 1.0
    if len(records) > 1:
        step_h = max(1e-6, (records[1]["time"] - records[0]["time"]).total_seconds() / 3600.0)
    dx = {d: info["dx_km"] for d, info in doms.items()}
    used_fine = max(r["domain"] for r in records)

    for dom, why in (skipped or {}).items():
        notes.append("%s: d%02d (%.0f km) was not used for tracking: %s." % (label, dom, dx[dom], why))

    fb = [r for r in records if r["domain"] < r["finest_available"]]
    if fb:
        by_dom = {}
        for r in fb:
            by_dom.setdefault((r["domain"], r["finest_available"]), []).append(r)
        for (got, want), rs in sorted(by_dom.items()):
            reasons = sorted({r["reason"] for r in rs})
            notes.append(
                "%s: tracking fell back to coarser d%02d (%.0f km) instead of d%02d (%.0f km) at "
                "%d of %d output times: %s (%s)."
                % (label, got, dx[got], want, dx[want], len(rs), len(records),
                   _ranges([r["time"] for r in rs], step_h), "; ".join(reasons)))
        # (c) storm left a nest interior after having been inside it
        for want in sorted({w for (_g, w) in by_dom}):
            inside = [r["time"] for r in records if r["domain"] == want]
            if not inside:
                continue
            gone = [r["time"] for r in fb if r["finest_available"] == want
                    and r["time"] > min(inside) and r["reason"].startswith("storm outside")]
            if gone:
                notes.append("%s: the storm left the d%02d interior (spec_bdy_width cells excluded) "
                             "from %s on; position/intensity after that are on the coarser grid."
                             % (label, want, min(gone).strftime("%Y-%m-%d %H:%MZ")))
    if dx[used_fine] > GRID_LIMITED_DX_KM:
        g = dx[used_fine]
        if len(doms) == 1:
            notes.append("%s: no nested domain -- the track is on a %.0f km grid, coarser than %.0f km: "
                         "position and intensity (MSLP, Vmax) are grid-limited (centre snaps to grid "
                         "cells; Vmax/MSLP underestimated)." % (label, g, GRID_LIMITED_DX_KM))
        else:
            notes.append("%s: finest domain used for tracking is %.0f km, coarser than %.0f km: position "
                         "and intensity (MSLP, Vmax) are grid-limited." % (label, g, GRID_LIMITED_DX_KM))
    return notes


def model_quadrant_radii(run_dir, track, domain=1):
    """Per-lead 34/50/64 kt quadrant wind radii (tc_metrics.quadrant_radii) from this
    domain's U10/V10, centred on the TempestExtremes-tracked point at each lead. Returns
    a list of {"time", "lead_h", radii dict...} for leads with a frame present."""
    netcdf4 = _netcdf4()
    import numpy as np

    if not track:
        return []
    t0 = min(datetime.strptime(t, WRF_TIME_FMT) for t in track)
    rows = []
    for time_str, pt in sorted(track.items()):
        path = os.path.join(run_dir, "wrfout_d%02d_%s" % (domain, time_str))
        if not os.path.exists(path):
            continue
        with netcdf4.Dataset(path) as ds:
            lat = np.asarray(ds.variables["XLAT"][0])
            lon = np.asarray(ds.variables["XLONG"][0])
            speed = np.hypot(ds.variables["U10"][0][...], ds.variables["V10"][0][...])
        rr = tm.dist_km(pt["lat"], pt["lon"], lat, lon)
        lead_h = (datetime.strptime(time_str, WRF_TIME_FMT) - t0).total_seconds() / 3600.0
        row = dict(time=time_str, lead_h=lead_h)
        row.update(tm.quadrant_radii(pt["lat"], pt["lon"], lat, lon, rr, speed))
        rows.append(row)
    return rows


def _radii_kt_for_lead(quadrant_row):
    radii_kt = {}
    for th in atcf.RAD_THRESHOLDS_KT:
        key = "r%d" % th
        quad = tuple(quadrant_row["%s_%s_km" % (key, q)] / NMI_TO_KM
                    for q in tm.QUADRANT_NAMES)
        if any(quad):
            radii_kt[th] = quad
    return radii_kt


def build_adeck(chosen_track, quadrant_rows, out_path, basin, cy, tech, storm_name=""):
    """chosen_track: TempestExtremes track (list of {time, lat, lon, pmin_hpa, vmax_ms}).
    quadrant_rows: model_quadrant_radii() output, keyed by time to attach radii. Writes
    one ATCF a-deck line per (threshold reached) via post.atcf.write_lines."""
    t0 = min(pt["time"] for pt in chosen_track)
    by_time = {r["time"]: r for r in quadrant_rows}
    rows = []
    for pt in sorted(chosen_track, key=lambda p: p["time"]):
        time_str = pt["time"].strftime(WRF_TIME_FMT)
        vmax_kt = (pt["vmax_ms"] or 0.0) / KT_TO_MS
        # ATCF field 3 is the forecast's initialization time, the same on every line; the
        # valid time is init + tau. Writing the valid time here made every hour its own
        # "forecast" and tc_pairs paired each one at lead 0.
        rows.append(dict(time=t0, tau_h=(pt["time"] - t0).total_seconds() / 3600.0,
                         lat=pt["lat"], lon=pt["lon"], vmax_kt=vmax_kt,
                         mslp_hpa=pt["pmin_hpa"] or 0,
                         radii_kt=_radii_kt_for_lead(by_time[time_str])
                         if time_str in by_time else {}))
    return atcf.write_lines(rows, out_path, basin=basin, cy=cy, tech=tech, stormname=storm_name)


# --------------------------------------------------------------------------------------
# orchestration + figures
# --------------------------------------------------------------------------------------

def run_tc_report(run_dirs, labels, hurdat2_path, storm_id, out_dir, ebtrk_path=None,
                  bdeck_path=None, max_dom=None, smooth_km=SMOOTH_RADIUS_KM):
    """Full `wtune post tc` pipeline over one or more runs.

    Always produced (needs only TempestExtremes + wrf-python): one ATCF a-deck per run
    (adeck_<label>.dat), quadrant_<label>.csv, obs.json (HURDAT2 (+EBTRK) as parsed), and
    track_map.png (drawn straight from the a-deck/b-deck files, not MET output).

    Produced only if MET (tc_pairs/tc_stat/tc_rmw) is installed: tc_pairs.tcst,
    tc_stat_track.stat/tc_stat_wind.stat and the error/RMW figures drawn from them. A
    missing MET tool does not fail the whole report -- its ToolNotFoundError is caught and
    appended to the returned "warnings" list, matching the "fail with a clear message,
    don't invent a fallback statistic" contract for a tool that genuinely isn't installed.

    Also written: track_domains_<label>.csv (per-time domain the storm was tracked on, dx,
    fallback flag + reason), tracking_notes.json ({label: [hint strings]}, read by
    post/recipes/tc_extra.py for the HTML report).

    Returns a summary dict: {"out_dir", "adecks", "bdeck", "figures", "warnings", "notes",
    "track_csvs", "track_steps" ({label: hourly_step_stats()})}. "warnings" = tool/obs problems; "notes" = tracking-grid hints
    (tracking_notes(): coarser-domain fallback, grid-limited, storm left the nest).
    """
    os.makedirs(out_dir, exist_ok=True)
    basin, cy, year = storm_id_parts(storm_id)
    hurdat2_points = parse_hurdat2_full(hurdat2_path, storm_id)
    obs = build_obs(hurdat2_path, storm_id, ebtrk_path)
    with open(os.path.join(out_dir, "obs.json"), "w") as fh:
        json.dump(obs, fh, indent=2)

    warnings = []
    try:
        bdeck = resolve_bdeck(storm_id, out_dir, bdeck_path=bdeck_path,
                              hurdat2_points=hurdat2_points)
    except TCError as e:
        warnings.append(str(e))
        bdeck = None

    adecks, precise, notes, track_csvs, dom_dx, steps = {}, {}, [], {}, {}, {}
    notes_by_label = {}
    for run_dir, label in zip(run_dirs, labels):
        doms = run_domains(run_dir, max_dom)
        if not doms:
            raise TCError("no wrfout_dNN_* files found in %s" % run_dir)
        parent_dom = min(doms)
        for d, info in doms.items():
            dom_dx[d] = info["dx_km"]
        start = _run_start(run_dir, parent_dom)
        best_guess = min(hurdat2_points, key=lambda r: abs((r["time"] - start).total_seconds()))
        work = os.path.join(out_dir, "_track_%s" % label)
        parent_track = track_this_run(run_dir, best_guess["lat"], best_guess["lon"], work,
                                      domain=parent_dom, smooth_km=smooth_km)
        from post.caseinfo import spec_bdy_width
        chosen, info = finest_domain_track(run_dir, parent_track, doms, parent_dom, work,
                                           spec_bdy_width(run_dir), smooth_km=smooth_km)
        run_notes = tracking_notes(label, chosen, doms, info["skipped"])
        steps[label] = hourly_step_stats(chosen, bdeck_points=_bdeck_fixes(bdeck))
        notes += run_notes
        notes_by_label[label] = run_notes
        _write_csv(os.path.join(out_dir, "track_domains_%s.csv" % label), [
            dict(time=r["time"].strftime(WRF_TIME_FMT), domain="d%02d" % r["domain"],
                 dx_km=doms[r["domain"]]["dx_km"], lat=r["lat"], lon=r["lon"],
                 pmin_hpa=r["pmin_hpa"], vmax_ms=r["vmax_ms"],
                 finest_available="d%02d" % r["finest_available"],
                 fallback=int(r["domain"] < r["finest_available"]), reason=r["reason"])
            for r in chosen])
        track_csvs[label] = os.path.join(out_dir, "track_domains_%s.csv" % label)
        # Quadrant radii stay on the PARENT domain: a nest is often smaller than the 600 km
        # radius search and would clip R34/R50/R64; only the centre comes from the finest track.
        quadrant_rows = model_quadrant_radii(run_dir, track_dict(chosen), domain=parent_dom)
        _write_csv(os.path.join(out_dir, "quadrant_%s.csv" % label), quadrant_rows)
        adeck_path = build_adeck(chosen, quadrant_rows,
                                 os.path.join(out_dir, "adeck_%s.dat" % label),
                                 basin, cy, tech=label)
        adecks[label] = adeck_path
        precise[label] = [(pt["time"], (pt["lat"], pt["lon"]), pt["domain"]) for pt in chosen]
    with open(os.path.join(out_dir, "tracking_notes.json"), "w") as fh:
        json.dump(notes_by_label, fh, indent=2)

    figures = [_track_map_figure(out_dir, precise, bdeck, storm_id, dom_dx)]

    if bdeck is not None:
        tcsts = {}
        for run_dir, label in zip(run_dirs, labels):
            met_dir = os.path.join(out_dir, "met_%s" % label)
            try:
                tcsts[label] = met_tc.run_tc_pairs(adecks[label], bdeck, met_dir,
                                                   model=label[:4].upper())
            except tools.ToolNotFoundError as e:
                warnings.append(str(e))
                break   # same tool missing for every remaining run -- one warning is enough
            except (met_tc.METError, tools.ToolError) as e:
                warnings.append("%s: tc_pairs: %s" % (label, e))
                continue
            try:
                figures.append(_rmw_figure(run_dir, adecks[label], met_dir, out_dir, label))
            except (met_tc.METError, tools.ToolError, tools.ToolNotFoundError) as e:
                warnings.append("%s: tc_rmw: %s" % (label, e))
        if tcsts:
            figures += _error_figures(out_dir, tcsts)
    else:
        warnings.append("no b-deck available -- skipping MET tc_pairs/tc_rmw")

    return dict(out_dir=out_dir, adecks=adecks, bdeck=bdeck, figures=[f for f in figures if f],
               warnings=warnings, notes=notes, track_csvs=track_csvs, track_steps=steps)


def _run_start(run_dir, domain=1):
    import glob
    prefix = "wrfout_d%02d_" % domain
    hits = sorted(glob.glob(os.path.join(run_dir, prefix + "*")))
    if not hits:
        raise TCError("no %s* files found in %s" % (prefix, run_dir))
    return datetime.strptime(os.path.basename(hits[0])[len(prefix):], WRF_TIME_FMT)


def _detect_max_dom(run_dir):
    import glob
    import re
    doms = set()
    for p in glob.glob(os.path.join(run_dir, "wrfout_d*_*")):
        m = re.search(r"wrfout_d(\d+)_", os.path.basename(p))
        if m:
            doms.add(int(m.group(1)))
    if not doms:
        raise TCError("no wrfout_dNN_* files found in %s" % run_dir)
    return max(doms)


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _track_map_figure(out_dir, model_tracks, bdeck, storm_id, dom_dx=None):
    """Model tracks at the tracker's full precision (model_tracks: {label: [(valid time,
    (lat, lon), domain)]} -- not the a-deck, whose 0.1-degree positions make an hourly track
    zig-zag) and the best track over the simulated window +-6 h, on a Natural Earth
    coastline when its data is cached locally. Each segment is coloured by the domain it was
    tracked on (the coarser of its two end points: it is only as good as that grid), with a
    legend; runs are told apart by line style."""
    from datetime import timedelta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from post import plotting

    dom_dx = dom_dx or {}

    def points(path):
        by_valid = {}
        for r in atcf.parse_file(path):
            by_valid.setdefault(r["time"] + timedelta(hours=r["tau_h"] or 0), (r["lat"], r["lon"]))
        return sorted(by_valid.items())

    tracks = {label: sorted(pts) for label, pts in model_tracks.items()}
    valid = [p[0] for pts in tracks.values() for p in pts]
    lo, hi = (min(valid) - timedelta(hours=6), max(valid) + timedelta(hours=6)) if valid else (None, None)
    best = [(t, ll) for t, ll in points(bdeck) if lo is None or lo <= t <= hi] if bdeck else []

    plotting._set_cartopy_data_dir()
    try:
        import cartopy.crs as ccrs
        coast = plotting._natural_earth_ready("50m")
    except ImportError:
        coast = False
    if coast:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        # coastline/lakes/borders/states come from the shared base map after the tracks set
        # the extent (below)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.8")
        gl.top_labels = gl.right_labels = False
        kw = dict(transform=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        kw = {}
    handles = []
    if best:
        # best track black with 6-hourly markers and a date label at 00 UTC (NHC style)
        ax.plot([ll[1] for _, ll in best], [ll[0] for _, ll in best], "k-", lw=1.4, zorder=1, **kw)
        six = [(t, ll) for t, ll in best if t.hour % 6 == 0 and t.minute == 0]
        ax.plot([ll[1] for _, ll in six], [ll[0] for _, ll in six], "ko", markersize=3.5,
                zorder=1, **kw)
        for t, ll in six:
            if t.hour == 0:
                ax.text(ll[1] + 0.15, ll[0], t.strftime("%d/00Z"), fontsize=7, zorder=4, **kw)
        handles.append(Line2D([], [], color="k", marker="o", markersize=3,
                              label="best track (6-hourly)"))
    cmap = plt.get_cmap("tab10")
    doms_seen = sorted({p[2] for pts in tracks.values() for p in pts})
    color = {d: cmap(k % 10) for k, d in enumerate(doms_seen)}
    styles = ["-", "--", ":", "-."]
    for n, (label, pts) in enumerate(tracks.items()):
        ls = styles[n % len(styles)]
        for a, b in zip(pts, pts[1:]):
            d = min(a[2], b[2])
            ax.plot([a[1][1], b[1][1]], [a[1][0], b[1][0]], ls, color=color[d], linewidth=1.6,
                    zorder=2, **kw)
        for d in doms_seen:
            # markers at the 6-hourly synoptic times only (the line carries the full track)
            sel = [p for p in pts if p[2] == d and p[0].hour % 6 == 0 and p[0].minute == 0]
            if sel:
                ax.plot([p[1][1] for p in sel], [p[1][0] for p in sel], "o", color=color[d],
                        markersize=3.5, zorder=3, **kw)
        if len(tracks) > 1:
            handles.append(Line2D([], [], color="0.3", linestyle=ls, label=label))
    for d in doms_seen:
        dx = dom_dx.get(d)
        handles.append(Line2D([], [], color=color[d], linewidth=2,
                              label="tracked on d%02d%s" % (d, " (%.0f km)" % dx if dx else "")))
    if coast:
        # shared base map over the plotted track extent: coastline, lakes, borders, states
        from post.basemap import add_base_map
        ax.autoscale_view()
        add_base_map(ax, ccrs, terrain=False, nest_rings=[], caption=False, labels=False)
    t0 = min(valid) if valid else None
    ax.set_title("%s track, model init %s\n(segments coloured by tracking domain; markers 6-hourly)"
                 % (storm_id, t0.strftime("%Y-%m-%d %HZ") if t0 else "?"), fontsize=10)
    ax.legend(handles=handles, fontsize=8)
    path = os.path.join(out_dir, "track_map.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _surface_pressure_profile(frame_path, lat0, lon0, rng_km):
    """Azimuthal-mean surface pressure (hPa) at each storm-centred range in rng_km, from one wrfout
    frame's PSFC. Self-computed because tc_rmw writes no surface field: it is only used to draw
    where the pressure-level data ends (below ground / sea surface) in the RMW figure."""
    import numpy as np
    with _netcdf4().Dataset(frame_path) as ds:
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        psfc = np.asarray(ds.variables["PSFC"][0]) / 100.0
    rr = tm.dist_km(lat0, lon0, lat, lon)
    rng = np.asarray(rng_km, dtype=float)
    half = np.gradient(rng) / 2.0 if len(rng) > 1 else np.array([5.0])
    out = np.full(rng.shape, np.nan)
    for i, r in enumerate(rng):
        sel = (rr >= r - half[i]) & (rr < r + half[i])
        if sel.any():
            out[i] = float(psfc[sel].mean())
    return out


def _fill_down_from_lowest_valid(vt, plev):
    """vt[p, r] with NaN where the pressure level is below the surface. Each column's NaN cells at
    higher pressure than its lowest valid level take that level's value. contourf drops every grid
    quad with a NaN corner (the terrain figures' lesson, post/recipes/terrain.py
    _fill_below_valid), which would leave a blank band above the real surface; the true
    below-ground region is drawn afterwards as a hatched polygon over these constant-filled cells."""
    import numpy as np
    out = vt.copy()
    order = np.argsort(-np.asarray(plev))            # highest pressure (lowest level) first
    for j in range(out.shape[1]):
        col = out[:, j]
        valid = [i for i in order if np.isfinite(col[i])]
        if valid:
            low = valid[0]
            for i in order:
                if plev[i] > plev[low] and not np.isfinite(col[i]):
                    col[i] = col[low]
    return out


def _rmw_figure(run_dir, adeck_path, met_dir, out_dir, label, every_h=6):
    """Radius-pressure cross-section of the azimuthal-mean tangential (filled, contourf) and radial
    (contours; negative = inflow) wind at the last lead, both computed by MET tc_rmw on its
    storm-centred grid from d01 frames every `every_h` hours (met_tc.regrid_for_tc_rmw), on
    met_tc.PRESSURE_LEVELS_HPA (every 50 hPa, 1000-100). Pressure levels below the surface are
    filled downward with the lowest valid value and covered by a grey hatched region bounded by
    the azimuthal-mean surface pressure of that frame."""
    import glob
    import netCDF4
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    frames = []
    for f in sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*"))):
        when = datetime.strptime(os.path.basename(f)[len("wrfout_d01_"):], WRF_TIME_FMT)
        if when.minute == 0 and when.hour % every_h == 0:
            frames.append((when, f))
    if not frames:
        return None
    data = met_tc.regrid_for_tc_rmw([f for _, f in frames], os.path.join(met_dir, "rmw_input.grb"))
    rmw_nc = met_tc.run_tc_rmw(data, adeck_path, os.path.join(met_dir, "rmw_output.nc"),
                               valid_hours=range(0, 24, every_h))

    with netCDF4.Dataset(rmw_nc) as ds:
        rng = np.asarray(ds["range"][:])
        plev = np.asarray(ds["pressure"][:])
        # (track_point, pressure, range, azimuth); VT/VR from tc_rmw's sampled u/v, not its
        # own VT/VR -- see met_tc.storm_relative_winds for why
        u = np.ma.filled(ds["UGRD"][:].astype(float), np.nan)
        v = np.ma.filled(ds["VGRD"][:].astype(float), np.nan)
        # MET's fill (~9.97e36) below ground is finite: NaN it explicitly
        u[np.abs(u) > 1e30] = np.nan
        v[np.abs(v) > 1e30] = np.nan
        vt, vr = met_tc.storm_relative_winds(u, v, np.asarray(ds["azimuth"][:]))
        lead_h = np.asarray(ds["lead_time_sec"][:]) / 3600.0
    ok = [k for k in range(vt.shape[0]) if np.isfinite(vt[k]).any()]
    if not ok or plev.size == 0:
        raise met_tc.METError("tc_rmw output has no finite VT on pressure levels")
    k = ok[-1]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vt_mean = np.nanmean(vt[k], axis=-1)
            vr_mean = np.nanmean(vr[k], axis=-1)
    vt_f = _fill_down_from_lowest_valid(vt_mean, plev)

    # surface pressure profile at the last lead's centre, for the below-ground region
    ps = None
    rows = [r for r in atcf.parse_file(adeck_path)]
    if rows:
        init = rows[0]["time"]
        valid = init + __import__("datetime").timedelta(hours=float(lead_h[k]))
        when, fpath = min(frames, key=lambda wf: abs((wf[0] - valid).total_seconds()))
        row = min(rows, key=lambda r: abs(r["tau_h"] - lead_h[k]))
        try:
            ps = _surface_pressure_profile(fpath, row["lat"], row["lon"], rng)
        except (KeyError, OSError):
            ps = None

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    # Vt shaded every 5 m/s on a sequential map, Vr lines every 2 m/s (blue dashed inflow,
    # red solid outflow), height on a second axis (decision 0064)
    vt_hi = float(np.nanmax(vt_f))
    levels = np.arange(5.0 * np.floor(min(float(np.nanmin(vt_f)), 0.0) / 5.0), vt_hi + 5.0, 5.0)
    cf = ax.contourf(rng, plev, vt_f, levels=levels, cmap="YlOrRd", extend="both")
    fig.colorbar(cf, ax=ax, label="azimuthal-mean tangential wind (m/s, every 5)", pad=0.14)
    # radial-wind lines from the masked (unfilled) array: the downward-filled copy would draw
    # contours running vertically through the constant-filled below-surface cells
    vr_m = np.ma.masked_invalid(vr_mean)
    for lv, col, ls in ((np.arange(-30.0, 0.0, 2.0), "tab:blue", "dashed"),
                        (np.arange(2.0, 32.0, 2.0), "tab:red", "solid")):
        cs = ax.contour(rng, plev, vr_m, levels=lv, colors=col, linestyles=ls, linewidths=0.8)
        ax.clabel(cs, fontsize=7, fmt="%d")
    ymax = float(np.max(plev))
    handles = []
    if ps is not None and np.isfinite(ps).any():
        below = np.clip(np.where(np.isfinite(ps), ps, ymax), None, ymax)
        ax.fill_between(rng, below, ymax, facecolor="0.85", edgecolor="0.5", hatch="///",
                        linewidth=0, zorder=5)
        ax.plot(rng, below, color="0.35", linewidth=0.8, zorder=6)
        handles.append(Patch(facecolor="0.85", edgecolor="0.5", hatch="///",
                             label="below the surface (mean PSFC)"))
    ax.set_ylim(ymax, float(np.min(plev)))
    ax.set_xlim(float(rng.min()), float(rng.max()))
    if handles:
        legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.16),
                           fontsize=8, frameon=False)
    ax.set_xlabel("radius from storm centre (km)")
    ax.set_ylabel("pressure (hPa)")
    # approximate height (ICAO standard atmosphere) for readers used to radius-height plots
    sec = ax.secondary_yaxis("right", functions=(
        lambda pp: 44.3308 * (1.0 - (np.clip(pp, 1e-3, None) / 1013.25) ** 0.190263),
        lambda zz: 1013.25 * (1.0 - np.asarray(zz) / 44.3308) ** (1.0 / 0.190263)))
    sec.set_ylabel("approx. height (km, standard atmosphere)")
    ax.set_title("%s, F%03d: azimuthal-mean tangential wind (shaded)\n"
                 "radial wind every 2 m/s: blue dashed = inflow, red = outflow"
                 % (label, int(round(lead_h[k]))), fontsize=10)
    path = os.path.join(out_dir, "%s_rmw_radius_height.png" % label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _error_figures(out_dir, tcsts):
    """Track error, central pressure, maximum wind and mean R34 against lead, every run
    overlaid with the best track -- read straight from each run's tc_pairs .tcst (no
    re-derivation of what tc_pairs computed). Units: TK_ERR/radii nmi -> km, winds kt -> m/s."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def lead_h(v):
        v = v.strip()
        return int(v[:-4]) + int(v[-4:-2]) / 60.0 if len(v) > 4 else float(v)

    def num(r, col, scale=1.0):
        try:
            x = float(r[col])
        except (KeyError, ValueError):
            return None
        return None if x <= -9999 else x * scale

    series = {}
    for label, path in tcsts.items():
        with open(path) as fh:
            rows = met_tc.parse_met_table(fh.read())
        series[label] = sorted(((lead_h(r["LEAD"]), r) for r in rows if "LEAD" in r),
                               key=lambda t: t[0])

    panels = [("TK_ERR", None, NMI_TO_KM, "track error (km)", "track_error"),
              ("AMSLP", "BMSLP", 1.0, "central pressure (hPa)", "mslp"),
              ("AMAX_WIND", "BMAX_WIND", KT_TO_MS, "maximum wind (m/s)", "vmax"),
              ("AAL_WIND_34", "BAL_WIND_34", NMI_TO_KM, "mean 34-kt radius (km)", "r34")]
    figures = []
    for acol, bcol, scale, ylabel, stem in panels:
        fig, ax = plt.subplots(figsize=(7, 4))
        drawn, best = False, None
        for label, rows in series.items():
            pts = [(h, num(r, acol, scale)) for h, r in rows]
            pts = [(h, y) for h, y in pts if y is not None]
            if pts:
                ax.plot(*zip(*pts), "-o", markersize=3, label=label)
                drawn = True
            if bcol and best is None:
                bp = [(h, num(r, bcol, scale)) for h, r in rows]
                best = [(h, y) for h, y in bp if y is not None]
        if best:
            ax.plot(*zip(*best), "k-o", markersize=3, label="best track")
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xlabel("forecast lead (h)")
        ax.set_ylabel(ylabel)
        # NHC verification units on a second axis (nmi for distances, kt for wind)
        if "(km)" in ylabel:
            sec = ax.secondary_yaxis("right", functions=(lambda v: v / 1.852, lambda v: v * 1.852))
            sec.set_ylabel(ylabel.replace("(km)", "(nmi)"))
        elif "(m/s)" in ylabel:
            sec = ax.secondary_yaxis("right", functions=(lambda v: v / KT_TO_MS,
                                                         lambda v: v * KT_TO_MS))
            sec.set_ylabel(ylabel.replace("(m/s)", "(kt)"))
        first = next((rows[0][1] for rows in series.values() if rows), {})
        init_txt = str(first.get("INIT", "?"))
        try:
            from datetime import datetime as _dt
            init_txt = _dt.strptime(init_txt, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %HZ")
        except ValueError:
            pass
        ax.set_title("%s  %s, init %s  (MET TC-Pairs)" % (first.get("STORM_ID", ""), ylabel,
                                                         init_txt), fontsize=10)
        ax.legend()
        path = os.path.join(out_dir, "tc_pairs_%s.png" % stem)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        figures.append(path)
    return figures

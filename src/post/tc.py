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


def track_this_run(run_dir, first_guess_lat, first_guess_lon, work_dir):
    """Runs analysis.diagnostics + analysis.tc_track end to end on this run's d01 files.
    Returns the chosen track (list of {time, lat, lon, pmin_hpa, vmax_ms})."""
    from analysis import diagnostics as diag_mod
    from analysis import tc_track

    os.makedirs(work_dir, exist_ok=True)
    fields_nc = os.path.join(work_dir, "fields.nc")
    diag_mod.wrfout_to_tc_fields(os.path.join(run_dir, "wrfout_d01_*"), fields_nc)
    tracks, _prov = tc_track.detect_and_stitch(fields_nc, work_dir)
    if not tracks:
        raise TCError("no track found in %s" % run_dir)
    chosen = tc_track.select_track(tracks, first_guess_lat, first_guess_lon)
    if chosen is None:
        raise TCError("select_track picked nothing for %s" % run_dir)
    return chosen


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
                  bdeck_path=None, max_dom=None):
    """Full `wtune post tc` pipeline over one or more runs.

    Always produced (needs only TempestExtremes + wrf-python): one ATCF a-deck per run
    (adeck_<label>.dat), quadrant_<label>.csv, obs.json (HURDAT2 (+EBTRK) as parsed), and
    track_map.png (drawn straight from the a-deck/b-deck files, not MET output).

    Produced only if MET (tc_pairs/tc_stat/tc_rmw) is installed: tc_pairs.tcst,
    tc_stat_track.stat/tc_stat_wind.stat and the error/RMW figures drawn from them. A
    missing MET tool does not fail the whole report -- its ToolNotFoundError is caught and
    appended to the returned "warnings" list, matching the "fail with a clear message,
    don't invent a fallback statistic" contract for a tool that genuinely isn't installed.

    Returns a summary dict: {"out_dir", "adecks", "bdeck", "figures", "warnings"}.
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

    adecks, precise = {}, {}
    for run_dir, label in zip(run_dirs, labels):
        dom = max_dom or _detect_max_dom(run_dir)
        start = _run_start(run_dir)
        best_guess = min(hurdat2_points, key=lambda r: abs((r["time"] - start).total_seconds()))
        chosen = track_this_run(run_dir, best_guess["lat"], best_guess["lon"],
                                os.path.join(out_dir, "_track_%s" % label))
        quadrant_rows = model_quadrant_radii(run_dir, track_dict(chosen), domain=1)
        _write_csv(os.path.join(out_dir, "quadrant_%s.csv" % label), quadrant_rows)
        adeck_path = build_adeck(chosen, quadrant_rows,
                                 os.path.join(out_dir, "adeck_%s.dat" % label),
                                 basin, cy, tech=label)
        adecks[label] = adeck_path
        precise[label] = [(pt["time"], (pt["lat"], pt["lon"])) for pt in chosen]

    figures = [_track_map_figure(out_dir, precise, bdeck, storm_id)]

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
               warnings=warnings)


def _run_start(run_dir):
    import glob
    hits = sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*")))
    if not hits:
        raise TCError("no wrfout_d01_* files found in %s" % run_dir)
    return datetime.strptime(os.path.basename(hits[0])[len("wrfout_d01_"):], WRF_TIME_FMT)


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


def _track_map_figure(out_dir, model_tracks, bdeck, storm_id):
    """Model tracks at the tracker's full precision (model_tracks: {label: [(valid time,
    (lat, lon))]} -- not the a-deck, whose 0.1-degree positions make an hourly track
    zig-zag) and the best track over the simulated window +-6 h, on a Natural Earth
    coastline when its data is cached locally."""
    from datetime import timedelta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from post import plotting

    def points(path):
        by_valid = {}
        for r in atcf.parse_file(path):
            by_valid.setdefault(r["time"] + timedelta(hours=r["tau_h"] or 0), (r["lat"], r["lon"]))
        return sorted(by_valid.items())

    tracks = {label: sorted(pts) for label, pts in model_tracks.items()}
    valid = [t for pts in tracks.values() for t, _ in pts]
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
        ax.coastlines("50m", linewidth=0.6, color="0.4")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.8")
        gl.top_labels = gl.right_labels = False
        kw = dict(transform=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        kw = {}
    if best:
        ax.plot([ll[1] for _, ll in best], [ll[0] for _, ll in best], "k-o", label="best track",
                markersize=3, **kw)
    for label, pts in tracks.items():
        ax.plot([ll[1] for _, ll in pts], [ll[0] for _, ll in pts], "-o", label=label,
                markersize=3, **kw)
    ax.set_title("%s track" % storm_id)
    ax.legend()
    path = os.path.join(out_dir, "track_map.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _rmw_figure(run_dir, adeck_path, met_dir, out_dir, label, every_h=6):
    """Radius-pressure cross-section of the azimuthal-mean tangential (filled) and radial
    (contours; negative = inflow) wind at the last lead, both computed by MET tc_rmw on its
    storm-centred grid from d01 frames every `every_h` hours (met_tc.regrid_for_tc_rmw)."""
    import glob
    import netCDF4
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    for f in sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*"))):
        when = datetime.strptime(os.path.basename(f)[len("wrfout_d01_"):], WRF_TIME_FMT)
        if when.minute == 0 and when.hour % every_h == 0:
            frames.append(f)
    if not frames:
        return None
    data = met_tc.regrid_for_tc_rmw(frames, os.path.join(met_dir, "rmw_input.grb"))
    rmw_nc = met_tc.run_tc_rmw(data, adeck_path, os.path.join(met_dir, "rmw_output.nc"),
                               valid_hours=range(0, 24, every_h))

    with netCDF4.Dataset(rmw_nc) as ds:
        rng = np.asarray(ds["range"][:])
        plev = np.asarray(ds["pressure"][:])
        # (track_point, pressure, range, azimuth); VT/VR from tc_rmw's sampled u/v, not its
        # own VT/VR -- see met_tc.storm_relative_winds for why
        u = np.ma.filled(ds["UGRD"][:].astype(float), np.nan)
        v = np.ma.filled(ds["VGRD"][:].astype(float), np.nan)
        vt, vr = met_tc.storm_relative_winds(u, v, np.asarray(ds["azimuth"][:]))
        lead_h = np.asarray(ds["lead_time_sec"][:]) / 3600.0
    ok = [k for k in range(vt.shape[0]) if np.isfinite(vt[k]).any()]
    if not ok or plev.size == 0:
        raise met_tc.METError("tc_rmw output has no finite VT on pressure levels")
    k = ok[-1]
    vt_mean = np.nanmean(vt[k], axis=-1)
    vr_mean = np.nanmean(vr[k], axis=-1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    pc = ax.pcolormesh(rng, plev, vt_mean, cmap="viridis", shading="auto")
    fig.colorbar(pc, ax=ax, label="azimuthal-mean tangential wind (m/s)")
    cs = ax.contour(rng, plev, vr_mean, levels=[-15, -10, -5, 0, 5, 10], colors="w",
                    linewidths=0.8)
    ax.clabel(cs, fontsize=7, fmt="%d")
    ax.invert_yaxis()
    ax.set_xlabel("radius from storm centre (km)")
    ax.set_ylabel("pressure (hPa)")
    ax.set_title("%s, lead %g h: azimuthal-mean tangential wind\n(contours: radial wind, <0 inflow)"
                 % (label, lead_h[k]), fontsize=10)
    path = os.path.join(out_dir, "%s_rmw_radius_height.png" % label)
    fig.tight_layout()
    fig.savefig(path)
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
        ax.set_xlabel("lead (h)")
        ax.set_ylabel(ylabel)
        ax.set_title("MET tc_pairs: %s" % ylabel)
        ax.legend()
        path = os.path.join(out_dir, "tc_pairs_%s.png" % stem)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        figures.append(path)
    return figures

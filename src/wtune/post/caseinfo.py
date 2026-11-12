"""post/caseinfo.py -- case-type detection for `wtune post recipe` (decision 0057 section 1):
multi-label, read straight from namelist.output (preferred: fully resolved, including defaults
the run never overrode) or namelist.input, plus a best-track fix check against HURDAT2 or a
b-deck. This module only reads the fields decision 0057's table names and evaluates the exact
thresholds it states -- no new domain rules are invented here.

Reuses wtune.cli.parse_namelist/first_value (the repo's one namelist reader, stdlib only) rather
than writing a second Fortran-namelist parser, and post/atcf.py + post/tc.py's HURDAT2 reader for
the best-track side, so this module adds no new file-format parsing of its own beyond a tiny
comma-separated-values-per-domain splitter (namelist arrays have no established-tool reader; they
are WRF/WTune-specific, like the missing-frame check in post/inventory.py).
"""
from __future__ import annotations

import glob
import os
from datetime import datetime, timedelta

from wtune import cli as _core
from wtune.post import atcf
from wtune.post.conventions import continuous_lons, lon_in_range
from wtune.post.extract import list_frames, segment_dirs

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"

# common is always produced; the rest are checked and, when detected, appended in this order
# (decision 0057 section 1: "順序：common 在最前面，其他類型依上表的順序"; terrain/moisture were
# added by the 2026-09-29 update, placed next to synoptic/pbl since they are also domain-scale
# 2-D/cross-section diagnostics).
ALL_TYPES_ORDER = ("common", "tc", "convection", "precip", "synoptic", "terrain", "moisture",
                    "cloud", "pbl", "chem", "climate", "ensemble")

SYNOPTIC_EXTENT_KM = 1500.0
SYNOPTIC_RUN_HOURS = 24.0
CONVECTION_DX_KM = 4.0
PBL_LES_DX_KM = 1.0
CLIMATE_RUN_DAYS = 30.0
TERRAIN_RELIEF_M = 1000.0
TERRAIN_PEAK_M = 1500.0


class CaseInfoError(RuntimeError):
    pass


def find_namelist(run_dir):
    """namelist.output is preferred, per decision 0057 section 1 ("from namelist.output/
    namelist.input"): it is fully resolved by WRF itself, so every field the detection table
    needs is present even when namelist.input relied on a compiled-in default. namelist.input
    is the fallback for a run that has not finished (no namelist.output written yet).

    Previously this preferred namelist.input instead, working around a wtune.cli.parse_namelist
    bug where a literal '/' anywhere on a line ended the current namelist group -- false-
    triggering on namelist.output's own machine-dumped string defaults that contain a path,
    e.g. ` PATH_TO_FILES = ~/WRFV3/,` inside &physics, silently truncating that group before
    cu_physics/bl_pbl_physics/ra_lw_physics were ever read. Fixed upstream in wtune/cli.py (a
    group now ends only at a slash standing alone at end of line); see
    tests/test_post_caseinfo.py::test_parse_namelist_keeps_group_past_unquoted_path_value."""
    # a segmented `wtune run --out` dir has no namelist of its own: its first segment's
    # (OUT/seg_0) carries the case's grid/physics/bdy_control unchanged (wtune/run/
    # render_segment_namelist only rewrites times, decomposition and restart settings).
    segs = segment_dirs(run_dir)
    for d in [run_dir] + segs[:1]:
        for name in ("namelist.output", "namelist.input"):
            path = os.path.join(d, name)
            if os.path.exists(path):
                return path
    raise CaseInfoError("no namelist.input or namelist.output found in %s" % run_dir)


def _num_list(raw, n):
    """Comma-separated per-domain namelist values -> a length-n list, coerced with
    wtune.cli.first_value (handles WRF's '.true./.false.'/int/float tokens). A short list is
    padded by repeating its last value, matching WRF's own namelist semantics for a value that
    was only specified for domain 1."""
    if raw is None:
        return [None] * n
    tokens = [t.strip() for t in raw.split(",")]
    out = [(_core.first_value(t) if t else None) for t in tokens]
    while len(out) < n:
        out.append(out[-1] if out else None)
    return out[:n]


def domain_physics(run_dir):
    """Per-domain grid/physics facts the detection table needs. Returns (max_dom, rows),
    rows[i] = {"domain": i+1, "dx_km", "dy_km", "e_we", "e_sn", "cu_physics",
    "bl_pbl_physics", "chem_opt"}."""
    nl = _core.parse_namelist(find_namelist(run_dir))
    dom = nl.get("domains", {})
    phys = nl.get("physics", {})
    chem = nl.get("chem", {})
    max_dom = int(_core.first_value(dom.get("max_dom")) or 1)

    dx = _num_list(dom.get("dx"), max_dom)
    dy = _num_list(dom.get("dy"), max_dom)
    e_we = _num_list(dom.get("e_we"), max_dom)
    e_sn = _num_list(dom.get("e_sn"), max_dom)
    cu = _num_list(phys.get("cu_physics"), max_dom)
    bl = _num_list(phys.get("bl_pbl_physics"), max_dom)
    # chem_opt lives in &chem on a WRF-Chem build; fall back to &physics in case a build files
    # it there instead -- either way it is simply absent (-> 0) on a non-chem build.
    chem_opt = _num_list(chem.get("chem_opt") or phys.get("chem_opt"), max_dom)

    rows = []
    for i in range(max_dom):
        rows.append(dict(
            domain=i + 1, dx_km=(dx[i] or 0.0) / 1000.0, dy_km=(dy[i] or 0.0) / 1000.0,
            e_we=e_we[i], e_sn=e_sn[i], cu_physics=cu[i], bl_pbl_physics=bl[i],
            chem_opt=chem_opt[i] or 0))
    return max_dom, rows


def d01_extent_km(domains):
    """(west-east, south-north) physical extent of d01 in km, from (e_we-1)*dx / (e_sn-1)*dy
    -- e_we/e_sn are staggered-grid point counts, so the physical span is one dx short of
    e_we*dx."""
    d1 = domains[0]
    we_km = ((d1["e_we"] or 1) - 1) * d1["dx_km"]
    sn_km = ((d1["e_sn"] or 1) - 1) * d1["dy_km"]
    return we_km, sn_km


def _frame_span_hours(run_dir):
    times = sorted(t for t, _ in list_frames(run_dir, domain=1))
    if len(times) < 2:
        return 0.0
    t0 = datetime.strptime(times[0], WRF_TIME_FMT)
    t1 = datetime.strptime(times[-1], WRF_TIME_FMT)
    return (t1 - t0).total_seconds() / 3600.0


def run_length_hours(run_dir):
    """run_days/run_hours/run_minutes from &time_control; falls back to the actual wrfout
    frame span on disk when the namelist's run_* fields are all zero (some configurations
    drive the run length from end_* instead -- decision 0057 doesn't distinguish, and the
    frame span is the ground truth either way). A segmented run dir always uses the frame
    span: its namelist is seg_0's, whose run_* is that one segment's length."""
    if segment_dirs(run_dir):
        return _frame_span_hours(run_dir)
    nl = _core.parse_namelist(find_namelist(run_dir))
    tc = nl.get("time_control", {})
    days = _core.first_value(tc.get("run_days")) or 0
    hours = _core.first_value(tc.get("run_hours")) or 0
    minutes = _core.first_value(tc.get("run_minutes")) or 0
    total_h = days * 24 + hours + minutes / 60.0
    return total_h if total_h > 0 else _frame_span_hours(run_dir)


def initial_state(case_dir, domain=1):
    """A domain's state before the run: wrfinput_dNN, else (a restart case) its netCDF
    wrfrst_dNN_*; None when neither exists."""
    path = os.path.join(case_dir, "wrfinput_d%02d" % domain)
    if os.path.exists(path):
        return path
    rst = sorted(p for p in glob.glob(os.path.join(case_dir, "wrfrst_d%02d_*" % domain))
                 if os.path.isfile(p))
    return rst[0] if rst else None


def _d01_bbox_and_window(run_dir):
    """(lat_min, lat_max, lon_min, lon_max, t_start, t_end) for d01, from its wrfout frames'
    own XLAT/XLONG (continuous across 180E, so lon_max may pass 180: test a longitude with
    conventions.lon_in_range) and valid times (each file's Times variable via post.extract, so a
    frames_per_outfile > 1 file contributes all its times; netCDF4, lazily imported -- this
    module is otherwise stdlib-only). Before the run (no frames yet): wrfinput_d01's (or a
    restart case's wrfrst_d01's) grid and time plus the namelist run length."""
    import netCDF4
    frames = list_frames(run_dir, domain=1)
    wrfinput = initial_state(run_dir)
    if not frames and wrfinput:
        with netCDF4.Dataset(wrfinput) as ds:
            t_start = datetime.strptime(b"".join(ds.variables["Times"][0]).decode(), WRF_TIME_FMT)
            lat, lon = ds.variables["XLAT"][0], continuous_lons(ds.variables["XLONG"][0])
            return (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()),
                    t_start, t_start + timedelta(hours=run_length_hours(run_dir)))
    if not frames:
        raise CaseInfoError("no wrfout_d01_* frames in %s -- cannot check tc/domain extent"
                            % run_dir)
    times = [t for t, _ in frames]
    t_start = datetime.strptime(min(times), WRF_TIME_FMT)
    t_end = datetime.strptime(max(times), WRF_TIME_FMT)
    with netCDF4.Dataset(frames[0][1]) as ds:
        lat = ds.variables["XLAT"][0]
        lon = continuous_lons(ds.variables["XLONG"][0])
        lat_min, lat_max = float(lat.min()), float(lat.max())
        lon_min, lon_max = float(lon.min()), float(lon.max())
    return lat_min, lat_max, lon_min, lon_max, t_start, t_end


def _best_track_points(hurdat2_path=None, storm_id=None, bdeck_path=None):
    """[(datetime, lat, lon), ...], b-deck preferred over HURDAT2 when both are given (same
    priority as post/tc.py's resolve_bdeck -- a real b-deck is the more authoritative track).
    None if neither was given at all."""
    if bdeck_path:
        return [(r["time"], r["lat"], r["lon"]) for r in atcf.parse_file(bdeck_path)]
    if hurdat2_path and storm_id:
        from wtune.post.tc import parse_hurdat2_full
        return [(r["time"], r["lat"], r["lon"]) for r in parse_hurdat2_full(hurdat2_path, storm_id)]
    return None


def detect_tc(run_dir, hurdat2_path=None, storm_id=None, bdeck_path=None):
    """(bool, evidence_str). True only when a best-track fix falls inside d01's own lat/lon
    box during the run's own time window (decision 0057 section 1). False with a stated
    reason when no --hurdat2/--bdeck was given -- tc is then simply not auto-detected, per
    the CLI contract in the task spec."""
    points = _best_track_points(hurdat2_path, storm_id, bdeck_path)
    if points is None:
        return False, "no --hurdat2/--bdeck given -- tc not auto-detected"
    lat_min, lat_max, lon_min, lon_max, t_start, t_end = _d01_bbox_and_window(run_dir)
    hits = [(t, lat, lon) for t, lat, lon in points
            if t_start <= t <= t_end and lat_min <= lat <= lat_max
            and lon_in_range(lon, lon_min, lon_max)]
    if not hits:
        return False, ("best track has no fix inside d01 (lat %.1f..%.1f, lon %.1f..%.1f) "
                       "during %s..%s" % (lat_min, lat_max, lon_min, lon_max, t_start, t_end))
    return True, ("%d best-track fix(es) inside d01 during the run window (e.g. %s at "
                  "%.2f,%.2f)" % (len(hits), hits[0][0], hits[0][1], hits[0][2]))


DEFAULT_SPEC_BDY_WIDTH = 5


def spec_bdy_width(run_dir):
    """WRF's own lateral boundary zone width (namelist &bdy_control spec_bdy_width; default
    5, WRF's own compiled-in default) -- the number of grid cells on every domain edge where
    the solution is specified/relaxed toward the driving lateral boundary condition, not
    produced by the model's own free dynamics (2026-09-29 fourth lead review: a "highest
    point" or cross-section line landing in this zone shows boundary-relaxation behaviour,
    not real terrain-forced flow -- katrina m1's d01 "peak" turned out to sit exactly there).
    Applied uniformly to every domain including nests (an approximation for a nest's own
    feedback/relaxation boundary, which WRF does not expose as a single namelist width the
    way spec_bdy_width does for d01 -- 2026-09-29 review: "for nests ... use 5 cells too")."""
    width = _core.first_value(nl_bdy_control(run_dir).get("spec_bdy_width"))
    return int(width) if width else DEFAULT_SPEC_BDY_WIDTH


def nl_bdy_control(run_dir):
    nl = _core.parse_namelist(find_namelist(run_dir))
    return nl.get("bdy_control", {})


def _interior_bounds(ny, nx, bdy_width):
    """(j0, j1, i0, i1): the domain interior's index range, excluding bdy_width cells from
    every edge -- hgt[j0:j1, i0:i1] is python-slice usable. None if the domain is too small
    for this exclusion to leave any interior at all."""
    j0, j1, i0, i1 = bdy_width, ny - bdy_width, bdy_width, nx - bdy_width
    if j1 <= j0 or i1 <= i0:
        return None
    return j0, j1, i0, i1


def _state_file(run_dir, domain, last=False):
    """A domain's first (last) wrfout frame, else -- before the run -- initial_state; or None."""
    frames = list_frames(run_dir, domain=domain)
    if frames:
        return frames[-1 if last else 0][1]
    return initial_state(run_dir, domain)


def _domain_hgt_stats(run_dir, domain, bdy_width=DEFAULT_SPEC_BDY_WIDTH):
    """{"min","max","relief"} (metres) of HGT on one domain's first available frame,
    restricted to the domain INTERIOR (excluding bdy_width cells on every edge -- 2026-09-29
    fourth lead review). None if that domain has no frames, no HGT variable, or is too small
    for the given bdy_width to leave any interior."""
    import netCDF4
    import numpy as np
    path = _state_file(run_dir, domain)
    if not path:
        return None
    with netCDF4.Dataset(path) as ds:
        if "HGT" not in ds.variables:
            return None
        hgt = np.asarray(ds.variables["HGT"][0])
    bounds = _interior_bounds(hgt.shape[0], hgt.shape[1], bdy_width)
    if bounds is None:
        return None
    j0, j1, i0, i1 = bounds
    interior = hgt[j0:j1, i0:i1]
    return dict(min=float(interior.min()), max=float(interior.max()),
               relief=float(interior.max() - interior.min()))


def detect_terrain(run_dir, max_dom, bdy_width=None):
    """(bool, evidence_str, qualifying_domains). True when any domain's INTERIOR (excluding
    the lateral boundary zone, spec_bdy_width cells on every edge -- 2026-09-29 fourth lead
    review) HGT relief >= TERRAIN_RELIEF_M or peak height >= TERRAIN_PEAK_M (decision 0057
    update: "任一 domain 的 HGT 起伏 ... >= 1000 m，或最高點 >= 1500 m"). qualifying_domains: the
    sorted list of domain numbers that INDIVIDUALLY satisfy the rule on their own interior --
    e.g. on katrina m1, d01's interior does NOT reach 1000 m relief once its own boundary
    zone (where the only real relief happened to sit) is excluded, so katrina may no longer
    be a terrain case at all after this fix; that is the correct outcome, not a bug to work
    around by lowering the threshold. Thresholds are decision 0057's own numbers, unchanged."""
    bdy_width = bdy_width if bdy_width is not None else spec_bdy_width(run_dir)
    hits = []
    for d in range(1, max_dom + 1):
        stats = _domain_hgt_stats(run_dir, d, bdy_width=bdy_width)
        if stats is None:
            continue
        if stats["relief"] >= TERRAIN_RELIEF_M or stats["max"] >= TERRAIN_PEAK_M:
            hits.append((d, stats))
    qualifying_domains = [d for d, _ in hits]
    if hits:
        ev = "; ".join("d%02d interior HGT relief=%.0fm max=%.0fm (bdy_width=%d)"
                       % (d, s["relief"], s["max"], bdy_width) for d, s in hits)
        return True, ev, qualifying_domains
    return False, ("no domain's interior (bdy_width=%d) has HGT relief>=%gm or peak>=%gm"
                   % (bdy_width, TERRAIN_RELIEF_M, TERRAIN_PEAK_M)), []


def detect_cloud(run_dir):
    """(bool, evidence): "cloud" whenever the last wrfout_d01 frame holds any hydrometeor
    mixing ratio (post/condensate.SPECIES) -- the cloud figure set is generic."""
    path = _state_file(run_dir, 1, last=True)
    if not path:
        return False, "no wrfout_d01_* frame to inspect"
    try:
        import netCDF4
        from wtune.post import condensate
        with netCDF4.Dataset(path) as ds:
            present = condensate.species_present(ds)
    except (ImportError, OSError) as e:
        return False, "wrfout_d01 not readable: %s" % e
    if present:
        return True, "hydrometeors in wrfout_d01: %s" % ", ".join(present)
    return False, "no hydrometeor mixing ratio in wrfout_d01"


def detect_types(run_dirs, types_override=None, ensemble=False, hurdat2_path=None,
                 storm_id=None, bdeck_path=None):
    """Multi-label case-type detection against run_dirs[0]'s namelist (ensemble members
    share one configuration by construction, so checking the first is sufficient).

    types_override: an explicit set (e.g. from --types tc) replacing auto-detection; "common"
    is always included. Returns {"types": [...ordered per ALL_TYPES_ORDER...],
    "evidence": {type: str}, "domains": [...domain_physics() rows...], "max_dom": int,
    "run_length_h": float, "d01_extent_km": (we_km, sn_km)}.
    """
    primary = run_dirs[0]
    max_dom, domains = domain_physics(primary)
    run_h = run_length_hours(primary)
    we_km, sn_km = d01_extent_km(domains)

    evidence = {"common": "always produced"}
    detected = {"common"}

    tc_ok, tc_ev = detect_tc(primary, hurdat2_path, storm_id, bdeck_path)
    evidence["tc"] = tc_ev
    if tc_ok:
        detected.add("tc")

    conv_hits = [d for d in domains
                if d["dx_km"] <= CONVECTION_DX_KM and (d["cu_physics"] or 0) == 0]
    if conv_hits:
        detected.add("convection")
        evidence["convection"] = "; ".join(
            "d%02d dx=%.3g km, cu_physics=%d" % (d["domain"], d["dx_km"], d["cu_physics"] or 0)
            for d in conv_hits)
    else:
        evidence["convection"] = "no domain with dx<=%g km and cu_physics=0" % CONVECTION_DX_KM

    evidence["precip"] = "always produced (RAINC/RAINNC checked at figure time)"
    detected.add("precip")

    synoptic_reasons = []
    if we_km >= SYNOPTIC_EXTENT_KM or sn_km >= SYNOPTIC_EXTENT_KM:
        synoptic_reasons.append("d01 extent %.0fx%.0f km >= %g km" % (we_km, sn_km, SYNOPTIC_EXTENT_KM))
    if run_h >= SYNOPTIC_RUN_HOURS:
        synoptic_reasons.append("run length %.1f h >= %g h" % (run_h, SYNOPTIC_RUN_HOURS))
    if synoptic_reasons:
        detected.add("synoptic")
        evidence["synoptic"] = "; ".join(synoptic_reasons)
    else:
        evidence["synoptic"] = ("d01 %.0fx%.0f km, run %.1f h -- below synoptic thresholds"
                                % (we_km, sn_km, run_h))

    terrain_ok, terrain_ev, terrain_domains = detect_terrain(primary, max_dom)
    evidence["terrain"] = terrain_ev
    if terrain_ok:
        detected.add("terrain")

    # decision 0057 update: "偵測到 precip、convection 或 tc 時就做" -- precip is itself always
    # produced, so this is effectively always on alongside it; evaluated literally rather than
    # re-deriving a stricter rule the decision does not state.
    moisture_triggers = detected & {"precip", "convection", "tc"}
    if moisture_triggers:
        detected.add("moisture")
        evidence["moisture"] = "triggered by: %s" % ", ".join(sorted(moisture_triggers))
    else:
        evidence["moisture"] = "none of precip/convection/tc detected"

    cloud_ok, evidence["cloud"] = detect_cloud(primary)
    if cloud_ok:
        detected.add("cloud")

    pbl_hits = [d for d in domains
               if (d["bl_pbl_physics"] or 0) == 0 or d["dx_km"] <= PBL_LES_DX_KM]
    if pbl_hits:
        detected.add("pbl")
        evidence["pbl"] = "; ".join(
            "d%02d bl_pbl_physics=%d dx=%.3g km" % (d["domain"], d["bl_pbl_physics"] or 0, d["dx_km"])
            for d in pbl_hits)
    else:
        evidence["pbl"] = "no domain with bl_pbl_physics=0 or dx<=%g km" % PBL_LES_DX_KM

    chem_hits = [d for d in domains if (d["chem_opt"] or 0) > 0]
    if chem_hits:
        detected.add("chem")
        evidence["chem"] = "; ".join("d%02d chem_opt=%d" % (d["domain"], d["chem_opt"])
                                     for d in chem_hits)
    else:
        evidence["chem"] = "chem_opt=0 on every domain"

    if run_h >= CLIMATE_RUN_DAYS * 24:
        detected.add("climate")
        evidence["climate"] = "run length %.1f h (%.1f d) >= %g d" % (run_h, run_h / 24, CLIMATE_RUN_DAYS)
    else:
        evidence["climate"] = "run length %.1f h < %g d" % (run_h, CLIMATE_RUN_DAYS)

    if ensemble and len(run_dirs) > 1:
        detected.add("ensemble")
        evidence["ensemble"] = "%d run dirs given with --ensemble" % len(run_dirs)
    elif ensemble:
        evidence["ensemble"] = "--ensemble given but only one RUN_DIR"
    else:
        evidence["ensemble"] = "--ensemble not given"

    if types_override:
        wanted = set(t.strip() for t in types_override.split(",") if t.strip())
        unknown = wanted - set(ALL_TYPES_ORDER)
        if unknown:
            raise CaseInfoError("--types: unknown type(s) %s (known: %s)"
                                % (sorted(unknown), ", ".join(ALL_TYPES_ORDER)))
        detected = wanted | {"common"}

    ordered = [t for t in ALL_TYPES_ORDER if t in detected]
    return dict(types=ordered, evidence=evidence, domains=domains, max_dom=max_dom,
               run_length_h=run_h, d01_extent_km=(we_km, sn_km),
               terrain_domains=terrain_domains)

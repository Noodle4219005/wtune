#!/usr/bin/env python3
"""wtune/advise.py -- `wtune advise <case> [--need chem|hydro|surge] [--json]` (wiki decision 0061).

Detect conditions in a case (and, when available, in a finished run's verification output) and
print concrete, costed, evidence-backed ADVICE for WRF modes / workflows the user can choose
(moving nest, tc.exe, WRFDA radar assimilation, spectral nudging, WRF-Chem, WRF-Hydro ...).
ADVICE ONLY (decision 0056): nothing is applied, no namelist/input/source is written.

Architecture
  * rules  -- src/wtune_advice_rules.json (data). Trigger = detector name + thresholds.
  * detectors -- functions below, each reusing existing code (see DETECTOR_SOURCES). A detector
    whose input is missing returns {"available": False, "reason": ...}; the engine reports it
    under "unavailable" instead of silently treating it as "no".
  * surfaces -- `wtune advise` (all rules), the tail of `wtune tune` (all rules), and `wtune post
    recipe` (only the verification-triggered rules: those with a post_* detector clause).

Detector reuse (file:line as of this commit):
  case_types         post/caseinfo.py:272 detect_types (when wrfout frames exist; thresholds
                     CONVECTION_DX_KM etc. caseinfo.py:33-37 otherwise), best-track fix test
                     modelled on caseinfo.py:172 detect_tc
  tc_vortex_deficit  best track: post/tc.py:64 parse_hurdat2_full, post/atcf.py:113 parse_file;
                     distance: post/tc_metrics.py:49 dist_km; SLP: wrf-python getvar("slp"),
                     same call as analysis/diagnostics.py:137
  tc_nest_exit       same best-track readers; nest grid from wrfinput_dNN XLAT/XLONG;
                     boundary-zone width post/caseinfo.py:190 DEFAULT_SPEC_BDY_WIDTH
  radar_coverage     post/fetch/mrms.py:29 FIRST_DAY + :31 REGIONS/:50 pick_region,
                     post/fetch/stage4.py:28 CONUS_BBOX (US domain), post/fetch/nexrad.py:11
                     (verified-present date)
  run_length         wtune/cli.py:127 extract_case['run_total_hours'] (resolve_run_seconds)
  post_track_growth / post_intensity_bias
                     TC-Pairs .tcst columns TK_ERR/AMSLP/BMSLP/LEAD, parsed with
                     post/met_tc.py:45 parse_met_table (same columns post/tc.py:1031 _error_figures plots)

Not found in src/wtune/post: any IBTrACS reader/fetcher (only HURDAT2 + NHC ATCF b-deck exist, see
post/tc.py:175 resolve_bdeck). So best track = --bdeck | --hurdat2(+--storm) | project.yaml storm
| registry observations{kind:track}; no download is done here.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timedelta

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import wtune.machine  # noqa: E402

RULES_PATH = os.path.join(_HERE, "wtune_advice_rules.json")
COEFFS_PATH = os.path.join(_HERE, "wtune_extension_coefficients.json")
CAMPAIGN_DIR = os.path.join(os.path.dirname(_HERE), "campaign")

NEEDS = ("chem", "hydro", "surge")

# --- detector parameters that are NOT rule thresholds (thresholds live in the rules file) ---
TC_CENTER_RADIUS_KM = 250.0      # model SLP minimum is searched this far from the best-track fix
TC_T0_TOLERANCE_H = 3.0          # best-track fix must be this close to the case start time
NEST_EDGE_FACTOR = 1.5           # a fix farther than 1.5 dx from the nearest nest cell is outside
NEST_REFINE_DX_M = 3000.0        # rule 1: "inner grid <= 3 km"
NEST_REFINE_SIZE_KM = 600.0      # ASSUMED footprint of an added inner nest (stated in the message)
# post/fetch/nexrad.py:11 -- "verified present for 2005-08-28 (KLIX, KMOB)". The bucket reaches
# further back but only this date is verified in the repo, so this is the conservative bound.
NEXRAD_VERIFIED_FROM = datetime(2005, 8, 28)
MIN_LEADS_FOR_GROWTH = 3
MIN_SPAN_H_FOR_GROWTH = 48.0
NMI_TO_KM = 1.852
WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"


class AdviseError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# case context
# --------------------------------------------------------------------------------------------
class AdviceCase:
    """Everything the detectors may read. All fields optional except `name` and `namelist`."""

    def __init__(self, name, namelist, first_state=None, case_dir=None, run_dir=None,
                 needs=(), hurdat2=None, storm=None, bdeck=None, registry=None, post_tc_dir=None):
        self.name = name
        self.namelist = namelist            # may be None: detectors then report unavailable
        self.first_state = first_state      # wrfinput_d01 (or earliest wrfout_d01): t0 fields
        self.case_dir = case_dir
        self.run_dir = run_dir              # dir with wrfout_* frames, if any
        self.needs = set(needs)
        self.hurdat2 = hurdat2
        self.storm = storm
        self.bdeck = bdeck
        self.registry = registry or {}
        self.post_tc_dir = post_tc_dir      # post recipe's <out>/tc (holds met_*/*.tcst)
        self.cache = {}


def _expand(p):
    return os.path.expanduser(os.path.expandvars(p)) if isinstance(p, str) else p


def load_registry_case(case_id, site=None):
    """cases.json overlaid by cases.<site>.json (same merge as campaign/configs.py:load_cases,
    re-implemented here only to avoid importing the predictor stack for a read-only lookup).
    $CASES_JSON replaces cases.json, as in configs.py."""
    with open(os.environ.get("CASES_JSON") or os.path.join(CAMPAIGN_DIR, "cases.json")) as f:
        cases = json.load(f)["cases"]
    site = site or wtune.machine.site()
    if site != "f1":
        ov_path = os.path.join(CAMPAIGN_DIR, "cases.%s.json" % site)
        if os.path.exists(ov_path):
            with open(ov_path) as f:
                for cid, over in json.load(f)["cases"].items():
                    base = dict(cases.get(cid, {}))
                    base.update({k: (_expand(v) if isinstance(v, str) else v)
                                 for k, v in over.items()})
                    cases[cid] = base
    if case_id not in cases:
        raise AdviseError("unknown case %r (not a directory, not in campaign/cases.json)" % case_id)
    return cases[case_id]


def _find_first_state(case_dir):
    if not case_dir or not os.path.isdir(case_dir):
        return None
    for pat in ("wrfinput_d01", "wrfinput_d01*"):
        hits = sorted(glob.glob(os.path.join(case_dir, pat)))
        if hits:
            return hits[0]
    frames = sorted(glob.glob(os.path.join(case_dir, "wrfout_d01_*")))
    return frames[0] if frames else None


def _find_namelist(d):
    # namelist.input first: it is what the user wrote and what wtune.predict.parse_namelist reads
    # everywhere else; WRF's namelist.output (fully resolved) is the fallback for a run dir
    for name in ("namelist.input", "namelist.output"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _project_storm_defaults(storm, hurdat2, project=None):
    """Fill an absent --storm/--hurdat2 from the project in effect (--project, $WTUNE_PROJECT,
    or a project.yaml at/above cwd; wtune.project.find_project) -- per field, as the
    project-dir branch of build_case does. Explicit flags win; no project -> unchanged."""
    if storm and hurdat2:
        return storm, hurdat2
    import wtune.project
    pdir = wtune.project.find_project(explicit=project)
    if not pdir:
        return storm, hurdat2
    p_storm, p_hurdat = wtune.project.project_storm(wtune.project.load_project(pdir))
    return storm or p_storm, hurdat2 or _expand(p_hurdat)


def _project_bdeck_default(project=None):
    """The project's storm.bdeck (`wtune project set --bdeck`, a storm outside HURDAT2), else None:
    without it tune saw haitang2017 (JMA track only) as common/convection, never tc."""
    import wtune.project
    try:
        pdir = wtune.project.find_project(explicit=project)
    except wtune.project.ProjectError:
        return None
    return _expand(wtune.project.project_bdeck(wtune.project.load_project(pdir))) if pdir else None


def build_case(arg, site=None, needs=(), hurdat2=None, storm=None, bdeck=None, post_tc_dir=None,
               project=None):
    """`arg` = a directory (case prep dir, run dir, or a wtune project) or a registry case id.
    `project` = --project DIR (storm/hurdat2 defaults, see _project_storm_defaults)."""
    registry = None
    case_dir = run_dir = namelist = None
    if os.path.isdir(_expand(arg)):
        d = os.path.abspath(_expand(arg))
        if os.path.isfile(os.path.join(d, "project.yaml")):
            import wtune.project
            doc = wtune.project.load_project(d)
            p_storm, p_hurdat = wtune.project.project_storm(doc)
            storm, hurdat2 = storm or p_storm, hurdat2 or p_hurdat
            runs = wtune.project.discover_runs(d)
            case_dir = os.path.join(d, "case")
            run_dir = runs[0][1] if runs else None
            namelist = _find_namelist(run_dir or case_dir) or _find_namelist(case_dir)
        else:
            case_dir = d
            run_dir = d if glob.glob(os.path.join(d, "wrfout_d*")) else None
            namelist = _find_namelist(d)
            storm, hurdat2 = _project_storm_defaults(storm, hurdat2, project)
        name = os.path.basename(d.rstrip("/"))
    else:
        registry = load_registry_case(arg, site)
        case_dir = _expand(registry["prep_dir"])
        namelist = _expand(registry.get("namelist_template") or
                           os.path.join(registry["prep_dir"], "namelist.input"))
        name = arg
        obs = registry.get("observations") or {}
        storm, hurdat2 = _project_storm_defaults(storm, hurdat2, project)
        if obs.get("kind") == "track":
            storm = storm or obs.get("storm_id")
            hurdat2 = hurdat2 or _expand(obs.get("path"))
    if namelist and not os.path.exists(namelist):
        raise AdviseError("namelist %s does not exist" % namelist)
    # a directory without a namelist is still advisable: detectors that need it are unavailable
    hurdat2 = hurdat2 or os.environ.get("WTUNE_HURDAT2") or None
    bdeck = bdeck or _project_bdeck_default(project)
    first_state = _find_first_state(run_dir or case_dir) or _find_first_state(case_dir)
    return AdviceCase(name, namelist, first_state=first_state, case_dir=case_dir, run_dir=run_dir,
                      needs=needs, hurdat2=hurdat2, storm=storm, bdeck=bdeck, registry=registry,
                      post_tc_dir=post_tc_dir)


# --------------------------------------------------------------------------------------------
# namelist helpers (wtune.predict.parse_namelist: typed, per-domain aware)
# --------------------------------------------------------------------------------------------
def _nl(case):
    if not case.namelist:
        raise AdviseError("no namelist.input/namelist.output in the case directory")
    if "nl" not in case.cache:
        import wtune.predict
        case.cache["nl"] = wtune.predict.parse_namelist(case.namelist)
    return case.cache["nl"]


def _g_dom(case, group, key, d, default=None):
    import wtune.predict
    return wtune.predict.g_dom(_nl(case), group, key, d, default)


def _max_dom(case):
    import wtune.predict
    return int(wtune.predict.get_max_dom(_nl(case)))


def _window(case):
    """(t_start, t_end) from &time_control start_*/end_* (d01). end falls back to start + run
    length (wtune.cli.extract_case run_total_hours)."""
    nl = _nl(case)

    def dt(prefix):
        tc = nl.get("time_control", {})
        try:
            vals = [int(tc[prefix + "_" + k][0]) for k in ("year", "month", "day", "hour")]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        mi = int(tc.get(prefix + "_minute", [0])[0] or 0)
        se = int(tc.get(prefix + "_second", [0])[0] or 0)
        return datetime(vals[0], vals[1], vals[2], vals[3], mi, se)

    t0 = dt("start")
    if t0 is None:
        raise AdviseError("namelist has no start_year/month/day/hour")
    t1 = dt("end")
    run_h = _run_hours(case)
    if run_h and run_h > 0:
        t1 = t0 + timedelta(hours=run_h)
    elif t1 is None:
        t1 = t0
    return t0, t1


def _run_hours(case):
    if not case.namelist:
        raise AdviseError("no namelist.input/namelist.output in the case directory")
    import wtune.cli
    return float(wtune.cli.extract_case(wtune.cli.parse_namelist(case.namelist))
                 .get("run_total_hours") or 0.0)


# --------------------------------------------------------------------------------------------
# grid / field readers (netCDF4 + wrf-python, lazily imported; tests monkeypatch these)
# --------------------------------------------------------------------------------------------
def _read_latlon(path):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        return (netCDF4.chartostring(ds.variables["Times"][:])[0].strip() if "Times" in ds.variables
                else None, ds.variables["XLAT"][0][:], ds.variables["XLONG"][0][:])


def _read_min_slp(path, lat, lon, radius_km):
    """Minimum SLP (hPa) within radius_km of (lat, lon) in the first time level of `path`;
    wrf-python getvar('slp'), as analysis/diagnostics.py:137. Returns (slp_min, (lat, lon))."""
    import netCDF4
    import numpy as np
    from wrf import getvar
    from wtune.post import tc_metrics as tm
    with netCDF4.Dataset(path) as ds:
        slp = np.asarray(getvar(ds, "slp", timeidx=0, units="hPa"))
        xlat = np.asarray(ds.variables["XLAT"][0])
        xlon = np.asarray(ds.variables["XLONG"][0])
    dist = tm.dist_km(lat, lon, xlat, xlon)
    masked = np.where(dist <= radius_km, slp, np.inf)
    j, i = np.unravel_index(int(np.argmin(masked)), masked.shape)
    if not np.isfinite(masked[j, i]):
        raise AdviseError("best-track position (%.2f, %.2f) is outside the domain" % (lat, lon))
    return float(masked[j, i]), (float(xlat[j, i]), float(xlon[j, i]))


def _domain_bbox(case):
    """(lat_min, lat_max, lon_min, lon_max) of d01: XLAT/XLONG when a state file exists, else a
    box around ref_lat/ref_lon from the namelist extent (flat-earth approximation, stated)."""
    if "bbox" in case.cache:
        return case.cache["bbox"]
    out = None
    if case.first_state:
        try:
            _t, lat, lon = _read_latlon(case.first_state)
            out = (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()), "grid")
        except Exception:                                   # noqa: BLE001  fall back to namelist
            out = None
    if out is None:
        ref_lat = _g_dom(case, "domains", "ref_lat", 0)
        ref_lon = _g_dom(case, "domains", "ref_lon", 0)
        if ref_lat is None or ref_lon is None:
            raise AdviseError("no wrfinput/wrfout and no ref_lat/ref_lon in namelist: domain "
                              "location unknown")
        dx = float(_g_dom(case, "domains", "dx", 0)) / 1000.0
        dy = float(_g_dom(case, "domains", "dy", 0, dx * 1000.0)) / 1000.0
        half_we = (int(_g_dom(case, "domains", "e_we", 0)) - 1) * dx / 2.0
        half_sn = (int(_g_dom(case, "domains", "e_sn", 0)) - 1) * dy / 2.0
        dlat = half_sn / 111.0
        dlon = half_we / (111.0 * max(0.2, math.cos(math.radians(ref_lat))))
        out = (ref_lat - dlat, ref_lat + dlat, ref_lon - dlon, ref_lon + dlon, "namelist-approx")
    case.cache["bbox"] = out
    return out


# --------------------------------------------------------------------------------------------
# best track
# --------------------------------------------------------------------------------------------
def _scan_hurdat2_for_storm(path, bbox, t_start, t_end):
    """Storm id in a HURDAT2 file with the most fixes inside (bbox, [t_start, t_end]); None if
    no storm qualifies. One pass, coordinates parsed with post/tc.py:58 _hurdat2_coord."""
    from wtune.post.tc import _hurdat2_coord
    lat0, lat1, lon0, lon1 = bbox[:4]
    best, best_n, cur, n = None, 0, None, 0
    with open(path, errors="replace") as fh:
        for raw in fh:
            f = [x.strip() for x in raw.split(",")]
            if len(f) >= 3 and f[0][:2].isalpha() and f[0][2:6].isdigit():
                if cur and n > best_n:
                    best, best_n = cur, n
                cur, n = f[0].upper(), 0
                continue
            if cur is None or len(f) < 6:
                continue
            try:
                when = datetime.strptime(f[0] + f[1], "%Y%m%d%H%M")
                lat, lon = _hurdat2_coord(f[4]), _hurdat2_coord(f[5])
            except ValueError:
                continue
            if t_start <= when <= t_end and lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
                n += 1
    if cur and n > best_n:
        best, best_n = cur, n
    return best


def best_track(case):
    """[{time, lat, lon, pmin_hpa}] sorted by time, or None when no best track is available.
    Priority as post/tc.py resolve_bdeck: --bdeck, then HURDAT2 (+storm id; found by scanning the
    file for the storm inside the domain/window when no id was given)."""
    if "track" in case.cache:
        return case.cache["track"]
    rows = None
    if case.bdeck and os.path.exists(case.bdeck):
        from wtune.post import atcf
        rows = sorted(({"time": r["time"], "lat": r["lat"], "lon": r["lon"],
                        "pmin_hpa": float(r["mslp_hpa"]) if r["mslp_hpa"] else None}
                       for r in atcf.parse_file(case.bdeck)), key=lambda r: r["time"])
    elif case.hurdat2 and os.path.exists(case.hurdat2):
        from wtune.post.tc import parse_hurdat2_full, resolve_storm
        storm = case.storm
        if storm:
            # NAME / NAME+YEAR -> ATCF id, restricted to storms overlapping the case window
            try:
                window = _window(case)
            except AdviseError:
                window = None
            storm = case.storm = resolve_storm(case.hurdat2, storm, window=window)
        else:
            t0, t1 = _window(case)
            storm = _scan_hurdat2_for_storm(case.hurdat2, _domain_bbox(case), t0, t1)
            case.cache["storm_found"] = storm
        if storm:
            rows = [{"time": r["time"], "lat": r["lat"], "lon": r["lon"],
                     "pmin_hpa": r["pmin_hpa"]} for r in parse_hurdat2_full(case.hurdat2, storm)]
    case.cache["track"] = rows
    return rows


# --------------------------------------------------------------------------------------------
# detectors  (each returns a dict with "available": bool)
# --------------------------------------------------------------------------------------------
def _unavail(reason):
    return {"available": False, "reason": reason}


def det_case_types(case):
    """Multi-label case type (decision 0057). Full detect_types when wrfout frames exist;
    otherwise the same thresholds read from the namelist plus a best-track fix test."""
    from wtune.post import caseinfo
    if case.run_dir and glob.glob(os.path.join(case.run_dir, "wrfout_d01_*")):
        try:
            best_track(case)                    # fills cache["storm_found"] when only HURDAT2 given
            info = caseinfo.detect_types([case.run_dir], hurdat2_path=case.hurdat2,
                                         storm_id=case.storm or case.cache.get("storm_found"),
                                         bdeck_path=case.bdeck)
            types = set(info["types"])
            return {"available": True, "source": "post.caseinfo.detect_types",
                    "types": info["types"], "tc": "tc" in types,
                    "convection": "convection" in types, "climate": "climate" in types,
                    "convection_evidence": info["evidence"].get("convection", ""),
                    "tc_evidence": info["evidence"].get("tc", "")}
        except Exception as e:                              # noqa: BLE001  fall through
            case.cache["case_types_note"] = "detect_types failed (%s); using namelist" % e
    max_dom = _max_dom(case)
    types, ev = {"common"}, {}
    conv = []
    for d in range(max_dom):
        dx = float(_g_dom(case, "domains", "dx", d, 0) or 0) / 1000.0
        cu = int(_g_dom(case, "physics", "cu_physics", d, 0) or 0)
        if dx <= caseinfo.CONVECTION_DX_KM and cu == 0:
            conv.append("d%02d dx=%.3g km, cu_physics=0" % (d + 1, dx))
    if conv:
        types.add("convection")
    ev["convection"] = "; ".join(conv) or "no domain with dx<=%g km and cu_physics=0" % \
        caseinfo.CONVECTION_DX_KM
    run_h = _run_hours(case)
    if run_h >= caseinfo.CLIMATE_RUN_DAYS * 24:
        types.add("climate")
    # tc: a best-track fix inside d01 during the run window (caseinfo.detect_tc's test)
    tc, tc_ev = False, "no best track given (--bdeck/--hurdat2/--storm) and registry case_family is not tropical_cyclone"
    try:
        track = best_track(case)
    except AdviseError as e:
        track, tc_ev = None, "best track not resolvable: %s" % e
    if track:
        try:
            t0, t1 = _window(case)
            lat0, lat1, lon0, lon1 = _domain_bbox(case)[:4]
            hits = [r for r in track if t0 <= r["time"] <= t1 and lat0 <= r["lat"] <= lat1
                    and lon0 <= r["lon"] <= lon1]
            tc = bool(hits)
            tc_ev = ("%d best-track fix(es) inside d01 during the run window" % len(hits)) if hits \
                else "best track has no fix inside d01 during the run window"
        except AdviseError as e:
            tc_ev = "domain/window unknown: %s" % e
    elif (case.registry or {}).get("case_family") == "tropical_cyclone":
        tc, tc_ev = True, "registry case_family=tropical_cyclone (no best track to confirm)"
    if tc:
        types.add("tc")
    return {"available": True, "source": "namelist+best-track", "types": sorted(types),
            "tc": tc, "convection": "convection" in types, "climate": "climate" in types,
            "convection_evidence": ev["convection"], "tc_evidence": tc_ev,
            "note": case.cache.get("case_types_note", "")}


def _t0_fix(case, track):
    t0, _ = _window(case)
    fix = min(track, key=lambda r: abs((r["time"] - t0).total_seconds()))
    if abs((fix["time"] - t0).total_seconds()) / 3600.0 > TC_T0_TOLERANCE_H:
        return None, t0
    return fix, t0


def det_tc_vortex_deficit(case):
    track = best_track(case)
    if not track:
        return _unavail("無最佳路徑：請給 --bdeck，或 --hurdat2（可加 --storm）")
    if not case.first_state:
        return _unavail("找不到 wrfinput_d01／wrfout 以讀取 t0 的 SLP")
    fix, t0 = _t0_fix(case, track)
    if fix is None:
        return _unavail("最佳路徑沒有落在 t0 ±%g h 內的紀錄" % TC_T0_TOLERANCE_H)
    if fix["pmin_hpa"] is None:
        return _unavail("最佳路徑 t0 紀錄沒有中心氣壓")
    try:
        slp, centre = _read_min_slp(case.first_state, fix["lat"], fix["lon"], TC_CENTER_RADIUS_KM)
    except Exception as e:                                  # noqa: BLE001
        return _unavail("讀 SLP 失敗：%s" % e)
    return {"available": True, "model_slp_hpa": slp, "best_pmin_hpa": fix["pmin_hpa"],
            "delta_hpa": slp - fix["pmin_hpa"], "t0": t0.strftime(WRF_TIME_FMT),
            "best_fix_time": fix["time"].strftime(WRF_TIME_FMT),
            "best_lat": fix["lat"], "best_lon": fix["lon"],
            "model_centre": list(centre), "search_radius_km": TC_CENTER_RADIUS_KM,
            "storm": case.storm or case.cache.get("storm_found")}


def _nest_path(case, dom):
    if not case.first_state:
        return None
    base = os.path.basename(case.first_state)
    cand = os.path.join(os.path.dirname(case.first_state), base.replace("d01", "d%02d" % dom, 1))
    return cand if os.path.exists(cand) else None


def det_tc_nest_exit(case):
    from wtune.post import caseinfo
    from wtune.post import tc_metrics as tm
    import numpy as np
    max_dom = _max_dom(case)
    track = best_track(case)
    if not track:
        return _unavail("無最佳路徑：請給 --bdeck，或 --hurdat2（可加 --storm）")
    if max_dom < 2:
        return {"available": True, "n_total": 0, "n_outside": 0, "nest_domain": None,
                "first_exit": None, "note": "單一 domain，沒有內層網格"}
    path = _nest_path(case, max_dom)
    if not path:
        return _unavail("找不到 wrfinput_d%02d／wrfout_d%02d 以取得內層網格範圍" % (max_dom, max_dom))
    t0, t1 = _window(case)
    fixes = [r for r in track if t0 <= r["time"] <= t1]
    if not fixes:
        return _unavail("最佳路徑在模擬時段內沒有紀錄")
    _t, lat2d, lon2d = _read_latlon(path)
    dx_km = float(_g_dom(case, "domains", "dx", max_dom - 1)) / 1000.0
    bw = caseinfo.DEFAULT_SPEC_BDY_WIDTH
    ny, nx = lat2d.shape
    outside = []
    for r in fixes:
        dist = tm.dist_km(r["lat"], r["lon"], lat2d, lon2d)
        j, i = np.unravel_index(int(np.argmin(dist)), dist.shape)
        inside = (float(dist[j, i]) <= NEST_EDGE_FACTOR * dx_km and bw <= j < ny - bw
                  and bw <= i < nx - bw)
        if not inside:
            outside.append(r)
    return {"available": True, "n_total": len(fixes), "n_outside": len(outside),
            "nest_domain": max_dom, "grid_source": os.path.basename(path),
            "first_exit": outside[0]["time"].strftime(WRF_TIME_FMT) if outside else None,
            "outside_frac": len(outside) / len(fixes)}


def det_radar_coverage(case):
    from wtune.post.fetch import mrms, stage4
    try:
        bbox = _domain_bbox(case)
        t0, t1 = _window(case)
    except AdviseError as e:
        return _unavail(str(e))
    lat0, lat1, lon0, lon1 = bbox[:4]
    clat, clon = (lat0 + lat1) / 2, (lon0 + lon1) / 2
    cb = stage4.CONUS_BBOX
    us = cb[0] <= clat <= cb[1] and cb[2] <= clon <= cb[3]
    region = mrms.pick_region((lat0, lat1, lon0, lon1))
    in_mrms = bool(region) and t1 >= mrms.FIRST_DAY
    in_nexrad = us and t0 >= NEXRAD_VERIFIED_FROM
    covered = us and (in_mrms or in_nexrad)
    parts = ["domain 中心 (%.1f, %.1f) %s美國範圍（stage4.CONUS_BBOX）" % (clat, clon, "在" if us else "不在")]
    parts.append("MRMS %s（區域 %s，自 %s 起）" % ("可用" if in_mrms else "不可用", region or "無",
                                              mrms.FIRST_DAY.date()))
    parts.append("NEXRAD Level II %s（已驗證自 %s 起）" % ("可用" if in_nexrad else "不可用",
                                                      NEXRAD_VERIFIED_FROM.date()))
    return {"available": True, "us_domain": us, "region": region, "in_mrms": in_mrms,
            "in_nexrad": in_nexrad, "covered": covered, "text": "；".join(parts)}


def det_run_length(case):
    try:
        h = _run_hours(case)
    except Exception as e:                                  # noqa: BLE001
        return _unavail("讀不到模擬長度：%s" % e)
    return {"available": True, "run_hours": h, "run_days": h / 24.0}


def det_user_need(case):
    return {"available": True, **{n: (n in case.needs) for n in NEEDS}}


def _tcst_rows(case):
    """TC-Pairs rows (all runs) under post_tc_dir; [] when there is no verification output."""
    if not case.post_tc_dir:
        return None
    paths = sorted(glob.glob(os.path.join(case.post_tc_dir, "met_*", "*.tcst")) +
                   glob.glob(os.path.join(case.post_tc_dir, "*.tcst")))
    if not paths:
        return None
    from wtune.post import met_tc
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += met_tc.parse_met_table(fh.read())
    return rows


def _lead_h(v):
    v = v.strip()
    return int(v[:-4]) + int(v[-4:-2]) / 60.0 if len(v) > 4 else float(v)


def _num(r, col):
    try:
        x = float(r[col])
    except (KeyError, ValueError):
        return None
    return None if x <= -9999 else x


def det_post_track_growth(case):
    rows = _tcst_rows(case)
    if rows is None:
        return _unavail("沒有 TC-Pairs 輸出（post tc 的 met_*/*.tcst）")
    pts = []
    for r in rows:
        if "LEAD" not in r:
            continue
        e = _num(r, "TK_ERR")
        if e is not None:
            pts.append((_lead_h(r["LEAD"]), e * NMI_TO_KM))
    leads = sorted({h for h, _ in pts})
    if len(leads) < MIN_LEADS_FOR_GROWTH or (leads[-1] - leads[0]) < MIN_SPAN_H_FOR_GROWTH:
        return _unavail("TC-Pairs 預報時間點不足（需 ≥ %d 個、跨度 ≥ %g h）"
                        % (MIN_LEADS_FOR_GROWTH, MIN_SPAN_H_FOR_GROWTH))
    n = len(pts)
    mx, my = sum(h for h, _ in pts) / n, sum(e for _, e in pts) / n
    sxx = sum((h - mx) ** 2 for h, _ in pts)
    slope_h = sum((h - mx) * (e - my) for h, e in pts) / sxx
    return {"available": True, "slope_km_per_day": slope_h * 24.0, "n": n,
            "lead_span_h": leads[-1] - leads[0], "max_track_err_km": max(e for _, e in pts)}


def det_post_intensity_bias(case):
    rows = _tcst_rows(case)
    if rows is None:
        return _unavail("沒有 TC-Pairs 輸出（post tc 的 met_*/*.tcst）")
    weak = []
    for r in rows:
        a, b = _num(r, "AMSLP"), _num(r, "BMSLP")
        if a is not None and b is not None:
            weak.append(a - b)          # > 0: model central pressure higher (weaker) than best track
    if not weak:
        return _unavail("TC-Pairs 沒有 AMSLP/BMSLP")
    return {"available": True, "median_weak_hpa": statistics.median(weak), "n": len(weak),
            "max_weak_hpa": max(weak)}


DETECTORS = {
    "case_types": det_case_types,
    "tc_vortex_deficit": det_tc_vortex_deficit,
    "tc_nest_exit": det_tc_nest_exit,
    "radar_coverage": det_radar_coverage,
    "run_length": det_run_length,
    "user_need": det_user_need,
    "post_track_growth": det_post_track_growth,
    "post_intensity_bias": det_post_intensity_bias,
}


def run_detector(case, name):
    if name not in case.cache.setdefault("det", {}):
        try:
            case.cache["det"][name] = DETECTORS[name](case)
        except Exception as e:                              # noqa: BLE001  never abort advice
            case.cache["det"][name] = _unavail("%s: %s" % (type(e).__name__, e))
    return case.cache["det"][name]


# --------------------------------------------------------------------------------------------
# cost estimates (WTune cost model / measured coefficients)
# --------------------------------------------------------------------------------------------
def _step_weight(case, d, dt):
    """Per-column per-step prior weight of domain d: same ingredients as
    wtune.case._build_cost_structure / wtune.physics_advice._cost_rate (dynamics + mp + cu +
    surface + pbl/sfclay + radiation, the last amortised by radt)."""
    from wtune import physics as phys
    from wtune.case import COST_WEIGHT, DYN_WEIGHT
    cats = ("mp_physics", "cu_physics", "bl_pbl_physics", "sf_sfclay_physics", "sf_surface_physics",
            "ra_lw_physics", "ra_sw_physics")
    suite = _g_dom(case, "physics", "physics_suite", 0)
    row = {"physics_suite": suite.strip("'\" ").lower() if isinstance(suite, str) else suite}
    for c in cats:
        v = _g_dom(case, "physics", c, d)
        row[c] = int(v) if v is not None else -1
    resolved, _ = phys.resolve_suite(row)
    w = {c: phys.cost_weight(c, resolved[c], COST_WEIGHT) for c in cats}
    radt = float(_g_dom(case, "physics", "radt", d, 0) or 0)
    spacing = max(1, int(round(radt * 60.0 / dt))) if radt > 0 else 1
    return (DYN_WEIGHT + w["mp_physics"] + w["cu_physics"] + w["sf_surface_physics"]
            + w["bl_pbl_physics"] + w["sf_sfclay_physics"]
            + (w["ra_lw_physics"] + w["ra_sw_physics"]) / spacing)


def _domain_dt_list(case, max_dom):
    dt0 = float(_g_dom(case, "domains", "time_step", 0, 60) or 60)
    out, cum = [], 1.0
    for d in range(max_dom):
        if d:
            r = _g_dom(case, "domains", "parent_time_step_ratio", d)
            if r is None:
                r = _g_dom(case, "domains", "parent_grid_ratio", d, 1)
            cum *= float(r or 1.0)
        out.append(dt0 / cum)
    return out


def nest_refinement_extra_pct(case):
    """Prior (cost-weight x cells / dt) extra cost of adding a NEST_REFINE_SIZE_KM square inner
    nest (integer ratio >= 2 giving dx <= NEST_REFINE_DX_M) under the finest existing domain;
    None when that domain is already <= 3 km. Returns (extra_pct, ratio, nest_dx_km)."""
    max_dom = _max_dom(case)
    dts = _domain_dt_list(case, max_dom)
    total, dxs = 0.0, []
    for d in range(max_dom):
        cells = 1.0
        for k in ("e_we", "e_sn", "e_vert"):
            cells *= float(_g_dom(case, "domains", k, d, 1) or 1)
        total += _step_weight(case, d, dts[d]) * cells / dts[d]
        dxs.append(float(_g_dom(case, "domains", "dx", d, 0) or 0))
    f = max_dom - 1
    if dxs[f] <= NEST_REFINE_DX_M:
        return None
    ratio = max(2, int(math.ceil(dxs[f] / NEST_REFINE_DX_M)))     # a nest needs ratio >= 2
    dx_n = dxs[f] / ratio
    n = int(round(NEST_REFINE_SIZE_KM * 1000.0 / dx_n))
    e_vert = float(_g_dom(case, "domains", "e_vert", f, 1) or 1)
    nest = _step_weight(case, f, dts[f] / ratio) * n * n * e_vert / (dts[f] / ratio)
    return 100.0 * nest / total, ratio, dx_n / 1000.0


def cost_vars(rule_id, case):
    if rule_id == "tc_weak_vortex_early_start":
        run_h = _run_hours(case)
        v = {"run_h": run_h, "extra_24_pct": 100.0 * 24 / run_h if run_h else float("nan"),
             "extra_36_pct": 100.0 * 36 / run_h if run_h else float("nan")}
        est = nest_refinement_extra_pct(case)
        v["nest_text"] = ("內層網格 ≤ 3 km：目前最細 domain 已 ≤ 3 km，無需另加。" if est is None else
                          "另加 %g×%g km、%.1f km 的內層 nest（比值 %d，假設的足跡）：先驗成本 +%.0f%%"
                          "（權重×格點/dt，排序用的先驗，非牆鐘）。" %
                          (NEST_REFINE_SIZE_KM, NEST_REFINE_SIZE_KM, est[2], est[1], est[0]))
        return v
    if rule_id == "tc_leaves_inner_nest_movenest":
        with open(COEFFS_PATH) as f:
            return {"const_frac_pct": 100.0 * json.load(f)["movenest"]["const_frac"]}
    return {}


# --------------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------------
def load_rules(path=None):
    with open(path or RULES_PATH) as f:
        doc = json.load(f)
    if doc.get("schema") != 1:
        raise AdviseError("unsupported rules schema %r" % doc.get("schema"))
    return doc


_OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b, "==": lambda a, b: a == b,
        "is_true": lambda a, b: bool(a) is True, "is_false": lambda a, b: bool(a) is False}


def _clause(case, c, unavailable):
    det = run_detector(case, c["detector"])
    if not det.get("available"):
        unavailable.setdefault(c["detector"], det.get("reason", ""))
        return False
    if c["field"] not in det or det[c["field"]] is None:
        return False
    return bool(_OPS[c["op"]](det[c["field"]], c.get("value")))


def _render_trigger(spec, values):
    """trigger_text is a string, or a list of {detector, text}: a part is printed only when its
    detector produced a value (so an absent post verification never prints a '—' sentence)."""
    if isinstance(spec, str):
        return _render(spec, values)
    return " ".join(_render(p["text"], values) for p in spec
                    if values.get(p["detector"], {}).get("available"))


def _render(template, values):
    """{det.field:fmt} / {name:fmt} substitution; any missing value renders as '—' so a
    template never raises (the unavailable detector is reported separately)."""
    def sub(m):
        key, fmt = m.group(1), m.group(2) or ""
        cur = values
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                return "—"
        try:
            return format(cur, fmt[1:]) if fmt else str(cur)
        except (ValueError, TypeError):
            return str(cur)
    return re.sub(r"\{([A-Za-z0-9_.]+)(:[^}]*)?\}", sub, template)


def _post_clause(rule):
    return any(c["detector"].startswith("post_")
               for c in rule["trigger"].get("any", []) + rule["trigger"].get("all", []))


def advise(case, surface="advise", rules_doc=None):
    """Evaluate the rules against `case`. surface='post' keeps only rules with a post_* clause
    (verification-triggered). Returns {"case", "surface", "advice": [...], "not_fired": [...],
    "unavailable": {detector: reason}, "footer"}."""
    doc = rules_doc or load_rules()
    unavailable, fired, not_fired = {}, [], []
    for rule in doc["rules"]:
        if surface == "post" and not _post_clause(rule):
            continue
        trig = rule["trigger"]
        missing = {}
        ok_all = all(_clause(case, c, missing) for c in trig.get("all", []))
        any_c = trig.get("any", [])
        # `any` is only evaluated once every `all` clause holds (a non-TC case never looks at
        # the TC detectors beyond the case_types gate)
        ok_any = ok_all and ((not any_c) or any([_clause(case, c, missing) for c in any_c]))
        if not (ok_all and ok_any):
            not_fired.append(rule["id"])
            # `missing` holds only clauses evaluated before the first real False (short-circuit)
            unavailable.update({k: v for k, v in missing.items() if k not in unavailable})
            continue
        spec = rule["trigger_text"]
        names = {c["detector"] for c in trig.get("any", []) + trig.get("all", [])}
        names |= set(re.findall(r"\{([a-z_0-9]+)\.", spec if isinstance(spec, str)
                                else " ".join(p["text"] for p in spec)))
        values = {n: run_detector(case, n) for n in names if n in DETECTORS}
        cost = rule["cost"]
        if cost["method"] == "uncalibrated":
            cost_text = cost["text"]
        else:
            try:
                cost_text = _render(cost["text_template"], dict(cost_vars(rule["id"], case)))
            except Exception as e:                          # noqa: BLE001  advice must not crash
                cost_text = "成本估計失敗（%s: %s）" % (type(e).__name__, e)
        fired.append({
            "id": rule["id"], "title": rule["title"],
            "trigger": _render_trigger(spec, values),
            "recommendation": rule["recommendation"], "how": list(rule["how"]),
            "cost": {"method": cost["method"], "text": cost_text},
            "evidence": dict(rule["evidence"]), "side_effects": rule["side_effects"],
            "footer": doc["footer"],
            "measured": {k: v for k, v in values.items() if v.get("available")},
        })
    ct = run_detector(case, "case_types")
    return {"case": case.name, "surface": surface, "advice": fired, "not_fired": not_fired,
            "case_types": ct.get("types") if ct.get("available") else None,
            "unavailable": unavailable, "footer": doc["footer"]}


def advise_case_id(case_id, site=None, needs=()):
    """Entry for the `wtune tune` hook: registry case id -> advice dict."""
    return advise(build_case(case_id, site=site, needs=needs))


# --------------------------------------------------------------------------------------------
# rendering (Traditional Chinese; the post report's own captions are zh+en, its chrome en)
# --------------------------------------------------------------------------------------------
def render_text(result):
    out = ["=== wtune advise: %s ===" % result["case"]]
    if result.get("case_types"):
        out.append("個案類型：%s" % ", ".join(result["case_types"]))
    if not result["advice"]:
        out.append("沒有規則觸發。")
    for i, m in enumerate(result["advice"], 1):
        out += ["", "[%d] %s  (%s)" % (i, m["title"], m["id"]),
                "  觸發：%s" % m["trigger"],
                "  建議：%s" % m["recommendation"], "  怎麼做："]
        out += ["    - %s" % h for h in m["how"]]
        out += ["  預估成本（%s）：%s" % (m["cost"]["method"], m["cost"]["text"]),
                "  證據：%s [[%s]]" % (m["evidence"]["text"], m["evidence"]["slug"]),
                "  已知副作用：%s" % m["side_effects"],
                "  ※ %s" % m["footer"]]
    if result["unavailable"]:
        out += ["", "未能評估的偵測器（輸入缺失，不等於「不觸發」）："]
        out += ["  - %s：%s" % (k, v) for k, v in sorted(result["unavailable"].items())]
    return "\n".join(out)


def render_html(result):
    import html
    if not result["advice"]:
        return ""
    parts = ["<h2>Advice 建議（advice only — not applied）</h2>"]
    for m in result["advice"]:
        parts.append("<div class='warnbox'><strong>%s</strong> <span class='stamp'>(%s)</span>"
                     "<p>觸發：%s</p><p>建議：%s</p><p>怎麼做：</p><ul>%s</ul>"
                     "<p>預估成本（%s）：%s</p><p>證據：%s <code>[[%s]]</code></p>"
                     "<p>已知副作用：%s</p><p class='stamp'>%s</p></div>" % (
                         html.escape(m["title"]), html.escape(m["id"]), html.escape(m["trigger"]),
                         html.escape(m["recommendation"]),
                         "".join("<li>%s</li>" % html.escape(h) for h in m["how"]),
                         html.escape(m["cost"]["method"]), html.escape(m["cost"]["text"]),
                         html.escape(m["evidence"]["text"]), html.escape(m["evidence"]["slug"]),
                         html.escape(m["side_effects"]), html.escape(m["footer"])))
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="wtune advise",
        description="Advice-only: detect case conditions and print costed, evidence-backed "
                    "suggestions for WRF modes/workflows (WRFDA, nudging, moving nest, tc.exe, "
                    "WRF-Chem/Hydro ...). Applies nothing (decision 0061/0056).")
    p.add_argument("case", help="registry case id (campaign/cases.json), a case/run directory, "
                                "or a wtune project directory")
    p.add_argument("--need", action="append", default=[], metavar="chem|hydro|surge",
                   help="declare a need (repeatable or comma list): chem | hydro | surge")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--site", default=None,
                   help="registry site overlay (default: the machine's, built-in scc)")
    p.add_argument("--hurdat2", default=None, help="HURDAT2 file (best track; storm auto-found "
                   "when --storm is omitted); also $WTUNE_HURDAT2")
    p.add_argument("--storm", default=None, help="storm id, e.g. AL122005, or NAME / NAME+YEAR "
                   "(KATRINA, KATRINA2005; matched in --hurdat2 within the case window)")
    p.add_argument("--bdeck", default=None, help="ATCF b-deck file (preferred over HURDAT2)")
    p.add_argument("--post-dir", default=None,
                   help="a `wtune post tc` output dir holding met_*/*.tcst (verification rules)")
    p.add_argument("--rules", default=None, help="alternate rules JSON")
    p.add_argument("--project", default=None,
                   help="wtune project dir whose project.yaml storm.id/storm.hurdat2 are used when "
                        "--storm/--hurdat2 are absent (default: $WTUNE_PROJECT, else a "
                        "project.yaml at/above cwd)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    needs = [n.strip() for item in args.need for n in item.split(",") if n.strip()]
    bad = [n for n in needs if n not in NEEDS]
    if bad:
        print("wtune advise: unknown --need %s (choose from %s)" % (",".join(bad), ",".join(NEEDS)),
              file=sys.stderr)
        return 2
    try:
        case = build_case(args.case, site=args.site, needs=needs, hurdat2=args.hurdat2,
                          storm=args.storm, bdeck=args.bdeck, post_tc_dir=args.post_dir,
                          project=args.project)
        result = advise(case, "advise", load_rules(args.rules))
    except Exception as e:                                  # noqa: BLE001  stdout stays valid
        msg = e.args[0] if isinstance(e, AdviseError) and e.args else "%s: %s" % (type(e).__name__, e)
        print("wtune advise: %s" % msg, file=sys.stderr)
        if args.json:
            print(json.dumps({"case": args.case, "advice": [], "unavailable": {}, "error": msg},
                             ensure_ascii=False))
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str) if args.json
          else render_text(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())

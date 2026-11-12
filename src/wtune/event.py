"""wtune event: a plain-language event description (event.txt) turned into a checked
event.yaml, and the observation plan WTune derives from it.

    wtune event init  event.txt            -> event.yaml template (source_sha256 filled in)
    wtune event vocab                      -> the allowed types, verify variables, obs kinds
    wtune event check event.yaml           -> schema, vocabulary, quotes, storm ids
    wtune event plan  event.yaml --run-dir D | --start/--end/--bbox

event.txt is written by the user; an agent fills event.yaml from it. Every value carries
`from:`, a verbatim quote of event.txt, so `check` can reject anything the text does not say.
The plan is computed here, not by the agent: each verify variable maps to an observation kind,
and each kind to a fetch module (`module`), to data an agent must find and convert to the
standard format (`agent_fetch`), or to `unsupported` when WTune has no comparison for it.
"""
import argparse
import hashlib
import os
import re
import sys

import yaml

SCHEMA = "wtune-event/1"

TYPES = {
    "tropical_cyclone": "hurricane / typhoon / tropical storm",
    "heavy_rain": "extreme or flooding rainfall",
    "mesoscale_convection": "thunderstorms, MCS, supercells, squall lines",
    "severe_wind": "derecho, damaging convective gusts",
    "frontal": "cold/warm/stationary front, Meiyu-Baiu front",
    "extratropical_cyclone": "mid-latitude low, bomb cyclone",
    "winter_storm": "snowstorm, freezing rain, lake-effect snow",
    "cold_air_outbreak": "cold surge, arctic outbreak",
    "heat_wave": "heat dome, extreme heat",
    "downslope_wind": "foehn, chinook, bora, mountain-wave windstorm",
    "fire_weather": "wildfire spread weather",
    "fog_low_cloud": "radiation / advection fog, stratus",
    "sea_breeze": "sea / lake breeze circulation",
    "urban": "urban heat island, city effects",
    "land_atmosphere": "soil moisture / surface flux feedback",
    "orographic": "terrain-forced rain or flow",
    "monsoon": "monsoon onset, rain band",
    "atmospheric_river": "atmospheric river, plume of vapour transport",
    "dust_air_quality": "dust storm, aerosol, air pollution (WRF-Chem)",
    "boundary_layer": "PBL / LES studies",
    "regional_climate": "multi-month or multi-year runs, seasonal bias",
    "idealized": "idealized case, no real-world observations",
}

# verify variable -> observation kind it is compared with
VERIFY = {
    "track": "best_track", "intensity": "best_track", "wind_radii": "best_track",
    "precip_1h": "gridded_precip", "precip_24h": "gridded_precip",
    "precip_total": "gridded_precip", "precip_gauge": "rain_gauges",
    "reflectivity": "radar",
    "t2": "stations", "t2_max": "stations", "t2_min": "stations", "td2": "stations",
    "rh2": "stations", "wind10": "stations", "mslp": "stations",
    "gust": "stations_extra", "visibility": "stations_extra",
    "z500": "reanalysis", "ivt": "reanalysis", "upper_air": "reanalysis",
    "sounding": "soundings", "pbl_height": "soundings",
    "cloud_top": "satellite_ir", "low_cloud": "satellite_ir",
    "snowfall": "snow", "snow_depth": "snow",
    "aod": "aerosol", "pm10": "air_quality", "pm25": "air_quality",
    "surface_flux": "flux_tower", "soil_moisture": "soil_moisture", "sst": "sst",
    "monthly_precip": "monthly_climate", "monthly_t2": "monthly_climate",
}

OBS_KINDS = {
    "best_track": "TC best track (ATCF b-deck / HURDAT2 / IBTrACS)",
    "gridded_precip": "gridded precipitation analysis (Stage IV, IMERG, national QPE)",
    "rain_gauges": "rain-gauge totals at points",
    "radar": "radar reflectivity composite",
    "stations": "surface stations: T2, Td2, RH2, 10 m wind, MSLP",
    "stations_extra": "surface stations: gusts, visibility",
    "reanalysis": "ERA5 pressure-level / column fields",
    "soundings": "radiosonde profiles",
    "satellite_ir": "geostationary IR brightness temperature",
    "snow": "snowfall / snow depth",
    "aerosol": "satellite aerosol optical depth",
    "air_quality": "PM10 / PM2.5 stations",
    "flux_tower": "surface sensible / latent heat flux",
    "soil_moisture": "soil moisture",
    "sst": "sea / lake surface temperature",
    "monthly_climate": "monthly reanalysis and precipitation climatology",
}

# kind -> (WTune comparison that consumes it, or None)
COMPARISON = {
    "best_track": "MET tc_pairs (post tc)",
    "gridded_precip": "MET grid_stat (precip recipe)",
    "radar": "radar composite -> grid_stat (convection recipe)",
    "stations": "MET point_stat (precip / terrain recipes); per-station series and "
                "statistics: `wtune post stations`",
    "reanalysis": "MET grid_stat on z500 (synoptic recipe)",
    "soundings": "Skew-T overlay (pbl recipe)",
    "monthly_climate": "monthly box means, seasonal bias (climate recipe)",
    "rain_gauges": "MET point_stat on gauge totals (precip recipe)",
    "stations_extra": "MET point_stat on UPP gust / Stoelinga-Warner visibility (precip recipe)",
    "satellite_ir": "MET grid_stat of OLR-derived Tb, cold tops (cloud recipe)",
}
# kinds the recipes read from an agent's folder (post/agent_obs.py); other agent-fetched data
# is kept with its provenance but no recipe reads it yet.
AGENT_READERS = ("gridded_precip", "rain_gauges", "radar", "satellite_ir", "stations_extra",
                 "stations")
AGENT_FORMAT_NOTE = ("write <project>/input/obs/<name>/ in format wtune-obs/1, then "
                     "`wtune event obs-check`")
BDECK_NOTE = "write an ATCF b-deck and pass `wtune post recipe --bdeck FILE`"
AGENT_FILES_NOTE = "no recipe reads this kind"

# Stage IV / NEXRAD are chosen when the domain centre lies over the lower 48 states (a box
# containing a corner of Canada or Mexico is fine: the comparisons mask to valid obs).
CONUS = (24.0, 49.5, -125.0, -66.0)
NHC_BASINS = ("AL", "EP", "CP")


class EventError(Exception):
    pass


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _norm(text):
    return re.sub(r"\s+", " ", text or "").strip()


def template(txt_path):
    return {"schema": SCHEMA, "source": os.path.basename(txt_path),
            "source_sha256": sha256_file(txt_path),
            "event": {"value": None, "from": None}, "types": [], "storms": [], "region":
            {"value": None, "from": None}, "verify": [], "diagnose": [], "obs_requests": [],
            "local_obs": [], "questions": []}


def load(path):
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    if not isinstance(doc, dict):
        raise EventError("%s: top level must be a mapping" % path)
    return doc


def _items(doc, key):
    v = doc.get(key) or []
    return v if isinstance(v, list) else [v]


def check(doc, txt=None, txt_sha=None, hurdat2=None):
    """(errors, warnings) for an event.yaml dict. txt: the event.txt text for the quote and
    sha checks; hurdat2: a HURDAT2 file to resolve storm names to ATCF ids."""
    err, warn = [], []
    if doc.get("schema") != SCHEMA:
        err.append("schema: must be %r" % SCHEMA)
    if txt_sha and doc.get("source_sha256") != txt_sha:
        err.append("source_sha256: does not match event.txt (%s) -- the text changed; "
                   "translate it again from `wtune event init`" % txt_sha)
    body = _norm(txt) if txt is not None else None

    def quoted(where, item):
        q = item.get("from") if isinstance(item, dict) else None
        if not q:
            err.append("%s: missing `from` (a verbatim quote of event.txt)" % where)
        elif body is not None and _norm(q) not in body:
            err.append("%s: `from` %r is not in event.txt" % (where, q))
        return q or ""

    ev = doc.get("event") or {}
    if not ev.get("value"):
        err.append("event.value: empty")
    else:
        quoted("event", ev)
    for i, t in enumerate(_items(doc, "types")):
        if not isinstance(t, dict) or t.get("value") not in TYPES:
            err.append("types[%d]: %r is not a type (wtune event vocab)"
                       % (i, t.get("value") if isinstance(t, dict) else t))
            continue
        quoted("types[%d]" % i, t)
    for i, v in enumerate(_items(doc, "verify")):
        if not isinstance(v, dict) or v.get("value") not in VERIFY:
            err.append("verify[%d]: %r is not a verify variable (wtune event vocab)"
                       % (i, v.get("value") if isinstance(v, dict) else v))
            continue
        quoted("verify[%d]" % i, v)
    for i, d in enumerate(_items(doc, "diagnose")):
        quoted("diagnose[%d]" % i, d)
    for i, r in enumerate(_items(doc, "obs_requests")):
        if not isinstance(r, dict) or r.get("kind") not in OBS_KINDS:
            err.append("obs_requests[%d]: kind %r is not an obs kind (wtune event vocab)"
                       % (i, r.get("kind") if isinstance(r, dict) else r))
            continue
        if not r.get("value"):
            err.append("obs_requests[%d]: needs value (the source as the user names it)" % i)
        quoted("obs_requests[%d]" % i, r)
    for i, o in enumerate(_items(doc, "local_obs")):
        if not isinstance(o, dict) or not o.get("path"):
            err.append("local_obs[%d]: needs path" % i)
            continue
        q = quoted("local_obs[%d]" % i, o)
        if q and o["path"] not in q:
            err.append("local_obs[%d]: path %r is not inside its quote" % (i, o["path"]))
        if o.get("kind") not in OBS_KINDS:
            err.append("local_obs[%d]: kind %r is not an obs kind" % (i, o.get("kind")))
    for i, s in enumerate(_items(doc, "storms")):
        if not isinstance(s, dict) or not s.get("name"):
            err.append("storms[%d]: needs name" % i)
            continue
        q = quoted("storms[%d]" % i, s)
        sid = s.get("id")
        if sid and sid.upper() not in q.upper():
            err.append("storms[%d]: id %r is not in its quote -- leave id empty unless "
                       "event.txt states it; `check --hurdat2` resolves names" % (i, sid))
        if hurdat2 and s.get("year"):
            from wtune.post.tc import TCError, resolve_storm
            try:
                found = resolve_storm(hurdat2, "%s%s" % (s["name"], s["year"]))
                if sid and sid.upper() != found:
                    err.append("storms[%d]: id %s, but HURDAT2 has %s for %s %s"
                               % (i, sid, found, s["name"], s["year"]))
                else:
                    s["id"] = found
            except TCError as e:
                warn.append("storms[%d]: %s" % (i, e))
    region = doc.get("region") or {}
    if region.get("value"):
        quoted("region", region)
    types = {t.get("value") for t in _items(doc, "types") if isinstance(t, dict)}
    if not types and not doc.get("questions"):
        err.append("types: empty and no questions -- say what is unclear in `questions`")
    real = [k for k in ("verify", "obs_requests", "local_obs") if _items(doc, k)]
    if "idealized" in types and real:
        err.append("types: idealized, but %s compare with observations -- an idealized case "
                   "has none; drop idealized, or move the items to diagnose or questions"
                   % ", ".join(real))
    if "tropical_cyclone" in types and not _items(doc, "storms") and not doc.get("questions"):
        err.append("storms: tropical_cyclone without a storm name -- name it or ask in "
                   "`questions`")
    if not _items(doc, "verify") and not _items(doc, "diagnose"):
        warn.append("verify and diagnose are both empty: nothing to compare or draw")
    return err, warn


def _centre_in(bbox, box):
    lat, lon = (bbox[0] + bbox[1]) / 2.0, (bbox[2] + bbox[3]) / 2.0
    return box[0] <= lat <= box[1] and box[2] <= lon <= box[3]


def plan(doc, start=None, end=None, bbox=None):
    """Rows {need, kind, source, status, note} for every verify variable, obs request and
    local file. start/end (datetime) and bbox (lat0, lat1, lon0, lon1) pick the datasets;
    unknown window -> region-specific modules are not chosen."""
    types = {t.get("value") for t in _items(doc, "types") if isinstance(t, dict)}
    conus = bbox is not None and _centre_in(bbox, CONUS)
    rows = []

    def row(need, kind, source, status, note=""):
        if status in ("agent_fetch", "local"):
            extra = (AGENT_FORMAT_NOTE if kind in AGENT_READERS else
                     BDECK_NOTE if kind == "best_track" else AGENT_FILES_NOTE)
            note = "; ".join(x for x in (note, extra) if x)
        rows.append(dict(need=need, kind=kind, source=source, status=status, note=note))

    if "idealized" in types:
        for v in _items(doc, "verify"):
            row(v.get("value"), VERIFY.get(v.get("value"), "?"), "-", "unsupported",
                "idealized case: no real-world observations")
        for r in _items(doc, "obs_requests"):
            row(r.get("value"), r.get("kind"), "-", "unsupported",
                "idealized case: no real-world observations")
        return rows

    storms = [s for s in _items(doc, "storms") if isinstance(s, dict)]
    for v in _items(doc, "verify"):
        var = v.get("value")
        kind = VERIFY.get(var)
        if kind is None:
            continue
        if kind == "best_track":
            ids = [s.get("id") for s in storms if s.get("id")]
            if ids and all(i[:2].upper() in NHC_BASINS for i in ids):
                row(var, kind, "NHC b-deck, else project HURDAT2", "module")
            elif not ids:
                row(var, kind, "-", "agent_fetch",
                    "storm id unknown: resolve with `check --hurdat2`, or find the basin's "
                    "best track (IBTrACS covers all basins)")
            else:
                row(var, kind, "IBTrACS / regional agency", "agent_fetch",
                    "basin outside NHC: no fetch module")
        elif kind == "gridded_precip":
            if conus:
                row(var, kind, "stage4", "module")
            else:
                row(var, kind, "IMERG or a national QPE", "agent_fetch",
                    "Stage IV covers CONUS only" if bbox else "region unknown")
        elif kind == "rain_gauges":
            row(var, kind, "national gauge network", "agent_fetch", "no fetch module")
        elif kind == "stations_extra":
            row(var, kind, "national station network (gust, visibility)", "agent_fetch",
                "no fetch module")
        elif kind == "satellite_ir":
            row(var, kind, "geostationary IR (GOES, Himawari, GridSat)", "agent_fetch",
                "no fetch module; scores count cold tops, low cloud shows only on the map"
                if var == "low_cloud" else "no fetch module")
        elif kind == "radar":
            if conus:
                row(var, kind, "nexrad", "module")
            else:
                row(var, kind, "national radar composite", "agent_fetch",
                    "NEXRAD/MRMS cover the US only" if bbox else "region unknown")
        elif kind == "stations":
            row(var, kind, "stations (ISD-Lite, global)", "module")
        elif kind == "reanalysis":
            if var == "z500":
                row(var, kind, "era5", "module")
            else:
                row(var, kind, "ERA5", "unsupported",
                    "the ERA5 comparison covers z500 only")
        elif kind == "soundings":
            row(var, kind, "igra (global)", "module",
                "" if var == "sounding" else "radiosonde bulk-Ri PBL height (pbl recipe)")
        elif kind == "monthly_climate":
            row(var, kind, "era5_monthly + gpcp_monthly", "module")
        else:
            row(var, kind, "-", "unsupported", "WTune has no comparison for %s" % kind)

    for r in _items(doc, "obs_requests"):
        kind = r.get("kind")
        if kind in COMPARISON:
            row(r.get("value"), kind, r.get("value"), "agent_fetch",
                "named by the user; find it, convert to the %s format" % kind)
        else:
            row(r.get("value"), kind, r.get("value"), "unsupported",
                "WTune has no comparison for %s" % kind)
    for o in _items(doc, "local_obs"):
        row(o.get("path"), o.get("kind"), o.get("path"), "local",
            "user file; convert to the %s format if it is not already" % o.get("kind"))
    for d in _items(doc, "diagnose"):
        row(d.get("value"), "model_only", "-", "model_only", "figure from the run itself")
    return rows


def _window(args):
    from wtune.post.sources import resolve_window
    try:
        return resolve_window(args)
    except ValueError:
        return None, None, None


def build_parser():
    p = argparse.ArgumentParser(prog="wtune event", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="event_command", required=True)
    s = sub.add_parser("init", help="write an event.yaml template for EVENT_TXT")
    s.add_argument("txt")
    s.add_argument("-o", "--out", help="default: event.yaml next to EVENT_TXT")
    s.add_argument("--force", action="store_true", help="overwrite an existing event.yaml")
    sub.add_parser("vocab", help="print the allowed types, verify variables and obs kinds")
    s = sub.add_parser("check", help="validate event.yaml against event.txt")
    s.add_argument("yaml")
    s.add_argument("--txt", help="default: `source` next to the yaml")
    s.add_argument("--hurdat2", help="resolve storm NAME+year to an ATCF id")
    s = sub.add_parser("obs-check", help="validate agent-fetched observation folders "
                                         "(format wtune-obs/1)")
    s.add_argument("dirs", nargs="*", help="dataset folders (default: every wtune-obs/1 "
                                           "folder under --project's input/obs)")
    s.add_argument("--project", help="default: the current directory")
    s = sub.add_parser("obs-convert", help="convert a provider's station file to a wtune-obs/1 "
                                           "folder")
    cs = s.add_subparsers(dest="convert_kind", required=True)
    a = cs.add_parser("asos", help="Iowa Environmental Mesonet ASOS CSV -> kind stations "
                                   "(degF -> degC, knot -> m/s)")
    a.add_argument("csv")
    a.add_argument("--out", required=True, help="dataset folder to write")
    a.add_argument("--dataset", required=True)
    a.add_argument("--provider", required=True)
    a.add_argument("--url", required=True)
    a.add_argument("--license", required=True)
    a.add_argument("--citation", default="")
    a.add_argument("--station", action="append", default=[], metavar="ID=LAT,LON,ELEV_M",
                   help="position of a station the CSV has no lat/lon/elevation columns for")
    s = sub.add_parser("plan", help="observation plan for event.yaml")
    s.add_argument("yaml")
    s.add_argument("--run-dir")
    from wtune.post.sources import add_window_arguments
    add_window_arguments(s)
    return p


def _txt_for(args, doc):
    if args.txt:
        return args.txt
    src = doc.get("source")
    if src:
        cand = os.path.join(os.path.dirname(os.path.abspath(args.yaml)), src)
        if os.path.exists(cand):
            return cand
    return None


def _obs_check(args):
    from wtune.post import agent_obs
    dirs = args.dirs or [d for _n, d, _p in agent_obs.datasets(args.project or ".")]
    if not dirs:
        print("error: no wtune-obs/1 folder under %s/input/obs" % (args.project or "."),
              file=sys.stderr)
        return 1
    bad = 0
    for d in dirs:
        err, warn = agent_obs.check(d)
        for w in warn:
            print("warning: %s: %s" % (d, w))
        for e in err:
            print("error: %s: %s" % (d, e))
        print("obs check %s: %s" % (d, "FAIL" if err else "OK"))
        bad += bool(err)
    return 1 if bad else 0


def _obs_convert(args):
    from wtune.post import agent_obs
    meta = {}
    for item in args.station:
        try:
            sid, rest = item.split("=", 1)
            lat, lon, elev = (float(v) for v in rest.split(","))
        except ValueError:
            print("error: --station expects ID=LAT,LON,ELEV_M, got %r" % item, file=sys.stderr)
            return 1
        meta[sid.strip()] = (lat, lon, elev)
    prov = dict(dataset=args.dataset, provider=args.provider, url=args.url,
                license=args.license, citation=args.citation)
    try:
        path, n = agent_obs.from_asos(args.csv, args.out, prov, meta)
    except (OSError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    print("wrote %s (%d rows)" % (path, n))
    return _obs_check(argparse.Namespace(dirs=[args.out], project=None))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.event_command == "init":
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.txt)), "event.yaml")
        if os.path.exists(out) and not args.force:
            print("error: %s exists (--force to overwrite)" % out, file=sys.stderr)
            return 1
        with open(out, "w") as fh:
            yaml.safe_dump(template(args.txt), fh, sort_keys=False, allow_unicode=True)
        print(out)
        return 0
    if args.event_command == "vocab":
        for title, table in (("types", TYPES), ("obs kinds", OBS_KINDS)):
            print("%s:" % title)
            for k, v in table.items():
                print("  %-22s %s" % (k, v))
        print("verify variables (-> obs kind):")
        for k, v in VERIFY.items():
            print("  %-22s %s" % (k, v))
        return 0
    if args.event_command == "obs-check":
        return _obs_check(args)
    if args.event_command == "obs-convert":
        return _obs_convert(args)
    try:
        doc = load(args.yaml)
    except (OSError, yaml.YAMLError, EventError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    if args.event_command == "check":
        txt_path = _txt_for(args, doc)
        txt = open(txt_path).read() if txt_path else None
        err, warn = check(doc, txt=txt, txt_sha=sha256_file(txt_path) if txt_path else None,
                          hurdat2=args.hurdat2)
        if txt_path is None:
            warn.insert(0, "event.txt not found: quotes and sha not checked")
        for w in warn:
            print("warning: %s" % w)
        for e in err:
            print("error: %s" % e)
        print("event check: %s (%d error(s), %d warning(s))"
              % ("FAIL" if err else "OK", len(err), len(warn)))
        return 1 if err else 0
    start, end, bbox = _window(args)
    rows = plan(doc, start, end, bbox)
    if bbox is None:
        print("note: no --run-dir or --start/--end/--bbox: region-specific datasets not chosen")
    for r in rows:
        print("%-12s %-16s %-28s %-28s %s" % (r["status"], r["kind"], str(r["need"])[:28],
                                              str(r["source"])[:28], r["note"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

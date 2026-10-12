"""post/fetch/igra.py -- radiosonde soundings from NOAA NCEI IGRA v2 (Integrated Global
Radiosonde Archive), for the pbl recipe's model-vs-observed profile comparison.

Fetch-module contract (NAME/HELP/PROVENANCE/add_arguments/run). Output:
  <project>/input/obs/igra/stations.csv      selected stations (id, lat, lon, elev_m, name)
  <project>/input/obs/igra/<STATION>.csv     one row per sounding level in the time window
  <project>/input/obs/igra/provenance.json   PROVENANCE + URLs, UTC fetch time, sha256s, window

IGRA2 ships one zip per station holding the WHOLE period of record (data-por, tens of MB for
a station active since the 1940s), a single deflate stream of text in chronological order.
This module therefore streams the zip and stops inflating as soon as a sounding is later than
the requested window, instead of downloading/unpacking the tail it does not need. Nothing is
kept besides the parsed window; re-running skips a station whose CSV sha256 matches the value
recorded in provenance.json for the same window.

Units in the CSV (IGRA native values converted): pressure hPa, height m (geopotential, MSL),
temp_C, rh_pct, dewpoint_C (= temp - dewpoint depression), wdir_deg, wspd_ms; missing = empty.
Levels IGRA quality assurance flagged (-8888) or that were missing (-9999) are empty.

Format reference: IGRA2 doc/igra2-data-format.txt (fixed columns). Data are public domain;
cite Durre et al. (2006, J. Climate 19, 53-68) and Durre et al. (2018, JTECH 35, 1753-1770).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone

NAME = "igra"
HELP = "NOAA NCEI IGRA v2 radiosonde soundings (00Z/12Z) for stations inside the WRF domain"

BASE = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive"
STATION_LIST_URL = BASE + "/doc/igra2-station-list.txt"
POR_URL = BASE + "/access/data-por/%s-data.txt.zip"

PROVENANCE = {
    "dataset": "Integrated Global Radiosonde Archive (IGRA) Version 2",
    "provider": "NOAA National Centers for Environmental Information (NCEI)",
    "url": BASE + "/",
    "license": "US Government work, public domain (NOAA NCEI); no registration",
    "citation": "Durre, I., R. S. Vose, D. B. Wuertz, 2006: Overview of the Integrated Global "
                "Radiosonde Archive. J. Climate, 19, 53-68; Durre et al., 2018: Enhancements "
                "to the IGRA. J. Atmos. Oceanic Technol., 35, 1753-1770; "
                "doi:10.7289/V5X63K0Q",
    "coverage_time": "1905-present, mostly 00Z/12Z launches",
    "coverage_space": "global, ~2800 stations",
    "resolution": "sounding levels (mandatory + significant), ~12 h",
    "access": "public, no credentials",
}

CSV_FIELDS = ["time_utc", "hour", "lvltyp1", "lvltyp2", "pressure_hPa", "height_m", "temp_C",
              "rh_pct", "dewpoint_C", "wdir_deg", "wspd_ms"]


# ---------------------------------------------------------------------------------------
# parsing (pure functions, unit-tested with small fixtures)
# ---------------------------------------------------------------------------------------

def parse_station_list(text):
    """igra2-station-list.txt fixed columns -> [{id, lat, lon, elev_m, name, first, last}]."""
    out = []
    for line in text.splitlines():
        if len(line) < 72 or not line.strip():
            continue
        try:
            out.append(dict(id=line[0:11].strip(), lat=float(line[12:20]), lon=float(line[21:30]),
                            elev_m=float(line[31:37]), name=line[38:71].strip(),
                            first=int(line[72:76]), last=int(line[77:81])))
        except ValueError:
            continue
    return out


def _int(field):
    try:
        return int(field)
    except ValueError:
        return None


def _valid(v):
    return v is not None and v > -8000     # -8888 (QA-removed) and -9999 (missing)


def parse_header(line):
    """'#ID YYYY MM DD HH RELT NUMLEV ...' -> (datetime|None, hour, numlev). hour 99 = unknown."""
    year, month, day, hour = _int(line[13:17]), _int(line[18:20]), _int(line[21:23]), _int(line[24:26])
    numlev = _int(line[32:36]) or 0
    try:
        t = datetime(year, month, day, hour if hour is not None and hour < 24 else 0,
                     tzinfo=timezone.utc)
    except (TypeError, ValueError):
        t = None
    return t, hour, numlev


def parse_level(line):
    """One IGRA data record -> dict of SI-ish values (None for missing)."""
    press = _int(line[9:15])
    gph = _int(line[16:21])
    temp = _int(line[22:27])
    rh = _int(line[28:33])
    dpdp = _int(line[34:39])
    wdir = _int(line[40:45])
    wspd = _int(line[46:51])
    temp_c = temp / 10.0 if _valid(temp) else None
    return dict(
        lvltyp1=line[0:1], lvltyp2=line[1:2],
        pressure_hPa=press / 100.0 if _valid(press) else None,
        height_m=float(gph) if _valid(gph) else None,
        temp_C=temp_c,
        rh_pct=rh / 10.0 if _valid(rh) else None,
        dewpoint_C=(temp_c - dpdp / 10.0) if (temp_c is not None and _valid(dpdp)) else None,
        wdir_deg=float(wdir) if _valid(wdir) and 0 <= wdir <= 360 else None,
        wspd_ms=wspd / 10.0 if _valid(wspd) and wspd >= 0 else None)


def iter_soundings(lines, start, end, hours=(0, 12)):
    """Yields (time, [level dicts]) for soundings with start <= time <= end (UTC datetimes) and
    nominal hour in `hours` (None = any). Stops (returns) at the first sounding later than end,
    relying on IGRA's chronological ordering, so a streaming caller can stop reading."""
    cur = None
    for line in lines:
        if line.startswith("#"):
            if cur is not None:
                yield cur
            cur = None
            t, hour, numlev = parse_header(line)
            if t is None:
                continue
            if t > end + timedelta(hours=1):
                return
            if t < start or (hours is not None and hour not in hours):
                continue
            cur = (t, [])
        elif cur is not None and line.strip():
            cur[1].append(parse_level(line))
    if cur is not None:
        yield cur


def write_station_csv(path, soundings):
    """Atomic write of the long-format CSV; returns the file's sha256."""
    tmp = path + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for t, levels in soundings:
            for lv in levels:
                w.writerow([t.strftime("%Y-%m-%dT%H:%M:%SZ"), t.hour, lv["lvltyp1"], lv["lvltyp2"]]
                           + ["" if lv[k] is None else round(lv[k], 2) for k in CSV_FIELDS[4:]])
    os.replace(tmp, path)
    return sha256_file(path)


def read_soundings(path):
    """{datetime(UTC): {field: np.ndarray}} from a station CSV written by this module, fields =
    CSV_FIELDS[4:] (pressure_hPa ... wspd_ms), levels in file order, NaN where missing."""
    import numpy as np
    by_time = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = datetime.strptime(row["time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            rec = by_time.setdefault(t, {k: [] for k in CSV_FIELDS[4:]})
            for k in CSV_FIELDS[4:]:
                rec[k].append(float(row[k]) if row[k] != "" else float("nan"))
    return {t: {k: np.asarray(v, dtype=float) for k, v in rec.items()} for t, rec in by_time.items()}


# ---------------------------------------------------------------------------------------
# streaming zip
# ---------------------------------------------------------------------------------------

def stream_zip_lines(url, timeout=120):
    """Yield decoded text lines of the single member of a remote zip, inflating incrementally
    (closing the generator closes the HTTP connection, so an early stop saves the download)."""
    req = urllib.request.Request(url, headers={"User-Agent": "wtune-post-fetch-igra"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        hdr = resp.read(30)
        if hdr[:4] != b"PK\x03\x04":
            raise RuntimeError("%s: not a zip local-file header" % url)
        method = int.from_bytes(hdr[8:10], "little")
        resp.read(int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little"))
        dec = zlib.decompressobj(-15) if method == 8 else None
        buf = b""
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            data = dec.decompress(chunk) if dec else chunk
            buf += data
            *lines, buf = buf.split(b"\n")
            for ln in lines:
                yield ln.decode("ascii", "replace")
            if dec is not None and dec.eof:
                break
        if dec is not None and not dec.eof:     # connection cut mid-member
            raise RuntimeError("%s: stream ended before the zip member was complete" % url)
        if buf:
            yield buf.decode("ascii", "replace")


def _http_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "wtune-post-fetch-igra"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode("ascii", "replace")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------
# window / bbox derivation and station selection
# ---------------------------------------------------------------------------------------

def _parse_time(s):
    for fmt in ("%Y%m%d%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d_%H:%M:%S", "%Y-%m-%dT%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError("cannot parse time %r (use YYYYmmddHH)" % s)


def window_from_run(run_dir, domain=1):
    """(start, end, bbox=(lat0, lat1, lon0, lon1) of the domain INTERIOR) from wrfout_d0N_*."""
    import netCDF4
    import numpy as np
    from wtune.post import caseinfo
    from wtune.post.extract import list_frames
    frames = list_frames(run_dir, domain=domain)
    if not frames:
        raise FileNotFoundError("no wrfout_d%02d_* in %s" % (domain, run_dir))
    start = datetime.strptime(frames[0][0], caseinfo.WRF_TIME_FMT).replace(tzinfo=timezone.utc)
    end = datetime.strptime(frames[-1][0], caseinfo.WRF_TIME_FMT).replace(tzinfo=timezone.utc)
    with netCDF4.Dataset(frames[0][1]) as ds:
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
    b = caseinfo._interior_bounds(lat.shape[0], lat.shape[1], caseinfo.spec_bdy_width(run_dir))
    j0, j1, i0, i1 = b if b else (0, lat.shape[0], 0, lat.shape[1])
    return start, end, (float(lat[j0:j1, i0:i1].min()), float(lat[j0:j1, i0:i1].max()),
                        float(lon[j0:j1, i0:i1].min()), float(lon[j0:j1, i0:i1].max()))


def land_stations(run_dir, stations, domain=1):
    """Keep stations whose nearest model grid column is land (LANDMASK == 1) -- a radiosonde on a
    small island or ship-like site samples a marine boundary layer the land-PBL comparison is
    not about. Nearest column by plain squared lat/lon distance on XLAT/XLONG (ample at 1-12 km)."""
    import netCDF4
    import numpy as np
    from wtune.post.extract import list_frames
    frames = list_frames(run_dir, domain=domain)
    with netCDF4.Dataset(frames[0][1]) as ds:
        lat = np.asarray(ds.variables["XLAT"][0])
        lon = np.asarray(ds.variables["XLONG"][0])
        land = np.asarray(ds.variables["LANDMASK"][0]) if "LANDMASK" in ds.variables else None
    if land is None:
        return stations
    keep = []
    for s in stations:
        d2 = (lat - s["lat"]) ** 2 + ((lon - s["lon"]) * math.cos(math.radians(s["lat"]))) ** 2
        j, i = np.unravel_index(np.argmin(d2), d2.shape)
        if land[j, i] == 1:
            keep.append(s)
    return keep


def select_stations(stations, bbox, start, end, max_n=6):
    """Stations inside bbox whose period of record overlaps [start, end]; when more than max_n,
    deterministic farthest-point sampling seeded at the station nearest the bbox centre (an even
    geographic spread rather than the first N in id order)."""
    lat0, lat1, lon0, lon1 = bbox
    cand = [s for s in stations if lat0 <= s["lat"] <= lat1 and lon0 <= s["lon"] <= lon1
            and s["first"] <= end.year and s["last"] >= start.year]
    cand.sort(key=lambda s: s["id"])
    if len(cand) <= max_n:
        return cand
    clat, clon = 0.5 * (lat0 + lat1), 0.5 * (lon0 + lon1)

    def dist(a, b):
        return math.hypot(a[0] - b[0], (a[1] - b[1]) * math.cos(math.radians(0.5 * (a[0] + b[0]))))

    chosen = [min(cand, key=lambda s: dist((s["lat"], s["lon"]), (clat, clon)))]
    while len(chosen) < max_n:
        rest = [s for s in cand if s not in chosen]
        chosen.append(max(rest, key=lambda s: min(dist((s["lat"], s["lon"]), (c["lat"], c["lon"]))
                                                  for c in chosen)))
    return sorted(chosen, key=lambda s: s["id"])


# ---------------------------------------------------------------------------------------
# contract entry points
# ---------------------------------------------------------------------------------------

def add_arguments(parser):
    parser.add_argument("--start", default=None, help="UTC YYYYmmddHH (default: first wrfout time)")
    parser.add_argument("--end", default=None, help="UTC YYYYmmddHH (default: last wrfout time)")
    parser.add_argument("--bbox", default=None, metavar="LAT0,LAT1,LON0,LON1",
                        help="default: interior of --run-dir's domain")
    parser.add_argument("--domain", type=int, default=1, help="wrfout domain for --run-dir (default 1)")
    parser.add_argument("--hours", default="0,12", help="nominal launch hours to keep (default 0,12; "
                        "'all' keeps every sounding)")
    parser.add_argument("--stations", default=None, help="comma-separated IGRA ids "
                        "(default: chosen inside the bbox)")
    parser.add_argument("--max-stations", type=int, default=6, dest="max_stations")


def _resolve_request(args):
    start = end = bbox = None
    if getattr(args, "run_dir", None):
        start, end, bbox = window_from_run(args.run_dir, getattr(args, "domain", 1))
    if args.start:
        start = _parse_time(args.start)
    if args.end:
        end = _parse_time(args.end)
    if args.bbox:
        bbox = tuple(float(x) for x in args.bbox.split(","))
    return start, end, bbox


def run(args):
    start, end, bbox = _resolve_request(args)
    if start is None or end is None:
        print("igra: need --run-dir or both --start and --end")
        return []
    hours = None if args.hours == "all" else tuple(int(h) for h in args.hours.split(","))
    out_dir = os.path.join(args.project, "input", "obs", NAME)
    os.makedirs(out_dir, exist_ok=True)
    prov_path = os.path.join(out_dir, "provenance.json")
    prov = {}
    if os.path.exists(prov_path):
        with open(prov_path) as fh:
            prov = json.load(fh)
    window = dict(start=start.strftime("%Y-%m-%dT%H:%M:%SZ"), end=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  bbox=list(bbox) if bbox else None, hours=list(hours) if hours else "all")
    files_prev = prov.get("files", {}) if prov.get("window") == window else {}

    try:
        stations = parse_station_list(_http_text(STATION_LIST_URL))
    except Exception as e:                      # network or provider failure: report, do not raise
        print("igra: cannot fetch the station list (%s): %s" % (STATION_LIST_URL, e))
        return []
    if args.stations:
        wanted = args.stations.split(",")
        chosen = [s for s in stations if s["id"] in wanted]
    elif bbox:
        if getattr(args, "run_dir", None):
            stations = land_stations(args.run_dir, stations, getattr(args, "domain", 1))
        chosen = select_stations(stations, bbox, start, end, args.max_stations)
    else:
        print("igra: need --bbox, --stations or --run-dir to choose stations")
        return []
    if not chosen:
        print("igra: no IGRA station inside bbox %s with a record covering %s..%s"
              % (bbox, window["start"], window["end"]))
        return []

    written, urls, files = [], [STATION_LIST_URL], {}
    kept_stations = []
    for s in chosen:
        path = os.path.join(out_dir, "%s.csv" % s["id"])
        url = POR_URL % s["id"]
        prev = files_prev.get(os.path.basename(path))
        if prev and os.path.exists(path) and sha256_file(path) == prev.get("sha256"):
            print("igra: %s up to date" % s["id"])
            files[os.path.basename(path)] = prev
            urls.append(url)
            kept_stations.append(s)
            written.append(path)
            continue
        print("igra: streaming %s (%s) ..." % (s["id"], s["name"]))
        try:
            sounds = list(iter_soundings(stream_zip_lines(url), start, end, hours))
        except Exception as e:
            print("igra: %s failed: %s (re-run to resume)" % (s["id"], e))
            continue
        urls.append(url)
        if not sounds:
            print("igra: %s has no soundings for %s..%s hours=%s" % (s["id"], window["start"], window["end"], hours))
            continue
        sha = write_station_csv(path, sounds)
        files[os.path.basename(path)] = dict(sha256=sha, soundings=len(sounds))
        kept_stations.append(s)
        written.append(path)
    if not written:
        return []

    st_path = os.path.join(out_dir, "stations.csv")
    with open(st_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "lat", "lon", "elev_m", "name"])
        for s in kept_stations:
            w.writerow([s["id"], s["lat"], s["lon"], s["elev_m"], s["name"]])
    files["stations.csv"] = dict(sha256=sha256_file(st_path))
    written.append(st_path)
    doc = dict(PROVENANCE)
    doc.update(urls=urls, fetched_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               files=files, window=window)
    with open(prov_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    written.append(prov_path)
    return written


def main(argv=None):                         # standalone: python -m post.fetch.igra --project P ...
    import argparse
    p = argparse.ArgumentParser(description=HELP)
    p.add_argument("--project", required=True)
    p.add_argument("--run-dir", default=None, dest="run_dir")
    add_arguments(p)
    for path in run(p.parse_args(argv)):
        print(path)


if __name__ == "__main__":
    main(sys.argv[1:])

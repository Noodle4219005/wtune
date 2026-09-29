"""post/fetch/stations.py -- surface station observations from NOAA NCEI ISD-Lite (hourly T, Td,
MSLP, wind speed/direction, 1 h and 6 h precipitation; public, no credentials). IEM's ASOS API
is the alternative public source but needs a per-network station list; ISD-Lite works for any
region (CONUS, Puerto Rico, ...) from one station catalogue (isd-history.csv), so it is the
only source here.

Files under <project>/input/obs/stations/: stations.csv (the selected catalogue rows:
id,name,icao,lat,lon,elev_m) and <USAF>-<WBAN>-<YYYY>.gz (the untouched ISD-Lite year files).
Stations are those inside the region whose record period covers the window; when more than
--max-stations qualify, those with an ICAO id (airport ASOS/AWOS, best quality) come first,
then the ones nearest the region centre.
Conversion to MET's 11-column point format is done at verification time by
post/met_pointstat.py (units: ISD-Lite stores T/Td in 0.1 degC, SLP in 0.1 hPa, wind speed in
0.1 m/s, precip in 0.1 mm; -9999 is missing).
"""
from __future__ import annotations

import csv
import io
import math
import os
from datetime import datetime

from post import sources

NAME = "stations"
HELP = "NOAA ISD-Lite hourly surface stations (T, Td, MSLP, wind, precip)"
PROVENANCE = dict(sources.REGISTRY["stations"])

HISTORY = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
LITE = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/%d/%s-%s-%d.gz"


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--max-stations", type=int, default=120)


def parse_history(text):
    """[dict(usaf, wban, name, icao, lat, lon, elev, begin, end)] for rows with a valid position
    (begin/end are datetime; rows with blank coordinates are dropped)."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            lat, lon = float(row["LAT"]), float(row["LON"])
            if lat == 0.0 and lon == 0.0:
                continue
            out.append(dict(usaf=row["USAF"], wban=row["WBAN"], name=row["STATION NAME"].strip(),
                            icao=row["ICAO"].strip(), lat=lat, lon=lon,
                            elev=float(row["ELEV(M)"]) if row["ELEV(M)"].strip() else float("nan"),
                            begin=datetime.strptime(row["BEGIN"], "%Y%m%d"),
                            end=datetime.strptime(row["END"], "%Y%m%d")))
        except (KeyError, ValueError):
            continue
    return out


def select(rows, bbox, start, end, n):
    clat, clon = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
    ok = [r for r in rows if bbox[0] <= r["lat"] <= bbox[1] and bbox[2] <= r["lon"] <= bbox[3]
          and r["begin"] <= end and r["end"] >= start and r["usaf"] != "999999"]
    ok.sort(key=lambda r: (0 if r["icao"] else 1,
                           math.hypot(r["lat"] - clat, (r["lon"] - clon) * math.cos(math.radians(clat)))))
    return ok[:n]


def station_id(r):
    return "%s-%s" % (r["usaf"], r["wban"])


def run(args):
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("stations: %s" % e)
        return []
    body = sources.http_get(HISTORY)
    if body is None:
        print("stations: cannot read %s" % HISTORY)
        return []
    chosen = select(parse_history(body.decode("utf-8", "replace")), bbox, start, end,
                    args.max_stations)
    if not chosen:
        print("stations: no ISD station inside %s with a record covering %s..%s" % (bbox, start, end))
        return []
    base = sources.dataset_dir(args.project, NAME)
    os.makedirs(base, exist_ok=True)
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    written, urls, files = [], [], {}
    years = range(start.year, end.year + 1)
    kept = []
    for r in chosen:
        got = False
        for y in years:
            url = LITE % (y, r["usaf"], r["wban"], y)
            dest = os.path.join(base, "%s-%s-%d.gz" % (r["usaf"], r["wban"], y))
            sha, dl = sources.download(url, dest, known_sha=known.get(os.path.basename(dest)))
            if sha is None:
                continue
            got = True
            written.append(dest)
            files[os.path.basename(dest)] = sha
            if dl:
                urls.append(url)
        if got:
            kept.append(r)
    if not kept:
        print("stations: %d catalogue stations in region but none has an ISD-Lite file for %s"
              % (len(chosen), ", ".join(str(y) for y in years)))
        return []
    cat = os.path.join(base, "stations.csv")
    with open(cat, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "icao", "lat", "lon", "elev_m"])
        for r in kept:
            w.writerow([station_id(r), r["name"], r["icao"], r["lat"], r["lon"], r["elev"]])
    written.append(cat)
    files["stations.csv"] = sources.sha256_file(cat)
    if urls:
        urls.append(HISTORY)
    sources.write_provenance(args.project, NAME, PROVENANCE, urls, files,
                             window=sources.window_dict(start, end), bbox=list(bbox),
                             extra={"n_stations": len(kept)})
    print("stations: %d station(s), %d file(s) under %s" % (len(kept), len(written), base))
    return written

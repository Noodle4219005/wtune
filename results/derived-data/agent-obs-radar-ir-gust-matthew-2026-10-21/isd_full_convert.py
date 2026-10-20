"""NCEI Global Hourly (ISD full) gust and visibility -> wtune-obs/1 stations_extra.

    python3 -I convert.py STATIONS_CSV OUT_DIR START END     (UTC, 'YYYY-MM-DDTHH:MM:SS')

Downloads each station through the NCEI Access Data Service (dataset global-hourly), keeps
routine METARs (REPORT_TYPE FM-15), and writes OC1 (wind gust, tenths m/s; 9999 missing) and
VIS (distance m; 999999 missing) with quality codes 0,1,4,5,9 (passed / not checked).
"""
import csv, hashlib, io, json, os, sys, time, urllib.request
from datetime import datetime, timezone

API = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly&stations=%s"
       "&startDate=%s&endDate=%s&format=csv&dataTypes=VIS,OC1,REPORT_TYPE&includeStationLocation=1")
GOOD = set("01459")
stations_csv, out, start, end = sys.argv[1:5]
os.makedirs(os.path.join(out, "raw"), exist_ok=True)
rows, urls = [], []
for st in csv.DictReader(open(stations_csv)):
    sid = st["id"].replace("-", "")
    url = API % (sid, start, end)
    raw = os.path.join(out, "raw", sid + ".csv")
    if not os.path.exists(raw):
        for attempt in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
                break
            except OSError:
                time.sleep(5)
        else:
            print("failed", sid); continue
        open(raw, "wb").write(data)
    urls.append(url)
    for r in csv.DictReader(io.StringIO(open(raw).read())):
        if r.get("REPORT_TYPE", "").strip() != "FM-15":
            continue
        t = datetime.strptime(r["DATE"], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M")
        base = [st["id"], r["LATITUDE"], r["LONGITUDE"], r["ELEVATION"], t]
        oc1 = (r.get("OC1") or "").split(",")
        if len(oc1) == 2 and oc1[0] != "9999" and oc1[1] in GOOD:
            rows.append(base + ["gust", int(oc1[0]) / 10.0])
        vis = (r.get("VIS") or "").split(",")
        if len(vis) == 4 and vis[0] != "999999" and vis[1] in GOOD:
            rows.append(base + ["visibility", float(vis[0])])
path = os.path.join(out, "stations_extra.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "lat", "lon", "elev_m", "time_utc", "var", "value"])
    w.writerows(rows)
json.dump({
    "format": "wtune-obs/1", "kind": "stations_extra",
    "dataset": "NOAA NCEI Global Hourly (ISD) METAR gust and visibility",
    "provider": "NOAA NCEI",
    "url": "https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly",
    "license": "US Government work; WMO Resolution 40 data may carry use restrictions outside the US",
    "citation": "Smith, A., N. Lott, and R. Vose, 2011: The Integrated Surface Database. BAMS, 92, 704-708",
    "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "urls": urls,
    "files": {"stations_extra.csv": hashlib.sha256(open(path, "rb").read()).hexdigest()},
}, open(os.path.join(out, "provenance.json"), "w"), indent=1)
print(len(rows), "rows", sum(r[5] == "gust" for r in rows), "gusts")

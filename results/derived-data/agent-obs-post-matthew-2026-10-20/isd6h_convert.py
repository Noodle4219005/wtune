"""NOAA ISD-Lite 6-hour precipitation (column 12) -> wtune-obs/1 rain_gauges.

    python3 -I convert.py STATIONS_DIR OUT_DIR START END    (START/END 'YYYY-MM-DD HH', UTC)

A pipeline test of the gauge reader: the same stations `wtune post fetch stations` downloads,
read for their precipitation instead of T/Td/wind/MSLP. ISD-Lite: tenths of mm, -9999
missing, -1 trace (-> 0). The 6-h value at hour H is the total of the 6 h ending at H UTC.
"""
import csv, glob, gzip, hashlib, json, os, sys
from datetime import datetime, timedelta, timezone

src, out = sys.argv[1], sys.argv[2]
t0, t1 = (datetime.strptime(x, "%Y-%m-%d %H") for x in sys.argv[3:5])
os.makedirs(out, exist_ok=True)
meta = {r["id"]: r for r in csv.DictReader(open(os.path.join(src, "stations.csv")))}
rows = []
for sid, m in sorted(meta.items()):
    for p in glob.glob(os.path.join(src, "%s-*.gz" % sid)):
        for line in gzip.open(p, "rt"):
            f = line.split()
            t = datetime(int(f[0]), int(f[1]), int(f[2]), int(f[3]))
            v = int(f[11])
            if not (t0 <= t <= t1) or v == -9999 or t.hour % 6:
                continue
            rows.append([sid, m["lat"], m["lon"], m["elev_m"] if m["elev_m"] != "nan" else "",
                         (t - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M"),
                         t.strftime("%Y-%m-%d %H:%M"), max(v, 0) / 10.0])
path = os.path.join(out, "gauges.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "lat", "lon", "elev_m", "start_utc", "end_utc", "precip_mm"])
    w.writerows(rows)
prov = {
    "format": "wtune-obs/1", "kind": "rain_gauges",
    "dataset": "NOAA ISD-Lite 6-hour precipitation (gauge-reader pipeline test)",
    "provider": "NOAA NCEI",
    "url": "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/",
    "license": "US Government work; WMO Resolution 40 data may carry use restrictions outside the US",
    "citation": "Smith, A., N. Lott, and R. Vose, 2011: The Integrated Surface Database. BAMS, 92, 704-708",
    "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "files": {"gauges.csv": hashlib.sha256(open(path, "rb").read()).hexdigest()},
}
json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
print(len(rows), "rows,", len({r[0] for r in rows}), "gauges")

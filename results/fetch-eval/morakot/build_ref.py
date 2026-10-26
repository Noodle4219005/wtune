#!/usr/bin/env python3
"""Reference for the morakot fetch task: CWA hourly station rainfall from CODiS, as wtune-obs/1
rain_gauges (gauges.csv). Kept under results/, which the fetch sandbox does not mount.

    python3 build_ref.py OUT_DIR

CODiS (https://codis.cwa.gov.tw): POST /api/station, one local day per call; a record stamped
local HH:00 is the hour ending then, 23:59 is the hour ending 24:00. Local time is UTC+8."""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

STATIONS = {   # id: (codis stn_type, name)
    "467530": ("cwb", "Alishan"), "C0R100": ("auto_C0", "Weiliaoshan"),
    "C0M530": ("auto_C0", "Fenqihu"), "C1M610": ("auto_C1", "Shipanlong"),
    "C1M630": ("auto_C1", "Fengshan"), "466920": ("cwb", "Taipei"),
}
WINDOW = (datetime(2009, 8, 7, 0), datetime(2009, 8, 10, 0))      # UTC
LOCAL = timedelta(hours=8)


def post(stn, typ, day, raw_dir):
    path = os.path.join(raw_dir, "%s_%s.json" % (stn, day))
    if not os.path.exists(path):
        data = ("date=%sT00:00:00.000+08:00&type=report_date&stn_ID=%s&stn_type=%s&more="
                "&start=%sT00:00:00&end=%sT23:59:59" % (day, stn, typ, day, day))
        out = subprocess.run(["curl", "-s", "--max-time", "60", "-X", "POST",
                              "https://codis.cwa.gov.tw/api/station", "--data", data],
                             capture_output=True, check=True).stdout
        with open(path, "wb") as fh:
            fh.write(out)
    return json.load(open(path))


def main(out_dir):
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    rows = []
    days = [(WINDOW[0] + LOCAL - timedelta(days=1) + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(5)]
    for stn, (typ, _name) in STATIONS.items():
        for day in days:
            js = post(stn, typ, day, raw)
            for blk in js.get("data") or []:
                for rec in blk.get("dts") or []:
                    t = rec["DataTime"]
                    loc = (datetime.strptime(t[:10], "%Y-%m-%d") + timedelta(days=1)
                           if t[11:16] == "23:59" else datetime.strptime(t[:16], "%Y-%m-%dT%H:%M"))
                    end = loc - LOCAL
                    if not WINDOW[0] < end <= WINDOW[1]:
                        continue
                    v = rec.get("Precipitation", {}).get("Accumulation")
                    rows.append((stn, end - timedelta(hours=1), end, v))
    meta = {}
    for blk in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "station_list.json")))["data"]:
        for s in blk["item"]:
            meta.setdefault(s["stationID"], s)
    ref = os.path.join(out_dir, "cwa_gauges")
    os.makedirs(ref, exist_ok=True)
    bad = 0
    with open(os.path.join(ref, "gauges.csv"), "w") as fh:
        fh.write("id,lat,lon,elev_m,start_utc,end_utc,precip_mm\n")
        for stn, a, b, v in sorted(rows):
            m = meta[stn]
            if v == -9.8:                   # trace (雨跡): 0 in CWA totals
                v = 0.0
            if v is None or v < 0:          # -999.x / -9997 / -9998: missing or faulty
                bad += 1
                continue
            fh.write("%s,%.6f,%.6f,%s,%s,%s,%s\n" % (stn, m["latitude"], m["longitude"], m["altitude"],
                                                  a.strftime("%Y-%m-%d %H:%M"),
                                                  b.strftime("%Y-%m-%d %H:%M"), v))
    print(len(rows), "records,", bad, "missing")


if __name__ == "__main__":
    main(sys.argv[1])

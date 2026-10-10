"""post/fetch/nexrad.py -- NEXRAD WSR-88D Level II volumes from the Unidata AWS bucket
`unidata-nexrad-level2` (public, no credentials; the older noaa-nexrad-level2 path no longer
allows listing, anonymous ListObjects returns AccessDenied). For every target time (default
hourly across the window) the volume scan nearest in time (within --tol-min, default 8 min) is
downloaded for each radar. Radars: --radars KLIX,KMOB, else the --max-radars sites nearest the
region centre among those within the region (+2 deg margin, about one radar range), from NCEI's HOMR station list
(needs that list to be reachable; --radars always works).

Files: <project>/input/obs/nexrad/<RADAR>/<original key name>. The composite reflectivity is
built from them later, on the compute side, by post/radar_composite.py (Py-ART gridding).
Coverage: verified present for 2005-08-28 (KLIX, KMOB); the bucket listing for some radars is
empty in recent years (KLIX 2024-01..2026 returned no keys), which run() reports as "no data".
"""
from __future__ import annotations

import math
import os
import re
from datetime import datetime, timedelta

from wtune.post import sources

NAME = "nexrad"
HELP = "NEXRAD Level II volumes (AWS unidata-nexrad-level2) nearest each hour, per radar"
PROVENANCE = dict(sources.REGISTRY["nexrad"])

BUCKET = "https://unidata-nexrad-level2.s3.amazonaws.com"
HOMR = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")
_TS_RE = re.compile(r"([A-Z0-9]{4})(\d{8})_(\d{6})")


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--radars", help="comma list of radar ids (e.g. KLIX,KMOB); default: "
                        "nearest sites to the region centre")
    parser.add_argument("--max-radars", type=int, default=4)
    parser.add_argument("--step-min", type=int, default=60, help="sampling step (default 60)")
    parser.add_argument("--tol-min", type=int, default=8, help="max |file time - target| (min)")


def key_time(key):
    m = _TS_RE.search(os.path.basename(key))
    return datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S") if m else None


def is_volume_key(key):
    base = os.path.basename(key)
    return key_time(key) is not None and not base.endswith("_MDM")


def choose(keys, wanted, tol):
    stamped = [(key_time(k), k) for k in keys if is_volume_key(k)]
    out = []
    for w in wanted:
        cands = [(abs((t - w).total_seconds()), k) for t, k in stamped]
        if cands:
            d, k = min(cands)
            if d <= tol.total_seconds() and k not in out:
                out.append(k)
    return out


def list_keys(prefix):
    keys, marker = [], ""
    while True:
        url = "%s/?prefix=%s&max-keys=1000%s" % (BUCKET, prefix, ("&marker=" + marker) if marker else "")
        body = sources.http_get(url)
        if body is None:
            return keys
        text = body.decode("utf-8", "replace")
        page = _KEY_RE.findall(text)
        keys += page
        if "<IsTruncated>true</IsTruncated>" not in text or not page:
            return keys
        marker = page[-1]


def parse_homr(text):
    """[(icao, lat, lon)] from HOMR's fixed-width nexrad-stations.txt (rows of type NEXRAD)."""
    out = []
    for line in text.splitlines():
        if "NEXRAD" not in line:
            continue
        f = line.split()
        try:
            icao = f[1]
            # LAT LON are the two floats before ELEV/UTC/STNTYPE
            i = next(i for i, x in enumerate(f) if x == "NEXRAD")
            lat, lon = float(f[i - 4]), float(f[i - 3])
        except (StopIteration, ValueError, IndexError):
            continue
        if re.fullmatch(r"[KPTL][A-Z0-9]{3}", icao):
            out.append((icao, lat, lon))
    return out


def pick_radars(sites, bbox, n, margin=2.0):
    clat, clon = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
    inside = [s for s in sites if bbox[0] - margin <= s[1] <= bbox[1] + margin
              and bbox[2] - margin <= s[2] <= bbox[3] + margin]
    inside.sort(key=lambda s: math.hypot(s[1] - clat, (s[2] - clon) * math.cos(math.radians(clat))))
    return [s[0] for s in inside[:n]]


def run(args):
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("nexrad: %s" % e)
        return []
    if args.radars:
        radars = [r.strip().upper() for r in args.radars.split(",") if r.strip()]
    else:
        body = sources.http_get(HOMR)
        if body is None:
            print("nexrad: cannot read the HOMR radar list; pass --radars KLIX,KMOB explicitly")
            return []
        radars = pick_radars(parse_homr(body.decode("utf-8", "replace")), bbox, args.max_radars)
        if not radars:
            print("nexrad: no NEXRAD site within region %s (+2 deg, ~ one radar range) -- pass --radars" % (bbox,))
            return []
    step = timedelta(minutes=args.step_min)
    t = start.replace(minute=0, second=0, microsecond=0)
    if t < start:
        t += timedelta(hours=1)
    wanted = []
    while t <= end:
        wanted.append(t)
        t += step
    tol = timedelta(minutes=args.tol_min)
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    written, urls, files = [], [], {}
    for radar in radars:
        keys = []
        for d in sorted({w.date() for w in wanted}):
            day = list_keys("%s/%s/" % (d.strftime("%Y/%m/%d"), radar))
            keys += choose(day, [w for w in wanted if w.date() == d], tol)
        if not keys:
            print("nexrad: %s has no Level II volume near %s..%s in %s (coverage/availability)"
                  % (radar, start, end, BUCKET))
            continue
        for key in keys:
            dest = os.path.join(sources.dataset_dir(args.project, NAME), radar, os.path.basename(key))
            rel = os.path.relpath(dest, sources.dataset_dir(args.project, NAME))
            sha, dl = sources.download("%s/%s" % (BUCKET, key), dest, known_sha=known.get(rel))
            if sha is None:
                continue
            written.append(dest)
            files[rel] = sha
            if dl:
                urls.append("%s/%s" % (BUCKET, key))
    if not written:
        return []
    # radar positions: pre-2008 Archive II volumes carry none (Py-ART reads 0, 0)
    homr = os.path.join(sources.dataset_dir(args.project, NAME), "nexrad_sites.txt")
    sha, dl = sources.download(HOMR, homr, known_sha=known.get("nexrad_sites.txt"))
    if sha:
        files["nexrad_sites.txt"] = sha
        if dl:
            urls.append(HOMR)
    sources.write_provenance(args.project, NAME, PROVENANCE, urls, files,
                             window=sources.window_dict(start, end), bbox=list(bbox),
                             extra={"radars": radars})
    print("nexrad: %d volume(s) for %s under %s (%d newly downloaded)"
          % (len(written), ",".join(radars), sources.dataset_dir(args.project, NAME), len(urls)))
    return written

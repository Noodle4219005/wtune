"""post/fetch/mrms.py -- NOAA MRMS from the AWS Open Data bucket noaa-mrms-pds (public, no
credentials): the merged reflectivity composite (MergedReflectivityQCComposite_00.50, 2 min,
0.01 deg) and the hourly gauge-corrected QPE (MultiSensor_QPE_01H_Pass2_00.00).

Coverage boundary, verified against the bucket listing: the first day of any product in the
CONUS and CARIB (Puerto Rico) sub-domains is 20201014, so nothing exists for Katrina (2005) or
Maria (2017). A window that ends before 2020-10-14, or a region outside the five MRMS
sub-domains (CONUS, CARIB, HAWAII, ALASKA, GUAM), returns [] and prints the reason.

Files: <project>/input/obs/mrms/<REGION>/<product>/MRMS_<product>_YYYYMMDD-HHMMSS.grib2, gunzipped
(sha256 recorded is that of the .grib2). Reflectivity is fetched at the file nearest each
--step-min boundary (default hourly), QPE at every whole hour.
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
from datetime import datetime, timedelta

from wtune.post import sources

NAME = "mrms"
HELP = "NOAA MRMS reflectivity composite / QPE (AWS noaa-mrms-pds; 2020-10-14 onward only)"
PROVENANCE = dict(sources.REGISTRY["mrms"])

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
FIRST_DAY = datetime(2020, 10, 14)
# sub-domain extents (lat0, lat1, lon0, lon1, east positive), from the grid headers of a real file
REGIONS = {"CONUS": (20.0, 55.0, -130.0, -60.0), "CARIB": (10.0, 25.0, -90.0, -60.0),
           "HAWAII": (15.0, 26.0, -164.0, -151.0), "ALASKA": (50.0, 72.0, -176.0, -126.0),
           "GUAM": (9.0, 18.0, 140.0, 150.0)}
PRODUCTS = {"refl": "MergedReflectivityQCComposite_00.50",
            "qpe": "MultiSensor_QPE_01H_Pass2_00.00"}
_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")
_TS_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--products", default="refl,qpe",
                        help="comma list of refl (reflectivity composite), qpe (1 h QPE)")
    parser.add_argument("--step-min", type=int, default=60,
                        help="reflectivity sampling step in minutes (default 60)")
    parser.add_argument("--region", choices=sorted(REGIONS),
                        help="force an MRMS sub-domain instead of choosing by overlap")


def pick_region(bbox):
    """MRMS sub-domain with the largest overlap with bbox (lat0, lat1, lon0, lon1), else None."""
    best, best_area = None, 0.0
    for name, (a0, a1, o0, o1) in REGIONS.items():
        dlat = min(bbox[1], a1) - max(bbox[0], a0)
        dlon = min(bbox[3], o1) - max(bbox[2], o0)
        if dlat > 0 and dlon > 0 and dlat * dlon > best_area:
            best, best_area = name, dlat * dlon
    return best


def list_keys(prefix):
    """All object keys under an S3 prefix (paginated ListObjects, anonymous)."""
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


def key_time(key):
    m = _TS_RE.search(key)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def targets(start, end, product, step_min):
    """(target time, tolerance) list: reflectivity every step_min, QPE at whole hours."""
    step = timedelta(minutes=60 if product == "qpe" else step_min)
    t = start.replace(minute=0, second=0, microsecond=0)
    if t < start:
        t += timedelta(hours=1)
    out = []
    while t <= end:
        out.append(t)
        t += step
    return out


def choose_keys(day_keys, wanted, tol):
    """For each wanted time the key of the nearest file within tol; dedup, order of wanted."""
    stamped = [(key_time(k), k) for k in day_keys]
    stamped = [(t, k) for t, k in stamped if t is not None]
    chosen = []
    for w in wanted:
        cands = [(abs((t - w).total_seconds()), k) for t, k in stamped]
        if cands:
            d, k = min(cands)
            if d <= tol.total_seconds() and k not in chosen:
                chosen.append(k)
    return chosen


def _fetch_gz(key, dest, known_sha):
    """Downloads key (.gz) and stores the gunzipped GRIB2 at dest; idempotent on the .grib2."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        sha = sources.sha256_file(dest)
        if known_sha is None or sha == known_sha:
            return sha, False
    gz = dest + ".gz"
    sha_gz, dl = sources.download("%s/%s" % (BUCKET, key), gz)
    if sha_gz is None:
        return None, False
    with gzip.open(gz, "rb") as src, open(dest + ".tmp", "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(dest + ".tmp", dest)
    os.remove(gz)
    return sources.sha256_file(dest), True


def run(args):
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("mrms: %s" % e)
        return []
    if end < FIRST_DAY:
        print("mrms: no data -- MRMS in the AWS bucket starts on %s (CONUS and CARIB alike); the "
              "requested window %s..%s is earlier (coverage limit; Katrina 2005 and Maria 2017 "
              "are not covered, use nexrad + stage4 instead)"
              % (FIRST_DAY.date(), start, end))
        return []
    region = args.region or pick_region(bbox)
    if region is None:
        print("mrms: no data -- region %s is outside every MRMS sub-domain %s"
              % (bbox, ", ".join(sorted(REGIONS))))
        return []
    start = max(start, FIRST_DAY)
    prods = [p.strip() for p in args.products.split(",") if p.strip()]
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    written, urls, files = [], [], {}
    for p in prods:
        if p not in PRODUCTS:
            print("mrms: unknown product %r (choose from %s)" % (p, ", ".join(PRODUCTS)))
            continue
        pname = PRODUCTS[p]
        tol = timedelta(minutes=(5 if p == "qpe" else max(2, min(args.step_min // 2, 10))))
        wanted = targets(start, end, p, args.step_min)
        days = sorted({w.date() for w in wanted})
        keys = []
        for d in days:
            day_keys = list_keys("%s/%s/%s/" % (region, pname, d.strftime("%Y%m%d")))
            keys += choose_keys(day_keys, [w for w in wanted if w.date() == d], tol)
        if not keys:
            print("mrms: no %s files found in %s for %s..%s" % (p, region, start, end))
            continue
        outdir = os.path.join(sources.dataset_dir(args.project, NAME), region, pname)
        for key in keys:
            dest = os.path.join(outdir, os.path.basename(key)[:-3])
            rel = os.path.relpath(dest, sources.dataset_dir(args.project, NAME))
            sha, dl = _fetch_gz(key, dest, known.get(rel))
            if sha is None:
                continue
            written.append(dest)
            files[rel] = sha
            if dl:
                urls.append("%s/%s" % (BUCKET, key))
    if not written:
        return []
    sources.write_provenance(args.project, NAME, PROVENANCE, urls, files,
                             window=sources.window_dict(start, end), bbox=list(bbox),
                             extra={"region": region, "products": prods})
    print("mrms: %d file(s) for region %s under %s (%d newly downloaded)"
          % (len(written), region, sources.dataset_dir(args.project, NAME), len(urls)))
    return written

"""post/fetch/stage4.py -- NCEP Stage IV QPE (1 h / 6 h / 24 h accumulations, ~4.8 km HRAP, GRIB1)
from the Iowa Environmental Mesonet archive (a plain HTTPS mirror of the NCEP/EMC files, public,
no credentials). NCAR RDA ds507.5 was tried first and is not used: data.rda.ucar.edu did not
answer for this dataset id from here, whereas the IEM archive returned Stage IV for every day
probed between 2001-01-01 and 2025-06-01 (Katrina 2005-08-27..29 included).

Files land in <project>/input/obs/stage4/ST4.YYYYMMDDHH.{01,06,24}h.grib; the valid time is the
END of the accumulation. Only accumulations lying completely inside the requested window are
fetched (so the 24 h product, which NCEP issues at 12 UTC only, appears only when the window
contains a full 12 UTC-to-12 UTC day). The grid is CONUS: a window/region entirely outside it
(e.g. Puerto Rico) returns [] with the reason.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from wtune.post import sources

NAME = "stage4"
HELP = "NCEP Stage IV QPE (1/6/24 h, CONUS ~4.8 km) from the IEM archive"
PROVENANCE = dict(sources.REGISTRY["stage4"])

# Approximate HRAP CONUS extent (lat0, lat1, lon0, lon1); used only to refuse regions that cannot
# be covered, the actual data outside the radar-covered area is simply missing in the file.
CONUS_BBOX = (23.0, 53.5, -127.5, -63.0)
SITES_FILE = "nexrad_sites.txt"
SITES_URL = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
BASE = "https://mesonet.agron.iastate.edu/archive/data/%Y/%m/%d/stage4/ST4.%Y%m%d%H.{acc:02d}h.grib"


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--accum", default="1,6,24",
                        help="accumulation lengths in hours to fetch (default 1,6,24)")
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")


def _overlaps(bbox):
    return not (bbox[1] < CONUS_BBOX[0] or bbox[0] > CONUS_BBOX[1]
                or bbox[3] < CONUS_BBOX[2] or bbox[2] > CONUS_BBOX[3])


def valid_times(start, end, accum):
    """Valid (end-of-accumulation) times whose whole accumulation lies inside [start, end];
    NCEP issues 6 h at 00/06/12/18 UTC and 24 h at 12 UTC only."""
    t = start.replace(minute=0, second=0, microsecond=0)
    if t < start:
        t += timedelta(hours=1)
    out = []
    while t <= end:
        if t - timedelta(hours=accum) >= start.replace(minute=0, second=0, microsecond=0):
            if accum == 1 or (accum == 6 and t.hour % 6 == 0) or (accum == 24 and t.hour == 12):
                out.append(t)
        t += timedelta(hours=1)
    return out


def url_for(t, accum):
    return t.strftime(BASE).format(acc=accum)


def path_for(project, t, accum):
    return os.path.join(sources.dataset_dir(project, NAME),
                        "ST4.%s.%02dh.grib" % (t.strftime("%Y%m%d%H"), accum))


def run(args):
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("stage4: %s" % e)
        return []
    if not _overlaps(bbox):
        print("stage4: no data -- region %s lies outside the Stage IV CONUS HRAP grid %s "
              "(coverage limit; Puerto Rico/Hawaii/Alaska are not in this product)"
              % (bbox, CONUS_BBOX))
        return []
    accums = [int(a) for a in str(args.accum).split(",") if a.strip()]
    prov_old = sources.read_provenance(args.project, NAME) or {}
    known = prov_old.get("files", {})
    jobs = [(t, a) for a in accums for t in valid_times(start, end, a)]
    if not jobs:
        print("stage4: window %s..%s contains no complete %s h accumulation"
              % (start, end, "/".join(map(str, accums))))
        return []

    def one(job):
        t, a = job
        dest = path_for(args.project, t, a)
        rel = os.path.basename(dest)
        try:
            sha, downloaded = sources.download(url_for(t, a), dest, known_sha=known.get(rel))
        except OSError as e:     # network, after download()'s own retries: keep the other files
            return t, a, dest, None, False, str(e)
        return t, a, dest, sha, downloaded, None

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        results = list(ex.map(one, jobs))
    written, urls, files = [], [], {}
    for t, a, dest, sha, downloaded, _err in results:
        if sha is None:
            continue
        written.append(dest)
        files[os.path.basename(dest)] = sha
        if downloaded:
            urls.append(url_for(t, a))
    failed = [(os.path.basename(d), err) for _t, _a, d, _s, _dl, err in results if err]
    missing = [(t, a) for t, a, _d, sha, _dl, err in results if sha is None and not err]
    if missing:
        print("stage4: %d of %d files do not exist on the server (e.g. %s %02dh) -- Stage IV "
              "6 h/24 h products are absent on some days" % (len(missing), len(results),
                                                             missing[0][0], missing[0][1]))
    if written:
        # radar site list (HOMR): the recipe masks the verification to radar range, because Stage
        # IV is 0 (not missing) beyond it, e.g. over the open Gulf
        sites = os.path.join(sources.dataset_dir(args.project, NAME), SITES_FILE)
        try:
            sha, dl = sources.download(SITES_URL, sites, known_sha=known.get(SITES_FILE))
        except OSError as e:
            sha, dl = None, False
            failed.append((SITES_FILE, str(e)))
        if sha:
            files[SITES_FILE] = sha
            if dl:
                urls.append(SITES_URL)
    if not written and not failed:
        print("stage4: no Stage IV file exists for %s..%s (coverage: 2001-present, CONUS)"
              % (start, end))
        return []
    if written:
        sources.write_provenance(args.project, NAME, PROVENANCE, urls, files,
                                 window=sources.window_dict(start, end), bbox=list(bbox),
                                 extra={"accumulations_h": accums,
                                        "failed": [name for name, _err in failed]})
    print("stage4: %d file(s) in %s (%d newly downloaded)"
          % (len(written), sources.dataset_dir(args.project, NAME), len(urls)))
    if failed:
        raise sources.FetchIncomplete(
            "stage4: %d of %d files could not be downloaded (%s: %s); run the same fetch again "
            "to get the rest" % (len(failed), len(results), failed[0][0], failed[0][1]))
    return written

"""post/sources.py -- provenance registry and the small download/provenance helpers shared by the
observation fetch modules (src/wtune/post/fetch/{stage4,nexrad,mrms,stations}.py) and by
post/report.py's "資料來源與授權" section.

Registry: REGISTRY[name] is the static PROVENANCE dict of every observation/reanalysis dataset
`wtune post` can use. A fetch run additionally writes <project>/input/obs/<name>/provenance.json
(era5: <project>/input/era5/provenance.json) = the static dict + the exact URLs fetched, the UTC
fetch time, sha256 of every file, and the requested time window/bbox. `dataset_records()` merges
both, so the report shows the fetch time when the data was fetched by this tool and the static
citation otherwise.

Helpers are stdlib only (urllib/hashlib/json) so the fetch modules run on a login node with
nothing but the default interpreter; the heavy numerics live elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

PROVENANCE_KEYS = ("dataset", "provider", "url", "license", "citation", "coverage_time",
                   "coverage_space", "resolution", "access")

REGISTRY = {
    "stage4": {
        "dataset": "NCEP Stage IV multi-sensor QPE (1 h / 6 h / 24 h)",
        "provider": "NOAA/NWS NCEP Environmental Modeling Center; archive mirrored by the Iowa "
                    "Environmental Mesonet (Iowa State University)",
        "url": "https://mesonet.agron.iastate.edu/archive/data/ (…/YYYY/MM/DD/stage4/); "
               "origin: https://www.emc.ncep.noaa.gov/mmb/ylin/pcpanl/stage4/",
        "license": "US Government work, public domain (no access restriction); IEM asks for "
                   "attribution",
        "citation": "Lin, Y., and K. E. Mitchell, 2005: The NCEP Stage II/IV hourly precipitation "
                    "analyses: development and applications. 19th Conf. on Hydrology, AMS, "
                    "San Diego, CA, Paper 1.2.",
        "coverage_time": "2001 to present (verified present 2001-01-01 .. 2025-06-01; 6 h/24 h "
                         "products missing on some early days)",
        "coverage_space": "CONUS HRAP grid (about 23-53N, 127-63W), no Puerto Rico/Hawaii/Alaska",
        "resolution": "4.763 km HRAP polar-stereographic grid, GRIB1",
        "access": "public, no credentials",
    },
    "nexrad": {
        "dataset": "NEXRAD WSR-88D Level II base data",
        "provider": "NOAA/NWS Radar Operations Center, via Unidata on AWS Open Data "
                    "(bucket unidata-nexrad-level2)",
        "url": "https://unidata-nexrad-level2.s3.amazonaws.com/ ; "
               "https://registry.opendata.aws/noaa-nexrad/",
        "license": "US Government work, public domain (AWS Open Data, no restriction)",
        "citation": "NOAA National Weather Service (NWS) Radar Operations Center, 1991: NOAA Next "
                    "Generation Radar (NEXRAD) Level 2 Base Data. NOAA National Centers for "
                    "Environmental Information, doi:10.7289/V5W9574V.",
        "coverage_time": "1991 to present (verified 2005-08-28 KLIX/KMOB)",
        "coverage_space": "radar sites of the WSR-88D network (CONUS, PR, HI, AK, Guam); "
                          "useful range ~230 km",
        "resolution": "volume scan every ~5-10 min, 250 m gates (reflectivity)",
        "access": "public, no credentials",
    },
    "mrms": {
        "dataset": "MRMS (Multi-Radar Multi-Sensor) merged reflectivity composite and QPE",
        "provider": "NOAA/NWS NCEP, via AWS Open Data (bucket noaa-mrms-pds)",
        "url": "https://noaa-mrms-pds.s3.amazonaws.com/ ; https://registry.opendata.aws/noaa-mrms-pds/",
        "license": "US Government work, public domain (AWS Open Data)",
        "citation": "Zhang, J., et al., 2016: Multi-Radar Multi-Sensor (MRMS) quantitative "
                    "precipitation estimation: Initial operating capabilities. Bull. Amer. "
                    "Meteor. Soc., 97, 621-638, doi:10.1175/BAMS-D-14-00174.1.",
        "coverage_time": "2020-10-14 to present (verified: first day in bucket for CONUS and "
                         "CARIB is 20201014); nothing earlier (so not 2005 Katrina, not 2017 Maria)",
        "coverage_space": "CONUS, CARIB (Puerto Rico/Caribbean), HAWAII, ALASKA, GUAM sub-domains",
        "resolution": "0.01 deg lat/lon, 2 min (reflectivity), hourly (QPE)",
        "access": "public, no credentials",
    },
    "stations": {
        "dataset": "NOAA Integrated Surface Database (ISD-Lite hourly)",
        "provider": "NOAA National Centers for Environmental Information (NCEI)",
        "url": "https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/ ; station list "
               "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
        "license": "public domain (NOAA/NCEI open data)",
        "citation": "Smith, A., N. Lott, and R. Vose, 2011: The Integrated Surface Database: "
                    "Recent developments and partnership with the National Climatic Data "
                    "Center. Bull. Amer. Meteor. Soc., 92, 704-708, doi:10.1175/2011BAMS3015.1.",
        "coverage_time": "1901 to present (station dependent)",
        "coverage_space": "global surface stations (ASOS/AWOS/synoptic)",
        "resolution": "hourly; T, Td, MSLP, wind speed/direction, 1 h and 6 h precipitation",
        "access": "public, no credentials",
    },
    "era5": {
        "dataset": "ERA5 hourly reanalysis on pressure levels (ARCO-ERA5 zarr)",
        "provider": "ECMWF / Copernicus Climate Change Service; ARCO copy on Google Cloud Storage",
        "url": "https://cloud.google.com/storage/docs/public-datasets/era5 ; "
               "gs://gcp-public-data-arco-era5",
        "license": "Copernicus licence (CC-BY 4.0 for ERA5)",
        "citation": "Hersbach, H., et al., 2020: The ERA5 global reanalysis. Q. J. R. Meteorol. "
                    "Soc., 146, 1999-2049, doi:10.1002/qj.3803.",
        "coverage_time": "1940 to present",
        "coverage_space": "global",
        "resolution": "0.25 deg, hourly",
        "access": "public, no credentials (the synoptic recipe reads the local subset written "
                  "by `wtune post fetch-era5 --project P --run-dir R` on a login node)",
    },
    "era5_monthly": {
        "dataset": "ERA5 monthly means of t2m/sst(/tp), aggregated by wtune from the ARCO-ERA5 hourly store",
        "provider": "ECMWF / Copernicus Climate Change Service; ARCO copy on Google Cloud public datasets",
        "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        "license": "Copernicus licence (CC-BY 4.0 for ERA5); attribution required",
        "citation": "Hersbach, H., et al., 2020: The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., "
                    "146, 1999-2049, doi:10.1002/qj.3803.",
        "coverage_time": "1940-01-01 to a few months before present (store time axis)",
        "coverage_space": "cropped to the run domain plus --margin degrees",
        "resolution": "0.25 deg, monthly mean of 6-hourly (t2m, sst) or hourly (tp) fields",
        "access": "public, anonymous (no credentials); needs internet (a login node on most clusters)",
    },
    "gpcp_monthly": {
        "dataset": "GPCP Version 2.3 Monthly Precipitation Analysis (v02r03)",
        "provider": "NOAA NCEI / GPCP (Global Precipitation Climatology Project)",
        "url": "https://www.ncei.noaa.gov/data/global-precipitation-climatology-project-gpcp-monthly/access/",
        "license": "US Government work / public (NOAA NCEI, CC0)",
        "citation": "Adler, R.F., et al., 2018: The Global Precipitation Climatology Project (GPCP) "
                    "Monthly Analysis (New Version 2.3) and a Review of 2017 Global Precipitation. "
                    "Atmosphere, 9(4), 138, doi:10.3390/atmos9040138.",
        "coverage_time": "1979-01 to about two months before present",
        "coverage_space": "global, 2.5 deg",
        "resolution": "monthly mean, mm/day",
        "access": "public, no credentials",
    },
    "igra": {
        "dataset": "Integrated Global Radiosonde Archive (IGRA) Version 2",
        "provider": "NOAA National Centers for Environmental Information (NCEI)",
        "url": "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/",
        "license": "US Government work, public domain (NOAA NCEI); no registration",
        "citation": "Durre, I., R. S. Vose, D. B. Wuertz, 2006: Overview of the Integrated Global "
                    "Radiosonde Archive. J. Climate, 19, 53-68; Durre et al., 2018: Enhancements "
                    "to the IGRA. J. Atmos. Oceanic Technol., 35, 1753-1770; doi:10.7289/V5X63K0Q",
        "coverage_time": "1905-present, mostly 00Z/12Z launches",
        "coverage_space": "global, ~2800 stations",
        "resolution": "sounding levels (mandatory + significant), ~12 h",
        "access": "public, no credentials",
    },
    "hurdat2": {
        "dataset": "HURDAT2 Atlantic best-track database",
        "provider": "NOAA/NWS National Hurricane Center",
        "url": "https://www.nhc.noaa.gov/data/#hurdat",
        "license": "US Government work, public domain",
        "citation": "Landsea, C. W., and J. L. Franklin, 2013: Atlantic hurricane database "
                    "uncertainty and presentation of a new database format. Mon. Wea. Rev., 141, "
                    "3576-3592, doi:10.1175/MWR-D-12-00254.1.",
        "coverage_time": "1851 to present",
        "coverage_space": "Atlantic basin",
        "resolution": "6-hourly best track",
        "access": "public, no credentials",
    },
}

# Figures of other recipe modules that use an external dataset without carrying an
# `obs_sources` key themselves (they are not mine): id prefix -> registry names. Used only by
# used_sources() as a fallback so the licence section still lists ERA5 / HURDAT2.
_FIGURE_ID_SOURCES = (("syn_z500_diff", ("era5",)), ("syn_gridstat", ("era5",)),
                      ("tc_track", ("hurdat2",)), ("tc_intensity", ("hurdat2",)),
                      ("climate_annual_cycle", ("era5_monthly", "gpcp_monthly")),
                      ("climate_bias_t2", ("era5_monthly",)),
                      ("climate_bias_precip", ("gpcp_monthly",)),
                      ("pbl_skewt_obs", ("igra",)), ("pbl_height_compare", ("igra",)))


class FetchIncomplete(RuntimeError):
    """Some files could not be downloaded (network). What was fetched is on disk and recorded
    in provenance.json; `wtune post fetch` exits 1 so the caller knows the set is partial."""


class FetchUnavailable(Exception):
    """Raised internally when data does not exist for the requested time/region."""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# paths and provenance.json
# --------------------------------------------------------------------------------------

def dataset_dir(project, name):
    """<project>/input/obs/<name>/ (era5: <project>/input/era5/)."""
    if name == "era5":
        return os.path.join(project, "input", "era5")
    return os.path.join(project, "input", "obs", name)


def provenance_path(project, name):
    return os.path.join(dataset_dir(project, name), "provenance.json")


def read_provenance(project, name):
    try:
        with open(provenance_path(project, name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_provenance(project, name, provenance, urls, files, window=None, bbox=None,
                     extra=None, prune_missing=False):
    """Writes/updates <dataset_dir>/provenance.json: `provenance` (the module's PROVENANCE dict)
    + `urls` (fetched) + `files` ({relative path: sha256}) + fetch time + requested window/bbox.
    An existing record is merged: urls/files accumulate, so a resumed or repeated fetch does not
    forget what an earlier one downloaded. prune_missing: drop files (keys relative to the
    dataset dir) that are no longer on disk. `fetched_utc` is the time of the newest download;
    a run that downloaded nothing keeps the old one."""
    d = dataset_dir(project, name)
    os.makedirs(d, exist_ok=True)
    old = read_provenance(project, name) or {}
    rec = dict(provenance)
    rec["urls"] = sorted(set(old.get("urls", [])) | set(urls))
    merged = dict(old.get("files", {}))
    merged.update(files)
    rec["files"] = ({k: v for k, v in merged.items() if os.path.exists(os.path.join(d, k))}
                    if prune_missing else merged)
    rec["fetched_utc"] = utc_now() if urls else old.get("fetched_utc", utc_now())
    rec["window"] = window if window is not None else old.get("window")
    rec["bbox"] = bbox if bbox is not None else old.get("bbox")
    if extra:
        rec.update(extra)
    tmp = provenance_path(project, name) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    os.replace(tmp, provenance_path(project, name))
    return provenance_path(project, name)


def dataset_records(project_dir, names):
    """[{name, dataset, provider, url, license, citation, fetched_utc, urls_fetched}] in the
    order of `names`: provenance.json under project_dir when it exists, else the static
    REGISTRY entry (fetched_utc "n/a (not fetched by wtune)")."""
    out = []
    for name in names:
        static = REGISTRY.get(name)
        prov = read_provenance(project_dir, name) if project_dir else None
        if static is None:
            if not (prov and prov.get("format") == "wtune-obs/1"):
                continue
            static = {k: "" for k in PROVENANCE_KEYS}     # an agent-fetched dataset
        rec = dict(static)
        if prov:
            rec.update({k: prov[k] for k in PROVENANCE_KEYS if k in prov})
        rec["name"] = name
        rec["fetched_utc"] = (prov or {}).get("fetched_utc", "n/a (not fetched by wtune)")
        rec["urls_fetched"] = len((prov or {}).get("urls", []))
        out.append(rec)
    return out


def source_tags(project_dir, names):
    """Compact per-figure `obs_sources` value: dataset/provider/url/license/citation/fetch time
    for each name (what the figure `extra` and the HTML stamp show)."""
    return [{k: r[k] for k in ("name", "dataset", "provider", "url", "license", "citation",
                               "fetched_utc")} for r in dataset_records(project_dir, names)]


def used_sources(figures):
    """Registry names used by a recipe's figure records: every real (non-missing) figure's
    `obs_sources[*].name`, plus the _FIGURE_ID_SOURCES fallback for other modules' figures.
    Order of first appearance."""
    names = []
    for f in figures:
        if f.get("missing"):
            continue
        for s in f.get("obs_sources") or []:
            if s["name"] not in names:
                names.append(s["name"])
        for prefix, srcs in _FIGURE_ID_SOURCES:
            if str(f.get("id", "")).startswith(prefix):
                for s in srcs:
                    if s not in names:
                        names.append(s)
    return names


# --------------------------------------------------------------------------------------
# download (idempotent, resumable)
# --------------------------------------------------------------------------------------

_UA = {"User-Agent": "wtune-post-fetch/1 (WRF post-processing; research use)"}


def http_get(url, timeout=60, retries=3):
    """Body bytes of url; None on HTTP 404/403 (data does not exist). Other errors retried,
    then raised."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            last = e
        except (urllib.error.URLError, OSError) as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


def url_exists(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers=_UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return False
        raise


def download(url, dest, known_sha=None, timeout=120, retries=3):
    """Downloads url -> dest. Idempotent: an existing non-empty dest whose sha256 equals
    `known_sha` (from provenance.json) is kept without a request; an existing dest with no known
    sha is trusted (size > 0) and its sha computed. Resumable: writes dest+'.part' and continues
    it with a Range request. Returns (sha256, downloaded: bool), or (None, False) when the
    server says 404/403 (data does not exist)."""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        sha = sha256_file(dest)
        if known_sha is None or sha == known_sha:
            return sha, False
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    last = None
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = dict(_UA)
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                expected, got = r.headers.get("Content-Length"), 0
                with open(part, mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
            # a connection cut mid-body makes read() return b"", not raise; .part is kept so
            # the next attempt resumes it with Range
            if expected is not None and got != int(expected):
                raise OSError("%s: got %d of %s bytes" % (url, got, expected))
            os.replace(part, dest)
            return sha256_file(dest), True
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None, False
            if e.code == 416 and have:       # .part already complete
                os.replace(part, dest)
                return sha256_file(dest), True
            last = e
        except (urllib.error.URLError, OSError) as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


# --------------------------------------------------------------------------------------
# the run's time window and bbox (for --run-dir)
# --------------------------------------------------------------------------------------

def run_window(run_dir):
    """{"start", "end" (datetime, naive UTC), "bbox": (lat0, lat1, lon0, lon1)} from the run's
    wrfout_d01 valid times (each file's Times, post.extract.list_frames) and its XLAT/XLONG. Raises ValueError when there are no frames."""
    from wtune.post.caseinfo import WRF_TIME_FMT
    from wtune.post.extract import list_frames
    frames = list_frames(run_dir, domain=1)
    if not frames:
        raise ValueError("no wrfout_d01_* frames in %s" % run_dir)
    import netCDF4
    with netCDF4.Dataset(frames[0][1]) as ds:
        lat, lon = ds.variables["XLAT"][0], ds.variables["XLONG"][0]
        bbox = (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))
    return dict(start=datetime.strptime(frames[0][0], WRF_TIME_FMT),
                end=datetime.strptime(frames[-1][0], WRF_TIME_FMT), bbox=bbox)


def parse_time(text):
    """'YYYY-MM-DD[_ T]HH[:MM[:SS]]' or 'YYYYMMDDHH' -> naive datetime."""
    t = text.strip().replace("_", " ").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d", "%Y%m%d%H",
                "%Y%m%d%H%M"):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("cannot parse time %r" % text)


def resolve_window(args, default_hours=None):
    """(start, end, bbox) from --run-dir, overridden by --start/--end/--bbox when given.
    Raises ValueError when neither source gives a window."""
    start = end = bbox = None
    if getattr(args, "run_dir", None):
        w = run_window(args.run_dir)
        start, end, bbox = w["start"], w["end"], w["bbox"]
    if getattr(args, "start", None):
        start = parse_time(args.start)
    if getattr(args, "end", None):
        end = parse_time(args.end)
    if getattr(args, "bbox", None):
        bbox = tuple(float(x) for x in args.bbox.split(","))
        if len(bbox) != 4:
            raise ValueError("--bbox must be lat0,lat1,lon0,lon1")
    if start is None or end is None or bbox is None:
        raise ValueError("need --run-dir, or all of --start/--end/--bbox, to know the time "
                         "window and region")
    return start, end, bbox


def add_window_arguments(parser):
    parser.add_argument("--start", help="override the window start (UTC, YYYY-MM-DD_HH)")
    parser.add_argument("--end", help="override the window end (UTC, YYYY-MM-DD_HH)")
    parser.add_argument("--bbox", help="override the region: lat0,lat1,lon0,lon1 (lon east +)")


def window_dict(start, end):
    return {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}

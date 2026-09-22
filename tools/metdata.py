#!/usr/bin/env python3
"""tools/metdata.py -- source selection, URL construction, availability checking, and download
for global GRIB analysis/forecast data (GFS via NOMADS/AWS/RDA-GDEX, or a local GRIB directory
e.g. NAM/NARR), for a WRF case. Owns every network/source concern so tools/grib_state.py (the
WRF-run-free physical-state trajectory tool) and any WPS/ungrib prep step can both just ask "what
do I need, where does it live, does it exist, fetch it" without re-deriving source/date rules.

Source selection (in order, by the requested valid time):
  nomads  within ~10 days of now -- NOMADS production retention window (real-time/recent)
  aws     >= 2021-01-01          -- noaa-gfs-bdp-pds S3 archive
  rda     >= 2015-01-15          -- NCAR RDA/GDEX ds084.1 GFS 0.25deg archive (THREDDS; RDA moved
                                     to gdex.ucar.edu -- host is configurable, see --rda-host)
  older   no GFS 0.25deg source exists; --grib-dir (local NAM/NARR) is used if given, else the
          alternative reanalysis/archive dataset ids are listed (ds083.2 FNL 1deg 1999-2019,
          ds633.0 ERA5, ds608.0 NARR North-America-only) -- fetch NOTHING automatically here.
An explicit --source overrides all of the above.

CLI:
  metdata.py find  --case-dir D [--namelist P] [--start ..] [--hours N] [--every H]
                    [--purpose state|wps] [--source auto|nomads|aws|rda|local] [--grib-dir G]
                    [--rda-host HOST]
      -> per required time: chosen source + reason, exact URL(s) (or local file path), which GRIB
         messages are needed, and any applicable trap-check warning. No network access.
  metdata.py check ... (same selection flags)
      -> HTTP HEAD (falling back to a 1-byte ranged GET when HEAD is refused) per URL, or a local
         os.path.exists check; reports which are missing. Never downloads.
  metdata.py fetch ... [--cache-dir C] [--bbox-crop]
      -> downloads. AWS/NOMADS: byte-range fetch of ONLY the needed message via the file's own
         .idx sidecar. RDA: whole-file fetch (THREDDS fileServer has no per-message subsetting).
         local: looked up under --grib-dir. Resumable/idempotent: a file already in --cache-dir is
         never re-downloaded. --bbox-crop additionally decodes+crops each state-purpose field to
         the case bbox (+2deg margin) and caches the cropped (lats, lons, values) as a small .npz
         (a full RDA global file can be several hundred MB; the crop keeps the cache bounded).

Built-in trap-check warnings (each hit and recorded in the lab wiki before this tool existed):
  - GFS analysis f000 carries NO accumulation/time-average message for APCP, ULWRF, or SHTFL --
    CONFIRMED live against a real AWS .idx (2026-09-21 cycle: all three absent from f000's message
    list, present at f003 with fcst='0-3 hour acc/ave fcst'; TCDC/HPBL/UGRD/VGRD were present at
    f000 with fcst='anl'). At fhour=0 this tool substitutes f003 of the SAME cycle for any of
    those three fields and prints a warning; see resolve_field_request / FLUX_FIELDS_NO_F000.
  - `wgrib2 -text` (and cfgrib/xarray's own decode) present GFS rows SOUTH -> NORTH: row 0 is the
    southernmost latitude, not the northernmost. Do not assume north-up when inspecting a decoded
    field by eye.
  - 850 hPa (and other low pressure-levels) can lie BELOW GROUND on high terrain; GFS pressure-
    level fields carry missing/NaN there. Downstream code must tolerate missing pressure-level
    data, not assume every level is populated everywhere.
  - For a WPS/ungrib prep run: ungrib needs BOTH $GFSDIR and $GFS_YMD set. GFSDIR alone makes
    ungrib silently write ZERO linked GRIBFILE.* files and exit 0; metgrid then fails much later
    with an opaque "mandatory field TT not found". `find --purpose wps` prints the exact date
    range/prefix ungrib needs so this can be checked before running metgrid.

VERIFIED LIVE (2026-09-21, real network + a real pixi env built from envs/grib/pixi.toml): the
AWS (noaa-gfs-bdp-pds) and RDA (thredds.rda.ucar.edu) URL formats both return HTTP 200 for a real
recent/archived cycle; STATE_FIELD_SPECS's var/level strings for APCP/TCDC/ULWRF/HPBL/UGRD/VGRD/
SHTFL all match real AWS .idx entries exactly (at f003, since three of them are absent at f000 --
see above). NOT verified: cfgrib's own shortName/typeOfLevel mapping for each of those messages
(open_grib_field was not run against a downloaded message in this pass -- do that next), the
NOMADS path/idx format (assumed identical to AWS's layout, which AWS mirrors from NOMADS prod --
plausible, not directly checked), the GFS pgrb2 APCP accumulation-bucket boundaries beyond
f000->f003 (assumed 3h-aligned, see resolve_apcp_request), and the exact Vtable.GFS shortName/
level strings for the WPS_FIELD_SPECS pressure-level + surface set (built from the public
Vtable.GFS field list, not checked against a real .idx).
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wtune as wt             # repo root: parse_namelist/extract_case/enrich_case/first_value
import wtune_solar as wsolar   # repo root: domain_bbox

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NOMADS_BASE = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod'
AWS_BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
RDA_HOST_DEFAULT = 'https://thredds.rda.ucar.edu/thredds'   # RDA moved to NCAR GDEX
                                                              # (gdex.ucar.edu); pass --rda-host /
                                                              # host= to point at the new host
                                                              # once its THREDDS path is confirmed
                                                              # -- this default was the one
                                                              # verified reachable 2026-09-22.

NOMADS_WINDOW_DAYS = 10
AWS_CUTOVER = datetime.date(2021, 1, 1)
RDA_CUTOVER = datetime.date(2015, 1, 15)
GFS_CYCLE_HOURS = (0, 6, 12, 18)

OLDER_ALTERNATIVES = [
    {'id': 'ds083.2', 'name': 'FNL 1deg', 'coverage': '1999-2019'},
    {'id': 'ds633.0', 'name': 'ERA5', 'coverage': '1940-present'},
    {'id': 'ds608.0', 'name': 'NARR (North America only)', 'coverage': '1979-2021ish'},
]

W_SOUTH_NORTH = ("decoded GFS rows run SOUTH -> NORTH (wgrib2 -text and cfgrib/xarray alike): "
                 "row 0 is the southernmost latitude, not the northernmost")
W_850_TERRAIN = ("pressure-level fields (e.g. 850 hPa) can lie BELOW GROUND on high terrain -- "
                  "expect missing/NaN there, not full coverage at every level")
W_WPS_GFSDIR = ("ungrib needs BOTH $GFSDIR and $GFS_YMD set -- GFSDIR alone silently writes ZERO "
                 "GRIBFILE.* links and exits 0; metgrid then fails later with 'mandatory field "
                 "TT not found'. Check ungrib.exe's own log lists nonzero links before metgrid.")


# ---------------------------------------------------------------------------
# GFS cycle timing + source selection (pure, offline-testable)
# ---------------------------------------------------------------------------

def gfs_cycle_for(valid_dt):
    """(cycle_dt, fhour) -- the nearest GFS cycle (00/06/12/18Z) at or before `valid_dt`, and the
    forecast hour from that cycle to `valid_dt`."""
    cycle_hour = max(h for h in GFS_CYCLE_HOURS if h <= valid_dt.hour)
    cycle_dt = valid_dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    fhour = int(round((valid_dt - cycle_dt).total_seconds() / 3600.0))
    return cycle_dt, fhour


def select_source(valid_dt, now=None, override=None, grib_dir=None):
    """(source, reason). source is one of 'nomads'/'aws'/'rda'/'local'/None (None only when no
    GFS 0.25deg archive covers `valid_dt` and no --grib-dir was given -- `reason` then lists the
    alternative datasets). An explicit `override` (not 'auto'/None) always wins. `now` is
    injectable for deterministic testing; defaults to the real UTC time."""
    if override and override != 'auto':
        return override, 'explicit --source override'
    now = now or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    age_days = (now - valid_dt).days
    if -1 <= age_days <= NOMADS_WINDOW_DAYS:
        return 'nomads', 'within ~{}d of now (NOMADS production retention window)'.format(NOMADS_WINDOW_DAYS)
    if valid_dt.date() >= AWS_CUTOVER:
        return 'aws', 'AWS noaa-gfs-bdp-pds archive (>= {})'.format(AWS_CUTOVER)
    if valid_dt.date() >= RDA_CUTOVER:
        return 'rda', 'NCAR RDA/GDEX ds084.1 GFS 0.25deg archive (>= {})'.format(RDA_CUTOVER)
    if grib_dir:
        return 'local', ('no GFS 0.25deg online archive before {}; using --grib-dir {} '
                          '(e.g. NAM/NARR)'.format(RDA_CUTOVER, grib_dir))
    alts = ', '.join('{} ({}, {})'.format(a['id'], a['name'], a['coverage']) for a in OLDER_ALTERNATIVES)
    return None, ('no GFS 0.25deg source before {}; consider {}, or pass --grib-dir for local '
                   'NAM/NARR files'.format(RDA_CUTOVER, alts))


# ---------------------------------------------------------------------------
# URL construction (pure, offline-testable)
# ---------------------------------------------------------------------------

def nomads_grib_url(dt, fhour):
    hh = dt.strftime('%H')
    return '{}/gfs.{}/{}/atmos/gfs.t{}z.pgrb2.0p25.f{:03d}'.format(
        NOMADS_BASE, dt.strftime('%Y%m%d'), hh, hh, fhour)


def nomads_idx_url(dt, fhour):
    return nomads_grib_url(dt, fhour) + '.idx'


def aws_grib_url(dt, fhour):
    hh = dt.strftime('%H')
    return '{}/gfs.{}/{}/atmos/gfs.t{}z.pgrb2.0p25.f{:03d}'.format(
        AWS_BASE, dt.strftime('%Y%m%d'), hh, hh, fhour)


def aws_idx_url(dt, fhour):
    return aws_grib_url(dt, fhour) + '.idx'


def rda_catalog_url(dt, host=None):
    base = host or RDA_HOST_DEFAULT
    return '{}/catalog/files/g/d084001/{}/{}/catalog.html'.format(
        base, dt.strftime('%Y'), dt.strftime('%Y%m%d'))


def rda_fileserver_url(dt, fhour, host=None):
    base = host or RDA_HOST_DEFAULT
    fname = 'gfs.0p25.{}.f{:03d}.grib2'.format(dt.strftime('%Y%m%d%H'), fhour)
    return '{}/fileServer/files/g/d084001/{}/{}/{}'.format(
        base, dt.strftime('%Y'), dt.strftime('%Y%m%d'), fname)


def nam_local_path(grib_dir, dt, fhour):
    """NAM local filename convention, e.g. 20190130.nam.t00z.awphys00.tm00.grib2 (NAM's own
    2-digit forecast-hour padding, not GFS's 3-digit f-hour)."""
    return os.path.join(grib_dir, '{}.nam.t{}z.awphys{:02d}.tm00.grib2'.format(
        dt.strftime('%Y%m%d'), dt.strftime('%H'), fhour))


def narr_local_path(grib_dir, dt):
    """NARR local filename convention -- BEST-EFFORT GUESS (NARR is a 3-hourly reanalysis with no
    forecast-hour axis, unlike NAM/GFS): merged_AWIP32.<YYYYMMDD><HH>.RS.grib2. Verify against
    the actual files under a real --grib-dir (e.g. /nfs/data/wrf/sc26-p2/grib-narr) before relying
    on this for anything beyond a `find`/`check` listing."""
    return os.path.join(grib_dir, 'merged_AWIP32.{}.RS.grib2'.format(dt.strftime('%Y%m%d%H')))


def local_candidates(grib_dir, dt, fhour):
    """Every local filename convention this module knows, in preference order -- `check`/`fetch`
    use the first one that exists on disk."""
    return [nam_local_path(grib_dir, dt, fhour), narr_local_path(grib_dir, dt)]


def url_for(source, cycle_dt, fhour, host=None, grib_dir=None):
    if source == 'aws':
        return aws_grib_url(cycle_dt, fhour)
    if source == 'nomads':
        return nomads_grib_url(cycle_dt, fhour)
    if source == 'rda':
        return rda_fileserver_url(cycle_dt, fhour, host=host)
    if source == 'local':
        for cand in local_candidates(grib_dir, cycle_dt, fhour):
            if os.path.exists(cand):
                return cand
        return local_candidates(grib_dir, cycle_dt, fhour)[0]   # first convention, for reporting
    raise ValueError('unknown source {!r}'.format(source))


# ---------------------------------------------------------------------------
# .idx parsing + byte-range selection (pure, offline-testable)
# ---------------------------------------------------------------------------

def parse_idx(text):
    """Parse a GFS/NAM .idx sidecar's text into an ordered list of message records.

    Format (one message per line, colon-separated):
      <msg_num>:<byte_offset>:d=<YYYYMMDDHH>:<VAR>:<LEVEL>:<FORECAST_DESC>[:...]
    The byte offset is the START of the message; the .idx carries no explicit length, so the END
    of a message is the NEXT message's start offset (or EOF for the last message) -- see
    `byte_range`.
    """
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) < 5:
            continue
        try:
            num = int(parts[0])
            offset = int(parts[1])
        except ValueError:
            continue
        date = parts[2][2:] if parts[2].startswith('d=') else parts[2]
        entries.append({'num': num, 'offset': offset, 'date': date, 'var': parts[3],
                         'level': parts[4], 'fcst': parts[5] if len(parts) > 5 else '',
                         'raw': line})
    entries.sort(key=lambda e: e['offset'])
    return entries


def find_message(entries, var, level_substr=None, fcst_substr=None):
    """First entry whose `var` matches exactly (case-insensitive) and whose `level`/`fcst`
    contain the given substrings (case-insensitive), or None."""
    for e in entries:
        if e['var'].upper() != var.upper():
            continue
        if level_substr and level_substr.lower() not in e['level'].lower():
            continue
        if fcst_substr and fcst_substr.lower() not in e['fcst'].lower():
            continue
        return e
    return None


def byte_range(entries, entry):
    """(start, end) inclusive byte offsets for one message's HTTP Range header. `end` is None
    for the last message in the file (open-ended range; the server reads to EOF)."""
    by_offset = sorted(entries, key=lambda e: e['offset'])
    idx = by_offset.index(entry)
    start = entry['offset']
    end = by_offset[idx + 1]['offset'] - 1 if idx + 1 < len(by_offset) else None
    return start, end


# ---------------------------------------------------------------------------
# HTTP (thin, swappable wrappers -- tests patch these directly, no `requests` import needed to
# test the callers: unittest.mock.patch.object(metdata, '_http_head', fake) etc.)
# ---------------------------------------------------------------------------

def _http_head(url, timeout=30):
    import requests
    r = requests.head(url, timeout=timeout, allow_redirects=True)
    return r.status_code, dict(r.headers)


def _http_get_range_status(url, timeout=30):
    """1-byte ranged GET, for servers that refuse HEAD (405/501)."""
    import requests
    r = requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=timeout, stream=True)
    status = r.status_code
    r.close()
    return status, dict(r.headers)


def _http_get_text(url, timeout=60):
    import requests
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _http_download(url, dest_path, range_header=None, timeout=120):
    import requests
    headers = {'Range': range_header} if range_header else None
    tmp = dest_path + '.part'
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    os.replace(tmp, dest_path)
    return dest_path


def check_url(url_or_path):
    """{'exists', 'status', 'detail'} for one URL (HTTP HEAD, falling back to a 1-byte ranged GET
    when HEAD is refused) or local path (os.path.exists). Never downloads."""
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
        try:
            status, _headers = _http_head(url_or_path)
        except Exception as exc:
            return {'exists': False, 'status': None, 'detail': str(exc)}
        if 200 <= status < 300:
            return {'exists': True, 'status': status, 'detail': url_or_path}
        if status in (405, 501):
            try:
                status2, _h2 = _http_get_range_status(url_or_path)
                return {'exists': status2 in (200, 206), 'status': status2, 'detail': url_or_path}
            except Exception as exc:
                return {'exists': False, 'status': status, 'detail': str(exc)}
        return {'exists': False, 'status': status, 'detail': url_or_path}
    return {'exists': os.path.exists(url_or_path), 'status': 'local', 'detail': url_or_path}


# ---------------------------------------------------------------------------
# Download (network -- not exercised by the offline test suite beyond mocked _http_* calls)
# ---------------------------------------------------------------------------

def _cache_path(cache_dir, key):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, key)


def download_full(url, dest_path, timeout=120):
    """Plain streamed GET to dest_path. Never re-downloads: a no-op if dest_path exists."""
    if os.path.exists(dest_path):
        return dest_path
    return _http_download(url, dest_path, timeout=timeout)


def _idx_text_cached(idx_url, cache_dir, key):
    idx_path = _cache_path(cache_dir, key + '.idx')
    if os.path.exists(idx_path):
        with open(idx_path) as fh:
            return fh.read()
    text = _http_get_text(idx_url)
    with open(idx_path, 'w') as fh:
        fh.write(text)
    return text


def fetch_idx_message(source, dt, fhour, var, level_substr, cache_dir, host=None, fcst_substr=None):
    """Fetch ONE GRIB message from an AWS/NOMADS pgrb2 file via its .idx byte range. Caches the
    .idx text and the extracted message bytes separately under cache_dir; never re-downloads
    either once cached."""
    if source == 'aws':
        grib_url, idx_url = aws_grib_url(dt, fhour), aws_idx_url(dt, fhour)
    elif source == 'nomads':
        grib_url, idx_url = nomads_grib_url(dt, fhour), nomads_idx_url(dt, fhour)
    else:
        raise ValueError('fetch_idx_message only supports aws/nomads, got {!r}'.format(source))
    key_base = '{}_gfs_{}_f{:03d}'.format(source, dt.strftime('%Y%m%d%H'), fhour)
    entries = parse_idx(_idx_text_cached(idx_url, cache_dir, key_base))
    entry = find_message(entries, var, level_substr, fcst_substr)
    if entry is None:
        raise RuntimeError('message {}/{} not found in {}.idx'.format(var, level_substr, key_base))
    start, end = byte_range(entries, entry)
    msg_path = _cache_path(cache_dir, key_base + '_{}_{}.grib2'.format(var, entry['num']))
    if os.path.exists(msg_path):
        return msg_path
    rng = 'bytes={}-{}'.format(start, end if end is not None else '')
    return _http_download(grib_url, msg_path, range_header=rng)


def fetch_rda_file(dt, fhour, cache_dir, host=None):
    """Fetch a FULL RDA/GDEX GFS file (THREDDS fileServer offers no per-message .idx subsetting
    the way AWS/NOMADS do) and cache it; caller crops/extracts fields after decoding."""
    dest = _cache_path(cache_dir, 'rda_gfs_{}_f{:03d}.grib2'.format(dt.strftime('%Y%m%d%H'), fhour))
    return download_full(rda_fileserver_url(dt, fhour, host=host), dest)


def fetch_local(grib_dir, dt, fhour):
    for cand in local_candidates(grib_dir, dt, fhour):
        if os.path.exists(cand):
            return cand
    raise RuntimeError('local GRIB not found under {} for {} f{:03d} (tried {})'.format(
        grib_dir, dt.strftime('%Y%m%d%H'), fhour, local_candidates(grib_dir, dt, fhour)))


def fetch_message(field_key, cycle_dt, fhour, source, cache_dir, grib_dir=None, host=None, specs=None):
    """Download (or locate locally) ONE GRIB message/file needed for `field_key` at
    (cycle_dt, fhour) from `source`. Returns a local file path."""
    specs = specs if specs is not None else STATE_FIELD_SPECS
    spec = specs[field_key]
    if source in ('aws', 'nomads'):
        return fetch_idx_message(source, cycle_dt, fhour, spec['var'], spec['level'], cache_dir, host=host)
    if source == 'rda':
        return fetch_rda_file(cycle_dt, fhour, cache_dir, host=host)
    if source == 'local':
        if not grib_dir:
            raise RuntimeError('--grib-dir is required for --source local')
        return fetch_local(grib_dir, cycle_dt, fhour)
    raise ValueError('unknown source {!r}'.format(source))


# ---------------------------------------------------------------------------
# GRIB decode + bbox crop (cfgrib/xarray lazy-imported; crop is pure numpy)
# ---------------------------------------------------------------------------

def open_grib_field(path, var_name, typeOfLevel_filter=None, extra_filters=None):
    """Decode ONE field from a GRIB file via cfgrib/xarray. Returns (lats_1d, lons_1d,
    values_2d); lons in the source's native convention (GFS: 0..360). Raises RuntimeError with a
    clear message if xarray itself is missing; if xarray is present but the cfgrib engine (and
    therefore eccodes) is not, xarray's own ValueError for an unrecognized engine propagates
    unchanged -- never returns fake data either way."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError('xarray/cfgrib not installed -- cannot decode GRIB ({})'.format(exc))
    filters = {'shortName': var_name}
    if typeOfLevel_filter:
        filters['typeOfLevel'] = typeOfLevel_filter
    if extra_filters:
        filters.update(extra_filters)
    dset = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'filter_by_keys': filters})
    try:
        varname = list(dset.data_vars)[0]
        values = np.asarray(dset[varname].values, dtype=float)
        lats = np.asarray(dset['latitude'].values, dtype=float)
        lons = np.asarray(dset['longitude'].values, dtype=float)
    finally:
        dset.close()
    return lats, lons, values


def fetch_and_decode_state_field(field_key, cycle_dt, fhour, source, cache_dir, grib_dir=None, host=None):
    spec = STATE_FIELD_SPECS[field_key]
    path = fetch_message(field_key, cycle_dt, fhour, source, cache_dir, grib_dir=grib_dir, host=host)
    return open_grib_field(path, spec['cfgrib_short'], typeOfLevel_filter=spec['cfgrib_level'])


def normalize_lon_0_360(lon):
    return np.mod(lon, 360.0)


def crop_bbox(lats_1d, lons_1d, values_2d, lat_min, lat_max, lon_min, lon_max, margin_deg=2.0):
    """Crop a regular lat/lon grid (values_2d shape (n_lat, n_lon)) to [lat_min-m, lat_max+m] x
    [lon_min-m, lon_max+m]. Transparently handles GFS's 0..360 longitude convention: the bbox
    (given in -180..180, WRF convention) is converted to 0..360 when the source grid itself uses
    0..360. Does not handle a bbox that wraps the antimeridian in 0..360 space (falls back to the
    full longitude range in that case -- rare for a WRF regional domain)."""
    lats_1d = np.asarray(lats_1d, dtype=float)
    lons_1d = np.asarray(lons_1d, dtype=float)
    lat0, lat1 = lat_min - margin_deg, lat_max + margin_deg
    lon0, lon1 = lon_min - margin_deg, lon_max + margin_deg
    if lons_1d.size and lons_1d.max() > 180.0:
        lon0, lon1 = float(normalize_lon_0_360(lon0)), float(normalize_lon_0_360(lon1))
        if lon1 < lon0:
            lon0, lon1 = 0.0, 360.0
    lat_idx = np.where((lats_1d >= min(lat0, lat1)) & (lats_1d <= max(lat0, lat1)))[0]
    lon_idx = np.where((lons_1d >= lon0) & (lons_1d <= lon1))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError('bbox crop produced an empty grid')
    cropped = values_2d[np.ix_(lat_idx, lon_idx)]
    return lats_1d[lat_idx], lons_1d[lon_idx], cropped


def case_bbox(case, wrfinput_path=None):
    """(lat_min, lat_max, lon_min, lon_max) for a case -- exact from wrfinput XLAT/XLONG extrema
    when available (needs netCDF4), else wtune_solar.domain_bbox's analytic approximation."""
    if wrfinput_path and os.path.exists(wrfinput_path):
        try:
            import netCDF4
            nc = netCDF4.Dataset(wrfinput_path)
            try:
                xlat = nc.variables['XLAT'][0]
                xlong = nc.variables['XLONG'][0]
                return (float(np.min(xlat)), float(np.max(xlat)),
                        float(np.min(xlong)), float(np.max(xlong)))
            finally:
                nc.close()
        except Exception:
            pass
    bbox = wsolar.domain_bbox(case)
    if bbox is None:
        raise RuntimeError('no wrfinput and no domain center lat/lon in namelist -- cannot '
                            'determine a bbox')
    return bbox


# ---------------------------------------------------------------------------
# Accumulation/average-window field trap (state purpose): f000 has no APCP -- CONFIRMED (live
# check, 2026-09-21 AWS cycle) that ULWRF and SHTFL are ALSO absent from f000's message set, not
# just APCP. TCDC/HPBL/UGRD/VGRD carry an instantaneous 'anl' message at f000 and need no
# substitution. See resolve_field_request's docstring for the exact verification command.
# ---------------------------------------------------------------------------

# Fields whose GRIB message is an accumulation or time-average over a forecast window, and
# therefore CANNOT exist at f000 (nothing to accumulate/average over at t=0 of a cycle). Verified
# live against a real AWS .idx (gfs.20260921/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx): APCP/ULWRF/
# SHTFL are absent from that file's message list; f003's .idx has all three with fcst='0-3 hour
# acc fcst' (APCP) / '0-3 hour ave fcst' (ULWRF, SHTFL). TCDC/HPBL/UGRD/VGRD were present at f000
# with fcst='anl' in the same check.
FLUX_FIELDS_NO_F000 = ('apcp', 'ulwrf', 'shtfl')


def resolve_field_request(field_key, cycle_dt, fhour):
    """(eff_cycle_dt, eff_fhour, warning_or_None) -- substitutes f003 of the SAME cycle for any
    FLUX_FIELDS_NO_F000 field requested at fhour=0 (confirmed absent there on the real product;
    see module-level comment above), with a warning naming the field. Every other field/fhour
    combination is returned unchanged."""
    if field_key in FLUX_FIELDS_NO_F000 and fhour == 0:
        return cycle_dt, 3, ('{} absent from analysis f000 (confirmed against a real .idx); '
                              'using f003 instead'.format(field_key.upper()))
    return cycle_dt, fhour, None


def resolve_apcp_request(cycle_dt, fhour):
    """(eff_cycle_dt, eff_fhour, window_hours, warning_or_None) for fetching APCP specifically --
    reuses resolve_field_request for the f000->f003 substitution (APCP is one of
    FLUX_FIELDS_NO_F000) and additionally resolves the accumulation WINDOW
    precip_rate_from_apcp_window needs. NEEDS CLUSTER VERIFICATION beyond the confirmed f000->f003
    substitution: the operational GFS pgrb2.0p25 product's APCP bucket is documented to reset at
    6h boundaries (0-3h, then 3-6h, i.e. NOT cumulative from init) -- assumed 3h-wide for every
    f000/f003/f006/... product hour here; confirm against a real .idx `fcst` field for forecast
    hours beyond f003 before trusting precip rate there.
    """
    eff_cycle, eff_fhour, warn = resolve_field_request('apcp', cycle_dt, fhour)
    if warn:
        return eff_cycle, eff_fhour, 3.0, warn
    window = 3.0 if eff_fhour % 3 == 0 else float(eff_fhour % 6 or 6)
    return eff_cycle, eff_fhour, window, None


# ---------------------------------------------------------------------------
# Field specs: which GRIB messages are needed (purpose='state' vs purpose='wps')
# ---------------------------------------------------------------------------

# Best-documented guess at the GFS pgrb2.0p25 message identity for each field tools/grib_state.py
# needs. VERIFY against a real downloaded .idx / cfgrib decode before trusting end to end.
STATE_FIELD_SPECS = {
    'apcp':  dict(var='APCP',  level='surface',            cfgrib_short='tp',    cfgrib_level='surface'),
    'tcdc':  dict(var='TCDC',  level='entire atmosphere',  cfgrib_short='tcc',   cfgrib_level='atmosphere'),
    'ulwrf': dict(var='ULWRF', level='top of atmosphere',  cfgrib_short='ulwrf', cfgrib_level='nominalTop'),
    'hpbl':  dict(var='HPBL',  level='surface',            cfgrib_short='blh',   cfgrib_level='surface'),
    'u10':   dict(var='UGRD',  level='10 m above ground',  cfgrib_short='u10',   cfgrib_level='heightAboveGround'),
    'v10':   dict(var='VGRD',  level='10 m above ground',  cfgrib_short='v10',   cfgrib_level='heightAboveGround'),
    'shtfl': dict(var='SHTFL', level='surface',            cfgrib_short='sshf',  cfgrib_level='surface'),
}

# GFS 0.25deg pgrb2 mandatory pressure levels (hPa), standard product set.
GFS_PRESSURE_LEVELS_HPA = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500,
                            450, 400, 350, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10)
_WPS_PLEVEL_VARS = ('HGT', 'TMP', 'RH', 'UGRD', 'VGRD')
_WPS_SURFACE_FIELDS = [
    ('PRMSL', 'mean sea level'), ('PRES', 'surface'), ('HGT', 'surface'),
    ('TMP', '2 m above ground'), ('RH', '2 m above ground'),
    ('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'),
    ('TMP', 'surface'),        # SKINTEMP in Vtable.GFS
    ('LAND', 'surface'), ('ICEC', 'surface'),
    ('WEASD', 'surface'), ('SNOD', 'surface'),
    ('TMP', '0-10 cm below ground'), ('TMP', '10-40 cm below ground'),
    ('TMP', '40-100 cm below ground'), ('TMP', '100-200 cm below ground'),
    ('SOILW', '0-10 cm below ground'), ('SOILW', '10-40 cm below ground'),
    ('SOILW', '40-100 cm below ground'), ('SOILW', '100-200 cm below ground'),
]


def _build_wps_field_specs():
    """The full WPS/ungrib GRIB message set (Vtable.GFS): mandatory pressure levels x
    (HGT, TMP, RH, UGRD, VGRD), plus the fixed surface/soil field list. See module docstring's
    'NEEDS THE CLUSTER TO VERIFY' -- built from the public Vtable.GFS field list, not confirmed
    against a real .idx."""
    specs = {}
    for var, level in _WPS_SURFACE_FIELDS:
        specs['{}_{}'.format(var, level).replace(' ', '_')] = dict(var=var, level=level)
    for var in _WPS_PLEVEL_VARS:
        for lev in GFS_PRESSURE_LEVELS_HPA:
            level = '{} mb'.format(lev)
            specs['{}_{}mb'.format(var, lev)] = dict(var=var, level=level)
    return specs


WPS_FIELD_SPECS = _build_wps_field_specs()


# ---------------------------------------------------------------------------
# Case / namelist loading + time resolution (shared with tools/grib_state.py)
# ---------------------------------------------------------------------------

def load_case(case_dir, namelist=None, wrfinput=None):
    """Resolve namelist + optional wrfinput and extract the namelist-derived case dict, the same
    convention wtune_case.find_case_inputs / wtune.extract_case use."""
    nl_path = namelist or os.path.join(case_dir, 'namelist.input')
    wi_path = wrfinput
    if wi_path is None and case_dir and os.path.isdir(case_dir):
        for name in sorted(os.listdir(case_dir)):
            if name.startswith('wrfinput_d0'):
                wi_path = os.path.join(case_dir, name)
                break
    if not os.path.exists(nl_path):
        raise RuntimeError('no namelist found at {} (pass --namelist)'.format(nl_path))
    nl = wt.parse_namelist(nl_path)
    case = wt.extract_case(nl)
    if wi_path and os.path.exists(wi_path):
        case, _err = wt.enrich_case(case, wi_path)
    return case, nl_path, wi_path


def resolve_times(case, start_arg, hours_arg, every):
    """List of datetimes at start, start+every, ... up to (and including, if it lands exactly)
    start+hours. start/hours default from the namelist case dict when not given explicitly.
    purpose='state' time axis."""
    if start_arg:
        start_dt = datetime.datetime.strptime(start_arg, '%Y-%m-%d_%H')
    else:
        start_dt = datetime.datetime(
            int(case.get('start_year') or 2020), int(case.get('start_month') or 1),
            int(case.get('start_day') or 1), int(case.get('start_hour') or 0))
    total_hours = float(hours_arg) if hours_arg is not None else float(case.get('run_total_hours') or 0)
    n_steps = int(total_hours // every) + 1
    return [start_dt + datetime.timedelta(hours=every * i) for i in range(n_steps)]


def resolve_wps_times(case, case_dir, namelist_wps=None):
    """Times for purpose='wps': every `interval_seconds` from namelist.wps's &share group (or a
    6h default when namelist.wps is absent/unparseable), spanning the case's start..end."""
    wps_path = namelist_wps or os.path.join(case_dir, 'namelist.wps')
    interval_s = 21600   # 6h -- WPS's own conventional default when interval_seconds is unset
    if os.path.exists(wps_path):
        try:
            nlw = wt.parse_namelist(wps_path)
            share = nlw.get('share', {})
            if 'interval_seconds' in share:
                interval_s = int(float(wt.first_value(share['interval_seconds'])))
        except Exception:
            pass
    every_h = interval_s / 3600.0
    start_dt = datetime.datetime(
        int(case.get('start_year') or 2020), int(case.get('start_month') or 1),
        int(case.get('start_day') or 1), int(case.get('start_hour') or 0))
    total_hours = float(case.get('run_total_hours') or 0)
    n_steps = (int(total_hours // every_h) + 1) if every_h > 0 else 1
    return [start_dt + datetime.timedelta(hours=every_h * i) for i in range(n_steps)]


# ---------------------------------------------------------------------------
# Plan construction (shared by find/check/fetch) + CLI
# ---------------------------------------------------------------------------

def build_plan(case_dir, namelist=None, wrfinput=None, start=None, hours=None, every=3.0,
               purpose='state', source='auto', grib_dir=None, host=None, now=None):
    """(plan, general_warnings). plan is a list of per-time dicts: valid_time, cycle, fhour,
    source, reason, messages (list of {field, var, level, fhour, url[, warning]}). Pure/offline
    except that `source`/reason selection and URL strings never touch the network -- only
    check()/fetch() do."""
    case, _nl_path, wi_path = load_case(case_dir, namelist, wrfinput)
    if purpose == 'wps':
        times = resolve_wps_times(case, case_dir)
        specs = WPS_FIELD_SPECS
        general_warnings = [W_SOUTH_NORTH, W_850_TERRAIN, W_WPS_GFSDIR]
    else:
        times = resolve_times(case, start, hours, every)
        specs = STATE_FIELD_SPECS
        general_warnings = [W_SOUTH_NORTH, W_850_TERRAIN]

    plan = []
    for t in times:
        cycle_dt, fhour = gfs_cycle_for(t)
        src, reason = select_source(t, now=now, override=source, grib_dir=grib_dir)
        entry = {'valid_time': t, 'cycle': cycle_dt, 'fhour': fhour, 'source': src,
                 'reason': reason, 'messages': []}
        if src is None:
            plan.append(entry)
            continue
        for key, spec in specs.items():
            eff_cycle, eff_fhour, msg_warn = cycle_dt, fhour, None
            if purpose == 'state':
                eff_cycle, eff_fhour, msg_warn = resolve_field_request(key, cycle_dt, fhour)
            url = url_for(src, eff_cycle, eff_fhour, host=host, grib_dir=grib_dir)
            msg = {'field': key, 'var': spec['var'], 'level': spec['level'], 'fhour': eff_fhour,
                   'url': url}
            if purpose == 'state':
                msg['cfgrib_short'] = spec['cfgrib_short']
                msg['cfgrib_level'] = spec['cfgrib_level']
            if msg_warn:
                msg['warning'] = msg_warn
            entry['messages'].append(msg)
        plan.append(entry)
    return plan, general_warnings


def print_find(plan, general_warnings):
    for w in general_warnings:
        print('WARNING: {}'.format(w))
    for entry in plan:
        print('{}  cycle={} fhour={:03d}  source={} ({})'.format(
            entry['valid_time'].strftime('%Y-%m-%d_%H:%M:%S'),
            entry['cycle'].strftime('%Y%m%d%H') if entry['source'] is not None else '-',
            entry['fhour'], entry['source'] or 'NONE', entry['reason']))
        for msg in entry['messages']:
            line = '    {:8s} {} / {}  f{:03d}  {}'.format(
                msg['field'], msg['var'], msg['level'], msg['fhour'], msg['url'])
            if msg.get('warning'):
                line += '   WARNING: {}'.format(msg['warning'])
            print(line)


def do_check(plan):
    n_ok, n_missing = 0, 0
    for entry in plan:
        if entry['source'] is None:
            continue
        for msg in entry['messages']:
            result = check_url(msg['url'])
            status = 'OK' if result['exists'] else 'MISSING'
            print('{} {} {} f{:03d} {} -- {}'.format(
                status, entry['valid_time'].strftime('%Y-%m-%d_%H'), msg['field'], msg['fhour'],
                msg['url'], result['detail']))
            if result['exists']:
                n_ok += 1
            else:
                n_missing += 1
    print('{} ok, {} missing'.format(n_ok, n_missing))
    return 1 if n_missing else 0


def _bbox_crop_and_cache(raw_path, msg, cache_dir, bbox):
    if 'cfgrib_short' not in msg:
        return raw_path   # purpose='wps' messages: no per-field decode mapping here, skip crop
    lats, lons, values = open_grib_field(raw_path, msg['cfgrib_short'],
                                          typeOfLevel_filter=msg.get('cfgrib_level'))
    lat0, lat1, lon0, lon1 = bbox
    clats, clons, cvals = crop_bbox(lats, lons, values, lat0, lat1, lon0, lon1)
    out_path = os.path.join(cache_dir, os.path.basename(raw_path) + '.{}.cropped.npz'.format(msg['field']))
    if not os.path.exists(out_path):
        np.savez_compressed(out_path, lats=clats, lons=clons, values=cvals)
    return out_path


def do_fetch(plan, cache_dir, bbox_crop, grib_dir, host, case=None, wrfinput_path=None):
    os.makedirs(cache_dir, exist_ok=True)
    bbox = case_bbox(case, wrfinput_path) if (bbox_crop and case is not None) else None
    results = []
    for entry in plan:
        if entry['source'] is None:
            results.append({'valid_time': entry['valid_time'], 'status': 'skipped', 'detail': entry['reason']})
            continue
        for msg in entry['messages']:
            try:
                path = fetch_message(msg['field'], entry['cycle'], msg['fhour'], entry['source'],
                                      cache_dir, grib_dir=grib_dir, host=host,
                                      specs=(STATE_FIELD_SPECS if 'cfgrib_short' in msg else WPS_FIELD_SPECS))
                if bbox_crop and bbox is not None:
                    path = _bbox_crop_and_cache(path, msg, cache_dir, bbox)
                results.append({'valid_time': entry['valid_time'], 'field': msg['field'],
                                 'status': 'ok', 'path': path})
                print('OK   {} {} -> {}'.format(entry['valid_time'].strftime('%Y-%m-%d_%H'),
                                                 msg['field'], path))
            except Exception as exc:
                results.append({'valid_time': entry['valid_time'], 'field': msg['field'],
                                 'status': 'error', 'detail': str(exc)})
                print('FAIL {} {} -- {}'.format(entry['valid_time'].strftime('%Y-%m-%d_%H'),
                                                 msg['field'], exc))
    return results


def _add_common_args(sp):
    sp.add_argument('--case-dir', required=True, help='directory with namelist.input (+ optional wrfinput_d0N)')
    sp.add_argument('--namelist', default=None)
    sp.add_argument('--wrfinput', default=None)
    sp.add_argument('--start', default=None, help='YYYY-MM-DD_HH (default: namelist start date)')
    sp.add_argument('--hours', type=float, default=None, help='default: namelist run_total_hours')
    sp.add_argument('--every', type=float, default=3.0, help='purpose=state sampling interval, hours')
    sp.add_argument('--purpose', default='state', choices=['state', 'wps'])
    sp.add_argument('--source', default='auto', choices=['auto', 'nomads', 'aws', 'rda', 'local'])
    sp.add_argument('--grib-dir', default=None)
    sp.add_argument('--rda-host', default=None, help='override the RDA/GDEX THREDDS base URL')


def main(argv=None):
    p = argparse.ArgumentParser(prog='metdata.py',
                                 description='GFS/NAM/NARR source selection, availability, and download')
    sub = p.add_subparsers(dest='cmd', required=True)
    find_p = sub.add_parser('find', help='print chosen source/URLs/messages per required time (no network)')
    _add_common_args(find_p)
    check_p = sub.add_parser('check', help='confirm each URL/local file exists (HEAD/listing only, no download)')
    _add_common_args(check_p)
    fetch_p = sub.add_parser('fetch', help='download (resumable, never re-downloads a cached file)')
    _add_common_args(fetch_p)
    fetch_p.add_argument('--cache-dir', required=True)
    fetch_p.add_argument('--bbox-crop', action='store_true',
                          help='decode+crop each purpose=state field to the case bbox (+2deg) before caching')
    args = p.parse_args(argv)

    plan, general_warnings = build_plan(
        args.case_dir, args.namelist, args.wrfinput, args.start, args.hours, args.every,
        args.purpose, args.source, args.grib_dir, args.rda_host)

    if args.cmd == 'find':
        print_find(plan, general_warnings)
        return 0
    if args.cmd == 'check':
        return do_check(plan)
    if args.cmd == 'fetch':
        case, _nl, wi_path = load_case(args.case_dir, args.namelist, args.wrfinput)
        results = do_fetch(plan, args.cache_dir, args.bbox_crop, args.grib_dir, args.rda_host,
                            case=case, wrfinput_path=wi_path)
        n_fail = sum(1 for r in results if r['status'] == 'error')
        return 1 if n_fail else 0
    return 1


if __name__ == '__main__':
    sys.exit(main())

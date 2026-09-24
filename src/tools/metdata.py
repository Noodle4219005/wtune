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
import calendar
import datetime
import os
import re
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import wtune_cli as wt         # src/wtune_cli.py: parse_namelist/extract_case/enrich_case/first_value
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
    # 2026-09-24 (case_rebuild.py analog, R6-C): pattern-matching fields for a candidate date's
    # weather vs. the case's own state. cfgrib short names follow the SAME convention as the
    # entries above (var/level VERIFIED against a real GFS pgrb2 .idx layout; cfgrib's own
    # shortName/typeOfLevel mapping is NOT independently verified here either, same caveat as
    # this module's docstring already states for the original 7 entries).
    'hgt500':  dict(var='HGT',  level='500 mb',              cfgrib_short='gh',   cfgrib_level='isobaricInhPa'),
    'tmp850':  dict(var='TMP',  level='850 mb',               cfgrib_short='t',    cfgrib_level='isobaricInhPa'),
    'ugrd850': dict(var='UGRD', level='850 mb',               cfgrib_short='u',    cfgrib_level='isobaricInhPa'),
    'vgrd850': dict(var='VGRD', level='850 mb',               cfgrib_short='v',    cfgrib_level='isobaricInhPa'),
    'pwat':    dict(var='PWAT', level='entire atmosphere (considered as a single layer)',
                    cfgrib_short='pwat', cfgrib_level='atmosphereSingleLayer'),
    'tmp2m':   dict(var='TMP',  level='2 m above ground',     cfgrib_short='2t',   cfgrib_level='heightAboveGround'),
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
    # 2026-09-24 fix: these 8 soil entries were 'TMP'/'cm' (e.g. ('TMP', '0-10 cm below
    # ground')) -- stale and WRONG. The live GFS .idx (verified 2023-03-31 00Z f000) names soil
    # temperature/moisture TSOIL/SOILW with METER-denominated layers, e.g.
    # 'TSOIL:0-0.1 m below ground'. The old strings matched NOTHING in a real file, and the
    # not-found path was never treated as fatal, so severe2023's met_em silently ended up with
    # NUM_METGRID_SOIL_LEVELS=0. See select_wps_messages/validate_wps_selection below for the
    # dynamic, idx-driven replacement the actual WPS fetch path now uses; this static list is
    # kept correct too, for any caller still using WPS_FIELD_SPECS directly (S3).
    ('TSOIL', '0-0.1 m below ground'), ('TSOIL', '0.1-0.4 m below ground'),
    ('TSOIL', '0.4-1 m below ground'), ('TSOIL', '1-2 m below ground'),
    ('SOILW', '0-0.1 m below ground'), ('SOILW', '0.1-0.4 m below ground'),
    ('SOILW', '0.4-1 m below ground'), ('SOILW', '1-2 m below ground'),
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
# Idx-driven WPS message selection (2026-09-24; aws/nomads per-message path only -- the WHOLE
# reason WPS_FIELD_SPECS/GFS_PRESSURE_LEVELS_HPA above are wrong: they hardcode which
# (var, level) pairs to ask for, so a real file whose pressure levels or soil-layer
# names/units differ (GFS actually carries 41 levels down to 0.01 mb, not
# GFS_PRESSURE_LEVELS_HPA's 26; soil layers are TSOIL/SOILW in METERS, not TMP/'cm') just
# silently returns nothing for those messages -- and the old code never treated "message not
# found" as fatal, so severe2023's met_em ended up with NUM_METGRID_SOIL_LEVELS=0,
# num_metgrid_levels=27 (should be 4 / 32). select_wps_messages selects directly from the
# FILE'S OWN parsed .idx (parse_idx's output) instead; validate_wps_selection makes "not enough
# messages" a hard failure, with the missing (var, level) list printed, not a silent partial
# fetch that looks like success.
# ---------------------------------------------------------------------------

_MB_LEVEL_RE = re.compile(r'^([\d.]+)\s*mb$', re.I)
_SOIL_LAYER_RE = re.compile(r'^[\d.]+\s*-\s*[\d.]+\s*m below ground$', re.I)

_WPS_SURFACE_SET = frozenset((var, level.lower()) for var, level in (
    ('PRMSL', 'mean sea level'),
    ('PRES', 'surface'), ('HGT', 'surface'), ('TMP', 'surface'),
    ('LAND', 'surface'), ('ICEC', 'surface'), ('WEASD', 'surface'), ('SNOD', 'surface'),
    ('TMP', '2 m above ground'), ('RH', '2 m above ground'),
    ('UGRD', '10 m above ground'), ('VGRD', '10 m above ground'),
))
_WPS_SOIL_VARS = frozenset(('TSOIL', 'SOILW'))
_WPS_SOIL_LAYERS_REQUIRED = frozenset((
    '0-0.1 m below ground', '0.1-0.4 m below ground',
    '0.4-1 m below ground', '1-2 m below ground',
))
_WPS_SURFACE_REQUIRED = (('LAND', 'surface'), ('TMP', 'surface'), ('PRES', 'surface'))
_WPS_MIN_PRESSURE_LEVELS = 20


def select_wps_messages(entries, fcst_substr=None):
    """Select the .idx `entries` (parse_idx's result, for ONE file/cycle/fhour) Vtable.GFS
    needs, driven entirely by what that file itself actually has:
      - var in {HGT, TMP, RH, UGRD, VGRD} at any '<p> mb' level with p >= 1 mb (GFS's own
        pressure-level set, whatever it is -- not a hardcoded list);
      - the fixed surface set (PRMSL/PRES/HGT/TMP/LAND/ICEC/WEASD/SNOD/2m TMP+RH/10m UGRD+VGRD);
      - TSOIL/SOILW at any 'X-Y m below ground' layer (whatever layers the file has).
    `fcst_substr`, when given, additionally requires each entry's fcst field to contain 'anl'
    OR `fcst_substr` (an idx normally covers exactly one fhour already, so this is a defensive
    filter, not the primary selection mechanism)."""
    selected = []
    for e in entries:
        if fcst_substr:
            fcst = e.get('fcst', '').lower()
            if 'anl' not in fcst and fcst_substr.lower() not in fcst:
                continue
        var = e['var'].upper()
        level = e['level'].strip()
        # NOTE: these three checks are NOT mutually exclusive by var (TMP/HGT/UGRD/VGRD each
        # appear in BOTH the pressure-level set AND the surface set, e.g. TMP:surface,
        # TMP:2 m above ground, HGT:surface, UGRD/VGRD:10 m above ground) -- an early
        # `continue` after only the pressure-level check wrongly skipped every one of those
        # surface-level messages for those vars. Check every category per entry.
        matched = False
        if var in _WPS_PLEVEL_VARS:
            m = _MB_LEVEL_RE.match(level)
            if m and float(m.group(1)) >= 1.0:
                matched = True
        if not matched and (var, level.lower()) in _WPS_SURFACE_SET:
            matched = True
        if not matched and var in _WPS_SOIL_VARS and _SOIL_LAYER_RE.match(level):
            matched = True
        if matched:
            selected.append(e)
    return selected


def validate_wps_selection(selected):
    """Raises RuntimeError (listing every missing (var, level)/shortfall) when `selected`
    (select_wps_messages' result) cannot actually support ungrib/Vtable.GFS: any of the 4
    TSOIL + 4 SOILW layers, LAND, TMP surface, or PRES surface missing, or fewer than
    _WPS_MIN_PRESSURE_LEVELS pressure levels for ANY of HGT/TMP/RH/UGRD/VGRD. This is the fix
    for 'the previous run printed OK for every message and still produced zero soil levels' --
    an incomplete selection must fail loudly, not silently pass."""
    have = set((e['var'].upper(), e['level'].strip().lower()) for e in selected)
    missing = []
    for soil_var in sorted(_WPS_SOIL_VARS):
        for layer in sorted(_WPS_SOIL_LAYERS_REQUIRED):
            if (soil_var, layer) not in have:
                missing.append((soil_var, layer))
    for var, level in _WPS_SURFACE_REQUIRED:
        if (var, level) not in have:
            missing.append((var, level))
    plevel_counts = {}
    for e in selected:
        var = e['var'].upper()
        if var in _WPS_PLEVEL_VARS:
            plevel_counts[var] = plevel_counts.get(var, 0) + 1
    for var in _WPS_PLEVEL_VARS:
        n = plevel_counts.get(var, 0)
        if n < _WPS_MIN_PRESSURE_LEVELS:
            missing.append((var, 'only %d pressure levels (need >= %d)'
                                 % (n, _WPS_MIN_PRESSURE_LEVELS)))
    if missing:
        raise RuntimeError('WPS message selection incomplete, missing: %s' % missing)
    return True


def fetch_wps_messages_idx_driven(cycle_dt, fhour, source, cache_dir, host=None):
    """The WPS-purpose, per-message fetch for aws/nomads (S1/S3, 2026-09-24): parses the
    file's OWN .idx (live), selects via select_wps_messages, and FAILS (validate_wps_selection
    raises RuntimeError, listing exactly what's missing) rather than silently returning a
    partial set -- do_fetch's WPS+aws/nomads path uses this instead of the WPS_FIELD_SPECS
    static-lookup fetch_message path. rda/local sources are unaffected (no per-message .idx
    concept there -- fetch_rda_file/fetch_local, unchanged).
    Returns [(selected_idx_entry, local_grib2_path), ...]."""
    if source == 'aws':
        grib_url, idx_url = aws_grib_url(cycle_dt, fhour), aws_idx_url(cycle_dt, fhour)
    elif source == 'nomads':
        grib_url, idx_url = nomads_grib_url(cycle_dt, fhour), nomads_idx_url(cycle_dt, fhour)
    else:
        raise ValueError('fetch_wps_messages_idx_driven only supports aws/nomads, got {!r}'.format(source))
    key_base = '{}_gfswps_{}_f{:03d}'.format(source, cycle_dt.strftime('%Y%m%d%H'), fhour)
    entries = parse_idx(_idx_text_cached(idx_url, cache_dir, key_base))
    selected = select_wps_messages(entries)
    validate_wps_selection(selected)
    out = []
    for sel in selected:
        start, end = byte_range(entries, sel)
        msg_path = _cache_path(cache_dir, '{}_{}_{}.grib2'.format(key_base, sel['var'], sel['num']))
        if not os.path.exists(msg_path):
            rng = 'bytes={}-{}'.format(start, end if end is not None else '')
            _http_download(grib_url, msg_path, range_header=rng)
        out.append((sel, msg_path))
    return out


# ---------------------------------------------------------------------------
# Forcing source fallback chain (2026-09-24, probed live from the scc login node) -- a
# WPS/ungrib-purpose, whole-file source resolution ADDITIVE to select_source()/fetch_message()
# above, which stay UNCHANGED (tools/grib_state.py's state-purpose path keeps using them
# unmodified). Exists because select_source()'s single date-rule (nomads/aws/rda/local) reports
# "no GFS 0.25deg source" for any date before 2015-01-15 -- true for GFS 0.25deg specifically,
# but WRONG as "no forcing data exists": NCEI's GFS analysis archive (0.5/1 deg) covers back to
# 2004, NCEP's FNL back to 1999-07-30, and CFSR reanalysis covers 1979-2010.
#
# URL patterns below are VERIFIED (live HTTP, scc login node, 2026-09-24) exactly as given, not
# invented:
#   NCEI g4 (0.5deg GRIB2): .../model-gfs-g4-anl-files-old/YYYYMM/YYYYMMDD/gfsanl_4_YYYYMMDD_HH00_FFF.grb2
#     FFF in 000/003/006, cycles 00/06/12/18 -> 3h valid times. 200 for 2007-01-01 and
#     2010-01-01; 404 for 2005-01-01 (g4 starts 2007-01-01). Catalog months 200403..202005.
#   NCEI g3 (1deg): same host, model-gfs-g3-anl-files-old, gfsanl_3_YYYYMMDD_HH00_FFF.grb
#     (GRIB1, verified for 2005-01-01) or .grb2 (unverified cutover date) -- BOTH extensions are
#     offered as probe alternatives, not a single hardcoded guess (see ncei_g3_wps_entry).
#   FNL d083002 (1deg, 6h, analysis): GRIB2
#     .../d083002/grib2/YYYY/YYYY.MM/fnl_YYYYMMDD_HH_00.grib2 (verified 2010-01-01); GRIB1 before
#     2007-12-06: .../d083002/grib1/YYYY/YYYY.MM/fnl_YYYYMMDD_HH_00.grib1 (verified 2005-01-01).
#     Coverage from 1999-07-30.
#   CFSR d093000 (0.5deg, 6h, reanalysis, 1979-2010): 5-day tar per stream per year dir:
#     .../d093000/YYYY/pgbh06.gdas.YYYYMMDD-YYYYMMDD.tar and flxf06.gdas.<same span>.tar (pgbh06
#     ~1.8 GB; verified 200 for 19900101-19900105). Spans: 01-05/06-10/11-15/16-20/21-25/
#     26-<last day of month>. Vtable.CFSR; pgbh06+flxf06 are linked together into ONE ungrib pass.
#   Vtables present in scc's WPS: Vtable.GFS, Vtable.CFSR, Vtable.NNRP, Vtable.NCEP2 (verified).
#   d090000 (NNRP)'s THREDDS catalog is EMPTY -> not auto-fetchable (not wired into the chain
#   below; report unsupported with that reason if ever asked for explicitly).
# ---------------------------------------------------------------------------

NCEI_HOST = 'https://www.ncei.noaa.gov/thredds/fileServer'
FNL_GRIB2_CUTOVER = datetime.date(2007, 12, 6)   # VERIFIED: 2010-01-01 is grib2, 2005-01-01 is grib1
NCEI_G4_START = datetime.date(2007, 1, 1)        # VERIFIED: 200 at 2007-01-01/2010-01-01, 404 at 2005-01-01
NCEI_G3_START = datetime.date(2004, 3, 2)
FNL_START = datetime.date(1999, 7, 30)
CFSR_START = datetime.date(1979, 1, 1)
CFSR_END = datetime.date(2010, 12, 31)


def _as_date(valid_dt):
    return valid_dt.date() if hasattr(valid_dt, 'date') else valid_dt


class ForcingSource:
    """One WPS/ungrib forcing source. `url_fn(valid_dt, host=None)` returns
    [(local_filename, [url, ...]), ...] -- one entry per physical file this source needs for
    that valid time (normally one; CFSR's own per-window helper below needs two, one per
    stream); the inner URL list holds ORDERED ALTERNATIVES (e.g. g3's grib1-vs-grib2 extension
    guess) -- the first one a probe finds existing wins, never both required at once.
    `prefix`/`vtable` are namelist.wps &ungrib prefix / the Vtable.* file to link -- see
    tools/coldstart_sample.py's fix-wps-namelist and run.sbatch's per-window ungrib step."""
    __slots__ = ('provider', 'dataset', 'product', 'resolution_deg', 'format',
                 'forcing_interval_h', 'source_kind', 'vtable', 'prefix', 'valid_from',
                 'valid_to', 'url_fn')

    def __init__(self, provider, dataset, product, resolution_deg, format, forcing_interval_h,
                source_kind, vtable, prefix, valid_from, valid_to, url_fn):
        self.provider = provider
        self.dataset = dataset
        self.product = product
        self.resolution_deg = resolution_deg
        self.format = format
        self.forcing_interval_h = forcing_interval_h
        self.source_kind = source_kind
        self.vtable = vtable
        self.prefix = prefix
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.url_fn = url_fn

    def urls_for(self, valid_dt, host=None):
        return self.url_fn(valid_dt, host=host)

    def to_dict(self):
        return {
            'provider': self.provider, 'dataset': self.dataset, 'product': self.product,
            'resolution_deg': self.resolution_deg, 'format': self.format,
            'forcing_interval_h': self.forcing_interval_h, 'source_kind': self.source_kind,
            'vtable': self.vtable, 'prefix': self.prefix,
            'valid_from': self.valid_from.isoformat() if self.valid_from else None,
            'valid_to': self.valid_to.isoformat() if self.valid_to else None,
        }

    def __repr__(self):
        return 'ForcingSource(%s)' % self.dataset


# ---- per-dataset URL builders (pure, offline-testable) --------------------

def gfs_nomads_wps_entry(valid_dt, host=None):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    fname = 'nomads_gfs_{}_f{:03d}.grib2'.format(cycle_dt.strftime('%Y%m%d%H'), fhour)
    return [(fname, [nomads_grib_url(cycle_dt, fhour)])]


def gfs_aws_wps_entry(valid_dt, host=None):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    fname = 'aws_gfs_{}_f{:03d}.grib2'.format(cycle_dt.strftime('%Y%m%d%H'), fhour)
    return [(fname, [aws_grib_url(cycle_dt, fhour)])]


def gfs_rda_wps_entry(valid_dt, host=None):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    fname = 'rda_gfs_{}_f{:03d}.grib2'.format(cycle_dt.strftime('%Y%m%d%H'), fhour)
    return [(fname, [rda_fileserver_url(cycle_dt, fhour, host=host)])]


def ncei_g4_url(valid_dt):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    return '{}/model-gfs-g4-anl-files-old/{}/{}/gfsanl_4_{}_{}00_{:03d}.grb2'.format(
        NCEI_HOST, cycle_dt.strftime('%Y%m'), cycle_dt.strftime('%Y%m%d'),
        cycle_dt.strftime('%Y%m%d'), cycle_dt.strftime('%H'), fhour)


def ncei_g4_wps_entry(valid_dt, host=None):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    fname = 'ncei_g4_{}_f{:03d}.grb2'.format(cycle_dt.strftime('%Y%m%d%H'), fhour)
    return [(fname, [ncei_g4_url(valid_dt)])]


def ncei_g3_url(valid_dt, ext):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    return '{}/model-gfs-g3-anl-files-old/{}/{}/gfsanl_3_{}_{}00_{:03d}.{}'.format(
        NCEI_HOST, cycle_dt.strftime('%Y%m'), cycle_dt.strftime('%Y%m%d'),
        cycle_dt.strftime('%Y%m%d'), cycle_dt.strftime('%H'), fhour, ext)


def ncei_g3_wps_entry(valid_dt, host=None):
    cycle_dt, fhour = gfs_cycle_for(valid_dt)
    fname = 'ncei_g3_{}_f{:03d}.grb'.format(cycle_dt.strftime('%Y%m%d%H'), fhour)
    # extension cutover unverified for g3 (only FNL's 2007-12-06 switch is confirmed) -- offer
    # BOTH, in date-informed order, and let the probe (check_url) pick the one that exists.
    exts = ('grb', 'grb2') if _as_date(valid_dt) < FNL_GRIB2_CUTOVER else ('grb2', 'grb')
    return [(fname, [ncei_g3_url(valid_dt, e) for e in exts])]


def fnl_url(valid_dt, fmt):
    return '{}/{}/{}/fnl_{}_{}_00.{}'.format(
        'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d083002', fmt,
        valid_dt.strftime('%Y/%Y.%m'), valid_dt.strftime('%Y%m%d'), valid_dt.strftime('%H'), fmt)


def fnl_wps_entry(valid_dt, host=None):
    fname = 'fnl_{}.grib'.format(valid_dt.strftime('%Y%m%d%H'))
    fmts = ('grib1', 'grib2') if _as_date(valid_dt) < FNL_GRIB2_CUTOVER else ('grib2', 'grib1')
    return [(fname, [fnl_url(valid_dt, f) for f in fmts])]


def cfsr_tar_span(dt):
    """(start_date, end_date) of the 5-day CFSR tar covering `dt`: 01-05/06-10/11-15/16-20/
    21-25/26-<last day of month> -- VERIFIED spans (1990-01: ...26-19900131)."""
    d = _as_date(dt)
    day = d.day
    if day <= 5:
        start, end = 1, 5
    elif day <= 10:
        start, end = 6, 10
    elif day <= 15:
        start, end = 11, 15
    elif day <= 20:
        start, end = 16, 20
    elif day <= 25:
        start, end = 21, 25
    else:
        start, end = 26, calendar.monthrange(d.year, d.month)[1]
    return (datetime.date(d.year, d.month, start), datetime.date(d.year, d.month, end))


def cfsr_tar_urls(dt, host=None):
    """(pgbh06_url, flxf06_url) for the 5-day tar span covering `dt` -- both streams are linked
    together into ONE ungrib pass (Vtable.CFSR)."""
    d = _as_date(dt)
    start, end = cfsr_tar_span(d)
    span = '{}-{}'.format(start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))
    base = 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d093000/{}'.format(d.year)
    return ('{}/pgbh06.gdas.{}.tar'.format(base, span), '{}/flxf06.gdas.{}.tar'.format(base, span))


def cfsr_wps_entry(valid_dt, host=None):
    pgb_url, flx_url = cfsr_tar_urls(valid_dt, host=host)
    start, end = cfsr_tar_span(valid_dt)
    span = '{}-{}'.format(start.strftime('%Y%m%d'), end.strftime('%Y%m%d'))
    return [('pgbh06.gdas.{}.tar'.format(span), [pgb_url]),
            ('flxf06.gdas.{}.tar'.format(span), [flx_url])]


SRC_NOMADS_WPS = ForcingSource('ncep', 'gfs_nomads', 'gfs.0p25', 0.25, 'grib2', 3, 'forecast',
                               'Vtable.GFS', 'GFS', None, None, gfs_nomads_wps_entry)
SRC_AWS_WPS = ForcingSource('noaa', 'gfs_aws', 'gfs.0p25', 0.25, 'grib2', 3, 'forecast',
                            'Vtable.GFS', 'GFS', AWS_CUTOVER, None, gfs_aws_wps_entry)
SRC_RDA_D084001 = ForcingSource('ncar-rda', 'd084001', 'gfs.0p25', 0.25, 'grib2', 3, 'forecast',
                                'Vtable.GFS', 'GFS', RDA_CUTOVER, None, gfs_rda_wps_entry)
# NCEI gfsanl: only _000 carries the full 3-D analysis; _003/_006 hold ~8 accumulation messages
# (verified 2005-08-27 12Z: f000 286 messages, f003 8) -> 6 h cadence, f000 only.
# T13 (2026-09-24, V5 sc26p2): ncei_g3/g4 and FNL use Vtable.GFS_PRMSL, not Vtable.GFS --
# real.exe FATAL'd 'ERROR in psfc: flag_psfc=1, flag_soilhgt=1, flag_slp=0' on a 2005 ncei_g3
# GRIB1 file. Stock Vtable.GFS's active PMSL line maps from MSLET (GRIB1 param 130 / GRIB2
# 0-3-192); the 2005 file has only PRMSL (indicatorOfParameter 2) -- Vtable.GFS's own comment
# says exactly this ("older GFS files carry only PRMSL") and gives the commented-out PRMSL line
# to use instead. d084001/aws/nomads (current-era GFS, always has MSLET) keep Vtable.GFS. FNL
# uses PRMSL too: the probe cannot know in advance whether a given FNL file has MSLET, and
# PRMSL is always present in GFS/GDAS pressure-grib files per Vtable.GFS's own comment, so it is
# the safe universal choice for FNL specifically. See
# coldstart_sample.generate_vtable_gfs_prmsl (the transform that produces Vtable.GFS_PRMSL from
# a real Vtable.GFS at stage time -- this constant just names it).
SRC_NCEI_G4 = ForcingSource('ncei', 'ncei_g4', 'gfsanl_4', 0.5, 'grib2', 6, 'analysis',
                            'Vtable.GFS_PRMSL', 'GFS', NCEI_G4_START, None, ncei_g4_wps_entry)
SRC_NCEI_G3 = ForcingSource('ncei', 'ncei_g3', 'gfsanl_3', 1.0, 'grib1', 6, 'analysis',
                            'Vtable.GFS_PRMSL', 'GFS', NCEI_G3_START, None, ncei_g3_wps_entry)
SRC_FNL = ForcingSource('ncar-rda', 'd083002', 'fnl', 1.0, 'grib1', 6, 'analysis',
                        'Vtable.GFS_PRMSL', 'FNL', FNL_START, None, fnl_wps_entry)
SRC_CFSR = ForcingSource('ncar-rda', 'd093000', 'cfsr', 0.5, 'grib2-tar', 6, 'reanalysis',
                         'Vtable.CFSR', 'CFSR', CFSR_START, CFSR_END, cfsr_wps_entry)


def resolve_forcing_chain(valid_dt, now=None, grib_dir=None):
    """Ordered [ForcingSource, ...] candidates for `valid_dt`, WPS/ungrib purpose. Date BANDS
    (newest-wins, mutually exclusive -- NOT cumulative: a 1990 date gets [cfsr] only, not
    [cfsr, fnl, ncei_g3, ...]) matching the coordinator's verified fallback design:
      >=2021-01-01: [aws, d084001, fnl]          >=2015-01-15: [d084001, fnl]
      >=2007-01-01: [ncei_g4, fnl]               >=2004-03-02: [ncei_g3, fnl]
      >=1999-07-30: [fnl]                        1979-01-01..2010-12-31: [cfsr]
      older: [] -- unsupported, unless --grib-dir (existing local path, unchanged).
    A 'nomads' entry is prepended whenever valid_dt is within NOMADS_WINDOW_DAYS of `now`
    (same rule as select_source), same as today. FNL is appended as the LAST fallback of the two
    newest bands too (2026-09-24: d083002's own THREDDS catalog root lists
    fnl_20210101_00_00.grib2 and continuing year directories -- it runs to the present, not just
    back to 1999-07-30, so it is a legitimate last-resort alternative to AWS/RDA GFS 0.25deg, not
    only the fallback once those stop existing)."""
    now = now or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    d = _as_date(valid_dt)
    age_days = (now - valid_dt).days
    recent = -1 <= age_days <= NOMADS_WINDOW_DAYS

    if d >= AWS_CUTOVER:
        chain = ([SRC_NOMADS_WPS] if recent else []) + [SRC_AWS_WPS, SRC_RDA_D084001, SRC_FNL]
    elif d >= RDA_CUTOVER:
        chain = ([SRC_NOMADS_WPS] if recent else []) + [SRC_RDA_D084001, SRC_FNL]
    elif d >= NCEI_G4_START:
        chain = [SRC_NCEI_G4, SRC_FNL]
    elif d >= NCEI_G3_START:
        chain = [SRC_NCEI_G3, SRC_FNL]
    elif d >= FNL_START:
        chain = [SRC_FNL]
    elif CFSR_START <= d <= CFSR_END:
        chain = [SRC_CFSR]
    else:
        chain = []

    if not chain and grib_dir:
        chain = [ForcingSource('local', 'local', 'local', None, 'grib', None, 'local', None,
                               'GFS', None, None, None)]
    return chain


# T4a (2026-09-24): three concurrent fetch processes against RDA THREDDS made d084001 probes
# fail (matthew 2016-10-06, sc26p1 72h span, conus12 12h span) -> fetch fell back to a coarser
# source, or reported nothing found -- re-probing the SAME URLs minutes later returned 200 for
# all. The probe was treating a transient failure (server momentarily overloaded/timing out) the
# same as a genuinely missing file. Retry backoff below; only a 404 (or 403, an unambiguous
# "this object doesn't exist" response) is treated as missing WITHOUT retrying.
PROBE_RETRY_BACKOFFS_S = (2, 5, 15)   # 4 total attempts: 1 initial + 3 retries


def _is_transient_probe_status(status):
    """True for a status worth retrying (timeout/connection error -> check_url's own status=
    None; HTTP 429 too-many-requests; any 5xx server error) -- False for a definitive "not
    found" (404, 403) or an outright unexpected status, which is reported as missing at once."""
    if status is None:
        return True
    if status == 429:
        return True
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return False


def probe_url_with_retry(url, backoffs=PROBE_RETRY_BACKOFFS_S, sleep_fn=None):
    """check_url(url), retrying on a TRANSIENT failure (see _is_transient_probe_status) up to
    len(backoffs) additional times with the given backoff delays (default 2s, 5s, 15s -> 4 total
    attempts). Returns (final_result_dict, attempts) -- attempts records EVERY probe, not just
    the last, as [{'status', 'detail'}, ...]. `sleep_fn` is injectable so tests never actually
    sleep through a real backoff."""
    sleep_fn = sleep_fn or time.sleep
    attempts = []
    result = None
    for i in range(len(backoffs) + 1):
        result = check_url(url)
        attempts.append({'status': result.get('status'), 'detail': result.get('detail')})
        if result['exists']:
            return result, attempts
        if not _is_transient_probe_status(result.get('status')):
            break
        if i < len(backoffs):
            sleep_fn(backoffs[i])
    return result, attempts


def probe_forcing_entries(entries, sleep_fn=None):
    """entries: [(filename, [url, ...]), ...] (one ForcingSource.urls_for() call's result, or
    several concatenated). Probes every URL with probe_url_with_retry (HTTP HEAD, falling back
    to a 1-byte ranged GET, retried on a transient failure -- see that function; tests mock
    check_url/_http_head/_http_get_range_status directly, never touch the network, and pass a
    no-op sleep_fn to avoid actually sleeping through a retry backoff in the test suite).
    PROBES RUN SEQUENTIALLY -- no thread pool (T4c, 2026-09-24): concurrent probing against the
    same upstream is what caused the transient failures T4a retries for in the first place, so
    adding concurrency back here would defeat the fix. Returns (all_found, [{'filename',
    'chosen_url', 'checked': [...]}, ...]); all_found is True iff EVERY entry has at least one
    existing URL among its alternatives. Each `checked` entry now also carries 'attempts' (every
    probe of that one URL, per probe_url_with_retry)."""
    results = []
    all_found = True
    for filename, urls in entries:
        chosen = None
        checked = []
        for u in urls:
            r, attempts = probe_url_with_retry(u, sleep_fn=sleep_fn)
            checked.append({'url': u, 'exists': r['exists'], 'status': r['status'],
                            'attempts': attempts})
            if r['exists'] and chosen is None:
                chosen = u
        if chosen is None:
            all_found = False
        results.append({'filename': filename, 'chosen_url': chosen, 'checked': checked})
    return all_found, results


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
        # WPS purpose + aws/nomads (S3, 2026-09-24): use the idx-driven selection
        # (fetch_wps_messages_idx_driven / select_wps_messages / validate_wps_selection) instead
        # of the static WPS_FIELD_SPECS lookup below -- that lookup silently matched nothing
        # against a real file's actual pressure/soil-layer set (severe2023:
        # NUM_METGRID_SOIL_LEVELS ended up 0) and never failed loudly. rda/local/state-purpose
        # entries are untouched -- they fall through to the unchanged per-message loop.
        is_wps_purpose = bool(entry['messages']) and 'cfgrib_short' not in entry['messages'][0]
        if is_wps_purpose and entry['source'] in ('aws', 'nomads'):
            vt_label = entry['valid_time'].strftime('%Y-%m-%d_%H')
            try:
                selected = fetch_wps_messages_idx_driven(
                    entry['cycle'], entry['fhour'], entry['source'], cache_dir, host=host)
                for sel, path in selected:
                    field = '{}_{}'.format(sel['var'], sel['level']).replace(' ', '_')
                    results.append({'valid_time': entry['valid_time'], 'field': field,
                                    'status': 'ok', 'path': path})
                print('OK   {} wps: {} messages selected from live .idx (cache={})'.format(
                    vt_label, len(selected), cache_dir))
            except Exception as exc:
                results.append({'valid_time': entry['valid_time'], 'field': 'wps',
                                'status': 'error', 'detail': str(exc)})
                print('FAIL {} wps -- {}'.format(vt_label, exc))
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

#!/usr/bin/env python3
"""tools/grib_state.py -- WRF-run-free domain-aggregate physical-state trajectory, built from
global analysis/forecast GRIB data (GFS via NCAR RDA THREDDS or AWS S3, or a local GRIB directory
e.g. NAM) instead of an actual WRF run. Produces the SAME feature set tools/state_timeline.py
computes from WRF output, so a cost model trained on WRF-run state trajectories can be APPLIED to
a prospective case that has never been simulated.

Feature set (names must match tools/state_timeline.py exactly):
  precip_frac    fraction of grid columns with precip rate > 1 mm/h
  precip_mean    domain-mean precip rate, mm/h
  cloud_frac     fraction of columns with TCDC (total cloud cover) > 50%, OR with OLR (outgoing
                 longwave radiation, proxied by ULWRF top-of-atmosphere) < 200 W/m^2 when that
                 field is the one available -- which one was used for a given row is recorded in
                 the `cloud_frac_source` CSV column ('tcdc' or 'olr')
  sunlit_frac    analytic, from lat/lon/UTC alone via wtune.solar.cos_zenith (no GRIB field)
  pblh_mean/pblh_p90   domain mean / 90th percentile of HPBL (boundary-layer height), m
  wind10_max     domain max of sqrt(UGRD_10m^2 + VGRD_10m^2), m/s
  hfx_mean       domain mean of SHTFL (surface sensible heat flux), W/m^2

Data source selection, URL construction, availability checking, and download are all owned by
tools/metdata.py (see its own module docstring for the nomads/aws/rda/local date rules and the
--rda-host override) -- this module imports it (`import metdata as md`) rather than re-deriving
any of that; `--source` here forwards straight to metdata.select_source.

What is left here is the regrid and the feature functions tools/coldstart_sample.py and
tools/case_rebuild.py import. The `trajectory` command, its CSV and the per-patch imbalance helpers
are in archive/unused-2026-10/src/tools/grib_state_legacy.py.

NEEDS THE CLUSTER TO VERIFY (cannot be checked offline -- see tools/test_grib_state.py and
tools/test_metdata.py for what IS covered offline): the exact GRIB `shortName`/`typeOfLevel`
strings cfgrib assigns to TCDC/ULWRF/HPBL/SHTFL on the real GFS pgrb2.0p25 product
(metdata.STATE_FIELD_SPECS is the best-documented guess), and GFS APCP's actual per-message
accumulation window beyond the f000->f003 substitution (see metdata.resolve_apcp_request).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # src/
if not __package__:                    # run by path: this directory is not an import root
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wtune import solar as wsolar           # repo root: cos_zenith/declination_deg/day_of_year/domain_bbox

# ---------------------------------------------------------------------------
# Data source / URL construction / availability / download / GRIB decode: all owned by
# tools/metdata.py now -- this module only regrids onto the WRF target grid, computes features,
# and assembles the trajectory. See metdata.py's own module docstring for source-selection rules,
# URL formats, and the NEEDS-CLUSTER-VERIFICATION caveats on message identity / APCP windows.
# ---------------------------------------------------------------------------

from wtune.tools import metdata as md    # tools/metdata.py: select_source/gfs_cycle_for/STATE_FIELD_SPECS/
                         # fetch_and_decode_state_field/crop_bbox/resolve_apcp_request/load_case/
                         # resolve_times


# ---------------------------------------------------------------------------
# nearest-neighbour regrid (pure numpy, offline-testable)
# ---------------------------------------------------------------------------

def regrid_nearest_regular(src_lats_1d, src_lons_1d, src_values_2d, tgt_lats_2d, tgt_lons_2d):
    """Nearest-neighbour regrid of a REGULAR (equally spaced) source lat/lon grid onto an
    arbitrary target lat/lon mesh (e.g. WRF's XLAT/XLONG, or the analytic bbox mesh). O(1) index
    lookup per target point -- no KDTree dependency: index = round((coord - coord[0]) /
    spacing), clipped to bounds. Valid because the source spacing is uniform (GFS 0.25deg)."""
    src_lats_1d = np.asarray(src_lats_1d, dtype=float)
    src_lons_1d = np.asarray(src_lons_1d, dtype=float)
    tgt_lats_2d = np.asarray(tgt_lats_2d, dtype=float)
    tgt_lons_2d = np.asarray(tgt_lons_2d, dtype=float)
    if src_lons_1d.size and src_lons_1d.max() > 180.0:
        tgt_lons_2d = md.normalize_lon_0_360(tgt_lons_2d)
    dlat = src_lats_1d[1] - src_lats_1d[0] if src_lats_1d.size > 1 else 1.0
    dlon = src_lons_1d[1] - src_lons_1d[0] if src_lons_1d.size > 1 else 1.0
    lat_idx = np.clip(np.rint((tgt_lats_2d - src_lats_1d[0]) / dlat).astype(int),
                       0, src_lats_1d.size - 1)
    lon_idx = np.clip(np.rint((tgt_lons_2d - src_lons_1d[0]) / dlon).astype(int),
                       0, src_lons_1d.size - 1)
    return src_values_2d[lat_idx, lon_idx]


# ---------------------------------------------------------------------------
# Feature math (pure numpy, offline-testable)
# ---------------------------------------------------------------------------

PRECIP_THRESHOLD_MMH = 1.0


def precip_frac(rate_mmh):
    return float(np.nanmean(np.asarray(rate_mmh, dtype=float) > PRECIP_THRESHOLD_MMH))


def precip_mean(rate_mmh):
    return float(np.nanmean(np.asarray(rate_mmh, dtype=float)))


def precip_rate_from_apcp_window(apcp_mm, window_hours):
    """precip rate (mm/h) when the fetched APCP message's OWN accumulation window already equals
    `window_hours` (e.g. GFS f006's APCP is documented as a 3-6h bucket, not 0-6h cumulative) --
    no differencing needed. Preferred over `precip_rate_from_apcp_diff` once the real
    accumulation window is confirmed."""
    if window_hours <= 0:
        raise ValueError('window_hours must be > 0')
    return np.asarray(apcp_mm, dtype=float) / window_hours


def cos_zenith_grid(lat2d, lon2d, valid_dt_utc):
    """cos(solar zenith) at every grid column, at `valid_dt_utc` (timezone-naive UTC
    datetime.datetime). Reuses wtune.solar.cos_zenith point-wise (np.vectorize) so this is
    EXACTLY the WRF-side terminator geometry, never a re-derivation of it. Factored out of
    `sunlit_frac` so the same per-column indicator also feeds the sunlit_frac patch-imbalance
    columns (see patch_stats / trajectory)."""
    lat2d = np.asarray(lat2d, dtype=float)
    lon2d = np.asarray(lon2d, dtype=float)
    doy = wsolar.day_of_year(valid_dt_utc.month, valid_dt_utc.day)
    decl = wsolar.declination_deg(doy)
    utc_hours = valid_dt_utc.hour + valid_dt_utc.minute / 60.0 + valid_dt_utc.second / 3600.0
    return np.vectorize(wsolar.cos_zenith)(lat2d, lon2d, utc_hours, decl)


def sunlit_frac(lat2d, lon2d, valid_dt_utc):
    """Fraction of grid columns with cos(solar zenith) > 0 (sun above the horizon) at
    `valid_dt_utc`. NOTE vs. tools/state_timeline.py's own sunlit_frac: this is a purely
    ANALYTIC definition (lat/lon/UTC only, no radiation field), per this module's own task spec
    -- the WRF side instead thresholds actual simulated SWDOWN (falling back to COSZEN when its
    Registry has it). The two are expected to agree closely but not bit-for-bit; flagged as a
    deliberate, spec-directed difference, not a bug."""
    cz = cos_zenith_grid(lat2d, lon2d, valid_dt_utc)
    return float(np.nanmean(cz > 0))


def pblh_mean(pblh_m):
    return float(np.nanmean(np.asarray(pblh_m, dtype=float)))


def pblh_p90(pblh_m):
    return float(np.nanpercentile(np.asarray(pblh_m, dtype=float), 90))


def wind10_max(u10, v10):
    u = np.asarray(u10, dtype=float)
    v = np.asarray(v10, dtype=float)
    return float(np.nanmax(np.sqrt(u * u + v * v)))


def hfx_mean(shtfl_wm2):
    return float(np.nanmean(np.asarray(shtfl_wm2, dtype=float)))


# ---------------------------------------------------------------------------
# Case / namelist loading + target grid (reuses wtune/cli.py / wtune/solar.py conventions)
# ---------------------------------------------------------------------------

# load_case/resolve_times are now owned by tools/metdata.py (find/check/fetch need the identical
# case-dir -> namelist -> times resolution) -- reused here as-is rather than duplicated.
load_case = md.load_case
resolve_times = md.resolve_times


def target_grid(case, wrfinput_path):
    """(lat2d, lon2d, mode) -- the WRF-grid regrid target. mode 'wrfinput' reads the exact
    XLAT/XLONG (needs netCDF4); mode 'bbox' falls back to an analytic uniform mesh from
    wtune.solar.domain_bbox (same flat-earth approximation wtune/case.py already relies on when
    no wrfinput exists)."""
    if wrfinput_path and os.path.exists(wrfinput_path):
        try:
            import netCDF4
            nc = netCDF4.Dataset(wrfinput_path)
            try:
                lat2d = np.array(nc.variables['XLAT'][0], dtype=float)
                lon2d = np.array(nc.variables['XLONG'][0], dtype=float)
                return lat2d, lon2d, 'wrfinput'
            finally:
                nc.close()
        except Exception:
            pass  # netCDF4 missing or unreadable -- fall through to bbox mode
    bbox = wsolar.domain_bbox(case)
    if bbox is None:
        raise RuntimeError('no wrfinput and no domain center lat/lon in namelist -- cannot '
                            'build a target grid')
    lat0, lat1, lon0, lon1 = bbox
    nx, ny = case.get('nx') or 100, case.get('ny') or 100
    lats = np.linspace(lat0, lat1, ny)
    lons = np.linspace(lon0, lon1, nx)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lat2d, lon2d, 'bbox'

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
  sunlit_frac    analytic, from lat/lon/UTC alone via wtune_solar.cos_zenith (no GRIB field)
  pblh_mean/pblh_p90   domain mean / 90th percentile of HPBL (boundary-layer height), m
  wind10_max     domain max of sqrt(UGRD_10m^2 + VGRD_10m^2), m/s
  hfx_mean       domain mean of SHTFL (surface sensible heat flux), W/m^2

Data source selection, URL construction, availability checking, and download are all owned by
tools/metdata.py (see its own module docstring for the nomads/aws/rda/local date rules and the
--rda-host override) -- this module imports it (`import metdata as md`) rather than re-deriving
any of that; `--source` here forwards straight to metdata.select_source.

CLI:
  python3 grib_state.py trajectory --case-dir D [--namelist P] [--wrfinput P]
      [--start YYYY-MM-DD_HH] [--hours N] [--every 3] [--source auto|nomads|aws|rda|local]
      [--grib-dir G] [--cache-dir C] [--out traj.csv] [--decomp NX NY]

NEEDS THE CLUSTER TO VERIFY (cannot be checked offline -- see tools/test_grib_state.py and
tools/test_metdata.py for what IS covered offline): the exact GRIB `shortName`/`typeOfLevel`
strings cfgrib assigns to TCDC/ULWRF/HPBL/SHTFL on the real GFS pgrb2.0p25 product
(metdata.STATE_FIELD_SPECS is the best-documented guess), and GFS APCP's actual per-message
accumulation window beyond the f000->f003 substitution (see metdata.resolve_apcp_request).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_PREDICTOR_DIR = os.path.join(_REPO_ROOT, 'predictor')
for _p in (_HERE, _REPO_ROOT, _PREDICTOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wtune_solar as wsolar           # repo root: cos_zenith/declination_deg/day_of_year/domain_bbox
import decomp_space as ds              # predictor/decomp_space.py: rsl_patch_extents (exact RSL rule)

# ---------------------------------------------------------------------------
# Data source / URL construction / availability / download / GRIB decode: all owned by
# tools/metdata.py now -- this module only regrids onto the WRF target grid, computes features,
# and assembles the trajectory. See metdata.py's own module docstring for source-selection rules,
# URL formats, and the NEEDS-CLUSTER-VERIFICATION caveats on message identity / APCP windows.
# ---------------------------------------------------------------------------

import metdata as md    # tools/metdata.py: select_source/gfs_cycle_for/STATE_FIELD_SPECS/
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
CLOUD_TCDC_THRESHOLD_PCT = 50.0
CLOUD_OLR_THRESHOLD_WM2 = 200.0


def precip_frac(rate_mmh):
    return float(np.nanmean(np.asarray(rate_mmh, dtype=float) > PRECIP_THRESHOLD_MMH))


def precip_mean(rate_mmh):
    return float(np.nanmean(np.asarray(rate_mmh, dtype=float)))


def precip_rate_from_apcp_diff(apcp_a_mm, apcp_b_mm, hours_delta):
    """precip rate (mm/h) from two CUMULATIVE-since-cycle-init APCP snapshots `hours_delta`
    apart -- 'APCP difference / hours' per the task spec, for the RDA path's f000/f003/f006
    analysis+short-forecast triples. CAVEAT (needs cluster verification): GFS's own APCP field is
    NOT always cumulative since init -- its accumulation bucket resets periodically (e.g. 0-3h,
    then 3-6h, not 0-6h) on the operational pgrb2 product. Confirm the real window from a
    downloaded message's `fcst` idx field before trusting this over
    `precip_rate_from_apcp_window`."""
    if hours_delta <= 0:
        raise ValueError('hours_delta must be > 0')
    a = np.asarray(apcp_a_mm, dtype=float)
    b = np.asarray(apcp_b_mm, dtype=float)
    return (b - a) / hours_delta


def precip_rate_from_apcp_window(apcp_mm, window_hours):
    """precip rate (mm/h) when the fetched APCP message's OWN accumulation window already equals
    `window_hours` (e.g. GFS f006's APCP is documented as a 3-6h bucket, not 0-6h cumulative) --
    no differencing needed. Preferred over `precip_rate_from_apcp_diff` once the real
    accumulation window is confirmed."""
    if window_hours <= 0:
        raise ValueError('window_hours must be > 0')
    return np.asarray(apcp_mm, dtype=float) / window_hours


def cloud_frac_tcdc(tcdc_pct):
    return float(np.nanmean(np.asarray(tcdc_pct, dtype=float) > CLOUD_TCDC_THRESHOLD_PCT))


def cloud_frac_olr(olr_wm2):
    return float(np.nanmean(np.asarray(olr_wm2, dtype=float) < CLOUD_OLR_THRESHOLD_WM2))


def cos_zenith_grid(lat2d, lon2d, valid_dt_utc):
    """cos(solar zenith) at every grid column, at `valid_dt_utc` (timezone-naive UTC
    datetime.datetime). Reuses wtune_solar.cos_zenith point-wise (np.vectorize) so this is
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


def _has_finite(arr):
    return bool(np.any(np.isfinite(np.asarray(arr, dtype=float))))


# ---------------------------------------------------------------------------
# Imbalance-feature helpers: per-MPI-patch aggregates under an (nproc_x, nproc_y) decomposition
# ---------------------------------------------------------------------------

def patch_slices(extent, nproc):
    """Concrete 0-indexed [start, end) boundaries for each of `nproc` ranks along one dimension
    of size `extent`, under RSL-lite's remainder-at-both-ends rule -- built from
    predictor/decomp_space.rsl_patch_width_for_rank, the SAME module's own per-rank-width helper
    for the rule rsl_patch_extents computes the (min,max) ENVELOPE of, so the two can never
    disagree by construction."""
    slices = []
    start = 0
    for r in range(nproc):
        w = ds.rsl_patch_width_for_rank(extent, nproc, r)
        slices.append((start, start + w))
        start += w
    return slices


def patch_stats(field2d, nproc_x, nproc_y):
    """Per-patch mean of `field2d` (shape (ny, nx)) under an RSL-lite (nproc_x, nproc_y) MPI
    decomposition, plus the exact legality envelope from decomp_space.rsl_patch_extents (the
    same floor-division rule WRF's own startup gate checks -- share/module_check_a_mundo.F).
    `patch_mean` is the UNWEIGHTED mean over the nproc_x*nproc_y patch means (every patch counted
    once regardless of size) -- matches tools/state_timeline.py's max_mean_over_patches
    (statistics.mean over its patch-value dict), so `patch_max`/`patch_mean` here and
    `imbalance_ratio(patch_max, patch_mean)` reproduce that module's *_patch_max/*_patch_mean/
    *_imbalance columns exactly when fed the same 0/1 indicator field."""
    field2d = np.asarray(field2d, dtype=float)
    ny, nx = field2d.shape
    min_w, max_w = ds.rsl_patch_extents(nx, nproc_x)
    min_h, max_h = ds.rsl_patch_extents(ny, nproc_y)
    xs = patch_slices(nx, nproc_x)
    ys = patch_slices(ny, nproc_y)
    means = np.full((nproc_y, nproc_x), np.nan)
    for j, (y0, y1) in enumerate(ys):
        for i, (x0, x1) in enumerate(xs):
            means[j, i] = np.nanmean(field2d[y0:y1, x0:x1])
    return {
        'patch_means': means,
        'patch_max': float(np.nanmax(means)),
        'patch_min': float(np.nanmin(means)),
        'patch_mean': float(np.nanmean(means)),
        'patch_spread': float(np.nanmax(means) - np.nanmin(means)),
        'min_patch_x': min_w, 'max_patch_x': max_w,
        'min_patch_y': min_h, 'max_patch_y': max_h,
        'legal': bool(min_w >= ds.MIN_PATCH and min_h >= ds.MIN_PATCH),
    }


def imbalance_ratio(patch_max, patch_mean):
    """max-over-patches / mean-over-patches -- None when either input is None/0. Matches
    tools/state_timeline.py's imbalance_ratio exactly (same formula, same None-guard on a
    domain-wide-zero field where the ratio is undefined)."""
    if patch_max is None or patch_mean is None or patch_mean == 0:
        return None
    return patch_max / patch_mean


# ---------------------------------------------------------------------------
# Case / namelist loading + target grid (reuses wtune.py / wtune_solar.py conventions)
# ---------------------------------------------------------------------------

# load_case/resolve_times are now owned by tools/metdata.py (find/check/fetch need the identical
# case-dir -> namelist -> times resolution) -- reused here as-is rather than duplicated.
load_case = md.load_case
resolve_times = md.resolve_times


def target_grid(case, wrfinput_path):
    """(lat2d, lon2d, mode) -- the WRF-grid regrid target. mode 'wrfinput' reads the exact
    XLAT/XLONG (needs netCDF4); mode 'bbox' falls back to an analytic uniform mesh from
    wtune_solar.domain_bbox (same flat-earth approximation wtune_case.py already relies on when
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


# ---------------------------------------------------------------------------
# Trajectory orchestration + CLI
# ---------------------------------------------------------------------------

CSV_FIELDS = ['t_hours', 'valid_time', 'precip_frac', 'precip_mean', 'cloud_frac',
              'cloud_frac_source', 'sunlit_frac', 'pblh_mean', 'pblh_p90', 'wind10_max',
              'hfx_mean', 'source', 'n_columns']
# The first 6 names/semantics MUST match tools/state_timeline.py's own STATE_COLUMNS imbalance
# block exactly (precip_frac_patch_max/patch_mean/imbalance, sunlit_frac_patch_max/patch_mean/
# imbalance -- max/mean-over-patches of the SAME 0/1 per-column indicator that produces
# precip_frac/sunlit_frac, ratio = max/mean) so a cost model sees identical columns from either
# producer. decomp_* is this module's own additive diagnostic (RSL-lite legality envelope for the
# requested (nproc_x, nproc_y); state_timeline.py has no equivalent column).
DECOMP_EXTRA_FIELDS = ['precip_frac_patch_max', 'precip_frac_patch_mean', 'precip_frac_imbalance',
                        'sunlit_frac_patch_max', 'sunlit_frac_patch_mean', 'sunlit_frac_imbalance',
                        'decomp_nproc_x', 'decomp_nproc_y', 'decomp_min_patch_x',
                        'decomp_max_patch_x', 'decomp_min_patch_y', 'decomp_max_patch_y',
                        'decomp_legal']


def _fetch_regridded(field_key, cycle_dt, fhour, source, grib_dir, cache_dir, lat2d, lon2d):
    slat, slon, sval = md.fetch_and_decode_state_field(field_key, cycle_dt, fhour, source,
                                                         cache_dir, grib_dir=grib_dir)
    slat, slon, sval = md.crop_bbox(slat, slon, sval, float(np.min(lat2d)), float(np.max(lat2d)),
                                     float(np.min(lon2d)), float(np.max(lon2d)))
    return regrid_nearest_regular(slat, slon, sval, lat2d, lon2d)


def trajectory(case_dir, namelist=None, wrfinput=None, start=None, hours=None, every=3.0,
               source='auto', grib_dir=None, cache_dir=None, decomp=None):
    """Compute the state trajectory rows for one case. Returns (rows, fieldnames) ready for
    write_csv. Network/GRIB-decoding calls happen here (via tools/metdata.py) -- not covered by
    the offline test suite; everything this function calls down to (URL/idx/crop/regrid/feature
    math) is."""
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'grib_state_cache')
    case, _nl_path, wi_path = load_case(case_dir, namelist, wrfinput)
    lat2d, lon2d, _grid_mode = target_grid(case, wi_path)
    times = resolve_times(case, start, hours, every)
    src_choice, _reason = md.select_source(times[0], override=source, grib_dir=grib_dir)
    if src_choice is None:
        raise RuntimeError('no GRIB source available for {}: {}'.format(times[0], _reason))
    if src_choice == 'local' and not grib_dir:
        raise RuntimeError('--grib-dir is required for --source local')
    t0 = times[0]
    rows = []
    for t in times:
        cycle_dt, fhour = md.gfs_cycle_for(t)
        row = {'t_hours': (t - t0).total_seconds() / 3600.0,
               'valid_time': t.strftime('%Y-%m-%d_%H:%M:%S'),
               'source': src_choice, 'n_columns': int(lat2d.size)}

        fields = {}
        for key in ('tcdc', 'ulwrf', 'hpbl', 'u10', 'v10', 'shtfl'):
            # ulwrf/shtfl (like apcp) are accumulation/time-average fields, CONFIRMED absent from
            # GFS's f000 analysis message set (see metdata.FLUX_FIELDS_NO_F000) -- substitute f003
            # of the same cycle for those two exactly as apcp does below; tcdc/hpbl/u10/v10 are
            # instantaneous ('anl') at f000 and pass through unchanged.
            eff_cycle, eff_fhour, _warn = md.resolve_field_request(key, cycle_dt, fhour)
            fields[key] = _fetch_regridded(key, eff_cycle, eff_fhour, src_choice, grib_dir,
                                            cache_dir, lat2d, lon2d)
        apcp_cycle, apcp_fhour, apcp_window, _apcp_warn = md.resolve_apcp_request(cycle_dt, fhour)
        apcp = _fetch_regridded('apcp', apcp_cycle, apcp_fhour, src_choice, grib_dir, cache_dir,
                                 lat2d, lon2d)
        precip_rate = precip_rate_from_apcp_window(apcp, apcp_window)

        row['precip_frac'] = precip_frac(precip_rate)
        row['precip_mean'] = precip_mean(precip_rate)
        if _has_finite(fields['ulwrf']):
            row['cloud_frac'] = cloud_frac_olr(fields['ulwrf'])
            row['cloud_frac_source'] = 'olr'
        else:
            row['cloud_frac'] = cloud_frac_tcdc(fields['tcdc'])
            row['cloud_frac_source'] = 'tcdc'
        cz_grid = cos_zenith_grid(lat2d, lon2d, t)
        row['sunlit_frac'] = float(np.nanmean(cz_grid > 0))
        row['pblh_mean'] = pblh_mean(fields['hpbl'])
        row['pblh_p90'] = pblh_p90(fields['hpbl'])
        row['wind10_max'] = wind10_max(fields['u10'], fields['v10'])
        row['hfx_mean'] = hfx_mean(fields['shtfl'])

        if decomp:
            nproc_x, nproc_y = decomp
            precip_ind = (precip_rate > PRECIP_THRESHOLD_MMH).astype(float)
            sunlit_ind = (cz_grid > 0).astype(float)
            ps = patch_stats(precip_ind, nproc_x, nproc_y)
            ss = patch_stats(sunlit_ind, nproc_x, nproc_y)
            row['precip_frac_patch_max'] = ps['patch_max']
            row['precip_frac_patch_mean'] = ps['patch_mean']
            row['precip_frac_imbalance'] = imbalance_ratio(ps['patch_max'], ps['patch_mean'])
            row['sunlit_frac_patch_max'] = ss['patch_max']
            row['sunlit_frac_patch_mean'] = ss['patch_mean']
            row['sunlit_frac_imbalance'] = imbalance_ratio(ss['patch_max'], ss['patch_mean'])
            row['decomp_nproc_x'], row['decomp_nproc_y'] = nproc_x, nproc_y
            row['decomp_min_patch_x'], row['decomp_max_patch_x'] = ps['min_patch_x'], ps['max_patch_x']
            row['decomp_min_patch_y'], row['decomp_max_patch_y'] = ps['min_patch_y'], ps['max_patch_y']
            row['decomp_legal'] = ps['legal']

        rows.append(row)
    fieldnames = CSV_FIELDS + DECOMP_EXTRA_FIELDS if decomp else CSV_FIELDS
    return rows, fieldnames


def write_csv(rows, fieldnames, out_path):
    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None):
    p = argparse.ArgumentParser(prog='grib_state.py',
                                 description='WRF-run-free physical-state trajectory from GRIB')
    sub = p.add_subparsers(dest='cmd', required=True)
    tp = sub.add_parser('trajectory', help='domain-aggregate state trajectory from GRIB data')
    tp.add_argument('--case-dir', required=True, help='directory with namelist.input (+ optional wrfinput_d0N)')
    tp.add_argument('--namelist', default=None)
    tp.add_argument('--wrfinput', default=None)
    tp.add_argument('--start', default=None, help='YYYY-MM-DD_HH (default: namelist start date)')
    tp.add_argument('--hours', type=float, default=None, help='default: namelist run_total_hours')
    tp.add_argument('--every', type=float, default=3.0, help='sampling interval, hours (default 3)')
    tp.add_argument('--source', default='auto', choices=['auto', 'nomads', 'aws', 'rda', 'local'])
    tp.add_argument('--grib-dir', default=None, help='local GRIB directory (required for --source local)')
    tp.add_argument('--cache-dir', default=os.path.join(tempfile.gettempdir(), 'grib_state_cache'))
    tp.add_argument('--out', default='traj.csv')
    tp.add_argument('--decomp', nargs=2, type=int, default=None, metavar=('NX', 'NY'))
    args = p.parse_args(argv)

    rows, fieldnames = trajectory(
        args.case_dir, args.namelist, args.wrfinput, args.start, args.hours, args.every,
        args.source, args.grib_dir, args.cache_dir, tuple(args.decomp) if args.decomp else None)
    write_csv(rows, fieldnames, args.out)
    print('wrote {} ({} rows)'.format(args.out, len(rows)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

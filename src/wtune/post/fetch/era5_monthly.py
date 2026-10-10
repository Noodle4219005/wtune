"""post/fetch/era5_monthly.py -- ERA5 monthly means of 2 m temperature (t2m), sea-surface
temperature (sst) and, on request, total precipitation (tp), cropped to the run's region, for the
climate recipe's annual-cycle and seasonal-bias figures.

Source choice (checked 2026-09-29 from the scc login node): no ANONYMOUS monthly-mean ERA5
product exists. The ARCO store (gs://gcp-public-data-arco-era5) holds hourly/6-hourly data only
(no monthly dataset under ar/ co/ raw/), NCAR's AWS mirror (s3://nsf-ncar-era5) lists no e5.moda*
objects, data.rda.ucar.edu (d633001) did not answer, and the Copernicus CDS monthly-means
dataset needs an API key. This module therefore aggregates the hourly ARCO store itself:

  * t2m, sst: mean over the 00/06/12/18 UTC analyses (--step-hours 6, the sampling of the WRF
    6-hourly output the recipe compares against; a 4-point diurnal sampling of a smooth diurnal
    cycle is within a few 0.1 K of the 24-point mean),
  * tp: sum of ALL hourly accumulations. ERA5 tp is a forecast accumulation over the preceding
    hour in metres of water; monthly mean rate = mean over hours of tp * 1000 (mm/h) * 24
    = mm/day. This needs every hour (8 760 chunk reads a year), so it is opt-in (--vars tp) and
    the recipe uses GPCP for precipitation instead (see gpcp_monthly.py: observation-based, no
    forecast-accumulation caveat).

Cost: one ARCO chunk is one global 0.25 deg field; a year of 6-hourly t2m+sst is ~2 920 chunk
reads (a few GB). Run it on the scc LOGIN node (compute nodes have no internet); it is
resumable per month (existing non-empty files are skipped) so an interruption loses at most
the current month.

Output: <project>/input/era5/monthly/era5_<var>_<YYYYMM>.nc (dims lat/lon, one field; variable
named as in --vars; t2m/sst in K, tp in mm/day; global attr n_samples) plus
<project>/input/obs/era5_monthly/provenance.json.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from wtune.post import sources

NAME = "era5_monthly"
HELP = "ERA5 monthly means (t2m, sst, tp) aggregated from ARCO hourly data -> <project>/input/era5/monthly/"
PROVENANCE = {
    "dataset": "ERA5 monthly means of t2m/sst(/tp), aggregated by wtune from the ARCO-ERA5 hourly store",
    "provider": "ECMWF / Copernicus Climate Change Service; ARCO copy on Google Cloud public datasets",
    "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    "license": "Copernicus licence (CC-BY 4.0 for ERA5)",
    "citation": "Hersbach, H., et al., 2020: The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., "
                "146, 1999-2049, doi:10.1002/qj.3803.",
    "coverage_time": "1940-01-01 to a few months before present (store time axis)",
    "coverage_space": "cropped to the run domain plus --margin degrees",
    "resolution": "0.25 deg, monthly mean of 6-hourly (t2m, sst) or hourly (tp) fields",
    "access": "public, anonymous (no credentials); needs internet (login node on scc)",
}
ARCO_NAMES = {"t2m": "2m_temperature", "sst": "sea_surface_temperature", "tp": "total_precipitation"}


def monthly_path(project, var, year, month):
    return os.path.join(project, "input", "era5", "monthly", "era5_%s_%04d%02d.nc" % (var, year, month))


def month_starts(start, end, skip_days=30, step_hours=6):
    """First-of-month datetimes of the calendar months entirely covered by [start, end] (the
    month's last sample, valid at the next month's 00 UTC minus one step, must exist) and that
    begin at or after start + skip_days (the spin-up exclusion the recipe applies too)."""
    cutoff = start + timedelta(days=skip_days)
    out = []
    y, m = start.year, start.month
    while True:
        ms = datetime(y, m, 1)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        nxt = datetime(ny, nm, 1)
        if nxt - timedelta(hours=step_hours) > end:
            break
        if ms >= cutoff and ms >= start.replace(day=1, hour=0, minute=0, second=0, microsecond=0):
            out.append(ms)
        y, m = ny, nm
    return out


def sample_times(ms, step_hours):
    ny, nm = (ms.year + 1, 1) if ms.month == 12 else (ms.year, ms.month + 1)
    nxt = datetime(ny, nm, 1)
    out, t = [], ms
    while t < nxt:
        out.append(t)
        t += timedelta(hours=step_hours)
    return out


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--vars", default="t2m,sst", help="comma list of t2m,sst,tp (default t2m,sst; "
                        "tp reads every hour of every month)")
    parser.add_argument("--skip-days", type=float, default=30.0, dest="skip_days",
                        help="do not fetch months starting before window start + this (spin-up; default 30)")
    parser.add_argument("--step-hours", type=int, default=6, dest="step_hours",
                        help="sampling interval of t2m/sst (default 6 = 00/06/12/18 UTC)")
    parser.add_argument("--jobs", type=int, default=8, help="concurrent chunk reads (default 8)")
    parser.add_argument("--margin", type=float, default=3.0, help="degrees around the bbox (default 3)")


def _write(path, var, lat, lon, field, units, n, ym):
    import netCDF4
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with netCDF4.Dataset(tmp, "w") as nc:
        nc.createDimension("lat", len(lat))
        nc.createDimension("lon", len(lon))
        for nm, arr, u in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            v = nc.createVariable(nm, "f8", (nm,))
            v[:] = arr
            v.units = u
        v = nc.createVariable(var, "f4", ("lat", "lon"), zlib=True, fill_value=float("nan"))
        v[:] = field
        v.units = units
        nc.month = ym
        nc.n_samples = n
        nc.source = PROVENANCE["url"]
    os.replace(tmp, path)


def _aggregate(ds, var, times, lat_idx, lon_idx, jobs=1):
    """NaN-aware mean over `times` of the cropped field (sst is NaN on land). The reads are
    network-latency bound (~1.5 s per 0.25 deg global chunk), so `jobs` threads fetch fields
    concurrently; the accumulation itself stays in the calling thread."""
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        return np.asarray(ds[ARCO_NAMES[var]].sel(time=np.datetime64(t)).isel(
            latitude=lat_idx, longitude=lon_idx).values, dtype="f8")

    acc = cnt = None
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        for a in ex.map(one, times):
            ok = np.isfinite(a)
            if acc is None:
                acc, cnt = np.zeros_like(a), np.zeros(a.shape)
            acc += np.where(ok, a, 0.0)
            cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


def run(args):
    import numpy as np
    from wtune.post.recipes import era5 as recipe_era5
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("era5_monthly: %s" % e)
        return []
    variables = [v for v in args.vars.split(",") if v]
    bad = [v for v in variables if v not in ARCO_NAMES]
    if bad:
        print("era5_monthly: unknown --vars %s (known: %s)" % (bad, ",".join(ARCO_NAMES)))
        return []
    months = month_starts(start, end, args.skip_days, args.step_hours)
    if not months:
        print("era5_monthly: no complete calendar month in %s..%s after skipping %g d of spin-up"
              % (start, end, args.skip_days))
        return []
    try:
        ds = recipe_era5.open_era5()
        t_last = ds["time"].values[-1]
    except recipe_era5.Era5UnavailableError as e:
        print("era5_monthly: no data -- %s" % e)
        return []
    era5_mod = recipe_era5._era5_mod()
    lat_idx, lon_idx, lon_sel = era5_mod.crop_indices(ds["latitude"].values, ds["longitude"].values,
                                                      bbox, args.margin)
    lat_sel = ds["latitude"].values[lat_idx]
    written, files = [], {}
    for ms in months:
        if np.datetime64(sample_times(ms, 1)[-1]) > t_last:
            print("era5_monthly: %s is beyond the store's last time %s, skipped" % (ms.strftime("%Y-%m"), t_last))
            continue
        for var in variables:
            path = monthly_path(args.project, var, ms.year, ms.month)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                written.append(path)
                files[os.path.relpath(path, os.path.join(args.project, "input"))] = sources.sha256_file(path)
                continue
            step = 1 if var == "tp" else args.step_hours
            times = sample_times(ms, step)
            print("era5_monthly: %s %s (%d fields) ..." % (var, ms.strftime("%Y-%m"), len(times)), flush=True)
            mean = _aggregate(ds, var, times, lat_idx, lon_idx, args.jobs)
            if var == "tp":                        # m/h accumulation -> mm/day
                mean, units = mean * 1000.0 * 24.0, "mm/day"
            else:
                units = "K"
            _write(path, var, lat_sel, lon_sel, mean, units, len(times), ms.strftime("%Y-%m"))
            written.append(path)
            files[os.path.relpath(path, os.path.join(args.project, "input"))] = sources.sha256_file(path)
    if not written:
        return []
    sources.write_provenance(args.project, NAME, PROVENANCE, [PROVENANCE["url"]], files,
                             window=sources.window_dict(start, end), bbox=list(bbox),
                             extra={"variables": variables, "step_hours": args.step_hours,
                                    "skip_days": args.skip_days})
    return written

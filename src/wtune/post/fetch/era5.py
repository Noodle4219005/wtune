"""post/fetch/era5.py -- ERA5 subsets from the ARCO-ERA5 zarr store (gs://gcp-public-data-arco-era5,
anonymous) for the synoptic comparison figures: 500 hPa height (syn_era5_diff) and 850 hPa
wind and specific humidity (syn_era5_850_diff), at the recipe's rain peak and its last
post-spin-up frame (plus --times).

Written to <project>/input/era5/z500_<YYYYMMDDHH>.nc and lev850_<YYYYMMDDHH>.nc;
post/recipes/era5.py reads them first and only
then falls back to streaming from gcsfs (which compute nodes without internet cannot do). Uses
tools/era5_intermediate.py's open_arco()/crop_indices() through post/recipes/era5.py -- the repo's
one ARCO reader -- and post/met_gridstat.py's writer, so the file is exactly what the recipe
consumes. The valid times are the recipe's own choices: post/recipe._map_time (the finest
domain's rain peak inside post.focus) and post/recipes/synoptic.py _post_spinup_time (last d01
frame at least --spinup-hours after the first).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from wtune.post import sources

NAME = "era5"
HELP = ("ERA5 z500 and 850 hPa wind/humidity subsets (ARCO-ERA5 zarr, anonymous) for the synoptic "
        "comparison -> <project>/input/era5/")
PROVENANCE = {
    "dataset": "ERA5 hourly reanalysis, 500 hPa geopotential and 850 hPa wind and specific humidity "
               "(ARCO-ERA5 zarr, 37 pressure levels)",
    "provider": "ECMWF / Copernicus Climate Change Service; ARCO copy on Google Cloud public datasets",
    "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3 "
           "(https://cloud.google.com/storage/docs/public-datasets/era5)",
    "license": "Copernicus licence (CC-BY 4.0 for ERA5); ARCO conversion is a Google public dataset",
    "citation": "Hersbach, H., et al., 2020: The ERA5 global reanalysis. Q. J. R. Meteorol. Soc., "
                "146, 1999-2049, doi:10.1002/qj.3803.",
    "coverage_time": "1940-01-01 to a few months before present (checked against the store's own "
                     "time axis at fetch time)",
    "coverage_space": "global",
    "resolution": "0.25 deg, hourly",
    "access": "public, anonymous (no credentials); needs internet (a login node on most clusters)",
}
MAX_TIME_MISMATCH = timedelta(hours=1)


def add_arguments(parser):
    parser.add_argument("--spinup-hours", type=float, default=6.0,
                        help="pick the last d01 frame at least this many hours after the first "
                        "(the synoptic recipe's own rule; default 6)")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="degrees added around the d01 bbox (default 2)")
    parser.add_argument("--end", default=None,
                        help="with --bbox, instead of --run-dir: the run's last valid time "
                        "(UTC, YYYYmmddHH), which the synoptic recipe compares at")
    parser.add_argument("--bbox", default=None, help="lat0,lat1,lon0,lon1 of d01")
    parser.add_argument("--times", default=None,
                        help="more valid times (UTC, comma-separated YYYYmmddHH) to fetch")


def _rain_peak(args):
    """The recipe's map time (post/recipe._map_time: the finest domain's rain peak inside the
    project's post.focus), or None."""
    from wtune.post.caseinfo import WRF_TIME_FMT
    from wtune.post.extract import list_frames
    from wtune.post.recipe import _map_time
    doms = [d for d in range(1, 10) if list_frames(args.run_dir, domain=d)]
    focus = times = None
    try:
        import wtune.project
        defaults = wtune.project.project_post_defaults(wtune.project.load_project(args.project))
        focus, times = defaults["focus"], defaults["times"] or None
    except Exception:                                   # noqa: BLE001 -- no project: no box
        pass
    peak = _map_time(args.run_dir, {"domains": [{"domain": d} for d in doms]},
                     args.spinup_hours, focus, post_times=times) if doms else None
    return datetime.strptime(peak, WRF_TIME_FMT) if peak else None


def _targets(args):
    """([valid datetimes], bbox) the recipe will ask for: the rain peak (the one-time maps) and
    the last post-spin-up frame, plus --times; or (None, reason)."""
    from wtune.post.caseinfo import WRF_TIME_FMT
    from wtune.post.recipes.synoptic import _post_spinup_time
    extra = [sources.parse_time(t) for t in (getattr(args, "times", None) or "").split(",") if t]
    if getattr(args, "end", None) and getattr(args, "bbox", None):
        return ([sources.parse_time(args.end)] + extra,
                tuple(float(x) for x in args.bbox.split(","))), None
    if not getattr(args, "run_dir", None):
        return None, "--run-dir is required (the valid time and region come from its wrfout_d01)"
    chosen, _frames = _post_spinup_time(args.run_dir, 1, args.spinup_hours)
    if chosen is None:
        return None, "no wrfout_d01_* frames in %s" % args.run_dir
    times = [_rain_peak(args), datetime.strptime(chosen[0], WRF_TIME_FMT)] + extra
    hours = sorted({t.replace(minute=0, second=0, microsecond=0) for t in times if t})
    return (hours, sources.run_window(args.run_dir)["bbox"]), None


def _target(args):
    """(first valid datetime, bbox), the old single-time contract (present())."""
    tgt, why = _targets(args)
    return ((tgt[0][0], tgt[1]), None) if tgt else (None, why)


def present(project, run_dir):
    """True when every subset the recipe would read for this run already exists."""
    from wtune.post.recipes import era5 as recipe_era5
    if not run_dir:
        return False
    args = type("A", (), dict(run_dir=run_dir, spinup_hours=6.0, project=project, times=None))()
    tgt, _ = _targets(args)
    return bool(tgt) and all(os.path.exists(recipe_era5.subset_path(project, t))
                             and os.path.exists(recipe_era5.level_path(project, t))
                             for t in tgt[0])


def _sha_ok(path, known, read=None):
    """Existing file that opens (with `read`, default the z500 reader) and, when a sha is
    recorded, still matches it."""
    if not os.path.exists(path):
        return False
    from wtune.post.recipes import era5 as recipe_era5
    try:
        (read or recipe_era5.read_subset)(path)
    except Exception:  # noqa: BLE001 -- unreadable/truncated: refetch
        return False
    return known is None or known == sources.sha256_file(path)


def run(args):
    from wtune.post.recipes import era5 as recipe_era5
    tgt, why = _targets(args)
    if tgt is None:
        print("era5: %s" % why)
        return []
    times, bbox = tgt
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    files, written, ds = {}, [], None
    for time in times:
        for path, write, read in ((recipe_era5.subset_path(args.project, time), _write_z500,
                                   recipe_era5.read_subset),
                                  (recipe_era5.level_path(args.project, time), _write_level,
                                   recipe_era5.read_level)):
            rel = os.path.basename(path)
            if _sha_ok(path, known.get(rel), read):
                print("era5: %s already present, nothing downloaded" % path)
            else:
                if ds is None:
                    try:
                        ds = recipe_era5.open_era5()
                        for t in times:
                            recipe_era5.check_time_covered(ds, t)
                    except recipe_era5.Era5UnavailableError as e:
                        print("era5: no data -- %s" % e)
                        return written
                write(ds, time, bbox, args.margin, path)
            files[rel] = sources.sha256_file(path)
            written.append(path)
    sources.write_provenance(args.project, NAME, PROVENANCE,
                             [recipe_era5.arco_url()] if ds is not None else [], files,
                             window=sources.window_dict(times[0], times[-1]), bbox=list(bbox),
                             extra={"valid_times": [t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in times],
                                    "fields": "Z500 (m); U850, V850 (m/s), Q850 (g/kg)"})
    return written


def _valid(ds, time):
    import numpy as np
    v = ds.sel(time=np.datetime64(time), method="nearest")["time"].values
    return datetime.utcfromtimestamp(v.astype("datetime64[s]").astype(int))


def _write_z500(ds, time, bbox, margin, path):
    from wtune.post.recipes import era5 as recipe_era5
    lat, lon, z = recipe_era5.z500_at(ds, time, bbox, margin=margin)
    recipe_era5.write_subset(lat, lon, z, _valid(ds, time), path)
    print("era5: wrote %s (%dx%d)" % (path, len(lat), len(lon)))


def _write_level(ds, time, bbox, margin, path):
    from wtune.post.recipes import era5 as recipe_era5
    lat, lon, fields = recipe_era5.level_fields_at(ds, time, bbox, margin=margin)
    recipe_era5.write_level(lat, lon, fields, _valid(ds, time), path)
    print("era5: wrote %s (%s, %dx%d)" % (path, ", ".join(fields), len(lat), len(lon)))

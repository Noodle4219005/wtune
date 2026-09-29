"""post/fetch/era5.py -- ERA5 500 hPa geopotential subset from the ARCO-ERA5 zarr store
(gs://gcp-public-data-arco-era5, anonymous) for the synoptic z500 comparison figure.

Written to <project>/input/era5/z500_<YYYYMMDDHH>.nc; post/recipes/era5.py reads it first and only
then falls back to streaming from gcsfs (which compute nodes without internet cannot do). Uses
tools/era5_intermediate.py's open_arco()/crop_indices() through post/recipes/era5.py -- the repo's
one ARCO reader -- and post/met_gridstat.py's writer, so the file is exactly what the recipe
consumes. The valid time is the recipe's own choice (post/recipes/synoptic.py _post_spinup_time:
last d01 frame at least --spinup-hours after the first).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from post import sources

NAME = "era5"
HELP = "ERA5 z500 subset (ARCO-ERA5 zarr, anonymous) for the synoptic comparison -> <project>/input/era5/"
PROVENANCE = {
    "dataset": "ERA5 hourly reanalysis, 500 hPa geopotential (ARCO-ERA5 zarr, 37 pressure levels)",
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
    "access": "public, anonymous (no credentials); needs internet (login node on scc)",
}
MAX_TIME_MISMATCH = timedelta(hours=1)


def add_arguments(parser):
    parser.add_argument("--spinup-hours", type=float, default=6.0,
                        help="pick the last d01 frame at least this many hours after the first "
                        "(the synoptic recipe's own rule; default 6)")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="degrees added around the d01 bbox (default 2)")


def _target(args):
    """(valid datetime, bbox) the synoptic recipe will ask for, or (None, reason)."""
    from post.caseinfo import WRF_TIME_FMT
    from post.recipes.synoptic import _post_spinup_time
    if not getattr(args, "run_dir", None):
        return None, "--run-dir is required (the valid time and region come from its wrfout_d01)"
    chosen, _frames = _post_spinup_time(args.run_dir, 1, args.spinup_hours)
    if chosen is None:
        return None, "no wrfout_d01_* frames in %s" % args.run_dir
    return (datetime.strptime(chosen[0], WRF_TIME_FMT), sources.run_window(args.run_dir)["bbox"]), None


def present(project, run_dir):
    """True when the subset the recipe would read for this run already exists."""
    from post.recipes import era5 as recipe_era5
    if not run_dir:
        return False
    args = type("A", (), dict(run_dir=run_dir, spinup_hours=6.0))()
    tgt, _ = _target(args)
    return bool(tgt) and os.path.exists(recipe_era5.subset_path(project, tgt[0]))


def _sha_ok(path, known):
    """Existing file that opens and (when a sha is recorded) still matches it."""
    if not os.path.exists(path):
        return False
    from post.recipes import era5 as recipe_era5
    try:
        recipe_era5.read_subset(path)
    except Exception:  # noqa: BLE001 -- unreadable/truncated: refetch
        return False
    return known is None or known == sources.sha256_file(path)


def run(args):
    from post.recipes import era5 as recipe_era5
    tgt, why = _target(args)
    if tgt is None:
        print("era5: %s" % why)
        return []
    time, bbox = tgt
    path = recipe_era5.subset_path(args.project, time)
    rel = os.path.basename(path)
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {}).get(rel)
    window = sources.window_dict(time, time)
    if _sha_ok(path, known):
        sha = sources.sha256_file(path)
        sources.write_provenance(args.project, NAME, PROVENANCE, [], {rel: sha}, window=window,
                                 bbox=list(bbox))
        print("era5: %s already present (sha256 %s...), nothing downloaded" % (path, sha[:12]))
        return [path]
    try:
        ds = recipe_era5.open_era5()
        recipe_era5.check_time_covered(ds, time)
    except recipe_era5.Era5UnavailableError as e:
        print("era5: no data -- %s" % e)
        return []
    lat, lon, z = recipe_era5.z500_at(ds, time, bbox, margin=args.margin)
    import numpy as np
    valid = ds.sel(time=np.datetime64(time), method="nearest")["time"].values
    valid = datetime.utcfromtimestamp(valid.astype("datetime64[s]").astype(int))
    recipe_era5.write_subset(lat, lon, z, valid, path)
    sha = sources.sha256_file(path)
    sources.write_provenance(args.project, NAME, PROVENANCE, [recipe_era5.arco_url()], {rel: sha},
                             window=window, bbox=list(bbox),
                             extra={"valid_time": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "level_hpa": 500, "variable": "geopotential/g -> Z500 (m)"})
    print("era5: wrote %s (%dx%d, valid %s)" % (path, len(lat), len(lon), valid))
    return [path]

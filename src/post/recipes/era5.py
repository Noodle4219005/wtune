"""post/recipes/era5.py -- ERA5 (ARCO) access for the synoptic/climate comparison figures
(decision 0057 section 3: "能匿名取得的優先：ERA5（ARCO）..."). Reuses
tools/era5_intermediate.py's own open_arco() (the repo's one ARCO-ERA5 zarr reader,
gcs://gcp-public-data-arco-era5, key-free) and crop_indices() (its bbox/longitude-wrap
cropping) rather than adding a second ARCO reader; this module only selects a pressure level
and valid time and hands back plain numpy arrays for post/recipes/synoptic.py (and, later,
climate.py) to difference against the model.

gcsfs/zarr were added to envs/wtune/pixi.toml (2026-09-29, alongside the wtune-project work) so
this environment can open the anonymous public ARCO-ERA5 bucket directly; open_era5() still
raises Era5UnavailableError (rather than crashing `wtune post recipe`) if either package is
missing from whatever interpreter actually runs this -- e.g. a stale pixi env that has not been
reinstalled since this change, or envs/ai's separate GPU env.
"""
from __future__ import annotations

from datetime import datetime

G = 9.80665


class Era5UnavailableError(RuntimeError):
    pass


def _era5_mod():
    import tools.era5_intermediate as era5_mod
    return era5_mod


def open_era5():
    era5_mod = _era5_mod()
    try:
        return era5_mod.open_arco()
    except ImportError as e:
        raise Era5UnavailableError(
            "ERA5 (ARCO): gcsfs/zarr not importable in this interpreter -- %s. "
            "envs/wtune/pixi.toml declares both; run `pixi install --manifest-path "
            "envs/wtune/pixi.toml` (or the equivalent under envs/ai) to enable this figure." % e)
    except Exception as e:
        raise Era5UnavailableError("ERA5 (ARCO): could not open the zarr store: %s" % e)


def z500_at(ds, time, bbox, margin=2.0):
    """500 hPa geopotential height (m), nearest `time`, cropped to bbox=(lat0,lat1,lon0,lon1)
    (lon in -180..180, per era5_intermediate.crop_indices' own contract) + margin degrees.
    Returns (lat_1d, lon_1d, z_2d)."""
    import numpy as np
    era5_mod = _era5_mod()
    lat_idx, lon_idx, lon_sel = era5_mod.crop_indices(ds["latitude"].values,
                                                      ds["longitude"].values, bbox, margin)
    lat_sel = ds["latitude"].values[lat_idx]
    snap = ds.sel(time=np.datetime64(time), method="nearest")
    levels = ds["level"].values
    k = int(np.argmin(np.abs(levels - 500)))
    g = np.asarray(snap["geopotential"].values[k])
    g = g[..., lat_idx, :][..., lon_idx]
    return lat_sel, lon_sel, g / G


# --------------------------------------------------------------------------------------
# Local subset: compute nodes on scc have no internet, so the ARCO zarr (GCS) is only
# reachable from a login node. `python -m post.recipes.era5 --project P --run-dir R` fetches
# the few MB the synoptic figure needs into <project>/input/era5/ there; the recipe then
# reads that file first and only falls back to gcsfs when it is absent.
# --------------------------------------------------------------------------------------

def subset_path(project_dir, time):
    import os
    return os.path.join(project_dir, "input", "era5", "z500_%s.nc" % time.strftime("%Y%m%d%H"))


def write_subset(lat, lon, z, valid, path):
    import os
    from post.met_gridstat import write_era5_z500_nc
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return write_era5_z500_nc(lat, lon, z, valid, path)


def read_subset(path):
    """(lat_1d, lon_1d, z_2d, valid datetime) from a file written by write_subset."""
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype="f8")
        lon = np.asarray(ds.variables["lon"][:], dtype="f8")
        z = np.ma.filled(ds.variables["Z500"][0], np.nan).astype("f8")
        t = netCDF4.num2date(ds.variables["time"][0], ds.variables["time"].units)
    return lat, lon, z, datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)


def z500_for(time, bbox, project_dir=None, margin=2.0):
    """(lat, lon, z500_m, valid, source): the project's local subset when present (it takes
    precedence, no network), else the ARCO store via open_era5()/z500_at()."""
    import os
    if project_dir:
        path = subset_path(project_dir, time)
        if os.path.exists(path):
            lat, lon, z, valid = read_subset(path)
            return lat, lon, z, valid, path
    import numpy as np
    ds = open_era5()
    lat, lon, z = z500_at(ds, time, bbox, margin=margin)
    valid = ds.sel(time=np.datetime64(time), method="nearest")["time"].values
    valid = datetime.utcfromtimestamp(valid.astype("datetime64[s]").astype(int))
    return lat, lon, z, valid, "ARCO"


def main(argv=None):
    import argparse
    import netCDF4
    import numpy as np
    ap = argparse.ArgumentParser(description="fetch the ERA5 z500 subset for a run's synoptic "
                                 "figure into <project>/input/era5/ (run on a login node)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--spinup-hours", type=float, default=6)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args(argv)
    from post.caseinfo import WRF_TIME_FMT
    from post.recipes.synoptic import _post_spinup_time
    chosen, _ = _post_spinup_time(args.run_dir, 1, args.spinup_hours)
    if chosen is None:
        raise SystemExit("no wrfout_d01 frames in %s" % args.run_dir)
    time_str, path = chosen
    with netCDF4.Dataset(path) as ds:
        la, lo = np.asarray(ds.variables["XLAT"][0]), np.asarray(ds.variables["XLONG"][0])
    bbox = (float(la.min()), float(la.max()), float(lo.min()), float(lo.max()))
    time = datetime.strptime(time_str, WRF_TIME_FMT)
    ds = open_era5()
    lat, lon, z = z500_at(ds, time, bbox, margin=args.margin)
    valid = ds.sel(time=np.datetime64(time), method="nearest")["time"].values
    valid = datetime.utcfromtimestamp(valid.astype("datetime64[s]").astype(int))
    out = write_subset(lat, lon, z, valid, subset_path(args.project, time))
    print("wrote %s (%dx%d, valid %s)" % (out, len(lat), len(lon), valid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

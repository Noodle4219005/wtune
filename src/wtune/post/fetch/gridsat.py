"""post/fetch/gridsat.py -- NOAA GridSat-B1 infrared window (~11 um) brightness temperature, the
geostationary IR climate record (GOES, Himawari, Meteosat merged; 0.07 deg, every 3 h, 1980 to
about a year before present) from NCEI's HTTPS archive (public, no credentials).

Writes the agent observation format wtune-obs/1, kind satellite_ir, so the cloud set's
cloud_ir_compare / cloud_ir_scores / cloud_ir_fss compare it with the model's OLR brightness
temperature with no agent conversion step: <project>/input/obs/gridsat/irtb_YYYYMMDDHHMM.nc
(1-D lat/lon, `tb` in K, NaN where missing) cropped to the run region, and provenance.json.
Each global file (~42 MB) is downloaded, cropped and deleted.

The packed `irwin_cdr` is read without netCDF4's automatic masking: its valid_range (140-375)
is in unpacked kelvin, and netCDF4 compares it with the packed int16 values, which masked
5110 of 5112 cells over Taiwan on 2017-08-01 21Z. Only _FillValue is masked here.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from wtune.post import sources

NAME = "gridsat"
HELP = "NOAA GridSat-B1 IR brightness temperature (0.07 deg, 3-hourly) -> satellite_ir obs"
PROVENANCE = {
    "format": "wtune-obs/1",
    "kind": "satellite_ir",
    "dataset": "NOAA GridSat-B1 geostationary IR window brightness temperature (irwin_cdr), v02r01",
    "provider": "NOAA NCEI",
    "url": "https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature",
    "license": "public domain (NOAA)",
    "citation": "Knapp, K. R., et al., 2011: Globally gridded satellite observations for climate "
                "studies. Bull. Amer. Meteor. Soc., 92, 893-907, doi:10.1175/2011BAMS3039.1.",
    "coverage_time": "1980-01-01 to about a year before present, every 3 h",
    "coverage_space": "70S-70N, all longitudes",
    "resolution": "0.07 deg, 3-hourly (00, 03, ... 21 UTC)",
    "access": "public, anonymous; needs internet (a login node on most clusters)",
}
BASE = ("https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/"
        "access/%Y/GRIDSAT-B1.%Y.%m.%d.%H.v02r01.nc")
MARGIN_DEG = 1.0


def add_arguments(parser):
    sources.add_window_arguments(parser)
    parser.add_argument("--jobs", type=int, default=4, help="parallel downloads (default 4)")


def valid_times(start, end):
    """The 3-hourly GridSat times inside [start, end]."""
    t = start.replace(minute=0, second=0, microsecond=0)
    t += timedelta(hours=(-t.hour) % 3)
    if t < start:
        t += timedelta(hours=3)
    out = []
    while t <= end:
        out.append(t)
        t += timedelta(hours=3)
    return out


def path_for(project, t):
    return os.path.join(sources.dataset_dir(project, NAME), "irtb_%s.nc" % t.strftime("%Y%m%d%H%M"))


def crop(src_path, dest, bbox, margin=MARGIN_DEG):
    """Writes irtb (1-D lat ascending, lon -180..180, tb K with NaN) for bbox + margin."""
    import netCDF4
    import numpy as np
    from wtune.post.conventions import lon_in_range
    lat0, lat1, lon0, lon1 = bbox
    with netCDF4.Dataset(src_path) as ds:
        ds.set_auto_maskandscale(False)
        lat = np.asarray(ds.variables["lat"][:], "f8")
        lon = np.asarray(ds.variables["lon"][:], "f8")
        v = ds.variables["irwin_cdr"]
        j = np.where((lat >= lat0 - margin) & (lat <= lat1 + margin))[0]
        i = np.where(lon_in_range(lon, lon0 - margin, lon1 + margin))[0]
        if not j.size or not i.size:
            raise ValueError("region %s lies outside GridSat-B1 (70S-70N)" % (bbox,))
        raw = np.asarray(v[0, j[0]:j[-1] + 1, :][:, i])
        fill = v.getncattr("_FillValue")
        tb = np.where(raw == fill, np.nan,
                      raw * float(v.getncattr("scale_factor")) + float(v.getncattr("add_offset")))
    tmp = dest + ".tmp"
    with netCDF4.Dataset(tmp, "w") as out:
        out.createDimension("lat", len(j))
        out.createDimension("lon", len(i))
        out.createVariable("lat", "f8", ("lat",))[:] = lat[j[0]:j[-1] + 1]
        out.createVariable("lon", "f8", ("lon",))[:] = lon[i]
        var = out.createVariable("tb", "f4", ("lat", "lon"), zlib=True, complevel=4,
                                 fill_value=np.float32(np.nan))
        var[:] = tb.astype("f4")
        var.units, var.long_name = "K", "GridSat-B1 irwin_cdr brightness temperature near 11 um"
    os.replace(tmp, dest)
    return dest


def run(args):
    try:
        start, end, bbox = sources.resolve_window(args)
    except ValueError as e:
        print("gridsat: %s" % e)
        return []
    times = valid_times(start, end)
    if not times:
        print("gridsat: window %s..%s holds no 3-hourly GridSat time" % (start, end))
        return []
    known = (sources.read_provenance(args.project, NAME) or {}).get("files", {})
    os.makedirs(sources.dataset_dir(args.project, NAME), exist_ok=True)

    def fetch_one(t):
        """Download only (threads); netCDF4/HDF5 is not thread-safe, so crops run serially."""
        dest = path_for(args.project, t)
        if os.path.isfile(dest) and known.get(os.path.basename(dest)) == sources.sha256_file(dest):
            return t, dest, None, None
        raw = dest + ".global.nc"
        try:
            sha, _dl = sources.download(t.strftime(BASE), raw)
        except OSError as e:
            return t, dest, None, str(e)
        return t, dest, raw if sha else "", None              # "": not on the server

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for t, dest, raw, err in ex.map(fetch_one, times):
            if raw is None and err is None:                   # already present
                results.append((t, dest, known[os.path.basename(dest)], False, None))
            elif err or not raw:
                results.append((t, dest, None, False, err))
            else:
                try:
                    crop(raw, dest, bbox)
                    results.append((t, dest, sources.sha256_file(dest), True, None))
                except (OSError, ValueError, KeyError) as e:
                    results.append((t, dest, None, False, str(e)))
                finally:
                    os.remove(raw)
    files = {os.path.basename(d): sha for _t, d, sha, _n, _e in results if sha}
    urls = [t.strftime(BASE) for t, _d, sha, new, _e in results if sha and new]
    failed = [(os.path.basename(d), e) for _t, d, _s, _n, e in results if e]
    absent = [t for t, _d, sha, _n, e in results if sha is None and not e]
    if absent:
        print("gridsat: %d of %d times are not on the server (e.g. %s)"
              % (len(absent), len(times), absent[0]))
    if files:
        sources.write_provenance(args.project, NAME, PROVENANCE, urls, files,
                                 window=sources.window_dict(start, end), bbox=list(bbox))
    print("gridsat: %d file(s) in %s (%d newly written)"
          % (len(files), sources.dataset_dir(args.project, NAME), len(urls)))
    if failed:
        raise sources.FetchIncomplete(
            "gridsat: %d of %d times could not be fetched (%s: %s); run the same fetch again"
            % (len(failed), len(times), failed[0][0], failed[0][1]))
    return [d for _t, d, sha, _n, _e in results if sha]

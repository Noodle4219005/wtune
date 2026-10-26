"""NOAA CDR GridSat-B1 (0.07 deg, 3-hourly) IR window brightness temperature -> wtune-obs/1
satellite_ir.

    python3 -I convert.py RAW_DIR OUT_DIR LAT0 LAT1 LON0 LON1

irwin_cdr: brightness temperature near 11 um in K, unpacked by hand (fill -> NaN),
valid at the file's `time`. Cut to the box. Over this region in 2016 the source is GOES-13.
"""
import glob, hashlib, json, os, sys
from datetime import datetime, timezone
import netCDF4
import numpy as np

raw, out = sys.argv[1:3]
lat0, lat1, lon0, lon1 = map(float, sys.argv[3:7])
os.makedirs(out, exist_ok=True)
files = {}
for p in sorted(glob.glob(os.path.join(raw, "GRIDSAT-B1.*.nc"))):
    with netCDF4.Dataset(p) as ds:
        tv = ds.variables["time"]
        t = netCDF4.num2date(tv[0], tv.units, only_use_cftime_datetimes=False)
        lat = np.asarray(ds.variables["lat"][:], "f8")
        lon = np.asarray(ds.variables["lon"][:], "f8")
        la = (lat >= lat0) & (lat <= lat1)
        lo = (lon >= lon0) & (lon <= lon1)
        v = ds.variables["irwin_cdr"]
        units = getattr(v, "units", "K")
        # valid_range is in K but netCDF4 tests it against the packed shorts, masking every
        # real value: unpack by hand
        v.set_auto_maskandscale(False)
        packed = v[0][np.ix_(la, lo)].astype("f8")
        tb = np.where(packed == v._FillValue, np.nan, packed * v.scale_factor + v.add_offset)
        tb = np.where((tb >= 140) & (tb <= 375), tb, np.nan)
    assert units.lower() in ("k", "kelvin"), units
    name = "irtb_%s.nc" % datetime(t.year, t.month, t.day, t.hour, t.minute).strftime("%Y%m%d%H%M")
    path = os.path.join(out, name)
    with netCDF4.Dataset(path, "w") as dst:
        dst.createDimension("lat", int(la.sum())); dst.createDimension("lon", int(lo.sum()))
        x = dst.createVariable("lat", "f8", ("lat",)); x[:] = lat[la]; x.units = "degrees_north"
        x = dst.createVariable("lon", "f8", ("lon",)); x[:] = lon[lo]; x.units = "degrees_east"
        x = dst.createVariable("tb", "f4", ("lat", "lon"), zlib=True, complevel=4,
                               fill_value=np.float32(np.nan))
        x[:] = tb; x.units = "K"
        dst.source = os.path.basename(p)
    files[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump({
    "format": "wtune-obs/1", "kind": "satellite_ir",
    "dataset": "NOAA CDR GridSat-B1 IR window (~11 um) brightness temperature, 0.07 deg",
    "provider": "NOAA NCEI",
    "url": "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/",
    "license": "NOAA Climate Data Record, US Government work, no restriction on use",
    "citation": "Knapp, K. R., et al., 2011: Globally gridded satellite observations for climate "
                "studies. BAMS, 92, 893-907, doi:10.1175/2011BAMS3039.1",
    "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "box": [lat0, lat1, lon0, lon1],
    "files": files,
}, open(os.path.join(out, "provenance.json"), "w"), indent=1)
print(len(files), "files")

"""CMORPH CDR v1.0 hourly 0.25 deg (NOAA NCEI) -> wtune-obs/1 gridded_precip.

    python3 -I convert.py RAW_DIR OUT_DIR LAT0 LAT1 LON0 LON1

Each raw file holds a rain RATE (mm/hr) for the hour STARTING at its `time`; the output is
precip_<start+1h>_01h.nc = the accumulation over the hour ENDING then (rate x 1 h), cut to
the box (lon -180..180). Missing (-999) -> NaN.
"""
import glob, hashlib, json, os, sys
from datetime import datetime, timedelta, timezone
import netCDF4
import numpy as np

raw, out = sys.argv[1], sys.argv[2]
lat0, lat1, lon0, lon1 = map(float, sys.argv[3:7])
os.makedirs(out, exist_ok=True)
files = {}
for p in sorted(glob.glob(os.path.join(raw, "CMORPH_V1.0_ADJ_0.25deg-HLY_*.nc"))):
    with netCDF4.Dataset(p) as ds:
        t = ds.variables["time"]
        start = netCDF4.num2date(t[0], t.units, only_use_cftime_datetimes=False)
        lat = np.asarray(ds.variables["lat"][:], "f8")
        lon = np.asarray(ds.variables["lon"][:], "f8")
        rate = np.ma.filled(ds.variables["cmorph"][0].astype("f8"), np.nan)   # mm/hr, scaled
    lon = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon)
    lon, rate = lon[order], rate[:, order]
    la = (lat >= lat0) & (lat <= lat1)
    lo = (lon >= lon0) & (lon <= lon1)
    end = datetime(start.year, start.month, start.day, start.hour) + timedelta(hours=1)
    name = "precip_%s_01h.nc" % end.strftime("%Y%m%d%H")
    dst_path = os.path.join(out, name)
    with netCDF4.Dataset(dst_path, "w") as dst:
        dst.createDimension("lat", int(la.sum()))
        dst.createDimension("lon", int(lo.sum()))
        v = dst.createVariable("lat", "f8", ("lat",)); v[:] = lat[la]; v.units = "degrees_north"
        v = dst.createVariable("lon", "f8", ("lon",)); v[:] = lon[lo]; v.units = "degrees_east"
        v = dst.createVariable("precip", "f4", ("lat", "lon"), fill_value=np.float32(np.nan))
        v[:] = rate[np.ix_(la, lo)] * 1.0
        v.units = "mm"
        v.long_name = "CMORPH CDR accumulation over the hour ending %s UTC" % end
        dst.source = os.path.basename(p)
    files[name] = hashlib.sha256(open(dst_path, "rb").read()).hexdigest()
prov = {
    "format": "wtune-obs/1", "kind": "gridded_precip",
    "dataset": "NOAA CDR CMORPH v1.0 bias-corrected, hourly 0.25 deg",
    "provider": "NOAA NCEI / NOAA CPC",
    "url": "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg/",
    "license": "NOAA Climate Data Record, US Government work, no restriction on use",
    "citation": "Xie, P., et al. (2017), Reprocessed, Bias-Corrected CMORPH Global High-Resolution "
                "Precipitation Estimates from 1998, J. Hydrometeorol., 18, 1617-1641, "
                "doi:10.1175/JHM-D-16-0168.1",
    "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "box": [lat0, lat1, lon0, lon1],
    "files": files,
}
with open(os.path.join(out, "provenance.json"), "w") as fh:
    json.dump(prov, fh, indent=1)
print(len(files), "files")

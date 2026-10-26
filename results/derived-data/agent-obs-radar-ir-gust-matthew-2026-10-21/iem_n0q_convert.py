"""IEM NEXRAD N0Q base-reflectivity mosaic (PNG, 0.005 deg) -> wtune-obs/1 radar.

    python3 -I convert.py RAW_DIR SITES_TXT OUT_DIR LAT0 LAT1 LON0 LON1

N0Q: palette index i -> dBZ = 0.5 i - 32.5 (i = 1..245 is -32..90 dBZ); i = 0 is no echo
inside coverage. Upper-left corner 50N 126W, 0.005 deg (the .wld world file). The mosaic is
the lowest tilt (base reflectivity), not a column maximum. Cut to the box, reduced 4x4 by the
maximum (0.02 deg), and cells farther than 230 km from every NEXRAD site set to NaN (no
coverage, not "no echo"). Sites: NCEI HOMR list (WTune's stage4/nexrad_sites.txt).
"""
import glob, hashlib, json, os, sys
from datetime import datetime, timezone
import netCDF4
import numpy as np
from PIL import Image

raw, sites_txt, out = sys.argv[1:4]
lat0, lat1, lon0, lon1 = map(float, sys.argv[4:8])
sys.path.insert(0, os.environ.get("WTUNE_SRC", ""))
from wtune.post.fetch.nexrad import parse_homr
sites = parse_homr(open(sites_txt).read())
slat = np.radians([s[1] for s in sites]); slon = np.radians([s[2] for s in sites])
os.makedirs(out, exist_ok=True)
DX, N = 0.005, 4
files = {}
mask = None
for p in sorted(glob.glob(os.path.join(raw, "n0q_*.png"))):
    t = datetime.strptime(os.path.basename(p)[4:16], "%Y%m%d%H%M")
    idx = np.array(Image.open(p))
    lats = 50.0 - DX * (np.arange(idx.shape[0]) + 0.5)
    lons = -126.0 + DX * (np.arange(idx.shape[1]) + 0.5)
    r = np.where((lats >= lat0) & (lats <= lat1))[0]
    c = np.where((lons >= lon0) & (lons <= lon1))[0]
    r = r[: len(r) // N * N]; c = c[: len(c) // N * N]
    dbz = 0.5 * idx[np.ix_(r, c)].astype("f4") - 32.5
    dbz = dbz.reshape(len(r) // N, N, len(c) // N, N).max(axis=(1, 3))
    la = lats[r].reshape(-1, N).mean(axis=1)[::-1]
    lo = lons[c].reshape(-1, N).mean(axis=1)
    dbz = dbz[::-1]
    if mask is None:
        LA, LO = np.radians(np.meshgrid(la, lo, indexing="ij"))
        near = np.zeros(LA.shape, bool)
        for a, b in zip(slat, slon):
            d = 2 * 6371 * np.arcsin(np.sqrt(np.sin((LA - a) / 2) ** 2 +
                                             np.cos(a) * np.cos(LA) * np.sin((LO - b) / 2) ** 2))
            near |= d <= 230.0
        mask = near
    dbz = np.where(mask, dbz, np.nan)
    name = "refl_%s.nc" % t.strftime("%Y%m%d%H%M")
    path = os.path.join(out, name)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(la)); ds.createDimension("lon", len(lo))
        v = ds.createVariable("lat", "f8", ("lat",)); v[:] = la; v.units = "degrees_north"
        v = ds.createVariable("lon", "f8", ("lon",)); v[:] = lo; v.units = "degrees_east"
        v = ds.createVariable("refl", "f4", ("lat", "lon"), zlib=True, complevel=4,
                              fill_value=np.float32(np.nan))
        v[:] = dbz; v.units = "dBZ"
        ds.source = os.path.basename(p)
    files[name] = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump({
    "format": "wtune-obs/1", "kind": "radar",
    "dataset": "IEM NEXRAD N0Q base-reflectivity mosaic (0.005 deg, max-reduced to 0.02 deg)",
    "provider": "Iowa Environmental Mesonet (Iowa State University), from NWS NEXRAD Level III",
    "url": "https://mesonet.agron.iastate.edu/archive/data/YYYY/MM/DD/GIS/uscomp/n0q_YYYYMMDDHHMM.png",
    "license": "IEM archive, free to use with attribution; NEXRAD data are US Government work",
    "citation": "Iowa Environmental Mesonet NEXRAD mosaic, https://mesonet.agron.iastate.edu/docs/nexrad_mosaic/",
    "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "box": [lat0, lat1, lon0, lon1],
    "files": files,
}, open(os.path.join(out, "provenance.json"), "w"), indent=1)
print(len(files), "files, grid", len(la), "x", len(lo), "covered", int(mask.sum()))

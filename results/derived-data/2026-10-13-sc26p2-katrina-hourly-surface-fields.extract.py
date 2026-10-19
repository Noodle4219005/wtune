"""Extract hourly surface fields of the SC26-P2 v090g run for the learning page (read-only on the run)."""
import csv, glob, json, os, re, sys
import numpy as np
from netCDF4 import Dataset
import wrf

P = os.path.expanduser("~/wtune-projects/sc26p2-20261002")
C = P + "/_nc_cache/v090g-8d170413/seg_0"
OUT = sys.argv[1]
files = sorted(glob.glob(C + "/wrfout_d01_*.nc"))
nt = len(files)
slp_q, wsp_q, dc_q, dn_q = [], [], [], []
prev_c = prev_n = None
stats = dict(slp_min=1e9, slp_max=-1e9, wsp_max=0.0, dc_max=0.0, dn_max=0.0, rainsh_max=0.0)
times = []
for k, f in enumerate(files):
    nc = Dataset(f)
    times.append(re.search(r"d01_(.*)\.nc$", f).group(1))
    slp = np.asarray(wrf.getvar(nc, "slp", units="hPa", meta=False), dtype="f8")
    u10 = nc["U10"][0]; v10 = nc["V10"][0]
    wsp = np.hypot(u10, v10)
    rc = np.asarray(nc["RAINC"][0], dtype="f8") + np.asarray(nc["RAINSH"][0], dtype="f8")
    rn = np.asarray(nc["RAINNC"][0], dtype="f8")
    stats["rainsh_max"] = max(stats["rainsh_max"], float(nc["RAINSH"][0].max()))
    if k == 0:
        hgt = np.asarray(nc["HGT"][0]); land = np.asarray(nc["LANDMASK"][0])
        lat1 = np.asarray(nc["XLAT"][0][:, 0]); lon1 = np.asarray(nc["XLONG"][0][0, :])
        lat_spread = float(np.abs(nc["XLAT"][0] - lat1[:, None]).max())
        lon_spread = float(np.abs(nc["XLONG"][0] - lon1[None, :]).max())
        attrs = {a: (float(getattr(nc, a)) if a != "MAP_PROJ" else int(getattr(nc, a)))
                 for a in ("DX", "CEN_LAT", "CEN_LON", "TRUELAT1", "STAND_LON", "MAP_PROJ")}
        dc = np.zeros_like(rc); dn = np.zeros_like(rn)
    else:
        dc = np.maximum(rc - prev_c, 0); dn = np.maximum(rn - prev_n, 0)
    prev_c, prev_n = rc, rn
    stats["slp_min"] = min(stats["slp_min"], float(slp.min())); stats["slp_max"] = max(stats["slp_max"], float(slp.max()))
    stats["wsp_max"] = max(stats["wsp_max"], float(wsp.max()))
    stats["dc_max"] = max(stats["dc_max"], float(dc.max())); stats["dn_max"] = max(stats["dn_max"], float(dn.max()))
    slp_q.append(np.clip(np.rint((slp - 940.0) / 0.4), 0, 255).astype("u1"))
    wsp_q.append(np.clip(np.rint(wsp / 0.25), 0, 255).astype("u1"))
    dc_q.append(np.clip(np.rint(np.sqrt(dc) * 25.0), 0, 255).astype("u1"))
    dn_q.append(np.clip(np.rint(np.sqrt(dn) * 25.0), 0, 255).astype("u1"))
    nc.close()
ny, nx = slp_q[0].shape
blob = b"".join(a.tobytes() for grp in (slp_q, wsp_q, dc_q, dn_q) for a in grp)
blob += np.clip(np.rint(hgt / 12.0), 0, 255).astype("u1").tobytes() + (land > 0.5).astype("u1").tobytes()
open(OUT + "/katrina.bin", "wb").write(blob)

track = []
for r in csv.DictReader(open(P + "/post/v090g/tc/track_domains_v090g.csv")):
    track.append([r["time"], round(float(r["lat"]), 3), round(float(r["lon"]), 3), round(float(r["pmin_hpa"]), 1), round(float(r["vmax_ms"]), 1)])
obs = [[r["time"], r["lat"], r["lon"], r["pmin_hpa"], round(r["vmax_ms"], 1)]
       for r in json.load(open(P + "/post/v090g/tc/obs.json"))["hurdat2"]
       if times[0] <= r["time"] <= times[-1]]
steps = []
pat = re.compile(r"Timing for main: time \S+ on domain\s+1:\s+([0-9.]+) elapsed")
for l in open(P + "/runs/v090g/seg_0/rsl.error.0000", errors="replace"):
    m = pat.search(l)
    if m: steps.append(round(float(m.group(1)), 4))
meta = dict(nx=nx, ny=ny, nt=nt, times=times, attrs=attrs, stats=stats,
            lat=[round(float(v), 4) for v in lat1], lon=[round(float(v), 4) for v in lon1],
            lat_spread=lat_spread, lon_spread=lon_spread,
            quant=dict(slp=[940.0, 0.4], wsp=[0.0, 0.25], rain_sqrt_scale=25.0, hgt_scale=12.0),
            order=["slp", "wsp", "d_rainc", "d_rainnc", "hgt", "landmask"],
            track=track, obs=obs, steps=steps,
            hgt_max=float(hgt.max()), land_frac=float((land > 0.5).mean()))
json.dump(meta, open(OUT + "/katrina.json", "w"), separators=(",", ":"))
print("frames", nt, "grid", nx, ny, "bytes", len(blob), "stats", {k: round(v, 2) for k, v in stats.items()},
      "lat_spread", lat_spread, "lon_spread", lon_spread, "steps", len(steps), "track", len(track), "obs", len(obs))

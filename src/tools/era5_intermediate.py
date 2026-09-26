"""ARCO-ERA5 -> WPS intermediate files, INCLUDING hydrometeors (decision 0054 phase 3).

GFS analyses carry no cloud/rain, so a rebuild segment cold-started from them spends its first
hour growing clouds (conus25 job 1238: per-step cost 0.999 -> 1.070 s over the 60-min spin-up,
truth ~1.10). ERA5 has cloud liquid/ice, rain and snow on its 37 pressure levels; WRF's
real.exe interpolates QC/QI/QR/QS from met_em whenever metgrid set FLAG_Q* (METGRID.TBL.ARW
entries QR/QC/QI/QS carry flag_in_output), with no namelist switch -- so a segment can start
with its clouds already present.

The files are written directly (scenario.wps_intermediate), bypassing ungrib; names and levels
follow ungrib's Vtable.ECMWF so METGRID.TBL.ARW's ECMWF entries apply unchanged (ST000007 ...,
SM000007 ..., LANDSEA used as the soil interpolation mask). Source: the public, key-free ARCO
store (Copernicus ERA5 licence: free, commercial use allowed, attribution required).

Runs in the envs/ai pixi environment (needs xarray/zarr/gcsfs); scenario.wps_intermediate needs
only numpy.
"""
import argparse
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scenario import wps_intermediate as wi  # noqa: E402

ARCO_ERA5_URL = "gcs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
# The 37-level store has cloud liquid/ice but NOT rain/snow; those exist only on the 137 model
# levels (hybrid a/b coefficients in data/ecmwf_l137_ab.csv, from the ECMWF L137 table).
ARCO_ERA5_ML_URL = "gcs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
L137_AB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ecmwf_l137_ab.csv")
MODEL_LEVEL = {
    "specific_rain_water_content": ("QR", "kg kg-1", "Rain water mixing ratio"),
    "specific_snow_water_content": ("QS", "kg kg-1", "Snow mixing ratio"),
}
G = 9.80665
MAP_SOURCE = "ERA5 (ARCO)"

# ARCO long name -> (WPS field, units, description, scale)
PRESSURE_LEVEL = {
    "temperature": ("TT", "K", "Temperature", 1.0),
    "u_component_of_wind": ("UU", "m s-1", "U", 1.0),
    "v_component_of_wind": ("VV", "m s-1", "V", 1.0),
    "specific_humidity": ("SPECHUMD", "kg kg-1", "Specific humidity", 1.0),
    "geopotential": ("GHT", "m", "Height", 1.0 / G),
    "specific_cloud_liquid_water_content": ("QC", "kg kg-1", "Cloud water mixing ratio", 1.0),
    "specific_cloud_ice_water_content": ("QI", "kg kg-1", "Cloud ice mixing ratio", 1.0),
    "specific_rain_water_content": ("QR", "kg kg-1", "Rain water mixing ratio", 1.0),
    "specific_snow_water_content": ("QS", "kg kg-1", "Snow mixing ratio", 1.0),
}
# ARCO long name -> (WPS field, units, description, xlvl, scale)
SURFACE = {
    "surface_pressure": ("PSFC", "Pa", "Surface pressure", wi.SURFACE_PA, 1.0),
    "mean_sea_level_pressure": ("PMSL", "Pa", "Sea-level pressure", wi.SEALEVEL_PA, 1.0),
    "skin_temperature": ("SKINTEMP", "K", "Skin temperature", wi.SURFACE_PA, 1.0),
    "2m_temperature": ("TT", "K", "Temperature at 2 m", wi.SURFACE_PA, 1.0),
    "10m_u_component_of_wind": ("UU", "m s-1", "U at 10 m", wi.SURFACE_PA, 1.0),
    "10m_v_component_of_wind": ("VV", "m s-1", "V at 10 m", wi.SURFACE_PA, 1.0),
    "land_sea_mask": ("LANDSEA", "proprtn", "Land/Sea flag", wi.SURFACE_PA, 1.0),
    "geopotential_at_surface": ("SOILHGT", "m", "Terrain field of source analysis",
                                wi.SURFACE_PA, 1.0 / G),
    "sea_ice_cover": ("SEAICE", "proprtn", "Ice flag", wi.SURFACE_PA, 1.0),
    "snow_depth": ("SNOW", "kg m-2", "Water equivalent snow depth", wi.SURFACE_PA, 1000.0),
    "soil_temperature_level_1": ("ST000007", "K", "T 0-7 cm below ground", wi.SURFACE_PA, 1.0),
    "soil_temperature_level_2": ("ST007028", "K", "T 7-28 cm below ground", wi.SURFACE_PA, 1.0),
    "soil_temperature_level_3": ("ST028100", "K", "T 28-100 cm below ground", wi.SURFACE_PA, 1.0),
    "soil_temperature_level_4": ("ST100289", "K", "T 100-289 cm below ground", wi.SURFACE_PA, 1.0),
    "volumetric_soil_water_layer_1": ("SM000007", "fraction", "Soil moisture 0-7 cm",
                                      wi.SURFACE_PA, 1.0),
    "volumetric_soil_water_layer_2": ("SM007028", "fraction", "Soil moisture 7-28 cm",
                                      wi.SURFACE_PA, 1.0),
    "volumetric_soil_water_layer_3": ("SM028100", "fraction", "Soil moisture 28-100 cm",
                                      wi.SURFACE_PA, 1.0),
    "volumetric_soil_water_layer_4": ("SM100289", "fraction", "Soil moisture 100-289 cm",
                                      wi.SURFACE_PA, 1.0),
}


def specific_humidity_from_dewpoint(td_k, p_pa):
    """Bolton (1980) vapour pressure over water; q = 0.622 e / (p - 0.378 e)."""
    td_c = np.asarray(td_k, dtype=np.float64) - 273.15
    e = 611.2 * np.exp(17.67 * td_c / (td_c + 243.5))
    return 0.622 * e / (np.asarray(p_pa, dtype=np.float64) - 0.378 * e)


def load_l137_ab(path=L137_AB):
    """(a_half[138] Pa, b_half[138]) from the ECMWF L137 table."""
    rows = [l.split(",") for l in open(path) if l.strip() and not l.startswith("#")][1:]
    return (np.array([float(r[1]) for r in rows]), np.array([float(r[2]) for r in rows]))


def model_levels_to_pressure(q_ml, ps, a_half, b_half, p_targets_pa):
    """Interpolate a (nlev, ny, nx) model-level field to pressure levels, linear in ln p,
    per column. Full-level pressure = mean of the bounding half levels (as calc_ecmwf_p does).
    Outside the column's range the nearest model level is used (metgrid's own fill_lev then
    handles below-ground levels)."""
    a_full = 0.5 * (a_half[:-1] + a_half[1:])
    b_full = 0.5 * (b_half[:-1] + b_half[1:])
    lnp = np.log(a_full[:, None, None] + b_full[:, None, None] * ps[None].astype(np.float64))
    nlev = lnp.shape[0]
    out = np.empty((len(p_targets_pa),) + ps.shape, dtype=np.float64)
    for i, ptarget in enumerate(p_targets_pa):
        lt = np.log(ptarget)
        k = np.clip((lnp < lt).sum(axis=0), 1, nlev - 1)
        lo = np.take_along_axis(lnp, (k - 1)[None], 0)[0]
        hi = np.take_along_axis(lnp, k[None], 0)[0]
        w = np.clip((lt - lo) / (hi - lo), 0.0, 1.0)
        qlo = np.take_along_axis(q_ml, (k - 1)[None], 0)[0]
        qhi = np.take_along_axis(q_ml, k[None], 0)[0]
        out[i] = qlo * (1.0 - w) + qhi * w
    return out


def crop_indices(lat, lon, bbox, margin):
    """Row/column index arrays for bbox (lat0, lat1, lon0, lon1; lon in -180..180) plus
    `margin` degrees, on a regular grid whose lon is 0..360. Returns (lat_idx ascending south->
    north, lon_idx west->east, lon values in -180..180 continuous across the crop)."""
    lat0, lat1, lon0, lon1 = bbox
    lat = np.asarray(lat)
    lon360 = np.mod(np.asarray(lon), 360.0)
    lat_idx = np.where((lat >= lat0 - margin) & (lat <= lat1 + margin))[0]
    lat_idx = lat_idx[np.argsort(lat[lat_idx])]
    w, e = np.mod(lon0 - margin, 360.0), np.mod(lon1 + margin, 360.0)
    if w <= e:
        lon_idx = np.where((lon360 >= w) & (lon360 <= e))[0]
    else:  # crosses 0 E
        lon_idx = np.concatenate([np.where(lon360 >= w)[0], np.where(lon360 <= e)[0]])
    lon_sel = lon360[lon_idx]
    lon_sel = np.where(lon_sel > 180.0, lon_sel - 360.0, lon_sel)
    lon_sel = np.unwrap(np.radians(lon_sel), discont=np.pi)
    return lat_idx, lon_idx, np.degrees(lon_sel)


def _record(hdate, field, units, desc, xlvl, slab, startlat, startlon, dlat, dlon):
    ny, nx = slab.shape
    return wi.Record(hdate=hdate, xfcst=0.0, map_source=MAP_SOURCE, field=field, units=units,
                     desc=desc, xlvl=float(xlvl), nx=nx, ny=ny, iproj=0, startloc="SWCORNER",
                     startlat=float(startlat), startlon=float(startlon), deltalat=float(dlat),
                     deltalon=float(dlon), earth_radius=wi.EARTH_RADIUS_KM,
                     is_wind_earth_rel=True, slab=np.ascontiguousarray(slab, dtype=np.float32))


def records_for_time(ds, t, bbox, margin=5.0, min_level_hpa=10, ml_ds=None, ab=None):
    """All intermediate records for one valid time from an ARCO-style xarray Dataset.
    ml_ds/ab (optional): the model-level store and L137 coefficients, to add QR/QS."""
    lat_idx, lon_idx, lon_vals = crop_indices(ds["latitude"].values, ds["longitude"].values,
                                              bbox, margin)
    lat_vals = ds["latitude"].values[lat_idx]
    startlat, startlon = lat_vals[0], lon_vals[0]
    dlat = float(np.round(lat_vals[1] - lat_vals[0], 6))
    dlon = float(np.round(lon_vals[1] - lon_vals[0], 6))
    hdate = t.strftime("%Y-%m-%d_%H:%M:%S")
    snap = ds.sel(time=np.datetime64(t))

    def grid(name):
        a = np.asarray(snap[name].values)
        return a[..., lat_idx, :][..., lon_idx]

    recs = []
    levels = [int(l) for l in ds["level"].values if int(l) >= min_level_hpa]
    for name, (field, units, desc, scale) in PRESSURE_LEVEL.items():
        if name not in ds:
            continue
        cube = grid(name)
        for k, lev in enumerate(ds["level"].values):
            if int(lev) not in levels:
                continue
            recs.append(_record(hdate, field, units, desc, int(lev) * 100.0,
                                cube[k] * scale, startlat, startlon, dlat, dlon))
    sfc = {}
    for name, (field, units, desc, xlvl, scale) in SURFACE.items():
        if name in ds:
            sfc[field] = (grid(name) * scale, units, desc, xlvl)
    if "sea_surface_temperature" in ds and "SKINTEMP" in sfc:
        sst = grid("sea_surface_temperature")
        sfc["SST"] = (np.where(np.isfinite(sst), sst, sfc["SKINTEMP"][0]), "K",
                      "Sea surface temperature", wi.SURFACE_PA)
    if "SEAICE" in sfc:
        a, u, d, x = sfc["SEAICE"]
        sfc["SEAICE"] = (np.where(np.isfinite(a), a, 0.0), u, d, x)
    if "2m_dewpoint_temperature" in ds and "PSFC" in sfc:
        q2 = specific_humidity_from_dewpoint(grid("2m_dewpoint_temperature"), sfc["PSFC"][0])
        sfc["SPECHUMD"] = (q2, "kg kg-1", "Specific humidity at 2 m", wi.SURFACE_PA)
    for field, (a, units, desc, xlvl) in sfc.items():
        recs.append(_record(hdate, field, units, desc, xlvl, a, startlat, startlon, dlat, dlon))
    if ml_ds is not None:
        msnap = ml_ds.sel(time=np.datetime64(t))
        p_targets = [int(l) * 100.0 for l in levels]
        for name, (field, units, desc) in MODEL_LEVEL.items():
            q = np.asarray(msnap[name].values)[..., lat_idx, :][..., lon_idx]
            qp = model_levels_to_pressure(q, sfc["PSFC"][0], ab[0], ab[1], p_targets)
            for k, pt in enumerate(p_targets):
                recs.append(_record(hdate, field, units, desc, pt, np.maximum(qp[k], 0.0),
                                    startlat, startlon, dlat, dlon))
    bad = [r.field for r in recs if not np.isfinite(r.slab).all()]
    if bad:
        raise ValueError("ERA5 %s: non-finite values in %s" % (hdate, sorted(set(bad))))
    return recs


def open_arco(url=ARCO_ERA5_URL):
    import gcsfs
    import xarray as xr
    fs = gcsfs.GCSFileSystem(token="anon")
    return xr.open_zarr(fs.get_mapper(url), consolidated=True, chunks=None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bbox", required=True, help="lat0,lat1,lon0,lon1 (lon -180..180)")
    ap.add_argument("--times", required=True,
                    help="comma list of YYYY-MM-DD_HH (hourly ERA5 valid times)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="ERA5")
    ap.add_argument("--margin", type=float, default=5.0)
    ap.add_argument("--min-level-hpa", type=int, default=10)
    ap.add_argument("--rain-snow", action="store_true",
                    help="also add QR/QS from the 137-level store (interpolated to the pressure levels)")
    a = ap.parse_args(argv)
    bbox = tuple(float(x) for x in a.bbox.split(","))
    times = [dt.datetime.strptime(t, "%Y-%m-%d_%H") for t in a.times.split(",")]
    os.makedirs(a.out, exist_ok=True)
    ds = open_arco()
    ml_ds, ab = (open_arco(ARCO_ERA5_ML_URL), load_l137_ab()) if a.rain_snow else (None, None)
    for t in times:
        recs = records_for_time(ds, t, bbox, a.margin, a.min_level_hpa, ml_ds=ml_ds, ab=ab)
        path = os.path.join(a.out, "%s:%s" % (a.prefix, t.strftime("%Y-%m-%d_%H")))
        wi.write_fields(path, recs)
        fields = sorted(set(r.field for r in recs))
        print("wrote %s: %d records, %d levels, fields %s"
              % (path, len(recs), len(set(r.xlvl for r in recs if r.field == "TT")) - 1,
                 " ".join(fields)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

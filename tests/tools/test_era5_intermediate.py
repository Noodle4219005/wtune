"""ERA5 (ARCO layout) -> WPS intermediate, offline with a synthetic dataset."""
import datetime as dt
import os
import sys

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from wtune.tools import era5_intermediate as e5  # noqa: E402
from wtune.scenario import wps_intermediate as wi  # noqa: E402


def _ds():
    lat = np.arange(90.0, -90.25, -0.25)          # ARCO: descending
    lon = np.arange(0.0, 360.0, 0.25)             # ARCO: 0..360
    lev = np.array([10, 500, 850, 1000])
    t = np.array([np.datetime64("2019-11-26T21:00")])
    shp3, shp2 = (1, len(lev), len(lat), len(lon)), (1, len(lat), len(lon))
    data = {n: (("time", "level", "latitude", "longitude"), np.full(shp3, v, np.float32))
            for n, v in (("temperature", 250.0), ("u_component_of_wind", 5.0),
                         ("v_component_of_wind", -3.0), ("specific_humidity", 0.005),
                         ("geopotential", 9.80665 * 1500.0),
                         ("specific_cloud_liquid_water_content", 1e-4),
                         ("specific_cloud_ice_water_content", 2e-5),
                         ("specific_rain_water_content", 3e-5),
                         ("specific_snow_water_content", 4e-5))}
    for n, v in (("surface_pressure", 100000.0), ("mean_sea_level_pressure", 101000.0),
                 ("skin_temperature", 290.0), ("2m_temperature", 288.0),
                 ("2m_dewpoint_temperature", 280.0), ("10m_u_component_of_wind", 1.0),
                 ("10m_v_component_of_wind", 2.0), ("land_sea_mask", 0.5),
                 ("geopotential_at_surface", 9.80665 * 100.0), ("snow_depth", 0.01),
                 ("soil_temperature_level_1", 285.0), ("volumetric_soil_water_layer_1", 0.3)):
        data[n] = (("time", "latitude", "longitude"), np.full(shp2, v, np.float32))
    sst = np.full(shp2, np.nan, np.float32)
    sst[:, :, :1100] = 295.0                       # ocean lon<275E; NaN (land) east of it
    data["sea_surface_temperature"] = (("time", "latitude", "longitude"), sst)
    data["sea_ice_cover"] = (("time", "latitude", "longitude"), np.full(shp2, np.nan, np.float32))
    return xr.Dataset(data, coords={"time": t, "level": lev, "latitude": lat, "longitude": lon})


def test_records_cover_hydrometeors_units_and_grid(tmp_path):
    recs = e5.records_for_time(_ds(), dt.datetime(2019, 11, 26, 21), (20.0, 55.0, -130.0, -60.0),
                               margin=5.0, min_level_hpa=10)
    by = {}
    for r in recs:
        by.setdefault(r.field, []).append(r)
    for f in ("QC", "QI", "QR", "QS", "TT", "UU", "VV", "SPECHUMD", "GHT"):
        assert f in by
    assert len(by["QR"]) == 4 and abs(by["QR"][0].slab.mean() - 3e-5) < 1e-9
    ght = [r for r in by["GHT"] if r.xlvl == 85000.0][0]
    assert abs(ght.slab.mean() - 1500.0) < 1e-3                     # geopotential / g
    assert abs(by["SOILHGT"][0].slab.mean() - 100.0) < 1e-3
    assert abs(by["SNOW"][0].slab.mean() - 10.0) < 1e-4             # m -> kg m-2
    assert by["SEAICE"][0].slab.max() == 0.0                         # NaN -> 0
    sst = by["SST"][0].slab
    assert np.isfinite(sst).all() and sst.min() == -1.0e30           # land NaN -> WPS missing
    assert sst.max() == pytest.approx(295.0)                         # ocean value untouched
    assert (sst[sst > -1e29] == 295.0).all()
    q2 = [r for r in by["SPECHUMD"] if r.xlvl == wi.SURFACE_PA][0].slab.mean()
    assert 0.0055 < q2 < 0.0065                                      # Td 280 K, 1000 hPa
    r0 = by["TT"][0]
    assert r0.startlat == pytest.approx(15.0) and r0.startlon == pytest.approx(-135.0)
    assert r0.deltalat == pytest.approx(0.25) and r0.deltalon == pytest.approx(0.25)
    assert r0.ny == len(np.arange(15.0, 60.25, 0.25)) and r0.nx == len(np.arange(-135.0, -54.75, 0.25))
    path = tmp_path / "ERA5:2019-11-26_21"
    wi.write_fields(path, recs)
    back = wi.read_fields(path)
    assert len(back) == len(recs) and back[0].hdate.startswith("2019-11-26_21:00:00")


def test_crop_across_dateline():
    lat = np.arange(90.0, -90.25, -0.25)
    lon = np.arange(0.0, 360.0, 0.25)
    li, lo, lv = e5.crop_indices(lat, lon, (-10.0, 10.0, 175.0, -175.0), margin=0.0)
    assert lv[0] == pytest.approx(175.0) and lv[-1] == pytest.approx(185.0)
    assert np.all(np.diff(lv) > 0)


def test_model_levels_to_pressure_is_exact_for_linear_in_lnp():
    a, b = e5.load_l137_ab()
    assert len(a) == 138 and a[0] == 0.0 and b[-1] == 1.0
    ps = np.full((2, 3), 100000.0)
    a_full, b_full = 0.5 * (a[:-1] + a[1:]), 0.5 * (b[:-1] + b[1:])
    lnp = np.log(a_full[:, None, None] + b_full[:, None, None] * ps[None])
    q = 1e-3 * (lnp - lnp.min())                      # linear in ln p
    targets = [50000.0, 85000.0, 92500.0]
    out = e5.model_levels_to_pressure(q, ps, a, b, targets)
    for i, pt in enumerate(targets):
        np.testing.assert_allclose(out[i], 1e-3 * (np.log(pt) - lnp.min()), rtol=1e-9)
    below = e5.model_levels_to_pressure(q, ps, a, b, [105000.0])   # below the surface
    np.testing.assert_allclose(below[0], q[-1])

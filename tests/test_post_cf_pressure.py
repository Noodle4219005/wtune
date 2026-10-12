"""post/cf_pressure.py (+ its recipe hook post/recipes/cf3d.py and `wtune post cf3d`): CF
metadata, isobaric axis, regular ascending lat/lon, model-top level dropping, below-ground
missing, exact bilinear reproduction of a field linear in native projected x/y, boundary-zone
exclusion, the QICE-missing path and file naming.

The synthetic wrfout is a Lambert-conformal grid whose XLAT/XLONG come from an INDEPENDENT pyproj
LCC on WRF's 6370 km sphere (not from wrf-python), so the native-grid fit is tested against a
consistent but separately built projection. wrf-python runs on it for real (no mocks): it
carries the variables getvar needs for tk/uvmet/wa/z/rh/pressure (same idea as
tests/test_post_extract.py's fixture).
"""
import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("wrf")
pyproj = pytest.importorskip("pyproj")
pytest.importorskip("scipy")

from wtune.post import cf_pressure as cp

NY, NX = 24, 30
DX = 12000.0
P_TOP = 20000.0
LCC = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=-98 +a=6370000 +b=6370000 +units=m"
# staggered sigma (1 = surface): a thin first layer puts the lowest mass level of the flat part
# below 1000 hPa, so 1000 hPa is valid there and missing only over the high terrain
SIGMA_STAG = np.concatenate([[1.0, 0.99], np.linspace(0.95, 0.0, 11)])
NZ = len(SIGMA_STAG) - 1
HIGH_COLS = NX // 3                        # west third: 1500 m plateau


def _native_lonlat():
    x = (np.arange(NX) - (NX - 1) / 2.0) * DX
    y = (np.arange(NY) - (NY - 1) / 2.0) * DX
    xx, yy = np.meshgrid(x, y)
    tr = pyproj.Transformer.from_crs(pyproj.CRS(LCC), "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(xx, yy)
    return xx, yy, lat, lon


def _write_wrfout(path, time_str="2005-08-28_06:00:00", qice=True):
    xx, yy, lat, lon = _native_lonlat()
    hgt = np.zeros((NY, NX))
    hgt[:, :HIGH_COLS] = 1500.0
    psfc = 103000.0 * np.exp(-hgt / 8000.0)
    pstag = P_TOP + (psfc[None] - P_TOP) * SIGMA_STAG[:, None, None]
    pmass = 0.5 * (pstag[1:] + pstag[:-1])
    zstag = hgt[None] + 8000.0 * np.log(psfc[None] / pstag[:-1].clip(min=1.0))
    zstag = np.concatenate([zstag, zstag[-1:] + 5000.0])     # top of the model, finite
    with netCDF4.Dataset(path, "w") as d:
        for n, s in [("Time", 1), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX),
                     ("south_north_stag", NY + 1), ("west_east_stag", NX + 1),
                     ("bottom_top", NZ), ("bottom_top_stag", NZ + 1)]:
            d.createDimension(n, s)
        d.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = \
            np.array([list(time_str)], dtype="S1")

        def var(name, dims, arr):
            d.createVariable(name, "f4", ("Time",) + dims)[:] = np.asarray(arr, "f4")[None]
        m2 = ("south_north", "west_east")
        m3 = ("bottom_top",) + m2
        s3 = ("bottom_top_stag",) + m2
        var("XLAT", m2, lat)
        var("XLONG", m2, lon)
        var("HGT", m2, hgt)
        var("PSFC", m2, psfc)
        var("P", m3, np.zeros_like(pmass))
        var("PB", m3, pmass)
        var("T", m3, np.zeros_like(pmass))                    # theta = 300 K
        var("QVAPOR", m3, np.full_like(pmass, 0.005))
        var("QCLOUD", m3, np.broadcast_to(1e-4 * (1.0 + xx / xx.max()), pmass.shape))
        if qice:
            var("QICE", m3, np.full_like(pmass, 2e-5))
        var("PH", s3, np.zeros_like(zstag))
        var("PHB", s3, zstag * 9.81)
        var("W", s3, np.full_like(zstag, 0.5))
        var("U", ("bottom_top", "south_north", "west_east_stag"), np.full((NZ, NY, NX + 1), 10.0))
        var("V", ("bottom_top", "south_north_stag", "west_east"), np.zeros((NZ, NY + 1, NX)))
        d.createVariable("P_TOP", "f4", ("Time",))[:] = [P_TOP]
        for k, v in dict(MAP_PROJ=1, TRUELAT1=30.0, TRUELAT2=60.0, STAND_LON=-98.0,
                         MOAD_CEN_LAT=38.0, CEN_LAT=38.0, CEN_LON=-98.0, POLE_LAT=90.0,
                         POLE_LON=0.0, DX=DX, DY=DX, P_TOP=P_TOP,
                         SIMULATION_START_DATE="2005-08-28_00:00:00",
                         TITLE=" OUTPUT FROM WRF V4.5.1 MODEL").items():
            d.setncattr(k, v)


def test_output_name_and_grid_spacing():
    name = cp.output_name(2, "2005-08-29_12:00:00")
    assert name == "wrf_d02_pl_20050829_1200.nc" and ":" not in name
    assert cp.grid_spacing_deg(12000.0) == 0.11
    assert cp.grid_spacing_deg(333.0) == 0.01          # floored, never a zero spacing


def test_bilinear_reproduces_field_linear_in_native_xy():
    crs = pyproj.CRS(LCC)
    xx, yy, lat, lon = _native_lonlat()
    x0, sx, y0, sy = cp.native_affine(lat, lon, crs)
    assert (x0, sx, y0, sy) == pytest.approx((xx[0, 0], DX, yy[0, 0], DX), abs=1.0)

    field = 3e-3 * xx - 2e-3 * yy + 50.0
    lat1d, lon1d = cp.target_grid(lat, lon, 0, 0.11)
    jj, ii = cp.regrid_indices(lat1d, lon1d, crs, (x0, sx, y0, sy), lat.shape, 0)
    out = cp.regrid(field, jj, ii)
    tx, ty = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        *np.meshgrid(lon1d, lat1d))
    expected = 3e-3 * tx - 2e-3 * ty + 50.0
    ok = np.isfinite(out)
    assert ok.sum() > 0.5 * ok.size
    np.testing.assert_allclose(out[ok], expected[ok], rtol=1e-5, atol=1e-3)


def test_target_grid_regular_ascending_and_boundary_zone_excluded():
    crs = pyproj.CRS(LCC)
    _xx, _yy, lat, lon = _native_lonlat()
    affine = cp.native_affine(lat, lon, crs)
    lat0, lon0 = cp.target_grid(lat, lon, 0, 0.11)
    lat5, lon5 = cp.target_grid(lat, lon, 5, 0.11)
    for a in (lat5, lon5):
        assert np.all(np.diff(a) > 0)
        np.testing.assert_allclose(np.diff(a), 0.11, atol=1e-6)
    assert len(lat5) < len(lat0) and len(lon5) < len(lon0)
    interior_lat, interior_lon = lat[5:-5, 5:-5], lon[5:-5, 5:-5]
    assert lat5[0] >= interior_lat.min() - 1e-6 and lat5[-1] <= interior_lat.max() + 1e-6
    assert lon5[0] >= interior_lon.min() - 1e-6 and lon5[-1] <= interior_lon.max() + 1e-6

    # every target point whose native index falls in the 5-cell zone is missing
    jj_all, ii_all = cp.regrid_indices(lat5, lon5, crs, affine, lat.shape, 0)
    jj5, _ii5 = cp.regrid_indices(lat5, lon5, crs, affine, lat.shape, 5)
    in_zone = np.isfinite(jj_all) & ((ii_all < 5) | (ii_all > NX - 6)
                                     | (jj_all < 5) | (jj_all > NY - 6))
    assert in_zone.any()
    assert np.isnan(jj5[in_zone]).all()
    assert np.isnan(cp.regrid(np.ones((NY, NX)), jj5, _ii5)[in_zone]).all()


def test_native_affine_refuses_a_grid_the_projection_does_not_describe():
    _xx, _yy, lat, lon = _native_lonlat()
    rng = np.random.default_rng(0)
    with pytest.raises(cp.CFPressureError, match="residual"):
        cp.native_affine(lat + rng.normal(0, 0.05, lat.shape), lon, pyproj.CRS(LCC))


def test_convert_frame_cf_layout(tmp_path):
    src = tmp_path / "wrfout_d01_2005-08-28_06:00:00"
    _write_wrfout(str(src))
    out = tmp_path / "o.nc"
    r = cp.convert_frame(str(src), str(out), domain=1, bdy_width=5, command="wtune post cf3d x")
    assert r["dropped_levels"] == [200.0, 150.0, 100.0]          # at/above P_TOP=200 hPa
    with netCDF4.Dataset(out) as d:
        assert d.Conventions == "CF-1.8" and d.source == "WRF V4.5.1"
        assert "wtune post cf3d x" in d.history and "institution" not in d.ncattrs()
        assert d.wtune_qice == "present"
        assert d.dimensions["time"].isunlimited() and len(d.dimensions["time"]) == 1
        t = d["time"]
        assert t.units == "hours since 2005-08-28 00:00:00" and t.calendar == "standard"
        assert t.standard_name == "time" and t.axis == "T" and t[0] == 6.0
        frt = d["forecast_reference_time"]
        assert frt.units == t.units and frt.standard_name == "forecast_reference_time"
        assert float(frt[...]) == 0.0
        p = d["isobaric"]
        assert (p.units, p.positive, p.axis, p.standard_name, p.long_name) == \
            ("hPa", "down", "Z", "air_pressure", "Isobaric surface")
        assert list(p[:]) == [lev for lev in cp.PRESSURE_LEVELS_HPA if lev > 200]
        for name, units, axis in (("lat", "degrees_north", "Y"), ("lon", "degrees_east", "X")):
            v = d[name]
            assert (v.units, v.axis, v.dimensions) == (units, axis, (name,))
            np.testing.assert_allclose(np.diff(v[:]), 0.11, atol=1e-6)

        std = dict(air_temperature="air_temperature", eastward_wind="eastward_wind",
                   northward_wind="northward_wind", upward_air_velocity="upward_air_velocity",
                   geopotential_height="geopotential_height",
                   relative_humidity="relative_humidity")
        for name, sn in std.items():
            v = d[name]
            assert v.standard_name == sn
            assert v.dimensions == ("time", "isobaric", "lat", "lon")
            assert v.dtype == np.float32 and v._FillValue == np.float32(9.96921e36)
            f = v.filters()
            assert f["zlib"] and f["complevel"] == 4
        assert d["relative_humidity"].units == "%"
        for name in ("QCLOUD", "QICE", "CLOUD_TOTAL"):
            assert "standard_name" not in d[name].ncattrs() and d[name].units == "kg kg-1"
        hs = d["surface_altitude"]
        assert hs.dimensions == ("lat", "lon") and hs.standard_name == "surface_altitude"

        # below ground: 1000 hPa is missing over the 1500 m plateau (regridded HGT exactly 1500
        # = all four native neighbours on it), valid over the flat part; 700 hPa valid on both
        hgt = hs[:]
        plateau, flat = (hgt == 1500.0).filled(False), (hgt == 0.0).filled(False)
        assert plateau.sum() > 5 and flat.sum() > 5
        t1000 = d["air_temperature"][0, 0]
        assert np.ma.count(t1000[plateau]) == 0 and np.ma.count(t1000[flat]) == flat.sum()
        t700 = d["air_temperature"][0, list(p[:]).index(700)]
        assert np.ma.count(t700[plateau]) == plateau.sum()

        qc, qi, tot = (d[n][0] for n in ("QCLOUD", "QICE", "CLOUD_TOTAL"))
        ok = ~np.ma.getmaskarray(tot)
        np.testing.assert_allclose(tot[ok], (qc + qi)[ok], rtol=1e-6)


def test_qice_missing_writes_cloud_total_as_qcloud(tmp_path):
    src = tmp_path / "wrfout_d01_2005-08-28_06:00:00"
    _write_wrfout(str(src), qice=False)
    out = tmp_path / "o.nc"
    r = cp.convert_frame(str(src), str(out), domain=1, bdy_width=5)
    assert r["has_qice"] is False
    with netCDF4.Dataset(out) as d:
        assert "QICE" not in d.variables
        assert d.wtune_qice.startswith("absent")
        assert "QCLOUD only" in d["CLOUD_TOTAL"].long_name
        np.testing.assert_array_equal(np.ma.filled(d["CLOUD_TOTAL"][:], 0),
                                      np.ma.filled(d["QCLOUD"][:], 0))


def _run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for t in ("2005-08-28_06:00:00", "2005-08-28_12:00:00"):
        _write_wrfout(str(run / ("wrfout_d01_%s" % t)), time_str=t)
    return run


def test_export_last_frame_by_default_and_every(tmp_path):
    run = _run_dir(tmp_path)
    res = cp.export(str(run), str(tmp_path / "out"), [1], 5)
    assert [os.path.relpath(r["path"], str(tmp_path / "out")) for r in res] == \
        [os.path.join("cf3d", "d01", "wrf_d01_pl_20050828_1200.nc")]
    res = cp.export(str(run), str(tmp_path / "out6"), [1], 5, every="6h")
    assert [r["valid_time"] for r in res] == ["2005-08-28_06:00:00", "2005-08-28_12:00:00"]
    res = cp.export(str(run), str(tmp_path / "out2"), [2], 5)
    assert res[0]["error"].startswith("no wrfout_d02_")


def test_recipe_hook_data_file_record_report_and_import_failure(tmp_path, monkeypatch):
    import sys
    from wtune.post import report
    from wtune.post.recipes import cf3d
    run = _run_dir(tmp_path)
    case_info = dict(domains=[dict(domain=1)], types=["common"], evidence={"common": "always"},
                     run_length_h=6.0, d01_extent_km=(348.0, 276.0))
    assert cf3d.cf_domains(dict(domains=[dict(domain=1), dict(domain=3)])) == [1, 3]
    out = tmp_path / "rec"
    recs = cf3d.figures(str(run), "base", str(out), case_info, 5)
    assert len(recs) == 1 and not recs[0]["missing"] and recs[0]["data_file"]
    assert recs[0]["id"] == "common_cf3d"
    assert recs[0]["size_bytes"] == os.path.getsize(recs[0]["path"])
    html_path = report.build_recipe_report(str(out), case_info, recs, [str(run)], ["base"], 6,
                                           {"common_cf3d": {}})
    text = open(html_path).read()
    assert "data file: <a href='cf3d/d01/wrf_d01_pl_20050828_1200.nc'>" in text and " KB)" in text

    monkeypatch.setitem(sys.modules, "wrf", None)      # import wrf -> ImportError
    recs = cf3d.figures(str(run), "base", str(out), case_info, 5)
    assert len(recs) == 1 and recs[0]["missing"] and recs[0]["id"] == "common_cf3d"


def test_cli_cf3d_writes_into_out(tmp_path):
    from wtune.post import cli
    run = _run_dir(tmp_path)
    out = tmp_path / "cli"
    assert cli.main(["cf3d", str(run), "--times", "2005082806", "--out", str(out)]) == 0
    path = out / "cf3d" / "d01" / "wrf_d01_pl_20050828_0600.nc"
    with netCDF4.Dataset(path) as d:
        assert "wtune post cf3d" in d.history and "--times 2005082806" in d.history

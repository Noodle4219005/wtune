"""Smoke test for post/extract.py against a minimal synthetic wrfout (same fixture shape
as tests/test_analysis_diagnostics.py's, since wrf.getvar needs the real diagnostic
inputs). Only exercises fields the fixture actually supports (slp, t2, wspd10/wdir10,
rain_acc/rain_rate, and one pressure-level field) -- mdbz/mcape/mcin need microphysics
mixing-ratio variables this minimal fixture does not carry, so they are not smoke-tested.
"""
import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
wrf = pytest.importorskip("wrf")

import post.extract as extract

NY, NX, NZ = 16, 20, 8
LAT0, LON0 = 24.0, -90.0
DLAT, DLON = 0.25, 0.25
TIMES = ["2005-08-25_00:00:00", "2005-08-25_06:00:00"]


def _write_wrfout(path, times, rainc_mm=0.0):
    nt = len(times)
    lat = (LAT0 + DLAT * np.arange(NY, dtype=np.float32))[:, None] * np.ones((1, NX), dtype=np.float32)
    lon = np.ones((NY, 1), dtype=np.float32) * (LON0 + DLON * np.arange(NX, dtype=np.float32))[None, :]

    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("Time", nt)
        d.createDimension("DateStrLen", 19)
        d.createDimension("south_north", NY)
        d.createDimension("west_east", NX)
        d.createDimension("south_north_stag", NY + 1)
        d.createDimension("west_east_stag", NX + 1)
        d.createDimension("bottom_top", NZ)
        d.createDimension("bottom_top_stag", NZ + 1)

        tvar = d.createVariable("Times", "S1", ("Time", "DateStrLen"))
        tvar[:] = np.array([list(t) for t in times], dtype="S1")

        d.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))[:] = np.repeat(lat[None], nt, axis=0)
        d.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))[:] = np.repeat(lon[None], nt, axis=0)

        base = np.linspace(100000.0, 50000.0, NZ, dtype=np.float32)
        pb = np.repeat(np.repeat(base[:, None, None], NY, axis=1), NX, axis=2)
        ph_stag = np.linspace(0.0, 6000.0, NZ + 1, dtype=np.float32) * 9.81
        phb = np.repeat(np.repeat(ph_stag[:, None, None], NY, axis=1), NX, axis=2)

        dims4 = ("Time", "bottom_top", "south_north", "west_east")
        dims4s = ("Time", "bottom_top_stag", "south_north", "west_east")
        d.createVariable("P", "f4", dims4)[:] = np.zeros((nt, NZ, NY, NX), dtype=np.float32)
        d.createVariable("PB", "f4", dims4)[:] = np.repeat(pb[None], nt, axis=0)
        d.createVariable("T", "f4", dims4)[:] = np.zeros((nt, NZ, NY, NX), dtype=np.float32)
        d.createVariable("QVAPOR", "f4", dims4)[:] = np.full((nt, NZ, NY, NX), 0.01, dtype=np.float32)
        d.createVariable("PH", "f4", dims4s)[:] = np.zeros((nt, NZ + 1, NY, NX), dtype=np.float32)
        d.createVariable("PHB", "f4", dims4s)[:] = np.repeat(phb[None], nt, axis=0)

        d.createVariable("PSFC", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.full((nt, NY, NX), 100000.0, dtype=np.float32)
        d.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.zeros((nt, NY, NX), dtype=np.float32)
        d.createVariable("U10", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.full((nt, NY, NX), 5.0, dtype=np.float32)
        d.createVariable("V10", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.zeros((nt, NY, NX), dtype=np.float32)
        d.createVariable("T2", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.full((nt, NY, NX), 290.0, dtype=np.float32)
        d.createVariable("Q2", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.full((nt, NY, NX), 0.008, dtype=np.float32)
        d.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))[:] = \
            np.zeros((nt, NZ, NY, NX + 1), dtype=np.float32)
        d.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))[:] = \
            np.zeros((nt, NZ, NY + 1, NX), dtype=np.float32)
        rainc = np.full((nt, NY, NX), rainc_mm, dtype=np.float32)
        d.createVariable("RAINC", "f4", ("Time", "south_north", "west_east"))[:] = rainc
        d.createVariable("RAINNC", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.zeros((nt, NY, NX), dtype=np.float32)

        d.setncattr("MAP_PROJ", 3)
        d.setncattr("TRUELAT1", 20.0)
        d.setncattr("TRUELAT2", 20.0)
        d.setncattr("STAND_LON", LON0)
        d.setncattr("CEN_LAT", LAT0 + DLAT * NY / 2.0)
        d.setncattr("CEN_LON", LON0)
        d.setncattr("MOAD_CEN_LAT", LAT0 + DLAT * NY / 2.0)
        d.setncattr("POLE_LAT", 90.0)
        d.setncattr("POLE_LON", 0.0)
        d.setncattr("DX", 25000.0)
        d.setncattr("DY", 25000.0)


@pytest.fixture
def run_dir(tmp_path):
    for mm, t in zip((0.0, 2.0), TIMES):
        _write_wrfout(tmp_path / ("wrfout_d01_%s" % t), [t], rainc_mm=mm)
    return tmp_path


def test_extract_points_csv(run_dir, tmp_path):
    out = tmp_path / "points.csv"
    result = extract.extract(str(run_dir), ["slp", "t2", "wspd10", "rain_acc", "rain_rate"],
                              str(out), points=[(LAT0 + 1.0, LON0 + 1.0, "P1")], workers=1)
    assert result["kind"] == "points"
    text = out.read_text()
    assert "# point" in text and "P1" in text
    assert "time" in text
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#") and not ln.startswith("time")]
    assert len(lines) == 2  # two frames


def test_extract_box_netcdf(run_dir, tmp_path):
    out = tmp_path / "box.nc"
    result = extract.extract(str(run_dir), ["slp", "z"], str(out),
                              box=(LAT0, LAT0 + 2.0, LON0, LON0 + 2.0), levels_hpa=[850.0],
                              workers=1)
    assert result["kind"] == "box"
    with netCDF4.Dataset(out) as ds:
        assert "slp" in ds.variables
        assert "z_850hPa" in ds.variables
        assert ds.variables["slp"].shape[0] == 2

#!/usr/bin/env python3
"""Tests for analysis/diagnostics.py (wrf-python extraction layer).

Builds a minimal but complete synthetic wrfout-shaped NetCDF -- the full 3-D state
(P, PB, T, QVAPOR, PH, PHB, PSFC, HGT, U10, V10, T2, Q2, U, V) plus the WRF map-projection
global attributes wrf.getvar needs (MAP_PROJ, TRUELAT1/2, STAND_LON, CEN_LAT/LON,
MOAD_CEN_LAT, POLE_LAT/LON, DX, DY) -- because wrf.getvar("slp")/("uvmet10") need the real
diagnostic inputs, not the vendored-formula subset predictor/verify_obs.py's synthetic
fixture uses. This exact variable/attribute set was verified by hand (wrf.getvar("slp"),
("uvmet10"), ("lat")/("lon") all running without error) before being written into this
test; see analysis/diagnostics.py's module docstring for the grid-separability contract
this module must uphold.

Requires wrf-python:
    pixi run --manifest-path envs/wtune/pixi.toml python -m pytest -q tests/test_analysis_diagnostics.py
"""
import os

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
wrf = pytest.importorskip("wrf")

import wtune.analysis.diagnostics as diag


# --------------------------------------------------------------------------------------
# grid separability -- pure numpy, no wrf-python needed
# --------------------------------------------------------------------------------------

def test_check_grid_separable_true_for_a_regular_lat_lon_grid():
    ny, nx = 10, 12
    lat2d = np.linspace(20.0, 35.0, ny)[:, None] * np.ones((1, nx))
    lon2d = np.ones((ny, 1)) * np.linspace(-100.0, -80.0, nx)[None, :]
    separable, max_lat_dev, max_lon_dev = diag.check_grid_separable(lat2d, lon2d)
    assert separable is True
    assert max_lat_dev == pytest.approx(0.0, abs=1e-12)
    assert max_lon_dev == pytest.approx(0.0, abs=1e-12)


def test_check_grid_separable_false_for_a_sheared_lambert_like_grid():
    ny, nx = 10, 12
    Y, X = np.meshgrid(np.linspace(20.0, 35.0, ny), np.linspace(-100.0, -80.0, nx), indexing="ij")
    lat2d = Y + 0.05 * X            # lat now depends on the column index too
    lon2d = X + 0.05 * Y            # lon now depends on the row index too
    separable, max_lat_dev, max_lon_dev = diag.check_grid_separable(lat2d, lon2d)
    assert separable is False
    assert max_lat_dev > 1e-3
    assert max_lon_dev > 1e-3


# --------------------------------------------------------------------------------------
# synthetic wrfout fixture
# --------------------------------------------------------------------------------------

NY, NX, NZ = 16, 20, 8
LAT0, LON0 = 24.0, -90.0
DLAT, DLON = 0.25, 0.25


def _write_synthetic_wrfout(path, times, dip_at=None, ring_at=None,
                            map_proj=3, non_separable=False):
    """A wrfout-shaped NetCDF wrf-python's getvar("slp")/("uvmet10")/("lat")/("lon") can
    read.

    dip_at:  {frame_index: (j, i, depth_pa)} -- the LOWEST MODEL LEVEL's perturbation
             pressure (P[...,0,j,i]) drops by depth_pa there, so SLP's per-frame minimum
             sits at (j, i) by construction. wrf.getvar("slp") hydrostatically extrapolates
             from the lowest model level's pressure (P+PB) through the T/QVAPOR/PH profile
             -- it does NOT read PSFC at all, confirmed by hand (a PSFC-only perturbation
             left slp completely flat).
    ring_at: {frame_index: (j, i, radius_cells, peak_ms)} -- a 10 m wind ring centred at
             (j, i).
    non_separable: write a sheared (Lambert-like) XLAT/XLONG instead of an axis-aligned
             one, to exercise the ncol/connectivity output path. MAP_PROJ is set
             accordingly but this is NOT a physically exact Lambert grid (see this file's
             module docstring) -- it only needs to make wrf.getvar run and to be
             numerically non-separable, both of which were checked by hand.
    """
    nt = len(times)
    if non_separable:
        Y, X = np.meshgrid(np.arange(NY, dtype=np.float32), np.arange(NX, dtype=np.float32),
                           indexing="ij")
        lat = (LAT0 + DLAT * Y + 0.02 * DLON * X).astype(np.float32)
        lon = (LON0 + DLON * X + 0.02 * DLAT * Y).astype(np.float32)
    else:
        lat = (LAT0 + DLAT * np.arange(NY, dtype=np.float32))[:, None] * np.ones(
            (1, NX), dtype=np.float32)
        lon = np.ones((NY, 1), dtype=np.float32) * (
            LON0 + DLON * np.arange(NX, dtype=np.float32))[None, :]

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

        d.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.repeat(lat[None], nt, axis=0)
        d.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.repeat(lon[None], nt, axis=0)

        base = np.linspace(100000.0, 50000.0, NZ, dtype=np.float32)
        pb = np.repeat(np.repeat(base[:, None, None], NY, axis=1), NX, axis=2)
        ph_stag = (np.linspace(0.0, 6000.0, NZ + 1, dtype=np.float32) * 9.81)
        phb = np.repeat(np.repeat(ph_stag[:, None, None], NY, axis=1), NX, axis=2)

        dims4 = ("Time", "bottom_top", "south_north", "west_east")
        dims4s = ("Time", "bottom_top_stag", "south_north", "west_east")
        p = np.zeros((nt, NZ, NY, NX), dtype=np.float32)
        Yi0, Xi0 = np.mgrid[0:NY, 0:NX]
        if dip_at:
            for t, (j, i, depth) in dip_at.items():
                d2 = (Yi0 - j) ** 2 + (Xi0 - i) ** 2
                p[t, 0] -= (depth * np.exp(-d2 / 6.0)).astype(np.float32)
        d.createVariable("P", "f4", dims4)[:] = p
        d.createVariable("PB", "f4", dims4)[:] = np.repeat(pb[None], nt, axis=0)
        d.createVariable("T", "f4", dims4)[:] = np.zeros((nt, NZ, NY, NX), dtype=np.float32)
        d.createVariable("QVAPOR", "f4", dims4)[:] = np.full(
            (nt, NZ, NY, NX), 0.01, dtype=np.float32)
        d.createVariable("PH", "f4", dims4s)[:] = np.zeros((nt, NZ + 1, NY, NX), dtype=np.float32)
        d.createVariable("PHB", "f4", dims4s)[:] = np.repeat(phb[None], nt, axis=0)

        psfc = np.full((nt, NY, NX), 100000.0, dtype=np.float32)
        u10 = np.zeros((nt, NY, NX), dtype=np.float32)
        v10 = np.zeros((nt, NY, NX), dtype=np.float32)
        Yi, Xi = np.mgrid[0:NY, 0:NX]
        for t in range(nt):
            if ring_at and t in ring_at:
                j, i, radius, peak = ring_at[t]
                d2 = (Yi - j) ** 2 + (Xi - i) ** 2
                mag = (peak * np.exp(-((np.sqrt(d2) - radius) ** 2) / 2.0)).astype(np.float32)
                u10[t] = mag
                v10[t] = 0.0
        d.createVariable("PSFC", "f4", ("Time", "south_north", "west_east"))[:] = psfc
        d.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.zeros((nt, NY, NX), dtype=np.float32)
        d.createVariable("U10", "f4", ("Time", "south_north", "west_east"))[:] = u10
        d.createVariable("V10", "f4", ("Time", "south_north", "west_east"))[:] = v10
        d.createVariable("T2", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.full((nt, NY, NX), 290.0, dtype=np.float32)
        d.createVariable("Q2", "f4", ("Time", "south_north", "west_east"))[:] = \
            np.full((nt, NY, NX), 0.008, dtype=np.float32)
        d.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))[:] = \
            np.zeros((nt, NZ, NY, NX + 1), dtype=np.float32)
        d.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))[:] = \
            np.zeros((nt, NZ, NY + 1, NX), dtype=np.float32)

        d.setncattr("MAP_PROJ", map_proj)
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
    return lat, lon


TIMES = ["2005-08-25_00:00:00", "2005-08-25_06:00:00"]


# --------------------------------------------------------------------------------------
# separable grid end to end
# --------------------------------------------------------------------------------------

def test_wrfout_to_tc_fields_separable_grid(tmp_path):
    dip_at = {0: (5, 6, 3000.0), 1: (7, 9, 3000.0)}       # the low moves between frames
    ring_at = {0: (5, 6, 3, 40.0), 1: (7, 9, 3, 40.0)}
    lat, lon = _write_synthetic_wrfout(
        tmp_path / "wrfout_d01_2005-08-25_00:00:00", TIMES, dip_at=dip_at, ring_at=ring_at)

    out_nc = tmp_path / "fields.nc"
    meta = diag.wrfout_to_tc_fields(str(tmp_path / "wrfout_d01_2005-08-25_00:00:00"),
                                    str(out_nc), variables=["slp", "wspd10"])

    assert meta["grid_type"] == "separable"
    assert meta["separable"] is True
    assert meta["n_times"] == 2
    assert meta["map_proj_name"] == "Mercator"

    with netCDF4.Dataset(out_nc) as ds:
        assert "lat" in ds.dimensions and "lon" in ds.dimensions
        assert ds.variables["lat"].ndim == 1 and ds.variables["lon"].ndim == 1
        np.testing.assert_allclose(ds.variables["lat"][:], lat[:, 0], atol=1e-4)
        np.testing.assert_allclose(ds.variables["lon"][:], lon[0, :], atol=1e-4)
        slp = np.asarray(ds.variables["slp"][:])
        wspd = np.asarray(ds.variables["wspd10"][:])
        assert slp.shape == (2, NY, NX)

        for t, (j, i, _) in dip_at.items():
            jj, ii = np.unravel_index(int(np.argmin(slp[t])), slp[t].shape)
            assert (jj, ii) == (j, i)
            # roughly hPa-scale and well below the undisturbed background
            assert slp[t, j, i] < slp[t, 0, 0] - 10.0

        for t, (j, i, radius, peak) in ring_at.items():
            assert float(np.max(wspd[t])) == pytest.approx(peak, rel=1e-3)


def test_wrfout_to_tc_fields_requires_slp_even_if_not_requested(tmp_path):
    lat, lon = _write_synthetic_wrfout(
        tmp_path / "wrfout_d01_2005-08-25_00:00:00", TIMES[:1])
    out_nc = tmp_path / "fields.nc"
    meta = diag.wrfout_to_tc_fields(str(tmp_path / "wrfout_d01_2005-08-25_00:00:00"),
                                    str(out_nc), variables=["wspd10"])
    with netCDF4.Dataset(out_nc) as ds:
        assert "slp" in ds.variables         # always included: DetectNodes needs it


def test_expand_wrfout_raises_when_nothing_matches(tmp_path):
    with pytest.raises(diag.DiagnosticsError):
        diag.expand_wrfout(str(tmp_path / "no_such_file_*"))


# --------------------------------------------------------------------------------------
# non-separable grid -> flattened ncol layout
# --------------------------------------------------------------------------------------

def test_wrfout_to_tc_fields_nonseparable_grid_writes_ncol_layout(tmp_path):
    lat2d, lon2d = _write_synthetic_wrfout(
        tmp_path / "wrfout_d01_2005-08-25_00:00:00", TIMES[:1],
        map_proj=1, non_separable=True)

    out_nc = tmp_path / "fields_curv.nc"
    meta = diag.wrfout_to_tc_fields(str(tmp_path / "wrfout_d01_2005-08-25_00:00:00"),
                                    str(out_nc), variables=["slp"])

    assert meta["grid_type"] == "ncol"
    assert meta["separable"] is False
    assert meta["max_lat_dev_deg"] > 1e-3 or meta["max_lon_dev_deg"] > 1e-3
    assert meta["map_proj_name"] == "Lambert conformal"

    with netCDF4.Dataset(out_nc) as ds:
        assert "ncol" in ds.dimensions
        assert ds.dimensions["ncol"].size == NY * NX
        assert ds.variables["lat"].shape == (NY * NX,)
        np.testing.assert_allclose(ds.variables["lat"][:], lat2d.reshape(-1), atol=1e-4)
        np.testing.assert_allclose(ds.variables["lon"][:], lon2d.reshape(-1), atol=1e-4)
        assert ds.variables["slp"].shape == (1, NY * NX)

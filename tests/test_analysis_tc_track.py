#!/usr/bin/env python3
"""Tests for analysis/tc_track.py (TempestExtremes DetectNodes+StitchNodes layer).

Exercises the TempestExtremes layer against a pre-made "fields.nc" written directly in
analysis.diagnostics' own output schema (see that module's wrfout_to_tc_fields), rather
than through wrf-python -- the wrf-python extraction itself is covered separately in
test_analysis_diagnostics.py. This split is explicitly sanctioned by the task: build the
synthetic file with the full WRF state for wrf-python, OR test the TempestExtremes layer
with a pre-made fields file and the wrf-python layer separately.

The central case (test_tracker_follows_vortex_naive_jumps_to_spurious_low) is the whole
point of this package: a moving low is tracked continuously across 4 frames (0/6/12/18h)
while a deeper, disconnected, single-frame low appears far away at t=12h. The naive
method (global argmin of SLP, no continuity -- predictor/verify_obs.py's storm_centres())
jumps to it because it is deeper; StitchNodes' --range continuity test rejects the jump
and keeps the track on the real, continuously-moving vortex. The exact grid/field
construction and TempestExtremes parameters below were verified by hand against
tempest-extremes 2.4.2 (conda-forge nompi build 103) before being written into this test;
see analysis/tc_track.py's module docstring for the underlying CLI facts (bool flags are
presence-only, --regional is mandatory for a limited-area domain, etc).

Requires DetectNodes/StitchNodes/GenerateConnectivityFile on PATH:
    pixi run --manifest-path envs/wtune/pixi.toml python -m pytest -q tests/test_analysis_tc_track.py
"""
import os
import shutil

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

import wtune.analysis.tc_track as tc_track

pytestmark = pytest.mark.skipif(
    shutil.which("DetectNodes") is None,
    reason="TempestExtremes (DetectNodes) not on PATH -- run under envs/wtune/pixi.toml")

NY, NX = 30, 36
LAT0, LON0 = 20.0, 260.0
DLAT, DLON = 0.5172413793103449, 0.5714285714285714   # (35-20)/29, (280-260)/35 -- matches
                                                       # the manually-verified /tmp scratch grid


def _write_fields_nc(path, nt=4, dt_hours=6.0, grid_type="separable"):
    """A fields.nc in analysis.diagnostics' own schema: a vortex moving in INDEX space
    (cy, cx) = (8 + 3k, 6 + 4k), a wind ring peaking exactly at grid distance-squared 9
    around it (so the analytic ring maximum lands exactly on a lattice point, giving an
    exact, non-interpolated expected value), and a deeper, stationary, single-frame
    spurious low at (25, 30) appearing only at k == 2 (t = 12h).
    """
    lat = np.linspace(LAT0, LAT0 + DLAT * (NY - 1), NY)
    lon = np.linspace(LON0, LON0 + DLON * (NX - 1), NX)
    Yi, Xi = np.mgrid[0:NY, 0:NX]

    centres = []
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", None)
        ds.createDimension("lat", NY)
        ds.createDimension("lon", NX)
        tvar = ds.createVariable("time", "f8", ("time",))
        tvar.units = "hours since 1970-01-01 00:00:00"
        tvar.calendar = "standard"
        tvar[:] = np.arange(nt) * dt_hours
        latv = ds.createVariable("lat", "f8", ("lat",))
        lonv = ds.createVariable("lon", "f8", ("lon",))
        latv.units = "degrees_north"
        lonv.units = "degrees_east"
        latv[:] = lat
        lonv[:] = lon

        slpv = ds.createVariable("slp", "f8", ("time", "lat", "lon"))
        windv = ds.createVariable("wspd10", "f8", ("time", "lat", "lon"))
        for k in range(nt):
            cy, cx = 8 + 3 * k, 6 + 4 * k
            centres.append((cy, cx))
            d2 = (Yi - cy) ** 2 + (Xi - cx) ** 2
            field = 1010.0 - 40.0 * np.exp(-d2 / 15.0)          # centre minimum: 970.0 hPa
            if k == 2:
                d2b = (Yi - 25) ** 2 + (Xi - 30) ** 2
                field = np.minimum(field, 1010.0 - 45.0 * np.exp(-d2b / 10.0))   # 965.0 hPa
            slpv[k] = field
            windv[k] = 5.0 + 45.0 * np.exp(-((d2 - 9) ** 2) / 8.0)   # ring peak: 50.0 m/s

        ds.title = "synthetic fields.nc for test_analysis_tc_track"
        ds.grid_type = grid_type
        ds.map_proj_name = "Mercator (synthetic, separable)"
        ds.ny = NY
        ds.nx = NX

    return {"lat": lat, "lon": lon, "centres": centres}


def _params(**overrides):
    kwargs = dict(mintime_hours=12.0, maxgap_hours=6.0, stitch_range_deg=8.0)
    kwargs.update(overrides)
    return tc_track.TCTrackParams(**kwargs)


# --------------------------------------------------------------------------------------
# the central acceptance test
# --------------------------------------------------------------------------------------

def test_tracker_follows_vortex_naive_jumps_to_spurious_low(tmp_path):
    grid = _write_fields_nc(tmp_path / "fields.nc")
    tracks, provenance = tc_track.detect_and_stitch(
        tmp_path / "fields.nc", str(tmp_path / "te"), params=_params())

    assert provenance["tool"] == "TempestExtremes"
    assert provenance["detect_cmd"][0].endswith("DetectNodes")
    assert provenance["stitch_cmd"][0].endswith("StitchNodes")
    assert provenance["connectivity_cmd"] is None       # separable grid: no connectivity needed
    assert provenance["grid_type"] == "separable"

    # exactly one persistent track: the spurious low is a single disconnected frame and
    # never meets --mintime on its own.
    assert len(tracks) == 1
    track = tracks[0]
    assert len(track) == 4

    lat, lon, centres = grid["lat"], grid["lon"], grid["centres"]
    for pt, (cy, cx) in zip(track, centres):
        assert pt["lat"] == pytest.approx(lat[cy], abs=1e-6)
        assert pt["lon"] == pytest.approx(lon[cx], abs=1e-6)
        # pmin at the centre matches the Gaussian minimum by construction (970.0 hPa
        # everywhere except the deep-low frame, where the vortex's OWN minimum is still
        # 970.0 -- the spurious low is a separate, deeper cell that does not touch it).
        assert pt["pmin_hpa"] == pytest.approx(970.0, abs=1e-6)
        # vmax within wind_radius_deg equals the ring's exact analytic maximum.
        assert pt["vmax_ms"] == pytest.approx(50.0, abs=1e-6)

    # never jumps to the spurious low's position (25, 30) at t=12h (track index 2).
    spurious_lat, spurious_lon = lat[25], lon[30]
    jump_km = tc_track.great_circle_km(track[2]["lat"], track[2]["lon"],
                                       spurious_lat, spurious_lon)
    assert jump_km > 500.0

    # the naive method (global argmin of SLP, no continuity) DOES jump: read the raw
    # field back out and confirm the domain-wide minimum at t=12h sits at the spurious
    # low, not at the vortex.
    with netCDF4.Dataset(tmp_path / "fields.nc") as ds:
        slp_t2 = np.asarray(ds.variables["slp"][2])
    j, i = np.unravel_index(int(np.argmin(slp_t2)), slp_t2.shape)
    assert (j, i) == (25, 30)
    assert float(slp_t2[j, i]) == pytest.approx(965.0, abs=1e-6)
    naive_jump_km = tc_track.great_circle_km(lat[j], lon[i], lat[centres[2][0]], lon[centres[2][1]])
    assert naive_jump_km > 500.0            # the naive centre is nowhere near the real vortex


def test_detectnodes_in_time_slices_matches_one_run(tmp_path, monkeypatch):
    _write_fields_nc(tmp_path / "fields.nc", nt=8)
    monkeypatch.setattr(tc_track, "_detect_workers", lambda n: 1)
    one, p1 = tc_track.detect_and_stitch(tmp_path / "fields.nc", str(tmp_path / "one"),
                                         params=_params())
    monkeypatch.setattr(tc_track, "_detect_workers", lambda n: 3)
    three, p3 = tc_track.detect_and_stitch(tmp_path / "fields.nc", str(tmp_path / "three"),
                                           params=_params())
    assert (p1["detect_parts"], p3["detect_parts"]) == (1, 3)
    with open(p1["nodes_file"]) as a, open(p3["nodes_file"]) as b:
        assert a.read() == b.read()
    assert one == three
    assert sorted(os.listdir(tmp_path / "three")) == ["nodes.dat", "track.csv"]


def test_select_track_picks_the_track_nearest_the_first_guess(tmp_path):
    grid = _write_fields_nc(tmp_path / "fields.nc")
    tracks, _ = tc_track.detect_and_stitch(
        tmp_path / "fields.nc", str(tmp_path / "te"), params=_params())
    cy0, cx0 = grid["centres"][0]
    chosen = tc_track.select_track(tracks, grid["lat"][cy0], grid["lon"][cx0])
    assert chosen is not None
    assert chosen[0]["lat"] == pytest.approx(grid["lat"][cy0], abs=1e-6)

    # a wildly wrong first guess still returns SOME track when only one exists.
    chosen_far = tc_track.select_track(tracks, -60.0, 10.0)
    assert chosen_far is tracks[0]

    assert tc_track.select_track([], 0.0, 0.0) is None


# --------------------------------------------------------------------------------------
# compare_best_track
# --------------------------------------------------------------------------------------

def _hurdat2_like_rows(centres, lat, lon, t0):
    from datetime import timedelta
    rows = []
    for k, (cy, cx) in enumerate(centres):
        rows.append({
            "time": t0 + timedelta(hours=6 * k),
            "lat": float(lat[cy]) + 0.01,      # small, realistic offset from the model
            "lon": float(lon[cx]) - 0.01,
            "pmin_hpa": 970.0 - k,             # a slow deepening trend to check against
            "vmax_ms": 48.0 + k,
        })
    return rows


def test_compare_best_track_reports_small_tracker_error_and_deepening(tmp_path):
    from datetime import datetime
    grid = _write_fields_nc(tmp_path / "fields.nc")
    tracks, _ = tc_track.detect_and_stitch(
        tmp_path / "fields.nc", str(tmp_path / "te"), params=_params())
    track = tracks[0]
    t0 = datetime(1970, 1, 1, 0, 0, 0)      # matches _write_fields_nc's epoch-hours base
    best = _hurdat2_like_rows(grid["centres"], grid["lat"], grid["lon"], t0)

    result = tc_track.compare_best_track(track, best)
    assert result["matched"] == 4
    assert result["naive"] is None                 # no wrfout_paths given
    tracker_stats = result["tracker"]["track_error_km"]
    assert tracker_stats["n"] == 4
    assert tracker_stats["max_abs"] < 5.0            # only the ~0.01 deg synthetic offset
    assert result["tracker"]["deepening_hpa"] == pytest.approx(970.0 - 970.0)  # constructed flat
    assert result["obs_deepening_hpa"] == pytest.approx(3.0)   # 970 -> 967 over 4 points
    assert "vmax" in result["vmax_caveat"]
    for row in result["rows"]:
        assert row["naive_track_error_km"] is None   # omitted without wrfout_paths


def test_compare_best_track_raises_on_empty_track():
    with pytest.raises(tc_track.TCTrackError):
        tc_track.compare_best_track([], [])


# --------------------------------------------------------------------------------------
# non-separable (curvilinear) grid via GenerateConnectivityFile
# --------------------------------------------------------------------------------------

def _write_curvilinear_fields_nc(path, nt=1):
    """2-D, genuinely non-separable lat/lon (lat depends on i too, lon on j too),
    flattened onto `ncol` -- the layout analysis.diagnostics writes for a non-separable
    grid. One Gaussian low, no track continuity needed; this only exercises the
    GenerateConnectivityFile + --in_connect code path (see tc_track.py docstring).
    """
    ny, nx = 20, 24
    Y, X = np.meshgrid(np.linspace(20, 35, ny), np.linspace(-100, -80, nx), indexing="ij")
    lat2d = Y + 0.01 * X
    lon2d = X + 0.01 * Y
    Yi, Xi = np.mgrid[0:ny, 0:nx]
    cy, cx = ny // 2, nx // 2
    d2 = ((Yi - cy) ** 2 + (Xi - cx) ** 2)
    slp2d = 1010.0 - 30.0 * np.exp(-d2 / 20.0)
    wind2d = 5.0 + 40.0 * np.exp(-((d2 - 9) ** 2) / 10.0)

    with netCDF4.Dataset(path, "w") as ds:
        ncol = ny * nx
        ds.createDimension("time", None)
        ds.createDimension("ncol", ncol)
        tvar = ds.createVariable("time", "f8", ("time",))
        tvar.units = "hours since 1970-01-01 00:00:00"
        tvar.calendar = "standard"
        tvar[:] = [0.0, 1.0]                  # 2 frames, 1h apart -- StitchNodes rejects
                                               # any mintime > 0 for a single-frame "path"
        latv = ds.createVariable("lat", "f8", ("ncol",))
        lonv = ds.createVariable("lon", "f8", ("ncol",))
        latv[:] = lat2d.reshape(-1)
        lonv[:] = lon2d.reshape(-1)
        slpv = ds.createVariable("slp", "f8", ("time", "ncol"))
        windv = ds.createVariable("wspd10", "f8", ("time", "ncol"))
        slpv[0] = slp2d.reshape(-1)
        slpv[1] = slp2d.reshape(-1)           # stationary low: only tests connectivity,
        windv[0] = wind2d.reshape(-1)         # not track continuity (that is covered by
        windv[1] = wind2d.reshape(-1)         # the separable-grid moving-vortex test)
        ds.grid_type = "ncol"
        ds.map_proj_name = "Lambert conformal (synthetic curvilinear)"
    return lat2d, lon2d, (cy, cx)


def test_nonseparable_grid_uses_connectivity_and_finds_the_low(tmp_path):
    lat2d, lon2d, (cy, cx) = _write_curvilinear_fields_nc(tmp_path / "fields_curv.nc")
    tracks, provenance = tc_track.detect_and_stitch(
        tmp_path / "fields_curv.nc", str(tmp_path / "te_curv"),
        params=_params(mintime_hours=1.0, maxgap_hours=1.0))
    assert provenance["connectivity_cmd"] is not None
    assert provenance["connectivity_cmd"][0].endswith("GenerateConnectivityFile")
    assert provenance["grid_type"] == "ncol"
    assert len(tracks) >= 1
    pt = tracks[0][0]
    assert pt["lat"] == pytest.approx(lat2d[cy, cx], abs=1e-3)
    assert pt["lon"] == pytest.approx(lon2d[cy, cx], abs=1e-3)
    assert pt["pmin_hpa"] == pytest.approx(980.0, abs=0.5)

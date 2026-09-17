#!/usr/bin/env python3
"""predictor/tests/test_verify_obs.py -- pytest suite for predictor/verify_obs.py.

Run with:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-predictor-env/pixi.toml \\
        python -m pytest predictor/tests/test_verify_obs.py -v

That env has no netCDF4, so every test that needs a wrfout is guarded by
`pytest.importorskip("netCDF4")` and SKIPS there. To actually exercise them:
    pixi run --manifest-path /home/u5727520/wrf-stack/wtune-pixi-env-20260830/pixi.toml \\
        python -m pytest predictor/tests/test_verify_obs.py -v
The parsers, the metrics and the refusal logic run in both.

The synthetic-netCDF pattern (createDimension / "S1" Times / "f4" fields, all under
tmp_path) is copied from test_wtune_config_gate.py:18-54 and test_fidelity_gate.py:16-82,
the repo's two existing synthetic-wrfout builders.

The HURDAT2 excerpt below is REAL: bytes copied verbatim out of
/home/u5727520/wrf-bench/atlas/sc26p2/hurdat2-1851-2024-040425.txt
(sha256 d595ada8aaa64e402c26eeb80efb344b52ef420b1627d8c0ee6ee65f0729923b, fetched
2026-09-17 from https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt,
HTTP 200, 7,034,638 B), lines 46323-46332 and the AL122005 records for the SC26-P2
window. A hand-written imitation would not catch a real formatting quirk.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verify_obs  # noqa: E402

# --------------------------------------------------------------------------------------
# real HURDAT2 excerpt (verbatim)
# --------------------------------------------------------------------------------------

AL122005_EXCERPT = """\
AL112005,             IRENE,     40,
20050804, 1800,  , TD, 12.9N,  35.9W,  25, 1007,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999
AL122005,            KATRINA,     10,
20050826, 1800,  , HU, 24.9N,  82.6W,  85,  968,   75,   75,   55,   35,   60,   60,   35,   20,   20,   20,   15,   10, -999
20050827, 0000,  , HU, 24.6N,  83.3W,  90,  959,   90,   75,   75,   75,   60,   60,   40,   30,   25,   25,   20,   15, -999
20050827, 0600,  , HU, 24.4N,  84.0W,  95,  950,  130,   90,   90,  130,   60,   60,   45,   60,   35,   30,   30,   25, -999
20050827, 1200,  , HU, 24.4N,  84.7W, 100,  942,  130,   90,   90,  130,   60,   60,   45,   60,   35,   30,   30,   25, -999
20050827, 1800,  , HU, 24.5N,  85.3W, 100,  948,  130,   90,   90,  130,   60,   60,   45,   60,   35,   30,   30,   25, -999
20050828, 0000,  , HU, 24.8N,  85.9W, 100,  941,  150,  100,   90,  150,   70,   60,   50,   70,   40,   30,   30,   30, -999
20050828, 0600,  , HU, 25.2N,  86.7W, 125,  930,  150,  100,   90,  150,   70,   60,   50,   70,   40,   30,   30,   30, -999
20050828, 1200,  , HU, 25.7N,  87.7W, 145,  909,  200,  150,  150,  200,  100,   80,   70,  100,   50,   40,   40,   50, -999
20050828, 1800,  , HU, 26.3N,  88.6W, 150,  902,  200,  150,  150,  200,  100,   80,   70,  100,   50,   40,   40,   50, -999
20050829, 0000,  , HU, 27.2N,  89.2W, 140,  905,  200,  150,  150,  200,  100,   80,   70,  100,   50,   40,   40,   50, -999
AL132005,              LEE,      5,
20050828, 1800,  , TD, 31.7N,  60.1W,  25, 1010,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0, -999
"""


def _hurdat2_file(tmp_path, text=AL122005_EXCERPT):
    path = tmp_path / "hurdat2.txt"
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------------------
# great-circle distance
# --------------------------------------------------------------------------------------

def test_great_circle_one_degree_of_latitude():
    """One degree of meridian = R*pi/180 exactly on a sphere. Checks the units."""
    expected = verify_obs.EARTH_RADIUS_KM * math.pi / 180.0        # 111.1949... km
    assert verify_obs.great_circle_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(expected, rel=1e-12)
    # and away from the equator, since a meridian arc does not depend on latitude
    assert verify_obs.great_circle_km(45.0, 10.0, 46.0, 10.0) == pytest.approx(expected, rel=1e-12)


def test_great_circle_quarter_circumference_and_identity():
    quarter = verify_obs.EARTH_RADIUS_KM * math.pi / 2.0           # 10007.54... km
    assert verify_obs.great_circle_km(0.0, 0.0, 0.0, 90.0) == pytest.approx(quarter, rel=1e-12)
    assert verify_obs.great_circle_km(0.0, 0.0, 0.0, -90.0) == pytest.approx(quarter, rel=1e-12)
    assert verify_obs.great_circle_km(31.4, -89.6, 31.4, -89.6) == pytest.approx(0.0, abs=1e-9)


def test_great_circle_known_pair_jfk_lax():
    """JFK (40.6397N 73.7789W) -> LAX (33.9425N 118.4081W): 3974 km on R=6371 km.

    A published pair, not a self-consistent one: it is the check that would fail if the
    haversine were coded with the wrong argument order or a degrees/radians slip, which
    the analytic cases above would not catch (they are symmetric in lat or in lon).
    """
    d = verify_obs.great_circle_km(40.6397, -73.7789, 33.9425, -118.4081)
    assert d == pytest.approx(3974.0, abs=5.0)


def test_great_circle_is_symmetric_and_vectorises():
    a = verify_obs.great_circle_km(24.6, -83.3, 25.2, -86.7)
    b = verify_obs.great_circle_km(25.2, -86.7, 24.6, -83.3)
    assert a == pytest.approx(b, rel=1e-12)
    arr = verify_obs.great_circle_km(24.6, -83.3, np.array([24.6, 25.6]), np.array([-83.3, -83.3]))
    assert arr.shape == (2,)
    assert arr[0] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------------------
# HURDAT2 parser
# --------------------------------------------------------------------------------------

def test_hurdat2_parses_the_right_storm_block(tmp_path):
    points = verify_obs.parse_hurdat2(_hurdat2_file(tmp_path), "AL122005")
    assert len(points) == 10                       # the header declares 10
    first = points[0]
    assert first["time"].strftime("%Y-%m-%d %H:%M") == "2005-08-26 18:00"
    assert first["lat"] == pytest.approx(24.9)
    assert first["lon"] == pytest.approx(-82.6)    # 82.6W must come back NEGATIVE
    assert first["pmin_hpa"] == pytest.approx(968.0)
    assert first["status"] == "HU"


def test_hurdat2_does_not_bleed_into_neighbouring_storms(tmp_path):
    """IRENE sits before KATRINA and LEE after it; neither may appear in the result."""
    points = verify_obs.parse_hurdat2(_hurdat2_file(tmp_path), "AL122005")
    lats = [p["lat"] for p in points]
    assert 12.9 not in lats                        # IRENE's only record
    assert 31.7 not in lats                        # LEE's only record
    assert all(p["time"].year == 2005 and p["time"].month == 8 for p in points)
    assert points == sorted(points, key=lambda p: p["time"])


def test_hurdat2_katrina_peak_and_units(tmp_path):
    points = verify_obs.parse_hurdat2(_hurdat2_file(tmp_path), "al122005")   # case-insensitive
    peak = min(points, key=lambda p: p["pmin_hpa"])
    assert peak["pmin_hpa"] == pytest.approx(902.0)
    assert peak["time"].strftime("%Y-%m-%d %H:%M") == "2005-08-28 18:00"
    assert peak["vmax_kt"] == pytest.approx(150.0)
    # 150 kt -> m/s. This is the conversion a track report's wind error depends on.
    assert peak["vmax_ms"] == pytest.approx(150.0 * 0.514444, rel=1e-9)
    assert peak["vmax_ms"] == pytest.approx(77.17, abs=0.01)


def test_hurdat2_missing_values_become_none_not_minus_999(tmp_path):
    text = ("AL122005,            KATRINA,      1,\n"
            "20050827, 0000,  , HU, 24.6N,  83.3W, -999, -999,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    points = verify_obs.parse_hurdat2(_hurdat2_file(tmp_path, text), "AL122005")
    assert points[0]["pmin_hpa"] is None
    assert points[0]["vmax_kt"] is None and points[0]["vmax_ms"] is None


def test_hurdat2_unknown_storm_raises_with_a_usable_message(tmp_path):
    with pytest.raises(verify_obs.VerifyObsError) as exc:
        verify_obs.parse_hurdat2(_hurdat2_file(tmp_path), "AL992099")
    assert "AL992099" in str(exc.value)


def test_hurdat2_record_count_mismatch_is_an_error(tmp_path):
    text = ("AL122005,            KATRINA,     99,\n"
            "20050827, 0000,  , HU, 24.6N,  83.3W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    with pytest.raises(verify_obs.VerifyObsError) as exc:
        verify_obs.parse_hurdat2(_hurdat2_file(tmp_path, text), "AL122005")
    assert "99" in str(exc.value)


def test_hurdat2_missing_file_names_the_file_and_the_fetch_command(tmp_path):
    with pytest.raises(verify_obs.MissingObservationFile) as exc:
        verify_obs.parse_hurdat2(str(tmp_path / "nope.txt"), "AL122005")
    message = str(exc.value)
    assert "nope.txt" in message
    assert "nhc.noaa.gov" in message          # says how to get it; does NOT fetch it


# --------------------------------------------------------------------------------------
# station observation CSV
# --------------------------------------------------------------------------------------

def _obs_csv(tmp_path, rows, header="TIME,T2_C", name="obs.csv"):
    path = tmp_path / name
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


def test_read_station_obs_autodetects_columns_and_sorts(tmp_path):
    path = _obs_csv(tmp_path, ["2019-01-28 01:00,-12.5", "2019-01-28 00:00,-10.0"])
    obs = verify_obs.read_station_obs(path)
    assert [v for _, v in obs] == [-10.0, -12.5]            # sorted by time
    assert obs[0][0].strftime("%Y-%m-%d %H:%M") == "2019-01-28 00:00"


def test_read_station_obs_unit_conversion():
    assert verify_obs._to_celsius(273.15, "K") == pytest.approx(0.0)
    assert verify_obs._to_celsius(32.0, "F") == pytest.approx(0.0)
    assert verify_obs._to_celsius(-18.9, "C") == pytest.approx(-18.9)
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs._to_celsius(1.0, "furlongs")


def test_read_station_obs_kelvin_file_becomes_celsius(tmp_path):
    path = _obs_csv(tmp_path, ["2019-01-28 00:00,263.15"], header="TIME,T2")
    obs = verify_obs.read_station_obs(path, units="K")
    assert obs[0][1] == pytest.approx(-10.0)


def test_read_station_obs_unidentifiable_column_raises_and_lists_columns(tmp_path):
    path = _obs_csv(tmp_path, ["2019-01-28 00:00,-10.0"], header="when,howcold")
    with pytest.raises(verify_obs.VerifyObsError) as exc:
        verify_obs.read_station_obs(path)
    assert "howcold" in str(exc.value) or "when" in str(exc.value)
    # ... and works once told explicitly
    obs = verify_obs.read_station_obs(path, time_column="when", value_column="howcold")
    assert obs[0][1] == pytest.approx(-10.0)


def test_read_station_obs_blank_is_a_gap_but_garbage_is_an_error(tmp_path):
    ok = _obs_csv(tmp_path, ["2019-01-28 00:00,-10.0", "2019-01-28 01:00,", "2019-01-28 02:00,-9"])
    assert len(verify_obs.read_station_obs(ok)) == 2
    bad = _obs_csv(tmp_path, ["2019-01-28 00:00,cold"], name="bad.csv")
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs.read_station_obs(bad)


def test_read_station_obs_missing_file_refuses_and_forbids_a_substitute(tmp_path):
    with pytest.raises(verify_obs.MissingObservationFile) as exc:
        verify_obs.read_station_obs(str(tmp_path / "south_bend.csv"))
    assert "south_bend" in str(exc.value)
    assert "substitute" in str(exc.value).lower()


# --------------------------------------------------------------------------------------
# time alignment
# --------------------------------------------------------------------------------------

def _dt(text):
    from datetime import datetime
    return datetime.strptime(text, "%Y-%m-%d_%H:%M:%S")


def test_match_times_pairs_within_tolerance_only():
    obs = [_dt("2019-01-28_00:00:00"), _dt("2019-01-28_01:00:00"), _dt("2019-01-28_02:00:00")]
    model = [_dt("2019-01-28_00:00:00"), _dt("2019-01-28_01:00:30")]
    pairs = verify_obs.match_times(obs, model, tol_s=60.0)
    assert [(o, m) for o, m, _ in pairs] == [(0, 0), (1, 1)]
    assert pairs[1][2] == pytest.approx(30.0)
    assert verify_obs.match_times(obs, model, tol_s=10.0) == [(0, 0, 0.0)]


def test_match_times_never_reuses_a_model_frame():
    obs = [_dt("2019-01-28_00:00:00"), _dt("2019-01-28_00:00:30")]
    model = [_dt("2019-01-28_00:00:00")]
    pairs = verify_obs.match_times(obs, model, tol_s=120.0)
    assert len(pairs) == 1


def test_require_matches_refuses_below_the_fraction():
    assert verify_obs._require_matches(9, 10, 0.9, "things") == pytest.approx(0.9)
    with pytest.raises(verify_obs.InsufficientMatches) as exc:
        verify_obs._require_matches(8, 10, 0.9, "observation times")
    message = str(exc.value)
    assert "8 of 10" in message and "REFUSING" in message
    with pytest.raises(verify_obs.InsufficientMatches):
        verify_obs._require_matches(0, 0, 0.9, "observation times")


# --------------------------------------------------------------------------------------
# synthetic wrfout
# --------------------------------------------------------------------------------------

LAT0, LON0 = 24.0, -90.0          # grid origin; 1 degree spacing keeps the arithmetic checkable
NY, NX, NZ = 5, 6, 6


def _grid():
    lat = LAT0 + np.arange(NY, dtype=np.float32)[:, None] * np.ones(NX, dtype=np.float32)
    lon = LON0 + np.ones((NY, 1), dtype=np.float32) * np.arange(NX, dtype=np.float32)
    return lat.astype(np.float32), lon.astype(np.float32)


def _write_wrfout(path, times, *, t2=None, low_at=None, land_at=(), u10=None):
    """A wrfout-shaped netCDF: Times + XLAT/XLONG (+ T2, or the 3-D state for track mode).

    low_at: {frame_index: (j, i, depth_pa)} -- a surface pressure deficit, so the frame's
            minimum sea-level pressure is at (j, i) by construction.
    land_at: [(j, i), ...] -- LANDMASK = 1 there.
    """
    netCDF4 = pytest.importorskip("netCDF4")
    lat, lon = _grid()
    nt = len(times)
    with netCDF4.Dataset(path, "w") as data:
        data.createDimension("Time", nt)
        data.createDimension("DateStrLen", 19)
        data.createDimension("south_north", NY)
        data.createDimension("west_east", NX)
        data.createDimension("bottom_top", NZ)
        data.createDimension("bottom_top_stag", NZ + 1)

        tvar = data.createVariable("Times", "S1", ("Time", "DateStrLen"))
        tvar[:] = np.array([list(t) for t in times], dtype="S1")
        for name, field in (("XLAT", lat), ("XLONG", lon)):
            var = data.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            var[:] = np.repeat(field[None, :, :], nt, axis=0)

        landmask = np.zeros((NY, NX), dtype=np.float32)
        for j, i in land_at:
            landmask[j, i] = 1.0
        var = data.createVariable("LANDMASK", "f4", ("Time", "south_north", "west_east"))
        var[:] = np.repeat(landmask[None, :, :], nt, axis=0)

        if t2 is not None:
            var = data.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
            var[:] = np.asarray(t2, dtype=np.float32)

        if low_at is None:
            return
        # 3-D state. Pressure decreases linearly with level so slp_from_frame always finds
        # a level above p0-100 hPa; geopotential is a uniform 0..6000 m column so the only
        # thing that moves SLP between grid points is the surface pressure deficit.
        base = np.linspace(100000.0, 50000.0, NZ, dtype=np.float32)
        pb = np.repeat(np.repeat(base[:, None, None], NY, axis=1), NX, axis=2)
        p = np.zeros((nt, NZ, NY, NX), dtype=np.float32)
        for t in range(nt):
            j, i, depth = low_at[t]
            p[t, 0, j, i] = -float(depth)
        ph_stag = np.linspace(0.0, 6000.0, NZ + 1, dtype=np.float32) * 9.81
        phb = np.repeat(np.repeat(ph_stag[:, None, None], NY, axis=1), NX, axis=2)

        dims4 = ("Time", "bottom_top", "south_north", "west_east")
        data.createVariable("P", "f4", dims4)[:] = p
        data.createVariable("PB", "f4", dims4)[:] = np.repeat(pb[None], nt, axis=0)
        data.createVariable("T", "f4", dims4)[:] = np.zeros((nt, NZ, NY, NX), dtype=np.float32)
        data.createVariable("QVAPOR", "f4", dims4)[:] = np.full(
            (nt, NZ, NY, NX), 0.01, dtype=np.float32)
        dims4s = ("Time", "bottom_top_stag", "south_north", "west_east")
        data.createVariable("PH", "f4", dims4s)[:] = np.zeros(
            (nt, NZ + 1, NY, NX), dtype=np.float32)
        data.createVariable("PHB", "f4", dims4s)[:] = np.repeat(phb[None], nt, axis=0)

        if u10 is not None:
            data.createVariable("U10", "f4", ("Time", "south_north", "west_east"))[:] = \
                np.asarray(u10, dtype=np.float32)
            data.createVariable("V10", "f4", ("Time", "south_north", "west_east"))[:] = \
                np.zeros((nt, NY, NX), dtype=np.float32)


# --------------------------------------------------------------------------------------
# nearest-grid-point selection
# --------------------------------------------------------------------------------------

def test_nearest_index_picks_the_closest_cell_and_reports_the_true_distance():
    lat, lon = _grid()
    # 26.4 N, -87.1 E rounds to row 2 (26 N) and column 3 (-87 E)
    j, i, glat, glon, dist = verify_obs.nearest_index(lat, lon, 26.4, -87.1)
    assert (j, i) == (2, 3)
    assert glat == pytest.approx(26.0) and glon == pytest.approx(-87.0)
    expected = verify_obs.great_circle_km(26.4, -87.1, 26.0, -87.0)
    assert dist == pytest.approx(expected, rel=1e-9)
    # exactly on a grid point -> that point, zero distance
    j, i, _, _, dist = verify_obs.nearest_index(lat, lon, 25.0, -88.0)
    assert (j, i) == (1, 2) and dist == pytest.approx(0.0, abs=1e-9)


def test_station_series_uses_the_nearest_point_not_the_corner(tmp_path):
    pytest.importorskip("netCDF4")
    times = ["2019-01-28_00:00:00", "2019-01-28_01:00:00"]
    t2 = np.zeros((2, NY, NX), dtype=np.float32)
    t2[:, :, :] = 250.0
    t2[0, 2, 3] = 263.15                      # a marker only at the cell we expect
    t2[1, 2, 3] = 264.15
    path = str(tmp_path / "wrfout_d01_2019-01-28_00:00:00")
    _write_wrfout(path, times, t2=t2)
    series, meta = verify_obs.station_series(path, 26.4, -87.1)
    assert (meta["j"], meta["i"]) == (2, 3)
    assert [v for _, v in series] == [pytest.approx(263.15), pytest.approx(264.15)]
    assert meta["frames"] == 2 and meta["files"] == 1


# --------------------------------------------------------------------------------------
# station mode end to end
# --------------------------------------------------------------------------------------

def _station_run(tmp_path, n=6, model_c=None, obs_c=None, obs_offset_min=0):
    """A model file and an obs CSV over n hourly steps. Returns (wrfout_path, obs_path)."""
    from datetime import datetime, timedelta
    t0 = datetime(2019, 1, 28, 0, 0, 0)
    times = [(t0 + timedelta(hours=k)).strftime("%Y-%m-%d_%H:%M:%S") for k in range(n)]
    model_c = [-10.0] * n if model_c is None else model_c
    obs_c = [-10.0] * n if obs_c is None else obs_c
    t2 = np.full((n, NY, NX), 200.0, dtype=np.float32)
    for k, c in enumerate(model_c):
        t2[k, 2, 3] = c + 273.15               # written as KELVIN, the metric is Celsius
    path = str(tmp_path / "wrfout_d01_2019-01-28_00:00:00")
    _write_wrfout(path, times, t2=t2)
    rows = [(t0 + timedelta(hours=k, minutes=obs_offset_min)).strftime("%Y-%m-%d %H:%M")
            + f",{c}" for k, c in enumerate(obs_c)]
    return path, _obs_csv(tmp_path, rows)


def test_verify_station_converts_kelvin_to_celsius_and_scores_zero_on_a_perfect_match(tmp_path):
    pytest.importorskip("netCDF4")
    wrfout, obs = _station_run(tmp_path, model_c=[-10, -12, -14, -16, -18, -20],
                               obs_c=[-10, -12, -14, -16, -18, -20])
    res = verify_obs.verify_station(wrfout, obs, 26.4, -87.1)
    assert res["matched"] == 6
    assert res["rmse_c"] == pytest.approx(0.0, abs=1e-4)
    assert res["bias_c"] == pytest.approx(0.0, abs=1e-4)
    assert res["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert res["model_mean_c"] == pytest.approx(-15.0, abs=1e-4)


def test_verify_station_rmse_bias_and_correlation_are_the_textbook_values(tmp_path):
    pytest.importorskip("netCDF4")
    # model - obs = [+1, -1, +1, -1, +1, -1] -> bias 0, RMSE 1
    wrfout, obs = _station_run(tmp_path, model_c=[-9, -13, -13, -17, -17, -21],
                               obs_c=[-10, -12, -14, -16, -18, -20])
    res = verify_obs.verify_station(wrfout, obs, 26.4, -87.1)
    assert res["rmse_c"] == pytest.approx(1.0, abs=1e-3)
    assert res["bias_c"] == pytest.approx(0.0, abs=1e-3)
    assert res["mae_c"] == pytest.approx(1.0, abs=1e-3)
    assert verify_obs.accuracy_error(res) == pytest.approx(res["rmse_c"])


def test_verify_station_sign_of_the_bias_is_model_minus_obs(tmp_path):
    pytest.importorskip("netCDF4")
    wrfout, obs = _station_run(tmp_path, model_c=[-8] * 6, obs_c=[-10] * 6)
    res = verify_obs.verify_station(wrfout, obs, 26.4, -87.1)
    assert res["bias_c"] == pytest.approx(2.0, abs=1e-3)   # model warmer -> positive


def test_verify_station_refuses_a_deliberate_time_mismatch(tmp_path):
    """Obs on the half hour, model on the hour: nothing matches inside the tolerance.

    This is the failure the tool must NOT paper over. Interpolating or widening the window
    silently would produce a plausible RMSE from a comparison that never happened.
    """
    pytest.importorskip("netCDF4")
    wrfout, obs = _station_run(tmp_path, obs_offset_min=30)
    with pytest.raises(verify_obs.InsufficientMatches) as exc:
        verify_obs.verify_station(wrfout, obs, 26.4, -87.1, time_tol_s=60.0)
    assert "REFUSING" in str(exc.value)
    # the same data DOES verify once the tolerance honestly covers the offset
    res = verify_obs.verify_station(wrfout, obs, 26.4, -87.1, time_tol_s=1800.0)
    assert res["matched"] == 6


def test_verify_station_refuses_a_partial_match_and_reports_the_fraction(tmp_path):
    """4 of 6 obs times land on a model frame -> 66.7% < 90%, refuse."""
    pytest.importorskip("netCDF4")
    from datetime import datetime, timedelta
    t0 = datetime(2019, 1, 28, 0, 0, 0)
    times = [(t0 + timedelta(hours=k)).strftime("%Y-%m-%d_%H:%M:%S") for k in (0, 1, 2, 3)]
    t2 = np.full((4, NY, NX), 263.15, dtype=np.float32)
    wrfout = str(tmp_path / "wrfout_d01_2019-01-28_00:00:00")
    _write_wrfout(wrfout, times, t2=t2)
    rows = []
    for k in range(4):
        rows.append((t0 + timedelta(hours=k)).strftime("%Y-%m-%d %H:%M") + ",-10.0")
    for minutes in (17, 43):                         # inside the window, on no frame
        rows.append((t0 + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M") + ",-10.0")
    obs = _obs_csv(tmp_path, rows)
    with pytest.raises(verify_obs.InsufficientMatches) as exc:
        verify_obs.verify_station(wrfout, obs, 26.4, -87.1)
    assert "4 of 6" in str(exc.value)
    # a lower bar accepts it -- the threshold is the caller's decision, not a hidden default
    res = verify_obs.verify_station(wrfout, obs, 26.4, -87.1, min_match_frac=0.5)
    assert res["matched"] == 4
    assert res["matched_fraction"] == pytest.approx(4 / 6)


def test_verify_station_ignores_obs_outside_the_model_window(tmp_path):
    """Obs from before the run started are not "unmatched" -- they are out of scope."""
    pytest.importorskip("netCDF4")
    from datetime import datetime, timedelta
    t0 = datetime(2019, 1, 28, 0, 0, 0)
    times = [(t0 + timedelta(hours=k)).strftime("%Y-%m-%d_%H:%M:%S") for k in range(3)]
    t2 = np.full((3, NY, NX), 263.15, dtype=np.float32)
    wrfout = str(tmp_path / "wrfout_d01_2019-01-28_00:00:00")
    _write_wrfout(wrfout, times, t2=t2)
    rows = [(t0 - timedelta(hours=k)).strftime("%Y-%m-%d %H:%M") + ",-10.0" for k in (3, 2, 1)]
    rows += [(t0 + timedelta(hours=k)).strftime("%Y-%m-%d %H:%M") + ",-10.0" for k in range(3)]
    res = verify_obs.verify_station(wrfout, _obs_csv(tmp_path, rows), 26.4, -87.1)
    assert res["obs_points_total"] == 6
    assert res["obs_points_in_window"] == 3
    assert res["matched"] == 3 and res["matched_fraction"] == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# track mode
# --------------------------------------------------------------------------------------

def test_storm_centre_is_the_sea_level_pressure_minimum(tmp_path):
    """The centre lands on the deficit, and the deficit survives the sea-level reduction.

    Note the ABSOLUTE sea-level pressure here is ~1058 hPa, not ~1000: the synthetic column
    sits with its lowest mass level at 500 m, and reducing to sea level adds ~58 hPa. That is
    the reduction doing its job, and it is exactly why min(PSFC) is the wrong centre finder --
    so the assertion is on the DIFFERENCE between a flat field and one with a low, not on an
    absolute value that only holds for this toy column.
    """
    pytest.importorskip("netCDF4")
    times = ["2005-08-27_00:00:00"]
    flat = str(tmp_path / "wrfout_d01_flat")
    deep = str(tmp_path / "wrfout_d01_2005-08-27_00:00:00")
    _write_wrfout(flat, times, low_at={0: (0, 0, 0.0)})
    _write_wrfout(deep, times, low_at={0: (2, 3, 4000.0)})
    centres = verify_obs.storm_centres(deep, ocean_only=False)
    assert len(centres) == 1
    assert centres[0]["lat"] == pytest.approx(26.0)
    assert centres[0]["lon"] == pytest.approx(-87.0)
    ambient = verify_obs.storm_centres(flat, ocean_only=False)[0]["pmin_hpa"]
    # a 40 hPa surface deficit stays a ~40 hPa sea-level deficit on a uniform column
    assert 35.0 < ambient - centres[0]["pmin_hpa"] < 48.0


def test_ocean_only_rejects_a_deeper_low_over_land(tmp_path):
    """The reason min(PSFC) is wrong: high/land points can outscore the storm centre."""
    pytest.importorskip("netCDF4")
    wrfout = str(tmp_path / "wrfout_d01_2005-08-27_00:00:00")
    _write_wrfout(wrfout, ["2005-08-27_00:00:00"], low_at={0: (0, 0, 9000.0)},
                  land_at=[(0, 0)])
    with_land = verify_obs.storm_centres(wrfout, ocean_only=False)[0]
    assert (with_land["j"], with_land["i"]) == (0, 0)
    ocean = verify_obs.storm_centres(wrfout, ocean_only=True)[0]
    assert (ocean["j"], ocean["i"]) != (0, 0)


def _track_run(tmp_path, centres, times, u10_peak=None):
    wrfout = str(tmp_path / "wrfout_d01_2005-08-27_00:00:00")
    u10 = None
    if u10_peak is not None:
        u10 = np.zeros((len(times), NY, NX), dtype=np.float32)
        for k, peak in enumerate(u10_peak):
            u10[k, 1, 1] = peak
    _write_wrfout(wrfout, times, low_at=centres, u10=u10)
    return wrfout


def test_verify_track_scores_a_perfect_track_as_zero_error(tmp_path):
    """Model centres put exactly on the best-track positions -> track error 0."""
    pytest.importorskip("netCDF4")
    # best track 00Z: 24.6N 83.3W; 06Z: 24.4N 84.0W. Nearest grid cells are
    # (j=1,i=7?) -- the grid only spans 24..28 N / -90..-85 E, so instead place the
    # model low on grid points and check against a best track written to match them.
    times = ["2005-08-27_00:00:00", "2005-08-27_06:00:00"]
    wrfout = _track_run(tmp_path, {0: (1, 4, 4000.0), 1: (2, 3, 6000.0)}, times)
    text = ("AL122005,            KATRINA,      2,\n"
            "20050827, 0000,  , HU, 25.0N,  86.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n"
            "20050827, 0600,  , HU, 26.0N,  87.0W,   95,  950,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    res = verify_obs.verify_track(wrfout, _hurdat2_file(tmp_path, text), "AL122005")
    assert res["matched"] == 2
    assert res["track_error_km_mean"] == pytest.approx(0.0, abs=1e-6)
    assert verify_obs.accuracy_error(res) == pytest.approx(0.0, abs=1e-6)
    assert [r["lead_h"] for r in res["series"]] == [0.0, 6.0]


def test_verify_track_error_equals_the_great_circle_offset(tmp_path):
    """Model one grid row north of the best track -> exactly one degree of latitude."""
    pytest.importorskip("netCDF4")
    times = ["2005-08-27_00:00:00"]
    wrfout = _track_run(tmp_path, {0: (2, 3, 4000.0)}, times)     # 26.0 N, -87.0 E
    text = ("AL122005,            KATRINA,      1,\n"
            "20050827, 0000,  , HU, 25.0N,  87.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    res = verify_obs.verify_track(wrfout, _hurdat2_file(tmp_path, text), "AL122005",
                                  min_match_frac=1.0)
    one_degree = verify_obs.EARTH_RADIUS_KM * math.pi / 180.0
    assert res["series"][0]["track_error_km"] == pytest.approx(one_degree, rel=1e-4)


def test_verify_track_pressure_and_wind_errors_have_the_right_sign_and_units(tmp_path):
    pytest.importorskip("netCDF4")
    times = ["2005-08-27_00:00:00"]
    wrfout = _track_run(tmp_path, {0: (2, 3, 4000.0)}, times, u10_peak=[30.0])
    text = ("AL122005,            KATRINA,      1,\n"
            "20050827, 0000,  , HU, 26.0N,  87.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    res = verify_obs.verify_track(wrfout, _hurdat2_file(tmp_path, text), "AL122005")
    row = res["series"][0]
    # model is far weaker than a 959 hPa Katrina -> positive pressure error
    assert row["pmin_error_hpa"] == pytest.approx(row["model_pmin_hpa"] - 959.0, rel=1e-9)
    assert row["pmin_error_hpa"] > 0
    # 90 kt = 46.3 m/s; model peak 30 m/s -> negative wind error
    assert row["obs_vmax_ms"] == pytest.approx(90.0 * 0.514444, rel=1e-9)
    assert row["model_vmax_ms"] == pytest.approx(30.0, rel=1e-5)
    assert row["vmax_error_ms"] == pytest.approx(30.0 - 90.0 * 0.514444, rel=1e-4)


def test_verify_track_reports_deepening_not_only_absolute_pressure(tmp_path):
    """The discriminating quantity for Katrina: how much the storm deepened."""
    pytest.importorskip("netCDF4")
    times = ["2005-08-27_00:00:00", "2005-08-27_06:00:00"]
    wrfout = _track_run(tmp_path, {0: (2, 3, 1000.0), 1: (2, 3, 4000.0)}, times)
    text = ("AL122005,            KATRINA,      2,\n"
            "20050827, 0000,  , HU, 26.0N,  87.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n"
            "20050827, 0600,  , HU, 26.0N,  87.0W,   95,  950,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    res = verify_obs.verify_track(wrfout, _hurdat2_file(tmp_path, text), "AL122005")
    assert res["obs_deepening_hpa"] == pytest.approx(9.0)          # 959 -> 950
    assert res["model_deepening_hpa"] == pytest.approx(30.0, abs=1.5)   # 10 hPa -> 40 hPa
    assert res["obs_pmin_min_hpa"] == pytest.approx(950.0)


def test_verify_track_skips_off_hour_landfall_records_by_default(tmp_path):
    """HURDAT2 carries 'L' records at odd minutes; history_interval=60 has no frame there."""
    pytest.importorskip("netCDF4")
    # Two hourly frames 6 h apart, and a landfall record at 02:30 BETWEEN them -- inside the
    # model window (so the window filter cannot quietly drop it) but on no history frame.
    times = ["2005-08-27_00:00:00", "2005-08-27_06:00:00"]
    wrfout = _track_run(tmp_path, {0: (2, 3, 4000.0), 1: (2, 3, 4000.0)}, times)
    text = ("AL122005,            KATRINA,      3,\n"
            "20050827, 0000,  , HU, 26.0N,  87.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n"
            "20050827, 0230, L, HU, 26.0N,  87.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n"
            "20050827, 0600,  , HU, 26.0N,  87.0W,   95,  950,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    res = verify_obs.verify_track(wrfout, _hurdat2_file(tmp_path, text), "AL122005")
    assert res["best_track_points_total"] == 3
    assert res["best_track_points_in_window"] == 2 and res["matched"] == 2
    # counting the 02:30Z landfall record drops the match rate to 2/3 and must refuse
    with pytest.raises(verify_obs.InsufficientMatches) as exc:
        verify_obs.verify_track(wrfout, _hurdat2_file(tmp_path, text), "AL122005",
                                synoptic_only=False)
    assert "2 of 3" in str(exc.value)


def test_verify_track_needs_the_3d_state(tmp_path):
    """A history stream without P/PB/PH/PHB cannot give a sea-level pressure. Say so."""
    pytest.importorskip("netCDF4")
    wrfout = str(tmp_path / "wrfout_d01_2005-08-27_00:00:00")
    _write_wrfout(wrfout, ["2005-08-27_00:00:00"], t2=np.zeros((1, NY, NX), dtype=np.float32))
    with pytest.raises(verify_obs.VerifyObsError) as exc:
        verify_obs.storm_centres(wrfout)
    assert "sea level" in str(exc.value).lower() or "'P'" in str(exc.value)


# --------------------------------------------------------------------------------------
# the case-driven library API recommend.py uses
# --------------------------------------------------------------------------------------

def test_observations_spec_returns_none_when_a_case_declares_none():
    """A case with no observations is not an error -- the caller falls back to fidelity."""
    assert verify_obs.observations_spec({"prep_dir": "/x", "dt": 72}) is None


def test_observations_spec_validates_rather_than_silently_falling_back():
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs.observations_spec({"observations": {"kind": "vibes", "path": "/x"}})
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs.observations_spec({"observations": {"kind": "track"}})          # no path
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs.observations_spec({"observations": {"kind": "track", "path": "/x"}})
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs.observations_spec(
            {"observations": {"kind": "station", "path": "/x", "station": {"lat": 1.0}}})
    spec = verify_obs.observations_spec(
        {"observations": {"kind": "track", "path": "/x", "storm_id": "AL122005"}})
    assert spec["storm_id"] == "AL122005"


def test_verify_refuses_the_wrong_domain_when_the_case_names_one(tmp_path):
    """SC26-P1 scores d02; handing it d01 would look like a config effect, not a mistake."""
    pytest.importorskip("netCDF4")
    wrfout, obs = _station_run(tmp_path)           # the helper writes wrfout_d01_*
    spec = {"kind": "station", "path": obs, "domain": "d02",
            "station": {"lat": 26.4, "lon": -87.1}}
    with pytest.raises(verify_obs.VerifyObsError) as exc:
        verify_obs.verify(spec, wrfout)
    assert "d02" in str(exc.value)
    spec["domain"] = "d01"
    assert verify_obs.verify(spec, wrfout)["matched"] == 6


def test_score_case_returns_none_triplet_without_observations():
    assert verify_obs.score_case({"dt": 72}, ["/nonexistent"]) == (None, None, None)


def test_score_case_gives_recommend_an_error_and_a_provenance_string(tmp_path):
    pytest.importorskip("netCDF4")
    times = ["2005-08-27_00:00:00"]
    wrfout = _track_run(tmp_path, {0: (2, 3, 4000.0)}, times)
    text = ("AL122005,            KATRINA,      1,\n"
            "20050827, 0000,  , HU, 25.0N,  87.0W,   90,  959,    0,    0,    0,    0,"
            "    0,    0,    0,    0,    0,    0,    0,    0, -999\n")
    case = {"observations": {"kind": "track", "path": _hurdat2_file(tmp_path, text),
                             "storm_id": "AL122005"}}
    error, source, result = verify_obs.score_case(case, wrfout)
    assert error > 0                                   # lower is better, and it is a distance
    assert error == pytest.approx(result["track_error_km_mean"])
    assert "observations" in source and "track" in source


def test_accuracy_error_is_the_metric_the_problem_statement_scores():
    assert verify_obs.PRIMARY_METRIC == {"station": "rmse_c", "track": "track_error_km_mean"}
    assert verify_obs.accuracy_error({"kind": "station", "rmse_c": 2.284}) == pytest.approx(2.284)
    assert verify_obs.accuracy_error(
        {"kind": "track", "track_error_km_mean": 46.3}) == pytest.approx(46.3)
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs.accuracy_error({"kind": "track", "track_error_km_mean": float("nan")})


def test_expand_wrfout_refuses_an_empty_glob(tmp_path):
    with pytest.raises(verify_obs.VerifyObsError) as exc:
        verify_obs.expand_wrfout([str(tmp_path / "wrfout_d01_*")])
    assert "no wrfout files matched" in str(exc.value)


def test_cases_json_documents_the_observations_block():
    """The schema lives in cases.json's _comment; a reader must be able to find it there."""
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "campaign", "cases.json")
    comment = json.load(open(path))["_comment"]
    assert "observations" in comment
    for token in ("station", "track", "storm_id", "verify_obs.py"):
        assert token in comment


# --------------------------------------------------------------------------------------
# how recommend.py's accuracy objective consumes this
# --------------------------------------------------------------------------------------

def _candidate(template, tts=-5.0, nd=0.02):
    return {"template": template, "family": "radiation", "status": "ok",
            "predicted_time_to_solution_pct": tts, "predicted_fidelity_nd": nd,
            "predicted_fidelity_label": "SMALL-DRIFT", "time_to_solution_interval_pct": None}


def test_recommend_falls_back_to_model_vs_model_without_observations():
    """No obs_scores -> byte-for-byte the pre-existing behaviour: nd, split by direction."""
    recommend = pytest.importorskip("recommend")
    view = recommend.build_accuracy_view([_candidate("P01"), _candidate("P02", nd=0.05)], [])
    assert view["judged_against"] == recommend.MODEL_ERROR
    assert view["ranked_observed"] == []
    assert [e["template"] for e in view["ranked_cost_levers"]] == ["P01", "P02"]
    assert all(e["fidelity_error_kind"] == recommend.MODEL_ERROR for e in view["entries"])


def test_recommend_prefers_the_observation_score_over_the_regression():
    recommend = pytest.importorskip("recommend")
    obs = {"P02": (46.3, "measured vs observations (track, ...)")}
    view = recommend.build_accuracy_view(
        [_candidate("P01"), _candidate("P02", nd=0.05)], [], obs_scores=obs,
        observations={"kind": "track", "path": "/x/hurdat2.txt"})
    assert view["judged_against"] == recommend.OBSERVED_ERROR
    assert [e["template"] for e in view["ranked_observed"]] == ["P02"]
    entry = view["ranked_observed"][0]
    assert entry["fidelity_error"] == pytest.approx(46.3)       # the obs score, not nd=0.05
    assert entry["fidelity_error_kind"] == recommend.OBSERVED_ERROR
    assert "observations" in entry["fidelity_error_source"]
    # the un-scored candidate still goes through the old path, in its own block
    assert [e["template"] for e in view["ranked_cost_levers"]] == ["P01"]


def test_the_chaos_floor_never_suppresses_an_observation_score():
    """floor_ratio is a model-vs-model concept; a real skill gain must not be filtered by it."""
    recommend = pytest.importorskip("recommend")
    rows = []          # no measured floor for P01; inject one directly instead
    view = recommend.build_accuracy_view(
        [_candidate("P01")], rows, obs_scores={"P01": (2.24, "observations")},
        observations={"kind": "station", "path": "/x/obs.csv"})
    entry = view["entries"][0]
    assert entry["below_floor"] is False
    assert entry["pareto_excluded_reason"] != "below the chaos floor"


def test_observation_scores_for_case_is_empty_without_a_run_or_a_block():
    recommend = pytest.importorskip("recommend")
    assert recommend.observation_scores_for_case("sc26p2", None) == {}
    assert recommend.observation_scores_for_case("sc26p2", {}) == {}
    fake_cases = {"nocase": {"dt": 72}}
    assert recommend.case_observations("nocase", fake_cases) is None
    assert recommend.case_observations("absent", fake_cases) is None


def test_format_report_renders_without_blowing_up(tmp_path):
    pytest.importorskip("netCDF4")
    wrfout, obs = _station_run(tmp_path, model_c=[-9, -13, -13, -17, -17, -21])
    text = verify_obs.format_report(verify_obs.verify_station(wrfout, obs, 26.4, -87.1))
    assert "RMSE" in text and "mean bias" in text and "nearest cell" in text

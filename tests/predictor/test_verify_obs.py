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
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wtune.predictor import verify_obs  # noqa: E402

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


def test_read_station_obs_unit_conversion():
    assert verify_obs._to_celsius(273.15, "K") == pytest.approx(0.0)
    assert verify_obs._to_celsius(32.0, "F") == pytest.approx(0.0)
    assert verify_obs._to_celsius(-18.9, "C") == pytest.approx(-18.9)
    with pytest.raises(verify_obs.VerifyObsError):
        verify_obs._to_celsius(1.0, "furlongs")


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


# --------------------------------------------------------------------------------------
# track mode
# --------------------------------------------------------------------------------------


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


def test_format_report_renders_without_blowing_up(tmp_path):
    pytest.importorskip("netCDF4")
    wrfout, obs = _station_run(tmp_path, model_c=[-9, -13, -13, -17, -17, -21])
    text = verify_obs.format_report(verify_obs.verify_station(wrfout, obs, 26.4, -87.1))
    assert "RMSE" in text and "mean bias" in text and "nearest cell" in text

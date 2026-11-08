"""Finest-domain TC tracking (post/tc.py finest_domain_track/tracking_notes): per-time domain
choice by grid-index interior containment, parent-seeded nest track selection, fallback
reasons, and the three user-facing hints. TempestExtremes and wrf-python are replaced by
fakes; the real chain is exercised on scc (see the delegation report)."""
import json
from datetime import datetime, timedelta

import numpy as np
import pytest

import wtune.post.tc as tc
from wtune.post.recipes import tc_extra

T0 = datetime(2005, 8, 27, 0)


def pt(h, lat, lon, slp=980.0, wind=40.0):
    return dict(time=T0 + timedelta(hours=h), lat=lat, lon=lon, pmin_hpa=slp, vmax_ms=wind)


def frames(hours):
    return {(T0 + timedelta(hours=h)).strftime(tc.WRF_TIME_FMT): "/fake/%d" % h for h in hours}


def doms(n_hours=6, nest_hours=None, dx=(12.0, 4.0)):
    d = {1: dict(frames=frames(range(n_hours)), dx_km=dx[0], ny=100, nx=100, moving=False)}
    if len(dx) > 1:
        d[2] = dict(frames=frames(nest_hours if nest_hours is not None else range(n_hours)),
                    dx_km=dx[1], ny=100, nx=100, moving=False)
    return d


@pytest.fixture
def fake_nest(monkeypatch):
    """nest tracks = whatever the test stores in box['tracks']; interior = lon < box['east']."""
    box = dict(tracks=[], east=-80.0)
    monkeypatch.setattr(tc, "_tracks_for_domain", lambda *a, **k: box["tracks"])
    monkeypatch.setattr(tc, "in_domain_interior",
                        lambda frame, lat, lon, bdy: lon < box["east"])
    from wtune import analysis  # noqa: F401  -- must import for finest_domain_track's own guard
    return box


PARENT = [pt(h, 25.0, -85.0 + 0.2 * h, 985.0, 35.0) for h in range(6)]


def test_storm_leaves_nest_interior_falls_back_and_warns(fake_nest):
    fake_nest["east"] = -84.35          # storm at lon -84.4..-84.2 (h=3,4,5 -> -84.4, -84.2, -84.0)
    fake_nest["tracks"] = [[pt(h, 25.0, -85.0 + 0.2 * h, 970.0, 50.0) for h in range(6)]]
    recs, _ = tc.finest_domain_track("/r", PARENT, doms(), 1, "/w", 5)
    assert [r["domain"] for r in recs] == [2, 2, 2, 2, 1, 1]
    late = [r for r in recs if r["domain"] == 1]
    assert late and all(r["reason"] == "storm outside d02 interior" for r in late)
    assert all(r["pmin_hpa"] == 985.0 for r in late)      # parent value where fallback
    notes = tc.tracking_notes("m1", recs, doms())
    joined = "\n".join(notes)
    assert "fell back to coarser d01" in joined and "storm outside d02 interior" in joined
    assert "left the d02 interior" in joined                # hint (c)


def test_nest_frame_missing_at_some_times_is_a_fallback(fake_nest):
    fake_nest["tracks"] = [[pt(h, 25.0, -85.0 + 0.2 * h) for h in range(6)]]
    d = doms(nest_hours=[0, 1, 4, 5])
    recs, _ = tc.finest_domain_track("/r", PARENT, d, 1, "/w", 5)
    assert [r["domain"] for r in recs] == [2, 2, 1, 1, 2, 2]
    # d02 has no frame at h=2,3 so d01 IS the finest available there: no fallback claimed
    assert all(r["reason"] == "" for r in recs)
    assert tc.tracking_notes("m1", recs, d) == []


def test_nest_minimum_far_from_parent_is_rejected(fake_nest):
    fake_nest["tracks"] = [[pt(h, 29.0, -85.0 + 0.2 * h, 900.0, 80.0) for h in range(6)]]  # ~440 km off
    recs, _ = tc.finest_domain_track("/r", PARENT, doms(), 1, "/w", 5)
    assert all(r["domain"] == 1 for r in recs)


def test_nest_tracker_failure_keeps_parent_and_reports(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("DetectNodes failed\nstack")
    monkeypatch.setattr(tc, "_tracks_for_domain", boom)
    recs, info = tc.finest_domain_track("/r", PARENT, doms(), 1, "/w", 5)
    assert all(r["domain"] == 1 for r in recs)
    assert "DetectNodes failed" in info["skipped"][2]
    # the nest existed but was not used: the records say the finest available is d01 only if the
    # skip is honoured -- the user must still be told why
    notes = tc.tracking_notes("m1", recs, doms(), info["skipped"])
    assert any("d02 (4 km) was not used" in n for n in notes)


def test_moving_nest_is_skipped(fake_nest):
    d = doms()
    d[2]["moving"] = True
    recs, info = tc.finest_domain_track("/r", PARENT, d, 1, "/w", 5)
    assert 2 in info["skipped"] and all(r["domain"] == 1 for r in recs)


def test_no_nest_coarse_grid_hint_b():
    d = doms(dx=(12.0,))
    recs = [dict(p, domain=1, finest_available=1, reason="") for p in PARENT]
    notes = tc.tracking_notes("run", recs, d)
    assert len(notes) == 1 and "no nested domain" in notes[0] and "12 km" in notes[0]
    assert "grid-limited" in notes[0]


def test_ranges_compress_consecutive_hours():
    ts = [T0 + timedelta(hours=h) for h in (0, 1, 2, 5, 7, 8)]
    assert tc._ranges(ts, 1.0) == "08-27 00:00Z..08-27 02:00Z, 08-27 05:00Z, 08-27 07:00Z..08-27 08:00Z"


def test_in_domain_interior_uses_grid_index_and_bdy_width(tmp_path, monkeypatch):
    import netCDF4
    p = str(tmp_path / "wrfout_d02_2005-08-27_00:00:00")
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("west_east", 50)
        ds.createDimension("south_north", 40)

    class FakeWrf:
        @staticmethod
        def ll_to_xy(ds, lat, lon, meta=False, as_int=True):
            return np.array([int(round(lon)), int(round(lat))])   # x=lon, y=lat, by index
    monkeypatch.setattr(tc, "_wrf", lambda: FakeWrf)
    assert tc.in_domain_interior(p, 20, 25, 5)
    assert not tc.in_domain_interior(p, 20, 4, 5)        # x=4 is inside the 5-cell boundary zone
    assert tc.in_domain_interior(p, 20, 5, 5)
    assert not tc.in_domain_interior(p, 20, 45, 5)       # x=45 >= nx-5
    assert not tc.in_domain_interior(p, 35, 25, 5)       # y=35 >= ny-5
    assert not tc.in_domain_interior(p, -999, -999, 5)   # wrf-python's out-of-domain value


def test_tc_extra_annotates_figure_but_emits_no_missing_entry(tmp_path, monkeypatch):
    (tmp_path / "tracking_notes.json").write_text(json.dumps({"m1": ["m1: hint text"], "m2": []}))
    assert tc_extra.tracking_notes_for("m1", str(tmp_path)) == ["m1: hint text"]
    assert tc_extra.tracking_notes_for("m2", str(tmp_path)) == []
    assert tc_extra.tracking_notes_for("m3", str(tmp_path)) == []
    monkeypatch.setattr(tc_extra, "radius_time_plot", lambda *a, **k: str(tmp_path / "x.png"))
    figs = tc_extra.figures("/r", "m1", "/a", 1, str(tmp_path))
    assert [f["id"] for f in figs] == ["tc_radius_time_wind"] and not figs[0]["missing"]
    assert figs[0]["note"] == "m1: hint text"


def test_hourly_step_stats_flags_a_hopping_track():
    smooth = [dict(time=T0 + timedelta(hours=h), lat=25.0, lon=-85.0 + 0.1 * h) for h in range(12)]
    st = tc.hourly_step_stats(smooth)
    assert st["ratio"] == pytest.approx(1.0, abs=0.05) and st["n"] == 5
    hop = [dict(p) for p in smooth]
    hop[9]["lat"] += 0.3
    st2 = tc.hourly_step_stats(hop)
    assert st2["ratio"] > 2.5 and st2["max_at"] in (T0 + timedelta(hours=9), T0 + timedelta(hours=10))
    assert tc.hourly_step_stats(smooth[:1]) is None


def test_refine_mslp_reads_unsmoothed_min_near_smoothed_centre(tmp_path):
    import netCDF4
    p = str(tmp_path / "fields.nc")
    lat, lon = np.linspace(24, 26, 21), np.linspace(-86, -84, 21)
    slp = np.full((1, 21, 21), 1000.0)
    slp[0, 10, 10] = 990.0        # eye minimum at (25, -85)
    slp[0, 2, 2] = 950.0          # deeper but far away: must not be picked
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", None); ds.createDimension("lat", 21); ds.createDimension("lon", 21)
        ds.createVariable("time", "f8", ("time",))[:] = [(T0 - datetime(1970, 1, 1)).total_seconds() / 3600]
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        ds.createVariable("slp", "f8", ("time", "lat", "lon"))[:] = slp
    tracks = [[dict(time=T0, lat=25.02, lon=-84.98, pmin_hpa=997.0, vmax_ms=40.0)]]
    tc._refine_mslp(tracks, p)
    assert tracks[0][0]["pmin_hpa"] == 990.0 and tracks[0][0]["lat"] == 25.02


def test_smooth_slp_field_uses_cdo_and_keeps_other_variables(tmp_path):
    import shutil
    import netCDF4
    if shutil.which("cdo") is None:
        pytest.skip("cdo not on PATH")
    p, out = str(tmp_path / "fields.nc"), str(tmp_path / "s.nc")
    slp = np.full((1, 21, 21), 1000.0)
    slp[0, 10, 10] = 980.0
    with netCDF4.Dataset(p, "w") as ds:
        ds.createDimension("time", None); ds.createDimension("lat", 21); ds.createDimension("lon", 21)
        ds.createVariable("time", "f8", ("time",)).units = "hours since 1970-01-01 00:00:00"
        ds.variables["time"][:] = [0.0]
        ds.createVariable("lat", "f8", ("lat",))[:] = np.linspace(24, 26, 21)
        ds.createVariable("lon", "f8", ("lon",))[:] = np.linspace(-86, -84, 21)
        ds.variables["lat"].units, ds.variables["lon"].units = "degrees_north", "degrees_east"
        ds.createVariable("slp", "f8", ("time", "lat", "lon"))[:] = slp
        ds.createVariable("wspd10", "f8", ("time", "lat", "lon"))[:] = np.full((1, 21, 21), 7.0)
        ds.grid_type = "separable"
    tc.smooth_slp_field(p, out, dx_km=11.0, radius_km=30.0)
    with netCDF4.Dataset(out) as ds:
        assert 980.0 < float(ds.variables["slp"][0, 10, 10]) < 1000.0     # spike smoothed
        assert float(ds.variables["wspd10"][0, 10, 10]) == 7.0             # untouched
        assert ds.grid_type == "separable"


def test_rmw_fill_down_reaches_the_lowest_valid_level_only():
    plev = np.array([1000.0, 950.0, 900.0, 850.0])
    vt = np.array([[np.nan, 5.0], [np.nan, 6.0], [10.0, 7.0], [12.0, 8.0]])
    out = tc._fill_down_from_lowest_valid(vt, plev)
    assert list(out[:, 0]) == [10.0, 10.0, 10.0, 12.0]       # below-surface cells copy 900 hPa
    assert list(out[:, 1]) == [5.0, 6.0, 7.0, 8.0]           # column with data everywhere untouched
    assert np.isnan(vt[0, 0])                                # input not modified


def test_met_tc_pressure_levels_are_every_50_hpa():
    from wtune.post import met_tc
    assert met_tc.PRESSURE_LEVELS_HPA[0] == 1000 and met_tc.PRESSURE_LEVELS_HPA[-1] == 100
    assert len(met_tc.PRESSURE_LEVELS_HPA) == 19
    assert set(np.diff(met_tc.PRESSURE_LEVELS_HPA)) == {-50}


def _track(zig=0.0):
    pts = []
    for h in range(13):
        lat = 25.0 + (zig if h % 2 else 0.0)
        pts.append(dict(time=T0 + timedelta(hours=h), lat=lat, lon=-85.0 + 0.135 * h))   # ~13.6 km/h
    return pts


BT = [(T0 + timedelta(hours=h), 25.0, -85.0 + 0.135 * h) for h in (0, 6, 12)]


def test_step_stats_without_bdeck_and_format_line():
    st = tc.hourly_step_stats(_track())
    assert st["bt_speed_km_h"] is None and st["speed_ratio"] is None
    assert "no best-track speed" in tc.format_track_steps("m1", st)
    line = tc.format_track_steps("m1", tc.hourly_step_stats(_track(), bdeck_points=BT))
    assert "tortuosity 1.00" in line and "median/best-track 1.0" in line
    assert "not enough fixes" in tc.format_track_steps("m1", None)


def test_bdeck_fixes_one_per_time(tmp_path):
    from wtune.post import atcf
    rows = [dict(time=T0, tau_h=0, lat=25.0, lon=-85.0, vmax_kt=70, mslp_hpa=980,
                 radii_kt={34: (10, 10, 10, 10), 50: (5, 5, 5, 5)})]
    path = atcf.write_lines(rows, str(tmp_path / "b.dat"), basin="AL", cy="12", tech="BEST")
    assert len(tc._bdeck_fixes(path)) == 1 and tc._bdeck_fixes(None) == []


def test_zero_smoothing_skips_cdo(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(tc, "smooth_slp_field", lambda *a, **k: called.append(1))
    import wtune.analysis.diagnostics as diag
    import wtune.analysis.tc_track as tct
    monkeypatch.setattr(diag, "wrfout_to_tc_fields", lambda pat, out: None)
    seen = {}
    monkeypatch.setattr(tct, "detect_and_stitch", lambda f, w, params=None: (seen.setdefault("f", f) and [], None))
    tc._tracks_for_domain("/r", 1, str(tmp_path), smooth_km=0)
    assert not called and seen["f"].endswith("fields.nc")

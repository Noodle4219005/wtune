"""Tests for wtune_cost_drivers (decision 0060): synthetic data only, no network."""
import datetime

import numpy as np
import pytest

import wtune_cost_drivers as cd
import wtune_probe_plan as plan

D = datetime.datetime
RUN_START = D(2022, 6, 24, 0)
RUN_END = RUN_START + datetime.timedelta(hours=24)
HOURS = [RUN_START + datetime.timedelta(hours=h) for h in range(25)]


class FakeProvider:
    def __init__(self, name, series=None, err=None):
        self.name, self.series, self.err, self.calls = name, series or {}, err, 0

    def fetch(self, bounds, t_start, t_end, interval_h):
        self.calls += 1
        if self.err:
            raise cd.ProviderError(self.err)
        return self.series, {"source": self.name}


def convective_series(peak_h=10, sigma=2.0):
    out = {}
    for h, t in enumerate(HOURS):
        g = float(np.exp(-0.5 * ((h - peak_h) / sigma) ** 2))
        out[t] = dict(cloud_col=0.2 + 0.8 * g, precip_area=0.6 * g, cape_area=0.5 * g, wind_max=8.0)
    return out


def _cs(providers, **kw):
    return cd.build_cost_series(providers, (30.0, 40.0, -100.0, -90.0), RUN_START, RUN_END, **kw)


def _pm(h, s, r=1.0):
    return plan.PointMeasurement(at=RUN_START + datetime.timedelta(hours=h), reason="t", source="rebuild",
                                 s_normal=s, n_rad=1, r=r)


def _linear_cs():
    """cloud_col = hour index -> c is the z-score of 0..24 (strictly increasing)."""
    return cd.CostSeries(RUN_START, HOURS, {"cloud_col": [float(h) for h in range(25)]},
                         {"cloud_col": 1.0}, [], {})


# ---- selection ------------------------------------------------------------------------
def _plan(cs, **cfg_kw):
    cfg = dict(steps="auto", points="auto", max_points=5, budget_pct=None, sources=["rebuild"],
               points_by="cost-indicators")
    cfg.update(cfg_kw)
    return plan.plan_points({}, RUN_START, RUN_END, dt=60.0, stepra=15, spinup_s=0.0, cfg=cfg,
                            forcing_interval_h=1.0, indicators=cs)


def test_convective_peak_at_hour_10_is_selected_and_t0_kept():
    cs = _cs([FakeProvider("fake", convective_series(10))])
    assert cs.usable
    points, notes = _plan(cs)
    hrs = [(p.at - RUN_START).total_seconds() / 3600.0 for p in points]
    assert 0.0 in hrs                                       # t0
    assert any(5.0 <= h <= 11.0 for h in hrs)               # the peak
    assert any("peak" in p.reason for p in points)
    assert any("stratum" in p.reason for p in points)
    assert len(points) <= 5
    assert any("cost-indicators" in n for n in notes)


def test_option_off_golden_identical_to_pre_change_output():
    sp = [(lat, lon) for lat in (-10.0, 0.0, 10.0) for lon in (-10.0, 0.0, 10.0)]
    rs = D(2020, 6, 1, 6)
    cfg = dict(steps="auto", points="auto", max_points=5, budget_pct=None, sources=["rebuild"])
    golden = [(D(2020, 6, 1, 6), "t0"), (D(2020, 6, 1, 7), "day"), (D(2020, 6, 1, 13), "farthest"),
              (D(2020, 6, 1, 19), "night"), (D(2020, 6, 2, 5), "late")]   # captured at HEAD

    def run(cfg_, **kw):
        pts, notes = plan.plan_points({}, rs, rs + datetime.timedelta(hours=24), dt=60.0, stepra=15,
                                      spinup_s=0.0, cfg=cfg_, sample_points=sp,
                                      forcing_interval_h=1.0, **kw)
        return [(p.at, p.reason) for p in pts], notes
    assert run(cfg) == (golden, [])
    assert run(dict(cfg, points_by="time")) == (golden, [])
    cs = _cs([FakeProvider("fake", convective_series())])
    assert run(cfg, indicators=cs) == (golden, [])         # indicators ignored when mode is time


def test_select_windows_default_unchanged_with_cost_lookup_none():
    import coldstart_sample as cs_mod
    cands = cs_mod.candidate_times(D(2020, 6, 1, 6), D(2020, 6, 2, 6), 3600.0, 3600.0)
    base = cs_mod.select_windows(cands, 0.0, 3600.0, 5)
    assert cs_mod.select_windows(cands, 0.0, 3600.0, 5, cost_lookup=None) == base
    assert [r for _t, r in base] == ["t0", "late", "farthest", "farthest", "farthest"]


# ---- providers / fallback -------------------------------------------------------------------
def test_provider_fallback_order_and_per_indicator_fill():
    bad = FakeProvider("bad", err="boom")
    a = FakeProvider("a", {t: {"cloud_col": v["cloud_col"]} for t, v in convective_series().items()})
    b = FakeProvider("b", convective_series())
    c = FakeProvider("c", convective_series())
    cs = _cs([bad, a, b, c])
    assert cs.used["cloud_col"] == "a"                      # first provider that supplies it
    assert cs.used["precip_area"] == "b" and cs.used["wind_max"] == "b"
    assert c.calls == 0                                     # all weather indicators covered: stop
    st = {p["provider"]: p["status"] for p in cs.provenance}
    assert st == {"bad": "failed", "a": "ok", "b": "ok"}
    assert "boom" in cs.provenance[0]["error"]


def test_all_providers_fail_gives_time_based_selection_with_note():
    cs = _cs([FakeProvider("x", err="e1"), FakeProvider("y", err="e2")])
    assert not cs.usable
    assert cd.build_cost_series([], (0, 1, 0, 1), RUN_START, RUN_END).usable is False
    base, _n = _plan(None, points_by="time")
    pts, notes = _plan(cs)
    assert [(p.at, p.reason) for p in pts] == [(p.at, p.reason) for p in base]
    assert any("time-based" in n and "x" in n and "e1" in n for n in notes)
    pts2, notes2 = _plan(None)                              # no indicators supplied at all
    assert [(p.at, p.reason) for p in pts2] == [(p.at, p.reason) for p in base]
    assert any("time-based" in n for n in notes2)


def test_precip_lookup_wired_from_indicators_in_time_fallback():
    # cost proxy unusable (constant) but precip_area present -> precip_peak is finally chosen
    ser = {t: dict(precip_area=(0.9 if t.hour == 14 else 0.1)) for t in HOURS}
    cs = _cs([FakeProvider("p", ser)])
    assert cs.usable                                        # precip_area varies -> c exists
    cs.c = None                                             # force the fallback path
    pts, _n = _plan(cs, sources=["rebuild"])
    assert any(p.reason == "precip_peak" and p.at.hour in (13, 14) for p in pts)


def test_imerg_is_skipped_with_reason_never_credentials():
    cs = _cs(cd.default_providers(["imerg"]))
    assert not cs.usable
    assert "Earthdata" in cs.provenance[0]["error"]


def test_constant_indicator_dropped_from_c():
    cs = _cs([FakeProvider("fake", convective_series())])
    assert "wind_max" in cs.dropped                         # constant 8 m/s
    assert cs.weights["cloud_col"] == cs.weights["precip_area"] == 1.0   # microphysics 2.0 / 2


def test_cost_structure_weights_from_case_profile():
    w = cd.indicator_weights({"phases_ranked": [{"phase": "microphysics", "weight": 4.0},
                                                {"phase": "cumulus", "weight": 0.0}]})
    assert w["cloud_col"] == 2.0 and w["cape_area"] == 0.0 and w["sunlit"] == 1.0


def test_sunlit_computed_with_sample_points_and_excluded_from_c_noradiation():
    sp = [(lat, lon) for lat in (-10.0, 0.0, 10.0) for lon in (-10.0, 0.0, 10.0)]
    cs = _cs([FakeProvider("fake", convective_series())], sample_points=sp)
    assert "sunlit" in cs.indicators and cs.used["sunlit"].startswith("computed")
    assert cs.c != cs.c_noradiation


# ---- weighting ----------------------------------------------------------------------------
def test_stratum_weights_sum_to_one_and_stratified_matches_hand_computation():
    cs = _linear_cs()
    ms = [_pm(2, 1.0), _pm(8, 2.0), _pm(14, 3.0), _pm(20, 4.0)]
    w = cd.weighted_totals(cs, ms, RUN_START, RUN_END, 60.0, 15, None, 1.0, 1.0)
    assert sum(s["weight"] for s in w["strata"]) == pytest.approx(1.0)
    assert [s["hours"] for s in w["strata"]] == [6, 6, 6, 6]
    steps_n = 1440
    # hand: w_k = 6/24 each, s_k = 1,2,3,4, r == 1 -> N * sum w_k s_k
    assert w["stratified_compute_s"] == pytest.approx(steps_n * 0.25 * (1 + 2 + 3 + 4))


def test_stratified_empty_stratum_takes_nearest_point_and_unequal_points_average():
    cs = _linear_cs()
    ms = [_pm(2, 1.0), _pm(3, 3.0), _pm(19, 4.0)]            # K=3 strata of 8 h: [0-7],[8-15],[16-23]
    w = cd.weighted_totals(cs, ms, RUN_START, RUN_END, 60.0, 15, None, 1.0, 1.0)
    s = {x["k"]: x["s"] for x in w["strata"]}
    assert s[1] == pytest.approx(2.0)                        # mean of 1 and 3
    assert s[3] == pytest.approx(4.0)
    assert s[2] == pytest.approx(4.0)                        # empty: median c (hour 11.5) is nearer h19 than h3


def test_stratified_applies_radiation_treatment_like_the_integrator():
    cs = _linear_cs()
    ms = [_pm(12, 2.0)]                                      # K=1 -> flat s=2
    stepra = 15
    w = cd.weighted_totals(cs, ms, RUN_START, RUN_END, 60.0, stepra, None, 3.0, 5.0)
    n_rad = sum(1 for i in range(1, 1441) if i == 1 or i % stepra == 0)
    r_t = 4.0                                                # no sample_points: (r_night+r_day)/2
    assert w["stratified_compute_s"] == pytest.approx(1440 * 2.0 + n_rad * (r_t - 1.0) * 2.0)


def test_regression_exact_on_linear_case_r2_one():
    cs = _linear_cs()
    cvals = {h: cs.value_at(RUN_START + datetime.timedelta(hours=h)) for h in range(24)}
    ms = [_pm(h, 1.0 + 0.5 * cvals[h]) for h in (1, 7, 13, 22)]
    w = cd.weighted_totals(cs, ms, RUN_START, RUN_END, 60.0, 15, None, 1.0, 1.0)
    assert w["alpha"] == pytest.approx(1.0) and w["beta"] == pytest.approx(0.5)
    assert w["r2"] == pytest.approx(1.0)
    expected = sum(60 * (1.0 + 0.5 * cvals[h]) for h in range(24))
    assert w["regression_compute_s"] == pytest.approx(expected)


def test_regression_needs_three_points_and_clamps_negative_slope():
    cs = _linear_cs()
    two = cd.weighted_totals(cs, [_pm(1, 1.0), _pm(20, 2.0)], RUN_START, RUN_END, 60.0, 15,
                             None, 1.0, 1.0)
    assert two["regression_compute_s"] is None and two["stratified_compute_s"] is not None
    ms = [_pm(2, 4.0), _pm(10, 3.0), _pm(20, 2.0)]           # s falls as c rises
    w = cd.weighted_totals(cs, ms, RUN_START, RUN_END, 60.0, 15, None, 1.0, 1.0)
    assert w["beta"] == 0.0 and w["alpha"] == pytest.approx(3.0)
    assert w["regression_compute_s"] == pytest.approx(1440 * 3.0)
    assert w["r2"] == pytest.approx(0.0)


def test_combine_adds_totals_and_leaves_total_s_untouched():
    cs = _linear_cs()
    ms = [_pm(2, 1.0), _pm(8, 2.0), _pm(14, 3.0), _pm(20, 4.0)]
    base = plan.combine(ms, RUN_START, RUN_END, 60.0, 15)
    res = plan.combine(ms, RUN_START, RUN_END, 60.0, 15, cost_series=cs)
    for k, v in base.items():
        assert res[k] == v                                    # nothing existing changes
    assert "stratified_total_s" not in base
    assert res["stratified_total_s"] == pytest.approx(res["compute_total_s"] * 0 + 1440 * 2.5)
    assert res["regression_total_s"] is not None and res["regression_r2"] is not None
    ci = res["cost_indicators"]
    assert ci["sources"] == {} or "cloud_col" in ci["weights"]
    vals = [res["total_s"], res["stratified_total_s"], res["regression_total_s"]]
    assert ci["spread_pct"] == pytest.approx(100 * (max(vals) - min(vals)) / res["total_s"])
    unusable = plan.combine(ms, RUN_START, RUN_END, 60.0, 15, cost_series=_cs([]))
    assert "stratified_total_s" not in unusable


# ---- case forcing providers -----------------------------------------------------------------
def _write_intermediate(path, hdate, q=1.0e-4, u=3.0, v=4.0):
    from scenario import wps_intermediate as wi

    def rec(field, xlvl, val):
        return wi.Record(hdate=hdate, xfcst=0.0, map_source="TEST", field=field, units="x", desc="d",
                         xlvl=xlvl, nx=3, ny=2, iproj=0, startloc="SWCORNER", startlat=30.0,
                         startlon=-100.0, deltalat=1.0, deltalon=1.0,
                         earth_radius=wi.EARTH_RADIUS_KM, is_wind_earth_rel=True,
                         slab=np.full((2, 3), val, dtype=np.float32))
    recs = [rec("QC", p, q) for p in (100000.0, 85000.0, 70000.0)]
    recs += [rec("UU", wi.SURFACE_PA, u), rec("VV", wi.SURFACE_PA, v)]
    wi.write_fields(path, recs)


def test_case_forcing_reads_wps_intermediate(tmp_path):
    for h in (0, 1):
        _write_intermediate(tmp_path / ("ERA5:2022-06-24_%02d" % h), "2022-06-24_%02d:00:00" % h)
    p = cd.CaseForcingProvider([str(tmp_path)])
    series, prov = p.fetch((30.0, 31.0, -100.0, -98.0), RUN_START, RUN_START + datetime.timedelta(hours=1), 1)
    assert sorted(series) == [RUN_START, RUN_START + datetime.timedelta(hours=1)]
    assert series[RUN_START]["cloud_col"] == pytest.approx(1.0e-4 * 30000.0 / cd.G, rel=1e-5)
    assert series[RUN_START]["wind_max"] == pytest.approx(5.0)
    assert len(prov["files"]) == 2 and prov["auth"] == "none"
    # a window with no files fails loudly (ProviderError), never fabricates data
    with pytest.raises(cd.ProviderError):
        cd.CaseForcingProvider([str(tmp_path)]).fetch((30.0, 31.0, -100.0, -98.0),
                                                      D(2030, 1, 1), D(2030, 1, 2), 1)


def test_case_forcing_reads_met_em(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    f = tmp_path / "met_em.d01.2022-06-24_00:00:00.nc"
    with netCDF4.Dataset(str(f), "w") as nc:
        for n, s in (("Time", None), ("num_metgrid_levels", 3), ("south_north", 2), ("west_east", 2),
                     ("west_east_stag", 3), ("south_north_stag", 3)):
            nc.createDimension(n, s)
        d3 = ("Time", "num_metgrid_levels", "south_north", "west_east")
        pres = nc.createVariable("PRES", "f4", d3)
        qc = nc.createVariable("QC", "f4", d3)
        uu = nc.createVariable("UU", "f4", ("Time", "num_metgrid_levels", "south_north", "west_east_stag"))
        vv = nc.createVariable("VV", "f4", ("Time", "num_metgrid_levels", "south_north_stag", "west_east"))
        for k, p in enumerate((100000.0, 85000.0, 70000.0)):
            pres[0, k] = p
        qc[0] = 1.0e-4
        uu[0] = 3.0
        vv[0] = 4.0
    series, prov = cd.CaseForcingProvider([str(tmp_path)]).fetch((0, 90, -180, 180), RUN_START,
                                                                 RUN_START, 1)
    assert series[RUN_START]["cloud_col"] == pytest.approx(1.0e-4 * 30000.0 / cd.G, rel=1e-5)
    assert series[RUN_START]["wind_max"] == pytest.approx(5.0)


def test_era5_provider_with_in_memory_store():
    xr = pytest.importorskip("xarray")
    times = np.array([np.datetime64(t) for t in HOURS[:3]])
    lat = np.arange(40.0, 29.0, -1.0)
    lon = np.arange(255.0, 266.0, 1.0)                       # 0..360, bounds below are -105..-95
    shape = (3, lat.size, lon.size)
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset({
        "total_column_cloud_liquid_water": (dims, np.full(shape, 0.3)),
        "total_column_cloud_ice_water": (dims, np.full(shape, 0.1)),
        "total_precipitation": (dims, np.full(shape, 0.002)),                  # 2 mm/h
        "convective_available_potential_energy": (dims, np.full(shape, 500.0)),  # below 1000
        "10m_u_component_of_wind": (dims, np.full(shape, 6.0)),
        "10m_v_component_of_wind": (dims, np.full(shape, 8.0)),
    }, coords=dict(time=times, latitude=lat, longitude=lon))
    series, prov = cd.Era5Provider(ds=ds).fetch((32.0, 38.0, -105.0, -95.0), HOURS[0], HOURS[2], 1)
    row = series[HOURS[1]]
    assert row == dict(cloud_col=pytest.approx(0.4), precip_area=1.0, cape_area=0.0,
                       wind_max=pytest.approx(10.0))
    assert prov["auth"] == "anonymous" and prov["missing_variables"] == []
    ds2 = ds.drop_vars(["convective_available_potential_energy"])
    series2, prov2 = cd.Era5Provider(ds=ds2).fetch((32.0, 38.0, -105.0, -95.0), HOURS[0], HOURS[2], 1)
    assert "cape_area" not in series2[HOURS[0]] and prov2["missing_variables"]


# ---- coordinator fixes (scc dry run of elevconv) -----------------------------------------------
def test_python_dash_m_help_runs_from_src():
    import os
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    r = subprocess.run([sys.executable, "-m", "wtune_cost_drivers", "--help"], cwd=src,
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                            if k != "PYTHONPATH"})
    assert r.returncode == 0, r.stderr
    assert "case_dir" in r.stdout


def _era5_store(times, lon, with_cape=True):
    xr = pytest.importorskip("xarray")
    lat = np.arange(45.0, 29.0, -1.0)
    shape = (len(times), lat.size, lon.size)
    dims = ("time", "latitude", "longitude")
    data = {
        "total_column_cloud_liquid_water": (dims, np.full(shape, 0.3)),
        "total_precipitation": (dims, np.full(shape, 0.002)),
        "10m_u_component_of_wind": (dims, np.full(shape, 6.0)),
        "10m_v_component_of_wind": (dims, np.full(shape, 8.0)),
    }
    if with_cape:
        data["convective_available_potential_energy"] = (dims, np.full(shape, 2000.0))
    return xr.Dataset(data, coords=dict(time=np.array([np.datetime64(t) for t in times]),
                                        latitude=lat, longitude=lon))


def test_era5_negative_longitudes_and_duplicate_hours_do_not_raise():
    # CONUS box with negative longitudes on a 0..360 store, a store whose time axis holds a
    # duplicated stamp (label-based .sel(method="nearest") raised InvalidIndexError), and the same
    # hour requested twice.
    stamps = HOURS[:4] + [HOURS[2]]                             # 02Z duplicated in the store
    ds = _era5_store(stamps, np.arange(255.0, 276.0, 0.25))
    series, prov = cd.Era5Provider(ds=ds).fetch((30.0, 45.0, -105.0, -85.0), HOURS[0], HOURS[3], 1)
    assert sorted(series) == HOURS[:4]
    assert series[HOURS[2]]["cape_area"] == 1.0 and series[HOURS[2]]["precip_area"] == 1.0
    # fractional / repeated interval requests still yield one row per distinct hour
    s2, _ = cd.Era5Provider(ds=ds).fetch((30.0, 45.0, -105.0, -85.0), HOURS[0], HOURS[3], 0.4)
    assert sorted(s2) == HOURS[:4]


def test_era5_hour_outside_store_is_provider_error():
    ds = _era5_store(HOURS[:2], np.arange(255.0, 276.0, 1.0))
    with pytest.raises(cd.ProviderError):
        cd.Era5Provider(ds=ds).fetch((30.0, 45.0, -105.0, -85.0), HOURS[0], HOURS[5], 1)


def test_sunlit_only_is_not_usable_and_plan_falls_back_with_loud_note():
    sp = [(lat, lon) for lat in (-10.0, 0.0, 10.0) for lon in (-10.0, 0.0, 10.0)]
    cs = _cs([FakeProvider("era5", err="InvalidIndexError: boom"),
              FakeProvider("gfs", err="no network")], sample_points=sp)
    assert "sunlit" in cs.indicators and not cs.usable
    why = cs.why_unusable()
    assert "era5: " in why and "InvalidIndexError: boom" in why and "gfs: " in why and "no network" in why
    base, _n = _plan(None, points_by="time")
    pts, notes = _plan(cs)
    assert [(p.at, p.reason) for p in pts] == [(p.at, p.reason) for p in base]
    assert any("WARNING" in n and "era5" in n and "gfs" in n for n in notes)
    # constant weather indicators are just as unusable
    flat = {t: dict(cloud_col=0.2, precip_area=0.0) for t in HOURS}
    assert not _cs([FakeProvider("flat", flat)], sample_points=sp).usable
    # a varying weather indicator makes it usable again
    assert _cs([FakeProvider("fake", convective_series())], sample_points=sp).usable


def test_expected_cost_structure_wired_into_weights(monkeypatch):
    sp = [(30.0, -100.0), (40.0, -90.0)]
    monkeypatch.setattr(cd.predict, "read_wrfinput_sample_points", lambda d: (sp, "x"))
    monkeypatch.setattr(cd, "case_cost_structure",
                        lambda d, n=None, c=None: {"phases_ranked": [
                            {"phase": "microphysics", "weight": 8.0},
                            {"phase": "dynamics", "weight": 5.0}]})
    cs = cd.cost_series_for_case("/nonexistent", RUN_START, RUN_END,
                                 providers=[FakeProvider("fake", convective_series())])
    assert cs.weights["cloud_col"] == 4.0 and cs.weights["precip_area"] == 4.0
    assert cs.weights["wind_max"] == pytest.approx(cd.WIND_SHARE_OF_DYNAMICS * 5.0)


def test_case_cost_structure_reads_profile_or_returns_none(tmp_path):
    assert cd.case_cost_structure(str(tmp_path)) is None        # no namelist: never raises


def _write_wrf_pair(d, with_bdy=True):
    netCDF4 = pytest.importorskip("netCDF4")
    nz, ny, nx = 3, 4, 5
    with netCDF4.Dataset(str(d / "wrfinput_d01"), "w") as nc:
        for n, s in (("Time", None), ("DateStrLen", 19), ("bottom_top", nz), ("south_north", ny),
                     ("west_east", nx)):
            nc.createDimension(n, s)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        t[0, :] = np.frombuffer(b"2022-05-12_00:00:00", dtype="S1")
        d3 = ("Time", "bottom_top", "south_north", "west_east")
        for name, val in (("P", 0.0), ("QCLOUD", 1.0e-4), ("QICE", 0.0)):
            nc.createVariable(name, "f4", d3)[:] = val
        pb = nc.createVariable("PB", "f4", d3)
        for k, p in enumerate((100000.0, 85000.0, 70000.0)):
            pb[0, k] = p
        for name in ("MU", "MUB"):
            nc.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = 5.0e4
        for name in ("U10", "V10"):
            nc.createVariable(name, "f4", ("Time", "south_north", "west_east"))[:] = 3.0 if name == "U10" else 4.0
    if not with_bdy:
        return
    with netCDF4.Dataset(str(d / "wrfbdy_d01"), "w") as nc:
        for n, s in (("Time", None), ("DateStrLen", 19), ("bdy_width", 2), ("bottom_top", nz),
                     ("along_x", nx), ("along_y", ny)):
            nc.createDimension(n, s)
        t = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        for i, hh in enumerate(("00", "06")):
            t[i, :] = np.frombuffer(("2022-05-12_%s:00:00" % hh).encode(), dtype="S1")
        for side, al in (("BXS", "along_y"), ("BXE", "along_y"), ("BYS", "along_x"), ("BYE", "along_x")):
            for name, val in (("QCLOUD", (1.0e-4 * 1.0e5, 2.0e-4 * 1.0e5)), ("QICE", (0.0, 0.0)),
                              ("U", (10.0 * 1.0e5,) * 2), ("V", (10.0 * 1.0e5,) * 2)):
                v = nc.createVariable("%s_%s" % (name, side), "f4", ("Time", "bdy_width", "bottom_top", al))
                for i in range(2):
                    v[i] = val[i]


def test_case_forcing_falls_back_to_wrfinput_and_wrfbdy(tmp_path):
    T0 = D(2022, 5, 12)
    _write_wrf_pair(tmp_path)
    t6 = T0 + datetime.timedelta(hours=6)
    p = cd.CaseForcingProvider([str(tmp_path)])
    series, prov = p.fetch((0.0, 90.0, -180.0, 180.0), T0, T0 + datetime.timedelta(hours=24), 1)
    assert "FALLBACK" in prov["source"] and len(prov["files"]) == 2
    # boundary column: layer thicknesses from the mean P+PB profile (3 x 15000 Pa), q decoupled by
    # MU+MUB = 1e5 (layer sum, not the level-centre trapezoid wrfinput's own t0 value would use)
    col0 = 1.0e-4 * 45000.0 / cd.G
    assert series[T0]["cloud_col"] == pytest.approx(col0, rel=1e-4)
    assert series[t6]["cloud_col"] == pytest.approx(2 * col0, rel=1e-4)
    assert series[t6]["wind_max"] == pytest.approx(10.0)
    # wrfinput alone: t0 only, with its own U10/V10 wind
    (tmp_path / "wrfbdy_d01").unlink()
    s2, prov2 = p.fetch((0.0, 90.0, -180.0, 180.0), T0, T0 + datetime.timedelta(hours=24), 1)
    assert list(s2) == [T0] and s2[T0]["wind_max"] == pytest.approx(5.0)
    # a real met_em/intermediate file takes precedence and the fallback is not used
    _write_intermediate(tmp_path / "ERA5:2022-05-12_00", "2022-05-12_00:00:00")
    s3, prov3 = p.fetch((30.0, 31.0, -100.0, -98.0), T0, T0 + datetime.timedelta(hours=24), 1)
    assert "FALLBACK" not in prov3["source"]

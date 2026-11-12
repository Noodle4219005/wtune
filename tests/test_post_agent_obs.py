"""post/agent_obs.py: the wtune-obs/1 folders an agent writes (gridded precipitation, rain
gauges), obs-check, window tiling, and the precip recipe's use of them."""
import hashlib
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from wtune import event
from wtune.post import agent_obs, met_pointstat, sources
from wtune.post.recipes import precip


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _write_grid(path, value=2.0, units="mm", lat=(20.0, 21.0, 22.0), lon=(120.0, 121.0)):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("lat", len(lat))
        ds.createDimension("lon", len(lon))
        ds.createVariable("lat", "f8", ("lat",))[:] = lat
        ds.createVariable("lon", "f8", ("lon",))[:] = lon
        v = ds.createVariable("precip", "f4", ("lat", "lon"))
        v[:] = np.full((len(lat), len(lon)), value)
        v.units = units


def _dataset(base, name="cmorph", kind="gridded_precip", hours=range(1, 25), **grid):
    d = base / "input" / "obs" / name
    d.mkdir(parents=True)
    files = {}
    if kind == "gridded_precip":
        for h in hours:
            f = "precip_%s_01h.nc" % (datetime(2016, 10, 7, 0) + timedelta(hours=h)).strftime("%Y%m%d%H")
            _write_grid(str(d / f), **grid)
            files[f] = _sha(d / f)
    else:
        (d / "gauges.csv").write_text(
            "id,lat,lon,elev_m,start_utc,end_utc,precip_mm\n"
            "G1,21.0,120.5,10,2016-10-07 00:00,2016-10-08 00:00,30.5\n"
            "G2,21.5,120.6,,2016-10-07 00:00,2016-10-08 00:00,0\n")
        files["gauges.csv"] = _sha(d / "gauges.csv")
    prov = dict(format=agent_obs.FORMAT, kind=kind, dataset="Test %s" % name, provider="p",
                url="https://example.org", license="public domain", citation="c",
                fetched_utc="2026-10-20T00:00:00Z", files=files)
    (d / "provenance.json").write_text(json.dumps(prov))
    return d


def test_grid_units_range_and_sha_are_checked(tmp_path):
    d = _dataset(tmp_path, units="kg m-2", value=9000.0)
    err, _ = agent_obs.check(str(d))
    assert any("units" in e for e in err) and any("outside" in e for e in err)
    _write_grid(str(d / "precip_2016100701_01h.nc"), value=3.0, units="kg m-2")
    assert any("sha256 differs" in e for e in agent_obs.check(str(d))[0])


def test_unknown_kind_and_missing_provenance(tmp_path):
    d = _dataset(tmp_path)
    prov = json.loads((d / "provenance.json").read_text())
    prov["kind"] = "lidar"
    (d / "provenance.json").write_text(json.dumps(prov))
    assert any("has no reader" in e for e in agent_obs.check(str(d))[0])
    os.remove(d / "provenance.json")
    assert agent_obs.check(str(d))[0] == ["provenance.json: missing or not JSON"]


def test_gauges_pass_and_bad_time_fails(tmp_path):
    d = _dataset(tmp_path, name="cwa", kind="rain_gauges")
    assert agent_obs.check(str(d))[0] == []
    rows = agent_obs.read_gauges(str(d / "gauges.csv"))
    assert rows[1]["elev_m"] == 0.0 and rows[0]["end"] == datetime(2016, 10, 8)
    (d / "gauges.csv").write_text("id,lat,lon,elev_m,start_utc,end_utc,precip_mm\n"
                                  "G1,21,120,0,2016-10-07T00:00+08:00,2016-10-08 00:00,1\n")
    prov = json.loads((d / "provenance.json").read_text())
    prov["files"]["gauges.csv"] = _sha(d / "gauges.csv")
    (d / "provenance.json").write_text(json.dumps(prov))
    assert any("line 2" in e for e in agent_obs.check(str(d))[0])


def test_read_grid_flips_descending_lat_and_wraps_lon(tmp_path):
    p = str(tmp_path / "g.nc")
    _write_grid(p, lat=(22.0, 21.0), lon=(359.0, 0.5, 1.5))
    lat, lon, _ = agent_obs.read_grid(p)
    assert list(lat) == [21.0, 22.0] and list(lon) == [-1.0, 0.5, 1.5]


def test_dataset_records_include_agent_dataset(tmp_path):
    _dataset(tmp_path)
    rec = sources.dataset_records(str(tmp_path), ["cmorph", "no_such"])
    assert [r["name"] for r in rec] == ["cmorph"] and rec[0]["dataset"] == "Test cmorph"


def test_gauge_periods_and_obs_text():
    rows = [dict(id="A 1", lat=1, lon=2, elev_m=3, start=datetime(2016, 10, 7, h),
                 end=datetime(2016, 10, 7, h + 1), precip_mm=1.5) for h in range(0, 10)]
    rows.append(dict(id="B", lat=1, lon=2, elev_m=0, start=datetime(2016, 10, 7, 0),
                     end=datetime(2016, 10, 7, 6), precip_mm=9.0))
    frames = [datetime(2016, 10, 6, 18) + timedelta(hours=h) for h in range(20)]
    hours, periods = met_pointstat.gauge_periods(rows, frames, 6)
    assert hours == 6 and periods == [(datetime(2016, 10, 7, 0), datetime(2016, 10, 7, 6))]
    text = met_pointstat.gauge_obs_text(rows, 1)
    assert text.splitlines()[0] == "ADPSFC A_1 20161007_010000 1 2 3 APCP_01 1 -9999 NA 1.500"
    assert len(text.splitlines()) == 10


def _frames(tmp_path):
    out = []
    for h in range(49):
        t = datetime(2016, 10, 6, 12) + timedelta(hours=h)
        out.append((t.strftime("%Y-%m-%d_%H:%M:%S"), str(tmp_path / "wrfout"), h))
    return out


def test_recipe_draws_agent_grid_figures(tmp_path, monkeypatch):
    _dataset(tmp_path)
    out_dir = tmp_path / "post"
    out_dir.mkdir()
    from wtune.post import caseinfo, met_gridstat, obs_figs
    lat2, lon2 = np.meshgrid([20.5, 21.5], [120.2, 120.8], indexing="ij")
    monkeypatch.setattr(precip, "_model_accum", lambda a, b: (lat2, lon2, np.ones((2, 2))))
    monkeypatch.setattr(precip, "_model_bbox", lambda p: (20.5, 21.5, 120.2, 120.8))
    monkeypatch.setattr(caseinfo, "spec_bdy_width", lambda r: 5)
    seen = {}
    monkeypatch.setattr(met_gridstat, "curvilinear_to_latlon", lambda *a, **k: "f.nc")
    monkeypatch.setattr(met_gridstat, "write_interior_polygon", lambda *a, **k: "p.poly")

    def verify(fcst, obs, poly, work, fname, oname, thr, **kw):
        with netCDF4.Dataset(obs) as ds:
            seen["obs"] = np.asarray(ds.variables["APCP_OBS"][0])
        seen["kw"], seen["thr"] = kw, thr
        return dict(cts="c", nbrcnt="n", pairs_nc="p")
    monkeypatch.setattr(met_gridstat, "run_verification", verify)
    monkeypatch.setattr(met_gridstat, "parse_cts_table", lambda p: [dict(
        thresh=1, pod=1, far=0, csi=1, fbias=1, total=4)])
    monkeypatch.setattr(met_gridstat, "parse_nbrcnt_table", lambda p: [dict(
        thresh=1, width=1, fss=1, fbs=0, f_rate=1, o_rate=1)])
    monkeypatch.setattr(met_gridstat, "read_pairs_fields",
                        lambda p: (lat2, lon2, np.ones((2, 2)), np.ones((2, 2)), "f", "o"))
    for fn in ("side_by_side", "diff_map", "scores_vs_threshold", "fss_vs_scale"):
        monkeypatch.setattr(obs_figs, fn, lambda *a, **k: "drawn.png")
    figs = precip._agent_grid_figures("run", "r", 1, _frames(tmp_path), str(out_dir), 6,
                                      str(tmp_path))
    assert [f["id"] for f in figs] == list(precip.AGENT_GRID_IDS.values())
    assert all(not f["missing"] for f in figs)
    assert figs[0]["obs_sources"][0]["dataset"] == "Test cmorph"
    assert "2016-10-07 00 -> 2016-10-08 00 UTC (24 h)" == figs[0]["window"]
    assert np.allclose(seen["obs"], 48.0) and seen["kw"]["to_grid"] == "OBS"
    assert os.path.isfile(out_dir / "precip_categorical_cmorph_A24.csv")
    # a grid over Taiwan is scored at CWA's warning levels, not the WPC inch values
    assert seen["thr"] == [">=1", ">=10", ">=40", ">=80", ">=200", ">=350"]
    lat_us, lon_us = np.meshgrid([30.0, 31.0], [-90.0, -89.0], indexing="ij")
    assert precip.event_thresholds(lat_us, lon_us) == precip.THRESHOLDS_MM


def test_recipe_notes_a_dataset_that_fails_check(tmp_path):
    _dataset(tmp_path, units="in")
    figs = precip._agent_grid_figures("run", "r", 1, _frames(tmp_path), str(tmp_path), 6,
                                      str(tmp_path))
    assert len(figs) == 1 and figs[0]["kind"] == "note" and "units" in figs[0]["text"]


def test_recipe_draws_gauge_figures(tmp_path, monkeypatch):
    _dataset(tmp_path, name="cwa", kind="rain_gauges")
    from wtune.post import obs_figs
    end = datetime(2016, 10, 8)
    monkeypatch.setattr(met_pointstat, "verify_gauges", lambda *a: dict(
        hours=24.0, periods=[(datetime(2016, 10, 7), end)], csv=["a.csv", "b.csv"],
        pairs=[("G1", 21.0, 120.5, end, 25.0, 30.5), ("G2", 21.5, 120.6, end, 1.0, 0.0)]))
    got = {}
    monkeypatch.setattr(obs_figs, "scatter_compare", lambda f, o, *a: got.setdefault("s", (f, o)) and "s.png")
    monkeypatch.setattr(obs_figs, "station_bias_maps", lambda panels, *a: got.setdefault("m", panels) and "m.png")
    figs = precip._gauge_figures("run", "r", 1, _frames(tmp_path), str(tmp_path), 6, str(tmp_path))
    assert [f["id"] for f in figs] == ["precip_gauge_scatter", "precip_gauge_map"]
    assert got["s"] == ([25.0, 1.0], [30.5, 0.0])
    assert list(got["m"][0]["bias"]) == [-5.5, 1.0]


def test_event_obs_check_cli_and_plan_note(tmp_path, capsys):
    d = _dataset(tmp_path)
    assert event.main(["obs-check", "--project", str(tmp_path)]) == 0
    _write_grid(str(d / "precip_2016100701_01h.nc"), value=5.0)
    assert event.main(["obs-check", str(d)]) == 1
    assert "sha256 differs" in capsys.readouterr().out
    rows = event.plan({"verify": [{"value": "precip_24h"}, {"value": "precip_gauge"},
                                  {"value": "gust"}]},
                      datetime(2016, 10, 7), datetime(2016, 10, 8), (20, 26, 118, 124))
    note = {r["need"]: r["note"] for r in rows}
    assert "obs-check" in note["precip_24h"] and "obs-check" in note["precip_gauge"]
    assert "obs-check" in note["gust"]
    rows = event.plan({"verify": [{"value": "track"}], "storms": [{"name": "Haiyan",
                                                                  "id": "WP312013"}]})
    assert "--bdeck" in rows[0]["note"]


def _snapshot_dataset(base, kind, var, units, value, times, name="snap"):
    d = base / "input" / "obs" / name
    d.mkdir(parents=True)
    prefix = {"radar": "refl", "satellite_ir": "irtb"}[kind]
    files = {}
    for t in times:
        f = "%s_%s.nc" % (prefix, t.strftime("%Y%m%d%H%M"))
        with netCDF4.Dataset(str(d / f), "w") as ds:
            ds.createDimension("lat", 2)
            ds.createDimension("lon", 2)
            ds.createVariable("lat", "f8", ("lat",))[:] = [20.0, 21.0]
            ds.createVariable("lon", "f8", ("lon",))[:] = [120.0, 121.0]
            v = ds.createVariable(var, "f4", ("lat", "lon"))
            v[:] = np.full((2, 2), value)
            v.units = units
        files[f] = _sha(d / f)
    (d / "provenance.json").write_text(json.dumps(dict(
        format=agent_obs.FORMAT, kind=kind, dataset="Test %s" % kind, provider="p", url="u",
        license="l", citation="c", fetched_utc="2026-10-21T00:00:00Z", files=files)))
    return d


@pytest.mark.parametrize("kind,var,units,good,bad", [("radar", "refl", "dBZ", 35.0, 120.0),
                                                     ("satellite_ir", "tb", "K", 220.0, 400.0)])
def test_snapshot_formats_are_checked(tmp_path, kind, var, units, good, bad):
    t = datetime(2016, 10, 7, 12)
    d = _snapshot_dataset(tmp_path, kind, var, units, good, [t])
    assert agent_obs.check(str(d)) == ([], [])
    assert agent_obs.snapshot_index(str(d), kind) == {t: str(d / os.listdir(d)[0])} or \
        list(agent_obs.snapshot_index(str(d), kind)) == [t]
    d2 = _snapshot_dataset(tmp_path, kind, var, "wrong", bad, [t], name="bad")
    err = agent_obs.check(str(d2))[0]
    assert any("units" in e for e in err) and any("outside" in e for e in err)


def test_nearest_respects_tolerance():
    idx = {datetime(2016, 10, 7, 12, 5): "a", datetime(2016, 10, 7, 13): "b"}
    assert agent_obs.nearest(idx, datetime(2016, 10, 7, 12), 8) == "a"
    assert agent_obs.nearest(idx, datetime(2016, 10, 7, 12, 30), 8) is None


def test_stations_extra_check_and_obs_text(tmp_path):
    d = tmp_path / "input" / "obs" / "isd_full"
    d.mkdir(parents=True)
    (d / "stations_extra.csv").write_text(
        "id,lat,lon,elev_m,time_utc,var,value\n"
        "S1,30,-81,5,2016-10-07 12:00,gust,25.7\n"
        "S1,30,-81,5,2016-10-07 12:00,visibility,40000\n")
    (d / "provenance.json").write_text(json.dumps(dict(
        format=agent_obs.FORMAT, kind="stations_extra", dataset="d", provider="p", url="u",
        license="l", fetched_utc="f", files={"stations_extra.csv": _sha(d / "stations_extra.csv")})))
    assert agent_obs.check(str(d))[0] == []
    rows = agent_obs.read_extra(str(d / "stations_extra.csv"))
    text = met_pointstat.extra_obs_text(rows, datetime(2016, 10, 7), datetime(2016, 10, 8))
    lines = text.splitlines()
    assert lines[0].split()[6:] == ["WGUST", "0", "-9999", "NA", "25.700"]
    assert lines[1].split()[6] == "VISIB" and float(lines[1].split()[-1]) == 16093.0
    (d / "stations_extra.csv").write_text("id,lat,lon,elev_m,time_utc,var,value\n"
                                          "S1,30,-81,5,2016-10-07 12:00,wind,3\n")
    prov = json.loads((d / "provenance.json").read_text())
    prov["files"]["stations_extra.csv"] = _sha(d / "stations_extra.csv")
    (d / "provenance.json").write_text(json.dumps(prov))
    assert any("var must be" in e for e in agent_obs.check(str(d))[0])


class _DS:
    def __init__(self, **v):
        self.variables = {k: np.asarray(a) for k, a in v.items()}


def test_visibility_clear_air_is_capped_and_fog_is_low():
    from wtune.post import surface_diag
    shape = (1, 1, 1, 2)
    qc = np.zeros(shape)
    qc[..., 1] = 3e-4                              # ~0.35 g/m3 of cloud water at the surface
    ds = _DS(P=np.full(shape, 0.0), PB=np.full(shape, 1.0e5), T=np.zeros(shape),
             QVAPOR=np.full(shape, 0.01), QCLOUD=qc)
    vis = surface_diag.visibility(ds)
    assert vis[0, 0] == surface_diag.VIS_CAP_M
    assert 50 < vis[0, 1] < 200


def test_gust_prefers_wspd10max():
    from wtune.post import surface_diag
    g, how = surface_diag.gust(_DS(WSPD10MAX=np.full((1, 2, 2), 17.0)))
    assert how == "WSPD10MAX" and float(g.max()) == 17.0


def test_cloud_recipe_draws_agent_ir(tmp_path, monkeypatch):
    from wtune.post import condensate, met_gridstat, obs_figs, snapshot_verify
    from wtune.post.recipes import cloud, convection
    proj = tmp_path
    t = datetime(2016, 10, 7, 12)
    _snapshot_dataset(proj, "satellite_ir", "tb", "K", 220.0, [t])
    monkeypatch.setattr(convection, "_project_of", lambda o: str(proj))
    monkeypatch.setattr(convection, "_post_spinup_frames",
                        lambda r, d, s: [("2016-10-07_12:00:00", "wrf", 0)])
    lat2, lon2 = np.meshgrid([20.0, 21.0], [120.0, 121.0], indexing="ij")
    real = netCDF4.Dataset
    monkeypatch.setattr(netCDF4, "Dataset", lambda p, *a, **k: _Ctx(XLAT=lat2[None], XLONG=lon2[None])
                        if p == "wrf" else real(p, *a, **k))
    monkeypatch.setattr(condensate, "olr_brightness_temperature", lambda ds, ti: np.full((2, 2), 215.0))
    monkeypatch.setattr(met_gridstat, "write_interior_polygon", lambda *a: "p")
    from wtune.post import caseinfo
    monkeypatch.setattr(caseinfo, "spec_bdy_width", lambda r: 5)
    seen = {}

    def verify(picks, model_field, obs_file, names, thresholds, work, poly, **kw):
        seen.update(picks=picks, thresholds=thresholds, model=model_field(*picks[0][1]))
        ctc = [dict(thresh=240.0, fy_oy=3, fy_on=1, fn_oy=0, fn_on=0)]
        nbr = [dict(thresh=240.0, width=1, fss=0.8, fbs=0.1, o_rate=0.7, f_rate=0.8)]
        cts = [dict(thresh=240.0, pod=1, far=0.25, csi=0.75, fbias=1.33, total=4)]
        late = t + timedelta(hours=3)
        return dict(per_ctc=[ctc], per_nbr=[nbr], cts_rows=[(t, c) for c in cts],
                    fields=[(t, lat2, lon2, np.full((2, 2), 215.0), np.full((2, 2), 220.0)),
                            (late, lat2, lon2, np.full((2, 2), 260.0), np.full((2, 2), 260.0))])
    monkeypatch.setattr(snapshot_verify, "verify", verify)
    drawn, args = {}, {}
    for fn in ("side_by_side", "scores_vs_threshold", "fss_vs_scale"):
        monkeypatch.setattr(obs_figs, fn, lambda *a, _f=fn, **k: args.__setitem__(_f, a)
                            or drawn.setdefault(_f, k) or "x.png")
    out = cloud._agent_ir_figures("run", "r", 1, {"domains": [{"domain": 1, "dx_km": 27}]},
                                  str(tmp_path / "post"))
    assert [f.get("id") or f.get("text") for f in out] == list(cloud.IR_IDS)
    assert seen["thresholds"] == ["<=240", "<=220", "<=210"] and seen["model"][3] == "K"
    assert drawn["scores_vs_threshold"]["op"] == "≤"
    assert "2016-10-07 12:00" in args["side_by_side"][5]        # most cold cloud
    # with the rain peak: the verified time nearest it; the panel names the dataset briefly
    # (its `dataset` sentence ran off the panel), one file set per domain
    out = cloud._agent_ir_figures("run", "r", 3, {"domains": [{"domain": 3, "dx_km": 3}]},
                                  str(tmp_path / "post"), map_time="2016-10-07_14:00:00",
                                  tag="_d03")
    assert "2016-10-07 15:00" in args["side_by_side"][5]
    assert args["side_by_side"][4][1] == "%s (obs)" % out[0]["obs_sources"][0]["name"].upper()
    assert args["side_by_side"][6].endswith("_d03.png") and out[0]["domain"] == "d03"


class _Ctx:
    def __init__(self, **v):
        self.variables = {k: np.asarray(a) for k, a in v.items()}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_recipe_draws_gust_and_visibility_figures(tmp_path, monkeypatch):
    from wtune.post import obs_figs
    d = tmp_path / "input" / "obs" / "isd_full"
    d.mkdir(parents=True)
    (d / "stations_extra.csv").write_text("id,lat,lon,elev_m,time_utc,var,value\n"
                                          "S1,30,-81,5,2016-10-07 12:00,gust,25\n")
    (d / "provenance.json").write_text(json.dumps(dict(
        format=agent_obs.FORMAT, kind="stations_extra", dataset="d", provider="p", url="u",
        license="l", fetched_utc="f", files={"stations_extra.csv": _sha(d / "stations_extra.csv")})))
    t = datetime(2016, 10, 7, 12)
    monkeypatch.setattr(met_pointstat, "verify_extra", lambda *a: dict(
        pairs={"WGUST": [("S1", 30.0, -81.0, t, 18.0, 25.0)], "VISIB": []},
        gust_method="UPP CALGUST", csv=["p.csv"], t_first=t, t_last=t))
    monkeypatch.setattr(obs_figs, "scatter_compare", lambda *a: "s.png")
    monkeypatch.setattr(obs_figs, "station_bias_maps", lambda *a: "m.png")
    figs = precip._extra_figures("run", "r", 1, _frames(tmp_path), str(tmp_path), 6, str(tmp_path))
    assert [(f["id"], f["missing"]) for f in figs] == [
        ("stn_gust_scatter", False), ("stn_gust_map", False),
        ("stn_vis_scatter", True), ("stn_vis_map", True)]
    assert "UPP CALGUST" in figs[0]["tool"]


def test_regrid_snapshot_leaves_cells_outside_the_obs_grid_missing(tmp_path):
    from wtune.post import tools
    if tools.find_on_path("cdo", install_hint="") is None:
        pytest.skip("no cdo")
    t = datetime(2016, 10, 7, 12)
    d = _snapshot_dataset(tmp_path, "radar", "refl", "dBZ", 30.0, [t])
    griddes = tmp_path / "g.griddes"
    griddes.write_text("gridtype = lonlat\nxsize = 4\nysize = 4\nxfirst = 119.0\nxinc = 1.0\n"
                       "yfirst = 18.0\nyinc = 1.0\n")
    out = agent_obs.regrid_snapshot(str(next(d.glob("refl_*.nc"))), "refl", t, str(griddes),
                                    str(tmp_path / "o.nc"), "REFC_OBS", "dBZ")
    with netCDF4.Dataset(out) as ds:
        a = np.ma.filled(np.ma.asarray(ds.variables["REFC_OBS"][0], "f8"), np.nan)
        a = np.where(a > 1e30, np.nan, a)
    assert np.isnan(a[0]).all() and np.isnan(a[:, 0]).all()       # 18N and 119E lie outside
    assert np.nanmax(a) == 30.0


def test_obs_check_warns_on_implausible_magnitudes(tmp_path):
    d = _dataset(tmp_path, value=0.02)                      # a scale applied twice
    err, warn = agent_obs.check(str(d))
    assert err == [] and any("scale applied twice" in w for w in warn)
    t = datetime(2016, 10, 7, 12)
    r = _snapshot_dataset(tmp_path, "radar", "refl", "dBZ", 5.0, [t], name="r")
    assert any("strongest echo" in w for w in agent_obs.check(str(r))[1])


def test_obs_check_warns_when_radar_has_no_coverage_mask(tmp_path):
    d = tmp_path / "input" / "obs" / "big"
    d.mkdir(parents=True)
    f = "refl_201610071200.nc"
    with netCDF4.Dataset(str(d / f), "w") as ds:
        ds.createDimension("lat", 11)
        ds.createDimension("lon", 11)
        ds.createVariable("lat", "f8", ("lat",))[:] = np.linspace(24, 34, 11)
        ds.createVariable("lon", "f8", ("lon",))[:] = np.linspace(-84, -74, 11)
        v = ds.createVariable("refl", "f4", ("lat", "lon"))
        v[:] = np.full((11, 11), -32.0)
        v[5, 5] = 40.0
        v.units = "dBZ"
    (d / "provenance.json").write_text(json.dumps(dict(
        format=agent_obs.FORMAT, kind="radar", dataset="d", provider="p", url="u", license="l",
        fetched_utc="f", files={f: _sha(d / f)})))
    err, warn = agent_obs.check(str(d))
    assert err == [] and any("beyond the radars' range" in w for w in warn)


def test_obs_check_warns_on_station_units(tmp_path):
    d = tmp_path / "input" / "obs" / "st"
    d.mkdir(parents=True)
    (d / "stations_extra.csv").write_text("id,lat,lon,elev_m,time_utc,var,value\n"
                                          "S1,30,-81,5,2016-10-07 12:00,gust,1.2\n"
                                          "S1,30,-81,5,2016-10-07 12:00,visibility,16\n")
    (d / "provenance.json").write_text(json.dumps(dict(
        format=agent_obs.FORMAT, kind="stations_extra", dataset="d", provider="p", url="u",
        license="l", fetched_utc="f", files={"stations_extra.csv": _sha(d / "stations_extra.csv")})))
    err, warn = agent_obs.check(str(d))
    assert err == [] and len(warn) == 2


def test_window_centres_on_the_rain_peak(tmp_path):
    """haitang2017 run72: the latest 24 h (08-02 00Z -> 08-03 00Z) began after the 08-01 23Z
    peak; the window now contains the peak, centred as near it as the files allow."""
    d = _dataset(tmp_path, hours=range(1, 37))
    frames = [datetime(2016, 10, 6, 12) + timedelta(hours=h) for h in range(49)]
    idx = agent_obs.grid_index(str(d))
    assert agent_obs.pick_window(frames, 6, idx)[1:3] == (datetime(2016, 10, 7, 12),
                                                         datetime(2016, 10, 8, 12))
    assert agent_obs.pick_window(frames, 6, idx, around=datetime(2016, 10, 7, 8))[1:3] == (
        datetime(2016, 10, 7), datetime(2016, 10, 8))


def test_grid_figures_over_taiwan_use_cwa_and_a_short_name(tmp_path, monkeypatch):
    """haitang2017 run72: the CMORPH comparison over Taiwan was drawn on the WPC inch scale."""
    from wtune.post import obs_figs, rainscale
    lat, lon = np.meshgrid(np.linspace(21, 26, 6), np.linspace(118, 123, 6), indexing="ij")
    f, o = np.full((6, 6), 40.0), np.full((6, 6), 120.0)
    o[0, 0] = 400.0
    levels, cmap, _name = rainscale.rain_scale(lat, lon, 400.0)
    p = obs_figs.side_by_side(lat, lon, f, o, ("a", "b"), "t", str(tmp_path / "s.png"), levels,
                              cmap=cmap)
    assert os.path.isfile(p)                    # a one-colour-per-bin cmap draws
    seen = {}
    monkeypatch.setattr(obs_figs, "side_by_side",
                        lambda *a, cmap=None, **k: seen.update(levels=a[7], cmap=cmap) or "x")
    for fn in ("diff_map", "scores_vs_threshold", "fss_vs_scale"):
        monkeypatch.setattr(obs_figs, fn, lambda *a, **k: "x")
    cts = [dict(thresh=1, pod=1, far=0, csi=1, fbias=1, total=36)]
    precip._grid_figures((lat, lon, f, o, cts, []), 25.0, "CMORPH", "CMORPH (obs)",
                         precip.AGENT_GRID_IDS, {k: k + ".png" for k in precip.AGENT_GRID_IDS},
                         "w", "m", {}, "2017-08-02_00:00:00", "run72", 3, str(tmp_path))
    assert list(seen["levels"]) == [float(v) for v in rainscale.CWA_LEVELS]
    assert seen["cmap"].name == "cwa_rain"


def test_ir_thinning_keeps_the_time_nearest_the_rain_peak(tmp_path, monkeypatch):
    from wtune.post import condensate, met_gridstat, snapshot_verify
    from wtune.post.recipes import cloud, convection
    t0 = datetime(2016, 10, 7, 0)
    times = [t0 + timedelta(hours=3 * k) for k in range(40)]
    _snapshot_dataset(tmp_path, "satellite_ir", "tb", "K", 220.0, times)
    monkeypatch.setattr(convection, "_project_of", lambda o: str(tmp_path))
    monkeypatch.setattr(convection, "_post_spinup_frames", lambda r, d, s: [
        (t.strftime("%Y-%m-%d_%H:%M:%S"), "wrf", 0) for t in times])
    monkeypatch.setattr(cloud, "MAX_IR_TIMES", 6)
    monkeypatch.setattr(met_gridstat, "write_interior_polygon", lambda *a: "p")
    from wtune.post import caseinfo
    monkeypatch.setattr(caseinfo, "spec_bdy_width", lambda r: 5)
    lat2, lon2 = np.meshgrid([20.0, 21.0], [120.0, 121.0], indexing="ij")
    real = netCDF4.Dataset
    monkeypatch.setattr(netCDF4, "Dataset", lambda p, *a, **k: _Ctx(XLAT=lat2[None], XLONG=lon2[None])
                        if p == "wrf" else real(p, *a, **k))
    monkeypatch.setattr(condensate, "olr_brightness_temperature", lambda ds, ti: np.full((2, 2), 215.0))
    seen = {}

    def verify(picks, *a, **k):
        seen["times"] = [p[0] for p in picks]
        raise ValueError("stop")
    monkeypatch.setattr(snapshot_verify, "verify", verify)
    cloud._agent_ir_figures("run", "r", 1, {"domains": [{"domain": 1, "dx_km": 27}]},
                            str(tmp_path / "post"), map_time="2016-10-08_23:00:00")
    assert datetime(2016, 10, 9, 0) in seen["times"] and len(seen["times"]) <= 7

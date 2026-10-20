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


def test_valid_grid_dataset_passes(tmp_path):
    d = _dataset(tmp_path)
    assert agent_obs.check(str(d)) == ([], [])
    ok, bad = agent_obs.usable(str(tmp_path), "gridded_precip")
    assert [n for n, _d, _p in ok] == ["cmorph"] and bad == []


def test_grid_units_range_and_sha_are_checked(tmp_path):
    d = _dataset(tmp_path, units="kg m-2", value=9000.0)
    err, _ = agent_obs.check(str(d))
    assert any("units" in e for e in err) and any("outside" in e for e in err)
    _write_grid(str(d / "precip_2016100701_01h.nc"), value=3.0, units="kg m-2")
    assert any("sha256 differs" in e for e in agent_obs.check(str(d))[0])


def test_unknown_kind_and_missing_provenance(tmp_path):
    d = _dataset(tmp_path)
    prov = json.loads((d / "provenance.json").read_text())
    prov["kind"] = "radar"
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


def test_hourly_files_tile_a_24h_window(tmp_path):
    d = _dataset(tmp_path)
    frames = [datetime(2016, 10, 6, 12) + timedelta(hours=h) for h in range(49)]
    span, start, end, paths = agent_obs.pick_window(frames, 6, agent_obs.grid_index(str(d)))
    assert (span, start, end, len(paths)) == (24, datetime(2016, 10, 7), datetime(2016, 10, 8), 24)
    lat, lon, total = agent_obs.sum_grids(paths)
    assert np.allclose(total, 48.0)
    la, lo, sub = agent_obs.subset(lat, lon, total, (20.5, 21.5, 120.2, 120.4), margin=0.5)
    assert list(la) == [20.0, 21.0, 22.0] and list(lo) == [120.0] and sub.shape == (3, 1)


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
        seen["kw"] = kw
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
    assert "do not read" in note["gust"]

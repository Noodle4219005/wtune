"""post/recipes/hovmoller.py: rain against distance along a line and time, model and observation."""
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("matplotlib")

from wtune.post import agent_obs  # noqa: E402
from wtune.post.extract import list_frame_refs  # noqa: E402
from wtune.post.recipes import hovmoller  # noqa: E402

T0 = datetime(2017, 8, 1)
NY, NX = 30, 40


def _run(tmp_path, hours=13):
    """Hourly wrfout frames on a 0.1 deg grid (22.5-25.4N, 119-122.9E); a 0.3 deg wide N-S rain
    band of 5 mm/h moves east 0.1 deg per hour from 120E."""
    run = tmp_path / "run"
    run.mkdir()
    lat = (22.5 + 0.1 * np.arange(NY))[:, None] * np.ones((1, NX))
    lon = np.ones((NY, 1)) * (119.0 + 0.1 * np.arange(NX))[None, :]
    acc = np.zeros((NY, NX))
    for h in range(hours):
        t = (T0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")
        if h:
            acc = acc + 5.0 * (np.abs(lon - (120.0 + 0.1 * h)) <= 0.15)
        with netCDF4.Dataset(str(run / ("wrfout_d01_" + t)), "w") as ds:
            for name, n in (("Time", 1), ("DateStrLen", 19), ("south_north", NY), ("west_east", NX)):
                ds.createDimension(name, n)
            ds.createVariable("Times", "S1", ("Time", "DateStrLen"))[:] = np.array([list(t)], "S1")
            dims = ("Time", "south_north", "west_east")
            for name, a in (("XLAT", lat), ("XLONG", lon), ("RAINC", np.zeros_like(acc)),
                            ("RAINNC", acc)):
                ds.createVariable(name, "f4", dims)[:] = a[None]
    return str(run)


def _cmorph(project, value=12.0):
    d = os.path.join(project, "input", "obs", "cmorph")
    os.makedirs(d)
    files = {}
    for end in (T0 + timedelta(hours=6), T0 + timedelta(hours=12)):
        f = "precip_%s_06h.nc" % end.strftime("%Y%m%d%H")
        with netCDF4.Dataset(os.path.join(d, f), "w") as ds:
            la, lo = np.arange(22.5, 25.5, 0.25), np.arange(119.0, 123.0, 0.25)
            ds.createDimension("lat", len(la))
            ds.createDimension("lon", len(lo))
            ds.createVariable("lat", "f8", ("lat",))[:] = la
            ds.createVariable("lon", "f8", ("lon",))[:] = lo
            v = ds.createVariable("precip", "f4", ("lat", "lon"))
            v[:] = np.full((len(la), len(lo)), value)
            v.units = "mm"
        files[f] = agent_obs._sha256(os.path.join(d, f))
    prov = dict(format=agent_obs.FORMAT, kind="gridded_precip", dataset="CMORPH test",
                provider="p", url="https://example.org", license="public domain",
                fetched_utc="2026-11-03T00:00:00Z", files=files)
    with open(os.path.join(d, "provenance.json"), "w") as fh:
        json.dump(prov, fh)


def test_off_grid_points_are_nan_not_the_edge_cell():
    lat = np.linspace(22.0, 24.0, 21)[:, None] * np.ones((1, 21))
    lon = np.ones((21, 1)) * np.linspace(120.0, 122.0, 21)[None, :]
    dist, v = hovmoller.sample_line([lon], lat, lon, (23.0, 119.0, 23.0, 121.0), n=21)
    assert np.isnan(v[0, :8]).all() and np.isfinite(v[0, -5:]).all()
    assert abs(dist[0] + dist[-1]) < 1e-6                        # centred on the line middle


def test_figure_shows_the_band_moving_and_the_observation_windows(tmp_path):
    run = _run(tmp_path)
    project = str(tmp_path / "proj")
    _cmorph(project)
    frames = list_frame_refs(run, domain=1)
    out = str(tmp_path / "out")
    os.makedirs(out)
    entry = hovmoller.figure(run, "r", 1, frames, out, 0, project, None, None,
                             line_override=dict(lat0=24.0, lon0=119.2, lat1=24.0, lon1=122.6))
    assert not entry.get("missing"), entry
    assert os.path.isfile(entry["path"]) and entry["observations"] == ["cmorph"]
    rows = list(__import__("csv").DictReader(open(entry["csv"])))
    hourly = [r for r in rows if r["panel"] == "wrf_hourly" and r["mm"]]
    peak = {}
    for r in hourly:
        k = r["start_utc"]
        if float(r["mm"]) > peak.get(k, (0, None))[0]:
            peak[k] = (float(r["mm"]), float(r["distance_km"]))
    xs = [peak[k][1] for k in sorted(peak)]
    assert all(b > a for a, b in zip(xs, xs[1:]))                 # the band moves east
    assert max(v for v, _x in peak.values()) == pytest.approx(5.0)
    obs = [float(r["mm"]) for r in rows if r["panel"] == "cmorph" and r["mm"]]
    assert obs and set(obs) == {12.0}
    windows = {(r["start_utc"], r["end_utc"]) for r in rows if r["panel"] == "wrf_cmorph"}
    assert windows == {("2017-08-01T00:00Z", "2017-08-01T06:00Z"),
                       ("2017-08-01T06:00Z", "2017-08-01T12:00Z")}


def test_default_line_crosses_the_rain_core_against_the_low_level_wind(tmp_path, monkeypatch):
    from wtune.post import focus
    run = _run(tmp_path)
    frames = list_frame_refs(run, domain=1)
    monkeypatch.setattr(focus, "rain_focus", lambda *a, **k: dict(
        lat=24.0, lon=120.5, j=15, i=15, peak_time=frames[5][0]))
    boxes = []
    monkeypatch.setattr(hovmoller, "_low_level_wind",
                        lambda path, ti, box: boxes.append(box) or (6.0, 6.0))
    line, rule = hovmoller.default_line(run, 1, frames, 0, None, None)
    assert line[0] < 24.0 < line[2] and line[1] > 120.5 > line[3]
    assert "from 225 deg" in rule and "24.00N 120.50E" in rule and "+-1 deg" in rule
    # the background flow over post.focus, not the rain core's own (convective) column
    hovmoller.default_line(run, 1, frames, 0, [22.5, 25.0, 119.5, 122.0], None)
    assert boxes == [(23.0, 25.0, 119.5, 121.5), (22.5, 25.0, 119.5, 122.0)]

"""post/station_compare.py and agent_obs.from_asos: stations dataset, circular wind-direction
statistics, calm exclusion, domain selection, nearest / bilinear sampling, output files."""
import csv
import hashlib
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from wtune import event
from wtune.post import agent_obs, station_compare as sc

T0 = datetime(2024, 3, 1, 0)


def _write_wrfout(path, tstr, lat0, lon0, d, n, t2_fn, u, v, hgt=100.0, gust=None, alpha=0.0):
    """A small wrfout: n x n mass points on a regular lat/lon grid, spacing d degrees."""
    lat1 = lat0 + d * np.arange(n)
    lon1 = lon0 + d * np.arange(n)
    lon, lat = np.meshgrid(lon1, lat1)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", n)
        ds.createDimension("west_east", n)
        ds.createDimension("DateStrLen", 19)
        tv = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        tv[0, :] = np.frombuffer(tstr.encode(), "S1")

        def put(name, a):
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = a
        put("XLAT", lat)
        put("XLONG", lon)
        put("HGT", np.full((n, n), hgt) if np.isscalar(hgt) else hgt)
        put("T2", t2_fn(lat, lon))
        put("U10", np.full((n, n), u))
        put("V10", np.full((n, n), v))
        put("Q2", np.full((n, n), 0.004))
        put("PSFC", np.full((n, n), 90000.0))
        put("SINALPHA", np.full((n, n), np.sin(alpha)))
        put("COSALPHA", np.full((n, n), np.cos(alpha)))
        if gust is not None:
            put("WSPD10MAX", np.full((n, n), gust))


def _run(tmp_path, hours=4, u=0.0, v=-5.0, d02_extent=True, alpha=0.0):
    """d01: 5x5 at 1 deg from (30N, 100E); d02: 5x5 at 0.1 deg from (32N, 102E).
    T2 = 273.15 + 10 + lat + 2*lon_offset (linear) on both."""
    run = tmp_path / "run"
    run.mkdir()
    for h in range(hours):
        ts = (T0 + timedelta(hours=h)).strftime("%Y-%m-%d_%H:%M:%S")
        fn = lambda la, lo: 273.15 + 10 + (la - 30) + 2 * (lo - 100)      # noqa: E731
        _write_wrfout(str(run / ("wrfout_d01_" + ts)), ts, 30.0, 100.0, 1.0, 5, fn, u, v,
                      hgt=100.0, gust=8.0, alpha=alpha)
        if d02_extent:
            _write_wrfout(str(run / ("wrfout_d02_" + ts)), ts, 32.0, 102.0, 0.1, 5, fn, u, v,
                          hgt=250.0, gust=9.0, alpha=alpha)
    return run


def _stations(tmp_path, rows, name="obs"):
    """rows: (id, lat, lon, elev, hour, var, value)."""
    d = tmp_path / name
    d.mkdir()
    with open(d / "stations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(agent_obs.EXTRA_COLS)
        for sid, la, lo, el, h, var, val in rows:
            w.writerow([sid, la, lo, el, (T0 + timedelta(minutes=h)).strftime(agent_obs.TIME_FMT),
                        var, val])
    sha = hashlib.sha256((d / "stations.csv").read_bytes()).hexdigest()
    (d / "provenance.json").write_text(json.dumps(dict(
        format=agent_obs.FORMAT, kind="stations", dataset="t", provider="p", url="u",
        license="l", fetched_utc="2026-01-01T00:00:00Z", files={"stations.csv": sha})))
    return d


# -- ASOS conversion --------------------------------------------------------------------------
ASOS_PROV = dict(dataset="IEM ASOS", provider="IEM", url="https://mesonet.agron.iastate.edu",
                 license="public domain")


def _asos(tmp_path, with_pos=True):
    """IEM layout: lon before lat, no elevation column, :55 observation minutes, upper-case M."""
    p = tmp_path / "asos.csv"
    pos = ",-119.8,34.4" if with_pos else ",,"
    p.write_text("\n".join([
        "station,valid,lon,lat,tmpf,dwpf,sknt,drct,gust",
        "SBA,2024-03-01 23:55" + pos + ",68.0,50.0,10.0,270.0,M",
        "SBA,2024-03-02 00:55" + pos + ",50.0,M,3.0,M,M",
        "SBA,2024-03-02 01:55" + pos + ",32,30,20,90,30.0"]) + "\n")
    return p


def test_from_asos_units_and_missing(tmp_path):
    out = tmp_path / "ds"
    path, n = agent_obs.from_asos(str(_asos(tmp_path)), str(out), ASOS_PROV)
    rows = agent_obs.read_stations(path)
    got = {(r["time"].hour, r["var"]): r["value"] for r in rows}
    assert got[(23, "t2")] == pytest.approx(20.0)
    assert got[(23, "td2")] == pytest.approx(10.0)
    assert got[(23, "wspd10")] == pytest.approx(10 * 1852 / 3600, abs=1e-3)
    assert got[(1, "gust")] == pytest.approx(30 * 1852 / 3600, abs=1e-3)
    assert (23, "gust") not in got and (0, "td2") not in got and (0, "wdir10") not in got
    assert got[(0, "wspd10")] == pytest.approx(3 * 1852 / 3600, abs=1e-3)
    assert n == len(rows) == 11
    assert rows[0]["lat"] == pytest.approx(34.4) and rows[0]["lon"] == pytest.approx(-119.8)
    assert rows[0]["elev_m"] is None
    err, warn = agent_obs.check(str(out))
    assert err == [] and any("elev_m" in w for w in warn)
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["kind"] == "stations" and prov["files"]["stations.csv"]


def test_from_asos_needs_position(tmp_path):
    with pytest.raises(ValueError, match="SBA.*lat/lon"):
        agent_obs.from_asos(str(_asos(tmp_path, False)), str(tmp_path / "ds"), ASOS_PROV)
    path, _n = agent_obs.from_asos(str(_asos(tmp_path, False)), str(tmp_path / "ds"), ASOS_PROV,
                                   {"SBA": (34.4, -119.8, 3.0)})
    r = agent_obs.read_stations(path)[0]
    assert r["lon"] == pytest.approx(-119.8) and r["elev_m"] == 3.0


def test_event_cli_obs_convert(tmp_path, capsys):
    csvp = _asos(tmp_path, False)  # no positions in the file
    rc = event.main(["obs-convert", "asos", str(csvp), "--out", str(tmp_path / "o"), "--dataset", "x",
                     "--provider", "p", "--url", "u", "--license", "l",
                     "--station", "SBA=34.4,-119.8,3"])
    assert rc == 0 and "OK" in capsys.readouterr().out
    assert event.main(["obs-convert", "asos", str(csvp), "--out", str(tmp_path / "o2"),
                       "--dataset", "x", "--provider", "p", "--url", "u", "--license", "l"]) == 1


def test_stations_vocabulary():
    assert "stations" in event.AGENT_READERS and "stations" in event.COMPARISON


# -- obs-check --------------------------------------------------------------------------------
def test_check_stations_ok_and_unit_warnings(tmp_path):
    ok = _stations(tmp_path, [("A", 34, -119, 5, 0, "t2", 15.0), ("A", 34, -119, 5, 0, "wspd10", 4.0),
                              ("A", 34, -119, 5, 0, "wdir10", 270), ("A", 34, -119, 5, 0, "slp", 1012)],
                   "ok")
    assert agent_obs.check(str(ok)) == ([], [])
    f = _stations(tmp_path, [("A", 34, -119, 5, m, "t2", 75.0) for m in range(0, 120, 10)] +
                  [("A", 34, -119, 5, m, "wspd10", 45.0) for m in range(0, 120, 10)], "bad")
    err, warn = agent_obs.check(str(f))
    assert err and any("outside" in e for e in err)
    assert any("degF" in w for w in warn) and any("knots" in w for w in warn)


def test_check_stations_unknown_var(tmp_path):
    d = _stations(tmp_path, [("A", 34, -119, 5, 0, "vis", 3.0)])
    err, _w = agent_obs.check(str(d))
    assert any("var must be one of" in e for e in err)


# -- statistics -------------------------------------------------------------------------------
def test_circular_difference():
    assert sc.circ_diff(350.0, 10.0) == pytest.approx(-20.0)
    assert sc.circ_diff(10.0, 350.0) == pytest.approx(20.0)
    assert sc.circ_diff(90.0, 270.0) == pytest.approx(-180.0)


def test_wind_dir_convention():
    assert sc.wind_dir(0.0, -5.0) == pytest.approx(0.0)       # southward flow = from the north
    assert sc.wind_dir(-5.0, 0.0) == pytest.approx(90.0)      # westward flow = from the east
    assert sc.wind_dir(5.0, 0.0) == pytest.approx(270.0)


def test_wdir_circular_bias_and_calm_exclusion(tmp_path):
    # model wind from 350 deg: u = -sin(350)*5? flow vector toward = from + 180
    sp, dr = 5.0, 350.0
    u, v = -sp * np.sin(np.radians(dr)), -sp * np.cos(np.radians(dr))
    run = _run(tmp_path, u=u, v=v)
    la, lo = 33.0, 101.0                                   # inside d01 only
    rows = [("A", la, lo, 100, 0, "wdir10", 10), ("A", la, lo, 100, 0, "wspd10", 5.0),
            ("A", la, lo, 100, 60, "wdir10", 330), ("A", la, lo, 100, 60, "wspd10", 5.0),
            ("A", la, lo, 100, 120, "wdir10", 90), ("A", la, lo, 100, 120, "wspd10", 1.0)]  # calm
    res = sc.compare(str(run), str(_stations(tmp_path, rows)), ["A"], ["wdir10", "wspd10"])
    st = {(s["domain"], s["variable"]): s for s in res["stats"]}[("d01", "wdir10")]
    assert st["n"] == 2 and st["n_calm_excluded"] == 1
    assert st["bias"] == pytest.approx((-20.0 + 20.0) / 2, abs=1e-3)
    assert st["rmse"] == pytest.approx(20.0, abs=1e-3)
    sp_st = {(s["domain"], s["variable"]): s for s in res["stats"]}[("d01", "wspd10")]
    assert sp_st["n"] == 3 and sp_st["bias"] == pytest.approx(((0) + 0 + 4.0) / 3, abs=1e-3)


def test_calm_threshold_is_in_knots(tmp_path):
    run = _run(tmp_path)
    rows = [("A", 33.0, 101.0, 100, 0, "wdir10", 10), ("A", 33.0, 101.0, 100, 0, "wspd10", 1.6)]
    ds = _stations(tmp_path, rows)
    # 1.6 m/s = 3.11 kt: not calm at 3 kt, calm at 4 kt
    assert sc.compare(str(run), str(ds), ["A"], ["wdir10"])["stats"][0]["n_calm_excluded"] == 0
    assert sc.compare(str(run), str(ds), ["A"], ["wdir10"], calm_kt=4.0)["stats"][0][
        "n_calm_excluded"] == 1


def test_exactly_3_kt_is_not_calm_despite_the_rounded_ms_value(tmp_path):
    run = _run(tmp_path)
    rows = [("A", 33.0, 101.0, 100, 0, "wdir10", 10), ("A", 33.0, 101.0, 100, 0, "wspd10", 1.543),  # 3.0 kt rounded
            ("A", 33.0, 101.0, 100, 60, "wdir10", 10), ("A", 33.0, 101.0, 100, 60, "wspd10", 1.49)]  # 2.9 kt
    ds = _stations(tmp_path, rows)
    st = [r for r in sc.compare(str(run), str(ds), ["A"], ["wdir10"])["stats"]]
    assert st[0]["n"] == 1 and st[0]["n_calm_excluded"] == 1


# -- domains and sampling ---------------------------------------------------------------------
def test_station_only_in_d02_gives_only_d02(tmp_path):
    run = _run(tmp_path)
    rows = [("B", 32.2, 102.2, 300, h, "t2", 20.0) for h in (0, 60)]
    res = sc.compare(str(run), str(_stations(tmp_path, rows)), ["B"], ["t2"], out_dir=str(tmp_path / "o"))
    # d01 contains it as well (d02 is nested), so both lines exist; a station outside d02 has d01 only
    assert {s["domain"] for s in res["stats"]} == {"d01", "d02"}
    rows = [("C", 31.2, 101.2, 10, 0, "t2", 20.0)]
    res = sc.compare(str(run), str(_stations(tmp_path, rows, "obs2")), ["C"], ["t2"])
    assert {s["domain"] for s in res["stats"]} == {"d01"}
    rows = [("D", 40.0, 100.5, 10, 0, "t2", 20.0)]
    res = sc.compare(str(run), str(_stations(tmp_path, rows, "obs3")), ["D"], ["t2"])
    assert res["stats"] == [] and any("outside every" in n for n in res["notes"])


def test_domains_filter(tmp_path):
    run = _run(tmp_path)
    ds = _stations(tmp_path, [("B", 32.2, 102.2, 300, 0, "t2", 20.0)])
    res = sc.compare(str(run), str(ds), ["B"], ["t2"], domains=[2])
    assert {s["domain"] for s in res["stats"]} == {"d02"}


def test_nearest_and_bilinear_on_linear_field(tmp_path):
    run = _run(tmp_path)
    # d01 point (31.3, 101.4): field = 10 + 1.3 + 2.8 = 14.1; nearest grid point (31, 101) = 13
    ds = _stations(tmp_path, [("E", 31.3, 101.4, 100, 0, "t2", 14.1)])
    nn = sc.compare(str(run), str(ds), ["E"], ["t2"], method="nearest")["series"][0]
    bl = sc.compare(str(run), str(ds), ["E"], ["t2"], method="bilinear")["series"][0]
    assert nn["model"] == pytest.approx(13.0, abs=1e-4)
    assert bl["model"] == pytest.approx(14.1, abs=1e-3) and bl["diff"] == pytest.approx(0, abs=1e-3)


def test_station_elevation_difference(tmp_path):
    run = _run(tmp_path)
    ds = _stations(tmp_path, [("F", 32.2, 102.2, 1000, 0, "t2", 20.0)])
    st = {s["domain"]: s for s in sc.compare(str(run), str(ds), ["F"], ["t2"])["stats"]}
    assert st["d02"]["model_hgt_minus_elev_m"] == pytest.approx(250 - 1000)
    assert st["d01"]["model_hgt_minus_elev_m"] == pytest.approx(100 - 1000)


def test_td2_and_gust_and_pairing_tolerance(tmp_path):
    run = _run(tmp_path)
    rows = [("G", 33.0, 101.0, 100, 5, "gust", 6.0),            # 5 min off, interval 60: paired
            ("G", 33.0, 101.0, 100, 95, "gust", 6.0),           # 35 min from 120 and 60 -> nearest 60: 35>30
            ("G", 33.0, 101.0, 100, 0, "td2", 0.0)]
    res = sc.compare(str(run), str(_stations(tmp_path, rows)), ["G"], ["gust", "td2"], domains=[1])
    g = [r for r in res["series"] if r["variable"] == "gust"]
    assert len(g) == 1 and g[0]["model"] == pytest.approx(8.0) and g[0]["diff"] == pytest.approx(2.0)
    td = [r for r in res["series"] if r["variable"] == "td2"][0]
    # w=0.004, p=900 hPa -> e = 5.75 hPa (saturation at 0 degC is 6.11) -> Td = -0.84 degC
    assert td["model"] == pytest.approx(-0.84, abs=0.05)


def test_grid_relative_wind_is_rotated(tmp_path):
    # grid rotated by 90 deg: grid u (east in grid) = north on the earth
    run = _run(tmp_path, u=5.0, v=0.0, alpha=np.pi / 2)
    ds = _stations(tmp_path, [("H", 33.0, 101.0, 100, 0, "wdir10", 180)])
    res = sc.compare(str(run), str(ds), ["H"], ["wdir10"], domains=[1])
    assert res["series"][0]["model"] == pytest.approx(180.0, abs=0.01)   # blowing from the south? no: toward north = from 180


def test_rejects_bad_input(tmp_path):
    run = _run(tmp_path)
    ds = _stations(tmp_path, [("A", 33, 101, 5, 0, "t2", 20.0)])
    with pytest.raises(sc.StationError, match="unsupported"):
        sc.compare(str(run), str(ds), ["A"], ["slp"])
    with pytest.raises(sc.StationError, match="method"):
        sc.compare(str(run), str(ds), ["A"], ["t2"], method="cubic")


# -- outputs ----------------------------------------------------------------------------------
def test_outputs_csv_columns_figures_and_index(tmp_path):
    pytest.importorskip("matplotlib")
    run = _run(tmp_path)
    rows = [("B", 32.2, 102.2, 300, h, v, x) for h in (0, 60, 120)
            for v, x in (("t2", 20.0), ("wdir10", (350 + 10 * (h // 60)) % 360), ("wspd10", 6.0))]
    out = tmp_path / "post"
    res = sc.compare(str(run), str(_stations(tmp_path, rows)), ["B"], ["t2", "wdir10"],
                     out_dir=str(out))
    head = next(csv.reader(open(out / "station_series.csv")))
    assert head[:7] == ["station", "domain", "variable", "time_utc", "obs", "model", "diff"]
    head = next(csv.reader(open(out / "station_stats.csv")))
    assert head == list(sc.STATS_COLS)
    assert len(res["figures"]) == 2 and all(os.path.isfile(f) for f in res["figures"])
    doc = json.loads((out / "figures.json").read_text())
    assert [f["id"] for f in doc["figures"]] == ["station_series"] * 2
    assert all(os.path.isfile(out / f["path"]) for f in doc["figures"])
    from wtune.post import recipe
    assert "station_series" in recipe.load_catalog()
    # a second run replaces, not duplicates, and keeps other entries
    doc["figures"].append(dict(id="other", missing=False, path="x.png"))
    (out / "figures.json").write_text(json.dumps(doc))
    sc.compare(str(run), str(_stations(tmp_path, rows, "o2")), ["B"], ["t2"], out_dir=str(out))
    ids = [f["id"] for f in json.loads((out / "figures.json").read_text())["figures"]]
    assert ids.count("station_series") == 1 and "other" in ids


def test_pairing_of_minute_55_obs_and_missing_elevation(tmp_path):
    """ASOS :55 observations pair with the next hourly output; empty elev_m gives empty height
    difference; a missing direction (calm) is skipped."""
    pytest.importorskip("matplotlib")
    run = _run(tmp_path)
    src = tmp_path / "a.csv"
    lines = ["station,valid,lon,lat,tmpf,dwpf,sknt,drct,gust"]
    for h in range(3):       # 2024-03-01 23:55 is before the run, the others are :55 of hours 0, 1
        t = (T0 + timedelta(hours=h - 1, minutes=55)).strftime("%Y-%m-%d %H:%M")
        lines.append("ABC,%s,101.0,33.0,50.0,M,%s,%s,M" % (t, "2.0" if h == 2 else "10.0",
                                                        "M" if h == 2 else "350.0"))
    src.write_text("\n".join(lines) + "\n")
    ds = tmp_path / "ds"
    agent_obs.from_asos(str(src), str(ds), ASOS_PROV)
    res = sc.compare(str(run), str(ds), ["ABC"], ["t2", "wdir10"], domains=[1],
                     out_dir=str(tmp_path / "o"))
    t2 = [r for r in res["series"] if r["variable"] == "t2"]
    assert [r["time_utc"] for r in t2] == ["2024-03-01 00:00", "2024-03-01 01:00",
                                           "2024-03-01 02:00"][:len(t2)] and len(t2) == 3
    wd = [r for r in res["series"] if r["variable"] == "wdir10"]
    assert len(wd) == 2                                   # the 'M' direction is not a row
    st = [s for s in res["stats"] if s["variable"] == "wdir10"][0]
    assert st["n"] == 2 and st["model_hgt_minus_elev_m"] is None
    row = list(csv.DictReader(open(tmp_path / "o" / "station_stats.csv")))[0]
    assert row["model_hgt_minus_elev_m"] == ""
    assert any("no elev_m" in n for n in res["notes"])


def test_outermost_ring_not_compared_and_grid_columns(tmp_path):
    run = _run(tmp_path)
    ds = _stations(tmp_path, [("R", 30.1, 100.1, 10, 0, "t2", 20.0),
                              ("A", 33.0, 101.0, 10, 0, "t2", 20.0)])
    res = sc.compare(str(run), str(ds), ["R", "A"], ["t2"], domains=[1])
    assert [s["station"] for s in res["stats"]] == ["A"]
    s = res["stats"][0]
    # (33N, 101E): row 3 (south_north), column 1 (west_east)
    assert (s["grid_j"], s["grid_i"], s["model_hgt_m"]) == (3, 1, 100.0)
    assert any("outermost" in n for n in res["notes"])


def test_default_pairing_tolerance_is_10_minutes(tmp_path):
    run = _run(tmp_path)
    rows = [("A", 33.0, 101.0, 10, 8, "t2", 20.0), ("A", 33.0, 101.0, 10, 70, "t2", 20.0),
            ("A", 33.0, 101.0, 10, 125, "t2", 20.0), ("A", 33.0, 101.0, 10, 111, "t2", 20.0)]
    res = sc.compare(str(run), str(_stations(tmp_path, rows)), ["A"], ["t2"], domains=[1])
    assert [r["time_utc"] for r in res["series"]] == ["2024-03-01 00:00", "2024-03-01 01:00",
                                                      "2024-03-01 02:00"]


def test_cli_stations(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from wtune.post import cli
    run = _run(tmp_path)
    ds = _stations(tmp_path, [("A", 33.0, 101.0, 10, 0, "t2", 20.0)])
    rc = cli.main(["stations", "--run", str(run), "--obs", str(ds), "--stations", "A",
                   "--vars", "t2", "--out", str(tmp_path / "post")])
    assert rc == 0 and (tmp_path / "post" / "station_stats.csv").is_file()
    assert "bias=" in capsys.readouterr().out


def test_default_is_every_station_and_variable_and_a_subset_is_flagged(tmp_path):
    run = _run(tmp_path)
    rows = [("A", 33.0, 101.0, 100, 0, "t2", 10), ("A", 33.0, 101.0, 100, 0, "wspd10", 5),
            ("B", 33.5, 101.5, 100, 0, "t2", 11), ("B", 33.5, 101.5, 100, 0, "slp", 1000)]
    ds = _stations(tmp_path, rows)
    full = sc.compare(str(run), str(ds), None, None)
    assert {(r["station"], r["variable"]) for r in full["stats"]} >= {("A", "t2"), ("A", "wspd10"), ("B", "t2")}
    assert any("slp" in n and "NOT compared" in n for n in full["notes"])
    part = sc.compare(str(run), str(ds), ["A"], ["t2"])
    assert any("NOT analysed" in n and "B" in n for n in part["notes"])
    assert any("wspd10" in n and "NOT compared" in n for n in part["notes"])

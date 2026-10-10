"""post/fetch/ framework (pkgutil discovery, contract check, CLI wiring, offline summary) and
the era5 module (idempotence, coverage guard, missing-subset hint). No network: the ARCO store
is replaced by a small in-memory xarray Dataset."""
import argparse
import os
from datetime import datetime

import numpy as np
import pytest

from wtune.post import cli, fetch, sources
from wtune.post.fetch import era5 as fetch_era5
from wtune.post.recipes import era5 as recipe_era5


def test_discovery_finds_era5_and_every_module_meets_the_contract():
    mods, errors = fetch.discover(refresh=True)
    assert "era5" in mods
    assert not errors, errors
    for name, m in mods.items():
        assert m.NAME == name and isinstance(m.HELP, str) and m.HELP
        assert set(fetch.REQUIRED_PROVENANCE) <= set(m.PROVENANCE)
        assert fetch.check_module(m) == []


def test_check_module_reports_contract_breaks():
    class Bad:
        NAME = "bad"
        PROVENANCE = {"dataset": "x"}
    problems = fetch.check_module(Bad)
    assert any("missing HELP" in p for p in problems)
    assert any("PROVENANCE lacks" in p for p in problems)


def test_list_text_has_name_help_and_coverage(capsys):
    assert cli.main(["fetch", "--list"]) == 0
    out = capsys.readouterr().out
    assert "era5" in out and "ERA5 z500 subset" in out and "coverage_time: 1940" in out


def test_fetch_and_alias_parse_module_args():
    p = cli.build_parser()
    a = p.parse_args(["fetch", "era5", "--project", "P", "--run-dir", "R", "--margin", "3"])
    assert (a.post_command, a.fetch_name, a.project, a.run_dir, a.margin) == ("fetch", "era5", "P", "R", 3.0)
    b = p.parse_args(["fetch-era5", "--project", "P", "--spinup-hours", "12"])
    assert (b.post_command, b.fetch_name, b.spinup_hours) == ("fetch-era5", "era5", 12.0)
    with pytest.raises(SystemExit):
        p.parse_args(["fetch", "era5"])                      # --project is required


def test_unknown_module_is_a_clear_error():
    with pytest.raises(SystemExit) as e:
        fetch.run_module("nope", argparse.Namespace())
    assert "unknown fetch module" in str(e.value)


def _project_with_run(tmp_path):
    run = tmp_path / "proj" / "runs" / "m1"
    run.mkdir(parents=True)
    (run / "wrfout_d01_2005-08-27_00:00:00").write_text("")
    return str(tmp_path / "proj"), str(run)


def test_offline_summary_lists_missing_datasets_with_commands(tmp_path):
    proj, run = _project_with_run(tmp_path)
    line = fetch.offline_summary(proj, run, online=False)
    assert line and "no internet" in line
    for name in fetch.discover()[0]:
        assert "`wtune post fetch %s --project %s --run-dir %s`" % (name, proj, run) in line
    assert fetch.offline_summary(proj, run, online=True) is None
    assert fetch.offline_summary(None, run, online=False) is None


def test_offline_summary_drops_datasets_already_local(tmp_path):
    proj, run = _project_with_run(tmp_path)
    d = sources.dataset_dir(proj, "stage4")
    os.makedirs(d)
    open(os.path.join(d, "a.grib"), "w").write("x")
    sources.write_provenance(proj, "stage4", fetch.discover()[0]["stage4"].PROVENANCE, [], {"a.grib": "0"})
    line = fetch.offline_summary(proj, run, online=False)
    assert "stage4" not in line.split("run on the login node")[0]


def test_missing_subset_reason_names_the_login_node_command(tmp_path, monkeypatch):
    proj, run = _project_with_run(tmp_path)
    monkeypatch.setattr(fetch, "has_internet", lambda timeout=2.0: False)
    with pytest.raises(recipe_era5.Era5UnavailableError) as e:
        recipe_era5.z500_for(datetime(2005, 8, 27, 6), (20, 30, -90, -80), proj)
    msg = str(e.value)
    assert "wtune post fetch-era5 --project %s --run-dir %s" % (proj, run) in msg
    assert "login node" in msg


def test_gcsfs_failure_also_names_the_command(tmp_path, monkeypatch):
    proj, run = _project_with_run(tmp_path)
    monkeypatch.setattr(fetch, "has_internet", lambda timeout=2.0: True)

    def boom():
        raise recipe_era5.Era5UnavailableError("ERA5 (ARCO): could not open the zarr store: dns")
    monkeypatch.setattr(recipe_era5, "open_era5", boom)
    with pytest.raises(recipe_era5.Era5UnavailableError) as e:
        recipe_era5.z500_for(datetime(2005, 8, 27, 6), (20, 30, -90, -80), proj)
    assert "could not open the zarr store" in str(e.value) and "fetch-era5" in str(e.value)


# ---- era5 fetch against a fake store -------------------------------------------------------

def _fake_arco(start="2005-08-26", hours=72):
    xr = pytest.importorskip("xarray")
    times = np.datetime64(start) + np.arange(hours) * np.timedelta64(1, "h")
    lat = np.arange(90, -90.5, -2.0)      # coarse: keep the fake store a few MB
    lon = np.arange(0, 360, 2.0)
    z = np.random.default_rng(0).uniform(4.5e4, 5.9e4, (len(times), 3, len(lat), len(lon))).astype("f4")
    return xr.Dataset(dict(geopotential=(("time", "level", "latitude", "longitude"), z)),
                      coords=dict(time=times, level=[400, 500, 600], latitude=lat, longitude=lon))


@pytest.fixture
def era5_env(tmp_path, monkeypatch):
    pytest.importorskip("netCDF4")
    proj, run = _project_with_run(tmp_path)
    ds = _fake_arco()
    calls = {"open": 0}

    def fake_open():
        calls["open"] += 1
        return ds
    monkeypatch.setattr(recipe_era5, "open_era5", fake_open)
    monkeypatch.setattr(fetch_era5, "_target", lambda args: (
        (datetime(2005, 8, 27, 6), (24.0, 26.0, -86.0, -84.0)), None) if args.run_dir else (None, "--run-dir is required"))
    args = argparse.Namespace(project=proj, run_dir=run, spinup_hours=6.0, margin=2.0)
    return args, calls


def test_era5_fetch_writes_subset_and_provenance_then_is_idempotent(era5_env):
    args, calls = era5_env
    paths = fetch_era5.run(args)
    assert len(paths) == 1 and paths[0].endswith("input/era5/z500_2005082706.nc")
    lat, lon, z, valid, = recipe_era5.read_subset(paths[0])
    assert valid == datetime(2005, 8, 27, 6) and z.shape == (len(lat), len(lon))
    prov = sources.read_provenance(args.project, "era5")
    assert prov["urls"] and prov["dataset"] == fetch_era5.PROVENANCE["dataset"]
    assert prov["files"]["z500_2005082706.nc"] == sources.sha256_file(paths[0])
    assert prov["window"]["start"] == "2005-08-27T06:00:00Z" and prov["bbox"] == [24.0, 26.0, -86.0, -84.0]
    fetched = prov["fetched_utc"]
    assert calls["open"] == 1
    mtime = os.path.getmtime(paths[0])

    assert fetch_era5.run(args) == paths                      # second run: nothing downloaded
    assert calls["open"] == 1 and os.path.getmtime(paths[0]) == mtime
    assert sources.read_provenance(args.project, "era5")["fetched_utc"] == fetched


def test_era5_refetches_a_corrupt_file(era5_env):
    args, calls = era5_env
    (path,) = fetch_era5.run(args)
    with open(path, "wb") as fh:
        fh.write(b"truncated")
    assert fetch_era5.run(args) == [path] and calls["open"] == 2
    recipe_era5.read_subset(path)


def test_era5_outside_coverage_returns_empty_with_reason(era5_env, monkeypatch, capsys):
    args, _ = era5_env
    monkeypatch.setattr(fetch_era5, "_target", lambda a: ((datetime(1939, 8, 27, 6), (24, 26, -86, -84)), None))
    assert fetch_era5.run(args) == []
    out = capsys.readouterr().out
    assert "outside the ARCO-ERA5 time coverage" in out
    assert not os.path.exists(os.path.join(args.project, "input", "era5", "z500_1939082706.nc"))


def test_era5_without_run_dir_says_why(capsys):
    assert fetch_era5.run(argparse.Namespace(project="/x", run_dir=None, spinup_hours=6.0, margin=2.0)) == []
    assert "--run-dir is required" in capsys.readouterr().out

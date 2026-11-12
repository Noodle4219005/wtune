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
    assert "era5" in out and "ERA5 z500 and 850 hPa" in out and "coverage_time: 1940" in out


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
        if name == "bdeck":                     # no storm in this project: nothing to fetch
            assert "fetch bdeck" not in line
            continue
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
    shape = (len(times), 4, len(lat), len(lon))
    rng = np.random.default_rng(0)
    dims = ("time", "level", "latitude", "longitude")
    return xr.Dataset(dict(geopotential=(dims, rng.uniform(4.5e4, 5.9e4, shape).astype("f4")),
                           u_component_of_wind=(dims, rng.uniform(-20, 20, shape).astype("f4")),
                           v_component_of_wind=(dims, rng.uniform(-20, 20, shape).astype("f4")),
                           specific_humidity=(dims, rng.uniform(0, 0.02, shape).astype("f4"))),
                      coords=dict(time=times, level=[400, 500, 600, 850], latitude=lat,
                                  longitude=lon))


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
    monkeypatch.setattr(fetch_era5, "_targets", lambda args: (
        ([datetime(2005, 8, 27, 6)], (24.0, 26.0, -86.0, -84.0)), None) if args.run_dir else (None, "--run-dir is required"))
    args = argparse.Namespace(project=proj, run_dir=run, spinup_hours=6.0, margin=2.0)
    return args, calls


def test_era5_fetch_writes_subset_and_provenance_then_is_idempotent(era5_env):
    args, calls = era5_env
    paths = fetch_era5.run(args)
    assert [os.path.basename(p) for p in paths] == ["z500_2005082706.nc", "lev850_2005082706.nc"]
    lat, lon, z, valid, = recipe_era5.read_subset(paths[0])
    assert valid == datetime(2005, 8, 27, 6) and z.shape == (len(lat), len(lon))
    lat, lon, fields, valid = recipe_era5.read_level(paths[1])
    assert sorted(fields) == ["Q850", "U850", "V850"] and valid == datetime(2005, 8, 27, 6)
    assert 0.0 <= np.nanmax(fields["Q850"]) <= 20.0                         # g/kg
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
    path, lev = fetch_era5.run(args)
    with open(path, "wb") as fh:
        fh.write(b"truncated")
    assert fetch_era5.run(args) == [path, lev] and calls["open"] == 2
    recipe_era5.read_subset(path)


def test_era5_outside_coverage_returns_empty_with_reason(era5_env, monkeypatch, capsys):
    args, _ = era5_env
    monkeypatch.setattr(fetch_era5, "_targets", lambda a: (([datetime(1939, 8, 27, 6)], (24, 26, -86, -84)), None))
    assert fetch_era5.run(args) == []
    out = capsys.readouterr().out
    assert "outside the ARCO-ERA5 time coverage" in out
    assert not os.path.exists(os.path.join(args.project, "input", "era5", "z500_1939082706.nc"))


def test_era5_without_run_dir_says_why(capsys):
    assert fetch_era5.run(argparse.Namespace(project="/x", run_dir=None, spinup_hours=6.0, margin=2.0)) == []
    assert "--run-dir is required" in capsys.readouterr().out


def test_tests_have_no_network():
    # conftest._no_network: compute nodes have no internet, so neither do the tests
    import urllib.request
    with pytest.raises(OSError):
        urllib.request.urlopen("http://93.184.216.34/", timeout=5)


def test_tests_keep_loopback_and_block_names():
    import socket
    import threading
    for host in ("127.0.0.1", "127.0.0.2", "::ffff:127.0.0.1"):
        fam = socket.AF_INET6 if ":" in host else socket.AF_INET
        srv = socket.socket(fam)
        srv.bind(("::" if fam == socket.AF_INET6 else "0.0.0.0", 0))
        srv.listen(1)
        threading.Thread(target=lambda: srv.accept()[0].close(), daemon=True).start()
        c = socket.socket(fam)
        c.settimeout(5)
        c.connect((host, srv.getsockname()[1]))      # loopback in any spelling stays open
        c.close()
        srv.close()
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 80)
    with pytest.raises(OSError):
        socket.socket().connect_ex(("93.184.216.34", 80))


def test_bdeck_storm_name_without_hurdat2_is_a_clear_error(tmp_path):
    from wtune.post import tc
    from wtune.post.fetch import bdeck
    (tmp_path / "project.yaml").write_text("storm:\n  id: KATRINA\n")
    assert bdeck.present(str(tmp_path)) is False
    with pytest.raises(tc.TCError, match="hurdat2"):
        bdeck._storm(str(tmp_path))


def test_bdeck_of_this_year_is_refetched_and_kept_when_the_refetch_fails(tmp_path, monkeypatch):
    from wtune.post import atcf
    from wtune.post.fetch import bdeck
    year = datetime.utcnow().year
    calls = []

    def fetch(y, basin, cy, dest, timeout=30):
        calls.append(dest)
        if len(calls) > 1:
            raise atcf.ATCFError("NHC unreachable")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "w").write("AL, 05, %d090100,   , BEST,   0, 200N,  600W,  50, 1000\n" % year)
        return dest
    monkeypatch.setattr(atcf, "fetch_bdeck", fetch)
    args = argparse.Namespace(project=str(tmp_path), storm="AL05%d" % year)
    first = bdeck.run(args)
    assert first and bdeck.run(args) == first and len(calls) == 2


def test_fetch_without_run_dir_uses_the_projects_newest_run(tmp_path, monkeypatch, capsys):
    # 2026-10-26: `wtune post fetch era5 --project P` fetched 0 files -- era5/igra/stations
    # need the run's wrfout for the window and region, and --run-dir was not derived
    import argparse
    from wtune.post import cli, fetch
    proj = tmp_path / "p"
    for rid in ("20261026113729", "merged72h"):
        (proj / "runs" / rid).mkdir(parents=True)
        (proj / "runs" / rid / "wrfout_d01_2017-07-31_00:00:00").write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(fetch, "run_module", lambda name, a: seen.setdefault("run_dir", a.run_dir) and [])
    args = argparse.Namespace(fetch_name="era5", project=str(proj), run_dir=None, list=False)
    assert cli.cmd_fetch(args) == 0
    assert seen["run_dir"] == str(proj / "runs" / "merged72h")
    assert "using the project's" in capsys.readouterr().out


def test_fetch_finds_the_project_it_runs_in(tmp_path, monkeypatch):
    # haitang2017 (2026-10-27): `wtune post fetch era5` inside the project still demanded --project
    from wtune.post import cli, fetch
    import wtune.project
    p = wtune.project.init_project(str(tmp_path / "p"))
    monkeypatch.chdir(p)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    seen = {}
    monkeypatch.setattr(fetch, "run_module", lambda name, args: seen.update(project=args.project) or [])
    args = cli.build_parser().parse_args(["fetch", "era5"])
    assert cli.cmd_fetch(args) == 0 and seen["project"] == p
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="no project here"):
        cli.cmd_fetch(cli.build_parser().parse_args(["fetch", "era5"]))

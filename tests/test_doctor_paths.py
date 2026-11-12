"""`wtune doctor` path checks, `--find` and `--json`: what an agent needs to set the paths
of a new machine by itself."""
import json
import stat

import pytest

from wtune import doctor


@pytest.fixture(autouse=True)
def _no_slurm(monkeypatch):
    # a login node would answer sinfo for real (the suite counts such calls)
    monkeypatch.setattr(doctor, "sinfo_partitions", lambda: None)


def _tree(tmp_path):
    wrf = tmp_path / "W"
    (wrf / "main").mkdir(parents=True)
    for exe in ("wrf.exe", "real.exe"):
        f = wrf / "main" / exe
        f.write_text("#!/bin/sh\n")
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    wps = tmp_path / "wps"
    wps.mkdir()
    for exe in ("geogrid.exe", "ungrib.exe", "metgrid.exe"):
        (wps / exe).write_text("")
    (tmp_path / "geog" / "topo_gmted2010_30s").mkdir(parents=True)
    return wrf, wps


def _env(monkeypatch, tmp_path, text, wrf_tree):
    env = tmp_path / "env.sh"
    env.write_text(text)
    monkeypatch.setenv("WTUNE_ENV_SOURCE", str(env))
    monkeypatch.setenv("WRF_TREE", str(wrf_tree))
    monkeypatch.setenv("WPS_GEOG", str(tmp_path / "geog"))
    monkeypatch.setenv("WTUNE_SITE", "scc")
    return env


def _by_level(checks, level):
    return [what for lv, what, _fix in checks if lv == level]


def test_a_complete_setup_passes_the_path_checks(tmp_path, monkeypatch):
    wrf, wps = _tree(tmp_path)
    adios2 = tmp_path / "adios2"
    adios2.mkdir()
    (adios2 / "pixi.toml").write_text("")
    coast = tmp_path / "ne" / "shapefiles" / "natural_earth" / "physical"
    coast.mkdir(parents=True)
    (coast / "ne_50m_coastline.shp").write_text("")
    _env(monkeypatch, tmp_path, "export WRF_TREE=%s WPS_DIR=%s WTUNE_PIXI_ENV=%s "
         "WTUNE_CARTOPY_DATA=%s\n" % (wrf, wps, adios2, tmp_path / "ne"), wrf)
    checks = doctor.path_checks()
    assert not _by_level(checks, "FAIL") and not _by_level(checks, "WARN"), checks


def test_an_empty_natural_earth_folder_is_warned_about(tmp_path, monkeypatch):
    wrf, wps = _tree(tmp_path)
    (tmp_path / "ne").mkdir()                     # a folder with no coastline shapefile in it
    _env(monkeypatch, tmp_path, "export WRF_TREE=%s WPS_DIR=%s WTUNE_CARTOPY_DATA=%s\n"
         % (wrf, wps, tmp_path / "ne"), wrf)
    warns = _by_level(doctor.path_checks(), "WARN")
    assert any("no coastline shapefile" in w for w in warns), warns


def test_the_env_file_and_machine_wrf_tree_must_agree(tmp_path, monkeypatch):
    # tune jobs source the env file, wtune run takes machine.yaml's wrf_tree (docs/guide/f1-arm.md)
    wrf, wps = _tree(tmp_path)
    _env(monkeypatch, tmp_path, "export WRF_TREE=%s/other WPS_DIR=%s\n" % (tmp_path, wps), wrf)
    fails = _by_level(doctor.path_checks(), "FAIL")
    assert any("WRF_TREE differs" in f for f in fails)
    assert any("no main/wrf.exe under %s/other (env file)" % tmp_path in f for f in fails)


def test_a_missing_env_file_or_unset_wrf_tree_fails_with_a_fix(tmp_path, monkeypatch):
    wrf, _wps = _tree(tmp_path)
    monkeypatch.setenv("WTUNE_ENV_SOURCE", str(tmp_path / "none.sh"))
    monkeypatch.setenv("WRF_TREE", str(wrf))
    checks = doctor.path_checks()
    assert [fix for lv, what, fix in checks if lv == "FAIL" and "does not exist" in what][0]
    _env(monkeypatch, tmp_path, "export WPS_DIR=/x\n", wrf)
    assert any("does not set WRF_TREE" in f for f in _by_level(doctor.path_checks(), "FAIL"))


def test_libraries_wrf_exe_cannot_load_fail_and_are_searched_for(tmp_path, monkeypatch, capsys):
    wrf, wps = _tree(tmp_path)
    _env(monkeypatch, tmp_path, "export WRF_TREE=%s WPS_DIR=%s\n" % (wrf, wps), wrf)
    lib = tmp_path / "spack" / "view" / "lib"
    lib.mkdir(parents=True)
    (lib / "libnetcdff.so.7").write_text("")
    real = doctor._sourced

    def fake(env_file, site_file, names):
        got = real(env_file, site_file, names)
        got["ldd_missing"] = ["libnetcdff.so.7"]
        return got
    monkeypatch.setattr(doctor, "_sourced", fake)
    assert any("cannot load libnetcdff.so.7" in f for f in _by_level(doctor.path_checks(), "FAIL"))
    assert doctor.main(["--json", "--find", "--find-root", str(tmp_path)]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and {"level", "what", "fix"} <= set(out["checks"][0])
    assert out["found"]["libs"] == {"libnetcdff.so.7": [str(lib)]}
    assert out["found"]["wrf_tree"] == [str(wrf)] and out["found"]["wps_dir"] == [str(wps)]
    assert out["found"]["geog"] == [str(tmp_path / "geog")]


def test_find_skips_environments_and_stops_on_time(tmp_path):
    hidden = tmp_path / ".pixi" / "W" / "main"
    hidden.mkdir(parents=True)
    (hidden / "wrf.exe").write_text("")
    assert doctor.find_paths([str(tmp_path)])["wrf_tree"] == []
    assert doctor.find_paths([str(tmp_path)], seconds=-1)["stopped"] is True


def test_find_lists_only_env_files_that_set_wrf_tree(tmp_path):
    d = tmp_path / "wtune-f1arm"
    d.mkdir()
    (d / "env.sh").write_text("module load nvhpc\nexport WRF_TREE=/w LD_LIBRARY_PATH=/l\n")
    (d / "env-old.sh").write_text("WRF_TREE=/w2; export WRF_TREE\n")
    (d / "envcheck.sh").write_text("echo $WRF_TREE\ntest -d \"$WRF_TREE\"\n")
    got = doctor.find_paths([str(tmp_path)])["env_files"]
    assert sorted(got) == [str(d / "env-old.sh"), str(d / "env.sh")]


def test_a_partition_sinfo_does_not_list_fails(monkeypatch):
    monkeypatch.setattr(doctor, "sinfo_partitions", lambda: {"arm144", "arm576"})
    monkeypatch.setenv("WTUNE_PARTITION", "compute")
    assert any("partition compute not in sinfo" in f
               for f in _by_level(doctor.path_checks(), "FAIL"))

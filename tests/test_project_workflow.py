"""The project-driven workflow: project.yaml defaults reach tune, run and post, and
`wtune submit` chains the three steps."""
import json
import os
import time

import pytest
import yaml

import wtune_project as proj
import wtune_submit
from post import cli as post_cli
from wtune_run import cli as run_cli

NAMELIST = """&time_control
 run_days = 0, run_hours = 72, run_minutes = 0, run_seconds = 0,
 start_year = 2019, start_month = 01, start_day = 28, start_hour = 00,
 end_year = 2019, end_month = 01, end_day = 31, end_hour = 00,
 history_interval = 60, 60,
/
&domains
 time_step = 10,
 max_dom = 2,
 e_we = 150, 100,
 e_sn = 150, 100,
 e_vert = 50, 50,
 dx = 9000,
 dy = 9000,
/
&physics
 mp_physics = 4, 4,
 radt = 9, 9,
/
"""


@pytest.fixture
def case_dir(tmp_path):
    d = tmp_path / "mycase"
    d.mkdir()
    (d / "namelist.input").write_text(NAMELIST)
    return str(d)


@pytest.fixture
def project(tmp_path, case_dir, monkeypatch):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work1")
    return proj.init_project(str(tmp_path / "p"), case_dir=case_dir)


def _doc(project_dir):
    with open(os.path.join(project_dir, "project.yaml")) as fh:
        return yaml.safe_load(fh)


def test_init_registers_case_and_writes_defaults(project):
    doc = _doc(project)
    assert doc["case"]["id"] == "mycase"
    assert doc["exclude_nodes"] == ["work1"]
    assert doc["post"] == {"types": "auto", "every": "6h",       # 72 h run
                           "fetch": ["era5", "igra", "stage4", "stations"]}
    with open(proj.project_case_table(project)) as fh:
        entry = json.load(fh)["cases"]["mycase"]
    assert entry["prep_dir"] == os.path.join(project, "case")
    assert (entry["dt"], entry["max_dom"]) == (10, 2)
    assert (entry["e_we_list"], entry["e_sn_list"], entry["e_vert_list"]) == (
        [150, 100], [150, 100], [50, 50])
    assert entry["physics"] == {"mp_physics": 4, "radt": 9}     # keys the probe job reads
    assert entry["case_family"] == "unclassified"


def test_registered_name_gets_proj_suffix(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    d = tmp_path / "sc26p1"                                     # an official case id
    d.mkdir()
    (d / "namelist.input").write_text(NAMELIST)
    p = proj.init_project(str(tmp_path / "p"), case_dir=str(d))
    assert _doc(p)["case"]["id"] == "sc26p1-proj"
    assert _doc(p)["exclude_nodes"] == []


def test_long_run_gets_no_default_every(tmp_path):
    d = tmp_path / "climate"
    d.mkdir()
    (d / "namelist.input").write_text(NAMELIST.replace("run_hours = 72", "run_hours = 240"))
    p = proj.init_project(str(tmp_path / "p"), case_dir=str(d))
    assert _doc(p)["post"]["every"] is None
    assert _doc(p)["post"]["fetch"] == []


def test_apply_project_env_merges_exclude_and_sets_case_table(project, monkeypatch):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work9")
    monkeypatch.delenv("CASES_JSON", raising=False)
    proj.apply_project_env(project, _doc(project))
    assert os.environ["WTUNE_SBATCH_EXCLUDE"] == "work9,work1"
    assert os.environ["CASES_JSON"] == proj.project_case_table(project)


def test_set_defaults_and_register_case(project):
    proj.set_project_defaults(project, exclude_nodes="work1,work9", post_every="none")
    doc = _doc(project)
    assert doc["exclude_nodes"] == ["work1", "work9"]
    assert doc["post"]["every"] is None
    os.remove(proj.project_case_table(project))
    proj.set_project_defaults(project, register_case=True)
    assert os.path.isfile(proj.project_case_table(project))


def _report(project_dir, name, mtime):
    d = os.path.join(project_dir, "tune", name)
    os.makedirs(d)
    path = os.path.join(d, "tune_report.json")
    with open(path, "w") as fh:
        json.dump({"winner": {"nodes": 2, "nx": 4, "ny": 4, "numtiles": 1, "omp": 1}}, fh)
    os.utime(path, (mtime, mtime))
    return path


def test_run_defaults_to_project_case_and_newest_report(project):
    now = time.time()
    _report(project, "old", now - 100)
    newest = _report(project, "new", now)
    opts = run_cli.parse_options(["--project", project])
    opts = run_cli.apply_project_defaults(opts, log=lambda *_: None)
    assert opts.case_dir == os.path.join(project, "case")
    assert opts.tune_report == newest


def test_run_explicit_nodes_or_other_case_keeps_no_report(project, tmp_path):
    _report(project, "t", time.time())
    opts = run_cli.apply_project_defaults(
        run_cli.parse_options(["--project", project, "--nodes", "3"]), log=lambda *_: None)
    assert opts.tune_report is None
    other = tmp_path / "other"
    other.mkdir()
    opts = run_cli.apply_project_defaults(
        run_cli.parse_options([str(other), "--project", project]), log=lambda *_: None)
    assert opts.tune_report is None


def test_run_without_case_outside_project_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(run_cli.RunError, match="CASE_DIR is required"):
        run_cli.apply_project_defaults(run_cli.parse_options([]), log=lambda *_: None)


def test_submit_steps_share_one_id(project):
    steps = wtune_submit.plan_steps(project, "ID", "2", skip_tune=False, no_post=False)
    assert [name for name, _cmd in steps] == ["tune", "run", "post"]
    tune, run, post = (cmd for _name, cmd in steps)
    assert tune[1:3] == ["tune", "-O2"] and tune[-1] == os.path.join(project, "tune", "ID")
    assert run[run.index("--tune-report") + 1] == os.path.join(project, "tune", "ID",
                                                               "tune_report.json")
    assert post[3] == os.path.join(project, "runs", "ID") and post[-1] == "--srun"
    steps = wtune_submit.plan_steps(project, "ID", "2", skip_tune=True, no_post=True)
    assert [name for name, _cmd in steps] == ["run"]
    assert "--tune-report" not in steps[0][1]


def test_fetch_steps_window_the_case_and_sit_between_run_and_post(project):
    import datetime as dt
    window = (dt.datetime(2019, 1, 28), dt.datetime(2019, 1, 31), (35.1, 47.5, -93.6, -77.2))
    fetch = wtune_submit.fetch_steps(project, ["era5", "stage4"], window)
    era5, stage4 = (cmd for _name, cmd in fetch)
    assert era5[2:4] == ["fetch", "era5"] and "--start" not in era5
    assert era5[era5.index("--end") + 1] == "2019013100"
    assert stage4[stage4.index("--start") + 1] == "2019012800"
    assert stage4[stage4.index("--bbox") + 1] == "35.100,47.500,-93.600,-77.200"
    steps = wtune_submit.plan_steps(project, "ID", "2", False, False, fetch)
    assert [n for n, _c in steps] == ["tune", "run", "fetch era5", "fetch stage4", "post"]
    steps = wtune_submit.plan_steps(project, "ID", "2", False, True, fetch)
    assert [n for n, _c in steps] == ["tune", "run"]


def test_case_window_from_namelist_and_wrfinput(project):
    netCDF4 = pytest.importorskip("netCDF4")
    import numpy as np
    case = os.path.join(project, "case")
    with netCDF4.Dataset(os.path.join(case, "wrfinput_d01"), "w") as ds:
        ds.createDimension("Time", 1), ds.createDimension("y", 2), ds.createDimension("x", 2)
        ds.createVariable("XLAT", "f4", ("Time", "y", "x"))[:] = np.array([[[30, 30], [40, 40]]])
        ds.createVariable("XLONG", "f4", ("Time", "y", "x"))[:] = np.array([[[-90, -80],
                                                                               [-90, -80]]])
    start, end, bbox = wtune_submit.case_window(case)
    assert (start.isoformat(), end.isoformat()) == ("2019-01-28T00:00:00", "2019-01-31T00:00:00")
    assert bbox == (30.0, 40.0, -90.0, -80.0)


def test_submit_refuses_outside_project(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert wtune_submit.main(["--dry-run"]) == 2
    assert "no project here" in capsys.readouterr().err


def test_post_srun_command_excludes_head_and_project_nodes(monkeypatch):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work1")
    cmd = post_cli.srun_command(["recipe", "runs/r1"])
    assert cmd[:2] == ["srun", "--exclude=head,work1"]
    assert cmd[-3:] == ["post", "recipe", "runs/r1"]

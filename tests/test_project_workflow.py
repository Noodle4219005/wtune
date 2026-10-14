"""The project-driven workflow: project.yaml defaults reach tune, run and post, and
`wtune submit` chains the three steps."""
import argparse
import json
import os
import re
import time

import pytest
import yaml

from wtune import project as proj
import wtune.machine
import wtune.submit
from wtune.post import cli as post_cli
from wtune.run import cli as run_cli

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
    case_id = doc["case"]["id"]
    assert re.fullmatch(r"mycase-[0-9a-f]{6}", case_id)
    assert doc["exclude_nodes"] == ["work1"]
    assert doc["post"] == {"types": "auto", "every": "6h",       # 72 h run
                           "fetch": ["era5", "igra", "stage4", "stations"]}
    with open(proj.project_case_table(project)) as fh:
        table = json.load(fh)["cases"]
    assert list(table) == [case_id]                             # only the project's own case
    entry = table[case_id]
    assert entry["prep_dir"] == os.path.join(project, "case")
    assert (entry["dt"], entry["max_dom"]) == (10, 2)
    assert (entry["e_we_list"], entry["e_sn_list"], entry["e_vert_list"]) == (
        [150, 100], [150, 100], [50, 50])
    assert entry["physics"] == {"mp_physics": 4, "radt": 9}     # keys the probe job reads
    assert entry["case_family"] == "unclassified"


def test_case_id_is_safe_and_distinct_per_case(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    ids = []
    for parent, text in (("a", NAMELIST), ("b", NAMELIST),
                         ("a2", NAMELIST.replace("time_step = 10", "time_step = 12"))):
        d = tmp_path / parent / "my_run.v2"                     # same dir name, '_' and '.'
        d.mkdir(parents=True)
        (d / "namelist.input").write_text(text)
        p = proj.init_project(str(tmp_path / ("p" + parent)), case_dir=str(d))
        ids.append(_doc(p)["case"]["id"])
        assert _doc(p)["exclude_nodes"] == []
    assert all(re.fullmatch(r"my-run-v2-[0-9a-f]{6}", i) for i in ids)   # no '_' (run-id split)
    assert len(set(ids)) == 3                 # the measurement ledger is keyed by this id


def test_init_validates_arguments_before_creating_anything(tmp_path, case_dir):
    target = tmp_path / "p"
    with pytest.raises(proj.ProjectError, match="hurdat2"):
        proj.init_project(str(target), case_dir=case_dir, hurdat2=str(tmp_path / "nope"))
    assert not target.exists()


def test_long_run_gets_no_default_every(tmp_path):
    d = tmp_path / "climate"
    d.mkdir()
    (d / "namelist.input").write_text(NAMELIST.replace("run_hours = 72", "run_hours = 240"))
    p = proj.init_project(str(tmp_path / "p"), case_dir=str(d))
    assert _doc(p)["post"]["every"] is None
    assert _doc(p)["post"]["fetch"] == []


def test_apply_project_env_merges_exclude_and_sets_case_table(project, monkeypatch):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work9")
    monkeypatch.setenv("CASES_JSON", "")      # empty = unset; setenv (not delenv) so it is restored
    proj.apply_project_env(project, _doc(project))
    assert os.environ["WTUNE_SBATCH_EXCLUDE"] == "work9,work1"
    with open(os.environ["CASES_JSON"]) as fh:
        merged = json.load(fh)["cases"]
    assert _doc(project)["case"]["id"] in merged
    assert "sc26p1" in merged                 # the shared table is read fresh, not snapshotted


def test_project_env_keeps_a_user_set_case_table(project, monkeypatch):
    monkeypatch.setenv("CASES_JSON", "/elsewhere/cases.json")
    proj.apply_project_env(project, _doc(project))
    assert os.environ["CASES_JSON"] == "/elsewhere/cases.json"


def test_register_case_error_is_a_project_error(project):
    os.remove(os.path.join(project, "case", "namelist.input"))
    with pytest.raises(proj.ProjectError, match="register-case"):
        proj.set_project_defaults(project, register_case=True)


def test_set_defaults_and_register_case(project):
    proj.set_project_defaults(project, exclude_nodes="work1,work9", post_every="none")
    doc = _doc(project)
    assert doc["exclude_nodes"] == ["work1", "work9"]
    assert doc["post"]["every"] is None
    os.remove(proj.project_case_table(project))
    proj.set_project_defaults(project, register_case=True)
    assert os.path.isfile(proj.project_case_table(project))


def _machine():
    m = wtune.machine
    return m.cores_per_node(), m.max_nodes(), m.partition(), m.login_nodes()


@pytest.fixture
def no_machine_env(monkeypatch):
    for name in wtune.machine.ENV.values():
        monkeypatch.setenv(name, "x")  # recorded, so the original state comes back afterwards
        monkeypatch.delenv(name)


def test_machine_block_reaches_every_command(project, no_machine_env, monkeypatch):
    proj.set_project_defaults(project, machine=dict(cores_per_node=64, max_nodes=20,
                                                    partition="batch", login_nodes=""))
    assert _doc(project)["machine"] == dict(cores_per_node=64, max_nodes=20, partition="batch",
                                            login_nodes=[])
    monkeypatch.setenv("WTUNE_PARTITION", "debug")       # set by hand: kept
    proj.apply_project_env(project, _doc(project))
    assert _machine() == (64, 20, "debug", [])
    assert ("machine: cores_per_node=64 max_nodes=20 partition=batch login_nodes=- account=-"
            in proj.info_project(project))


def test_without_a_machine_block_the_built_in_values_apply(project, no_machine_env):
    proj.apply_project_env(project, _doc(project))
    assert _machine() == (72, 13, "compute", ["head"])
    assert ("machine: cores_per_node=72 max_nodes=13 partition=compute login_nodes=head"
            in proj.info_project(project))


def test_account_goes_on_every_job_of_the_project(project, no_machine_env, monkeypatch):
    assert proj.main(["set", "--project", project, "--account", "proj42"]) == 0
    proj.apply_project_env(project, _doc(project))
    assert wtune.machine.account_arg() == "--account=proj42"
    cmd = post_cli.srun_command(["recipe", "runs/r1"])
    assert cmd[:5] == ["srun", "-p", "compute", "--exclude=head,work1", "--account=proj42"]


def test_machine_description_and_calibration_in_the_project(project, no_machine_env, tmp_path):
    cal = tmp_path / "newbox.json"
    cal.write_text("{}")
    assert proj.main(["set", "--project", project, "--hardware-id", "newbox",
                      "--numa-per-node", "2", "--calibration", str(cal)]) == 0
    assert _doc(project)["machine"] == dict(hardware_id="newbox", numa_per_node=2,
                                            calibration=str(cal))
    proj.apply_project_env(project, _doc(project))
    assert (wtune.machine.hardware_id(), wtune.machine.numa_per_node(),
            wtune.machine.calibration()) == ("newbox", 2, str(cal))
    assert proj.main(["set", "--project", project, "--calibration", str(cal) + ".missing"]) == 1


def test_project_set_user_writes_the_settings_every_project_shares(
        project, no_machine_env, monkeypatch, tmp_path, capsys):
    path = tmp_path / "machine.yaml"
    monkeypatch.setenv("WTUNE_MACHINE_FILE", str(path))
    assert proj.main(["set", "--user", "--partition", "mine", "--max-nodes", "4"]) == 0
    assert yaml.safe_load(path.read_text()) == dict(max_nodes=4, partition="mine")
    assert "partition=mine" in proj.info_project(project)            # no project block yet
    proj.set_project_defaults(project, machine=dict(partition="here"))
    assert "max_nodes=4 partition=here" in proj.info_project(project)
    assert "project.yaml" not in capsys.readouterr().out             # --user leaves it alone
    assert proj.main(["set", "--user"]) == 1
    assert "at least one machine flag" in capsys.readouterr().err


def test_measured_numbers_and_the_longest_job_in_the_project(project, no_machine_env, capsys):
    assert proj.main(["set", "--project", project, "--max-hours", "8",
                      "--measured", "noise_frac=0.05", "--measured", "init_s=60"]) == 0
    assert proj.main(["set", "--project", project, "--measured", "io_factor=3"]) == 0
    assert _doc(project)["machine"] == dict(
        max_hours=8.0, measured=dict(noise_frac=0.05, init_s=60.0, io_factor=3.0))
    assert ("max_hours=8 max_omp=0 node_preflight=on measured=init_s=60,io_factor=3,noise_frac=0.05"
            in proj.info_project(project))
    proj.apply_project_env(project, _doc(project))
    assert (wtune.machine.max_hours(), wtune.machine.measured("io_factor")) == (8.0, 3.0)
    capsys.readouterr()
    assert proj.main(["set", "--project", project, "--measured", "nope"]) == 1
    assert "NAME=NUMBER" in capsys.readouterr().err


def test_project_set_machine_flags(project, capsys):
    assert proj.main(["set", "--project", project, "--max-nodes", "8",
                      "--login-nodes", "login1,login2"]) == 0
    assert _doc(project)["machine"] == {"max_nodes": 8, "login_nodes": ["login1", "login2"]}
    assert ("machine: cores_per_node=72 max_nodes=8 partition=compute login_nodes=login1,login2"
            in capsys.readouterr().out)
    assert proj.main(["set", "--project", project, "--cores-per-node", "0"]) == 1
    assert "machine.cores_per_node: 0 must be a positive integer" in capsys.readouterr().err
    assert _doc(project)["machine"] == {"max_nodes": 8, "login_nodes": ["login1", "login2"]}


@pytest.mark.parametrize("block, message", [
    ({"cores": 64}, "unknown key"), ({"cores_per_node": 0}, "positive integer"),
    ({"max_nodes": "many"}, "positive integer"), ({"partition": " "}, "must be a name"),
    ({"login_nodes": 3}, "list of node names"), ("compute", "must be a mapping"),
    ({"numa_per_node": 0}, "positive integer"), ({"hardware_id": ""}, "must be a name"),
    ({"calibration": "scc.json"}, "absolute path of an existing file")])
def test_a_bad_machine_block_is_a_project_error(block, message):
    with pytest.raises(proj.ProjectError, match=message):
        proj.project_machine({"machine": block})


def _report(project_dir, name, mtime, case=None):
    d = os.path.join(project_dir, "tune", name)
    os.makedirs(d)
    path = os.path.join(d, "tune_report.json")
    with open(path, "w") as fh:
        json.dump({"case": case or _doc(project_dir)["case"]["id"],
                   "winner": {"nodes": 2, "nx": 4, "ny": 4, "numtiles": 1, "omp": 1}}, fh)
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


def test_run_skips_a_newer_report_of_another_case(project):
    now = time.time()
    own = _report(project, "own", now - 100)
    _report(project, "other", now, case="some-other-case")
    skipped = []
    opts = run_cli.apply_project_defaults(run_cli.parse_options(["--project", project]),
                                          log=skipped.append)
    assert opts.tune_report == own
    assert any("some-other-case" in line for line in skipped)
    assert proj.latest_tune_report(project, _doc(project)["case"]["id"]) == own


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
    steps = wtune.submit.plan_steps(project, "ID", "2", skip_tune=False, no_post=False)
    assert [name for name, _cmd in steps] == ["tune", "run", "post"]
    tune, run, post = (cmd for _name, cmd in steps)
    assert tune[1:3] == ["tune", "-O2"] and tune[-1] == os.path.join(project, "tune", "ID")
    assert run[run.index("--tune-report") + 1] == os.path.join(project, "tune", "ID",
                                                               "tune_report.json")
    assert post[3] == os.path.join(project, "runs", "ID") and post[-1] == "--srun"
    steps = wtune.submit.plan_steps(project, "ID", "2", skip_tune=True, no_post=True)
    assert [name for name, _cmd in steps] == ["run"]
    assert "--tune-report" not in steps[0][1]


def test_submit_tunes_with_the_projects_own_config(project):
    assert "--config" not in wtune.submit.plan_steps(project, "ID", "2", False, True)[0][1]
    config = os.path.join(project, "wtune.yaml")
    with open(config, "w") as f:
        f.write("search:\n  omp: [1]\n")
    tune = wtune.submit.plan_steps(project, "ID", "2", False, True)[0][1]
    assert tune[tune.index("--config") + 1] == config


def test_fetch_steps_window_the_case_and_sit_between_run_and_post(project):
    import datetime as dt
    window = (dt.datetime(2019, 1, 28), dt.datetime(2019, 1, 31, 6), (-35.1, 47.5, -93.6, -77.2),
              dt.datetime(2019, 1, 31))
    fetch = wtune.submit.fetch_steps(project, ["era5", "stage4"], window)
    era5, stage4 = (cmd for _name, cmd in fetch)
    assert era5[2:4] == ["fetch", "era5"] and "--start" not in era5
    assert era5[era5.index("--end") + 1] == "201901310000"      # last history frame, not run end
    assert stage4[stage4.index("--start") + 1] == "2019012800"
    assert stage4[stage4.index("--end") + 1] == "2019013106"
    # one argv element: a leading '-' (southern hemisphere) must not parse as an option
    assert "--bbox=-35.100,47.500,-93.600,-77.200" in stage4 and "--bbox" not in stage4
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox")
    assert ap.parse_args([a for a in stage4 if a.startswith("--bbox")]).bbox.startswith("-35.1")
    steps = wtune.submit.plan_steps(project, "ID", "2", False, False, fetch)
    assert [n for n, _c in steps] == ["tune", "run", "fetch era5", "fetch stage4", "post"]
    steps = wtune.submit.plan_steps(project, "ID", "2", False, True, fetch)
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
    start, end, bbox, last = wtune.submit.case_window(case)
    assert (start.isoformat(), end.isoformat()) == ("2019-01-28T00:00:00", "2019-01-31T00:00:00")
    assert bbox == (30.0, 40.0, -90.0, -80.0) and last == end


def test_case_window_last_frame_and_odd_time_step(project):
    netCDF4 = pytest.importorskip("netCDF4")
    case = os.path.join(project, "case")
    path = os.path.join(case, "namelist.input")
    text = open(path).read().replace("run_hours = 72", "run_hours = 30").replace(
        "history_interval = 60, 60", "history_interval = 720, 720").replace(
        "time_step = 10", "time_step = 7")     # 30 h is not a multiple of 7 s: still a window
    with open(path, "w") as fh:
        fh.write(text)
    with netCDF4.Dataset(os.path.join(case, "wrfinput_d01"), "w") as ds:
        ds.createDimension("Time", 1), ds.createDimension("y", 1), ds.createDimension("x", 1)
        ds.createVariable("XLAT", "f4", ("Time", "y", "x"))[:] = 30.0
        ds.createVariable("XLONG", "f4", ("Time", "y", "x"))[:] = -90.0
    start, end, _bbox, last = wtune.submit.case_window(case)
    assert (end - start).total_seconds() == 30 * 3600
    assert (last - start).total_seconds() == 24 * 3600          # frames at +0, +12, +24 h


def test_detach_child_gets_the_resolved_project(project, monkeypatch):
    seen = {}

    class FakePopen:
        pid = 1

        def __init__(self, cmd, **kw):
            seen["cmd"], seen["cwd"] = cmd, kw["cwd"]
    monkeypatch.setattr(wtune.submit.subprocess, "Popen", FakePopen)
    wtune.submit._detach(["--project", "./rel", "--detach", "-O", "1", "--project=./rel2"],
                         project, "ID")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--project") + 1] == project and cmd.count("--project") == 1
    assert "./rel" not in cmd and "--project=./rel2" not in cmd and "--detach" not in cmd
    assert cmd[-2:] == ["-O", "1"] and seen["cwd"] == project


def test_submit_refuses_outside_project(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    assert wtune.submit.main(["--dry-run"]) == 2
    assert "no project here" in capsys.readouterr().err


def test_post_srun_command_excludes_head_and_project_nodes(monkeypatch):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work1")
    cmd = post_cli.srun_command(["recipe", "runs/r1"])
    assert cmd[:4] == ["srun", "-p", "compute", "--exclude=head,work1"]
    assert cmd[-3:] == ["post", "recipe", "runs/r1"]


def test_submit_passes_compress_history_to_the_run(project):
    steps = dict(wtune.submit.plan_steps(project, "ID", "2", True, True, compress_history=True))
    assert "--compress-history" in steps["run"]
    assert "--compress-history" not in dict(wtune.submit.plan_steps(project, "ID", "2", True,
                                                                    True))["run"]

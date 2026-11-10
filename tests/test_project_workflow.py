"""The project-driven workflow: project.yaml defaults reach tune, run and post, and
`wtune submit` chains the three steps."""
import argparse
import json
import os
import time

import pytest
import yaml

from wtune import project as proj
import wtune.machine
import wtune.submit
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
    assert ("max_hours=8 max_omp=0 node_preflight=on node_memory_gib=0 copy_probe_inputs=on exclusive=on hardware_cores_per_node=0 measured=init_s=60,io_factor=3,noise_frac=0.05"
            in proj.info_project(project))
    proj.apply_project_env(project, _doc(project))
    assert (wtune.machine.max_hours(), wtune.machine.measured("io_factor")) == (8.0, 3.0)
    capsys.readouterr()
    assert proj.main(["set", "--project", project, "--measured", "nope"]) == 1
    assert "NAME=NUMBER" in capsys.readouterr().err


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


def test_run_without_case_outside_project_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(run_cli.RunError, match="CASE_DIR is required"):
        run_cli.apply_project_defaults(run_cli.parse_options([]), log=lambda *_: None)


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


def test_submit_passes_compress_history_to_the_run(project):
    steps = dict(wtune.submit.plan_steps(project, "ID", "2", True, True, compress_history=True))
    assert "--compress-history" in steps["run"]
    assert "--compress-history" not in dict(wtune.submit.plan_steps(project, "ID", "2", True,
                                                                    True))["run"]


def test_init_keeps_the_wrfrst_of_a_case_without_wrfinput(tmp_path, case_dir, monkeypatch):
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    rst = ["wrfrst_d%02d_2019-01-28_00:00:00" % d for d in (1, 2)]
    for name in rst + ["wrfbdy_d01", "wrfrst_d01_2019-01-29_00:00:00"]:
        open(os.path.join(case_dir, name), "w").close()
    p = proj.init_project(str(tmp_path / "p"), case_dir=case_dir)
    case = os.path.join(p, "case")
    assert sorted(n for n in os.listdir(case) if n.startswith("wrfrst")) == rst  # not the later one
    with open(os.path.join(case, "namelist.input")) as fh:
        assert "restart = .true." in fh.read()
    with open(os.path.join(case_dir, "namelist.input")) as fh:
        assert "restart" not in fh.read()                           # the source is never written
    with open(proj.project_case_table(p)) as fh:
        entry = list(json.load(fh)["cases"].values())[0]
    assert (entry["restart"], entry["input_files"], entry["restart_time"],
            entry["restart_io_form"]) == (True, rst, "2019-01-28_00:00:00", 2)


def test_submit_passes_run_options_and_the_space_check_bypass(project):
    """2026-10-14: --stall-minutes, --checkpoint-minutes and --resume reached only `wtune run`."""
    args = wtune.submit.build_parser().parse_args(
        ["--resume", "0", "--checkpoint-minutes", "60", "--stall-minutes", "5", "--no-space-check"])
    assert wtune.submit.run_args(args) == ["--resume", "0", "--checkpoint-minutes", "60",
                                           "--stall-minutes", "5"]
    steps = dict(wtune.submit.plan_steps(project, "ID", "2", False, True, None, False,
                                         wtune.submit.run_args(args), args.no_space_check))
    run = " ".join(steps["run"])
    assert "--resume 0 --checkpoint-minutes 60 --stall-minutes 5 --no-space-check" in run
    assert steps["tune"][-1] == "--no-space-check"
    assert wtune.submit.run_args(wtune.submit.build_parser().parse_args([])) == []

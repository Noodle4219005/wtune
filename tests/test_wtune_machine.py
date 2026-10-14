"""wtune.machine: the built-in scc values, what replaces them, and that the generated jobs
follow."""
import os

import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
import wtune.config
import wtune.machine
import wtune.nodes
from wtune import predict
from wtune.predictor import decomp_space, features
from wtune.run import layout, sbatch
from wtune.tools import scc_sweep


def _unset(monkeypatch, name):
    monkeypatch.setenv(name, "x")      # recorded, so the original state comes back afterwards
    monkeypatch.delenv(name)


@pytest.fixture
def scc(monkeypatch):
    for name in list(wtune.machine.ENV.values()) + ["WTUNE_SBATCH_EXCLUDE",
                                                   "WTUNE_PREFLIGHT_EXCLUDE"]:
        _unset(monkeypatch, name)


@pytest.fixture
def other_machine(scc, monkeypatch):
    for name, value in (("WTUNE_CORES_PER_NODE", "64"), ("WTUNE_MAX_NODES", "20"),
                        ("WTUNE_PARTITION", "batch"), ("WTUNE_LOGIN_NODES", ""),
                        ("WTUNE_ACCOUNT", "proj42")):
        monkeypatch.setenv(name, value)


def _machine():
    m = wtune.machine
    return m.cores_per_node(), m.max_nodes(), m.partition(), m.login_nodes()


def test_built_in_values_are_scc(scc):
    assert _machine() == (72, 13, "compute", ["head"])


def test_environment_replaces_them(other_machine):
    assert _machine() == (64, 20, "batch", [])


@pytest.mark.parametrize("bad", ["0", "-4", "many", "7.5"])
def test_a_bad_count_is_refused(scc, monkeypatch, bad):
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", bad)
    with pytest.raises(wtune.machine.MachineError, match="WTUNE_CORES_PER_NODE"):
        wtune.machine.cores_per_node()


def test_segment_job_asks_for_that_machine(other_machine):
    text = sbatch.render_segment_sbatch("/seg", 0, layout.make_config(8, 16, 1, 1, nodes=2),
                                        "01:00:00", "job")
    assert ("#SBATCH -p batch -N 2 --exclusive --ntasks-per-node=64 --time=01:00:00 "
            "--account=proj42\n") in text
    assert "--exclude" not in text


def test_segment_job_on_scc_is_unchanged(scc):
    text = sbatch.render_segment_sbatch("/seg", 0, layout.make_config(12, 12, 1, 1, nodes=2),
                                        "01:00:00", "job")
    assert ("#SBATCH -p compute -N 2 --exclusive --ntasks-per-node=72 --time=01:00:00 "
            "--exclude=head\n") in text
    assert "--account" not in text


def test_layout_is_checked_against_the_node_size(other_machine):
    layout.make_config(8, 8, 1, 1, nodes=1)
    with pytest.raises(layout.RunError, match=r"1 node\(s\) x 64 cores"):
        layout.make_config(5, 13, 1, 1, nodes=1)


def test_probe_job_asks_for_that_machine(other_machine):
    assert scc_sweep.exclude_arg() == ""
    cmd = scc_sweep.build_sbatch_command("c", 2, wtune.machine.cores_per_node(), "00:10:00",
                                         600, 1, 20, "/c.json", "/out")
    assert "sbatch -p batch -N 2 --ntasks-per-node=64 --exclusive --account=proj42 " in cmd
    assert "--exclude" not in cmd


def test_config_node_cap_follows_the_machine(other_machine):
    cfg, errors = wtune.config.resolve_config({})
    assert not errors and cfg["budget"]["max_nodes"] == 20
    cfg, errors = wtune.config.resolve_config({"budget": {"max_nodes": 6}})
    assert not errors and cfg["budget"]["max_nodes"] == 6


CASE = {"e_we": [150], "e_sn": [130], "max_dom": 1}


def test_a_machine_the_table_does_not_know_is_described_by_its_settings(scc, monkeypatch):
    for name, value in (("WTUNE_HARDWARE_ID", "newbox"), ("WTUNE_CORES_PER_NODE", "144"),
                        ("WTUNE_NUMA_PER_NODE", "2")):
        monkeypatch.setenv(name, value)
    assert features.hardware("newbox") == dict(cores_per_node=144, cores_per_numa=72,
                                               numa_per_node=2)
    assert features.hardware("scc-gh200")["cores_per_node"] == 72      # the table still answers
    assert features.hardware("elsewhere") == {}
    assert decomp_space.default_omp_options(144, 1, "newbox") == [1, 2, 3, 4, 6, 8, 9, 12, 18,
                                                                  24, 36, 72]
    with pytest.raises(ValueError, match="unknown hardware_id 'elsewhere'"):
        decomp_space.default_omp_options(144, 1, "elsewhere")


def test_settings_that_contradict_a_known_machine_are_refused(scc, monkeypatch):
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", "64")           # still called scc-gh200
    with pytest.raises(ValueError, match="give this machine its own machine.hardware_id"):
        features.hardware("scc-gh200")
    monkeypatch.setenv("WTUNE_HARDWARE_ID", "odd")
    monkeypatch.setenv("WTUNE_NUMA_PER_NODE", "3")
    with pytest.raises(ValueError, match="not a multiple of numa_per_node=3"):
        features.hardware("odd")


def test_calibration_file_follows_the_machine(scc, monkeypatch, tmp_path):
    assert predict.calibration_path() == str(predict.DEFAULT_CALIBRATION)
    mine = tmp_path / "newbox.json"
    mine.write_text('{"site": "newbox", "cores_per_node": 144}')
    monkeypatch.setenv("WTUNE_CALIBRATION", str(mine))
    assert predict.calibration_path() == str(mine)
    assert predict.cores_per_node(predict.load_calibration(predict.calibration_path())) == 144


def test_jobs_name_the_site_and_source_the_machine_environment(scc, monkeypatch):
    monkeypatch.setenv("WTUNE_SITE", "mysite")
    monkeypatch.setenv("WTUNE_ENV_SOURCE", "/opt/env.sh")
    text = sbatch.render_segment_sbatch("/seg", 0, layout.make_config(12, 12, 1, 1, nodes=2),
                                        "01:00:00", "job")
    assert "\nSITE=mysite\n" in text and '\nsource "/opt/env.sh"\n' in text
    cmd = scc_sweep.build_sbatch_command("c", 2, 72, "00:10:00", 600, 1, 20, "/c.json", "/out")
    assert cmd.startswith("source /opt/env.sh; sbatch ") and "ALL,SITE=mysite," in cmd


def test_user_file_sits_between_the_built_in_values_and_the_project(scc, monkeypatch, tmp_path):
    path = tmp_path / "cfg" / "machine.yaml"
    monkeypatch.setenv("WTUNE_MACHINE_FILE", str(path))
    assert wtune.machine.load_user() == {}
    wtune.machine.save_user(dict(partition="mine", max_nodes=4))
    assert wtune.machine.load_user() == dict(partition="mine", max_nodes=4)
    wtune.machine.export(dict(partition="project"))     # the dispatcher exports the project first
    wtune.machine.export(wtune.machine.load_user())
    assert (wtune.machine.partition(), wtune.machine.max_nodes()) == ("project", 4)
    path.write_text("cores: 9\n")
    with pytest.raises(wtune.machine.MachineError, match="unknown key"):
        wtune.machine.load_user()


def test_paths_of_the_machine(scc, monkeypatch):
    assert wtune.machine.geog() == "/nfs/data/wrf/WPS_GEOG/"
    wtune.machine.export(dict(geog="~/geog", wrf_tree="$HOME/wrf", env_source="$HOME/env.sh"))
    home = os.path.expanduser("~")
    assert (wtune.machine.geog(), wtune.machine.wrf_tree()) == (home + "/geog", home + "/wrf")
    assert wtune.machine.env_source() == "$HOME/env.sh"      # sourced by a shell: kept as written


def test_measured_numbers_can_be_replaced_per_machine(scc):
    measured = wtune.machine.measured
    rows = [dict(nodes=1, median=1.0), dict(nodes=2, median=0.96)]
    assert measured("noise_frac") == 0.03 and not wtune.nodes.stop_early(rows)    # 4% > 3%
    wtune.machine.export(dict(measured={"noise_frac": 0.05}))
    assert measured("noise_frac") == 0.05 and wtune.nodes.stop_early(rows)
    wtune.machine.export(dict(measured={"noise_frac": 0.2, "init_s": 60.0}))   # exported later
    assert (measured("noise_frac"), measured("init_s")) == (0.05, 60.0)
    assert measured("io_factor") == 2.0                                        # untouched
    with pytest.raises(wtune.machine.MachineError, match="unknown name"):
        wtune.machine.check({"measured": {"nope": 1}})
    with pytest.raises(wtune.machine.MachineError, match="must be a number"):
        wtune.machine.check({"measured": {"init_s": "fast"}})


def test_longest_job_of_the_partition_caps_time_limits(scc, monkeypatch):
    assert wtune.machine.max_hours() == 0 and wtune.machine.cap_seconds(1e6) == 1e6
    assert scc_sweep.time_cap_hours() == 12
    monkeypatch.setenv("WTUNE_MAX_HOURS", "8")
    assert wtune.machine.cap_seconds(1e6) == 8 * 3600 and scc_sweep.time_cap_hours() == 8
    monkeypatch.setenv("WTUNE_MAX_HOURS", "soon")
    with pytest.raises(wtune.machine.MachineError, match="WTUNE_MAX_HOURS"):
        wtune.machine.max_hours()


def test_max_omp_caps_threads_together_with_a_case_cap(monkeypatch):
    monkeypatch.setenv("WTUNE_MAX_OMP", "")                 # restored afterwards; "" = not set
    assert wtune.machine.max_omp() == 0 and wtune.machine.omp_cap() is None
    assert wtune.machine.omp_cap(1) == 1                    # the case's own cap alone
    wtune.machine.export(wtune.machine.check({"max_omp": 2}))
    assert os.environ["WTUNE_MAX_OMP"] == "2"
    caps = (wtune.machine.omp_cap(), wtune.machine.omp_cap(1), wtune.machine.omp_cap(4))
    assert caps == (2, 1, 2)
    with pytest.raises(wtune.machine.MachineError, match="number of threads"):
        wtune.machine.check({"max_omp": -1})
    monkeypatch.setenv("WTUNE_MAX_OMP", "two")
    with pytest.raises(wtune.machine.MachineError, match="WTUNE_MAX_OMP"):
        wtune.machine.max_omp()


def test_node_preflight_can_be_turned_off_for_the_machine(monkeypatch):
    monkeypatch.setenv("WTUNE_NODE_PREFLIGHT", "")          # restored afterwards; "" = not set
    assert wtune.machine.node_preflight() is True
    wtune.machine.export(wtune.machine.check({"node_preflight": False}))
    assert os.environ["WTUNE_NODE_PREFLIGHT"] == "0" and wtune.machine.node_preflight() is False
    with pytest.raises(wtune.machine.MachineError, match="true or false"):
        wtune.machine.check({"node_preflight": "no"})
    monkeypatch.setenv("WTUNE_NODE_PREFLIGHT", "maybe")
    with pytest.raises(wtune.machine.MachineError, match="WTUNE_NODE_PREFLIGHT"):
        wtune.machine.node_preflight()


def test_a_refused_partition_moves_the_job_to_the_next(scc, monkeypatch):
    wtune.machine.export(wtune.machine.check({"partition": ["small", "mid", "large"]}))
    assert wtune.machine.partitions() == ["small", "mid", "large"]
    assert wtune.machine.partition() == "small"
    seen = []

    def sbatch_runner(cmd):
        seen.append(cmd)
        if " -p large " not in cmd:
            raise RuntimeError("sbatch failed (rc=1): Job violates accounting/QOS policy")
        return "Submitted batch job 7\n"

    cmd = scc_sweep.build_sbatch_command("c", 4, 72, "00:10:00", 600, 1, 20, "/c.json", "/out")
    assert scc_sweep.submit_job(cmd, sbatch_runner) == "7"
    assert [c.split("; ")[1].split()[:3] for c in seen] == [
        ["sbatch", "-p", "small"], ["sbatch", "-p", "mid"], ["sbatch", "-p", "large"]]
    seen.clear()
    assert scc_sweep.submit_job("sbatch /x/nc2bp.sbatch", sbatch_runner) == "7"
    assert seen[-1] == "sbatch -p large /x/nc2bp.sbatch"


def test_every_partition_refusing_is_reported(scc):
    wtune.machine.export(dict(partition="small,mid"))

    def sbatch_runner(cmd):
        raise RuntimeError("refused by " + cmd.split()[2])

    with pytest.raises(RuntimeError, match="small: refused by small; mid: refused by mid"):
        scc_sweep.submit_job("sbatch /seg.sbatch", sbatch_runner)
    with pytest.raises(wtune.machine.MachineError, match="list of names"):
        wtune.machine.check({"partition": []})

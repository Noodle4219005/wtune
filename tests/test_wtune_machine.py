"""wtune.machine: the built-in scc values, what replaces them, and that the generated jobs
follow."""
import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
import wtune.config
import wtune.machine
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

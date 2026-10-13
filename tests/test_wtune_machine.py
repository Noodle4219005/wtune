"""wtune.machine: the built-in scc values, what replaces them, and that the generated jobs
follow."""
import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
import wtune.config
import wtune.machine
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
                        ("WTUNE_PARTITION", "batch"), ("WTUNE_LOGIN_NODES", "")):
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
    assert "#SBATCH -p batch -N 2 --exclusive --ntasks-per-node=64 --time=01:00:00\n" in text
    assert "--exclude" not in text


def test_segment_job_on_scc_is_unchanged(scc):
    text = sbatch.render_segment_sbatch("/seg", 0, layout.make_config(12, 12, 1, 1, nodes=2),
                                        "01:00:00", "job")
    assert ("#SBATCH -p compute -N 2 --exclusive --ntasks-per-node=72 --time=01:00:00 "
            "--exclude=head\n") in text


def test_layout_is_checked_against_the_node_size(other_machine):
    layout.make_config(8, 8, 1, 1, nodes=1)
    with pytest.raises(layout.RunError, match=r"1 node\(s\) x 64 cores"):
        layout.make_config(5, 13, 1, 1, nodes=1)


def test_probe_job_asks_for_that_machine(other_machine):
    assert scc_sweep.exclude_arg() == ""
    cmd = scc_sweep.build_sbatch_command("c", 2, wtune.machine.cores_per_node(), "00:10:00",
                                         600, 1, 20, "/c.json", "/out")
    assert "sbatch -p batch -N 2 --ntasks-per-node=64 --exclusive " in cmd
    assert "--exclude" not in cmd


def test_config_node_cap_follows_the_machine(other_machine):
    cfg, errors = wtune.config.resolve_config({})
    assert not errors and cfg["budget"]["max_nodes"] == 20
    cfg, errors = wtune.config.resolve_config({"budget": {"max_nodes": 6}})
    assert not errors and cfg["budget"]["max_nodes"] == 6

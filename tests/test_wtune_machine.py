"""wtune.machine: the built-in scc values, what replaces them, and that the generated jobs
follow."""
import os

import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
import wtune.config
import wtune.machine
import wtune.nodes
from wtune.predictor import decomp_space, features
from wtune.tools import scc_sweep


def _unset(monkeypatch, name):
    monkeypatch.setenv(name, "x")      # recorded, so the original state comes back afterwards
    monkeypatch.delenv(name)


@pytest.fixture
def scc(monkeypatch):
    for name in list(wtune.machine.ENV.values()) + ["WTUNE_SBATCH_EXCLUDE",
                                                   "WTUNE_PREFLIGHT_EXCLUDE"]:
        _unset(monkeypatch, name)


def _machine():
    m = wtune.machine
    return m.cores_per_node(), m.max_nodes(), m.partition(), m.login_nodes()


@pytest.mark.parametrize("bad", ["0", "-4", "many", "7.5"])
def test_a_bad_count_is_refused(scc, monkeypatch, bad):
    monkeypatch.setenv("WTUNE_CORES_PER_NODE", bad)
    with pytest.raises(wtune.machine.MachineError, match="WTUNE_CORES_PER_NODE"):
        wtune.machine.cores_per_node()


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


def test_paths_of_the_machine(scc, monkeypatch):
    assert wtune.machine.geog() == "/nfs/data/wrf/WPS_GEOG/"
    wtune.machine.export(dict(geog="~/geog", wrf_tree="$HOME/wrf", env_source="$HOME/env.sh"))
    home = os.path.expanduser("~")
    assert (wtune.machine.geog(), wtune.machine.wrf_tree()) == (home + "/geog", home + "/wrf")
    assert wtune.machine.env_source() == home + "/env.sh"    # a job's site file sources the
    # variable as is, so a literal $HOME was a missing file (f1arm.sh:23, f1-arm 2026-10-19)


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


def test_a_job_pending_on_a_size_limit_moves_to_the_next_partition(scc):
    # f1-arm: arm144 takes 2 x 72 tasks --exclusive (288 CPUs) and leaves it pending for ever
    wtune.machine.export(dict(partition="arm144,arm576,arm1440"))
    ids = iter(["11", "12"])
    submitted, cancelled = [], []

    def sbatch_runner(cmd):
        submitted.append(cmd.split()[2])
        return "Submitted batch job %s\n" % next(ids)

    reasons = {"11": "QOSMaxCpuPerJobLimit", "12": "Priority"}
    assert scc_sweep.submit_job("sbatch /x.sbatch", sbatch_runner,
                                pending_reason=reasons.get, cancel=cancelled.append) == "12"
    assert submitted == ["arm144", "arm576"] and cancelled == ["11"]

    with pytest.raises(RuntimeError, match="arm1440: job 3 pending on PartitionNodeLimit"):
        scc_sweep.submit_job("sbatch /x.sbatch", lambda c: "Submitted batch job 3\n",
                             pending_reason=lambda j: "PartitionNodeLimit", cancel=cancelled.append)
    assert cancelled == ["11", "3", "3", "3"]


def test_every_partition_refusing_is_reported(scc):
    wtune.machine.export(dict(partition="small,mid"))

    def sbatch_runner(cmd):
        raise RuntimeError("refused by " + cmd.split()[2])

    with pytest.raises(RuntimeError, match="small: refused by small; mid: refused by mid"):
        scc_sweep.submit_job("sbatch /seg.sbatch", sbatch_runner)
    with pytest.raises(wtune.machine.MachineError, match="list of names"):
        wtune.machine.check({"partition": []})


def test_the_per_user_submit_limit_is_waited_out(scc):
    wtune.machine.export(dict(partition="small,large"))
    calls, sleeps = [], []

    def sbatch_runner(cmd):
        calls.append(cmd.split()[2])
        if cmd.split()[2] == "small":
            raise RuntimeError("sbatch failed (rc=1): QOSMaxCpuPerJobLimit")
        if len(calls) < 6:
            raise RuntimeError("sbatch failed (rc=1): QOSMaxSubmitJobPerUserLimit")
        return "Submitted batch job 9\n"

    assert scc_sweep.submit_job("sbatch /x.sbatch", sbatch_runner, sleeps.append) == "9"
    assert calls == ["small", "large"] * 3 and sleeps == [60, 60]

    def never(cmd):
        raise RuntimeError("sbatch failed (rc=1): QOSMaxSubmitJobPerUserLimit")

    sleeps.clear()
    with pytest.raises(RuntimeError, match="MaxSubmitJob"):
        scc_sweep.submit_job("sbatch /x.sbatch", never, sleeps.append)
    assert len(sleeps) == scc_sweep.SUBMIT_LIMIT_MAX_WAIT_S // 60


def test_probe_input_copies_can_be_turned_off(scc, monkeypatch):
    assert wtune.machine.copy_probe_inputs() is True
    wtune.machine.export(wtune.machine.check({"copy_probe_inputs": False}))
    assert wtune.machine.copy_probe_inputs() is False
    with pytest.raises(wtune.machine.MachineError, match="true or false"):
        wtune.machine.check({"copy_probe_inputs": "no"})


def test_a_node_count_no_partition_takes_is_skipped_not_fatal(monkeypatch):
    import dataclasses
    from wtune.tune import node_search as ns

    @dataclasses.dataclass
    class C:
        nodes: int = 1
        steps: int = 10
        budget_s: float = 1800.0

        def elapsed(self):
            return 0.0

    def submit(cn, configs, **kw):
        if cn.nodes == 2:
            raise scc_sweep.NoPartitionError("no partition took the job -- arm576: refused")
        return "5%d" % cn.nodes, "/c.json", {}

    monkeypatch.setattr(ns, "submit_configs_job", submit)
    monkeypatch.setattr(ns, "_sweep_steps", lambda ctx, s: 10)
    monkeypatch.setattr(ns, "_job_time_est", lambda *a: {"total_run_s": 60})
    monkeypatch.setattr(ns, "_cold_budget_s_per_step", lambda *a: (1.0, "test"))
    monkeypatch.setattr(ns._jobs, "wait_for_jobs", lambda ctx, ids, label: {})
    monkeypatch.setattr(ns, "_anchor_results", lambda ctx, items, jobs, states, skipped: dict(jobs))
    anchor = dict(ranks=72, omp=1, nx=8, ny=9)
    items = [dict(nodes=n, key=n, tag={"nodes": n}, suffix="nodes-n%d" % n, anchor=anchor)
             for n in (1, 2, 4)]
    skipped, jobs = [], {}
    assert ns._anchor_wave(C(), items, lambda n: (1.0, "test"), skipped, 1, jobs) == {1: "51", 4: "54"}
    assert skipped == [{"nodes": 2, "reason": "no partition took the job -- arm576: refused"}]


def test_hardware_cores_per_node_makes_jobs_request_the_billed_cpus(scc, tmp_path):
    # f1-arm with 72 of 144 cores used: N x 72 tasks --exclusive is billed N x 144 CPUs
    assert wtune.machine.cpus_per_task_arg(72) == ""          # scc: hardware = cores_per_node
    wtune.machine.export(wtune.machine.check({"cores_per_node": 72,
                                              "hardware_cores_per_node": 144}))
    assert wtune.machine.hardware_cores_per_node() == 144
    assert [wtune.machine.cpus_per_task_arg(t) for t in (72, 70, 48, 36, 100, 144)] == [
        "--cpus-per-task=2", "--cpus-per-task=2", "--cpus-per-task=3", "--cpus-per-task=4", "", ""]
    cmd = scc_sweep.build_sbatch_command("c", 7, 72, "00:10:00", 600, 1, 20, "/c.json", "/out")
    assert " -N 7 --ntasks-per-node=72 --cpus-per-task=2 --exclusive " in cmd
    from wtune.run import sbatch as run_sbatch
    head = run_sbatch._header_lines(str(tmp_path), 2, "01:00:00", "seg", "/env.sh", str(tmp_path))
    assert "-N 2 --exclusive --ntasks-per-node=72 --cpus-per-task=2 --time=01:00:00" in head[2]
    with pytest.raises(wtune.machine.MachineError, match="number of cores"):
        wtune.machine.check({"hardware_cores_per_node": -1})


def test_exclusive_off_jobs_hold_only_their_tasks_cores_on_one_socket(scc, tmp_path):
    # f1-arm 2026-11-09: no idle node, so a whole-node tune waited 12 h; shared, each job takes
    # one Grace (72 of 144 cores) and is billed for those
    wtune.machine.export(wtune.machine.check({"cores_per_node": 72, "hardware_cores_per_node": 144,
                                              "numa_per_node": 1, "exclusive": False}))
    assert not wtune.machine.exclusive() and wtune.machine.cpus_per_task_arg(72) == ""
    cmd = scc_sweep.build_sbatch_command("c", 2, 72, "00:10:00", 600, 1, 20, "/c.json", "/out")
    from wtune import proxy
    from wtune.run import sbatch as run_sbatch
    pool = proxy.render_sbatch(str(tmp_path / "q"), "python", "/camp", "00:16:00", 4, mode="_pool")
    head = run_sbatch._header_lines(str(tmp_path), 2, "01:00:00", "seg", "/env.sh", str(tmp_path))[2]
    for line in (cmd, pool, head):
        assert "--exclusive" not in line and "--cpus-per-task" not in line
        assert "--ntasks-per-socket=72 -m block:block" in line
    with pytest.raises(wtune.machine.MachineError, match="true or false"):
        wtune.machine.check({"exclusive": "no"})

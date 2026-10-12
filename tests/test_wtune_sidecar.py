#!/usr/bin/env python3
"""`wtune run --sidecar`: restart validation records and in-allocation history conversion.
Every srun is a fake handle; the conversion it stands for is wtune.sidecar's own CLI run
in-process on test_post_bp_cache's fake BP directories (no adios2)."""
from conftest import swap_wtune_module
import datetime as dt
import json
import os
import sys

import pytest

pytest.importorskip("netCDF4")

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from wtune import sidecar as ws  # noqa: E402
from wtune.post import extract  # noqa: E402
from test_post_bp_cache import _bp, _fresh_process, fake_bp  # noqa: E402,F401
from test_wtune_run_resume import FMT, T0, _real_bp, _rsl  # noqa: E402

CFG = {"ranks": 80, "nodes": 5, "omp": 3}             # 48 of 72 cores per node
REAL_BOUND_WITHIN = ws.bound_within


def _t(hours):
    return (T0 + dt.timedelta(hours=hours)).strftime(FMT)


def _frame(seg, dom, hours):
    return _bp(os.path.join(str(seg), "wrfout_d%02d_%s" % (dom, _t(hours))), [_t(hours)])


class Proc:
    """A started srun: converts (runs the argv's own `convert` CLI) when the test lets it end."""

    def __init__(self, argv):
        self.argv, self.rc = argv, None

    def end(self, killed=False):
        self.rc = 143 if killed else ws.main(self.argv[self.argv.index("convert"):])

    def poll(self):
        return self.rc

    def wait(self):
        if self.rc is None:
            self.end()
        return self.rc


class Runner:
    def __init__(self):
        self.procs = []

    def __call__(self, argv):
        assert all(p.rc is not None for p in self.procs), "two conversion sruns at once"
        self.procs.append(Proc(argv))
        return self.procs[-1]

    def frames(self, i):
        argv = self.procs[i].argv
        return argv[argv.index("--nc-cache") + 2:]


@pytest.fixture(autouse=True)
def bound(monkeypatch):
    """The test process stands for a task srun bound to the idle cores."""
    monkeypatch.setattr(ws, "bound_within", lambda mask: True)


def _sidecar(tmp_path, runner, project=None, max_dom=1):
    run = tmp_path / "out"
    report, lines = {}, []
    side = ws.Sidecar(str(run), ws.post_cache_root(str(run), project), max_dom, os.path.isdir,
                      report, save=lambda: None, log=lines.append, cores_per_node=72,
                      exclude_arg="--exclude=head", runner=runner)
    return side, run, report, lines


# ---------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------
def test_idle_cores_and_mask():
    assert ws.idle_cores(CFG, 72) == list(range(48, 72))
    assert ws.idle_cores({"ranks": 72, "nodes": 1, "omp": 1}, 72) == []
    assert ws.idle_cores({"ranks": 70, "nodes": 4, "omp": 2}, 72) == list(range(36, 72))
    assert ws.cpu_mask([0, 1, 4]) == "0x13"
    assert ws.in_job_srun("77", [70, 71], 72, 5) == [
        "srun", "--jobid=77", "--overlap", "-N1", "-n1", "--relative=4", "-c", "72",
        "--cpu-bind=verbose,mask_cpu:0xc00000000000000000"]     # the allocation's last node
    assert ws.own_srun(72, "--exclude=head,work5") == [
        "srun", "-p", "compute", "-N1", "-n1", "-c", "72", "--time=01:00:00",
        "--exclude=head,work5"]


def test_convert_refuses_to_run_outside_the_idle_cores(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ws, "bound_within", REAL_BOUND_WITHIN)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 48}, raising=False)
    assert ws.bound_within("0x7") is False and ws.bound_within(ws.cpu_mask([0, 1, 48])) is True
    bound = []
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cores: bound.append(set(cores)),
                        raising=False)
    assert ws.bind_to(ws.cpu_mask(range(48, 72))) is True and bound == [{48}]   # self-bind
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    rc = ws.main(["convert", "--cpu-mask", ws.cpu_mask(range(48, 72)), "--run-dir",
                  str(tmp_path), "--nc-cache", str(tmp_path / "c")])
    assert rc == 3 and "nothing converted" in capsys.readouterr().err


def test_finished_frames_need_a_later_frame_of_the_domain(tmp_path, fake_bp):  # noqa: F811
    seg = tmp_path / "seg_0"
    a, b = _frame(seg, 1, 0), _frame(seg, 1, 1)
    c = _frame(seg, 2, 0)
    os.makedirs(seg / ("wrfout_d01_%s" % _t(2)))          # pre-created, not written yet
    assert ws.finished_frames(str(seg), False) == [a]     # 01:00 (d01), 00:00 (d02): maybe open
    assert ws.finished_frames(str(seg), True) == [a, b, c]


def test_post_cache_root_is_posts_own(tmp_path, monkeypatch):
    monkeypatch.delenv(extract.NC_CACHE_ENV, raising=False)
    run, proj = str(tmp_path / "out"), str(tmp_path / "proj")
    assert ws.post_cache_root(run) == os.path.join(run, "_nc_cache")
    assert ws.post_cache_root(run, proj) == os.path.join(proj, "_nc_cache")
    monkeypatch.setenv(extract.NC_CACHE_ENV, str(tmp_path / "shared"))
    assert ws.post_cache_root(run, proj) == str(tmp_path / "shared")


# ---------------------------------------------------------------------------
# conversion batches
# ---------------------------------------------------------------------------
def test_batches_only_finished_frames_one_srun_at_a_time(tmp_path, fake_bp):  # noqa: F811
    r = Runner()
    side, run, report, lines = _sidecar(tmp_path, r)
    seg = run / "seg_0"
    f0, f1 = _frame(seg, 1, 0), _frame(seg, 1, 1)
    side.poll(str(seg), "77", CFG, "PENDING")
    assert r.procs == []                                  # no allocation yet
    side.poll(str(seg), "77", CFG, "RUNNING")
    assert r.procs[0].argv[:9] == ws.in_job_srun("77", list(range(48, 72)), 72, 5)
    assert r.procs[0].argv[12:14] == ["--cpu-mask", ws.cpu_mask(range(48, 72))]
    assert r.frames(0) == [f0]                            # 01:00 is still WRF's open frame
    f2 = _frame(seg, 1, 2)
    side.poll(str(seg), "77", CFG, "RUNNING")
    assert len(r.procs) == 1                              # the first srun is still alive
    r.procs[0].end()
    side.poll(str(seg), "77", CFG, "RUNNING")
    assert r.frames(1) == [f1] and report["sidecar"]["converted"] == 1
    r.procs[1].end()
    side.poll(str(seg), "77", CFG, "RUNNING")
    assert len(r.procs) == 2 and report["sidecar"]["converted"] == 2   # nothing new finished
    assert fake_bp.conversions() == [f0, f1] and f2 not in fake_bp.conversions()
    assert sum("converting 1 frame(s) in job 77 on 24 idle core(s)" in ln for ln in lines) == 2


def test_batch_size_is_capped(tmp_path, fake_bp, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ws, "MAX_BATCH", 2)
    r = Runner()
    side, run, _report, _ = _sidecar(tmp_path, r)
    frames = [_frame(run / "seg_0", 1, h) for h in range(4)]
    side.poll(str(run / "seg_0"), "77", CFG, "RUNNING")
    assert r.frames(0) == frames[:2]


def test_no_idle_core_converts_only_after_the_run(tmp_path, fake_bp):  # noqa: F811
    r = Runner()
    side, run, report, lines = _sidecar(tmp_path, r)
    seg = run / "seg_0"
    frames = [_frame(seg, 1, h) for h in range(3)]
    side.poll(str(seg), "77", {"ranks": 72, "nodes": 1, "omp": 1}, "RUNNING")
    side.job_ended(str(seg))
    assert r.procs == []
    side.finish()
    assert r.procs[0].argv[:9] == ws.own_srun(72, "--exclude=head") and r.frames(0) == []
    assert sorted(fake_bp.conversions()) == frames
    assert report["sidecar"]["converted"] == 3 and report["sidecar"]["not_converted"] == []


def test_leftover_and_cut_off_frames_are_converted_at_the_end(tmp_path, fake_bp):  # noqa: F811
    r = Runner()
    side, run, report, lines = _sidecar(tmp_path, r)
    seg = run / "seg_0"
    f0, f1, f2 = [_frame(seg, 1, h) for h in range(3)]
    side.poll(str(seg), "77", CFG, "RUNNING")
    assert r.frames(0) == [f0, f1]
    r.procs[0].end(killed=True)                           # the allocation ended under it
    side.job_ended(str(seg))
    assert report["sidecar"]["converted"] == 0
    side.finish()
    assert len(r.procs) == 2 and r.procs[1].argv[0:3] == ["srun", "-p", "compute"]
    assert sorted(fake_bp.conversions()) == [f0, f1, f2]
    assert report["sidecar"]["converted"] == 3
    assert lines[-1] == "  sidecar: 3 frame(s) in %s" % report["sidecar"]["cache"]


def test_empty_batch_ends_in_job_conversion(tmp_path, fake_bp, monkeypatch):  # noqa: F811
    monkeypatch.setattr(ws, "bound_within", lambda mask: False)     # the binding did not take
    monkeypatch.setattr(ws, "bind_to", lambda mask: False)
    r = Runner()
    side, run, report, lines = _sidecar(tmp_path, r)
    seg = run / "seg_0"
    frames = [_frame(seg, 1, h) for h in range(3)]
    side.poll(str(seg), "77", CFG, "RUNNING")
    r.procs[0].end()
    assert r.procs[0].rc == 3 and fake_bp.conversions() == []
    _frame(seg, 1, 3)
    side.poll(str(seg), "77", CFG, "RUNNING")
    side.poll(str(seg), "77", CFG, "RUNNING")
    assert len(r.procs) == 1 and sum("converted nothing" in ln for ln in lines) == 1
    side.job_ended(str(seg))
    side.finish()                                         # the node of its own has no mask
    assert len(r.procs) == 2 and len(fake_bp.conversions()) == 4
    assert frames[0] in fake_bp.conversions() and report["sidecar"]["converted"] == 4


def test_backlog_of_an_ended_job_goes_into_the_next_job(tmp_path, fake_bp):  # noqa: F811
    r = Runner()
    side, run, _report, _ = _sidecar(tmp_path, r)
    f0, f1 = _frame(run / "seg_0", 1, 0), _frame(run / "seg_0", 1, 1)
    side.job_ended(str(run / "seg_0"))
    g0, _g1 = _frame(run / "seg_1", 1, 2), _frame(run / "seg_1", 1, 3)
    side.poll(str(run / "seg_1"), "78", CFG, "RUNNING")
    assert r.procs[0].argv[1] == "--jobid=78" and r.frames(0) == [f0, f1, g0]


def test_cache_is_exactly_what_post_looks_up(tmp_path, fake_bp, monkeypatch):  # noqa: F811
    proj = tmp_path / "proj"
    r = Runner()
    side, run, report, _ = _sidecar(tmp_path, r, project=str(proj))
    frames = [_frame(run / "seg_0", 1, h) for h in range(2)]
    frames += [_frame(run / "seg_0_r1", 1, 2)]
    side.job_ended(str(run / "seg_0"))
    side.job_ended(str(run / "seg_0_r1"))
    side.finish()
    done = len(fake_bp.conversions())
    assert done == 3 and report["sidecar"]["cache"] == str(proj / "_nc_cache")
    _fresh_process(monkeypatch)
    with extract.nc_cache(base=str(proj)):                # what `wtune post recipe` sets
        prep = extract.prepare_history(str(run))
        want = [extract.cached_nc_path(str(run), f) for f in frames]
        assert [p for _t, p in extract.list_frames(str(run))] == want
    assert prep["n_bp"] == 3 and prep["failed"] == []
    assert len(fake_bp.conversions()) == done             # post converted nothing
    assert all(os.path.isfile(p) for p in want)
    assert want[2].endswith(os.path.join("seg_0_r1", "wrfout_d01_%s.nc" % _t(2)))


def test_a_sidecar_error_never_stops_the_run(tmp_path, fake_bp):  # noqa: F811
    def boom(argv):
        raise OSError("srun: command not found")
    side, run, _report, lines = _sidecar(tmp_path, boom)
    _frame(run / "seg_0", 1, 0)
    _frame(run / "seg_0", 1, 1)
    side.poll(str(run / "seg_0"), "77", CFG, "RUNNING")
    side.finish()
    assert lines[-1] == "  sidecar: switched off (OSError: srun: command not found)"


# ---------------------------------------------------------------------------
# restart validation
# ---------------------------------------------------------------------------
def test_restart_validation_records_each_new_valid_set(tmp_path, fake_bp):  # noqa: F811
    r = Runner()
    side, run, report, lines = _sidecar(tmp_path, r, max_dom=2)
    seg = run / "seg_0"
    seg.mkdir(parents=True)
    rst = lambda d, h: str(seg / ("wrfrst_d%02d_%s" % (d, _t(h))))   # noqa: E731
    at = lambda h: T0 + dt.timedelta(hours=h)                        # noqa: E731
    for d in (1, 2):
        _real_bp(rst(d, 2))
    _real_bp(rst(1, 4))                                   # 04:00: d02 not written yet
    (seg / "rsl.error.0000").write_text(_rsl(T0, at(4), [at(2)], 2))
    side.poll(str(seg), "77", CFG, "PENDING")
    rec = report["sidecar"]
    assert rec["valid_restarts"] == [{"dir": str(seg), "time": _t(2)}]
    _real_bp(rst(2, 4))
    side.poll(str(seg), "77", CFG, "RUNNING")             # on disk, the rsl has no line yet
    assert len(rec["valid_restarts"]) == 1
    (seg / "rsl.error.0000").write_text(_rsl(T0, at(5), [at(2), at(4)], 2))
    side.job_ended(str(seg))
    assert [v["time"] for v in rec["valid_restarts"]] == [_t(2), _t(4)]
    assert rec["last_valid_restart"] == {"dir": str(seg), "time": _t(4)}
    assert sum("restart" in ln and "valid" in ln for ln in lines) == 2


# ---------------------------------------------------------------------------
# through `wtune run`
# ---------------------------------------------------------------------------
def test_wtune_run_sidecar_hook(tmp_path, monkeypatch):
    from wtune import predict as wp
    from wtune import run as wr
    from wtune import watch as wtune_watch
    import test_wtune_run as twr
    swap_wtune_module(monkeypatch, "bp", twr.FakeBP().module())
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    monkeypatch.delenv(extract.NC_CACHE_ENV, raising=False)

    class Job(twr.Fakes):
        """The job has written three d01 frames (into their pre-created dirs) and a valid
        restart at its end; sacct says RUNNING twice before COMPLETED."""
        polls = 0

        def sbatch(self, cmd):
            seg = os.path.dirname(cmd.split()[-1].strip("'"))
            info = wtune_watch.run_info(wp.parse_namelist(os.path.join(seg, "namelist.input")))
            for h in range(3):
                open(os.path.join(seg, "wrfout_d01_%s" % _t(h), "md.idx"), "w").close()
            with open(os.path.join(seg, "rsl.error.0000"), "w") as f:
                f.write(_rsl(info["start"], info["end"], [info["end"]], 3))
            return super().sbatch(cmd)

        def sacct(self, cmd):
            self.polls += 1
            jid = cmd[cmd.index("-j") + 1]
            return "%s|%s|00:10:00\n" % (jid, "RUNNING" if self.polls < 3 else "COMPLETED")

    class Srun(Runner):
        """Stands for the conversion: writes the cache manifest of every frame it was given."""
        def __call__(self, argv):
            proc = super().__call__(argv)
            run = argv[argv.index("--run-dir") + 1]
            with extract.nc_cache(directory=argv[argv.index("--nc-cache") + 1]):
                for bp in self.frames(-1) or [p for _d, _t, p, _b in extract._raw_entries(run)]:
                    nc = extract.cached_nc_path(run, bp)
                    os.makedirs(os.path.dirname(nc), exist_ok=True)
                    open(nc + ".json", "w").close()
            proc.rc = 0
            return proc

    case, out, r, lines = twr._make_case(tmp_path), tmp_path / "out", Srun(), []
    f = Job()
    rc = wr.main([str(case), "--nodes", "1", "--nx", "6", "--ny", "8", "--out", str(out),
                  "--poll-interval", "0", "--sidecar"], sbatch_runner=f.sbatch,
                 sacct_runner=f.sacct, snapshot_fn=f.snapshot, sleep=lambda s: None,
                 log=lines.append, sidecar_runner=r)
    rep = json.loads((out / "run_report.json").read_text())
    assert rc == 0 and rep["status"] == "completed"
    seg = str(out / "seg_0")
    assert [p.argv[0:2] for p in r.procs] == [["srun", "--jobid=1001"], ["srun", "-p"]]
    assert r.procs[0].argv[5] == "--relative=0"           # one node: the last is the first
    assert r.procs[0].argv[8] == "--cpu-bind=verbose,mask_cpu:" + ws.cpu_mask(range(48, 72))
    assert r.frames(0) == [os.path.join(seg, "wrfout_d01_%s" % _t(h)) for h in (0, 1)]
    side = rep["sidecar"]
    assert side["cache"] == str(out / "_nc_cache") and side["converted"] == 3
    assert side["not_converted"] == []
    assert side["last_valid_restart"] == {"dir": seg, "time": "2020-06-01_12:00:00"}

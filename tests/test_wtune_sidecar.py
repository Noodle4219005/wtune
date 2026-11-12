#!/usr/bin/env python3
"""`wtune run --sidecar`: restart validation records and in-allocation history conversion.
Every srun is a fake handle; the conversion it stands for is wtune.sidecar's own CLI run
in-process on test_post_bp_cache's fake BP directories (no adios2)."""
from conftest import swap_wtune_module
import datetime as dt
import json
import os

import pytest

pytest.importorskip("netCDF4")

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from wtune import sidecar as ws  # noqa: E402
from wtune.post import extract  # noqa: E402
from test_post_bp_cache import _bp, _fresh_process, fake_bp  # noqa: E402,F401
from test_wtune_run_resume import FMT, T0, _rsl  # noqa: E402

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


# ---------------------------------------------------------------------------
# --compress-history: the checked netCDF takes the BP's place
# ---------------------------------------------------------------------------
def _compressing(tmp_path, runner, monkeypatch, check_ok=True):
    import wtune.bp

    def check(argv, what):
        if not check_ok:
            raise RuntimeError("values differ")
    monkeypatch.setattr(wtune.bp, "_run_adios2", check, raising=False)
    run = tmp_path / "out"
    report, lines = {}, []
    side = ws.Sidecar(str(run), ws.post_cache_root(str(run)), 1, os.path.isdir, report,
                      save=lambda: None, log=lines.append, cores_per_node=72,
                      exclude_arg="--exclude=head", runner=runner, replace=True)
    return side, run, report


def test_compress_history_puts_each_checked_netcdf_in_place_of_its_bp(tmp_path, fake_bp,  # noqa: F811
                                                                      monkeypatch):
    side, run, report = _compressing(tmp_path, Runner(), monkeypatch)
    seg = run / "seg_0"
    frames = [_frame(seg, 1, h) for h in range(3)]
    side.job_ended(str(seg))
    side.finish()
    assert all(os.path.isfile(f) for f in frames)                 # netCDF under the BP's name
    assert not [n for n in os.listdir(seg) if n.endswith(".bp-replaced")]
    assert not os.path.exists(side.cache_root)                     # no empty cache dirs left
    assert report["sidecar"]["converted"] == 3 and report["sidecar"]["not_converted"] == []
    _fresh_process(monkeypatch)
    prep = extract.prepare_history(str(run))
    assert prep["n_bp"] == 0                                      # post converts nothing
    assert [p for _t, p in extract.list_frames(str(run))] == frames


def test_a_frame_that_fails_the_check_stays_bp(tmp_path, fake_bp, monkeypatch):  # noqa: F811
    side, run, _report = _compressing(tmp_path, Runner(), monkeypatch, check_ok=False)
    frame = _frame(run / "seg_0", 1, 0)
    side.job_ended(str(run / "seg_0"))
    side.finish()
    assert os.path.isdir(frame)
    with extract.nc_cache(directory=side.cache_root):
        assert os.path.isfile(extract.cached_nc_path(str(run), frame))


def test_a_full_node_shares_its_last_core_when_compressing(tmp_path, fake_bp, monkeypatch):  # noqa: F811
    r = Runner()
    side, run, _report = _compressing(tmp_path, r, monkeypatch)
    seg = run / "seg_0"
    for h in range(3):
        _frame(seg, 1, h)
    side.poll(str(seg), "77", {"ranks": 72, "nodes": 1, "omp": 1}, "RUNNING")
    argv = r.procs[0].argv
    assert argv[argv.index("nice"):argv.index("nice") + 3] == ["nice", "-n", "19"]
    assert "--cpu-bind=verbose,mask_cpu:%s" % ws.cpu_mask([71]) in argv and "--replace" in argv


def test_finish_drops_the_partial_files_of_a_killed_conversion(tmp_path, fake_bp):  # noqa: F811
    side, run, _report, _lines = _sidecar(tmp_path, Runner())
    part = os.path.join(side.cache_root, "out-x", "seg_0")
    os.makedirs(part)
    junk = os.path.join(part, ".wrfout_d01_x.wtune-tmp-n1-7.nc.tmp-7-1")
    keep = os.path.join(part, "wrfout_d01_x.nc")
    for p in (junk, keep):
        open(p, "w").close()
    side.finish()
    assert not os.path.exists(junk) and os.path.exists(keep)

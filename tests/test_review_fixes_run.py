"""Regression tests for the 8bdcd3c source-review findings in wtune_run / wtune_watch /
wtune_sidecar. No job is ever submitted: sbatch/sacct/snapshot/srun are fakes in tmp dirs."""
import datetime as dt
import json
import os
import time

import pytest

import wtune_project as proj
import wtune_run as wr
import wtune_watch as ww
from wtune_run import cli as run_cli
from wtune_run._deps import coldstart

import test_refactor_golden_wtune_run as G
import test_watch_run_root as W
import test_wtune_run as T
import test_wtune_run_resume as TR
from test_wtune_run import fake_bp  # noqa: F401  (autouse: a fake src/wtune_bp)

LAYOUT = ["--nodes", "1", "--nx", "8", "--ny", "9"]


def _main(tmp_path, args, fakes=None, text=T.NAMELIST_3DOM, **kw):
    case = tmp_path / "case" if (tmp_path / "case").exists() else T._make_case(tmp_path, text)
    out, lines = tmp_path / "out", []
    if fakes is not None:
        kw.update(sbatch_runner=fakes.sbatch, sacct_runner=fakes.sacct,
                  snapshot_fn=fakes.snapshot)
    rc = wr.main([str(case), "--out", str(out), "--poll-interval", "0"] + args,
                 sleep=lambda s: None, log=lines.append, **kw)
    rep = out / "run_report.json"
    return rc, lines, json.loads(rep.read_text()) if rep.exists() else None


# 1 -- wait_for_job: only consecutive empty sacct replies give up
def test_wait_for_job_counts_consecutive_unknowns_only():
    replies = ["", "7|RUNNING|00:01:00\n"] * 15 + ["7|COMPLETED|00:20:00\n"]
    state, _elapsed = wr.wait_for_job("7", lambda cmd: replies.pop(0), sleep=lambda s: None,
                                      log=lambda s: None)
    assert state == "COMPLETED"
    state, elapsed = wr.wait_for_job("7", lambda cmd: "", sleep=lambda s: None,
                                     log=lambda s: None)
    assert (state, elapsed) == ("UNKNOWN", None)


# 2 -- a job the time-limit trap interrupted is not a CFL model failure
class Interrupted(TR.Dying):
    """Dying + the sbatch trap's INTERRUPTED record and a routine cfl line in an rsl."""
    status = "SEGMENT 0 rc=1 ok=0 INTERRUPTED sig=USR1\n"

    def __init__(self, outcomes):
        super().__init__(outcomes, {"rsl.error.0003": " 2 points exceeded cfl=2 in domain "
                                                      "d01 at time 2020-06-01_05:00:00\n"})

    def sbatch(self, cmd):
        died = self.outcomes and isinstance(self.outcomes[0], tuple)
        out = super().sbatch(cmd)
        if died and self.status:
            with open(os.path.join(self.submitted[-1][0], "segment_status.txt"), "w") as f:
                f.write(self.status)
        return out


def test_time_limit_interrupt_with_cfl_line_is_resumed_not_rescued(tmp_path):
    f = Interrupted([("died", "FAILED", 9)])        # the trap exits 3: sacct says FAILED
    rc, _lines, rep = _main(tmp_path, LAYOUT + ["--segment-minutes", "720", "--cfl-rescue"], f,
                            text=T._with_restart(120))
    assert rc == 0 and rep["physics_gray"] is False
    assert rep["segments"][0]["attempts"][0]["failure_class"] == "infrastructure"
    assert [os.path.basename(s) for s, _ in f.submitted] == ["seg_0", "seg_0_r1"]
    assert "cfl_rescue" not in [d["decision"] for d in rep["decisions"]]


def test_cancelled_job_with_cfl_line_is_not_rescued(tmp_path):
    f = Interrupted([("died", "CANCELLED", 9)])
    f.status = None
    rc, _lines, rep = _main(tmp_path, LAYOUT + ["--segment-minutes", "720", "--cfl-rescue"], f,
                            text=T._with_restart(120))
    assert rc == 1 and rep["physics_gray"] is False
    assert [os.path.basename(s) for s, _ in f.submitted] == ["seg_0"]


def test_cfl_crash_is_still_rescued(tmp_path):
    f = Interrupted([("died", "FAILED", 9)])        # wrf.exe died by itself: no INTERRUPTED
    f.status = None
    rc, _lines, rep = _main(tmp_path, LAYOUT + ["--segment-minutes", "720", "--cfl-rescue"], f,
                            text=T._with_restart(120))
    assert rc == 0 and rep["physics_gray"] is True
    assert [os.path.basename(s) for s, _ in f.submitted] == ["seg_0", "seg_0_rescue"]


# 3 -- a kept case restart interval spelled restart_interval_h only
RST_H = T.NAMELIST_3DOM.replace(T._RST_LINE, " restart_interval_h                  = 6,")


def test_case_with_only_restart_interval_h_runs(tmp_path):
    assert "restart_interval_h" in RST_H
    f = T.Fakes()
    rc, lines, rep = _main(tmp_path, LAYOUT + ["--segment-minutes", "360"], f, text=RST_H)
    assert rc == 0 and rep["status"] == "completed"
    assert any("case's restart_interval kept" in ln for ln in lines)
    nl = (tmp_path / "out" / "seg_1" / "namelist.input").read_text()
    assert "restart_interval_h" in nl and coldstart.count_key(nl, "restart_interval") == 0


def test_dry_run_namelist_error_is_a_clean_error(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise coldstart.ColdstartError("segment 0 namelist: keys not present exactly once")
    monkeypatch.setattr(run_cli, "render_segment_namelist", boom)
    rc, _lines, _rep = _main(tmp_path, LAYOUT + ["--dry-run"])
    assert rc == 2
    assert "error: segment 0 namelist" in capsys.readouterr().err


# 4 -- Slurm must not requeue a segment job behind the controller's back
def test_segment_sbatch_is_not_requeued(tmp_path):
    text = wr.render_segment_sbatch(str(tmp_path), 0, wr.make_config(8, 9, 1, 1, 1), "01:00:00",
                                    "j")
    assert "#SBATCH --no-requeue\n" in text


# 5 -- explicit layout flags beat the tune report's ranking
def test_cli_layout_overrides_win_over_ranking(tmp_path):
    rep = G._tune_report(tmp_path, ranking=G.RANKING)
    rc, lines, _ = _main(tmp_path, ["--tune-report", rep, "--nx", "4", "--ny", "18",
                                    "--dry-run"])
    assert rc == 0
    segs = [ln for ln in lines if ln.startswith("  seg ")]
    assert segs and all("4x18 t2 omp1" in ln for ln in segs)
    assert any("no layout matching the CLI overrides" in ln for ln in lines)
    # a ranked layout named by the flags keeps its prediction
    rc, lines, _ = _main(tmp_path, ["--tune-report", rep, "--nx", "6", "--ny", "12",
                                    "--dry-run"])
    segs = [ln for ln in lines if ln.startswith("  seg ")]
    assert rc == 0 and segs and all(" B " in ln and "6x12 t2 omp1" in ln for ln in segs)
    # no flags: the ranking decides, as before
    rc, lines, _ = _main(tmp_path, ["--tune-report", rep, "--dry-run"])
    assert rc == 0 and not any("CLI overrides" in ln for ln in lines)


# 6 -- wtune watch on a run the controller gave up on
def _root(tmp_path, status):
    root = tmp_path / "out"
    W._rundir(root / "seg_0", hours=6, n_steps=30)
    seg = {"start": W.T0.strftime(W.FMT),
           "end": (W.T0 + dt.timedelta(hours=6)).strftime(W.FMT)}
    (root / "run_report.json").write_text(json.dumps({"status": status, "segments": [seg]}))
    return str(root)


def test_watch_reports_a_failed_run_and_stops(tmp_path, capsys):
    assert ww.main([_root(tmp_path, "failed"), "--interval", "0"]) == 3
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("seg 0/1 FAILED")


def test_watch_healthy_run_is_unchanged(tmp_path, capsys):
    assert ww.main([_root(tmp_path, "running"), "--once"]) == 0
    assert capsys.readouterr().out.startswith("seg 0/1 RUNNING")


# 7 -- the sidecar is finalized for a failed run too
class _Srun:
    def __init__(self):
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        return self

    def poll(self):
        return 0

    def wait(self):
        return 0


class _FailingWithFrame(T.Fakes):
    def sbatch(self, cmd):
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        T._mk_bp(os.path.join(seg, "wrfout_d01_2020-06-01_00:00:00"))
        return super().sbatch(cmd)

    def sacct(self, cmd):
        return "%s|FAILED|00:10:00\n" % cmd[cmd.index("-j") + 1]


def test_sidecar_is_finished_when_the_run_fails(tmp_path):
    srun = _Srun()
    rc, lines, rep = _main(tmp_path, LAYOUT + ["--sidecar"], _FailingWithFrame(["fail"]),
                           sidecar_runner=srun)
    assert rc == 1 and rep["status"] == "failed"
    assert [c[0] for c in srun.calls] == ["srun"]          # the after-run conversion
    assert [os.path.basename(p) for p in rep["sidecar"]["not_converted"]] == [
        "wrfout_d01_2020-06-01_00:00:00"]
    assert any("NOT converted" in ln for ln in lines)


# 8 -- the t=0 frame of a cold start is not progress
def test_initial_wrfout_alone_is_init_not_rsl_lagging(tmp_path):
    now = time.time()
    d = W._rundir(tmp_path / "r", wrfout_hours=[0], mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "init" and "RSL_LAGGING" not in W._codes(snap)
    d = W._rundir(tmp_path / "r2", wrfout_hours=[0, 1], mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "running" and "RSL_LAGGING" in W._codes(snap)


# 9 -- usage errors are `error: ...` + rc 2, not tracebacks
def test_no_out_outside_a_project_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    case = T._make_case(tmp_path)
    assert wr.main([str(case)] + LAYOUT + ["--dry-run"], log=lambda s: None) == 2
    assert "error: --out is required" in capsys.readouterr().err


def test_bad_project_is_a_clean_error(tmp_path, monkeypatch, capsys):
    case = T._make_case(tmp_path)
    args = [str(case)] + LAYOUT + ["--dry-run"]
    assert wr.main(args + ["--project", str(tmp_path / "nope")], log=lambda s: None) == 2
    assert "error: --project" in capsys.readouterr().err
    monkeypatch.setenv("WTUNE_PROJECT", str(tmp_path / "nope"))
    assert wr.main(args, log=lambda s: None) == 2
    assert "error: WTUNE_PROJECT=" in capsys.readouterr().err


# 10 -- the auto-picked tune report must be for the project's case
def _project_report(project_dir, name, mtime, **doc):
    d = os.path.join(project_dir, "tune", name)
    os.makedirs(d)
    path = os.path.join(d, "tune_report.json")
    with open(path, "w") as fh:
        json.dump(dict(doc, winner={"nodes": 2, "nx": 4, "ny": 4}), fh)
    os.utime(path, (mtime, mtime))
    return path


def test_newest_report_of_another_case_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    project = proj.init_project(str(tmp_path / "p"), case_dir=str(T._make_case(tmp_path)))
    case_id = proj.project_case_id(proj.load_project(project))
    assert case_id
    now, lines = time.time(), []
    mine = _project_report(project, "mine", now - 200, case=case_id)
    _project_report(project, "nocase", now - 100)
    other = _project_report(project, "other", now, case="sc26p1")
    opts = run_cli.apply_project_defaults(run_cli.parse_options(["--project", project]),
                                          log=lines.append)
    assert opts.tune_report == mine
    assert any("skipped" in ln and other in ln for ln in lines)
    os.remove(mine)
    opts = run_cli.apply_project_defaults(run_cli.parse_options(["--project", project]),
                                          log=lines.append)
    assert opts.tune_report is None


def test_project_without_registered_case_keeps_the_newest_report(tmp_path, monkeypatch):
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    project = proj.init_project(str(tmp_path / "p"))
    os.makedirs(os.path.join(project, "case"), exist_ok=True)
    newest = _project_report(project, "t", time.time(), case="whatever")
    opts = run_cli.apply_project_defaults(run_cli.parse_options(["--project", project]),
                                          log=lambda s: None)
    assert opts.tune_report == newest


# 11 -- an unexpected exception never leaves the report at "running"
def test_interrupt_finalizes_the_report(tmp_path):
    def sbatch(cmd):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _main(tmp_path, LAYOUT, sbatch_runner=sbatch)
    rep = json.loads((tmp_path / "out" / "run_report.json").read_text())
    assert rep["status"] == "failed" and "KeyboardInterrupt" in rep["error"]

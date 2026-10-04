#!/usr/bin/env python3
"""`wtune run --resume`: failure classification, valid restarts, the resumed attempt, and how
post/watch list its directory. Fakes only: no job, no adios2."""
import datetime as dt
import json
import os

import pytest

import test_wtune_run  # noqa: F401  first: its wtune_run import restores SITE/configs
import wtune_predict as wp
import wtune_resume as wres
import wtune_run as wr
import wtune_watch
from test_wtune_run import (REAL_BP, Fakes, _make_case, _run, _with_restart,
                            fake_bp)  # noqa: F401  (autouse fixture)

FMT = "%Y-%m-%d_%H:%M:%S"
T0 = dt.datetime(2020, 6, 1, 0)
MAIN = "Timing for main: time %s on domain   1:    1.00000 elapsed seconds"
RST = "Timing for Writing restart for domain %8d:    2.00000 elapsed seconds"


def _at(hours):
    return T0 + dt.timedelta(hours=hours)


def _rsl(start, stop, rst_times, max_dom, step_s=60):
    """rsl of a job that ran d01 steps from start to stop and finished a restart at rst_times."""
    lines, t = [], start
    while t < stop:
        t += dt.timedelta(seconds=step_s)
        lines.append(MAIN % t.strftime(FMT))
        if t in rst_times:
            lines += [RST % d for d in range(1, max_dom + 1)]
    return "\n".join(lines) + "\n"


class Dying(Fakes):
    """An outcome ("died", STATE, HOURS) is a job that wrote the restarts its namelist asks for
    (restart_interval from the run start), stopped HOURS after the run start without SUCCESS,
    and that sacct reports as STATE."""

    def __init__(self, outcomes=None, rsl_extra=None):
        super().__init__(outcomes)
        self.states, self.steps, self.rsl_extra = {}, {}, rsl_extra or {}

    def sbatch(self, cmd):
        if not (self.outcomes and isinstance(self.outcomes[0], tuple)):
            return super().sbatch(cmd)
        _died, state, hours = self.outcomes.pop(0)
        seg = os.path.dirname(cmd.split()[-1].strip("'"))
        nl = wp.parse_namelist(os.path.join(seg, "namelist.input"))
        info = wtune_watch.run_info(nl)
        every = dt.timedelta(minutes=nl["time_control"]["restart_interval"][0])
        rst = [T0 + j * every for j in range(1, 100)
               if info["start"] < T0 + j * every <= _at(hours)]
        with open(os.path.join(seg, "rsl.error.0000"), "w") as f:
            f.write(_rsl(info["start"], _at(hours), rst, len(info["domains"])))
        for name, text in self.rsl_extra.items():
            with open(os.path.join(seg, name), "w") as f:
                f.write(text)
        for t in rst:
            for d in range(1, len(info["domains"]) + 1):
                p = os.path.join(seg, "wrfrst_d%02d_%s" % (d, t.strftime(FMT)))
                os.makedirs(p, exist_ok=True)
                open(os.path.join(p, "md.idx"), "w").close()
        self.submitted.append((seg, "died"))
        self.snap_by_dir[seg] = "died"
        self.steps[seg] = int((_at(hours) - info["start"]).total_seconds() // 60)
        self.states[str(1000 + len(self.submitted))] = state
        return "Submitted batch job %d\n" % (1000 + len(self.submitted))

    def sacct(self, cmd):
        jid = cmd[cmd.index("-j") + 1]
        return "%s|%s|00:10:00\n" % (jid, self.states.get(jid, "COMPLETED"))

    def snapshot(self, rundir):
        if self.snap_by_dir[rundir] != "died":
            return super().snapshot(rundir)
        n = self.steps[rundir]
        return {"status": "running", "warnings": [], "sum_d01_elapsed_s": 2.0 * n,
                "n_d01_steps": n}


def _run_one_segment(tmp_path, fakes, extra=()):
    """A 12 h single-segment run whose case writes a restart every 2 h."""
    if not (tmp_path / "case").exists():
        _make_case(tmp_path, _with_restart(120))
    return _run(tmp_path, fakes, ["--segment-minutes", "720"] + list(extra))


def _names(fakes):
    return [os.path.basename(s) for s, _ in fakes.submitted]


# ---------------------------------------------------------------------------
# the resumed attempt
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["NODE_FAIL", "TIMEOUT", "PREEMPTED", "BOOT_FAIL", "FAILED",
                                   "COMPLETED"])
def test_infrastructure_failure_resumes_once_and_finishes(tmp_path, state):
    f = Dying([("died", state, 9)])
    rc, out, rep, lines = _run_one_segment(tmp_path, f)
    assert rc == 0 and rep["status"] == "completed" and rep["resumes"] == 1
    assert _names(f) == ["seg_0", "seg_0_r1"]
    # valid restarts 02/04/06/08; wrfbdy records exist every 6 h, so 06:00 is the resume point
    first, second = rep["segments"][0]["attempts"]
    assert (first["failure_class"], first["resume_time"], first["job_id"]) == (
        "infrastructure", "2020-06-01_06:00:00", "1001")
    assert second["start"] == "2020-06-01_06:00:00" and second["ok"] is True
    assert "  resume 1/1: infrastructure at job 1001; restart 2020-06-01_06:00:00" in lines
    assert rep["segments"][0]["run_dir"] == str(out / "seg_0_r1")


def test_resumed_attempt_dir_layout_dt_restart_link_and_time(tmp_path, fake_bp):  # noqa: F811
    f = Dying([("died", "NODE_FAIL", 9)])
    rc, out, rep, lines = _run_one_segment(tmp_path, f)
    r1 = out / "seg_0_r1"
    nl = wp.parse_namelist(str(r1 / "namelist.input"))
    tc, dom = nl["time_control"], nl["domains"]
    assert tc["start_hour"] == [6, 6, 6] and tc["end_hour"] == [12, 12, 12]
    assert tc["restart"] == [True] and tc["run_seconds"] == [6 * 3600]
    assert (dom["time_step"], dom["nproc_x"], dom["nproc_y"]) == ([60], [8], [9])
    for d in (1, 2, 3):
        name = "wrfrst_d%02d_2020-06-01_06:00:00" % d
        assert os.readlink(r1 / name) == str(out / "seg_0" / name)
    xml = (r1 / "adios2.xml").read_text()
    assert all("wrfrst_d01_2020-06-01_%02d:00:00" % h in xml for h in (6, 8, 10, 12))
    # --time from the observed 2 s/step over the 360 steps left, + init + restart read
    wall = 2.0 * 360 + rep["overheads"]["init_s"] + rep["overheads"]["restart_read_s"]
    assert "--time=%s " % wr._hms(wr.time_limit_s(wall)) in (r1 / "segment.sbatch").read_text()
    # a fresh input copy and pre-created output dirs, as for any attempt
    assert str(r1 / "_bp_inputs" / "wrfbdy_d01") in [c[4] for c in fake_bp.cp_calls]
    assert sum("pre-created" in ln for ln in lines) == 2


def test_without_input_records_the_latest_valid_restart_is_used(tmp_path):
    f = Dying([("died", "NODE_FAIL", 9)])
    rc, out, rep, _ = _run_one_segment(tmp_path, f, ["--io", "netcdf"])
    assert rc == 0 and rep["segments"][0]["attempts"][0]["resume_time"] == "2020-06-01_08:00:00"


def test_later_segment_hands_off_from_the_resumed_attempt(tmp_path):
    _make_case(tmp_path, _with_restart(120))
    f = Dying([("died", "NODE_FAIL", 5)])
    rc, out, rep, _ = _run(tmp_path, f, ["--io", "netcdf"])        # two 6 h segments
    assert rc == 0 and _names(f) == ["seg_0", "seg_0_r1", "seg_1"]
    assert rep["segments"][0]["attempts"][0]["resume_time"] == "2020-06-01_04:00:00"
    name = "wrfrst_d01_2020-06-01_06:00:00"
    assert os.readlink(out / "seg_1" / name) == str(out / "seg_0_r1" / name)


def test_resume_budget_is_per_run(tmp_path):
    f = Dying([("died", "NODE_FAIL", 9), ("died", "NODE_FAIL", 11)])
    rc, out, rep, _ = _run_one_segment(tmp_path, f, ["--io", "netcdf"])
    assert rc == 1 and _names(f) == ["seg_0", "seg_0_r1"]
    assert "not resumed: --resume 1 already used" in rep["error"]
    f = Dying([("died", "NODE_FAIL", 9), ("died", "NODE_FAIL", 11)])
    rc, _out, _rep, _ = _run(tmp_path, f, ["--segment-minutes", "720", "--io", "netcdf",
                                           "--resume", "2", "--out", str(tmp_path / "out2")])
    rep = json.loads((tmp_path / "out2" / "run_report.json").read_text())
    assert rc == 0 and _names(f) == ["seg_0", "seg_0_r1", "seg_0_r2"] and rep["resumes"] == 2
    assert [a.get("resume_time") for a in rep["segments"][0]["attempts"]] == [
        "2020-06-01_08:00:00", "2020-06-01_10:00:00", None]


# ---------------------------------------------------------------------------
# failures that are not resumed
# ---------------------------------------------------------------------------
def _stopped(tmp_path, fakes, extra=()):
    rc, out, rep, _ = _run_one_segment(tmp_path, fakes, extra)
    assert rc == 1 and rep["status"] == "failed" and len(fakes.submitted) == 1
    assert rep["resumes"] == 0
    return rep, rep["segments"][0]["attempts"][0]


def test_model_failure_is_not_resumed(tmp_path):
    seg = "forrtl: severe (174): SIGSEGV, segmentation fault occurred\n"
    rep, att = _stopped(tmp_path, Dying([("died", "FAILED", 9)], {"rsl.error.0003": seg}))
    assert att["failure_class"] == "model" and "not resumed" not in rep["error"]


def test_cfl_failure_keeps_todays_behaviour(tmp_path):
    rep, att = _stopped(tmp_path, Dying(["cfl"]))
    assert att["failure_class"] == "model" and "CFL" in rep["error"]


def test_cancelled_is_not_resumed(tmp_path):
    rep, att = _stopped(tmp_path, Dying([("died", "CANCELLED", 9)]))
    assert att["failure_class"] == "cancelled"


def test_no_valid_restart_is_not_resumed(tmp_path):
    rep, att = _stopped(tmp_path, Dying([("died", "NODE_FAIL", 1)]))
    assert att["failure_class"] == "infrastructure"
    assert "not resumed: no valid restart" in rep["error"]
    assert att["no_resume_reason"].startswith("no valid restart")


def test_resume_0_disables(tmp_path):
    rep, att = _stopped(tmp_path, Dying([("died", "NODE_FAIL", 9)]), ["--resume", "0"])
    assert "not resumed: --resume 0 disables it" in rep["error"]


# ---------------------------------------------------------------------------
# wtune_resume (pure / files)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state,model,want", [
    ("NODE_FAIL", False, "infrastructure"), ("TIMEOUT", True, "infrastructure"),
    ("PREEMPTED", False, "infrastructure"), ("BOOT_FAIL", False, "infrastructure"),
    ("FAILED", False, "infrastructure"), ("COMPLETED", False, "infrastructure"),
    ("FAILED", True, "model"), ("COMPLETED", True, "model"),
    ("CANCELLED", False, "cancelled"), ("CANCELLED", True, "cancelled"),
    ("OUT_OF_MEMORY", False, "other"), ("UNKNOWN", False, "other")])
def test_classify_failure(state, model, want):
    assert wres.classify_failure(state, model) == want


def test_rsl_mentions_any_rank(tmp_path):
    (tmp_path / "rsl.error.0000").write_text("all fine\n")
    assert not wres.rsl_mentions(str(tmp_path), wres.MODEL_NEEDLES)
    (tmp_path / "rsl.out.0007").write_text("Signal: Segmentation fault (11)\n")
    assert wres.rsl_mentions(str(tmp_path), wres.MODEL_NEEDLES)


def test_restart_write_finished_needs_every_domain(tmp_path):
    text = _rsl(T0, _at(4), [_at(2)], 2) + (RST % 1) + "\n"      # 04:00: killed writing d02
    assert wres.restart_writes_finished(text, 2) == [_at(2)]
    assert wres.restart_writes_finished(text, 1) == [_at(2), _at(4)]   # an end-of-run restart


def _real_bp(path):
    os.makedirs(path)
    for name in ("md.idx", "data.0"):
        with open(os.path.join(path, name), "w") as f:
            f.write("x")


def test_valid_restart_needs_complete_dirs_and_the_rsl_line(tmp_path):
    seg = tmp_path / "seg_0"
    seg.mkdir()
    name = lambda d, h: str(seg / ("wrfrst_d%02d_%s" % (d, _at(h).strftime(FMT))))  # noqa: E731
    for d in (1, 2):
        _real_bp(name(d, 2))                             # complete, write finished
        _real_bp(name(d, 8))                             # complete, but no rsl line
    _real_bp(name(1, 4))
    os.makedirs(name(2, 4))                              # d02 empty (pre-created only)
    _real_bp(name(1, 6))
    os.makedirs(name(2, 6))
    open(os.path.join(name(2, 6), "data.0"), "w").close()   # d02 partial: no md.idx
    (seg / "rsl.error.0000").write_text(_rsl(T0, _at(7), [_at(2), _at(4), _at(6)], 2))
    valid = lambda **kw: wres.valid_restart_times(str(seg), 2, REAL_BP.is_bp, **kw)  # noqa: E731
    assert valid() == ["2020-06-01_02:00:00"]
    assert valid(skip={"2020-06-01_02:00:00"}) == []
    assert wres.restart_sets_on_disk(str(seg), 2, REAL_BP.is_bp) == [
        "2020-06-01_02:00:00", "2020-06-01_08:00:00"]
    assert wres.last_d01_time(str(seg)) == _at(7)


def test_linked_restart_is_an_input_and_netcdf_restart_counts(tmp_path):
    seg = tmp_path / "seg_1"
    seg.mkdir()
    _real_bp(str(tmp_path / "prev"))
    os.symlink(str(tmp_path / "prev"), seg / "wrfrst_d01_2020-06-01_06:00:00")
    (seg / "wrfrst_d01_2020-06-01_08:00:00").write_text("CDF")
    (seg / "wrfrst_d01_2020-06-01_10:00:00").write_text("")
    assert wres.restart_sets_on_disk(str(seg), 1, REAL_BP.is_bp) == ["2020-06-01_08:00:00"]


def test_resume_point_rules():
    assert wres.resume_point([120, 240, 360, 480], 0, 720, 360, 540) == (360, None)
    assert wres.resume_point([120, 240, 360, 480], 0, 720, 1, 540) == (480, None)
    assert wres.resume_point([120, 240], 0, 720, 1, 240) == (120, None)    # run not past 240
    assert wres.resume_point([120, 240], 240, 720, 1, 700)[0] is None      # none after the start
    assert wres.resume_point([None, 720], 0, 720, 1, 720)[0] is None       # off grid / the end
    assert wres.resume_point([], 0, 720, 1, 700)[1].startswith("no valid restart")


# ---------------------------------------------------------------------------
# post / watch see the attempts as one run
# ---------------------------------------------------------------------------
def _frames(d, hours):
    d.mkdir(parents=True, exist_ok=True)
    for h in hours:
        (d / ("wrfout_d01_%s" % _at(h).strftime(FMT))).write_text("x")


def test_frame_listing_merges_attempts_later_wins(tmp_path):
    from post import extract
    out = tmp_path / "out"
    _frames(out / "seg_0", [0, 1, 2, 3])                 # died after 03:00, restart at 02:00
    _frames(out / "seg_0_r1", [3, 4])
    _frames(out / "seg_1", [5, 6])
    _frames(out / "seg_1_rescue", [5])
    _frames(out / "seg_1_rescue_r2", [6])
    _frames(out / "seg_10", [7])
    assert [os.path.basename(d) for d in extract.segment_dirs(str(out))] == [
        "seg_0", "seg_0_r1", "seg_1", "seg_1_rescue", "seg_1_rescue_r2", "seg_10"]
    got = [os.path.basename(os.path.dirname(p)) for _t, p in extract.list_frames(str(out))]
    assert got == ["seg_0", "seg_0", "seg_0", "seg_0_r1", "seg_0_r1", "seg_1_rescue",
                   "seg_1_rescue_r2", "seg_10"]


def test_watch_follows_the_latest_attempt(tmp_path):
    for name in ("seg_0", "seg_0_r1", "seg_0_r2"):
        (tmp_path / name).mkdir()
    assert os.path.basename(wtune_watch.follow_run_root(str(tmp_path))["dir"]) == "seg_0_r2"
    (tmp_path / "seg_0_rescue").mkdir()
    (tmp_path / "seg_0_rescue_r3").mkdir()
    fol = wtune_watch.follow_run_root(str(tmp_path))
    assert os.path.basename(fol["dir"]) == "seg_0_rescue_r3" and fol["segment"] == 0

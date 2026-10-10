"""wtune_watch: `wtune run --out` root following (seg_<k>/ + run_report.json) and the wrfout
progress floor for an rsl lagging behind on NFS. Synthetic dirs: 1-domain namelist (dt 60 s),
rsl.error.0000 with d01 Timing lines, empty wrfout_d01_<time> files."""
import datetime as dt
import json
import os
import time

from wtune import watch as ww

T0 = dt.datetime(2021, 7, 19, 0)
FMT = "%Y-%m-%d_%H:%M:%S"


def _namelist(start, hours):
    end = start + dt.timedelta(hours=hours)
    return ("&time_control\n run_hours = %d,\n" % hours
            + " start_year = %d,\n start_month = %d,\n start_day = %d,\n start_hour = %d,\n"
            % (start.year, start.month, start.day, start.hour)
            + " end_year = %d,\n end_month = %d,\n end_day = %d,\n end_hour = %d,\n"
            % (end.year, end.month, end.day, end.hour)
            + " history_interval = 60,\n/\n"
            + "&domains\n time_step = 60,\n max_dom = 1,\n e_we = 100,\n e_sn = 100,\n"
            + " e_vert = 40,\n/\n&physics\n radt = 10,\n/\n")


def _rsl(start, n_steps, tail=()):
    out = ["starting wrf task 0"]
    for i in range(1, n_steps + 1):
        t = start + dt.timedelta(seconds=60 * i)
        out.append("Timing for main: time %s on domain   1:    1.00000 elapsed seconds"
                   % t.strftime(FMT))
    return "\n".join(out + list(tail)) + "\n"


def _rundir(path, start=T0, hours=24, n_steps=0, tail=(), wrfout_hours=(), mtime=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "namelist.input").write_text(_namelist(start, hours))
    (path / "rsl.error.0000").write_text(_rsl(start, n_steps, tail))
    for h in wrfout_hours:
        p = path / ("wrfout_d01_" + (start + dt.timedelta(hours=h)).strftime(FMT))
        p.write_bytes(b"")
        if mtime is not None:
            os.utime(p, (mtime, mtime))
    return path


def _codes(snap):
    return [w["code"] for w in snap["warnings"]]


# ---------------------------------------------------------------------------
# wrfout floor
# ---------------------------------------------------------------------------
def test_wrfout_floor_with_empty_rsl_is_running_not_init(tmp_path):
    now = time.time() + 3600               # rsl is "old": HANG_INIT would fire without frames
    d = _rundir(tmp_path / "r", wrfout_hours=range(0, 14), mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "running"
    assert snap["d01_last_time"] == "2021-07-19_13:00:00"
    assert snap["completed_steps"] == 780
    assert abs(snap["fraction"] - 780 / 1440) < 1e-9
    assert "RSL_LAGGING" in _codes(snap) and "HANG_INIT" not in _codes(snap)
    assert "HANG" not in _codes(snap)
    assert "54.2%" in ww.format_line(snap)
    # the same rsl without any frame: init + HANG_INIT, as before
    d2 = _rundir(tmp_path / "r2")
    snap2 = ww.snapshot(str(d2), now=now)
    assert snap2["status"] == "init" and "HANG_INIT" in _codes(snap2)
    assert "RSL_LAGGING" not in _codes(snap2)


def test_wrfout_floor_over_stale_timing_lines(tmp_path):
    now = time.time()
    d = _rundir(tmp_path / "r", n_steps=60, wrfout_hours=range(0, 7), mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "running"
    assert snap["d01_last_time"] == "2021-07-19_06:00:00"
    assert snap["rsl_d01_last_time"] == "2021-07-19_01:00:00"
    assert snap["completed_steps"] == 360 and snap["remaining_steps"] == 1080
    assert snap["n_d01_steps"] == 60 and snap["rate_s_per_step"] is not None
    assert "RSL_LAGGING" in _codes(snap)


def test_wrfout_floor_frames_go_stale_then_hang(tmp_path):
    now = time.time() + 7200
    d = _rundir(tmp_path / "r", wrfout_hours=range(0, 3), mtime=now - 3600)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "running" and "HANG" in _codes(snap)


def test_wrfout_not_ahead_of_rsl_changes_nothing(tmp_path):
    now = time.time()
    d = _rundir(tmp_path / "r", n_steps=60, wrfout_hours=(0, 1), mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["d01_last_time"] == "2021-07-19_01:00:00" and snap["completed_steps"] == 60
    assert "RSL_LAGGING" not in _codes(snap) and "wrfout_last_time" not in snap


def test_wrfout_floor_never_masks_failed_or_done(tmp_path):
    now = time.time()
    d = _rundir(tmp_path / "f", n_steps=10, tail=("forrtl: severe (174): SIGSEGV",),
                wrfout_hours=range(0, 5), mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "failed" and "FAILED" in _codes(snap)
    assert "RSL_LAGGING" not in _codes(snap)
    d = _rundir(tmp_path / "d", n_steps=10, tail=("d01 SUCCESS COMPLETE WRF",),
                wrfout_hours=range(0, 5), mtime=now)
    snap = ww.snapshot(str(d), now=now)
    assert snap["status"] == "done" and snap["fraction"] == 1.0
    assert "RSL_LAGGING" not in _codes(snap)


def test_replay_ignores_wrfout(tmp_path):
    d = _rundir(tmp_path / "r", n_steps=60, wrfout_hours=range(0, 7))
    snap = ww.snapshot(str(d), replay_fraction=0.5)
    assert snap["status"] == "replay" and "RSL_LAGGING" not in _codes(snap)
    assert snap["d01_last_time"] == "2021-07-19_00:30:00"


# ---------------------------------------------------------------------------
# run root
# ---------------------------------------------------------------------------
def _root(tmp_path, report=True, rescue=False):
    root = tmp_path / "o2"
    s1 = T0 + dt.timedelta(hours=8)
    _rundir(root / "seg_0", T0, 8, n_steps=480, tail=("d01 SUCCESS COMPLETE WRF",))
    _rundir(root / "seg_1", s1, 8, n_steps=120)               # at T0 + 10 h
    if rescue:
        _rundir(root / "seg_1_rescue", s1, 8, n_steps=240)    # at T0 + 12 h
    if report:
        segs = [{"start": (T0 + dt.timedelta(hours=8 * k)).strftime(FMT),
                 "end": (T0 + dt.timedelta(hours=8 * (k + 1))).strftime(FMT)} for k in range(3)]
        (root / "run_report.json").write_text(json.dumps({"status": "running",
                                                          "segments": segs}))
    return root


def test_run_root_follows_latest_segment_with_whole_run_percentage(tmp_path, capsys):
    root = _root(tmp_path)
    fol = ww.follow_run_root(str(root))
    assert fol["dir"] == str(root / "seg_1") and fol["segment"] == 1 and fol["n_segments"] == 3
    assert ww.main([str(root), "--once"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("seg 1/3 RUNNING  41.7%"), line
    assert "NO_NAMELIST" not in line and "NO_RSL" not in line
    snap = ww.annotate_run_root(ww.snapshot(fol["dir"]), str(root), fol)
    assert abs(snap["segment_fraction"] - 0.25) < 1e-9
    assert snap["whole_run_start"] == "2021-07-19_00:00:00"
    assert snap["whole_run_end"] == "2021-07-20_00:00:00"


def test_run_root_without_report_uses_segment_window(tmp_path, capsys):
    root = _root(tmp_path, report=False)
    assert ww.main([str(root), "--once"]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("seg 1/? RUNNING  25.0%"), line


def test_run_root_prefers_rescue_attempt(tmp_path):
    fol = ww.follow_run_root(str(_root(tmp_path, rescue=True)))
    assert fol["dir"].endswith("seg_1_rescue") and fol["segment"] == 1


def test_run_root_finished_rule():
    fol = {"segment": 1, "n_segments": 3, "run_status": "running"}
    assert not ww.root_finished({"status": "done"}, fol)          # seg 1 of 3: keep polling
    assert ww.root_finished({"status": "failed"}, fol)
    assert ww.root_finished({"status": "done"}, dict(fol, segment=2))
    assert ww.root_finished({"status": "running"}, dict(fol, run_status="failed"))
    assert ww.root_finished({"status": "done"}, dict(fol, n_segments=None, run_status=None))


def test_single_dir_is_not_a_root(tmp_path, capsys):
    d = _rundir(tmp_path / "r", n_steps=60)
    (d / "seg_0").mkdir()                    # a stray seg_ dir does not matter with a namelist
    assert ww.follow_run_root(str(d)) is None
    assert ww.follow_run_root(str(tmp_path / "missing")) is None
    ww.main([str(d), "--once"])
    line = capsys.readouterr().out.strip()
    assert line.startswith("RUNNING   4.2%"), line

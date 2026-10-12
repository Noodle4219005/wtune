#!/usr/bin/env python3
"""`wtune run --checkpoint-minutes`: the restarts --resume resumes from, when the case's own
restart_interval puts none inside a segment. Fakes only."""
import json

import pytest

import test_wtune_run  # noqa: F401  first: its wtune.run import restores SITE/configs
from wtune import predict as wp
from wtune import run as wr
from test_wtune_run import (_geo, _make_case, _Spy, _with_restart,
                            fake_bp)  # noqa: F401  (autouse fixture)
from test_wtune_run_resume import Dying

ALIGN = {"steps": 360, "seconds": 21600.0, "label": "interval_seconds 06:00:00"}   # dt 60 s
HOUR = 60                                                                          # d01 steps


def _main(tmp_path, extra=(), fakes=None, case_text=None):
    if not (tmp_path / "case").exists():
        _make_case(tmp_path, *([case_text] if case_text else []))
    f, lines, out = fakes or _Spy(), [], tmp_path / "out"
    rc = wr.main([str(tmp_path / "case"), "--nodes", "1", "--nx", "8", "--ny", "9", "--out",
                  str(out), "--poll-interval", "0"] + list(extra), sbatch_runner=f.sbatch,
                 sacct_runner=f.sacct, snapshot_fn=f.snapshot, sleep=lambda s: None,
                 log=lines.append)
    return rc, out, f, lines


# ---------------------------------------------------------------------------
# the cadence (pure)
# ---------------------------------------------------------------------------
def test_auto_is_the_smallest_record_multiple_from_an_eighth_of_the_run():
    assert wr.auto_checkpoint_steps(72 * HOUR, None, 6 * HOUR) == 12 * HOUR   # SC26-P1 shape
    assert wr.auto_checkpoint_steps(72 * HOUR, None, 1) == 9 * HOUR
    assert wr.auto_checkpoint_steps(12 * HOUR, None, 6 * HOUR) == 6 * HOUR
    assert wr.auto_checkpoint_steps(6 * HOUR, None, 6 * HOUR) is None         # none inside
    # with segments: inside each one and dividing L
    assert wr.auto_checkpoint_steps(72 * HOUR, 24 * HOUR, 6 * HOUR) == 12 * HOUR
    assert wr.auto_checkpoint_steps(72 * HOUR, 18 * HOUR, 6 * HOUR) is None   # 12 h cuts 18 h
    assert wr.auto_checkpoint_steps(12 * HOUR, 6 * HOUR, 6 * HOUR) is None


def test_setting_auto_for_a_p1_like_run(tmp_path):
    g, n = _geo(tmp_path), 72 * HOUR
    rst = wr.restart_setting(g, None, n, 100000 * 60.0, "auto", ALIGN)   # case: 100000 min
    assert rst == {"set_min": 720, "cadence_s": 43200.0,
                   "label": "checkpoints every 720 min (auto, for --resume)"}
    assert wr.restart_write_steps(g, 0, n, rst["cadence_s"]) == [h * HOUR for h in
                                                                range(12, 73, 12)]
    assert wr.restart_setting(g, None, n, 100000 * 60.0)["set_min"] == wr.RESTART_OFF_MIN


def test_case_interval_wins_when_it_writes_inside_the_run(tmp_path):
    g = _geo(tmp_path)
    keep = wr.restart_setting(g, None, 720, 180 * 60.0, "auto", ALIGN)
    assert keep["set_min"] is None and keep["cadence_s"] == 10800.0
    keep = wr.restart_setting(g, 360, 720, 120 * 60.0, "auto")         # inside every segment
    assert keep["set_min"] is None and keep["cadence_s"] == 7200.0
    # the case's 6 h equals the segment: nothing inside it, so a checkpoint cadence is set
    assert wr.restart_setting(g, 360, 720, 360 * 60.0, "auto")["set_min"] == 90
    for kw in ({}, {"checkpoint": "auto", "align": ALIGN}):            # no checkpoint fits
        assert wr.restart_setting(g, 360, 720, 360 * 60.0, **kw)["set_min"] is None


def test_explicit_minutes_are_checked_against_the_record_step(tmp_path):
    g = _geo(tmp_path)
    assert wr.checkpoint_steps_for_minutes(g, 720, ALIGN) == 720
    assert wr.checkpoint_steps_for_minutes(g, 45) == 45
    with pytest.raises(wr.RunError, match=r"not a multiple of the 360 min input record step "
                                          r"\(interval_seconds 06:00:00\)"):
        wr.checkpoint_steps_for_minutes(g, 300, ALIGN)
    rst = wr.restart_setting(g, None, 720, 180 * 60.0, 360, ALIGN)     # wins over the case
    assert rst["set_min"] == 360 and rst["label"] == "checkpoints every 360 min " \
                                                     "(--checkpoint-minutes)"
    with pytest.raises(wr.RunError, match="does not divide the 06:00:00 segment length"):
        wr.restart_setting(g, 360, 720, 0.0, 240)


def test_request_auto_only_with_resume():
    assert wr.checkpoint_request("auto", 1) == "auto" and wr.checkpoint_request("auto", 0) is None
    assert wr.checkpoint_request("0", 1) is None and wr.checkpoint_request("90", 0) == 90
    with pytest.raises(wr.RunError, match="--checkpoint-minutes"):
        wr.checkpoint_request("soon", 1)


def test_checkpoint_writes_are_in_the_predicted_wall(tmp_path):
    g, over = _geo(tmp_path), wr.overheads(restart_write_s=50.0)
    cand = [{"config": wr.make_config(8, 9, 1, 1, 1), "compute": wr.constant_compute_fn(1.0)}]
    _L, off = wr.plan_run(g, 720, cand, over)
    _L, on = wr.plan_run(g, 720, cand, over, align=ALIGN, checkpoint="auto")   # 06:00, 12:00
    assert on[0]["predicted_wall_s"] - off[0]["predicted_wall_s"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# through `wtune run` (12 h case, wrfbdy records every 6 h, case restart_interval > run)
# ---------------------------------------------------------------------------
def test_auto_checkpoints_reach_namelist_xml_and_precreated_dirs(tmp_path):
    rc, out, f, lines = _main(tmp_path)
    assert rc == 0
    assert "1 segment(s), restart checkpoints every 360 min (auto, for --resume)" in lines[0]
    tc = wp.parse_namelist(str(out / "seg_0" / "namelist.input"))["time_control"]
    assert tc["restart_interval"] == [360] and tc["restart"] == [False]
    xml = (out / "seg_0" / "adios2.xml").read_text()
    names = ["wrfrst_d%02d_2020-06-01_%s:00:00" % (d, h) for h in ("06", "12") for d in (1, 2, 3)]
    assert all('<io name="%s">' % n in xml for n in names)
    assert all(n in f.seen["seg_0"] for n in names)                     # pre-created at submit
    rep = json.loads((out / "run_report.json").read_text())
    assert rep["restart_interval_min"] == 360 and rep["restart_cadence_s"] == 21600.0


def test_explicit_checkpoints_every_one_listed(tmp_path):
    rc, out, f, lines = _main(tmp_path, ["--io", "netcdf", "--checkpoint-minutes", "120"])
    assert rc == 0 and "restart checkpoints every 120 min (--checkpoint-minutes)" in lines[0]
    xml = (out / "seg_0" / "adios2.xml").read_text()
    assert all("wrfrst_d03_2020-06-01_%02d:00:00" % h in xml for h in range(2, 13, 2))


def test_explicit_off_the_record_step_is_a_clear_error(tmp_path, capsys):
    rc, out, f, _ = _main(tmp_path, ["--checkpoint-minutes", "300"])
    assert rc == 2 and f.submitted == []
    assert "--checkpoint-minutes 300 is not a multiple of the 360 min input record step" \
        in capsys.readouterr().err


def test_zero_disables_with_one_note(tmp_path):
    rc, out, f, lines = _main(tmp_path, ["--checkpoint-minutes", "0", "--dry-run"])
    assert rc == 0 and "restart off (one segment" in lines[0]
    assert sum("--resume has nothing to resume a failed job from" in ln for ln in lines) == 1


def test_resume_0_keeps_todays_plan_without_a_note(tmp_path):
    rc, out, f, lines = _main(tmp_path, ["--resume", "0"])
    assert rc == 0 and "restart off (one segment" in lines[0]
    assert not any("--resume has nothing" in ln for ln in lines)
    tc = wp.parse_namelist(str(out / "seg_0" / "namelist.input"))["time_control"]
    assert tc["restart_interval"] == [wr.RESTART_OFF_MIN]
    assert not [n for n in f.seen["seg_0"] if n.startswith("wrfrst")]


def test_case_interval_is_left_alone_through_main(tmp_path):
    rc, out, f, lines = _main(tmp_path, case_text=_with_restart(180))
    assert rc == 0 and "restart every 03:00:00 (case's restart_interval kept" in lines[0]


def test_a_node_failure_resumes_from_an_auto_checkpoint(tmp_path):
    f = Dying([("died", "NODE_FAIL", 9)])
    rc, out, _f, lines = _main(tmp_path, fakes=f)             # the case itself writes no restart
    assert rc == 0 and [s.rsplit("/", 1)[1] for s, _ in f.submitted] == ["seg_0", "seg_0_r1"]
    assert "  resume 1/1: infrastructure at job 1001; restart 2020-06-01_06:00:00" in lines
    tc = wp.parse_namelist(str(out / "seg_0_r1" / "namelist.input"))["time_control"]
    assert tc["restart_interval"] == [360] and tc["restart"] == [True]

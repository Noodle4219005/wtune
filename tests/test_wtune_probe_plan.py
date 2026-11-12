#!/usr/bin/env python3
"""pytest suite for wtune/probe_plan.py (2026-09-26 multi-point probe redesign).

See wtune/probe_plan.py's module docstring for the design (RANKING layer restart-start probes;
PREDICTION layer plan_points/combine) and this task's report for the conus25 evidence that
motivated it.
"""
import datetime

import pytest

from wtune import predict as predict
from wtune import probe_plan as plan


def _dt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")


# ---------------------------------------------------------------------------
# steps_for_stepra / resolve_ranking_steps
# ---------------------------------------------------------------------------
def test_steps_auto_is_50_for_radt10_dt15():
    # radt=10min, dt=15s -> stepra = round(10*60/15) = 40 -> steps = max(30, 40+10) = 50
    assert plan.resolve_ranking_steps(10.0, 15.0) == 50
    assert plan.steps_for_stepra(40) == 50


# ---------------------------------------------------------------------------
# explicit point parsing
# ---------------------------------------------------------------------------
def test_parse_explicit_point_all_three_forms():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = _dt("2020-06-01_10:00:00")   # 10h run

    at, reason, source = plan.parse_explicit_point("+90m", run_start, run_end)
    assert at == run_start + datetime.timedelta(minutes=90)

    at, reason, source = plan.parse_explicit_point("+2h", run_start, run_end)
    assert at == run_start + datetime.timedelta(hours=2)

    at, reason, source = plan.parse_explicit_point("70%", run_start, run_end)
    assert at == run_start + datetime.timedelta(hours=7)   # 70% of 10h

    at, reason, source = plan.parse_explicit_point("2019-11-26_23:00:00", run_start, run_end)
    assert at == _dt("2019-11-26_23:00:00")

    at, reason, source = plan.parse_explicit_point(
        {"at": "+1h", "reason": "night", "source": "rebuild"}, run_start, run_end)
    assert at == run_start + datetime.timedelta(hours=1)
    assert reason == "night"
    assert source == "rebuild"

    # legacy alias normalized
    at, reason, source = plan.parse_explicit_point({"at": "+1h", "source": "coldstart"},
                                                     run_start, run_end)
    assert source == "rebuild"


def test_parse_explicit_point_rejects_garbage():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = _dt("2020-06-01_10:00:00")
    with pytest.raises(plan.ProbePlanError):
        plan.parse_explicit_point("banana", run_start, run_end)


# ---------------------------------------------------------------------------
# plan_points
# ---------------------------------------------------------------------------
_DEFAULT_CFG = dict(steps="auto", points="auto", max_points=5, budget_pct=None,
                     sources=["trajectory", "restart", "rebuild"])


def test_plan_points_short_run_falls_back_to_single_point_with_warning():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(minutes=5)   # far shorter than spinup+window
    points, notes = plan.plan_points({}, run_start, run_end, dt=15.0, stepra=40,
                                      spinup_s=3600.0, cfg=dict(_DEFAULT_CFG))
    assert len(points) == 1
    assert points[0].at == run_start
    assert points[0].reason == "t0_fallback"
    assert any("too short" in n for n in notes)


def test_plan_points_explicit_out_of_range_is_an_error():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=6)
    cfg = dict(_DEFAULT_CFG, points=["+7h"])   # past run_end
    with pytest.raises(plan.ProbePlanError):
        plan.plan_points({}, run_start, run_end, dt=15.0, stepra=40, spinup_s=0.0, cfg=cfg)


def test_plan_points_explicit_duplicates_merge_not_error():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=6)
    cfg = dict(_DEFAULT_CFG,
               points=[{"at": "+1h", "reason": "a"}, {"at": "+1h", "reason": "b"}, "+61m"],
               sources=["rebuild"])
    points, notes = plan.plan_points({}, run_start, run_end, dt=15.0, stepra=40,
                                      spinup_s=0.0, cfg=cfg, forcing_interval_h=1.0)
    assert len(points) == 1   # all three collapse into one (well within one measure window)
    assert points[0].source == "rebuild"
    tokens = points[0].reason.split("+")
    assert "a" in tokens and "b" in tokens   # merged reasons kept, not dropped


def test_plan_points_budget_caps_the_count():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=24)
    cfg = dict(_DEFAULT_CFG, sources=["rebuild"], max_points=5, budget_pct=10.0)
    points_uncapped, _n = plan.plan_points({}, run_start, run_end, dt=60.0, stepra=15,
                                           spinup_s=0.0, cfg=dict(cfg, budget_pct=None),
                                           forcing_interval_h=1.0)
    assert len(points_uncapped) > 1
    points_capped, notes = plan.plan_points({}, run_start, run_end, dt=60.0, stepra=15,
                                            spinup_s=0.0, cfg=cfg, s_est=1.0,
                                            predicted_run_s=100.0, forcing_interval_h=1.0)
    assert len(points_capped) < len(points_uncapped)
    assert any("budget_pct" in n for n in notes)


def test_resolve_rebuild_point_drops_when_M_would_overrun_run_end():
    # Same grid/spinup, but a SHORT run (ends 22:15): T=21:00 (the latest grid time <=
    # p-spinup) gives M=22:00; M's own 1200s (20min) measure window would end at 22:20, past
    # run_end=22:15 -> dropped (M is None), T is still reported.
    run_start = _dt("2019-11-26_18:00:00")
    run_end = _dt("2019-11-26_22:15:00")
    p = _dt("2019-11-26_23:50:00")
    M, T = plan.resolve_rebuild_point(p, run_start, run_end, spinup_s=3600.0,
                                      rebuild_measure_window_s=1200.0, forcing_interval_h=3.0)
    assert M is None
    assert T == _dt("2019-11-26_21:00:00")


def test_plan_points_rebuild_merges_with_existing_trajectory_point_at_3h_cadence():
    # The exact conus25/G0014 C3 scenario: 3h cadence, 6h run 18Z-00Z, spinup 60min. A rebuild
    # point targeted at +2h (20:00, farther than the trajectory horizon) resolves to M=19:00,
    # which collides with the already-placed trajectory point at +1h (19:00) -- merged, not two
    # separate points.
    run_start = _dt("2019-11-26_18:00:00")
    run_end = run_start + datetime.timedelta(hours=6)
    cfg = dict(_DEFAULT_CFG,
               points=[{"at": "+1h", "source": "trajectory"}, {"at": "+2h", "source": "rebuild"}])
    points, notes = plan.plan_points({}, run_start, run_end, dt=15.0, stepra=40, spinup_s=3600.0,
                                      cfg=cfg, forcing_interval_h=3.0)
    assert len(points) == 1
    assert points[0].at == _dt("2019-11-26_19:00:00")
    assert points[0].source == "trajectory"   # the trajectory point was placed FIRST (idx 0),
                                               # so it keeps priority in the merge
    assert "rebuild" in points[0].reason


# ---------------------------------------------------------------------------
# ranking-layer restart-start namelist (offline, no cluster)
# ---------------------------------------------------------------------------
_RESTART_NAMELIST = (
    "&time_control\n"
    " run_days = 0, run_hours = 6, run_minutes = 0, run_seconds = 0,\n"
    " start_year = 2019, 2019,\n"
    " start_month = 11, 11,\n"
    " start_day = 25, 25,\n"
    " start_hour = 12, 12,\n"
    " start_minute = 0, 0,\n"
    " restart = .false.,\n"
    " restart_interval = 1440,\n"
    " history_interval = 60, 0,\n"
    "/\n"
    "&domains\n"
    " max_dom = 2,\n"
    " nproc_x = 8, nproc_y = 8,\n"
    " numtiles = 4,\n"
    "/\n"
)


def test_build_restart_start_namelist_run_length_is_steps_times_dt():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0)
    # 50 * 15 = 750s
    assert "run_seconds = 750," in out
    assert "run_days = 0," in out
    assert "run_hours = 0," in out
    assert "run_minutes = 0," in out


def test_build_restart_start_namelist_writes_decomp_when_given():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0,
                                                  nproc_x=4, nproc_y=16, numtiles=2)
    assert "nproc_x = 4," in out
    assert "nproc_y = 16," in out
    assert "numtiles = 2," in out


    # a phase-0 reader (i%4==0) would instead sample the plain 1.0-valued steps as "radiation"
    # and report r~=1.0, not 5.0 -- the assertion above already discriminates that bug.


def test_combine_integral_matches_linear_rise_within_1_percent():
    # s(t) rises 1.0 -> 1.1 linearly over 6h; points measured at +1h, +3h, +5h.
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=6)
    dt = 15.0
    total_s_seconds = 6 * 3600.0

    def s_true(t_s):
        return 1.0 + 0.1 * (t_s / total_s_seconds)

    points = []
    for hours in (1, 3, 5):
        t_s = hours * 3600.0
        points.append(plan.PointMeasurement(at=run_start + datetime.timedelta(hours=hours),
                                             reason="probe", source="trajectory",
                                             s_normal=s_true(t_s), n_rad=0, r=None))
    result = plan.combine(points, run_start, run_end, dt, stepra=10_000_000, calibration={})
    # ground truth: integral (in STEPS) of the piecewise-linear interpolation through the three
    # measured points, held flat before/after -- no radiation correction (stepra huge -> no
    # radiation steps at all in this run length).
    steps_n = int((run_end - run_start).total_seconds() / dt)
    anchors_s = [(1 * 3600.0, s_true(1 * 3600.0)), (3 * 3600.0, s_true(3 * 3600.0)),
                 (5 * 3600.0, s_true(5 * 3600.0))]
    anchors_s.sort()
    total_expected = 0.0
    for i in range(1, steps_n + 1):
        t = i * dt
        if t <= anchors_s[0][0]:
            total_expected += anchors_s[0][1]
        elif t >= anchors_s[-1][0]:
            total_expected += anchors_s[-1][1]
        else:
            for j in range(len(anchors_s) - 1):
                t0, s0 = anchors_s[j]
                t1, s1 = anchors_s[j + 1]
                if t0 <= t <= t1:
                    frac = (t - t0) / (t1 - t0)
                    total_expected += s0 + frac * (s1 - s0)
                    break
    assert result["compute_total_s"] == pytest.approx(total_expected, rel=0.01)

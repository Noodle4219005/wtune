#!/usr/bin/env python3
"""pytest suite for wtune_probe_plan.py (2026-09-26 multi-point probe redesign).

See wtune_probe_plan.py's module docstring for the design (RANKING layer restart-start probes;
PREDICTION layer plan_points/combine) and this task's report for the conus25 evidence that
motivated it.
"""
import datetime

import pytest

import wtune_predict as predict
import wtune_probe_plan as plan


def _dt(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")


# ---------------------------------------------------------------------------
# steps_for_stepra / resolve_ranking_steps
# ---------------------------------------------------------------------------
def test_steps_auto_is_50_for_radt10_dt15():
    # radt=10min, dt=15s -> stepra = round(10*60/15) = 40 -> steps = max(30, 40+10) = 50
    assert plan.resolve_ranking_steps(10.0, 15.0) == 50
    assert plan.steps_for_stepra(40) == 50


def test_steps_for_stepra_floors_at_30():
    assert plan.steps_for_stepra(5) == 30   # max(30, 5+10)=30


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


def test_plan_points_night_day_respect_sunlit():
    # lon=0: solar_hour == UTC hour: noon UTC is sunlit, midnight UTC is not. A 3x3 lattice
    # (not a single point) avoids the exact-tie plateaus a single binary sample point produces
    # near sunrise/sunset. run_start is deliberately NOT midnight/noon, so t0 does not coincide
    # with either extreme.
    run_start = _dt("2020-06-01_06:00:00")
    run_end = run_start + datetime.timedelta(hours=24)
    sample_points = [(lat, lon) for lat in (-10.0, 0.0, 10.0) for lon in (-10.0, 0.0, 10.0)]
    cfg = dict(_DEFAULT_CFG, sources=["rebuild"])
    points, notes = plan.plan_points({}, run_start, run_end, dt=60.0, stepra=15,
                                      spinup_s=0.0, cfg=cfg, sample_points=sample_points)
    by_reason = {p.reason: p for p in points}
    assert "night" in by_reason and "day" in by_reason
    night_frac = predict.sunlit_fraction(sample_points, by_reason["night"].at)
    day_frac = predict.sunlit_fraction(sample_points, by_reason["day"].at)
    assert day_frac > night_frac
    assert day_frac > 0.5
    assert night_frac < 0.5
    for p in points:
        assert p.source == "rebuild"


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
                                      spinup_s=0.0, cfg=cfg)
    assert len(points) == 1   # all three collapse into one (well within one measure window)
    assert "a" in points[0].reason and "b" in points[0].reason   # merged reasons kept, not dropped


def test_plan_points_budget_caps_the_count():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=24)
    cfg = dict(_DEFAULT_CFG, sources=["rebuild"], max_points=5, budget_pct=10.0)
    points_uncapped, _n = plan.plan_points({}, run_start, run_end, dt=60.0, stepra=15,
                                           spinup_s=0.0, cfg=dict(cfg, budget_pct=None))
    assert len(points_uncapped) > 1
    points_capped, notes = plan.plan_points({}, run_start, run_end, dt=60.0, stepra=15,
                                            spinup_s=0.0, cfg=cfg, s_est=1.0,
                                            predicted_run_s=100.0)
    assert len(points_capped) < len(points_uncapped)
    assert any("budget_pct" in n for n in notes)


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


def test_build_restart_start_namelist_sets_start_for_every_domain_and_restart_true():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0)
    assert "start_year = 2019, 2019," in out
    assert "start_month = 11, 11," in out
    assert "start_day = 26, 26," in out
    assert "start_hour = 23, 23," in out
    assert "start_minute = 00, 00," in out
    assert "start_second = 00, 00," in out
    assert "restart = .true.," in out
    assert "restart = .false." not in out


def test_build_restart_start_namelist_run_length_is_steps_times_dt():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0)
    # 50 * 15 = 750s
    assert "run_seconds = 750," in out
    assert "run_days = 0," in out
    assert "run_hours = 0," in out
    assert "run_minutes = 0," in out


def test_build_restart_start_namelist_history_off():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0)
    assert "history_interval = 999999, 999999," in out


def test_build_restart_start_namelist_restart_interval_forced_huge():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0)
    assert "restart_interval = 999999," in out


def test_build_restart_start_namelist_writes_decomp_when_given():
    t = _dt("2019-11-26_23:00:00")
    out = plan.build_restart_start_namelist_text(_RESTART_NAMELIST, t, steps=50, dt_seconds=15.0,
                                                  nproc_x=4, nproc_y=16, numtiles=2)
    assert "nproc_x = 4," in out
    assert "nproc_y = 16," in out
    assert "numtiles = 2," in out


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------
def test_combine_single_point_equals_constant_model():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=6)
    dt = 15.0
    steps_n = int((run_end - run_start).total_seconds() / dt)
    pm = plan.PointMeasurement(at=run_start + datetime.timedelta(hours=1), reason="t0",
                                source="trajectory", s_normal=1.0, n_rad=2, r=3.0)
    result = plan.combine([pm], run_start, run_end, dt, stepra=40, calibration={})
    # constant model: every normal step costs 1.0s, every radiation step costs r*1.0=3.0s
    stepra = 40
    n_rad_steps = sum(1 for i in range(1, steps_n + 1) if i == 1 or i % stepra == 0)
    expected = (steps_n - n_rad_steps) * 1.0 + n_rad_steps * 3.0
    assert result["compute_total_s"] == pytest.approx(expected, rel=1e-6)
    assert result["r_night"] == pytest.approx(3.0)
    assert result["r_day"] == pytest.approx(3.0)


def test_combine_radiation_ratio_pooled_from_points():
    run_start = _dt("2020-06-01_00:00:00")
    run_end = run_start + datetime.timedelta(hours=12)
    dt = 60.0
    night_pm = plan.PointMeasurement(at=run_start, reason="night", source="rebuild",
                                      s_normal=1.0, n_rad=1, r=2.0)
    day_pm = plan.PointMeasurement(at=run_start + datetime.timedelta(hours=12), reason="day",
                                    source="rebuild", s_normal=1.0, n_rad=1, r=6.0)
    sample_points = [(0.0, 0.0)]   # night_pm at 00:00 UTC (dark), day_pm at 12:00 UTC (lit)
    result = plan.combine([night_pm, day_pm], run_start, run_end, dt, stepra=15,
                          calibration={}, sample_points=sample_points)
    assert result["r_source"] == "fit_day_night"
    assert result["r_night"] == pytest.approx(2.0, abs=0.5)
    assert result["r_day"] == pytest.approx(6.0, abs=0.5)


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


# ---------------------------------------------------------------------------
# post-hoc check (labelled post-hoc -- checks the COMBINER arithmetic against the G0014 truth,
# not the placement policy): feeds G0014's measured hourly normal-step medians as if they were
# point measurements at +1h/+3h/+5h and compares against the truth wall time 1783.4s.
# ---------------------------------------------------------------------------
def test_posthoc_g0014_combine_vs_truth():
    # conus25 G0014 truth (2026-09-26 evidence, see wtune_probe_plan.py's module docstring):
    # hourly normal-step medians for hours 0-5, dt=15s, 1440 steps total, 37 radiation steps
    # measured at ratio 4.97, init=69s. Truth: wall - writes = 1783.4s.
    run_start = _dt("2020-01-01_00:00:00")
    dt = 15.0
    n_steps_total = 1440
    run_end = run_start + datetime.timedelta(seconds=n_steps_total * dt)
    hourly_medians = [1.018, 1.084, 1.088, 1.093, 1.112, 1.115]   # hours 0..5
    points = [plan.PointMeasurement(at=run_start + datetime.timedelta(hours=h), reason="probe",
                                     source="trajectory", s_normal=hourly_medians[h],
                                     n_rad=(37 if h in (1, 3, 5) else 0),
                                     r=(4.97 if h in (1, 3, 5) else None))
              for h in (1, 3, 5)]
    calibration = {"radiation": {"r_rad_default": 4.97}}
    result = plan.combine(points, run_start, run_end, dt, stepra=40, calibration=calibration)
    total_no_init = result["compute_total_s"]
    truth = 1783.4
    init_s = 69.0
    total_with_init = total_no_init + init_s
    rel_err = (total_with_init - truth) / truth
    print("posthoc: combine(compute)=%.1f +init(%.1f)=%.1f truth=%.1f rel_err=%.2f%%"
          % (total_no_init, init_s, total_with_init, truth, rel_err * 100))
    # This is a post-hoc arithmetic check, not a placement check -- 3 hourly points reproducing a
    # continuous ramp is expected to be within a working margin, not exact; require materially
    # better than the old single-t0-probe model (which under-predicted by 9.6%).
    assert abs(rel_err) < 0.05

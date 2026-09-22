#!/usr/bin/env python3
"""pytest suite for tools/temporal_validation.py (decision 0053 step 2, offline temporal
validation). Fake sbatch runners, synthetic ledger/rsl files -- no cluster access. Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest -q tools/test_temporal_validation.py
"""
import argparse
import csv
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import temporal_validation as tv  # noqa: E402

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45, dt=20, prep_dir="/nonexistent", bp_dir="",
                     input_files=[], case_family="synthetic", steps=20, enabled=True)

NML_1DOM = """&time_control
 run_days = 0,
 run_hours = 12,
 run_minutes = 0,
 run_seconds = 0,
 start_year = 2018,
 start_month = 08,
 start_day = 25,
 start_hour = 00,
 start_minute = 00,
 start_second = 00,
 history_interval = 180,
 restart = .false.,
 restart_interval = 5000,
 io_form_history = 2
 io_form_restart = 2
 io_form_input = 2
 io_form_boundary = 2
/
&domains
 max_dom = 1,
 e_we = 901,
 e_sn = 901,
/
&namelist_quilt
 nio_tasks_per_group = 4,
 nio_groups = 1,
/
"""

NML_2DOM = NML_1DOM.replace(
    "start_year = 2018,", "start_year = 2018, 2018,").replace(
    "start_month = 08,", "start_month = 08, 08,").replace(
    "start_day = 25,", "start_day = 25, 25,").replace(
    "start_hour = 00,", "start_hour = 00, 00,").replace(
    "start_minute = 00,", "start_minute = 00, 00,").replace(
    "start_second = 00,", "start_second = 00, 00,").replace(
    "max_dom = 1,", "max_dom = 2,")

# job 587/588 postmortem shapes, transcribed from the real cluster namelists cited in the
# coordinator's rejection (isc21: /home/nkli/wrf-cases/isc21/WRF_ISC21_3_DOMAIN/namelist.input;
# lafe: /nfs/data/wrf/doi4-lafe2017/namelist.input.LAFE_2500_500_100_MYNN) -- only the
# &time_control keys read_run_length_hours/read_case_start_datetime/read_case_end_datetime
# actually consume.
NML_ISC21_SHAPE = """&time_control
 run_days = 0,
 run_hours = 0,
 run_minutes = 30,
 run_seconds = 00,
 start_year = 2016,
 start_month = 08,
 start_day = 29,
 start_hour = 09,
 start_minute = 00,
 start_second = 00,
 end_year = 2016,
 end_month = 08,
 end_day = 30,
 end_hour = 00,
 end_minute = 00,
 end_second = 00,
/
&domains
 max_dom = 1,
/
"""

NML_LAFE_SHAPE = """&time_control
 run_days = 0,
 run_hours = 0,
 run_minutes = 0,
 run_seconds = 0,
 start_year = 2017,
 start_month = 08,
 start_day = 23,
 start_hour = 06,
 start_minute = 00,
 start_second = 00,
 end_year = 2017,
 end_month = 08,
 end_day = 24,
 end_hour = 06,
 end_minute = 00,
 end_second = 00,
/
&domains
 max_dom = 1,
/
"""


# ---------------------------------------------------------------------------
# restart time offsets
# ---------------------------------------------------------------------------
def test_restart_offsets_default_12_3():
    assert tv.restart_offsets_hours(12, 3) == [0, 3, 6, 9, 12]


def test_restart_offsets_nonmultiple_appends_final():
    assert tv.restart_offsets_hours(10, 3) == [0, 3, 6, 9, 10]


def test_restart_offsets_every_equals_hours():
    assert tv.restart_offsets_hours(3, 3) == [0, 3]


def test_restart_offsets_zero_hours():
    assert tv.restart_offsets_hours(0, 3) == [0]


def test_steps_to_minutes_rounds_up():
    # 30 steps * 25 s = 750 s = 12.5 min -> ceil 13
    assert tv.steps_to_minutes(30, 25) == 13
    # exact minute boundary stays exact
    assert tv.steps_to_minutes(24, 15) == 6


def test_offset_datetime_and_fmt():
    start = datetime.datetime(2018, 8, 25, 0, 0, 0)
    d = tv.offset_datetime(start, 3)
    assert d == datetime.datetime(2018, 8, 25, 3, 0, 0)
    assert tv.fmt_wrf_time(d) == "2018-08-25_03:00:00"


def test_wrfrst_filenames_two_domains():
    dt = datetime.datetime(2018, 8, 25, 6, 0, 0)
    assert tv.wrfrst_filenames(2, dt) == [
        "wrfrst_d01_2018-08-25_06:00:00", "wrfrst_d02_2018-08-25_06:00:00"]


# ---------------------------------------------------------------------------
# namelist edits: long run (cold start, netCDF restart writer)
# ---------------------------------------------------------------------------
def test_edit_namelist_long_cold_start_netcdf_restart():
    out = tv.edit_namelist_long(NML_2DOM, nproc_x=16, nproc_y=18, numtiles=1, ranks=288,
                                 max_dom=2, run_minutes=720, restart_every_minutes=180)
    assert "restart = .false." in out
    assert "restart_interval = 180" in out
    assert "io_form_restart = 2" in out
    assert "io_form_input = 2" in out
    assert "io_form_boundary = 2" in out
    assert "run_minutes = 720" in out
    assert "history_interval = 0, 0," in out          # history off
    assert "nio_tasks_per_group=0" in out              # quilting disabled
    assert "nproc_x=16, nproc_y=18, numtiles=1," in out
    # native start_* untouched (still the template's own 2018-08-25_00:00:00)
    assert "start_year = 2018, 2018," in out
    assert "start_hour = 00, 00," in out or "start_hour = 0, 0," in out


def test_edit_namelist_long_never_writes_adios2_group():
    out = tv.edit_namelist_long(NML_1DOM, 10, 10, 1, 100, 1, 60, 60)
    assert "&namelist_adios2" not in out
    assert "io_form_restart = 14" not in out


# ---------------------------------------------------------------------------
# namelist edits: samples (restart-from-T reader, and t0 cold start)
# ---------------------------------------------------------------------------
def test_edit_namelist_sample_restart_from_t_single_domain():
    restart_dt = datetime.datetime(2018, 8, 25, 3, 0, 0)
    out = tv.edit_namelist_sample(NML_1DOM, nproc_x=9, nproc_y=8, numtiles=1, ranks=72,
                                   max_dom=1, run_minutes=13, restart_dt=restart_dt)
    assert "restart = .true." in out
    assert "io_form_restart = 2" in out
    assert "run_minutes = 13" in out
    assert "start_year = 2018," in out
    assert "start_hour = 3," in out
    assert "restart_interval = 1000000" in out   # never restart-writes during this short window


def test_edit_namelist_sample_restart_from_t_multi_domain_aligns_all_domains():
    restart_dt = datetime.datetime(2018, 8, 25, 6, 0, 0)
    out = tv.edit_namelist_sample(NML_2DOM, 8, 9, 1, 72, 2, 13, restart_dt=restart_dt)
    assert "restart = .true." in out
    assert "start_year = 2018, 2018," in out
    assert "start_hour = 6, 6," in out


def test_edit_namelist_sample_t0_is_cold_start():
    # restart_dt=None (T=0): task spec says "t0 uses the normal cold start" -- restart=.false.,
    # native start_* left untouched (never rewritten to a computed value).
    out = tv.edit_namelist_sample(NML_1DOM, 9, 8, 1, 72, 1, 13, restart_dt=None)
    assert "restart = .false." in out
    assert "restart = .true." not in out
    assert "start_year = 2018," in out   # untouched native value, not rewritten
    assert "io_form_restart = 2" in out


# ---------------------------------------------------------------------------
# rsl timing parsing / drift curve
# ---------------------------------------------------------------------------
def _rsl_line(step_second, domain, elapsed):
    return ("Timing for main: time 2018-08-25_00:%02d:00 on domain %4d:   %8.5f elapsed seconds"
            % (step_second, domain, elapsed))


def test_parse_domain1_step_series_ignores_nested_domain():
    text = "\n".join([
        _rsl_line(0, 1, 2.0), _rsl_line(0, 2, 0.4),
        _rsl_line(1, 1, 1.8), _rsl_line(1, 2, 0.5),
        _rsl_line(2, 1, 2.2),
    ])
    assert tv.parse_domain1_step_series(text) == [2.0, 1.8, 2.2]


def test_parse_domain1_step_series_tolerates_stray_null_bytes():
    text = _rsl_line(0, 1, 3.0) + "\x00garbled\n" + _rsl_line(1, 1, 3.1)
    assert tv.parse_domain1_step_series(text) == [3.0, 3.1]


def test_median_s_per_step_drops_first_step():
    assert tv.median_s_per_step([100.0, 2.0, 2.0, 2.0]) == 2.0
    assert tv.median_s_per_step([100.0]) is None
    assert tv.median_s_per_step([]) is None


def test_drift_curve_buckets_by_simulated_hour_and_normalizes():
    # dt=1000s: step i (0-based) starts at sim_time i*1000s. Step 1 (index 0, excluded) starts
    # hour 0; indices 1-3 (starts 1000/2000/3000s) stay in hour 0; indices 4-7 (starts
    # 4000..7000s) fall in hour 1 ([3600,7200)); indices 8-10 (starts 8000..10000s) in hour 2.
    series = [999.0] + [4.0] * 3 + [8.0] * 4 + [16.0] * 3
    rows = tv.drift_curve(series, dt_seconds=1000)
    by_hour = {r["sim_hour"]: r for r in rows}
    assert by_hour[0]["median_s_per_step"] == 4.0
    assert by_hour[0]["n_steps"] == 3
    assert by_hour[0]["normalized"] == 1.0
    assert by_hour[1]["median_s_per_step"] == 8.0
    assert by_hour[1]["n_steps"] == 4
    assert by_hour[1]["normalized"] == 2.0
    assert by_hour[2]["median_s_per_step"] == 16.0
    assert by_hour[2]["normalized"] == 4.0


def test_drift_curve_sub_hour_steps_bucket_together():
    # dt=900s: step starts at 900/1800/2700s (indices 1-3) all stay under the 3600s hour-0
    # boundary, so all three land in the same bucket.
    series = [1.0, 2.0, 2.0, 2.0]
    rows = tv.drift_curve(series, dt_seconds=900)
    assert len(rows) == 1
    assert rows[0]["sim_hour"] == 0
    assert rows[0]["n_steps"] == 3
    assert rows[0]["median_s_per_step"] == 2.0


def test_drift_curve_crosses_hour_boundary_into_new_bucket():
    # dt=1200s: index 3 starts at exactly 3600s -> hour 1, not hour 0 (floor at the boundary).
    series = [1.0, 2.0, 2.0, 6.0]
    rows = tv.drift_curve(series, dt_seconds=1200)
    by_hour = {r["sim_hour"]: r for r in rows}
    assert by_hour[0]["n_steps"] == 2
    assert by_hour[0]["median_s_per_step"] == 2.0
    assert by_hour[1]["n_steps"] == 1
    assert by_hour[1]["median_s_per_step"] == 6.0
    assert by_hour[1]["normalized"] == 3.0


# ---------------------------------------------------------------------------
# Kendall tau
# ---------------------------------------------------------------------------
def test_kendall_tau_perfect_agreement():
    assert tv.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_kendall_tau_perfect_reversal():
    assert tv.kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_kendall_tau_partial_agreement():
    # one adjacent swap out of 4 items: 5 concordant, 1 discordant pair -> tau = 4/6
    tau = tv.kendall_tau([1, 2, 3, 4], [1, 2, 4, 3])
    assert abs(tau - (4.0 / 6.0)) < 1e-9


def test_kendall_tau_constant_sequence_is_nan():
    import math
    assert math.isnan(tv.kendall_tau([1, 1, 1], [1, 2, 3]))


# ---------------------------------------------------------------------------
# rankings / within-2% toy scenario
# ---------------------------------------------------------------------------
def _toy_samples_rows():
    # 3 configs (A best at t0, B best at +6h), 2 restart times.
    return [
        dict(restart_hours=0, config_id="A", ok=1, steps=[99, 1.0, 1.0, 1.0]),
        dict(restart_hours=0, config_id="B", ok=1, steps=[99, 1.5, 1.5, 1.5]),
        dict(restart_hours=0, config_id="C", ok=1, steps=[99, 2.0, 2.0, 2.0]),
        dict(restart_hours=6, config_id="A", ok=1, steps=[99, 1.3, 1.3, 1.3]),
        dict(restart_hours=6, config_id="B", ok=1, steps=[99, 1.0, 1.0, 1.0]),
        dict(restart_hours=6, config_id="C", ok=1, steps=[99, 2.0, 2.0, 2.0]),
    ]


def test_build_rankings_orders_by_median_ascending():
    rankings = tv.build_rankings(_toy_samples_rows())
    at_t0 = [r for r in rankings if r["restart_hours"] == 0]
    assert [r["config_id"] for r in sorted(at_t0, key=lambda r: r["rank"])] == ["A", "B", "C"]
    at_t6 = [r for r in rankings if r["restart_hours"] == 6]
    assert [r["config_id"] for r in sorted(at_t6, key=lambda r: r["rank"])] == ["B", "A", "C"]


def test_build_rankings_failed_run_ranks_last_with_none():
    rows = _toy_samples_rows()
    rows.append(dict(restart_hours=0, config_id="D", ok=0, steps=[]))
    rankings = tv.build_rankings(rows)
    d_row = next(r for r in rankings if r["restart_hours"] == 0 and r["config_id"] == "D")
    assert d_row["median_s_per_step"] is None
    assert d_row["rank"] is None


def test_summarize_tau_and_within_2pct():
    rankings = tv.build_rankings(_toy_samples_rows())
    summary, final_hours = tv.summarize(rankings, tau_threshold=0.7, pct_threshold=0.02)
    assert final_hours == 6
    row0 = next(r for r in summary if r["restart_hours"] == 0)
    assert row0["kendall_tau_vs_t0"] is None       # T==0 has no tau vs itself
    assert row0["best_config"] == "A"
    row6 = next(r for r in summary if r["restart_hours"] == 6)
    assert row6["is_final"] is True
    assert row6["best_config"] == "B"
    assert row6["best_config_t0"] == "A"
    # A's own s/step at T=6 is 1.3, B (T=6's actual best) is 1.0 -> 30% over, well outside 2%
    assert abs(row6["pct_over_using_t0_choice"] - 0.3) < 1e-9
    assert row6["within_pct_threshold"] is False
    # ranking reverses (A>B>C at t0 vs B>A>C at +6h among A/B, C tied last both times) -> tau<1
    assert row6["kendall_tau_vs_t0"] < 1.0


def test_summarize_within_2pct_true_when_close():
    rows = [
        dict(restart_hours=0, config_id="A", ok=1, steps=[99, 1.00, 1.00]),
        dict(restart_hours=0, config_id="B", ok=1, steps=[99, 1.01, 1.01]),
        dict(restart_hours=3, config_id="A", ok=1, steps=[99, 1.00, 1.00]),
        dict(restart_hours=3, config_id="B", ok=1, steps=[99, 1.005, 1.005]),  # 0.5% faster than A
    ]
    rankings = tv.build_rankings(rows)
    summary, _final = tv.summarize(rankings, pct_threshold=0.02)
    row3 = next(r for r in summary if r["restart_hours"] == 3)
    assert row3["best_config_t0"] == "A"
    assert row3["within_pct_threshold"] is True
    assert row3["pct_over_using_t0_choice"] < 0.02


def test_summarize_raises_without_t0_baseline():
    rows = [dict(restart_hours=3, config_id="A", ok=1, steps=[99, 1.0])]
    rankings = tv.build_rankings(rows)
    try:
        tv.summarize(rankings)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# CSV round trips
# ---------------------------------------------------------------------------
def test_rankings_summary_drift_csv_round_trip(tmp_path):
    rankings = tv.build_rankings(_toy_samples_rows())
    summary, _final = tv.summarize(rankings)
    drift = tv.drift_curve([10.0, 4.0, 4.0, 4.0], dt_seconds=900)

    tv.write_rankings_csv(str(tmp_path / "rankings.csv"), rankings)
    tv.write_summary_csv(str(tmp_path / "summary.csv"), summary)
    tv.write_drift_csv(str(tmp_path / "drift.csv"), drift)

    with open(tmp_path / "rankings.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(rankings)
    assert rows[0]["config_id"] in ("A", "B", "C")

    with open(tmp_path / "summary.csv") as f:
        srows = list(csv.DictReader(f))
    assert {r["restart_hours"] for r in srows} == {"0", "6"}

    with open(tmp_path / "drift.csv") as f:
        drows = list(csv.DictReader(f))
    assert drows[0]["normalized"] == "1.0"


def test_read_samples_ledger_round_trip(tmp_path):
    path = tmp_path / "temporal_karina_samples_h3_999.csv"
    with open(path, "w") as f:
        f.write("restart_hours,config_id,ranks,omp,nx,ny,numtiles,dist,rc,ok,steps\n")
        f.write("3,ANCHOR,72,1,8,9,1,block,0,1,1.900000;2.000000;2.100000\n")
        f.write("3,G0001,72,1,9,8,1,cyclic,124,0,\n")
    rows = tv.read_samples_ledger(str(path))
    assert rows[0]["restart_hours"] == 3
    assert rows[0]["config_id"] == "ANCHOR"
    assert rows[0]["steps"] == [1.9, 2.0, 2.1]
    assert rows[1]["ok"] == "0"
    assert rows[1]["steps"] == []


def test_read_long_timing_csv(tmp_path):
    path = tmp_path / "timing.csv"
    with open(path, "w") as f:
        f.write("step_index,s_per_step\n1,2.5\n2,1.9\n3,2.0\n")
    assert tv.read_long_timing_csv(str(path)) == [2.5, 1.9, 2.0]


# ---------------------------------------------------------------------------
# case selection / plan (auto_select_cases, read_run_length_hours)
# ---------------------------------------------------------------------------
def _write_namelist(path, run_hours):
    path.write_text(
        "&time_control\n run_days = 0,\n run_hours = %d,\n run_minutes = 0,\n"
        " start_year = 2018,\n start_month = 1,\n start_day = 1,\n/\n&domains\n/\n" % run_hours)


def test_read_run_length_hours():
    assert tv.read_run_length_hours(NML_1DOM) == 12.0


# ---------------------------------------------------------------------------
# read_run_length_hours: job 587/588 postmortem -- 0 from run_* keys is ambiguous between
# "genuinely short" and "length is defined by start_date -> end_date"; these two must resolve
# to DIFFERENT, correct answers, never to the same wrong 0.
# ---------------------------------------------------------------------------
def test_read_run_length_hours_isc21_shape_sub_hour_run_minutes_wins_over_date_span():
    # run_minutes=30 alone is nonzero -> the explicit 30-minute run_* value is authoritative;
    # the 15-hour start->end date span must NOT override it.
    assert tv.read_run_length_hours(NML_ISC21_SHAPE) == 0.5


def test_read_run_length_hours_lafe_shape_all_run_star_zero_uses_date_span():
    # every run_* key is 0 -> WRF derives the length from start_date (2017-08-23 06:00) ->
    # end_date (2017-08-24 06:00) = 24 hours (1440 minutes), NOT 0 and NOT a 1-minute floor.
    hours = tv.read_run_length_hours(NML_LAFE_SHAPE)
    assert hours == 24.0
    assert hours * 60 == 1440.0


def test_read_run_length_hours_raises_when_neither_source_available():
    text = "&time_control\n run_days = 0,\n run_hours = 0,\n run_minutes = 0,\n run_seconds = 0,\n/\n&domains\n/\n"
    try:
        tv.read_run_length_hours(text, path="/fake/namelist.input")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "/fake/namelist.input" in str(e)


def test_read_case_end_datetime_single_domain():
    dt = tv.read_case_end_datetime(NML_LAFE_SHAPE)
    assert dt == datetime.datetime(2017, 8, 24, 6, 0, 0)


def test_read_case_end_datetime_missing_fields_returns_none():
    assert tv.read_case_end_datetime("&time_control\n/\n") is None


def test_read_case_start_datetime_single_domain():
    dt = tv.read_case_start_datetime(NML_1DOM)
    assert dt == datetime.datetime(2018, 8, 25, 0, 0, 0)


def test_read_case_start_datetime_missing_fields_returns_none():
    assert tv.read_case_start_datetime("&time_control\n/\n") is None


def test_auto_select_cases_filters_by_measured_run_length(tmp_path, monkeypatch):
    long_dir = tmp_path / "long"
    short_dir = tmp_path / "short"
    long_dir.mkdir(); short_dir.mkdir()
    _write_namelist(long_dir / "namelist.input", run_hours=12)
    _write_namelist(short_dir / "namelist.input", run_hours=4)

    fake_cases = {
        "longcase": dict(prep_dir=str(long_dir), enabled=True),
        "shortcase": dict(prep_dir=str(short_dir), enabled=True),
        "disabledcase": dict(prep_dir=str(long_dir), enabled=False),
    }
    monkeypatch.setattr(tv.campaign_configs, "CASES", fake_cases)
    monkeypatch.setattr(tv.campaign_configs, "CASE_ORDER",
                         ["longcase", "shortcase", "disabledcase"])
    assert tv.auto_select_cases(min_hours=12) == ["longcase"]


def test_auto_select_cases_skips_unreadable_template(monkeypatch):
    fake_cases = {"nowhere": dict(prep_dir="/definitely/not/a/real/path", enabled=True)}
    monkeypatch.setattr(tv.campaign_configs, "CASES", fake_cases)
    monkeypatch.setattr(tv.campaign_configs, "CASE_ORDER", ["nowhere"])
    assert tv.auto_select_cases(min_hours=12) == []


# ---------------------------------------------------------------------------
# plan CLI output (auto-selection + explicit --cases, capsys)
# ---------------------------------------------------------------------------
def test_cmd_plan_requires_cases_or_measurable_templates(monkeypatch):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {})
    monkeypatch.setattr(tv.campaign_configs, "CASE_ORDER", [])
    args = argparse.Namespace(cases=None, hours=12, every=3, steps=30, dry_run=True,
                               nodes=4, cores_per_node=72, hardware_id="scc-gh200",
                               k=7, seed=1, root="/tmp/does-not-matter")
    try:
        tv.cmd_plan(args)
        assert False, "expected SystemExit"
    except SystemExit:
        pass


def test_cmd_plan_prints_case_line_for_explicit_case(monkeypatch, capsys):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    args = argparse.Namespace(cases=["synth"], hours=12, every=3, steps=30, dry_run=True,
                               nodes=1, cores_per_node=72, hardware_id="scc-gh200",
                               k=3, seed=1, root="/tmp/does-not-matter")
    tv.cmd_plan(args)
    out = capsys.readouterr().out
    assert "synth" in out
    assert "restart_times=[0, 3, 6, 9, 12]" in out
    assert "unmeasured" in out   # /nonexistent prep_dir -> template unreadable, falls back to --hours


# ---------------------------------------------------------------------------
# config generation reuse (delegates to tools/scc_sweep.py, not re-derived)
# ---------------------------------------------------------------------------
def test_build_sample_configs_anchor_plus_k():
    built = tv.build_sample_configs("synth", MATTHEW_LIKE, nodes=1, cores_per_node=72,
                                     hardware_id="scc-gh200", k=5, seed=1)
    assert built["skip_reason"] is None
    assert built["configs"][0]["id"] == "ANCHOR"
    assert 1 <= len(built["configs"]) <= 6


def test_anchor_config_matches_scc_sweep_ground_truth():
    a = tv.anchor_config(MATTHEW_LIKE, nodes=1, cores_per_node=112, hardware_id="scc-gh200")
    assert (a["ranks"], a["nx"], a["ny"]) == (90, 9, 10)   # tools/test_scc_sweep.py's own anchor


# ---------------------------------------------------------------------------
# sbatch command builders (fake runner, mirrors tools/test_scc_sweep.py's pattern)
# ---------------------------------------------------------------------------
def test_build_long_sbatch_command_contents():
    cmd = tv.build_long_sbatch_command(
        "karina", nodes=4, cores_per_node=72, hours=12, every=3, time_limit="02:00:00",
        out_dir="/home/u/wtune-runs/campaign", env_source="$HOME/env.sh",
        sbatch_script="/home/u/wtune/tools/temporal_run.sbatch", site="scc")
    assert "TV_MODE=long" in cmd
    assert "CASE=karina" in cmd
    assert "HOURS=12" in cmd
    assert "EVERY=3" in cmd
    assert "-N 4" in cmd
    assert "temporal_run.sbatch" in cmd


def test_build_samples_sbatch_command_contents():
    cmd = tv.build_samples_sbatch_command(
        "karina", restart_hours=6, long_run_dir="/out/temporal_karina_long_123",
        configs_path="/out/karina/samples_configs.json", steps=30, nodes=4, cores_per_node=72,
        time_limit="00:30:00", timeout_s=900, out_dir="/home/u/wtune-runs/campaign")
    assert "TV_MODE=samples" in cmd
    assert "RESTART_HOURS=6" in cmd
    assert "LONG_RUN_DIR=/out/temporal_karina_long_123" in cmd
    assert "STEPS=30" in cmd
    assert "TIMEOUT_S=900" in cmd


def _fake_sbatch_runner(start_job_id=5000):
    calls = []
    counter = {"n": start_job_id}

    def runner(cmd_str):
        calls.append(cmd_str)
        counter["n"] += 1
        return "Submitted batch job %d\n" % counter["n"]

    return runner, calls


def test_cmd_submit_long_dry_run_builds_command_without_submitting(monkeypatch, capsys):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=12, every=3, nodes=1, cores_per_node=72, hardware_id="scc-gh200",
        k=7, seed=1, root="/tmp/x", out_dir="/tmp/out", env_source="$HOME/env.sh",
        sbatch_script="/tmp/temporal_run.sbatch", site="scc", ledger_dir="/tmp/ledger",
        dry_run=True, sbatch_runner=runner)
    rc = tv.cmd_submit_long(args)
    assert rc == 0
    assert calls == []   # dry-run never invokes the runner
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "TV_MODE=long" in out


def test_cmd_submit_samples_submits_one_job_per_restart_time(monkeypatch, tmp_path):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", long_run_dir="/out/long", hours=6, every=3, restart_times=None, steps=30,
        nodes=1, cores_per_node=72, hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path),
        out_dir="/tmp/out", env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch",
        site="scc", ledger_dir="/tmp/ledger", dry_run=False, sbatch_runner=runner)
    rc = tv.cmd_submit_samples(args)
    assert rc == 0
    assert len(calls) == 3   # restart_offsets_hours(6, 3) == [0, 3, 6]
    for cmd in calls:
        assert "TV_MODE=samples" in cmd
    assert (tmp_path / "synth" / "samples_configs.json").exists()


# ---------------------------------------------------------------------------
# submit-timeline (TIMELINE mode, decision 0053 step 2's training-data extension)
# ---------------------------------------------------------------------------
def test_ensure_timeline_configs_caps_at_default_timeline_k(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    configs_list, path = tv.ensure_timeline_configs(
        str(tmp_path), "synth", nodes=1, cores_per_node=72, hardware_id="scc-gh200",
        k=7, seed=1)
    # anchor + up to DEFAULT_TIMELINE_K(2) alternatives = at most 3, never scc_sweep's own k=7
    assert len(configs_list) <= 1 + tv.DEFAULT_TIMELINE_K
    assert os.path.basename(path) == "timeline_configs.json"


def test_ensure_timeline_configs_reuses_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    first, path = tv.ensure_timeline_configs(str(tmp_path), "synth", 1, 72, "scc-gh200", 2, 1)
    # Corrupt CASES so a second call would raise if it tried to regenerate instead of reusing.
    monkeypatch.setattr(tv.campaign_configs, "CASES", {})
    second, path2 = tv.ensure_timeline_configs(str(tmp_path), "synth", 1, 72, "scc-gh200", 2, 1)
    assert path == path2
    assert first == second


def test_ensure_timeline_configs_path_distinct_from_samples_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    tv.ensure_timeline_configs(str(tmp_path), "synth", 1, 72, "scc-gh200", 2, 1)
    assert tv.sample_configs_path(str(tmp_path), "synth") != tv.timeline_configs_path(str(tmp_path), "synth")


def test_build_timeline_sbatch_command_shape():
    cmd = tv.build_timeline_sbatch_command(
        "synth", nodes=2, cores_per_node=72, hours=12, every=3, run_minutes=30,
        configs_path="/tmp/tl.json", time_limit="01:00:00", out_dir="/tmp/out",
        aux_interval_minutes=30, aux_stream=23, frames_per_auxhist=500,
        fields=["RAINNC", "SWDOWN"])
    assert "TV_MODE=timeline" in cmd
    assert "HOURS=12" in cmd
    assert "RUN_MINUTES=30" in cmd
    assert "EVERY=3" in cmd
    assert "CONFIG_FILE=/tmp/tl.json" in cmd
    assert "AUX_INTERVAL_MIN=30" in cmd
    assert "AUX_STREAM=23" in cmd
    assert "FRAMES_PER_AUXHIST=500" in cmd
    assert "AUX_FIELDS=RAINNC,SWDOWN" in cmd
    assert "-N 2" in cmd


def test_build_timeline_sbatch_command_omits_aux_stream_when_none():
    cmd = tv.build_timeline_sbatch_command(
        "synth", nodes=1, cores_per_node=72, hours=6, every=3, run_minutes=360,
        configs_path="/tmp/tl.json", time_limit="00:30:00", out_dir="/tmp/out",
        aux_interval_minutes=30)
    assert "AUX_STREAM=" not in cmd
    assert "AUX_FIELDS=" not in cmd


def test_build_timeline_sbatch_command_run_minutes_independent_of_hours():
    # job 587/588 postmortem: RUN_MINUTES must carry the exact (possibly sub-hour, possibly
    # date-span-derived) value -- it must never be silently re-derived as hours*60.
    cmd = tv.build_timeline_sbatch_command(
        "synth", nodes=1, cores_per_node=72, hours=0, every=3, run_minutes=30,
        configs_path="/tmp/tl.json", time_limit="00:30:00", out_dir="/tmp/out",
        aux_interval_minutes=30)
    assert "HOURS=0" in cmd
    assert "RUN_MINUTES=30" in cmd


def test_cmd_submit_timeline_dry_run_builds_command_without_submitting(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=12, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=7, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir="/tmp/ledger", dry_run=True, s_per_step=4.94, sbatch_runner=runner)
    rc = tv.cmd_submit_timeline(args)
    assert rc == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "TV_MODE=timeline" in out
    assert "configs=" in out
    assert "[override]" in out


def test_cmd_submit_timeline_submits_one_job_for_all_configs(monkeypatch, tmp_path):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=6, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir="/tmp/ledger", dry_run=False, s_per_step=4.94, sbatch_runner=runner)
    rc = tv.cmd_submit_timeline(args)
    assert rc == 0
    assert len(calls) == 1   # ALL configs run sequentially inside ONE sbatch job, unlike SAMPLES
    assert "TV_MODE=timeline" in calls[0]
    assert (tmp_path / "synth" / "timeline_configs.json").exists()


# ---------------------------------------------------------------------------
# DEFECT A regression (job 587/588 postmortem): sub-hour run length must not truncate to a
# 1-step estimate while the sbatch side still runs the case's full (sub-hour) length.
# ---------------------------------------------------------------------------
def test_resolve_run_minutes_sub_hour_case_is_nonzero():
    # full_hours=0.5 (a real isc21/lafe-shaped case), cap=12 (default, does not bind) -> 30 min,
    # never int(0.5)==0 the way the old `min(int(full_hours), args.hours)` computation did.
    assert tv.resolve_run_minutes(0.5, 12) == 30


def test_resolve_run_minutes_cap_binds_when_smaller():
    assert tv.resolve_run_minutes(12.0, 3) == 180


def test_resolve_run_minutes_unmeasured_template_uses_cap():
    assert tv.resolve_run_minutes(None, 6) == 360
    assert tv.resolve_run_minutes(None, 0) == tv.DEFAULT_HOURS * 60


def test_cmd_submit_timeline_isc21_shape_run_minutes_reaches_namelist_edit(monkeypatch, tmp_path, capsys):
    # isc21 shape (job 587/588 evidence): namelist run length is a genuine 30 minutes
    # (run_hours=0, run_minutes=30), template's own start->end date span is 15h. dt=20s
    # (MATTHEW_LIKE) -> steps_est must reflect the true 30-minute run (30*60/20 = 90 steps), and
    # -- the part that matters -- the value that reaches the namelist edit (RUN_MINUTES in the
    # exported sbatch command) must be 30, NOT 0 (which would select the 15h date span via WRF's
    # own end-date fallback: HOURS=0 is exported unchanged for other consumers, but RUN_MINUTES,
    # not HOURS, is what tools/state_timeline.py's edit-namelist-timeline now uses).
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "namelist.input").write_text(NML_ISC21_SHAPE)
    fake_case = dict(MATTHEW_LIKE, prep_dir=str(case_dir))
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": fake_case})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=12, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir="/tmp/ledger", dry_run=True, s_per_step=4.94, sbatch_runner=runner)
    rc = tv.cmd_submit_timeline(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "hours=0" in out          # HOURS is still exported unchanged (kept for other
                                      # consumers) -- it is RUN_MINUTES, not HOURS, that now
                                      # drives the actual namelist edit; see the DRY-RUN line below
    assert "run_min=30" in out       # Python-side estimate sees the true length
    dry_run_line = next(ln for ln in out.splitlines() if ln.startswith("DRY-RUN:"))
    assert "RUN_MINUTES=30" in dry_run_line   # the value that actually reaches the namelist edit
    assert "HOURS=0" in dry_run_line
    # time_limit must not be sized for a 1-step run: 90 steps at s_est=4.94s/step is well over
    # the old code's ~120s (STARTUP_S) + 1*4.94s floor.
    time_str = next(ln for ln in out.splitlines() if ln.startswith("TIMELINE"))
    assert "time=00:20:00" not in time_str


def test_cmd_submit_timeline_lafe_shape_uncapped_run_minutes_reaches_namelist_edit(monkeypatch, tmp_path, capsys):
    # lafe shape (job 587/588 evidence): all four run_* keys are 0, start->end date span is 24h
    # (1440 min). With a cap that does not bind (--hours=30), the full 1440 minutes must reach the
    # namelist edit -- NOT 0 and NOT a 1-minute floor (the old resolve_run_minutes(0.0, cap) bug).
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "namelist.input").write_text(NML_LAFE_SHAPE)
    fake_case = dict(MATTHEW_LIKE, prep_dir=str(case_dir))
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": fake_case})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=30, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir="/tmp/ledger", dry_run=True, s_per_step=4.94, sbatch_runner=runner)
    rc = tv.cmd_submit_timeline(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_min=1440" in out
    dry_run_line = next(ln for ln in out.splitlines() if ln.startswith("DRY-RUN:"))
    assert "RUN_MINUTES=1440" in dry_run_line
    assert "HOURS=24" in dry_run_line


def test_cmd_submit_timeline_lafe_shape_cap_still_binds(monkeypatch, tmp_path, capsys):
    # Same lafe shape, but with the DEFAULT --hours=12 cap, which DOES bind (24h measured > 12h
    # cap): the operator's cap must still be honoured -- 720 minutes, not 1440 and not 0/1.
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "namelist.input").write_text(NML_LAFE_SHAPE)
    fake_case = dict(MATTHEW_LIKE, prep_dir=str(case_dir))
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": fake_case})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=12, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir="/tmp/ledger", dry_run=True, s_per_step=4.94, sbatch_runner=runner)
    rc = tv.cmd_submit_timeline(args)
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_min=720" in out
    dry_run_line = next(ln for ln in out.splitlines() if ln.startswith("DRY-RUN:"))
    assert "RUN_MINUTES=720" in dry_run_line
    assert "HOURS=12" in dry_run_line


# ---------------------------------------------------------------------------
# DEFECT B regression (job 587/588 postmortem): no ledger row -> refuse rather than submit on
# the unmeasured fallback_10s constant.
# ---------------------------------------------------------------------------
def test_cmd_submit_timeline_refuses_on_fallback_estimate(monkeypatch, tmp_path):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=6, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir=str(tmp_path / "no-ledger-here"), dry_run=False, s_per_step=None,
        sbatch_runner=runner)
    try:
        tv.cmd_submit_timeline(args)
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "fallback" in str(e)
        assert "tools/scc_sweep.py submit --cases synth --nodes 1 --hardware-id scc-gh200" in str(e)
    assert calls == []   # never submitted


def test_cmd_submit_timeline_dry_run_still_refuses_on_fallback_estimate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=6, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir=str(tmp_path / "no-ledger-here"), dry_run=True, s_per_step=None,
        sbatch_runner=runner)
    try:
        tv.cmd_submit_timeline(args)
        assert False, "expected SystemExit even under --dry-run"
    except SystemExit as e:
        assert "fallback" in str(e)
    out = capsys.readouterr().out
    assert "DRY-RUN" in out          # the plan is still printed before the refusal
    assert "TV_MODE=timeline" in out
    assert calls == []


def test_cmd_submit_timeline_s_per_step_override_bypasses_refusal(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(tv.campaign_configs, "CASES", {"synth": dict(MATTHEW_LIKE)})
    runner, calls = _fake_sbatch_runner()
    args = argparse.Namespace(
        case="synth", hours=6, every=3, aux_interval_minutes=30, aux_stream=None,
        frames_per_auxhist=2000, fields=None, nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", k=2, seed=1, root=str(tmp_path), out_dir="/tmp/out",
        env_source="$HOME/env.sh", sbatch_script="/tmp/temporal_run.sbatch", site="scc",
        ledger_dir=str(tmp_path / "no-ledger-here"), dry_run=False, s_per_step=4.94,
        sbatch_runner=runner)
    rc = tv.cmd_submit_timeline(args)
    assert rc == 0
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "s_est=4.940s/step[override]" in out


def test_build_arg_parser_submit_timeline_defaults_k_to_timeline_k():
    args = tv.build_arg_parser().parse_args(["submit-timeline", "--case", "synth"])
    assert args.k == tv.DEFAULT_TIMELINE_K
    assert args.aux_interval_minutes == 30


# ---------------------------------------------------------------------------
# analyze end-to-end
# ---------------------------------------------------------------------------
def test_cmd_analyze_end_to_end(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    root_dir = tmp_path / "root"
    raw_dir.mkdir()

    samples_t0 = raw_dir / "temporal_karina_samples_h0_1.csv"
    samples_t3 = raw_dir / "temporal_karina_samples_h3_2.csv"
    with open(samples_t0, "w") as f:
        f.write("restart_hours,config_id,ranks,omp,nx,ny,numtiles,dist,rc,ok,steps\n")
        f.write("0,ANCHOR,72,1,8,9,1,block,0,1,99;1.0;1.0\n")
        f.write("0,G0001,72,1,9,8,1,cyclic,0,1,99;1.5;1.5\n")
    with open(samples_t3, "w") as f:
        f.write("restart_hours,config_id,ranks,omp,nx,ny,numtiles,dist,rc,ok,steps\n")
        f.write("3,ANCHOR,72,1,8,9,1,block,0,1,99;1.0;1.0\n")
        f.write("3,G0001,72,1,9,8,1,cyclic,0,1,99;1.02;1.02\n")

    long_csv = raw_dir / "temporal_karina_long_9_timing.csv"
    with open(long_csv, "w") as f:
        f.write("step_index,s_per_step\n")
        for i, v in enumerate([99.0, 2.0, 2.0, 2.0, 2.0], start=1):
            f.write("%d,%s\n" % (i, v))

    args = argparse.Namespace(
        case="karina", root=str(root_dir), raw_dir=str(raw_dir), samples_csv=None,
        long_timing_csv=None, dt=3600.0, tau_threshold=0.7, pct_threshold=0.02)
    rc = tv.cmd_analyze(args)
    assert rc == 0

    out = capsys.readouterr().out
    assert "temporal validation: karina" in out
    assert (root_dir / "karina" / "rankings.csv").exists()
    assert (root_dir / "karina" / "summary.csv").exists()
    assert (root_dir / "karina" / "drift.csv").exists()

    with open(root_dir / "karina" / "summary.csv") as f:
        srows = list(csv.DictReader(f))
    row3 = next(r for r in srows if r["restart_hours"] == "3")
    assert row3["best_config_t0"] == "ANCHOR"
    assert row3["within_pct_threshold"] == "True"

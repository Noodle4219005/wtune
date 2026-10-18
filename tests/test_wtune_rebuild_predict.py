import datetime as dt
import types

import pytest

from wtune import rebuild_predict as wrp


def _fake_plan(case_dir, forcing, spinup_min, max_points, sources=("rebuild",)):
    t0 = dt.datetime(2021, 7, 19, 18)
    pts = [types.SimpleNamespace(at=t0 + dt.timedelta(hours=h, minutes=spinup_min), reason="auto",
                                 source="rebuild") for h in (0, 7)]
    return None, t0, t0 + dt.timedelta(hours=30), 36.0, 17, pts, []


def test_dry_run_prints_plan_and_defaults(monkeypatch, capsys, tmp_path):
    seen = {}

    def plan(case_dir, forcing, spinup_min, max_points, sources=("rebuild",)):
        seen.update(forcing=forcing, spinup=spinup_min, max_points=max_points)
        return _fake_plan(case_dir, forcing, spinup_min, max_points)

    monkeypatch.setattr(wrp, "plan", plan)
    assert wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert seen == {"forcing": "era5_arco", "spinup": 10, "max_points": None}
    assert "measure 30 steps" in out                          # short window: max(30, 17+10)
    assert "288 ranks, 16x18" in out and "2 parallel jobs x 4 nodes" in out
    wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18", "--forcing", "auto",
              "--max-points", "3", "--dry-run"])
    assert seen == {"forcing": "auto", "spinup": 60, "max_points": 3}


def test_refuses_output_inside_case_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(wrp, "plan", _fake_plan)
    monkeypatch.setenv("WPS_DIR", "/x")
    monkeypatch.setenv("WRF_TREE", "/x")
    assert wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18",
                     "--out", str(tmp_path / "inside")]) == 2
    assert not (tmp_path / "inside").exists()


def test_rebuild_points_fetch_into_each_point_plan_dir(monkeypatch, tmp_path):
    import datetime as _dt
    import wtune.tune
    seen, seen_timeout = [], []
    monkeypatch.setenv("WPS_DIR", "/x")
    monkeypatch.setenv("WRF_TREE", "/x")
    monkeypatch.setattr(wtune.tune.probe_plan, "plan_rebuild_segments",
                        lambda case_dir, pts, cfg, pdir, **k: {"windows": [], "pdir": pdir})
    monkeypatch.setattr(wtune.tune.coldstart, "resolve_and_fetch_forcing",
                        lambda plan, pdir, cache, grib_dir=None: (seen.append((pdir, cache)) or (0, {})))
    monkeypatch.setattr(wtune.tune.budget, "rebuild_segment_timeout_min", lambda *a, **k: 45)
    monkeypatch.setattr(wtune.tune.coldstart, "stage_plan",
                        lambda pdir, site="scc", per_window_min=None: (seen_timeout.append(per_window_min)
                                                                       or pdir + "/run.sbatch"))
    monkeypatch.setattr(wtune.tune.sweep, "submit_job", lambda cmd, sbatch_runner=None: "1")
    monkeypatch.setattr(wtune.tune.jobs, "wait_for_jobs", lambda ctx, ids, label: {j: ("COMPLETED", "") for j in ids})
    monkeypatch.setattr(wtune.tune.predict, "read_wrfinput_sample_points", lambda d: ([], None))
    monkeypatch.setattr(wtune.tune.predict, "load_calibration", lambda p: {})
    monkeypatch.setattr(wtune.tune.coldstart, "score_windows_from_rsl", lambda plan, sp, cal: ([], None, []))
    ctx = types.SimpleNamespace(sbatch_runner=None, owned_jobs=[], state_file=None)
    pts = [types.SimpleNamespace(at=_dt.datetime(2021, 7, 20, h)) for h in (1, 8)]
    wtune.tune.run_rebuild_points(ctx, pts, dict(ranks=288, omp=1, nx=16, ny=18), 4, str(tmp_path),
                                  str(tmp_path / "out"), _dt.datetime(2021, 7, 19, 18))
    assert sorted(seen) == [(str(tmp_path / "out" / ("pt_%d" % i)), str(tmp_path / "out" / ("pt_%d" % i) / "grib"))
                            for i in (0, 1)]
    assert seen_timeout == [45, 45]


def test_segment_timeout_scales_with_case_cost_and_has_a_floor(monkeypatch):
    import wtune.tune
    wp = wtune.tune.predict
    monkeypatch.setattr(wp, "parse_namelist", lambda p: {})
    monkeypatch.setattr(wp, "get_d01_steps", lambda nl: (432, 10800, 25.0, ""))
    monkeypatch.setattr(wp, "steps_per_radiation_call", lambda radt, dt: 29)
    monkeypatch.setattr(wp, "g", lambda nl, grp, key, default=None: 10 if key == "radt" else 2)
    monkeypatch.setattr(wp, "load_calibration", lambda p: {"fallback_s_per_cell_step_ns": 4437.0})
    monkeypatch.setattr(wp, "nested_equivalent_cells", lambda nl, md: 5.0e8)   # ~7.7 s/step on 288
    t = wtune.tune.rebuild_segment_timeout_min("/case", 20, 288)
    # 48 spin-up + 58 window steps ~ 106 x 7.7 x 1.5 + 120 s ~ 22 min, x3 -> ~66 min
    assert 55 <= t <= 80
    monkeypatch.setattr(wp, "nested_equivalent_cells", lambda nl, md: 1.0e6)
    assert wtune.tune.rebuild_segment_timeout_min("/case", 20, 288) == 30


def test_measure_steps_option():
    from wtune.rebuild_predict import _measure_steps
    assert _measure_steps(None, 17) is None
    assert _measure_steps("short", 17) == 30   # 30-step floor
    assert _measure_steps("short", 29) == 39
    assert _measure_steps("short", 6) == 30
    assert _measure_steps("20", 17) == 20
    with pytest.raises(SystemExit):
        _measure_steps("17", 17)


# --- I/O estimate and --time recommendation -----------------------------------------------

_NL_IO = """\
 &time_control
 run_hours = 24,
 start_year = 2019, start_month = 11, start_day = 26, start_hour = 0,
 history_interval = 60, 180,
 io_form_history = 2,
 restart_interval = 720,
 /
 &domains
 time_step = 60, max_dom = 2,
 /
"""


def _nl(tmp_path, text):
    from wtune import predict as wp
    p = tmp_path / "namelist.input"
    p.write_text(text)
    return wp.parse_namelist(str(p))


def test_user_frames_history_on_off_begin_and_restart(tmp_path):
    f = wrp.user_io_frames(_nl(tmp_path, _NL_IO), 86400)
    assert f["history"] == [25, 9] and f["restart"] == 2
    f = wrp.user_io_frames(_nl(tmp_path, _NL_IO.replace("= 60, 180,", "= 60, 0,")), 86400)
    assert f["history"] == [25, 0]                                   # interval 0 -> off
    f = wrp.user_io_frames(_nl(tmp_path, _NL_IO.replace(" io_form_history = 2,",
                                                        " history_begin_h = 6, 6,")), 86400)
    assert f["history"] == [19, 7]                                   # frames from 06:00 on
    f = wrp.user_io_frames(_nl(tmp_path, _NL_IO.replace("io_form_history = 2", "io_form_history = 0")
                               .replace("restart_interval = 720", "restart_interval = 7200")), 86400)
    assert f["history"] == [0, 0] and f["restart"] == 0
    f = wrp.user_io_frames(_nl(tmp_path, _NL_IO.replace(" io_form_history", " restart = .true.,\n io_form_history")), 86400)
    assert f["history"] == [24, 8] and f["notes"]                    # resumed: no t=0 frame


def test_io_estimate_multiplies_medians_by_frames():
    frames = {"history": [25, 0], "restart": 2, "notes": []}
    writes = {"history": {1: [10.0, 14.0, 12.0], 2: [3.0]}, "restart": {1: [100.0], 2: [20.0]}}
    io = wrp.io_estimate(frames, writes, dt.datetime(2019, 11, 26, 3, 40))
    assert io["ok"] and io["io_s"] == pytest.approx(25 * 12.0 + 2 * 120.0)
    assert io["history"][0]["seconds"] == pytest.approx(300.0) and "seconds" not in io["history"][1]
    assert io["source"] == "measured (1 frame at 2019-11-26_03:40:00)"
    bad = wrp.io_estimate(frames, {"history": {}, "restart": {}}, None)
    assert not bad["ok"] and bad["io_s"] is None and bad["missing"] == ["history d01", "restart"]
    assert wrp.io_estimate({"history": [0, 0], "restart": 0}, None, None)["io_s"] == 0.0


def test_recommendation_arithmetic_and_rounding():
    r = wrp.recommend_time(3600.0, 600.0)        # 3600 + 1200 + min(360, 300) = 5100 (x60)
    assert r["time_s"] == 5100 and r["time"] == "01:25:00" and r["covers_io"]
    assert wrp.recommend_time(3601.0, 600.0)["time_s"] == 5160   # < 2 h: rounded to 1 min
    assert wrp.recommend_time(100000.0, 0.0)["time"] == "27:55:00"  # >= 2 h: 5 min
    f = wrp.recommend_time(3601.0, None)          # probe failed: range_high + margin, rounded up
    assert f["time_s"] == 3960 and not f["covers_io"]
    # dorian2019 T1 (scc): range_high 1502.5 s, io 299.1 s -> margin 150 s -> 38:00, not 45:00
    d = wrp.recommend_time(1502.52, 299.14)
    assert d["time_s"] == 2280 and d["margin_s"] == pytest.approx(150.252)
    assert wrp.recommend_time(500.0, 0.0)["margin_s"] == 120.0     # floor


def test_probe_marks_only_the_last_point(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(wrp, "plan", _fake_plan)
    wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18", "--dry-run"])
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("  point")]
    assert ["I/O probe" in l for l in lines] == [False, True]
    wrp.main([str(tmp_path), "--nodes", "4", "--nx", "16", "--ny", "18", "--dry-run",
              "--no-io-probe"])
    assert "I/O probe" not in capsys.readouterr().out


def test_read_probe_writes_from_the_probe_plan_dir(tmp_path):
    import json
    plan = {"windows": [{"t": "2019-11-26_03:00:00"}], "measure_steps": 20, "spinup_min": 20,
            "dt_s": 60.0, "io_probe": {"window": 0, "history_prefix": "wrfout_d"}}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    probe, writes, at = wrp.read_probe_writes(str(tmp_path))
    assert writes is None and at == dt.datetime(2019, 11, 26, 3, 40)
    (tmp_path / "win_0" / "wrf").mkdir(parents=True)
    (tmp_path / "win_0" / "wrf" / "rsl.error.0000").write_text(
        "Timing for Writing wrfout_d01_2019-11-26_03:40:00 for domain        1:   13.90594 elapsed seconds\n")
    assert wrp.read_probe_writes(str(tmp_path))[1]["history"] == {1: [13.90594]}


def test_ctx_charges_running_time_only(tmp_path):
    ctx = wrp._ctx(str(tmp_path), 8, str(tmp_path / "o"), max_wait_hours=12)
    assert ctx.budget_mode == "running" and ctx.max_wall_s == 12 * 3600.0
    assert ctx.budget_s == 4 * 3600.0


def test_points_by_flag_defaults_to_time_and_plan_stays_time_based(monkeypatch, tmp_path, capsys):
    from wtune import rebuild_predict as rp
    seen = {}

    def fake_plan(case_dir, forcing, spinup, max_points, **kw):
        seen.update(kw)
        raise SystemExit(0)
    monkeypatch.setattr(rp, "plan", fake_plan)
    monkeypatch.setattr("wtune.extensions.refusal_predict_rebuild", lambda d: None)
    with pytest.raises(SystemExit):
        rp.main([str(tmp_path), "--nodes", "1", "--nx", "1", "--ny", "1", "--dry-run"])
    assert seen == {}          # option off: plan() is called exactly as before

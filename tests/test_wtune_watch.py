"""Tests for wtune/watch.py -- synthetic namelist.input + rsl.error/out.0000 in tmp dirs.

Synthetic run: 3 domains (ratio 1:3:3), 24 h at dt=60 s -> 1440 d01 steps, radt=10 min
(stepra 10), history every 60 min for every domain. d01 compute = 1.0 s, x4 on a radiation step
(WRF schedule: step 1 and every stepra-th) -> mean 1.3 s/step. History frame at sim minute T is
written at the start of the step beginning at T ('Timing for Writing' lines before that step's
d01 line; the end-of-run frame after the last d01 line), 20 s per frame split d01/d02/d03 =
10/5/5 s -> writes are ~20 % of wall. Stationary, so a whole-period mean is exact.
"""
import datetime as dt
import json
import os
import sys
import time

import pytest

from wtune import watch as ww

T0 = dt.datetime(2021, 7, 19, 0)
DT = 60
N = 1440
STEPRA = 10
HIST = 60
WRITE = 20.0
SPLIT = (0.5, 0.25, 0.25)


def _namelist(end_hours=(24, 24, 24), run_hours=24, run_minutes=0, dt_s=DT, radt=10, extra=""):
    ends = [T0 + dt.timedelta(hours=h) for h in end_hours]

    def row(key, vals):
        return " %s = %s,\n" % (key, ", ".join(str(v) for v in vals))

    return ("&time_control\n" + row("run_hours", [run_hours]) + row("run_minutes", [run_minutes])
            + row("start_year", [T0.year] * 3) + row("start_month", [T0.month] * 3)
            + row("start_day", [T0.day] * 3) + row("start_hour", [T0.hour] * 3)
            + row("end_year", [e.year for e in ends]) + row("end_month", [e.month for e in ends])
            + row("end_day", [e.day for e in ends]) + row("end_hour", [e.hour for e in ends])
            + row("history_interval", [HIST] * 3) + extra + "/\n"
            + "&domains\n time_step = %d,\n max_dom = 3,\n" % dt_s
            + row("e_we", [100, 100, 100]) + row("e_sn", [100, 100, 100])
            + row("e_vert", [40, 40, 40]) + row("parent_id", [0, 1, 2])
            + row("parent_time_step_ratio", [1, 3, 3]) + "/\n"
            + "&physics\n" + row("radt", [radt] * 3) + "/\n")


def _s_normal(i, ramp=0.0, n=N):
    return 1.0 + ramp * i / n


def _cost(i, ramp=0.0, r_rad=4.0, stepra=STEPRA, n=N):
    """d01 step i COMPUTE: s(t_i) x r_rad on a WRF-schedule radiation step."""
    s = _s_normal(i, ramp, n)
    return s * (r_rad if (i == 1 or i % stepra == 0) else 1.0)


def _write_lines(t, write, doms=(1, 2, 3)):
    ts = t.strftime("%Y-%m-%d_%H:%M:%S")
    return ["Timing for Writing wrfout_d%02d_%s for domain        %d: %10.5f elapsed seconds"
            % (d, ts, d, write * SPLIT[d - 1]) for d in doms]


def _rsl(n_d01=N, nest_stop=None, head=("starting wrf task 0",), tail=(), cost=_cost, dt_s=DT,
         write=WRITE, hist=HIST, n_total=N):
    """d01 steps 1..n_d01; nest lines (d02 3 per d01 step, d03 3 per d02 step) printed before the
    d01 line of the same step, and omitted after d01 step `nest_stop`. A history frame every
    `hist` steps is written at the start of the following step (its d01 line = compute + write);
    the end-of-run frame follows the last d01 line when the log is complete."""
    out = list(head)
    for i in range(1, n_d01 + 1):
        base = T0 + dt.timedelta(seconds=(i - 1) * dt_s)
        wr = write if write and (i - 1) % hist == 0 else 0.0
        if wr:
            out += _write_lines(base, wr, (1,) if nest_stop is not None and i > nest_stop
                                else (1, 2, 3))
            wr = wr if nest_stop is None or i <= nest_stop else wr * SPLIT[0]
        if nest_stop is None or i <= nest_stop:
            for j in range(1, 4):
                for k in range(1, 4):
                    t3 = base + dt.timedelta(seconds=((j - 1) * 3 + k) * dt_s / 9)
                    out.append("Timing for main: time %s on domain   3:    0.01000 elapsed seconds"
                               % t3.strftime("%Y-%m-%d_%H:%M:%S"))
                t2 = base + dt.timedelta(seconds=j * dt_s / 3)
                out.append("Timing for main: time %s on domain   2:    0.05000 elapsed seconds"
                           % t2.strftime("%Y-%m-%d_%H:%M:%S"))
        t1 = T0 + dt.timedelta(seconds=i * dt_s)
        out.append("Timing for main: time %s on domain   1: %10.5f elapsed seconds"
                   % (t1.strftime("%Y-%m-%d_%H:%M:%S"), cost(i) + wr))
    if write and n_d01 == n_total and n_total % hist == 0:
        out += _write_lines(T0 + dt.timedelta(seconds=n_total * dt_s), write)
    return "\n".join(out + list(tail)) + "\n"


def _step_total(i, cost=_cost, write=WRITE, hist=HIST):
    return cost(i) + (write if (i - 1) % hist == 0 else 0.0)


def _rundir(tmp_path, rsl=None, rsl_out=None, **nml):
    (tmp_path / "namelist.input").write_text(_namelist(**nml))
    if rsl is not None:
        (tmp_path / "rsl.error.0000").write_text(rsl)
    if rsl_out is not None:
        (tmp_path / "rsl.out.0000").write_text(rsl_out)
    return str(tmp_path)


def _codes(snap):
    return [w["code"] for w in snap["warnings"]]


SUCCESS = ("d01 2021-07-20_00:00:00 wrf: SUCCESS COMPLETE WRF",)


def test_replay_estimate_within_2pct_and_rate_is_whole_period_mean(tmp_path, capsys):
    d = _rundir(tmp_path, rsl=_rsl(tail=SUCCESS))
    snap = ww.snapshot(d, replay_fraction=0.25)
    assert snap["stepra"] == STEPRA and snap["total_steps"] == N
    assert snap["completed_steps"] == 360 and snap["periods"] == [STEPRA]
    assert abs(snap["replay_error_pct"]) < 2.0
    w = snap["rate_window"]
    assert w % STEPRA == 0 and w >= max(2 * STEPRA, 50, 90)
    costs = [_cost(i) for i in range(1, 361)]
    assert snap["rate_s_per_step"] == pytest.approx(sum(costs[-w:]) / w)  # mean, not median
    assert snap["rate_s_per_step"] == pytest.approx(1.3)
    assert snap["rate_s_per_step"] > ww.statistics.median(costs[-w:])
    assert snap["n_write_absorbing_steps"] == 6  # steps 1, 61, ..., 301
    actual = sum(_step_total(i) for i in range(361, N + 1)) + WRITE  # + end-of-run frame
    assert snap["actual_remaining_s"] == pytest.approx(actual)
    assert snap["remaining_writes_s"] == pytest.approx(19 * WRITE)  # minutes 360, 420, ..., 1440
    assert snap["remaining_s"] == pytest.approx(snap["remaining_compute_s"]
                                                + snap["remaining_writes_s"])
    assert not snap["actual_partial"] and _codes(snap) == []
    assert snap["replay_error_pct"] is not None and snap["replay_error_pct"] == snap["error_pct"]
    assert snap["est_for_actual_s"] == pytest.approx(snap["remaining_s"])
    assert ww.exit_code(snap) == 0

    assert ww.main([d, "--replay-fraction", "0.1,0.25,0.5"]) == 0
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 3 and all(r.startswith("replay F=") and "err " in r for r in rows)
    for f in (0.1, 0.5):
        assert abs(ww.snapshot(d, replay_fraction=f)["replay_error_pct"]) < 2.0


def test_unobserved_stream_takes_the_prediction_io_block_else_zero_with_note(tmp_path):
    # restart every 12 h: nothing written by step 500 (first restart at minute 720)
    d = _rundir(tmp_path, rsl=_rsl(n_d01=500), extra=" restart_interval = 720,\n")
    snap = ww.snapshot(d, now=time.time())
    rst = [r for r in snap["writes"] if r["stream"] == "restart"]
    assert len(rst) == 3 and all(r["remaining_frames"] == 2 and r["seconds"] == 0 for r in rst)
    assert any("restart d01" in n for n in snap["write_notes"])
    pj = _pred_json(tmp_path / "p.json", io={"history": [{"domain": 1, "median_s": 99.0}],
                                            "restart": {"per_write_s": 30.0}})
    snap = ww.snapshot(d, now=time.time(), prediction=ww.load_prediction(pj, d))
    rows = {(r["stream"], r["dom"]): r for r in snap["writes"]}
    assert rows[("restart", 1)]["source"] == "prediction"
    assert rows[("restart", 1)]["seconds"] == pytest.approx(60.0)
    assert rows[("restart", 2)]["seconds"] == 0.0
    assert rows[("history", 1)]["source"] == "observed"  # observed wins over the prediction
    assert rows[("history", 1)]["per_frame_s"] == pytest.approx(10.0)


def test_init_spike_excluded_from_short_run_replay(tmp_path):
    # 30-min run at dt=15 s -> 120 d01 steps, stepra 120 (radt 30): a 25 % replay has 30 lines,
    # and the first 3 are 8x slower. The lead-in rule keeps them out of the rate.
    def cost(i):
        return (8.0 if i <= 3 else 0.0) + _cost(i, stepra=120, n=120)
    d = _rundir(tmp_path, rsl=_rsl(n_d01=120, cost=cost, dt_s=15, tail=SUCCESS, write=0.0),
                run_hours=0, run_minutes=30, dt_s=15, radt=30)
    snap = ww.snapshot(d, replay_fraction=0.25)
    assert snap["total_steps"] == 120 and snap["completed_steps"] == 30
    assert snap["rate_lead_dropped"] == 3 and snap["rate_s_per_step"] == pytest.approx(1.0)
    # actual 90 steps: 89 x 1.0 + one radiation step (120) x 4.0 = 93 -> est 90 = -3.2 %
    assert snap["replay_error_pct"] == pytest.approx(100 * (90 - 93) / 93)
    assert snap["replay_error_pct_rate"] == snap["replay_error_pct"]
    assert snap["replay_error_pct_pred"] is None and snap["estimate"] == "rate"


def test_truncated_rsl_error_loses_to_complete_rsl_out(tmp_path):
    full = _rsl(n_d01=500)
    trunc = full[: len(full) // 3]
    d = _rundir(tmp_path, rsl=trunc, rsl_out=full)
    snap = ww.snapshot(d, now=time.time())
    assert snap["rsl_file"] == "rsl.out.0000" and snap["n_d01_steps"] == 500
    # tie -> rsl.error.0000
    (tmp_path / "rsl.error.0000").write_text(full)
    assert ww.snapshot(d, now=time.time())["rsl_file"] == "rsl.error.0000"


def test_hang_with_injected_clock(tmp_path):
    d = _rundir(tmp_path, rsl=_rsl(n_d01=500))
    t_file = time.time() - 50
    os.utime(tmp_path / "rsl.error.0000", (t_file, t_file))
    fresh = ww.snapshot(d, now=t_file + 10)
    assert fresh["status"] == "running" and "HANG" not in _codes(fresh)
    assert fresh["eta"] and fresh["remaining_steps"] == N - 500
    assert ww.exit_code(fresh) == 0
    old = ww.snapshot(d, now=t_file + 301)  # median step 1.0 s -> limit = hang_min 300 s
    assert "HANG" in _codes(old) and ww.exit_code(old) == 2
    assert "HANG" not in _codes(ww.snapshot(d, now=t_file + 301, hang_min=400))
    assert "HANG" in _codes(ww.snapshot(d, now=t_file + 301, hang_min=10, hang_factor=200))
    assert "HANG" not in _codes(ww.snapshot(d, now=t_file + 150, hang_min=10, hang_factor=200))


def test_hang_init_quotes_problem_opening(tmp_path):
    head = ("starting wrf task 0", "med_initialdata_input: calling input_input",
            " W A R N I N G : Problem opening storm_diags_d01.txt")
    d = _rundir(tmp_path, rsl="\n".join(head) + "\n")
    now = time.time()
    early = ww.snapshot(d, now=now + 10)
    assert early["status"] == "init" and "HANG_INIT" not in _codes(early)
    late = ww.snapshot(d, now=now + 3600)
    hi = [w for w in late["warnings"] if w["code"] == "HANG_INIT"]
    assert hi and "storm_diags_d01.txt" in hi[0]["text"] and "hint" in hi[0]["text"]
    assert ww.exit_code(late) == 2
    assert "HANG_INIT" not in _codes(ww.snapshot(d, now=now + 3600, init_limit=7200))
    assert "HANG_INIT" in _codes(ww.snapshot(d, replay_fraction=0.5))


def test_nest_end_early_from_namelist_alone(tmp_path):
    d = _rundir(tmp_path, end_hours=(24, 12, 12))
    snap = ww.snapshot(d)
    assert snap["status"] == "no_rsl"
    early = [w for w in snap["warnings"] if w["code"] == "NEST_END_EARLY"]
    assert len(early) == 2 and "d02" in early[0]["text"] and ww.exit_code(snap) == 2
    (tmp_path / "ok").mkdir()
    ok = ww.snapshot(_rundir(tmp_path / "ok", end_hours=(24, 24, 24)))
    assert "NEST_END_EARLY" not in _codes(ok)


def test_nest_stopped_while_d01_continues(tmp_path):
    d = _rundir(tmp_path, rsl=_rsl(n_d01=500, nest_stop=300))
    snap = ww.snapshot(d, now=time.time())
    stopped = [w for w in snap["warnings"] if w["code"] == "NEST_STOPPED"]
    assert {w["text"][:3] for w in stopped} == {"d02", "d03"} and ww.exit_code(snap) == 2
    (tmp_path / "healthy").mkdir()
    healthy = ww.snapshot(_rundir(tmp_path / "healthy", rsl=_rsl(n_d01=500)), now=time.time())
    assert "NEST_STOPPED" not in _codes(healthy)
    doms = {x["id"]: x for x in healthy["domains"]}
    assert doms[2]["last_time"] == doms[1]["last_time"] == "2021-07-19_08:20:00"
    # replay before the nests stopped: clean; after: flagged
    assert "NEST_STOPPED" not in _codes(ww.snapshot(d, replay_fraction=0.5))
    assert "NEST_STOPPED" in _codes(ww.snapshot(d, replay_fraction=0.9))


def test_failed_and_done_exit_codes(tmp_path, capsys):
    d = _rundir(tmp_path, rsl=_rsl(n_d01=200, tail=("-------------- FATAL CALLED ---------------",)))
    snap = ww.snapshot(d, now=time.time())
    assert snap["status"] == "failed" and "FAILED" in _codes(snap) and ww.exit_code(snap) == 3
    assert ww.main([d, "--once"]) == 3

    nan = _rsl(n_d01=5) + "Timing for main: time 2021-07-19_00:06:00 on domain   1:        NaN elapsed seconds\n"
    snap = ww.snapshot(_rundir(tmp_path, rsl=nan), now=time.time())
    assert snap["status"] == "failed" and ww.exit_code(snap) == 3

    d = _rundir(tmp_path, rsl=_rsl(tail=SUCCESS))
    snap = ww.snapshot(d, now=time.time() + 1e6)  # old log, but done -> no HANG
    assert snap["status"] == "done" and snap["fraction"] == 1.0 and snap["remaining_s"] == 0.0
    assert _codes(snap) == [] and ww.exit_code(snap) == 0
    capsys.readouterr()
    assert ww.main([d, "--json"]) == 0  # polling loop stops by itself on DONE
    assert json.loads(capsys.readouterr().out)["status"] == "done"


def test_cfl_informational(tmp_path):
    cfl = ("d01 2021-07-19_00:10:00   3 points exceeded cfl=2 in domain d01 at time",)
    d = _rundir(tmp_path, rsl=_rsl(n_d01=100, tail=cfl))
    snap = ww.snapshot(d, now=time.time())
    assert snap["cfl_count"] == 1 and "CFL" in _codes(snap) and ww.exit_code(snap) == 0
    assert "CFL" not in _codes(ww.snapshot(d, now=time.time(), prev_cfl=1))
    assert "CFL" in _codes(ww.snapshot(d, now=time.time(), prev_cfl=0))


def test_cli_registration(tmp_path, monkeypatch, capsys):
    import wtune.cli
    monkeypatch.setattr(sys, "argv", ["wtune", "watch", "--help"])
    with pytest.raises(SystemExit) as e:
        wtune.cli.main()
    assert e.value.code == 0 and "watch" in capsys.readouterr().out
    d = _rundir(tmp_path, rsl=_rsl(n_d01=50))
    monkeypatch.setattr(sys, "argv", ["wtune", "watch", d, "--help"])
    with pytest.raises(SystemExit) as e:
        wtune.cli.main()
    assert e.value.code == 0 and "--replay-fraction" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["wtune", "watch", d, "--once"])
    with pytest.raises(SystemExit) as e:
        wtune.cli.main()
    assert e.value.code == 0 and "RUNNING" in capsys.readouterr().out


def _pred_json(path, scale=0.8, ramp=0.3, r_rad=4.0, hours=(0, 6, 12, 18, 24), io=None):
    """A pred_rebuild.json whose s_normal anchors carry the run's ramp, at `scale` x the true
    level (the level is what k corrects)."""
    pts = [dict(at=(T0 + dt.timedelta(hours=h)).isoformat(), reason="auto", source="rebuild",
                s_normal=scale * _s_normal(h * 3600 / DT, ramp), n_rad=2, r=r_rad)
           for h in hours]
    rec = {"result": dict(compute_total_s=1.0, init_s=30.0, total_s=31.0, steps_n=N,
                          r_night=r_rad, r_day=r_rad, points=pts), "io": io}
    path.write_text(json.dumps(rec))
    return str(path)


def test_cli_prediction_and_multi_fraction_json_list(tmp_path, capsys):
    d = _rundir(tmp_path, rsl=_rsl(tail=SUCCESS))
    pj = _pred_json(tmp_path / "pred_rebuild.json", ramp=0.0)
    assert ww.main([d, "--replay-fraction", "0.1,0.25,0.5", "--json", "--prediction", pj]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert isinstance(rows, list) and [r["replay_fraction"] for r in rows] == [0.1, 0.25, 0.5]
    for r in rows:
        assert r["estimate"] == "prediction"
        assert r["replay_error_pct"] is not None and r["replay_error_pct_rate"] is not None
    assert ww.main([d, "--replay-fraction", "0.25", "--json"]) == 0
    assert isinstance(json.loads(capsys.readouterr().out), dict)  # single F -> one object
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"result": {"points": []}}))
    with pytest.raises(SystemExit):
        ww.main([d, "--once", "--prediction", str(bad)])

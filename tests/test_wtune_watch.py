"""Tests for wtune_watch.py -- synthetic namelist.input + rsl.error/out.0000 in tmp dirs.

Synthetic run: 3 domains (ratio 1:3:3), 24 h at dt=60 s -> 1440 d01 steps, radt=10 min
(stepra 10), history every 60 min (every 60 d01 steps). d01 step cost = 1.0 s, +3.0 s on a
radiation step, +20.0 s on a history-write step: stationary, so a whole-period mean is exact.
"""
import datetime as dt
import os
import sys
import time

import pytest

import wtune_watch as ww

T0 = dt.datetime(2021, 7, 19, 0)
DT = 60
N = 1440
STEPRA = 10
HIST = 60


def _namelist(end_hours=(24, 24, 24), run_hours=24):
    ends = [T0 + dt.timedelta(hours=h) for h in end_hours]

    def row(key, vals):
        return " %s = %s,\n" % (key, ", ".join(str(v) for v in vals))

    return ("&time_control\n" + row("run_hours", [run_hours])
            + row("start_year", [T0.year] * 3) + row("start_month", [T0.month] * 3)
            + row("start_day", [T0.day] * 3) + row("start_hour", [T0.hour] * 3)
            + row("end_year", [e.year for e in ends]) + row("end_month", [e.month for e in ends])
            + row("end_day", [e.day for e in ends]) + row("end_hour", [e.hour for e in ends])
            + row("history_interval", [HIST] * 3) + "/\n"
            + "&domains\n time_step = %d,\n max_dom = 3,\n" % DT
            + row("e_we", [100, 100, 100]) + row("e_sn", [100, 100, 100])
            + row("e_vert", [40, 40, 40]) + row("parent_id", [0, 1, 2])
            + row("parent_time_step_ratio", [1, 3, 3]) + "/\n"
            + "&physics\n" + row("radt", [10, 10, 10]) + "/\n")


def _cost(i):
    return 1.0 + (3.0 if i % STEPRA == 1 else 0.0) + (20.0 if i % HIST == 0 else 0.0)


def _rsl(n_d01=N, nest_stop=None, head=("starting wrf task 0",), tail=()):
    """d01 steps 1..n_d01; nest lines (d02 3 per d01 step, d03 3 per d02 step) printed before the
    d01 line of the same step, and omitted after d01 step `nest_stop`."""
    out = list(head)
    for i in range(1, n_d01 + 1):
        if nest_stop is None or i <= nest_stop:
            base = T0 + dt.timedelta(seconds=(i - 1) * DT)
            for j in range(1, 4):
                for k in range(1, 4):
                    t3 = base + dt.timedelta(seconds=(j - 1) * 20 + k * 20 / 3)
                    out.append("Timing for main: time %s on domain   3:    0.01000 elapsed seconds"
                               % t3.strftime("%Y-%m-%d_%H:%M:%S"))
                t2 = base + dt.timedelta(seconds=j * 20)
                out.append("Timing for main: time %s on domain   2:    0.05000 elapsed seconds"
                           % t2.strftime("%Y-%m-%d_%H:%M:%S"))
        t1 = T0 + dt.timedelta(seconds=i * DT)
        out.append("Timing for main: time %s on domain   1: %10.5f elapsed seconds"
                   % (t1.strftime("%Y-%m-%d_%H:%M:%S"), _cost(i)))
    return "\n".join(out + list(tail)) + "\n"


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
    assert snap["completed_steps"] == 360 and snap["periods"] == [STEPRA, HIST]
    assert abs(snap["error_pct"]) < 2.0
    w = snap["rate_window"]
    assert w % HIST == 0 and w >= max(2 * STEPRA, 50, 90)
    costs = [_cost(i) for i in range(1, 361)]
    assert snap["rate_s_per_step"] == pytest.approx(sum(costs[-w:]) / w)  # mean, not median
    assert snap["rate_s_per_step"] > ww.statistics.median(costs[-w:])
    assert snap["actual_remaining_s"] == pytest.approx(sum(_cost(i) for i in range(361, N + 1)))
    assert not snap["actual_partial"] and _codes(snap) == []
    assert snap["est_for_actual_s"] == pytest.approx(snap["remaining_s"])
    assert ww.exit_code(snap) == 0

    assert ww.main([d, "--replay-fraction", "0.1,0.25,0.5"]) == 0
    rows = capsys.readouterr().out.strip().splitlines()
    assert len(rows) == 3 and all(r.startswith("replay F=") and "err " in r for r in rows)


def test_replay_on_partial_log_compares_only_the_logged_span(tmp_path):
    snap = ww.snapshot(_rundir(tmp_path, rsl=_rsl(n_d01=700)), replay_fraction=0.5)
    assert snap["actual_partial"] and snap["completed_steps"] == 350
    assert snap["remaining_steps"] == N - 350
    assert snap["est_for_actual_s"] == pytest.approx(350 * snap["rate_s_per_step"])
    assert abs(snap["error_pct"]) < 2.0 and "CFL" not in _codes(snap)


def test_rate_window_falls_back_to_stepra_multiple_when_lcm_too_long():
    rate, w, med = ww.rate_window([1.0] * 70, 10, [10, 600])
    assert w == 50 and rate == 1.0 and med == 1.0  # LCM 600 > 70 done -> whole stepra: 5 x 10
    rate, w, _ = ww.rate_window([1.0] * 200, 10, [10, 60])
    assert w == 60  # max(20, 50, 50)=50 -> one 60-step LCM period


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


def test_nest_end_later_than_run_is_not_early(tmp_path):
    # isc21 shape: run_* shorter than the end_* span -> d01 run end wins, nests end later: fine.
    snap = ww.snapshot(_rundir(tmp_path, end_hours=(30, 30, 30)))
    assert "NEST_END_EARLY" not in _codes(snap)


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
    import json
    assert json.loads(capsys.readouterr().out)["status"] == "done"


def test_cfl_informational(tmp_path):
    cfl = ("d01 2021-07-19_00:10:00   3 points exceeded cfl=2 in domain d01 at time",)
    d = _rundir(tmp_path, rsl=_rsl(n_d01=100, tail=cfl))
    snap = ww.snapshot(d, now=time.time())
    assert snap["cfl_count"] == 1 and "CFL" in _codes(snap) and ww.exit_code(snap) == 0
    assert "CFL" not in _codes(ww.snapshot(d, now=time.time(), prev_cfl=1))
    assert "CFL" in _codes(ww.snapshot(d, now=time.time(), prev_cfl=0))


def test_cli_registration(tmp_path, monkeypatch, capsys):
    import wtune_cli
    monkeypatch.setattr(sys, "argv", ["wtune", "watch", "--help"])
    with pytest.raises(SystemExit) as e:
        wtune_cli.main()
    assert e.value.code == 0 and "watch" in capsys.readouterr().out
    d = _rundir(tmp_path, rsl=_rsl(n_d01=50))
    monkeypatch.setattr(sys, "argv", ["wtune", "watch", d, "--help"])
    with pytest.raises(SystemExit) as e:
        wtune_cli.main()
    assert e.value.code == 0 and "--replay-fraction" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["wtune", "watch", d, "--once"])
    with pytest.raises(SystemExit) as e:
        wtune_cli.main()
    assert e.value.code == 0 and "RUNNING" in capsys.readouterr().out

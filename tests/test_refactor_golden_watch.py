"""Golden snapshots of wtune_watch.snapshot (dict, key order included) and format_line, pinned
before the round-2 readability refactor split snapshot(). Run dirs come from the synthetic
builders of test_wtune_watch.py / test_watch_run_root.py (imported, not copied); file mtimes,
`now`, TZ and the file-age origin are pinned so nothing depends on the wall clock.

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_watch.py
"""
import datetime as dt
import json
import os
import time

import pytest

import test_watch_run_root as trr
import test_wtune_watch as tw
import wtune_watch as ww

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixtures", "golden", "watch")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"
MTIME = 1_700_000_000.0   # epoch s, every rsl/wrfout file


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    monkeypatch.setattr(ww, "_file_age_origin", lambda st: st.st_mtime)
    yield
    time.tzset()


def _pin_mtimes(d):
    for name in os.listdir(d):
        os.utime(os.path.join(d, name), (MTIME, MTIME))
    return str(d)


def _check(name, snap, tmp_path):
    text = (json.dumps(snap, indent=1) + "\n" + ww.format_line(snap) + "\n"
            + "exit=%d\n" % ww.exit_code(snap)).replace(str(tmp_path), "<TMP>")
    path = os.path.join(GOLDEN_DIR, name + ".txt")
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        expected = fh.read()
    assert text == expected, "snapshot differs from golden %s" % path


def _cost_nan(i):
    return float("nan") if i == 150 else tw._cost(i)


def _ramp(i):
    return tw._cost(i, ramp=0.3)


# name -> (rundir builder kwargs for tw._rundir, snapshot kwargs). now is relative to MTIME.
CASES = {
    "live_mid_run": (dict(rsl=tw._rsl(n_d01=700)), dict(now=10.0)),
    "live_big_writes": (dict(rsl=tw._rsl(n_d01=500, write=100.0)), dict(now=5.0)),
    "live_done": (dict(rsl=tw._rsl(tail=tw.SUCCESS)), dict(now=1e6)),
    "live_failed": (dict(rsl=tw._rsl(n_d01=200, tail=("--- FATAL CALLED ---",))), dict(now=1.0)),
    "live_nan": (dict(rsl=tw._rsl(n_d01=300, cost=_cost_nan)), dict(now=1.0)),
    "live_hang": (dict(rsl=tw._rsl(n_d01=500)), dict(now=301.0)),
    "live_hang_custom": (dict(rsl=tw._rsl(n_d01=500)),
                         dict(now=301.0, hang_min=10.0, hang_factor=200.0)),
    "live_hang_init": (dict(rsl="starting wrf task 0\nmed_initialdata_input: calling input_input\n"
                                "Problem opening storm_diags_d01.txt\n"), dict(now=3600.0)),
    "live_init_fresh": (dict(rsl="starting wrf task 0\n"), dict(now=10.0)),
    "live_nest_stopped": (dict(rsl=tw._rsl(n_d01=500, nest_stop=300)), dict(now=1.0)),
    "live_nest_end_early": (dict(rsl=tw._rsl(n_d01=100), end_hours=(24, 12, 24)), dict(now=1.0)),
    "live_cfl_first": (dict(rsl=tw._rsl(n_d01=100, tail=(
        "d01 2021-07-19_00:10:00   3 points exceeded cfl=2 in domain d01 at time",))),
        dict(now=1.0)),
    "live_cfl_prev": (dict(rsl=tw._rsl(n_d01=100, tail=(
        "d01 cfl=2", "d01 cfl=3", "d01 cfl=4"))), dict(now=1.0, prev_cfl=1)),
    "live_restart_unobserved": (dict(rsl=tw._rsl(n_d01=500),
                                     extra=" restart_interval = 720,\n"), dict(now=1.0)),
    "live_truncated_error": (dict(rsl=tw._rsl(n_d01=500)[:20000], rsl_out=tw._rsl(n_d01=500)),
                             dict(now=1.0)),
    "replay_025": (dict(rsl=tw._rsl(tail=tw.SUCCESS)), dict(replay_fraction=0.25)),
    "replay_partial": (dict(rsl=tw._rsl(n_d01=700)), dict(replay_fraction=0.5)),
    "replay_nest_stopped": (dict(rsl=tw._rsl(n_d01=500, nest_stop=300)),
                            dict(replay_fraction=0.9)),
    "replay_init_only": (dict(rsl="starting wrf task 0\nProblem opening x.txt\n"),
                         dict(replay_fraction=0.5)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_snapshot_golden(name, tmp_path):
    build, kw = CASES[name]
    d = _pin_mtimes(tw._rundir(tmp_path, **build))
    kw = dict(kw)
    kw["now"] = MTIME + kw.get("now", 0.0)
    _check("snapshot_%s" % name, ww.snapshot(d, **kw), tmp_path)


def test_snapshot_no_namelist_golden(tmp_path):
    (tmp_path / "rsl.error.0000").write_text(tw._rsl(n_d01=100))
    d = _pin_mtimes(tmp_path)
    _check("snapshot_no_namelist", ww.snapshot(d, now=MTIME + 1.0), tmp_path)


def test_snapshot_no_rsl_golden(tmp_path):
    (tmp_path / "namelist.input").write_text(tw._namelist())
    _check("snapshot_no_rsl", ww.snapshot(str(tmp_path), now=MTIME + 1.0), tmp_path)


@pytest.mark.parametrize("name,kw", [
    ("replay_ramp", dict(replay_fraction=0.25)),
    ("live_done_ramp", dict(now=1.0)),
])
def test_snapshot_prediction_golden(name, kw, tmp_path):
    d = _pin_mtimes(tw._rundir(tmp_path, rsl=tw._rsl(cost=_ramp, tail=tw.SUCCESS)))
    pred = ww.load_prediction(tw._pred_json(tmp_path / "pred_rebuild.json"), d)
    kw = dict(kw)
    kw["now"] = MTIME + kw.get("now", 0.0)
    _check("snapshot_pred_%s" % name, ww.snapshot(d, prediction=pred, **kw), tmp_path)


def test_snapshot_prediction_live_mismatch_golden(tmp_path):
    d = _pin_mtimes(tw._rundir(tmp_path, rsl=tw._rsl(n_d01=500, cost=_ramp)))
    pj = tmp_path / "p.json"
    tw._pred_json(pj, io={"history": [{"domain": 1, "median_s": 99.0}],
                          "restart": {"per_write_s": 30.0}})
    rec = json.loads(pj.read_text())
    rec["result"]["steps_n"] = 999
    pj.write_text(json.dumps(rec))
    pred = ww.load_prediction(str(pj), d)
    _check("snapshot_pred_live_mismatch", ww.snapshot(d, now=MTIME + 2.0, prediction=pred),
           tmp_path)


# wrfout floor / run root (test_watch_run_root builders: 1 domain, 1.0 s steps)
@pytest.mark.parametrize("name,build,now", [
    ("lagging_empty_rsl", dict(wrfout_hours=(0, 1)), 30.0),
    ("lagging_stale_timing", dict(n_steps=60, wrfout_hours=(0, 1, 2)), 30.0),
    ("lagging_frames_stale", dict(n_steps=0, wrfout_hours=(0, 1)), 5000.0),
    ("wrfout_behind_rsl", dict(n_steps=180, wrfout_hours=(0, 1)), 30.0),
])
def test_snapshot_wrfout_floor_golden(name, build, now, tmp_path):
    d = trr._rundir(tmp_path / "run", mtime=MTIME, **build)
    _pin_mtimes(d)
    _check("snapshot_%s" % name, ww.snapshot(str(d), now=MTIME + now), tmp_path)


def test_snapshot_run_root_golden(tmp_path):
    root = tmp_path / "root"
    seg_start = trr.T0 + dt.timedelta(hours=6)
    trr._rundir(root / "seg_0", start=trr.T0, hours=6, n_steps=360)
    seg = trr._rundir(root / "seg_1", start=seg_start, hours=6, n_steps=90)
    _pin_mtimes(seg)
    (root / "run_report.json").write_text(json.dumps({"status": "running", "segments": [
        {"start": trr.T0.strftime(trr.FMT), "end": seg_start.strftime(trr.FMT)},
        {"start": seg_start.strftime(trr.FMT),
         "end": (seg_start + dt.timedelta(hours=6)).strftime(trr.FMT)},
        {"start": (seg_start + dt.timedelta(hours=6)).strftime(trr.FMT),
         "end": (seg_start + dt.timedelta(hours=12)).strftime(trr.FMT)}]}))
    fol = ww.follow_run_root(str(root))
    snap = ww.annotate_run_root(ww.snapshot(fol["dir"], now=MTIME + 3.0), str(root), fol)
    _check("snapshot_run_root", snap, tmp_path)

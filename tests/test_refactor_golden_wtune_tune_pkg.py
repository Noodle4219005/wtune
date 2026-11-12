#!/usr/bin/env python3
"""Golden tests added before the wtune.tune package split (round 2): the paths the first golden
file does not pin -- the CLI surface (help text, main() exit codes, the script / `-m` entry
points), the node-search fallback, -O0 without a known run length, and two more pipeline paths.

Snapshots live next to the first file's, under tests/fixtures/golden/wtune/tune/ (prefix pkg_).
Regenerate (only with the UNCHANGED code): WTUNE_GOLDEN_REGEN=1 pytest tests/<this file>.
"""
from conftest import swap_wtune_module
import json
import os
import subprocess
import sys
import types

import pytest

import wtune.tune
import test_wtune_tune as T
import test_refactor_golden_wtune_tune as G

REPO = G.REPO
SRC = os.path.join(REPO, "src")

# the module each monkeypatched name is looked up in by the code under test; the pre-split
# single module (which regenerates the snapshots) looks every name up in itself
SEAM = dict(resolve_case="context", _total_steps="context", cold_start_prediction="prediction",
            run_rebuild_points="prediction",
            plan_predicted_points="points", measure_candidates_at_points="points",
            finish_rebuilt_points="points")


@pytest.fixture(autouse=True)
def _ascending_sweep(monkeypatch):
    # the snapshots were recorded with the ascending node sweep; the default is now promising-first
    monkeypatch.setenv("WTUNE_SWEEP_ORDER", "ascending")


def _patch(monkeypatch, name, value):
    target = getattr(wtune.tune, SEAM[name]) if hasattr(wtune.tune, "__path__") else wtune.tune
    monkeypatch.setattr(target, name, value)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Same isolation as the first golden file: no node exclusions, a stub wtune.advise."""
    from wtune.tools import scc_sweep
    for var in ("WTUNE_SBATCH_EXCLUDE", scc_sweep.PREFLIGHT_EXCLUDE_ENV):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    stub = types.ModuleType("wtune_advise")
    stub.advise_case_id = lambda case_id, site="scc", needs=(): dict(
        advice=[dict(id="stub", case=case_id)], unavailable=[])
    stub.render_text = lambda res: "advice: %d stub item(s)" % len(res["advice"])
    swap_wtune_module(monkeypatch, "advise", stub)
    monkeypatch.setenv("COLUMNS", "100")


def test_golden_main_exit_codes(monkeypatch, capsys):
    monkeypatch.setenv("SITE", "scc")
    out = {}
    for name, argv in (("bad_site", ["matthew", "--site", "x"]),
                       ("no_minutes", ["matthew", "--nodes", "2"]),
                       ("no_case", ["--minutes", "5", "--nodes", "2"])):
        rc = wtune.tune.main(argv)
        cap = capsys.readouterr()
        out[name] = dict(rc=rc, out=cap.out, err=cap.err)
    G._check("pkg_main_exit_codes.json", json.dumps(out, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("entry", ["script", "module"])
def test_golden_entry_points(entry):
    env = dict(os.environ, COLUMNS="100", PYTHONPATH=SRC)
    argv = ([sys.executable, os.path.join(SRC, "wtune", "tune"), "--help"] if entry == "script"
            else [sys.executable, "-m", "wtune.tune", "--help"])
    r = subprocess.run(argv, capture_output=True, text=True, cwd=REPO, env=env, timeout=120)
    G._check("pkg_entry_%s.txt" % entry, "rc=%d\n%s" % (r.returncode, r.stdout))


def _sinfo_missing(argv, timeout):
    raise FileNotFoundError(2, "No such file or directory", "sinfo")


def test_golden_node_search_fallback(monkeypatch, tmp_path, capsys):
    """--nodes search where no sweep job produces a measurement: the proxy fallback report."""
    sbatch, _reader, job_nodes = T.make_scaling_fakes(T.NONMONO_CURVE)
    monkeypatch.setattr(T, "make_scaling_fakes",
                        lambda curve, core_mult=None: (sbatch, lambda case, ids: [], job_nodes))
    T._run_cli(monkeypatch, tmp_path, ["--minutes", "600"],
               preflight_runner=_sinfo_missing)  # fixture recorded without sinfo; never call the real one
    out = capsys.readouterr().out
    with open(tmp_path / "out" / "tune_report.json") as f:
        js = f.read()
    G._check("pkg_node_search_fallback.txt", G._norm(out, tmp_path))
    G._check("pkg_node_search_fallback.json", G._norm(js, tmp_path))


def test_golden_level0_unknown_run_length(monkeypatch, tmp_path, capsys):
    _patch(monkeypatch, "resolve_case",
           lambda case, cpn: (dict(T.BIG_GRID, prep_dir=str(tmp_path / "m")),
                              str(tmp_path / "namelist.input"), T.ALL_NODES_PROFILE))
    _patch(monkeypatch, "_total_steps", lambda nl: None)
    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--site", "scc", "--out", str(tmp_path / "out"), "-O0",
         "--ledger-dir", str(tmp_path / "ledger")])
    wtune.tune.run(args, sbatch_runner=lambda cmd: None, clock=T.FakeClock())
    out = capsys.readouterr().out
    assert "unavailable (run length unknown)" in out
    assert os.path.exists(tmp_path / "out" / "tune_report.json")
    G._check("pkg_level0_unknown_steps.txt", G._norm(out, tmp_path))


def _env_with(**fakes):
    """_PipeEnv with some of its fakes replaced (name -> factory(env) -> fake)."""
    class Env(T._PipeEnv):
        def __init__(self, monkeypatch, *a, **k):
            super().__init__(monkeypatch, *a, **k)
            for name, make in fakes.items():
                _patch(monkeypatch, name, make(self))
    return Env


def _raising(msg):
    def f(*a, **k):
        raise RuntimeError(msg)
    return f


def _other_point(planned, T_, call):
    import datetime as _dt
    if call == 0:
        return planned, 0.07
    return [types.SimpleNamespace(at=planned[0].at + _dt.timedelta(minutes=30), reason="x")], 0.07


PIPELINES = {
    # the re-plan after phase 1 keeps 1 of the 3 submitted rebuild points: 2 are cancelled
    "fewer_points": dict(planner=lambda planned, T_, call: (planned if call == 0 else planned[:1],
                                                            0.07)),
    # the re-plan needs no point any more: every rebuild job is cancelled, t0-anchor prediction
    "replan_none": dict(planner=lambda planned, T_, call: (planned if call == 0 else [], 0.15)),
    # the re-plan wants a point that was not submitted: everything submitted is kept
    "replan_other": dict(planner=_other_point),
    # decomposition candidates whose runs fail: untried reasons + partial results
    "failed_runs": dict(drop=lambda cid: cid.startswith("D")),
    # phase 0: no ERA5-rebuildable point / the point planning itself fails
    "p0_no_points": dict(n_points=0),
    "p0_plan_fails": dict(env=dict(plan_predicted_points=lambda env: _raising("no plan"))),
    # phase 2: the rebuild jobs fail / every point measurement fails
    "p2_rebuild_fails": dict(env=dict(finish_rebuilt_points=lambda env: _raising("rebuild"))),
    "p2_measure_fails": dict(env=dict(measure_candidates_at_points=lambda env: _raising("meas"))),
}


@pytest.mark.parametrize("name", sorted(PIPELINES))
def test_golden_pipeline_more(name, monkeypatch, tmp_path, capsys):
    kw = dict(PIPELINES[name], preflight_runner=G._no_sinfo)
    if "env" in kw:
        monkeypatch.setattr(T, "_PipeEnv", _env_with(**kw.pop("env")))
    T._pipe_run(monkeypatch, tmp_path, **kw)
    out = capsys.readouterr().out
    with open(tmp_path / "out" / "tune_report.json") as f:
        js = f.read()
    G._check("pkg_pipeline_%s.txt" % name, G._norm(out, tmp_path))
    G._check("pkg_pipeline_%s.json" % name, G._norm(js, tmp_path))


# --- prediction paths no other test reaches: combine() over measured points, the C3 driver and
# the pipeline's prediction context, with wtune.predict / wtune.probe_plan faked -----------------
def _fake_predict_stack(monkeypatch, calls):
    import datetime as _dt
    wp, pp = wtune.tune.predict, wtune.tune.probe_plan
    monkeypatch.setattr(wp, "parse_namelist", lambda path: calls.append(("nl", path)) or {"nl": 1})
    monkeypatch.setattr(wp, "get_d01_steps", lambda nl: (1440, 21600.0, 15.0, None))
    monkeypatch.setattr(wp, "get_time_control_datetime",
                        lambda nl, which: _dt.datetime(2026, 1, 1, 6, 0))
    monkeypatch.setattr(wp, "g", lambda nl, grp, key, default=None: 10 if key == "radt" else 1)
    monkeypatch.setattr(wp, "steps_per_radiation_call", lambda radt, dt: int(radt * 60 / dt))
    monkeypatch.setattr(wp, "load_calibration",
                        lambda path: calls.append(("cal", os.path.basename(path))) or
                        {"drift_band_pct": 5.0, "r_rad_default_day": 3.9})
    monkeypatch.setattr(wp, "read_wrfinput_sample_points",
                        lambda case_dir: calls.append(("sp", case_dir)) or ([(30.0, -90.0)], "fake"))

    def combine(meas, run_start, run_end, dt, stepra, **kw):
        calls.append(("combine", len(meas), str(run_start), str(run_end), dt, stepra,
                      sorted((k, repr(v)) for k, v in kw.items())))
        t0 = _dt.datetime(2026, 1, 1, 7, 0)
        return dict(compute_total_s=20000.0, init_s=90.0, total_s=20090.0, range_low_s=19000.0,
                    range_high_s=21000.0, drift_pct=5.0, r_night=1.8, r_day=3.9, r_source="fake",
                    stratified_total_s=20500.0, regression_total_s=None, regression_r2=0.81,
                    cost_indicators=dict(spread_pct=4.2, sources=["fake"]),
                    points=[dict(at=t0, reason="t0", source="rebuild", s_normal=0.8, n_rad=2,
                                 r=3.7), dict(at=t0, reason="sunlit", source="trajectory",
                                              s_normal=0.9, n_rad=1, r=None)],
                    init_note="fake init note")
    monkeypatch.setattr(pp, "combine", combine)


@pytest.mark.parametrize("drift", [None, 7.5])
def test_golden_full_run_prediction_multipoint(drift, monkeypatch, tmp_path):
    calls = []
    _fake_predict_stack(monkeypatch, calls)
    ctx = T._ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"),
                 case_json=dict(T.MATTHEW_LIKE, prep_dir="$HOME/wrf-cases/matthew"))
    if drift is not None:
        ctx.prediction_cfg["drift_band_pct"] = drift
    best = dict(id="D1", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    meas = [types.SimpleNamespace(at=None), types.SimpleNamespace(at=None)]
    text = wtune.tune.build_full_run_prediction(ctx, best, 2, {}, None, probe_measurements=meas)
    G._check("pkg_multipoint_%s.txt" % drift, G._norm(json.dumps(
        dict(text=text, calls=calls), indent=1, default=repr), tmp_path) + "\n")


def test_golden_prediction_context(monkeypatch, tmp_path):
    calls = []
    _fake_predict_stack(monkeypatch, calls)
    ctx = T._ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"),
                 case_json=dict(T.MATTHEW_LIKE, prep_dir=str(tmp_path / "case")))
    ctx.prediction_cfg["drift_band_pct"] = 9.0
    target = wtune.tune.prediction if hasattr(wtune.tune, "__path__") else wtune.tune
    out = target._prediction_context(ctx)
    G._check("pkg_prediction_context.txt", G._norm(json.dumps(
        dict(out=out, calls=calls), indent=1, default=repr), tmp_path) + "\n")

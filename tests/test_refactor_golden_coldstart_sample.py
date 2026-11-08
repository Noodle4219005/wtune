"""Golden snapshots of tools/coldstart_sample.render_run_sbatch, pinned before the readability
refactor split it into private helpers. The fixtures under tests/fixtures/golden/coldstart_sample/
were written by the UNCHANGED renderer; any byte of difference in the rendered run.sbatch is a
behaviour change, not a cosmetic one (the script is what Slurm runs).

Regenerate (only when a behaviour change is intended) with:
    WTUNE_GOLDEN_REGEN=1 python -m pytest -q tests/test_refactor_golden_coldstart_sample.py
"""
import os

import pytest

from wtune.tools import coldstart_sample as cs
from wtune.tools import scc_sweep

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fixtures", "golden", "coldstart_sample")
REGEN = os.environ.get("WTUNE_GOLDEN_REGEN") == "1"


def _config(ranks=72, omp=1, nx=9, ny=8, numtiles=1):
    return {"ranks": ranks, "omp": omp, "nx": nx, "ny": ny, "numtiles": numtiles}


def _era5(n, interval_h=1):
    return [{"prefix": "ERA5", "vtable": "none", "forcing_interval_h": interval_h}
            for _ in range(n)]


# name -> (plan, render_run_sbatch kwargs). Each one exercises a branch of the renderer:
# window count, plan_dir present/absent, nests (max_dom), ML hook, --real-only, forcing chain
# (default GFS / plan-recorded primary / all-intermediate ERA5 / mixed), estimates.
CASES = {
    "single_window": (
        {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
         "nodes": 1, "config": _config()},
        {}),
    "multi_window": (
        {"windows": [{"t": "2019-11-26_12:00:00", "t_end": "2019-11-26_18:00:00", "reason": "t0"},
                     {"t": "2019-11-27_00:00:00", "t_end": "2019-11-27_06:00:00", "reason": "mid"},
                     {"t": "2019-11-27_12:00:00", "t_end": "2019-11-27_18:00:00", "reason": "late"}],
         "nodes": 4, "config": _config(ranks=90, omp=2, nx=9, ny=10, numtiles=2),
         "window_interval_seconds": 21600, "_plan_dir": "/scratch/cs plan/severe2023"},
        {"per_window_estimate_min": 25, "wps_estimate_min": 30}),
    "nested_domains": (
        {"windows": [{"t": "2017-08-23_06:00:00", "t_end": "2017-08-23_12:00:00", "reason": "t0"},
                     {"t": "2017-08-24_00:00:00", "t_end": "2017-08-24_03:00:00", "reason": "late"}],
         "nodes": 2, "config": _config(ranks=144, omp=1, nx=12, ny=12, numtiles=4),
         "max_dom": 3, "interval_seconds": 10800, "_plan_dir": "/scratch/plans/harvey"},
        {}),
    "ml_model": (
        {"windows": [{"t": "2019-11-26_15:00:00", "t_end": "2019-11-26_21:00:00", "reason": "t0"}],
         "nodes": 1, "config": _config(), "ml_model": "/plan/dir/report.json",
         "_plan_dir": "/plan/dir"},
        {}),
    "real_only": (
        {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"},
                     {"t": "2019-11-26_18:00:00", "reason": "late"}],
         "nodes": 1, "config": _config(), "real_only": True},
        {}),
    "ml_and_real_only": (
        {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
         "nodes": 1, "config": _config(), "real_only": True, "ml_model": "/plan/dir/report.json"},
        {}),
    "era5_forcing": (
        {"windows": [{"t": "2023-03-31_18:00:00", "t_end": "2023-03-31_21:00:00", "reason": "t0"},
                     {"t": "2023-04-01_00:00:00", "t_end": "2023-04-01_02:00:00", "reason": "late"}],
         "nodes": 2, "config": _config(ranks=128, omp=1, nx=8, ny=16),
         "_plan_dir": "/scratch/plans/severe2023-era5"},
        {"per_window_forcing": _era5(2)}),
    "mixed_forcing": (
        {"windows": [{"t": "2023-03-31_18:00:00", "t_end": "2023-03-31_21:00:00", "reason": "t0"},
                     {"t": "2023-04-01_00:00:00", "t_end": "2023-04-01_06:00:00", "reason": "late"}],
         "nodes": 1, "config": _config()},
        {"per_window_forcing": [
            {"prefix": "ERA5", "vtable": "none", "forcing_interval_h": 1},
            {"prefix": "FNL", "vtable": "Vtable.GFS", "forcing_interval_h": 6}]}),
    # The renderer itself never reads a bbox (the bbox only matters to the ERA5 fetch), so this
    # plan pins that a southern-hemisphere case's plan -- with its plan-recorded forcing chain
    # (namelist_wps_fields / forcing_chain_primary) -- renders exactly as before.
    "southern_hemisphere": (
        {"windows": [{"t": "2016-02-20_00:00:00", "t_end": "2016-02-20_06:00:00", "reason": "t0"},
                     {"t": "2016-02-20_12:00:00", "t_end": "2016-02-20_18:00:00", "reason": "late"}],
         "nodes": 3, "config": _config(ranks=200, omp=1, nx=10, ny=20, numtiles=1),
         "bbox": [-25.5, 170.0, -10.0, -170.0], "case_dir": "/cases/winston2016",
         "namelist_wps_fields": {"prefix": "CFSR"},
         "forcing_chain_primary": {"prefix": "CFSR", "vtable": "Vtable.CFSR"},
         "window_interval_seconds": 21600, "_plan_dir": "/scratch/plans/winston2016"},
        {"site": "scc", "per_window_estimate_min": 40}),
}


@pytest.fixture(autouse=True)
def _pinned_environment(monkeypatch):
    # The rendered text embeds the tool's own install path and the env-driven --exclude list:
    # pin both so the snapshot does not depend on where the checkout lives or the caller's env.
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    monkeypatch.delenv(scc_sweep.PREFLIGHT_EXCLUDE_ENV, raising=False)
    monkeypatch.setattr(cs, "_SELF_PATH", "/opt/wtune/src/tools/coldstart_sample.py")
    monkeypatch.setattr(cs, "_TREE", "/opt/wtune")
    monkeypatch.setattr(cs, "_SCM_PATH", "/opt/wtune/src/tools/state_correction_model.py")


def _check_golden(name, text):
    path = os.path.join(GOLDEN_DIR, name)
    if REGEN:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
    with open(path) as fh:
        expected = fh.read()
    assert text == expected, "rendered text differs from golden %s" % path


@pytest.mark.parametrize("name", sorted(CASES))
def test_render_run_sbatch_golden(name):
    plan, kwargs = CASES[name]
    _check_golden("run_sbatch_%s.sh" % name, cs.render_run_sbatch(plan, **kwargs))


def test_render_run_sbatch_overpacked_node_assertion():
    plan = {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}],
            "nodes": 1, "config": _config(ranks=73)}
    with pytest.raises(AssertionError) as exc:
        cs.render_run_sbatch(plan)
    assert str(exc.value) == "ppn=73 exceeds CORES_PER_NODE=72 (ranks=73 nodes=1)"


# ---------------------------------------------------------------------------
# inspect_case / build_plan / main: pinned on the namelist fixtures test_coldstart_sample.py uses
# (imported, not copied). Results are snapshotted as JSON with the temp dir replaced by <TMP>;
# `now` and the geog directory are always explicit so nothing depends on the date or $WPS_GEOG.
# ---------------------------------------------------------------------------
import contextlib  # noqa: E402
import datetime  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
import test_coldstart_sample as tcs  # noqa: E402

NOW = datetime.datetime(2026, 10, 1, 0, 0, 0)


def _json_golden(name, obj, tmp_path):
    text = json.dumps(obj, indent=1, default=repr).replace(str(tmp_path), "<TMP>") + "\n"
    _check_golden(name + ".json", text)


def _case(tmp_path, extra="", wps=True, files=(), replace=(), **nl_kwargs):
    """A case dir from test_coldstart_sample's namelist template. `extra` can only ADD keys (a
    repeated &group keeps the first value of a key it already has), so `replace` rewrites
    template lines in place for keys the template already sets."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    tcs.write_namelist(str(case_dir / "namelist.input"), extra=extra, **nl_kwargs)
    text = (case_dir / "namelist.input").read_text()
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    (case_dir / "namelist.input").write_text(text)
    if wps:
        tcs.write_namelist_wps(str(case_dir / "namelist.wps"))
    for f in files:
        (case_dir / f).write_text("x")
    return str(case_dir)


def _geog(tmp_path, names=tcs.SCC_GEOG_LISTING):
    return tcs._make_geog_dir(str(tmp_path), names=names)


def _outcome(fn, *args, **kwargs):
    """{'result'|'error', 'stdout', 'stderr'} for one call."""
    out, err = io.StringIO(), io.StringIO()
    rec = {}
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rec["result"] = fn(*args, **kwargs)
        except (cs.ColdstartError, SystemExit) as exc:
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
    rec["stdout"] = out.getvalue()
    rec["stderr"] = err.getvalue()
    return rec


def _kv(key, value):
    return " %-35s = %s," % (key, value)


# d02 starts an hour after d01 (and a nest with the same start, for the consistent branch).
NEST_REPLACE = (
    (_kv("max_dom", 1), _kv("max_dom", "2")),
    (_kv("e_we", 425), _kv("e_we", "425, 151")),
    (_kv("e_sn", 300), _kv("e_sn", "300, 151")),
    (_kv("parent_id", 0), _kv("parent_id", "0, 1")),
    (_kv("i_parent_start", 0), _kv("i_parent_start", "0, 100")),
    (_kv("j_parent_start", 0), _kv("j_parent_start", "0, 100")),
    (_kv("parent_grid_ratio", 1), _kv("parent_grid_ratio", "1, 3")),
    (_kv("start_year", 2020), _kv("start_year", "2020, 2020")),
    (_kv("start_month", 1), _kv("start_month", "1, 1")),
    (_kv("start_day", 1), _kv("start_day", "1, 1")),
    (_kv("start_minute", "00"), _kv("start_minute", "00, 00")),
    (_kv("start_second", "00"), _kv("start_second", "00, 00")),
)
NEST_MISMATCH = NEST_REPLACE + ((_kv("start_hour", 0), _kv("start_hour", "0, 1")),)
NEST_MATCH = NEST_REPLACE + ((_kv("start_hour", 0), _kv("start_hour", "0, 0")),)

# name -> (_case kwargs, geog listing or None for a missing geog dir, inspect_case kwargs)
INSPECT_CASES = {
    "completable": (dict(), tcs.SCC_GEOG_LISTING, {}),
    "no_projection": (dict(wps=False), tcs.SCC_GEOG_LISTING, {}),
    "ifire": (dict(ifire=2), tcs.SCC_GEOG_LISTING, {}),
    "lake_missing": (dict(extra="&physics\n sf_lake_physics = 1,\n /\n"),
                     tcs.SCC_GEOG_LISTING, {}),
    "lake_present": (dict(extra="&physics\n sf_lake_physics = 1,\n /\n"),
                     tcs.SCC_GEOG_LISTING + ("lake_depth_30s",), {}),
    "lake_unlisted_geog": (dict(extra="&physics\n sf_lake_physics = 1,\n /\n"), None, {}),
    "urban_bep": (dict(extra="&physics\n sf_urban_physics = 2,\n /\n"), tcs.SCC_GEOG_LISTING, {}),
    "chem_missing_inputs": (dict(extra="&chem\n chem_opt = 401,\n chem_in_opt = 1,\n /\n"),
                            tcs.SCC_GEOG_LISTING, {}),
    "chem_with_inputs": (dict(extra="&chem\n chem_opt = 401,\n chem_in_opt = 1,\n /\n",
                              files=("wrfchemi_d01", "wrf_chem_input_d01")),
                         tcs.SCC_GEOG_LISTING, {}),
    "obs_nudge": (dict(extra="&fdda\n obs_nudge_opt = 1,\n /\n"), tcs.SCC_GEOG_LISTING, {}),
    "filled_differently": (
        dict(extra="&physics\n mp_physics = 28,\n sf_surface_physics = 3,\n"
                   " num_soil_layers = 6,\n /\n&domains\n num_land_cat = 24,\n /\n"
                   "&fdda\n grid_fdda = 1,\n /\n"),
        tcs.SCC_GEOG_LISTING, {}),
    "nest_start_mismatch": (dict(replace=NEST_MISMATCH), tcs.SCC_GEOG_LISTING, {}),
    "nest_start_match": (dict(replace=NEST_MATCH), tcs.SCC_GEOG_LISTING, {}),
    "date_2010": (dict(sy=2010, ey=2010), tcs.SCC_GEOG_LISTING, {}),
    "date_1975": (dict(sy=1975, ey=1975), tcs.SCC_GEOG_LISTING, {}),
    "date_1975_grib_dir": (dict(sy=1975, ey=1975), tcs.SCC_GEOG_LISTING,
                           {"grib_dir": "/data/grib"}),
}


@pytest.mark.parametrize("name", sorted(INSPECT_CASES))
def test_inspect_case_golden(name, tmp_path):
    case_kwargs, geog_names, kwargs = INSPECT_CASES[name]
    case_dir = _case(tmp_path, **case_kwargs)
    geog = _geog(tmp_path, geog_names) if geog_names is not None else str(tmp_path / "nogeog")
    rows, blockers, nl = cs.inspect_case(case_dir, now=NOW, geog_path=geog, **kwargs)
    _json_golden("inspect_" + name, {"rows": rows, "blockers": blockers,
                                     "nl_is_none": nl is None}, tmp_path)


def test_inspect_case_no_namelist_golden(tmp_path):
    rows, blockers, nl = cs.inspect_case(str(tmp_path), now=NOW, geog_path=_geog(tmp_path))
    _json_golden("inspect_no_namelist", {"rows": rows, "blockers": blockers,
                                         "nl_is_none": nl is None}, tmp_path)


def _plan_outcome(tmp_path, case_dir, **kwargs):
    out_dir = str(tmp_path / "plan")
    kwargs.setdefault("now", NOW)
    kwargs.setdefault("geog_path", _geog(tmp_path))
    rec = _outcome(cs.build_plan, case_dir, out_dir, **kwargs)
    for fname in ("plan.json", "namelist.wps"):
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            with open(path) as fh:
                rec[fname] = fh.read()
    return rec


def _dt(s):
    return datetime.datetime.strptime(s, cs._WRF_TIME_FMT)


# name -> (_case kwargs, build_plan kwargs)
PLAN_CASES = {
    "smoke_k4": (dict(ed=2, eh=0), dict(k=4, nodes=1, ranks=72, omp=1, nx=9, ny=8,
                                        spinup_min=1)),
    "defaults_two_day": (dict(run_hours=48), dict(spinup_min=1)),
    "pinned_times": (dict(run_hours=48), dict(spinup_min=1, times=[
        _dt("2020-01-01_03:00:00"), _dt("2020-01-01_09:00:00")])),
    "pinned_times_k1": (dict(run_hours=48), dict(spinup_min=1, k=1, times=[
        _dt("2020-01-01_03:00:00"), _dt("2020-01-01_09:00:00"), _dt("2020-01-01_15:00:00")])),
    "pinned_off_grid": (dict(run_hours=48), dict(spinup_min=1, times=[_dt("2020-01-01_01:00:00")])),
    "pinned_does_not_fit": (dict(run_hours=6), dict(spinup_min=200,
                                                    times=[_dt("2020-01-01_03:00:00")])),
    "date_shift_to": (dict(run_hours=24), dict(spinup_min=1, k=2,
                                               date_shift_to="2021-07-15_06:00:00")),
    "forcing_pin_fnl_offgrid_start": (dict(sh=1, run_hours=24),
                                      dict(spinup_min=1, k=2, forcing="d083002")),
    "unknown_forcing_pin": (dict(run_hours=24), dict(spinup_min=1, forcing="nosuchdataset")),
    "real_only_ml_numtiles": (dict(run_hours=24), dict(spinup_min=1, k=2, real_only=True,
                                                       ml_model="/models/report.json",
                                                       numtiles=4, nodes=2, ranks=144, omp=2)),
    "measure_steps": (dict(run_hours=24), dict(spinup_min=1, k=2, measure_steps=20)),
    "measure_steps_too_few": (dict(run_hours=24), dict(spinup_min=1, k=2, measure_steps=5)),
    "io_probe": (dict(run_hours=24), dict(spinup_min=1, k=2, io_probe=True)),
    # dt=72 s: 16 measured steps is not a whole-minute window, so the probe grows it to 20.
    "io_probe_nwp_diagnostics": (
        dict(run_hours=24, extra="&physics\n nwp_diagnostics = 1,\n /\n"),
        dict(spinup_min=1, k=2, io_probe=True, measure_steps=16)),
    "nested": (dict(run_hours=24, replace=NEST_MATCH), dict(spinup_min=1, k=2)),
    "blocked": (dict(ifire=1), dict(spinup_min=1)),
    "radt_zero": (dict(replace=((_kv("radt", 15), _kv("radt", 0)),)), dict(spinup_min=1)),
    "window_too_long": (dict(run_hours=1), dict(spinup_min=300)),
    "both_shift_and_analog": (dict(run_hours=24), dict(
        spinup_min=1, date_shift_to="2021-07-15_06:00:00", analog_path="/no/analog.json")),
}


@pytest.mark.parametrize("name", sorted(PLAN_CASES))
def test_build_plan_golden(name, tmp_path):
    case_kwargs, kwargs = PLAN_CASES[name]
    rec = _plan_outcome(tmp_path, _case(tmp_path, **case_kwargs), **kwargs)
    _json_golden("plan_" + name, rec, tmp_path)


@pytest.mark.parametrize("chosen", [{"date": "2019-03-02_00:00:00"}, {}])
def test_build_plan_analog_golden(chosen, tmp_path):
    analog = tmp_path / "analog.json"
    analog.write_text(json.dumps({"chosen": chosen, "quality": 0.83}))
    rec = _plan_outcome(tmp_path, _case(tmp_path, run_hours=24), spinup_min=1, k=2,
                        analog_path=str(analog))
    _json_golden("plan_analog_%s" % ("chosen" if chosen else "no_chosen"), rec, tmp_path)


@pytest.mark.parametrize("name", ["inspect_ok", "inspect_blocked", "plan_ok", "plan_blocked",
                                  "plan_error", "stage_error", "score_error"])
def test_main_dispatch_golden(name, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")
    geog = _geog(tmp_path)
    case_dir = _case(tmp_path, ifire=(1 if name.endswith("blocked") else 0), run_hours=24)
    out = str(tmp_path / "plan")
    argv = {
        "inspect_ok": ["inspect", "--case-dir", case_dir, "--geog-path", geog],
        "inspect_blocked": ["inspect", "--case-dir", case_dir, "--geog-path", geog],
        "plan_ok": ["plan", "--case-dir", case_dir, "--out", out, "--geog-path", geog,
                    "--spinup-min", "1", "--k", "2", "--times",
                    "2020-01-01_03:00:00, 2020-01-01_09:00:00", "--real-only"],
        "plan_blocked": ["plan", "--case-dir", case_dir, "--out", out, "--geog-path", geog],
        "plan_error": ["plan", "--case-dir", case_dir, "--out", out, "--geog-path", geog,
                       "--spinup-min", "1", "--times", "2020-01-01_01:00:00"],
        "stage_error": ["stage", "--plan", str(tmp_path / "noplan")],
        "score_error": ["score", "--plan", str(tmp_path / "noplan")],
    }[name]
    try:
        rec = _outcome(cs.main, argv)
    except (OSError, ValueError) as exc:   # a non-ColdstartError escapes main() unchanged
        rec = {"raised": "%s: %s" % (type(exc).__name__, exc)}
    if os.path.exists(os.path.join(out, "plan.json")):
        with open(os.path.join(out, "plan.json")) as fh:
            rec["plan.json"] = fh.read()
    _json_golden("main_" + name, rec, tmp_path)

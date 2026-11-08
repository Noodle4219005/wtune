#!/usr/bin/env python3
"""pytest suite for wtune/tune/ -- fake Slurm (sbatch/sacct/scancel), a synthetic-time clock and
synthetic ledgers throughout, so this never touches a real cluster and never needs
numpy/scikit-learn/catboost (predictor/decomp_search.py, the real adaptive-round proposer, is only
imported lazily by wtune.tune._default_proposer when no proposer is injected -- every test here
injects one). Run with:

  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest test_wtune_tune.py
"""
import csv
import json
import os
import atexit
import re
import shutil
import subprocess
import tempfile

import pytest

import wtune.config
import wtune.tune

# per-session scratch for _ctx's out/ledger dirs: a fixed /tmp path is owned by whichever user
# ran the tests first on a shared node (scc work3: another user's /tmp/wtune-tune-test-out
# made 16 tests fail with PermissionError)
_SCRATCH = tempfile.mkdtemp(prefix="wtune-tune-test-")
atexit.register(shutil.rmtree, _SCRATCH, True)

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ascending_sweep(monkeypatch):
    # the node sweep defaults to most-promising-first; these tests pin the ascending path
    monkeypatch.setenv("WTUNE_SWEEP_ORDER", "ascending")


class FakeClock:
    """Virtual time: .sleep() advances .t instantly instead of blocking, so a budget-deadline
    test can simulate 30 minutes elapsing without a real wait."""

    def __init__(self, start=0.0):
        self.t = start

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(0.0, seconds)


def make_fake_sbatch(start=1000):
    """Also records, per job id, the CONFIG_FILE path from --export=... (parsed out of the exact
    command scc_sweep.build_sbatch_command produces), so a fake ledger_reader can read back the
    real configs.json wtune.tune wrote and fabricate a plausible measurement per config id."""
    counter = {"n": start}
    calls = []
    job_to_configs_path = {}

    def runner(cmd_str):
        calls.append(cmd_str)
        counter["n"] += 1
        job_id = str(counter["n"])
        m = re.search(r"CONFIG_FILE=(\S+)", cmd_str)
        if m:
            job_to_configs_path[job_id] = m.group(1)
        return "Submitted batch job %s\n" % job_id

    return runner, calls, job_to_configs_path


def make_fake_sacct(states):
    """`states`: {job_id(str): (State, Elapsed)}, mutable by the test; any job id not in it is
    reported COMPLETED (so a test that doesn't care about polling gets instant termination)."""

    def runner(cmd):
        idx = cmd.index("-j")
        ids = cmd[idx + 1].split(",")
        lines = []
        for jid in ids:
            state, elapsed = states.get(jid, ("COMPLETED", "00:01:00"))
            lines.append("%s|%s|%s" % (jid, state, elapsed))
        return "\n".join(lines)

    return runner


def make_fake_scancel():
    calls = []

    def runner(job_ids):
        calls.append(list(job_ids))

    return runner, calls


def make_ledger_reader_from_configs(job_to_configs_path, base_s=1.0, reps=1):
    """Fabricates one successful measurement per config id actually written to disk for that job
    (distinct, deterministic medians so ranking is unambiguous)."""

    def reader(case, job_ids):
        rows = []
        for j in job_ids:
            path = job_to_configs_path.get(j)
            if not path or not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            for i, cfg in enumerate(data["configs"]):
                s = base_s / (1.0 + 0.1 * i)
                for rep in range(1, reps + 1):
                    rows.append(dict(run_id="%s_%s_r%d_%s" % (case, cfg["id"], rep, j), job_id=j,
                                      s_per_step_median=str(s)))
        return rows

    return reader


def _result(cid, median, config=None):
    return dict(cid=cid, config=config if config is not None else {"id": cid}, s_values=[median],
                median=median, lo=median, hi=median, n=1)


def _ctx(**overrides):
    clock = overrides.pop("clock", None) or FakeClock()
    sbatch_runner = overrides.pop("sbatch_runner", None)
    if sbatch_runner is None:
        sbatch_runner, _calls, _paths = make_fake_sbatch()
    sacct_runner = overrides.pop("sacct_runner", None) or make_fake_sacct({})
    scancel_runner = overrides.pop("scancel_runner", None)
    if scancel_runner is None:
        scancel_runner, _ = make_fake_scancel()
    ledger_reader = overrides.pop("ledger_reader", None) or (lambda case, job_ids: [])
    proposer = overrides.pop("proposer", None)

    base = dict(
        case="matthew", case_json=dict(MATTHEW_LIKE, prep_dir="/fake/matthew"),
        namelist_path="/fake/matthew/namelist.input", nodes=1, cores_per_node=72,
        hardware_id="scc-gh200", steps=20, seed=1, kappa=0.5, budget_s=1800.0, max_nodes=13,
        out_dir=os.path.join(_SCRATCH, "out"), ledger_dir=os.path.join(_SCRATCH, "ledger"),
        env_source="$HOME/wtune-scc/env.sh",
        sbatch_script="$HOME/wtune/campaign/strict_campaign.sbatch", sacct_cmd="sacct",
        poll_interval=0.0, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
        scancel_runner=scancel_runner, ledger_reader=ledger_reader, proposer=proposer, clock=clock,
    )
    base.update(overrides)
    ctx = wtune.tune.Ctx(**base)
    ctx.start_time = clock.now()
    return ctx


def _profile(table):
    return {"core_count_priors": {"efficiency_table": table}}


def test_choose_nodes_knee_stops_at_first_step_when_doubling_never_pays_off():
    # speedup(1)=0.5, speedup(2)=0.6 -- ratio 1.2 < 1.5x on the very first step.
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.5},
        {"nodes": 2, "legal": True, "efficiency_E": 0.3},
    ]
    nodes, note = wtune.tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 1
    assert "1.5" in note


def test_choose_nodes_knee_empty_table_defaults_to_one():
    nodes, note = wtune.tune.choose_nodes_knee(_profile([]), max_nodes=13)
    assert nodes == 1
    assert "no legal" in note


def test_choose_nodes_dispatches_by_policy_and_always_warns_uncalibrated():
    table = [{"nodes": 1, "legal": True, "efficiency_E": 0.9},
             {"nodes": 2, "legal": True, "efficiency_E": 0.85}]
    for policy, expect_nodes in (("min", 1), ("cap", 2), ("knee", 2)):
        nodes, _note, warning = wtune.tune.choose_nodes(_profile(table), max_nodes=13,
                                                          policy=policy)
        assert nodes == expect_nodes
        assert "UNCALIBRATED" in warning
        assert policy in warning


def test_choose_nodes_rejects_unknown_policy():
    with pytest.raises(SystemExit):
        wtune.tune.choose_nodes(_profile([]), max_nodes=13, policy="bogus")


# ---------------------------------------------------------------------------
# failed configs dropped
# ---------------------------------------------------------------------------
def test_aggregate_by_cid_drops_failed_rows():
    configs = {"ANCHOR": {"id": "ANCHOR", "ranks": 8}, "G0001": {"id": "G0001", "ranks": 8}}
    rows = [
        dict(run_id="matthew_ANCHOR_r1_1000", s_per_step_median="0.5"),
        dict(run_id="matthew_G0001_r1_1000", s_per_step_median=""),          # failed: empty
        dict(run_id="matthew_G0001_r1_1001", s_per_step_median="not-a-number"),  # malformed
        dict(run_id="not-a-recognized-run-id-format", s_per_step_median="1.0"),  # unparsable
    ]
    out = wtune.tune.aggregate_by_cid(rows, configs)
    assert set(out) == {"ANCHOR"}
    assert out["ANCHOR"]["median"] == 0.5
    assert out["ANCHOR"]["n"] == 1


def test_run_confirm_skips_only_when_even_the_cheapest_candidate_alone_exceeds_budget():
    all_results = {"SLOW": _result("SLOW", 50.0, dict(id="SLOW", ranks=8, omp=1, nx=2, ny=4,
                                                        numtiles=1, dist="block"))}
    ctx = _ctx(budget_s=60.0, steps=20)   # 60s budget, nowhere near enough for one 50s/step config
    job_id, confirm_results = wtune.tune.run_confirm(ctx, {}, all_results, anchor_cid=None,
                                                       s_est=50.0)
    assert job_id is None
    assert confirm_results is None


def test_run_round0_submits_nothing_when_budget_too_small():
    # budget_s=60 << even a single 1-config job's padded estimate at the 10s/step fallback
    # (no ledger rows -> CONSERVATIVE_S_PER_STEP): round0 must refuse to submit any slot.
    sbatch_runner, calls, _paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, budget_s=60.0)
    job_ids, configs_by_cid, rows = wtune.tune.run_round0(ctx)
    assert job_ids == []
    assert calls == []
    assert ctx.owned_jobs == []


def test_parse_slurm_elapsed():
    assert wtune.tune._parse_slurm_elapsed("00:01:30") == 90.0
    assert wtune.tune._parse_slurm_elapsed("01:23:45") == 5025.0
    assert wtune.tune._parse_slurm_elapsed("23:45") == 23 * 60 + 45
    assert wtune.tune._parse_slurm_elapsed("1-00:01:23") == 86400 + 83
    assert wtune.tune._parse_slurm_elapsed("") is None
    assert wtune.tune._parse_slurm_elapsed(None) is None
    assert wtune.tune._parse_slurm_elapsed("garbage") is None


def test_measure_startup_s_ignores_non_completed_jobs():
    cfg_a = dict(id="A", ranks=8, omp=1, nx=2, ny=4, numtiles=1, dist="block")
    job_configs = {"1001": [cfg_a]}
    states = {"1001": ("CANCELLED", "00:30:00")}   # elapsed reflects the deadline, not real cost
    wave_results = {"A": dict(median=2.0)}
    startup_s, n_runs = wtune.tune.measure_startup_s(job_configs, states, wave_results, steps=10)
    assert startup_s is None
    assert n_runs == 0


def test_run_adaptive_rounds_stops_at_75pct_projected_finish():
    def fake_proposer(state, batch_size, kappa=None):
        cfg = dict(id="RX", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block", blocktime="0")
        return dict(stopped=False, round=1), [cfg] * batch_size

    sbatch_runner, calls, _paths = make_fake_sbatch()
    clock = FakeClock(start=0.0)
    clock.t = 1000.0   # already 100% of a tiny budget elapsed
    ctx = _ctx(sbatch_runner=sbatch_runner, proposer=fake_proposer, budget_s=100.0, clock=clock)
    all_configs_by_cid, all_results = {}, {}
    wtune.tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)
    assert calls == []
    assert ctx.owned_jobs == []


def test_strict_guard_reads_every_namelist_line_the_report_hands_out(tmp_path):
    # a solution-changing key slipped into namelist_lines must be caught, not just the
    # three decomposition keys the guard happens to expect today
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text("&domains\n time_step = 60,\n nproc_x = -1,\n/\n"
                             "&physics\n mp_physics = 8,\n/\n")
    ctx = _ctx(namelist_path=str(namelist_path))
    ok = {"namelist_lines": ["nproc_x=2,", "nproc_y=4,", "numtiles=1,", "mp_physics=8,"]}
    wtune.tune._assert_report_is_strict_only(ctx, ok)          # unchanged physics: passes
    bad = {"namelist_lines": ["nproc_x=2,", "mp_physics=6,"]}
    with pytest.raises(wtune.tune.wc.ComplianceError, match="mp_physics"):
        wtune.tune._assert_report_is_strict_only(ctx, bad)
    with pytest.raises(wtune.tune.wc.ComplianceError, match="time_step"):
        wtune.tune._assert_report_is_strict_only(ctx, {"namelist_lines": ["time_step=45,"]})


def test_build_report_propagates_a_strict_guard_violation(tmp_path, monkeypatch):
    # wiring test: a ComplianceError from the guard must not be swallowed by build_report.
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text("&domains\n nproc_x = -1,\n nproc_y = -1,\n numtiles = 1,\n/\n")

    def always_raise(orig, final, **kw):
        raise wtune.tune.wc.ComplianceError("simulated non-STRICT change")

    monkeypatch.setattr(wtune.tune.wc, "assert_strict_only", always_raise)

    ctx = _ctx(namelist_path=str(namelist_path))
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    with pytest.raises(wtune.tune.wc.ComplianceError):
        wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                confirm_results=None, anchor_cid=None)


def test_round0_measuring_0_3s_per_step_lets_adaptive_and_confirm_both_run_within_budget(
        monkeypatch, tmp_path):
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    profile = _profile([{"nodes": 1, "legal": True, "efficiency_E": 0.95}])
    monkeypatch.setattr(
        wtune.tune.context, "resolve_case",
        lambda case, cores_per_node: (case_json, str(tmp_path / "namelist.input"), profile))

    sbatch_runner, sbatch_calls, job_to_configs_path = make_fake_sbatch(start=9700)
    sacct_runner = make_fake_sacct({})
    scancel_runner, scancel_calls = make_fake_scancel()
    # round0 measures 0.30/0.27/0.25 s/step here -- matching the real conus12 run's 0.30-0.93
    # s/step -- instead of the no-ledger fallback (CONSERVATIVE_S_PER_STEP=10s/step) every later
    # stage used to stay pinned to.
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=0.3, reps=2)

    proposer_calls = {"n": 0}

    def one_round_proposer(state, batch_size, kappa=None):
        proposer_calls["n"] += 1
        if proposer_calls["n"] == 1:
            cfg = dict(id="R01D0001", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                       blocktime="0")
            return dict(stopped=False, round=1), [cfg]
        return dict(stopped=True, reason="test stub"), []

    args = wtune.tune.build_arg_parser().parse_args([
        "matthew", "--site", "scc", "--minutes", "30", "--nodes", "1", "--max-nodes", "1",
        "--out", str(tmp_path / "out"), "--steps", "20",
        "--ledger-dir", str(tmp_path / "ledger"),
    ])
    report = wtune.tune.run(args, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
                             scancel_runner=scancel_runner, ledger_reader=ledger_reader,
                             proposer=one_round_proposer, clock=FakeClock())

    # adaptive round DID submit a batch (it may then stop early: round 0's leader stays clear)
    assert any("round1" in c for c in sbatch_calls)
    assert any(row["confirmed"] for row in report["ranking"])    # confirm DID run
    assert len(report["ranking"]) > 1                             # round0's top-up widened coverage
    assert scancel_calls == []                                   # everything fit inside the budget
    assert report["elapsed_minutes"] < report["budget_minutes"]  # budget was not exhausted


# ---------------------------------------------------------------------------
# regression: the full-run-prediction hook must feed wtune.predict this run's own measurements,
# not call it with ledger_paths=None (which always fell back to source=no_probe_fallback, +-50%,
# even right after measuring the winning config directly).
# ---------------------------------------------------------------------------
_PREDICT_TEST_NAMELIST = """\
&time_control
 run_days = 0,
 run_hours = 1,
 run_minutes = 0,
 run_seconds = 0,
 history_interval = 60,
 io_form_history = 2,
 /
&domains
 max_dom = 1,
 e_we = 91,
 e_sn = 100,
 e_vert = 45,
 dx = 3000,
 dy = 3000,
 time_step = 15,
 /
&physics
 radt = 10,
 /
"""


def test_write_probe_ledger_makes_full_run_prediction_use_measured_source(tmp_path):
    assert wtune.tune.predict is not None, "wtune predict must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)

    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}
    confirm_results = {"ANCHOR": dict(cid="ANCHOR", config=best_config, s_values=[0.90, 0.93],
                                      median=0.915, lo=0.90, hi=0.93, n=2)}

    probe_path = wtune.tune.write_probe_ledger(ctx, all_results, confirm_results)
    assert probe_path and os.path.exists(probe_path)
    with open(probe_path) as f:
        probe_rows = list(csv.DictReader(f))
    assert len(probe_rows) == 2   # confirm's own REPS=2 replicates, not round0's single value
    assert {r["s_per_step_median"] for r in probe_rows} == {"0.9", "0.93"}

    text = wtune.tune.build_full_run_prediction(ctx, best_config, nodes=1, all_results=all_results,
                                                 confirm_results=confirm_results)
    assert "source=measured" in text
    assert "no_probe_fallback" not in text


def test_sample_more_diverse_configs_takes_the_machine_rankers_first_half(monkeypatch, capsys):
    for k, v in (("HARDWARE_ID", "f1-grace"), ("CORES_PER_NODE", "144"), ("NUMA_PER_NODE", "2")):
        monkeypatch.setenv("WTUNE_" + k, v)
    ctx = _ctx(case_json=dict(MATTHEW_LIKE, prep_dir="/fake/matthew", e_we=400, e_sn=400),
               nodes=1, hardware_id="f1-grace", cores_per_node=144)
    fresh = wtune.tune.sample_more_diverse_configs(ctx, [], k=6, seed=1)
    assert len(fresh) == 6 and "ranker's first picks" in capsys.readouterr().out
    assert all(c["omp"] == 1 and not c["tile_strategy"] for c in fresh[:3])   # model-fitted layouts
    monkeypatch.setenv("WTUNE_SHORTLIST", "0")
    again = wtune.tune.sample_more_diverse_configs(ctx, [], k=6, seed=1)
    assert len(again) == 6 and "ranker" not in capsys.readouterr().out


def test_build_full_run_prediction_applies_drift_band_override_and_flags_measure_r_rad(tmp_path):
    assert wtune.tune.predict is not None, "wtune predict must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"),
               prediction_cfg=dict(io_target="nfs", drift_band_pct=9.5, measure_r_rad=True))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    text = wtune.tune.build_full_run_prediction(ctx, best_config, nodes=1, all_results=all_results,
                                                 confirm_results=None)

    assert "prediction.drift_band_pct=9.5 applied via calibration override" in text
    override_path = os.path.join(str(tmp_path / "out"), "calibration_override.json")
    assert os.path.exists(override_path)
    with open(override_path) as f:
        assert json.load(f)["drift_band_pct"] == 9.5
    assert "measure_r_rad requested" in text
    assert "NOT YET IMPLEMENTED" in text


def test_rank_predicted_entries_failed_prediction_ranks_last_and_is_flagged(tmp_path, monkeypatch):
    ctx = _ctx(out_dir=str(tmp_path / "out"))
    cfg_ok = dict(id="G_OK", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    cfg_bad = dict(id="G_BAD", ranks=48, omp=1, nx=6, ny=8, numtiles=1, dist="block")
    results = {
        "G_OK": _result("G_OK", 2.0, cfg_ok),     # far WORSE median s/step than G_BAD...
        "G_BAD": _result("G_BAD", 0.1, cfg_bad),  # ...but G_BAD's own prediction raises
    }

    def fake_build_prediction(namelist_path, case_dir, nodes, config, ledger_paths,
                               calibration_path, io_form_override, io_target, case_name=None,
                               iofields_dir=None, rsl_paths=None, include_io=False):
        if config["ranks"] == 48:
            raise RuntimeError("synthetic prediction failure")
        return {"total_s": 999.0, "range_low_s": 950.0, "range_high_s": 1050.0,
                "compute": {"r_rad_source": "fake_test_source"}}

    monkeypatch.setattr(wtune.tune.predict, "build_prediction", fake_build_prediction)

    predicted = wtune.tune.predict_candidates(ctx, nodes=1, results=results)
    assert predicted["G_BAD"]["pred_total_s"] is None
    assert "synthetic prediction failure" in predicted["G_BAD"]["pred_error"]
    assert predicted["G_OK"]["pred_error"] is None

    ranked = wtune.tune.rank_predicted_entries(predicted, anchor_cid=None)
    # the failed prediction ranks LAST despite its far better raw median s/step
    assert [r["cid"] for r in ranked] == ["G_OK", "G_BAD"]

    report = wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=results,
                                      confirm_results=None, anchor_cid=None)
    assert report["ranking"][-1]["id"] == "G_BAD"
    assert report["ranking"][-1]["pred_error"] and "synthetic prediction failure" \
        in report["ranking"][-1]["pred_error"]
    text = wtune.tune.render_report_text(report)
    assert "PREDICTION FAILED for G_BAD" in text


def _patched_load_config(raw):
    """A stand-in for wtune.config.load_config that skips real YAML parsing entirely -- resolves
    `raw` (a plain dict, the same shape a parsed YAML file would produce) through
    wtune.config.resolve_config directly. Lets the --config wiring in wtune.tune.run() (CLI > file
    > defaults precedence, search filtering, probe/prediction plumbing) be exercised without
    requiring PyYAML in this test environment -- only wtune/config.py's OWN load_config/CLI tests
    (test_wtune_config.py) need real YAML and are skipped there when PyYAML is absent."""
    def fake(path):
        cfg, errs = wtune.config.resolve_config(raw)
        if errs:
            raise wtune.config.ConfigError(errs)
        return cfg
    return fake


def _run_e2e_with_config(tmp_path, raw_config=None, extra_argv=(), monkeypatch=None):
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    profile = _profile([{"nodes": 1, "legal": True, "efficiency_E": 0.95}])
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                         lambda case, cores_per_node: (case_json, str(tmp_path / "namelist.input"),
                                                        profile))
    if raw_config is not None:
        monkeypatch.setattr(wtune.config, "load_config", _patched_load_config(raw_config))

    sbatch_runner, sbatch_calls, job_to_configs_path = make_fake_sbatch(start=30000)
    sacct_runner = make_fake_sacct({})
    scancel_runner, _scancel_calls = make_fake_scancel()
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path)

    def stop_immediately_proposer(state, batch_size, kappa=None):
        return dict(stopped=True, reason="test stub"), []

    argv = ["matthew", "--site", "scc", "--max-nodes", "13",
            "--out", str(tmp_path / "out"), "--ledger-dir", str(tmp_path / "ledger")]
    if raw_config is not None:
        argv += ["--config", str(tmp_path / "wtune.yaml")]   # path is never actually read (patched)
    argv += list(extra_argv)

    args = wtune.tune.build_arg_parser().parse_args(argv)
    report = wtune.tune.run(args, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
                             scancel_runner=scancel_runner, ledger_reader=ledger_reader,
                             proposer=stop_immediately_proposer, clock=FakeClock())
    return report, sbatch_calls


def test_config_probe_steps_and_confirm_reps_are_wired_through(tmp_path, monkeypatch):
    report, calls = _run_e2e_with_config(
        tmp_path,
        raw_config=dict(case="matthew", budget=dict(minutes=30),
                         probe=dict(steps=7, confirm_reps=4)),
        monkeypatch=monkeypatch)
    assert any("STEPS=7" in c for c in calls)
    assert any("REPS=4" in c for c in calls)


def test_config_search_omp_restriction_constrains_round0_and_confirm(tmp_path, monkeypatch):
    report, _calls = _run_e2e_with_config(
        tmp_path,
        raw_config=dict(case="matthew", budget=dict(minutes=30),
                         search=dict(omp=[2], numtiles=[1])),
        monkeypatch=monkeypatch)
    non_anchor_rows = [r for r in report["ranking"] if r["id"] != "ANCHOR"]
    assert non_anchor_rows   # the restriction did not eliminate everything
    assert all(r["omp"] == 2 for r in non_anchor_rows)
    assert all(r["numtiles"] == 1 for r in non_anchor_rows)


def test_run_raises_systemexit_with_every_violation_for_an_invalid_config(tmp_path, monkeypatch):
    monkeypatch.setattr(wtune.config, "load_config",
                         _patched_load_config(dict(case="matthew", search=dict(omp=[0]), bogus=1)))
    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit) as exc_info:
        wtune.tune.run(args)
    msg = str(exc_info.value)
    assert "search.omp" in msg
    assert "bogus" in msg


def test_run_rejects_case_dir_only_config_as_not_yet_runnable(tmp_path, monkeypatch):
    monkeypatch.setattr(wtune.config, "load_config",
                         _patched_load_config(dict(case_dir="/some/where", budget=dict(minutes=10))))
    args = wtune.tune.build_arg_parser().parse_args(
        ["--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit, match="case_dir"):
        wtune.tune.run(args)


def test_run_rejects_a_multi_value_budget_nodes_list(tmp_path, monkeypatch):
    monkeypatch.setattr(wtune.config, "load_config",
                         _patched_load_config(dict(case="matthew",
                                                    budget=dict(minutes=10, nodes=[1, 2, 4]))))
    args = wtune.tune.build_arg_parser().parse_args(
        ["--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit, match="not yet run"):
        wtune.tune.run(args)


# ---------------------------------------------------------------------------
# --nodes search: measured node sweep, objective choice, finalist search
# ---------------------------------------------------------------------------
def _interp(points, max_n=13):
    """{N: s/step} for N=1..max_n, linear between the given points, flat after the last."""
    ks = sorted(points)
    out = {}
    for n in range(1, max_n + 1):
        if n in points:
            out[n] = points[n]
        elif n > ks[-1]:
            out[n] = points[ks[-1]]
        else:
            lo = max(k for k in ks if k < n)
            hi = min(k for k in ks if k > n)
            out[n] = points[lo] + (points[hi] - points[lo]) * (n - lo) / float(hi - lo)
    return out


NONMONO_CURVE = _interp({1: 1.0, 2: 0.52, 4: 0.30, 8: 0.32, 13: 0.40})   # 8 slower than 4
ALL_NODES_PROFILE = _profile([{"nodes": n, "legal": True, "efficiency_E": 0.9}
                              for n in (1, 2, 4, 8, 13)])


def make_scaling_fakes(curve, start=7000, core_mult=None):
    """(sbatch_runner, ledger_reader, job_nodes): every job's configs get s = curve[N] / (1 +
    0.05*i), so ANCHOR (i=0) measures exactly curve[N]; job_nodes[job] = N parsed from `-N`."""
    counter = {"n": start}
    job_nodes, job_paths = {}, {}

    def sbatch(cmd_str):
        counter["n"] += 1
        j = str(counter["n"])
        job_nodes[j] = int(re.search(r"-N (\d+)", cmd_str).group(1))
        job_paths[j] = re.search(r"CONFIG_FILE=(\S+)", cmd_str).group(1)
        return "Submitted batch job %s\n" % j

    def reader(case, job_ids):
        rows = []
        for j in job_ids:
            with open(job_paths[j]) as f:
                cfgs = json.load(f)["configs"]
            for i, c in enumerate(cfgs):
                mult = 1.0
                if core_mult:       # cores used per node = ranks * omp / nodes
                    mult = core_mult(c["ranks"] * c["omp"] // job_nodes[j])
                for rep in (1, 2):
                    rows.append(dict(run_id="%s_%s_r%d_%s" % (case, c["id"], rep, j),
                                     s_per_step_median=str(mult * curve[job_nodes[j]]
                                                           / (1 + 0.05 * i))))
        return rows

    return sbatch, reader, job_nodes


def _run_cli(monkeypatch, tmp_path, argv, curve=NONMONO_CURVE, profile=ALL_NODES_PROFILE,
             total_steps=1000, grid=MATTHEW_LIKE, core_mult=None, sbatch_calls=None,
             preflight_runner=None, pending_nodes=(), on_sbatch=None):
    case_json = dict(grid, prep_dir=str(tmp_path / "matthew"))
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                        lambda case, cpn: (case_json, str(tmp_path / "namelist.input"), profile))
    monkeypatch.setattr(wtune.tune.context, "_total_steps", lambda nl: total_steps)
    sbatch, reader, job_nodes = make_scaling_fakes(curve, core_mult=core_mult)
    if sbatch_calls is not None:
        inner = sbatch

        def sbatch(cmd_str, _inner=inner):
            sbatch_calls.append(cmd_str)
            return _inner(cmd_str)
    states = {}                              # jobs at an N in pending_nodes never start
    if pending_nodes or on_sbatch:
        submit = sbatch

        def sbatch(cmd_str, _submit=submit):
            out = _submit(cmd_str)
            j = out.split()[-1]
            if job_nodes[j] in pending_nodes:
                states[j] = ("PENDING", "00:00:00")
            if on_sbatch:
                on_sbatch(cmd_str, job_nodes[j])
            return out
        measured = reader
        reader = lambda case, ids: measured(case, [j for j in ids if j not in states])  # noqa: E731
    scancel, scancel_calls = make_fake_scancel()

    def stop_proposer(state, batch_size, kappa=None):
        return dict(stopped=True, reason="test stub"), []

    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--site", "scc", "--out", str(tmp_path / "out"), "--steps", "5",
         "--ledger-dir", str(tmp_path / "ledger")] + argv)
    report = wtune.tune.run(args, sbatch_runner=sbatch, sacct_runner=make_fake_sacct(states),
                             scancel_runner=scancel, ledger_reader=reader,
                             proposer=stop_proposer, clock=FakeClock(),
                             preflight_runner=preflight_runner)
    return report, job_nodes, scancel_calls


ISOLATED_CURVE = {**NONMONO_CURVE, 7: 0.20}      # 7 alone is fast: outside the 3..5 bracket


def test_costly_untried_n_wait_for_the_time_the_finalists_leave(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wtune.tune.node_search, "EARLY_UNTRIED_FRAC", -1.0)  # never cheap enough
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path,
                                      ["-O2", "--minutes", "600", "--hotspot-fill", "off"],
                                      curve=ISOLATED_CURVE)
    ns = report["node_selection"]
    assert "N=7 is now the fastest ANCHOR" in capsys.readouterr().out
    assert ns["finalists"] == [4, 7] and report["nodes"] == 7
    sweep_jobs = set(ns["sweep_jobs"])
    assert 7 in {n for j, n in job_nodes.items() if j not in sweep_jobs}   # layouts tuned at 7


def test_queue_cap_in_the_sweep_keeps_the_n_measured(monkeypatch, tmp_path, capsys):
    report, job_nodes, cancelled = _run_cli(
        monkeypatch, tmp_path, ["-O2", "--minutes", "600", "--hotspot-fill", "off",
                                "--max-queue-hours", "1"], pending_nodes={8})
    out = capsys.readouterr().out
    assert "keeping the N measured so far" in out and "finalist N=" not in out
    assert report["nodes"] == 4 and not report.get("pipeline_aborted")
    assert report["pipeline"]["queue_cap_hit"]
    assert [job_nodes[j] for c in cancelled for j in c] == [8]
    assert job_nodes[max(job_nodes, key=int)] == 8                # nothing sent after the cap


def test_ctrl_c_in_the_sweep_keeps_the_n_measured_and_writes_the_report(monkeypatch, tmp_path,
                                                                        capsys):
    stop = wtune.tune.context.STOP
    monkeypatch.setitem(stop, "reason", None)

    def on_sbatch(cmd, nodes):                     # Ctrl-C right after the N=8 job is sent
        if nodes == 8:
            stop["reason"] = "stopped by Ctrl-C"
    report, job_nodes, cancelled = _run_cli(
        monkeypatch, tmp_path, ["-O2", "--minutes", "600", "--hotspot-fill", "off"],
        pending_nodes={8}, on_sbatch=on_sbatch)
    out = capsys.readouterr().out
    assert report["nodes"] == 4 and report["stopped"] == "stopped by Ctrl-C"
    assert report["nodes_note"].endswith("stopped by Ctrl-C -- ANCHOR config")
    assert [job_nodes[j] for c in cancelled for j in c] == [8]
    assert job_nodes[max(job_nodes, key=int)] == 8                # nothing sent after the stop
    assert "tune_report.json holds the best of what was measured" in out
    assert (tmp_path / "out" / "tune_report.json").exists()
    assert not (tmp_path / "out" / "tune.state.json").exists()


def test_latest_tune_report_skips_a_tune_stopped_before_any_result(tmp_path):
    import json as _json
    for name, doc in (("a", {"case": "c", "nodes": 4}),
                      ("b", {"case": "c", "nodes": None, "pipeline_aborted": True,
                             "stopped": "stopped by Ctrl-C"})):
        (tmp_path / "tune" / name).mkdir(parents=True)
        (tmp_path / "tune" / name / "tune_report.json").write_text(_json.dumps(doc))
        os.utime(tmp_path / "tune" / name / "tune_report.json", (0, {"a": 100, "b": 200}[name]))
    log = []
    got = wtune.project.latest_tune_report(str(tmp_path), "c", log.append)
    assert got == str(tmp_path / "tune" / "a" / "tune_report.json")
    assert log == ["tune report skipped: %s has no result (stopped by Ctrl-C)"
                   % (tmp_path / "tune" / "b" / "tune_report.json")]


def test_a_stop_while_layouts_are_measured_keeps_the_layouts(monkeypatch, tmp_path, capsys):
    stop = wtune.tune.context.STOP
    monkeypatch.setitem(stop, "reason", None)

    def on_sbatch(cmd, nodes):                     # stop once round0's first wave is in
        if "round0-topup" in cmd:
            stop["reason"] = "stopped by wtune stop"
    report, _jn, _sc = _run_cli(monkeypatch, tmp_path,
                                ["-O2", "--minutes", "600", "--hotspot-fill", "off"],
                                on_sbatch=on_sbatch)
    assert "keeping the" in capsys.readouterr().out
    assert report["stopped"] == "stopped by wtune stop"
    assert report["best"]["id"] != "ANCHOR"        # a measured layout, not the sweep's ANCHOR


def test_node_sweep_budget_skip_is_per_n(tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes({1: 0.01, 2: 0.01, 4: 0.01})
    # per-job estimate ~ 120 + 20*10 = 320 s at the fallback (one run) -> budget 400 s fits one
    # job per wave only while elapsed is 0; elapsed does not advance (fake clock), so all fit...
    ctx = _ctx(hotspot_fill=False, probe_cfg=dict(wtune.config.DEFAULTS["probe"], sweep_steps=None),
               sbatch_runner=sbatch, ledger_reader=reader, budget_s=400.0, max_nodes=13,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    ctx.clock.t = 100.0                # ...so advance the clock: 100 + 320 > 400 -> all skipped
    ctx.start_time = 0.0
    sw = wtune.tune.run_node_sweep(ctx, [1, 2, 4])
    assert job_nodes == {} and len(sw["skipped"]) == 3
    assert "exceed the budget" in sw["skipped"][0]["reason"]


def test_small_n_that_cannot_fit_are_skipped_and_the_largest_n_goes_first(tmp_path, monkeypatch,
                                                                           capsys):
    import wtune.tune.memory as mem
    # 300 GiB of grid on 100 GiB nodes, nothing per rank: N <= 2 is skipped up front
    monkeypatch.setattr(mem, "model_for", lambda case_json, node_gib: mem.MemoryModel(
        300 * mem.GIB / 1000.0, 100.0, 1000.0, 0.0))
    sbatch, reader, job_nodes = make_scaling_fakes(OPT6_CURVE)
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, max_nodes=13,
               case_json=dict(BIG_GRID, prep_dir="/fake/big"), hotspot_fill=False,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, list(range(1, 14)), budget_frac=1.0)
    out = capsys.readouterr().out
    assert "N=1 may not fit in memory; largest N first" in out
    assert job_nodes[min(job_nodes)] == 13                       # the first job is the largest N
    assert not {1, 2} & set(sw["measurements"]) and not {1, 2} & set(job_nodes.values())
    assert all("memory: predicted" in s["reason"] for s in sw["skipped"] if s["nodes"] in (1, 2))


def test_after_the_proxy_round0_runs_each_pick_twice_at_once_and_the_race_skips_them(
        tmp_path, capsys, monkeypatch):
    from wtune.tune import candidates, proxy_stage, rounds
    sbatch, reader, _jn = make_scaling_fakes({1: 0.30})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, proxy=True,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    a = candidates._sweep_anchor(ctx, 1)[0]
    picks = [a, dict(a, id="D0001", numtiles=2)]

    def select(c, configs):
        c.proxy_used = True
        return picks
    monkeypatch.setattr(proxy_stage, "select", select)
    job_ids, by_cid, rows = rounds.run_round0(ctx)
    out = capsys.readouterr().out
    assert "round0: 2 config(s) x 2 runs, one job each" in out and len(job_ids) == 4
    res = candidates.aggregate_by_cid(rows, by_cid)
    assert sorted(res) == ["ANCHOR", "D0001"] and all(r["n"] >= 2 for r in res.values())
    rounds._race_leaders(ctx, res, 0.3)
    assert "race: submitted" not in capsys.readouterr().out


def test_node_sweep_uses_fewer_cores_when_full_nodes_have_no_legal_decomposition(tmp_path):
    """f1-arm matthew 2026-10-14: 91x100 cuts into 90 ranks at most. With the thread cap at 1 a
    node count whose full nodes need more ranks was skipped (here N=2: 144), and on a 144-core
    node the whole tune measured one layout."""
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0, 2: 0.6})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, omp_cap=1,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, [1, 2], budget_frac=1.0)
    assert sorted(sw["measurements"]) == [1, 2] and sw["skipped"] == []
    assert sw["cores"] == {1: 72, 2: 45}                # N=2: 90 ranks, 45 on each node
    assert all(a["omp"] == 1 for a in sw["anchors"].values())          # the cap still holds
    assert (sw["anchors"][2]["ranks"], sw["anchors"][2]["nx"], sw["anchors"][2]["ny"]) == (90, 9, 10)
    # the cores/node sweep starts from those 45, not from the 72 of a node
    info, _skipped = wtune.tune.node_search.run_core_sweep(ctx, [2], sw, [72, 64, 48, 36])
    assert max(info[2]["anchors"]) == 45 and {40, 36, 30} <= set(info[2]["results"])
    assert info[2]["chosen"] == 45                      # all tie here: the fullest wins
    rows = wtune.tune.node_search.build_layout_table(sw, info, {}, ctx)["rows"]
    assert {(r["nodes"], r["cores_per_node"]) for r in rows} >= {(1, 72), (2, 45), (2, 36)}


def test_node_sweep_applies_chem_j_patch_guard(monkeypatch, tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0})
    monkeypatch.setattr(wtune.extensions, "chem_j_patch_reason",
                        lambda e_sn, ny, nx=None: "j-patch too narrow (test)")
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=36000.0, chem_e_sn=[100],
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, [1])
    assert job_nodes == {} and "j-patch too narrow" in sw["skipped"][0]["reason"]


def test_chem_guard_keeps_the_widest_j_patch_when_it_would_leave_nothing(capsys):
    ctx = _ctx(chem_e_sn=[100])             # CHEM_MIN_J_PATCH = 21 rows: 100/8 = 12 rows is below it
    configs = [dict(id="G0001", ranks=72, omp=1, nx=9, ny=8, numtiles=1, dist="block"),
               dict(id="G0002", ranks=72, omp=1, nx=6, ny=12, numtiles=1, dist="block"),
               dict(id="G0003", ranks=72, omp=1, nx=4, ny=18, numtiles=1, dist="block")]
    kept = wtune.tune._apply_search_filter(ctx, configs)
    assert [c["id"] for c in kept] == ["G0001"]                      # ny=8: the widest j-patch
    assert "every candidate is below the WRF-Chem j-patch guard" in capsys.readouterr().out
    ok = dict(id="G0004", ranks=72, omp=1, nx=18, ny=4, numtiles=1, dist="block")   # 25 rows: passes
    assert [c["id"] for c in wtune.tune._apply_search_filter(ctx, configs + [ok])] == ["G0004"]


# ---------------------------------------------------------------------------
# coarse-to-fine node search, cores-per-node dimension, MaxRSS
# ---------------------------------------------------------------------------
BIG_GRID = dict(e_we=400, e_sn=400, e_vert=45)       # every N in 1..13 has a legal layout
OPT6_CURVE = _interp({1: 1.0, 2: 0.55, 4: 0.33, 5: 0.29, 6: 0.27, 7: 0.28, 8: 0.30, 13: 0.50})


def test_cores_option_disables_dimension_and_validates(monkeypatch, tmp_path):
    report, job_nodes, _sc = _run_cli(
        monkeypatch, tmp_path, ["--minutes", "6000", "--cores-per-node-options", "72"],
        grid=BIG_GRID)
    assert report["node_selection"]["core_tables"] == {}
    assert not (tmp_path / "out" / "cores-n4-c64").exists()     # no cores job directories
    with pytest.raises(SystemExit, match="cores-per-node-options"):
        _run_cli(monkeypatch, tmp_path, ["--minutes", "30", "--cores-per-node-options", "99"])


def test_scc_site_launch_places_ranks_per_node_for_an_undersubscribed_run():
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    site = os.path.join(here, "campaign", "sites", "scc.sh")
    # N=2, c=64, omp=1 -> 128 ranks, 64 per node, one core each; cores 64-71 stay idle
    out = subprocess.run(
        ["bash", "-c", 'source "%s"; SLURM_JOB_NUM_NODES=2 site_launch 128 64 1 block; '
                       'echo "${SITE_LAUNCH_CMD[@]}"' % site],
        capture_output=True, text=True, check=True).stdout.strip()
    assert out == "mpirun -np 128 --map-by ppr:64:node:PE=1 --bind-to core ./wrf.exe"


def test_maxrss_is_read_from_step_rows_and_never_invented():
    ctx = _ctx(sacct_runner=lambda cmd: "1001|\n1001.batch|6044K\n1001.0|2048M\n1002|\n1002.batch|")
    got = wtune.tune._maxrss_mb_by_job(ctx, ["1001", "1002"])
    assert got["1001"] == pytest.approx(2048.0)
    assert got["1002"] is None
    seen = []
    ctx.sacct_runner = lambda cmd: seen.append(cmd) or ""
    assert wtune.tune._maxrss_mb_by_job(ctx, ["7"]) == {"7": None}     # empty sacct -> None
    assert "--format=JobID,MaxRSS" in seen[0] and "-X" not in seen[0]


# ---------------------------------------------------------------------------
# --pipeline full (decision 0059): fakes for the rebuild/0055 measurement path + physics advice
# ---------------------------------------------------------------------------
import datetime as _dt
import types as _types

_PT_S = {0: 0.30, 1: 0.40, 2: 0.33}          # s_normal of the winner per rebuilt point (1 = costliest)


class _Cands(list):
    note = "fake"
    skipped = []


class _PipeEnv:
    """Fake rebuild/prediction/physics layer with an event log shared with the fake sbatch."""

    def __init__(self, monkeypatch, tmp_path, clock, minutes, *, extension=False, physics=True,
                 runner_ok=True, phase2_cost_s=0.0, bump_after_first_read=0.0, steps=32,
                 n_points=3, planner=None, missing_env=()):
        self.events = []
        self.measure_modes = []
        self.planner_calls = []
        self.measure_cfgs = []
        self.tmp_path = tmp_path
        self.clock = clock
        self.phase2_cost_s = phase2_cost_s
        self.bump = bump_after_first_read
        self.physics_calls = []
        t0 = _dt.datetime(2026, 1, 1, 0, 0)
        self.pts = [_types.SimpleNamespace(at=t0 + _dt.timedelta(hours=h), reason="sunlit")
                    for h in (1, 3, 5, 7, 9)[:n_points]]
        ev, mp = self.events, monkeypatch
        mp.setattr(wtune.tune.context, "_namelist_dt", lambda ctx: 60.0)
        mp.setattr(wtune.tune.points, "plan_predicted_points",
                   lambda ctx, spinup: (None, t0, t0 + _dt.timedelta(hours=6), 60.0, 10,
                                        list(self.pts), []))

        def submit(ctx, points, anchor, *a, **k):
            ev.append("rebuild_submit")
            ids = ["90%02d" % i for i in range(len(points))]
            ctx.owned_jobs.extend(ids)
            return dict(job_ids=ids, points=list(points), done=[None] * len(points))

        def finish(ctx, handle):
            ev.append("rebuild_finish")
            return [dict(index=i, point=p, case_id="matthewp%d" % i, T=t0, cases_json="x",
                         prep_dir="x", template="x", steps=steps, spinup_min=20, stepra=10, dt=60.0,
                         plan_dir="x", job_id="9001") for i, p in enumerate(handle["points"])]

        def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
            ev.append(("measure", ctx.nodes, [c["id"] for c in configs],
                       [rp["index"] for rp in rebuilt]))
            self.measure_cfgs.append((anchor, list(configs)))
            self.measure_modes.append(k.get("mode"))
            ctx.owned_jobs.append("8%03d" % len(ev))
            jobs = ["8%03d" % (len(ev) + i) for i in range(len(rebuilt))]
            # the runner keeps the window frames of configs asking for them (mid + end)
            for cfg in [anchor, dict(anchor, id="ANCHOREND")] + list(configs):
                if cfg.get("window_frame") == "mid_end":
                    for rp, jid in zip(rebuilt, jobs):
                        d = os.path.join(ctx.ledger_dir, "%s_%s_r1_%s"
                                         % (rp["case_id"], "ANCHOR" if cfg is anchor else cfg["id"], jid))
                        os.makedirs(d, exist_ok=True)
                        for t in ("mid", "end"):
                            open(os.path.join(d, "wrfout_d01_%s.nc" % t), "w").write("x")
            clock.t += self.phase2_cost_s
            ctx.add_running(self.phase2_cost_s)          # the budget counts only RUNNING time
            phys = any(c["id"].startswith("PH") for c in configs)
            meas = {}
            for cid in ["ANCHOR"] + [c["id"] for c in configs]:
                f = 0.9 if cid.startswith("PH") else 1.0
                meas[cid] = [_types.SimpleNamespace(at=rp["point"].at,
                                                    s_normal=_PT_S.get(rp["index"], 0.31) * f)
                             for rp in rebuilt]
            return meas, [], jobs

        def rank(meas, run_start, run_end, dt, stepra, cal, sp, case_dir, nl, ranks, nodes):
            out = []
            for cid, pms in meas.items():
                tot = sum(pm.s_normal for pm in pms) * 1000.0
                out.append(dict(cid=cid, pred_total_s=tot, pred_low_s=tot * 0.95,
                                pred_high_s=tot * 1.05,
                                result=dict(points=[1], compute_total_s=tot, init_s=10.0,
                                            total_s=tot + 10, range_low_s=tot, range_high_s=tot,
                                            drift_pct=1.0, r_night=1.0, r_day=3.9, r_source="fake")))
            return sorted(out, key=lambda e: e["pred_total_s"])

        pcalls = self.planner_calls

        def plan_points(case_types, run_start, run_end, precision_min, T_est_s, indicators=None,
                        best_track=None, window_s=0.0, min_gap_s=1800.0, max_points=6):
            # R's real signature (wtune.cost_drivers.plan_points_for_case)
            n = len(pcalls)
            pcalls.append(dict(types=list(case_types), precision_min=precision_min, T=T_est_s,
                               call=n, window_s=window_s, run_start=run_start, run_end=run_end))
            if planner is not None:
                pts, band = planner(list(self.pts), T_est_s, n)
            else:
                pts, band = list(self.pts), 0.07
            return dict(points=[(run_start, "t0")] + [(p.at, "r") for p in pts],
                        band_pct=band * 100.0, rationale=["fake case-type plan"],
                        needs_rebuild=[p.at for p in pts], confidence="normal",
                        meets_precision=True, types=list(case_types))

        mp.setattr(wtune.tune.prediction, "cost_drivers",
                   _types.SimpleNamespace(plan_points_for_case=plan_points))
        mp.setattr(wtune.tune.prediction, "_case_types_and_track", lambda ctx: (["common"], None))
        mp.setattr(wtune.tune.points, "submit_rebuilt_points", submit)
        mp.setattr(wtune.tune.points, "finish_rebuilt_points", finish)
        mp.setattr(wtune.tune.points, "measure_candidates_at_points", measure)
        mp.setattr(wtune.tune.points, "rank_by_predicted_total", rank)
        # default phase 2 (2026-10-27): predict-rebuild's procedure at the phase-1 winner
        from wtune import probe_plan as _pp_mod, rebuild_predict as _rp_mod
        mp.setattr(_rp_mod, "plan",
                   lambda case_dir, forcing, spinup_min, max_points, namelist_path=None, **k:
                   (None, t0, t0 + _dt.timedelta(hours=6), 60.0, 10, list(self.pts)[:max_points],
                    []))
        self.rebuild_winner = []

        def rebuild_winner(ctx, points, best, nodes, case_dir, out_dir, run_start, **k):
            ev.append("rebuild_winner")
            self.rebuild_winner.append(dict(best=best, nodes=nodes, n=len(points),
                                            measure_steps=k.get("measure_steps"),
                                            forcing=k.get("forcing"),
                                            spinup=k.get("spinup_min")))
            return ([_types.SimpleNamespace(at=p.at, s_normal=0.3) for p in points],
                    ",".join("77%02d" % i for i in range(len(points))))
        mp.setattr(wtune.tune.pipeline, "run_rebuild_points", rebuild_winner)
        mp.setattr(wtune.tune.prediction, "rebuild_env_missing", lambda: list(missing_env))
        mp.setattr(_pp_mod, "combine",
                   lambda meas, *a, **k: dict(compute_total_s=1000.0, init_s=20.0, init_note="t",
                                              total_s=1020.0, range_low_s=950.0,
                                              range_high_s=1090.0))
        mp.setattr(wtune.tune.prediction, "_prediction_context",
                   lambda ctx: (None, t0, t0 + _dt.timedelta(hours=6), 60.0, 10, {}, [], "/x"))
        mp.setattr(wtune.tune.physics_phase, "physics_runner_supported", lambda script: runner_ok)
        mp.setattr(wtune.tune.physics_phase, "build_physics_window_namelists",
                   lambda ctx, cands, rsub, anchor: (
                       dict(anchor, physics_namelists={}),
                       [dict(anchor, id=cd["id"], physics_namelists={}) for cd in cands]))
        calls = self.physics_calls

        def plan(path, *, max_candidates=6, case_name=None):
            calls.append(("plan", max_candidates, case_name))
            comp = [dict(id=i, label=i, rationale="r", restart_compatible=True,
                         prior_gain_pct=5.0) for i in ("PH1", "PH2", "PH3", "PH4")][:max_candidates]
            est = [dict(id=i, label=i, rationale="r", restart_compatible=False,
                        prior_gain_pct=12.0) for i in ("EO1", "EO2")]
            return _Cands(comp + est)

        def fidelity(b, c, **kw):
            assert isinstance(b, list) and isinstance(c, list) and len(b) == len(c) == 2
            calls.append(("fidelity", kw))
            return dict(passed=True, worst=[dict(name="T2", rel_rms=0.01)])

        def table(cands, speed, fid):
            return [dict(id=c["id"], label=c["label"], measured=c["id"] in speed,
                         predicted_saving_pct=(speed[c["id"]]["saving_pct"] if c["id"] in speed
                                               else None),
                         prior_saving_pct=c.get("prior_gain_pct"),
                         fidelity="pass" if fid.get(c["id"]) else "not run",
                         recommendation=("consider" if c["id"] in speed else
                                         "estimate only — needs a cold start to measure"))
                    for c in cands]

        api = _types.SimpleNamespace(
            plan_candidates=plan,
            weights_from_combine=lambda pms, *a, **k: {pm.at.isoformat(): 1.0 for pm in pms},
            predicted_speedup=lambda b, c, w: dict(saving_pct=100.0 * (1 - sum(
                w.get(k, 0) * c[k] for k in c) / sum(w.get(k, 0) * b[k] for k in c)),
                ratio_min=0.9, ratio_max=0.9),
            fidelity_verdict=fidelity, advice_table=table)
        api.PROGNOSTIC_FIELDS = ("T", "U", "V", "QVAPOR", "PSFC", "T2", "Q2", "U10", "V10")
        mp.setattr(wtune.tune.physics_phase, "physics_advice", api if physics else None)
        if extension:
            mp.setattr(wtune.extensions, "omp_cap_for_namelist", lambda p: 1)


def make_pending_sacct(pending_polls):
    """sacct fake: every job reports PENDING for its first `pending_polls` state polls, then
    COMPLETED (None = never completes). Other sacct formats (MaxRSS, Submit/Start/End) are empty."""
    seen = {}

    def runner(cmd):
        fmt = [c for c in cmd if c.startswith("--format=")]
        if not fmt or fmt[0] != "--format=JobID,State,Elapsed":
            return ""
        lines = []
        for jid in cmd[cmd.index("-j") + 1].split(","):
            seen[jid] = seen.get(jid, 0) + 1
            pend = pending_polls is None or seen[jid] <= pending_polls
            lines.append("%s|%s|%s" % (jid, "PENDING" if pend else "COMPLETED", "00:00:00"))
        return "\n".join(lines)

    return runner


def _pipe_run(monkeypatch, tmp_path, minutes=None, extra_args=(), sacct=None, scancel=None,
              preflight_runner=None, short_run_guard=False, total_steps=None, level=3,
              cold=None, physics_advice=True, full_cap=8, grid=None, drop=None, **kw):
    clock = FakeClock()
    if total_steps is None:      # a long run under a --minutes cap
        total_steps = 10 ** 6 if minutes is not None else 1000
    env = _PipeEnv(monkeypatch, tmp_path, clock, minutes, **kw)
    case_json = dict(grid or BIG_GRID, prep_dir=str(tmp_path / "matthew"))
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                        lambda case, cpn: (case_json, str(tmp_path / "namelist.input"),
                                           ALL_NODES_PROFILE))
    monkeypatch.setattr(wtune.tune.context, "_total_steps", lambda nl: total_steps)
    if cold is not None:       # (whole-run seconds, s/step at the proxy node count)
        monkeypatch.setattr(wtune.tune.prediction, "cold_start_prediction",
                            lambda ctx, nodes, cfg, steps: (cold[0], cold[1], "fake cold start"))
    sbatch, reader, _jn = make_scaling_fakes(NONMONO_CURVE)
    calls = {"n": 0}

    def logging_sbatch(cmd):
        env.events.append("sbatch")
        return sbatch(cmd)

    def logging_reader(case, job_ids):
        rows = reader(case, job_ids)
        if drop is not None:         # runs that "fail": their ledger rows never appear
            rows = [r for r in rows if not drop(wtune.tune.parse_run_id(r["run_id"])["cid"])]
        return rows

    if env.bump:        # the first phase-1 wait "runs" for bump x B of job running time
        real_wait = wtune.tune.wait_for_jobs
        waited = {"n": 0}

        def bumping_wait(ctx, job_ids, stage, *a, **k):
            out = real_wait(ctx, job_ids, stage, *a, **k)
            waited["n"] += 1
            if waited["n"] == 1 and stage == "node-sweep":
                ctx.add_running(env.bump * minutes * 60.0)
            return out

        monkeypatch.setattr(wtune.tune.jobs, "wait_for_jobs", bumping_wait)

    def stop_proposer(state, batch_size, kappa=None):
        return dict(stopped=True, reason="test stub"), []

    argv = ["matthew", "--site", "scc", "--out", str(tmp_path / "out"), "--steps", "5",
            "--ledger-dir", str(tmp_path / "ledger"), "-O%d" % level]
    if short_run_guard:              # opt-in guard (OFF by default)
        argv.append("--stop-if-short")
    if physics_advice and level >= 3:
        argv.append("--physics-advice")
    real_full = wtune.tune.enumerate_full_space

    def capped_full_space(ctx):      # the real enumeration is hundreds of layouts: keep tests fast
        configs, info = real_full(ctx)
        return (configs if full_cap is None else configs[:full_cap]), info

    monkeypatch.setattr(wtune.tune.candidates, "enumerate_full_space", capped_full_space)
    if minutes is not None:
        argv += ["--minutes", str(minutes)]
    args = wtune.tune.build_arg_parser().parse_args(argv + list(extra_args))
    scancel_calls = []
    if scancel is None:
        def scancel(job_ids):
            scancel_calls.append(list(job_ids))
    env.scancel_calls = scancel_calls
    report = wtune.tune.run(args, sbatch_runner=logging_sbatch,
                            sacct_runner=sacct or make_fake_sacct({}), scancel_runner=scancel,
                            ledger_reader=logging_reader, proposer=stop_proposer, clock=clock,
                            preflight_runner=preflight_runner)
    return report, env


def _status(report):
    return {p["name"]: p["status"] for p in report["pipeline"]["phases"]}


def test_pipeline_full_phase3_truncated_when_phase2_uses_the_budget(monkeypatch, tmp_path):
    minutes = 6000
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=minutes)
    base = report["pipeline"]["phases"][2]["end_s"]
    # phase 2's measurement runs until 60 s before B: phase 3 cannot fit its estimate
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=minutes,
                            phase2_cost_s=minutes * 60.0 - 60.0 - base)
    ph = _status(report)
    assert ph["predict"].startswith(("ok", "partial"))
    assert ph["physics"].startswith("not run (budget)")
    assert report["pipeline"]["physics_advice"] is None
    assert report["pipeline"]["prediction"] is not None


def test_physics_runner_supported_reads_the_campaign_files(tmp_path):
    (tmp_path / "strict_campaign.sbatch").write_text("echo nothing\n")
    (tmp_path / "configs.py").write_text("# no marker\n")
    script = str(tmp_path / "strict_campaign.sbatch")
    assert wtune.tune.physics_runner_supported(script) is False
    (tmp_path / "configs.py").write_text("cfg.get('physics_namelists')\n")
    assert wtune.tune.physics_runner_supported(script) is True


# ---------------------------------------------------------------------------
# node health pre-flight
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clean_exclude_env(monkeypatch):
    # the pre-flight publishes its failures in a process-wide env var: keep tests independent
    monkeypatch.delenv("WTUNE_PREFLIGHT_EXCLUDE", raising=False)
    monkeypatch.delenv("WTUNE_SBATCH_EXCLUDE", raising=False)
    yield
    os.environ.pop("WTUNE_PREFLIGHT_EXCLUDE", None)


def make_health_runner(states, bad=None, log=None):
    """sinfo + srun + scancel fake. states {node: state}; bad {node: 'timeout' | 'srun: error:
    ...' (rc 1, from srun) | stderr text (rc 3, from the probe script)}."""
    bad = bad or {}
    log = log if log is not None else []

    def runner(argv, timeout):
        log.append(argv)
        if argv[0] == "sinfo":
            return 0, "\n".join("%s %s" % kv for kv in list(states.items()) + [("head", "idle")])
        if argv[0] == "scancel":
            return 0, ""
        n = argv[argv.index("-w") + 1]
        b = bad.get(n)
        if b == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        if b and b.startswith("srun: error:"):
            return 1, b
        return (3, b) if b else (0, "ok")

    return runner, log


def _preflight(runner, requested=13, **kw):
    return wtune.tune.node_preflight(runner, case_dir="/case", src_dir="/src",
                                     wrf_exe="/wrf/main/wrf.exe", requested=requested, **kw)


def test_preflight_timeout_fails_an_idle_node_but_not_a_busy_one():
    states = {"work2": "idle", "work3": "allocated", "work4": "mixed"}
    runner, _log = make_health_runner(states, bad={n: "timeout" for n in states})
    h = _preflight(runner, requested=13, timeout_s=7)
    assert h["nodes"]["work2"]["reason"] == "probe timed out twice (7s each)" and not h["nodes"]["work2"]["ok"]
    # busy: unprobed. A mixed node need not answer either: on f1-arm (shared nodes, busy queue)
    # every mixed node timed out and got excluded, leaving only the nodes that were taken.
    for n in ("work3", "work4"):
        assert h["nodes"][n]["ok"] and h["nodes"][n]["probed"] is False
    assert h["failed"] == ["work2"] and h["healthy"] == 2


def test_preflight_reports_a_configuration_error_as_such():
    """dbbf82d review: 'Unable to allocate resources: Invalid account ...' is a configuration
    error, not a busy node; failing the node instead excluded every node of a partition that a
    later sbatch falls back to (5dac859 review)."""
    runner, _log = make_health_runner({"work2": "idle"}, bad={
        "work2": "srun: error: Unable to allocate resources: Invalid account or "
                 "account/partition combination specified"})
    h = _preflight(runner, requested=13)
    w = h["nodes"]["work2"]
    assert h["failed"] == [] and w["ok"] and w["probed"] is False
    assert w["reason"].startswith("scheduler configuration error, not probed: srun: error:")


def test_preflight_merges_with_the_user_exclude_in_every_sbatch(monkeypatch):
    monkeypatch.setenv("WTUNE_SBATCH_EXCLUDE", "work9,work2")
    runner, _log = make_health_runner({"work2": "idle", "work3": "idle", "work4": "idle",
                                       "work9": "idle"}, bad={"work3": "no $HOME"})
    h = _preflight(runner, requested=13)
    assert h["user_exclude"] == ["work9", "work2"]
    assert h["nodes"]["work9"]["reason"] == "excluded by WTUNE_SBATCH_EXCLUDE"
    assert h["healthy"] == 1 and h["max_nodes_effective"] == 1   # only work4
    assert h["exclusions"] == ["head", "work9", "work2", "work3"]  # merged, de-duplicated, ordered
    assert wtune.tune.sweep.exclude_arg() == "--exclude=head,work9,work2,work3"
    cmd = wtune.tune.sweep.build_sbatch_command("c", 1, 72, "00:10:00", 600, 1, 5, "/x.json", "/o")
    assert "--exclude=head,work9,work2,work3" in cmd
    # the coldstart sbatch renderer reads the same merged set
    from wtune.tools import coldstart_sample
    plan = {"windows": [{"t": "2019-11-26_15:00:00", "reason": "t0"}], "nodes": 1,
            "config": {"ranks": 72, "omp": 1, "nx": 9, "ny": 8, "numtiles": 1}}
    assert "#SBATCH --exclude=head,work9,work2,work3\n" in coldstart_sample.render_run_sbatch(
        plan, site="scc")


def test_run_caps_max_nodes_lists_skipped_sweep_sizes_and_reports_node_health(
        monkeypatch, tmp_path, capsys):
    states = {"work%d" % i: "idle" for i in range(1, 14)}
    states["work1"] = "down*"
    runner, _log = make_health_runner(states, bad={"work2": "no $HOME", "work10": "timeout"})
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "6000"],
                                      grid=BIG_GRID, preflight_runner=runner)
    nh = report["node_health"]
    assert nh["healthy"] == 10 and nh["max_nodes_requested"] == 13 and nh["max_nodes_effective"] == 10
    assert nh["exclusions"] == ["head", "work1", "work10", "work2"]
    assert nh["nodes"]["work2"]["reason"].startswith("probe failed rc=3")
    skipped = {s["nodes"]: s["reason"] for s in report["node_selection"]["skipped"]}
    for n in (11, 12, 13):
        assert skipped[n] == "skipped (only 10 healthy nodes)"
    assert max(job_nodes.values()) <= 10                          # nothing above the healthy count
    out = capsys.readouterr().out
    assert "node health pre-flight" in out and "work2" in out
    with open(tmp_path / "out" / "tune_report.json") as f:
        assert json.load(f)["node_health"]["max_nodes_effective"] == 10


def test_o2_estimate_follows_the_proxy_rounds():
    from wtune.tune import estimates as E
    kw = dict(s1=0.1261, steps_probe=30, startup_s=120.0, max_nodes=10)
    real, prox = wtune.tune.layout_estimates(proxy=False, **kw), wtune.tune.layout_estimates(**kw)
    # proxy runs: PROXY_MAX_STEPS = 8 steps; the one-node pool at one crop position, leaders at
    # PROXY_POSITIONS = 2; one job per layout, up to 4 x max_nodes = 40 jobs at once
    st = lambda n: E.EST_PROXY_RUN_START_S + 8 * 0.1261 / n                            # noqa: E731
    one = E.EST_PROXY_RUN_START_S + 8 * 0.1261 * sum(1.0 / n for n in range(1, 11)) / 10
    band = E.EST_PROXY_RUN_START_S + 8 * 0.1261 * sum(1.0 / n for n in range(4, 11)) / 7
    # 60 pool layouts on 40 jobs -> 2 rounds; 21 band leaders (3 x 7 N > 3) on min(40 // 3, 21) =
    # 13 jobs -> 2 rounds of 2 positions; the 3 leaders of N=2 and of N=3 on their own nodes, one
    # round; no real probe after the proxy
    assert prox[2] == pytest.approx(2 * one + 2 * 2 * band + st(2) + st(3))
    assert prox[2] < real[2]                       # no real t0 sweep before it
    assert prox[3] == real[3]                      # O3 enumerates the full space: no proxy
    # 2 time points: every band leader 4 runs, every whole-domain leader 3
    assert E.proxy_joint_s(120.0, 30, 0.1261, 10, time_points=2) == pytest.approx(
        2 * one + 2 * 4 * band + 3 * (st(2) + st(3)))
    # O1: one leader per N -- 7 band leaders on 13 jobs, one round of 2 positions
    assert prox[1] == pytest.approx(2 * one + 2 * band + st(2) + st(3))
    # --proxy-max-nodes 8: 60 layouts on 8 nodes -> 8 rounds; 21 band leaders on 8 // 3 = 2 jobs ->
    # 11 rounds of 2 positions; N=2's 3 leaders on 4 jobs -> 1 round, N=3's on 2 jobs -> 2
    assert E.proxy_joint_s(120.0, 30, 0.1261, 10, node_cap=8) == pytest.approx(
        8 * one + 22 * band + st(2) + 2 * st(3))
    # max_nodes 3: 18 layouts on 12 jobs -> 2 rounds, no band round, the whole-domain leaders of
    # N=2 and N=3
    assert E.proxy_joint_s(120.0, 30, 0.1261, 3) == pytest.approx(
        2 * (E.EST_PROXY_RUN_START_S + 8 * 0.1261 * (1 + 1 / 2 + 1 / 3) / 3) + st(2) + st(3))
    # round0 at N=3: 18 layouts on 12 jobs -> 2 rounds, no band round, 4 real probes in one slot
    assert E.proxy_round0_s(120.0, 30, 0.1261 / 3, 3, 3) == pytest.approx(2 * st(3) + 4 * (120.0 + 30 * 0.1261 / 3))
    # round0 at N=8: 18 layouts on 18 jobs, 4 band leaders x (2 positions + 1 time point) in one
    # round, 4 real probes in one slot
    assert E.proxy_round0_s(120.0, 30, 0.1261 / 8, 8, 8, time_points=1) == pytest.approx(
        4 * st(8) + 4 * (120.0 + 30 * 0.1261 / 8))


def _planner_ctx():
    return _ctx()


def _hourly_points(n=48, t0=_dt.datetime(2026, 1, 1)):
    return [dict(index=i, point=_types.SimpleNamespace(at=t0 + _dt.timedelta(hours=i), reason="r"))
            for i in range(n)]


def test_plan_prediction_points_calls_the_REAL_planner_common_case_needs_t0_only(monkeypatch):
    from wtune import cost_drivers as real                     # the real module, not a stub
    monkeypatch.setattr(wtune.tune.prediction, "cost_drivers", real)
    monkeypatch.setattr(wtune.tune.prediction, "_case_types_and_track", lambda ctx: (["common"], None))
    t0 = _dt.datetime(2026, 1, 1)
    wrapped = _hourly_points(30, t0)
    r = wtune.tune.plan_prediction_points(_planner_ctx(), wrapped, 15.0, 25 * 60.0, False,
                                          run_start=t0, run_end=t0 + _dt.timedelta(hours=24),
                                          window_s=3600.0)
    assert r["source"] == "wtune.cost_drivers"                 # NOT the silent fallback table
    assert r["selected"] == []                                  # needs_rebuild == []: t0 only
    assert r["band"] == pytest.approx(0.15) and r["meets_precision"] is True
    assert [p[1] for p in r["points"]] == ["t0"] and r["types"] == ["common"]
    assert "t0 only" in r["method"]


def test_plan_prediction_points_falls_back_loudly_on_a_real_error(monkeypatch, capsys):
    from wtune import cost_drivers as real
    monkeypatch.setattr(wtune.tune.prediction, "cost_drivers", real)

    def boom(ctx):
        raise RuntimeError("no case")

    monkeypatch.setattr(wtune.tune.prediction, "_case_types_and_track", boom)
    t0 = _dt.datetime(2026, 1, 1)
    r = wtune.tune.plan_prediction_points(_planner_ctx(), _hourly_points(5, t0), 15.0, 25 * 60.0,
                                          False, run_start=t0,
                                          run_end=t0 + _dt.timedelta(hours=24))
    assert r["source"] == "fallback band table"
    assert "FAILED (RuntimeError: no case)" in capsys.readouterr().out
    # no run window (planning failed upstream): also the fallback, never a TypeError
    r = wtune.tune.plan_prediction_points(_planner_ctx(), _hourly_points(5, t0), 15.0, 25 * 60.0,
                                          False)
    assert r["source"] == "fallback band table"


def test_step_down_after_the_first_measurement_wave(monkeypatch, tmp_path, capsys):
    # cold start is big (no step-down up front); the first wave shows 1000 steps x 0.3 s = 7 min
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=None, level=3, short_run_guard=True,
                            extra_args=("--hotspot-fill", "off"))
    pipe = report["pipeline"]
    sd = pipe["level"]["stepped_down"]
    assert sd and sd[-1]["when"] == "after first wave" and sd[-1]["to"] == 1
    assert sd[-1]["message"].startswith("模擬本身只要約 7 分鐘") and "已自動降為 O1" in sd[-1]["message"]
    assert pipe["level"]["effective"] == 1 and "STEPPED DOWN" in capsys.readouterr().out
    table = [r["nodes"] for r in report["node_selection"]["table"]]
    assert 3 not in table and 5 not in table                      # no refinement at O1
    assert report["node_selection"]["finalists"] == []             # no finalist search
    assert pipe["prediction"] is not None                          # the prediction still runs
    # surplus rebuild jobs are cancelled only when the planner needs fewer points; here it asks
    # for all of them, so the winner is predicted at every point
    assert not env.scancel_calls


def test_add_run_length_io_without_a_model_keeps_the_total():
    pred = dict(predicted_total_s=1000.0, predicted_range_s=[900.0, 1100.0],
                candidates=[dict(predicted_total_s=1000.0)])
    wtune.tune.add_run_length_io(pred, None, 880.0, 120.0, 0.10)
    assert pred["predicted_total_s"] == 1000.0 and pred["predicted_range_s"] == [900.0, 1100.0]
    assert pred["candidates"][0]["predicted_total_s"] == 1000.0
    assert "NOT included" in pred["breakdown"] and pred["segment_note"] is None


def test_layout_report_table_winner_runner_up_noise_and_untried(monkeypatch, tmp_path, capsys):
    # every 7th enumerated layout "fails" (no ledger row): it must show up as NOT TRIED
    # the fake times a layout by its place in the job, so a second run elsewhere would differ
    monkeypatch.setitem(wtune.config.DEFAULTS["probe"], "sweep_reps", 2)
    report, env = _pipe_run(monkeypatch, tmp_path, full_cap=None, grid=MATTHEW_LIKE,
                            physics_advice=False,
                            drop=lambda cid: cid.startswith("D") and int(cid[1:]) % 7 == 0)
    ns = report["node_selection"]
    rows = ns["layouts"]
    for k in ("nodes", "cores_per_node", "ranks", "omp", "nx", "ny", "numtiles", "median", "spread",
              "reps", "source"):
        assert all(k in r for r in rows), k
    meds = [r["median"] for r in rows]
    assert meds == sorted(meds)                                   # best first
    w, ru, cmp_ = ns["winner"], ns["runner_up"], ns["winner_vs_runner_up"]
    assert w["median"] <= ru["median"]
    assert (w["nodes"], w["ranks"], w["omp"], w["nx"], w["ny"], w["numtiles"]) != (
        ru["nodes"], ru["ranks"], ru["omp"], ru["nx"], ru["ny"], ru["numtiles"])
    assert cmp_["gap"] == pytest.approx((ru["median"] - w["median"]) / w["median"])
    assert cmp_["noise"] == 0.03 and cmp_["beyond_noise"] == (cmp_["gap"] > 0.03)
    untried = ns["untried"]
    assert any(u["reason"].startswith("no measurement") for u in untried)
    assert any(u["reason"].startswith("not measured") or "no legal decomposition" in u["reason"]
               for u in untried)                                   # unmeasured / illegal N too
    out = capsys.readouterr().out
    assert "--- measured layouts" in out and "WINNER:" in out and "RUNNER-UP:" in out
    assert "winner vs runner-up" in out and "NOT TRIED:" in out and "no measurement" in out
    assert "P1" not in out                                         # no tune-vs-run verdict line


def test_main_defaults_to_the_o2_layout_search_unless_a_mode_is_chosen(monkeypatch):
    seen = {}
    monkeypatch.setattr(wtune.tune.cli, "run", lambda args, **k: seen.update(level=args.level,
                                                                       nodes=args.nodes))
    wtune.tune.main(["matthew"])
    assert seen["level"] == 2 and wtune.tune.DEFAULT_LEVEL == 2
    wtune.tune.main(["matthew", "-O1"])
    assert seen["level"] == 1
    wtune.tune.main(["matthew", "--nodes", "4", "--minutes", "30"])      # legacy fixed N: untouched
    assert seen["level"] is None and seen["nodes"] == "4"


def test_point_job_modes_and_exports():
    anchor = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1)
    one = dict(anchor, id="W0")
    two = dict(anchor, id="L1")
    framed = dict(one, window_frame="mid_end")
    assert wtune.tune.choose_point_mode([one], anchor) == "fused"           # winner-only
    assert wtune.tune.choose_point_mode([], anchor) == "no_end"             # ANCHOR alone
    assert wtune.tune.choose_point_mode([two], anchor) == "fused"           # exactly one candidate
    assert wtune.tune.choose_point_mode([one, two], anchor) == "default"    # 3 configs: keep END
    assert wtune.tune.choose_point_mode([framed], dict(anchor, window_frame="mid_end")) == "no_end"
    assert wtune.tune.point_export("fused") == [("POINT_FUSED", 1), ("ANCHOR_END", 0),
                                                ("WINDOW_EXACT", 1)]
    assert wtune.tune.point_export("no_end") == [("ANCHOR_END", 0)]
    assert wtune.tune.point_export("default") == []
    wrapped = wtune.tune.wrap_with_anchors([one], anchor)
    ids = lambda m: [c["id"] for c in wtune.tune.point_runs(wrapped, m)]      # noqa: E731
    assert ids("default") == ["ANCHOR", "W0", "ANCHOREND"]
    assert ids("no_end") == ["ANCHOR", "W0"] and ids("fused") == ["W0"]


def _point_job(monkeypatch, tmp_path, mode, configs):
    """Run the REAL measure_candidates_at_points against fakes; return (sbatch cmd, written
    configs.json ids, meas_by_cid, gates)."""
    cmds, rows_for = [], {}

    def sbatch(cmd):
        cmds.append(cmd)
        return "Submitted batch job 5001\n"

    ctx = _ctx(sbatch_runner=sbatch, out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"),
               nodes=4, ledger_reader=lambda case, ids: rows_for["rows"])
    monkeypatch.setattr(wtune.tune.points, "_measure_run",
                        lambda ctx, rp, cid, jid, dt, first_global_step=None: (
                            _types.SimpleNamespace(at=rp["point"].at, s_normal=0.3), 1.0 + 8))
    rp = dict(index=0, point=_types.SimpleNamespace(at=_dt.datetime(2026, 1, 1, 6), reason="r"),
              case_id="matthewp0", T=_dt.datetime(2026, 1, 1), cases_json="/x/c.json", steps=32,
              spinup_min=20, stepra=6, dt=150.0)
    anchor = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                  blocktime="0")
    run_ids = {"fused": ["W0"], "no_end": ["ANCHOR", "L1"],
               "default": ["ANCHOR", "L1", "L2", "ANCHOREND"]}[mode]
    rows_for["rows"] = [dict(run_id="matthewp0_%s_r1_5001" % cid,
                             notes="restart_used=1" + (" fused=1" if mode == "fused" else ""),
                             s_per_step_median="0.3") for cid in run_ids]
    meas, gates, jobs = wtune.tune.measure_candidates_at_points(
        ctx, configs, [rp], anchor, s_by_cid={"ANCHOR": 0.3, "W0": 0.3, "L1": 0.3, "L2": 0.3},
        mode=mode)
    with open(re.search(r"CONFIG_FILE=(\S+)", cmds[0]).group(1)) as f:
        written = [c["id"] for c in json.load(f)["configs"]]
    return cmds[0], written, meas, gates


def test_two_config_point_job_drops_anchorend_and_the_drift_gate(monkeypatch, tmp_path):
    l1 = dict(id="L1", ranks=64, omp=1, nx=8, ny=8, numtiles=1, dist="block", blocktime="0")
    cmd, written, meas, gates = _point_job(monkeypatch, tmp_path, "no_end", [l1])
    assert "ANCHOR_END=0" in cmd and "POINT_FUSED" not in cmd and "WINDOW_EXACT" not in cmd
    assert written == ["ANCHOR", "L1"] and set(meas) == {"ANCHOR", "L1"}
    assert gates[0]["ok"] and gates[0]["anchor_drift"] is None


def test_o0_on_a_grid_too_small_for_all_cores_uses_fewer():
    """f1-arm 2026-10-15: -O0 on matthew, yagi, c34 at one thread per rank stopped with "no legal
    layout at any node count" -- it tried all 144 cores per node only; the node sweep steps down."""
    ctx = _ctx(cores_per_node=144, max_nodes=4, exclude_rules=[dict(omp_gt=1, when="fire")],
               is_fire=True)
    report = wtune.tune.pipeline.run_level0(ctx, ALL_NODES_PROFILE, None)
    assert report["best"]["omp"] == 1 and report["cores_used_per_node"] < 144
    assert report["best"]["ranks"] == report["nodes"] * report["cores_used_per_node"]
    assert report["pipeline"]["prediction"]["cores_per_node"] == report["cores_used_per_node"]


def _r(cid, median):
    return dict(cid=cid, config={}, s_values=[median], median=median, lo=median, hi=median, n=1)


def test_a_job_s_input_copy_is_removed_once_it_has_ended(tmp_path):
    copy = tmp_path / "bp_fresh"
    (copy / "copy_0").mkdir(parents=True)
    ctx = _ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    ctx.input_copies["7001"] = str(copy)
    wtune.tune.jobs.wait_for_jobs(ctx, ["7001"], "test")        # the fake sacct says COMPLETED
    assert not copy.exists() and ctx.input_copies == {}


def test_busy_time_is_the_union_of_the_jobs_start_to_end():
    out = ("7001|2026-10-15T03:00:00|2026-10-15T03:01:00\n"
           "7002|2026-10-15T03:00:30|2026-10-15T03:02:00\n"      # overlaps 7001
           "7003|2026-10-15T03:05:00|2026-10-15T03:05:20\n"      # after a gap
           "7004|Unknown|Unknown")
    ctx = _ctx(sacct_runner=lambda cmd: out)
    assert wtune.tune.jobs.busy_s(ctx, ["7001", "7002", "7003", "7004"]) == 140.0
    from wtune.tune import report as rep
    row = dict(nodes=1, cores_per_node=144, ranks=144, omp=1, nx=9, ny=16, numtiles=1,
               median=0.17, reps=1)
    lines = rep.summary_lines(dict(winner=row, pipeline=dict(jobs_busy_s=113.0,
                                                              total_wall_s=216.0)))
    assert lines[-1] == "tuning:   jobs running 1:53 of 3:36 wall (the rest is queue and controller)"


def test_without_phase2_the_whole_run_comes_from_the_winners_s_per_step(monkeypatch, capsys):
    from wtune.tune import pipeline as PL
    ctx = _ctx(clock=FakeClock(), budget_s=float("inf"))
    st = PL._pipeline_start(ctx, None, 2, False, False, None, 1000, False)
    monkeypatch.setattr(PL, "_delete_window_frames", lambda ctx, runs: 0)
    monkeypatch.setattr(PL, "_attach_phase_timing", lambda ctx, phases, slices: None)
    report = dict(best=dict(median_s_per_step=3.4))          # joint proxy: no node-sweep callback
    out = PL._pipeline_finish(st, report, None, None, 0.1, None, [], True)
    assert out["pipeline"]["predicted_real_s"] == pytest.approx(1000 * 3.4 + ctx.startup_s)
    assert out["pipeline"]["predicted_real_source"] == "phase-1 anchor s/step x steps"


def test_a_disk_guard_abort_stops_the_tune_instead_of_dropping_the_n():
    from wtune.tune import candidates
    rows = [dict(run_id="conus25-e60237_ANCHOR_r1_2026498", job_id="2026498", s_per_step_median="",
                 notes="status=ABORTED reason=low_disk free_gib=2.722 min_free_gib=5 context=ANCHOR_r1")]
    with pytest.raises(SystemExit, match="disk guard"):
        candidates.aggregate_by_cid(rows, {})

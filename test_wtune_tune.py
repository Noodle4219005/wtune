#!/usr/bin/env python3
"""pytest suite for wtune_tune.py -- fake Slurm (sbatch/sacct/scancel), a synthetic-time clock and
synthetic ledgers throughout, so this never touches a real cluster and never needs
numpy/scikit-learn/catboost (predictor/decomp_search.py, the real adaptive-round proposer, is only
imported lazily by wtune_tune._default_proposer when no proposer is injected -- every test here
injects one). Run with:

  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest test_wtune_tune.py
"""
import csv
import json
import os
import re

import pytest

import wtune_tune

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
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
    real configs.json wtune_tune wrote and fabricate a plausible measurement per config id."""
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


def make_ledger_reader_from_configs(job_to_configs_path, base_s=1.0):
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
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cfg["id"], j), job_id=j,
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
        out_dir="/tmp/wtune-tune-test-out", ledger_dir="/tmp/wtune-tune-test-ledger",
        env_source="$HOME/wtune-scc/env.sh",
        sbatch_script="$HOME/wtune/campaign/strict_campaign.sbatch", sacct_cmd="sacct",
        poll_interval=0.0, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
        scancel_runner=scancel_runner, ledger_reader=ledger_reader, proposer=proposer, clock=clock,
    )
    base.update(overrides)
    ctx = wtune_tune.Ctx(**base)
    ctx.start_time = clock.now()
    return ctx


def _profile(table):
    return {"core_count_priors": {"efficiency_table": table}}


# ---------------------------------------------------------------------------
# slots computation
# ---------------------------------------------------------------------------
def test_compute_slots():
    assert wtune_tune.compute_slots(13, 4) == 3
    assert wtune_tune.compute_slots(13, 13) == 1
    assert wtune_tune.compute_slots(13, 20) == 1    # N alone exceeds max_nodes -> still 1 job
    assert wtune_tune.compute_slots(8, 2) == 4


def test_chunk_round_robin_keeps_anchor_in_first_chunk():
    chunks = wtune_tune.chunk_round_robin([{"id": "ANCHOR"}, {"id": "G1"}, {"id": "G2"}], 2)
    assert chunks[0][0]["id"] == "ANCHOR"
    assert sum(len(c) for c in chunks) == 3


def test_chunk_round_robin_drops_empty_chunks():
    chunks = wtune_tune.chunk_round_robin([{"id": "A"}], 5)
    assert chunks == [[{"id": "A"}]]


# ---------------------------------------------------------------------------
# --nodes auto choice from a synthetic profile (--nodes-policy min/cap/knee)
# ---------------------------------------------------------------------------
def test_choose_nodes_knee_stops_when_next_doubling_drops_below_1_5x():
    # speedup(n)=n*E(n): 0.99, 1.90, 3.32, 4.96, 5.20 -- 4->8 is the first step whose ratio
    # (4.96/3.32=1.494) drops below 1.5x, so knee must stop at nodes=4.
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.99},
        {"nodes": 2, "legal": True, "efficiency_E": 0.95},
        {"nodes": 4, "legal": True, "efficiency_E": 0.83},
        {"nodes": 8, "legal": True, "efficiency_E": 0.62},
        {"nodes": 13, "legal": True, "efficiency_E": 0.40},
    ]
    nodes, note = wtune_tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 4
    assert "1.5" in note


def test_choose_nodes_knee_caps_by_max_nodes():
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.99},
        {"nodes": 2, "legal": True, "efficiency_E": 0.95},
        {"nodes": 4, "legal": True, "efficiency_E": 0.90},
    ]
    nodes, _note = wtune_tune.choose_nodes_knee(_profile(table), max_nodes=2)
    assert nodes == 2


def test_choose_nodes_knee_stops_at_first_step_when_doubling_never_pays_off():
    # speedup(1)=0.5, speedup(2)=0.6 -- ratio 1.2 < 1.5x on the very first step.
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.5},
        {"nodes": 2, "legal": True, "efficiency_E": 0.3},
    ]
    nodes, note = wtune_tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 1
    assert "1.5" in note


def test_choose_nodes_knee_skips_illegal_rows():
    table = [
        {"nodes": 1, "legal": False},
        {"nodes": 2, "legal": True, "efficiency_E": 0.85},
    ]
    nodes, _note = wtune_tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 2


def test_choose_nodes_knee_empty_table_defaults_to_one():
    nodes, note = wtune_tune.choose_nodes_knee(_profile([]), max_nodes=13)
    assert nodes == 1
    assert "no legal" in note


def test_choose_nodes_min_picks_smallest_legal():
    table = [
        {"nodes": 1, "legal": False},
        {"nodes": 2, "legal": True, "efficiency_E": 0.9},
        {"nodes": 4, "legal": True, "efficiency_E": 0.8},
    ]
    nodes, note = wtune_tune.choose_nodes_min(_profile(table), max_nodes=13)
    assert nodes == 2
    assert "policy=min" in note


def test_choose_nodes_cap_picks_largest_legal_within_cap():
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.9},
        {"nodes": 2, "legal": True, "efficiency_E": 0.7},
        {"nodes": 4, "legal": True, "efficiency_E": 0.5},
        {"nodes": 8, "legal": True, "efficiency_E": 0.1},
    ]
    nodes, note = wtune_tune.choose_nodes_cap(_profile(table), max_nodes=4)
    assert nodes == 4
    assert "policy=cap" in note


def test_choose_nodes_dispatches_by_policy_and_always_warns_uncalibrated():
    table = [{"nodes": 1, "legal": True, "efficiency_E": 0.9},
             {"nodes": 2, "legal": True, "efficiency_E": 0.85}]
    for policy, expect_nodes in (("min", 1), ("cap", 2), ("knee", 2)):
        nodes, _note, warning = wtune_tune.choose_nodes(_profile(table), max_nodes=13,
                                                          policy=policy)
        assert nodes == expect_nodes
        assert "UNCALIBRATED" in warning
        assert policy in warning


def test_choose_nodes_rejects_unknown_policy():
    with pytest.raises(SystemExit):
        wtune_tune.choose_nodes(_profile([]), max_nodes=13, policy="bogus")


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
    out = wtune_tune.aggregate_by_cid(rows, configs)
    assert set(out) == {"ANCHOR"}
    assert out["ANCHOR"]["median"] == 0.5
    assert out["ANCHOR"]["n"] == 1


def test_aggregate_by_cid_medians_across_replicates():
    configs = {"G0001": {"id": "G0001"}}
    rows = [
        dict(run_id="matthew_G0001_r1_2000", s_per_step_median="1.0"),
        dict(run_id="matthew_G0001_r2_2000", s_per_step_median="3.0"),
    ]
    out = wtune_tune.aggregate_by_cid(rows, configs)
    assert out["G0001"]["median"] == 2.0
    assert out["G0001"]["lo"] == 1.0 and out["G0001"]["hi"] == 3.0
    assert out["G0001"]["n"] == 2


# ---------------------------------------------------------------------------
# confirm stage picks top-3 + anchor
# ---------------------------------------------------------------------------
def test_select_confirm_configs_top3_plus_anchor():
    all_results = {
        "ANCHOR": _result("ANCHOR", 1.0),
        "G0001": _result("G0001", 0.5),
        "G0002": _result("G0002", 0.6),
        "G0003": _result("G0003", 0.7),
        "G0004": _result("G0004", 0.8),
    }
    ids = [c["id"] for c in wtune_tune.select_confirm_configs(all_results, anchor_cid="ANCHOR")]
    assert ids == ["G0001", "G0002", "G0003", "ANCHOR"]


def test_select_confirm_configs_anchor_already_in_top3_not_duplicated():
    all_results = {
        "ANCHOR": _result("ANCHOR", 0.4),
        "G0001": _result("G0001", 0.5),
        "G0002": _result("G0002", 0.6),
        "G0003": _result("G0003", 0.7),
    }
    ids = [c["id"] for c in wtune_tune.select_confirm_configs(all_results, anchor_cid="ANCHOR")]
    assert ids == ["ANCHOR", "G0001", "G0002"]
    assert len(ids) == len(set(ids))


def test_select_confirm_configs_missing_anchor_is_fine():
    all_results = {"G0001": _result("G0001", 0.5)}
    ids = [c["id"] for c in wtune_tune.select_confirm_configs(all_results, anchor_cid=None)]
    assert ids == ["G0001"]


# ---------------------------------------------------------------------------
# budget stop
# ---------------------------------------------------------------------------
def test_would_exceed_budget():
    assert wtune_tune.would_exceed_budget(elapsed_s=100, estimated_job_s=50, budget_s=200) is False
    assert wtune_tune.would_exceed_budget(elapsed_s=100, estimated_job_s=150, budget_s=200) is True


def test_run_round0_submits_nothing_when_budget_too_small():
    # budget_s=60 << even a single 1-config job's padded estimate at the 10s/step fallback
    # (no ledger rows -> CONSERVATIVE_S_PER_STEP): round0 must refuse to submit any slot.
    sbatch_runner, calls, _paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, budget_s=60.0)
    job_ids, configs_by_cid, rows = wtune_tune.run_round0(ctx)
    assert job_ids == []
    assert calls == []
    assert ctx.owned_jobs == []


def test_run_round0_submits_when_budget_allows():
    sbatch_runner, calls, paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, budget_s=3600.0, max_nodes=2, nodes=1)
    job_ids, configs_by_cid, rows = wtune_tune.run_round0(ctx)
    assert job_ids
    assert calls
    assert set(job_ids) == set(ctx.owned_jobs)
    assert "ANCHOR" in configs_by_cid


# ---------------------------------------------------------------------------
# cancel-only-own-jobs
# ---------------------------------------------------------------------------
def test_wait_for_jobs_cancels_only_given_job_ids_at_deadline():
    clock = FakeClock(start=0.0)
    states = {"1001": ("RUNNING", "00:10:00"), "1002": ("RUNNING", "00:10:00"),
              "9999": ("RUNNING", "00:10:00")}   # 9999: some other job, never passed to us
    sacct_runner = make_fake_sacct(states)
    scancel_runner, scancel_calls = make_fake_scancel()
    ctx = _ctx(clock=clock, sacct_runner=sacct_runner, scancel_runner=scancel_runner,
               budget_s=100.0)
    clock.t = 150.0   # already past the deadline on the very first poll
    wtune_tune.wait_for_jobs(ctx, ["1001", "1002"], "test")
    assert scancel_calls == [["1001", "1002"]]


def test_wait_for_jobs_does_not_cancel_when_jobs_finish_in_time():
    scancel_runner, scancel_calls = make_fake_scancel()
    ctx = _ctx(sacct_runner=make_fake_sacct({}), scancel_runner=scancel_runner, budget_s=1800.0)
    wtune_tune.wait_for_jobs(ctx, ["1001", "1002"], "test")   # default fake state: COMPLETED
    assert scancel_calls == []


# ---------------------------------------------------------------------------
# adaptive rounds (proposer injected -- never imports predictor/decomp_search.py)
# ---------------------------------------------------------------------------
def test_run_adaptive_rounds_submits_one_round_then_stops():
    calls = {"n": 0}

    def fake_proposer(state, batch_size):
        calls["n"] += 1
        if calls["n"] == 1:
            cfg = dict(id="R01D0001", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                       blocktime="0")
            return dict(stopped=False, round=1), [cfg]
        return dict(stopped=True, reason="no more candidates"), []

    sbatch_runner, _calls, _paths = make_fake_sbatch(start=2000)

    def ledger_reader(case, job_ids):
        rows = []
        for j in job_ids:
            rows.append(dict(run_id="%s_R01D0001_r1_%s" % (case, j), job_id=j,
                              s_per_step_median="1.234"))
        return rows

    ctx = _ctx(sbatch_runner=sbatch_runner, ledger_reader=ledger_reader, proposer=fake_proposer,
               budget_s=3600.0)
    all_configs_by_cid, all_results = {}, {}
    wtune_tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)

    assert calls["n"] == 2
    assert ctx.owned_jobs == ["2001"]
    assert all_results["R01D0001"]["median"] == pytest.approx(1.234)


def test_run_adaptive_rounds_stops_at_75pct_projected_finish():
    def fake_proposer(state, batch_size):
        cfg = dict(id="RX", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block", blocktime="0")
        return dict(stopped=False, round=1), [cfg] * batch_size

    sbatch_runner, calls, _paths = make_fake_sbatch()
    clock = FakeClock(start=0.0)
    clock.t = 1000.0   # already 100% of a tiny budget elapsed
    ctx = _ctx(sbatch_runner=sbatch_runner, proposer=fake_proposer, budget_s=100.0, clock=clock)
    all_configs_by_cid, all_results = {}, {}
    wtune_tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)
    assert calls == []
    assert ctx.owned_jobs == []


# ---------------------------------------------------------------------------
# report fields
# ---------------------------------------------------------------------------
def test_build_report_fields():
    ctx = _ctx()
    ctx.owned_jobs = ["1001"]
    all_results = {
        "ANCHOR": _result("ANCHOR", 1.0,
                           {"id": "ANCHOR", "ranks": 8, "omp": 1, "nx": 4, "ny": 2, "numtiles": 1,
                            "dist": "block"}),
        "G0001": _result("G0001", 0.5,
                          {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2, "ny": 4, "numtiles": 1,
                           "dist": "cyclic"}),
    }
    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                      confirm_results=None, anchor_cid="ANCHOR")
    for key in ("case", "nodes", "nodes_note", "cores_per_node", "budget_minutes",
                "elapsed_minutes", "job_ids", "cancelled_job_ids", "anchor_id", "ranking", "best",
                "param_table", "full_run_prediction"):
        assert key in report, key

    assert report["ranking"][0]["id"] == "G0001"     # 0.5 s/step beats the 1.0 s/step anchor
    assert report["best"]["id"] == "G0001"
    assert report["ranking"][0]["speedup_vs_anchor"] == pytest.approx(2.0)
    assert report["param_table"]["launch_line"].startswith("mpirun -np 8")
    assert "--rank-by node" in report["param_table"]["launch_line"]   # dist=cyclic
    assert report["param_table"]["env"]["OMP_STACKSIZE"] == "1G"
    assert isinstance(report["full_run_prediction"], str) and report["full_run_prediction"]


def test_build_report_marks_confirmed_configs():
    ctx = _ctx()
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    confirm_results = {"G0001": _result("G0001", 0.55, all_results["G0001"]["config"])}
    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                      confirm_results=confirm_results, anchor_cid=None)
    assert report["ranking"][0]["confirmed"] is True
    assert report["ranking"][0]["median_s_per_step"] == pytest.approx(0.55)   # confirm wins


# ---------------------------------------------------------------------------
# end-to-end (real round0/decomp_space config generation against a synthetic case, fake Slurm,
# adaptive rounds stubbed to stop immediately so no numpy/sklearn/catboost is ever required)
# ---------------------------------------------------------------------------
def test_run_end_to_end(monkeypatch, tmp_path):
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    profile = _profile([{"nodes": 1, "legal": True, "efficiency_E": 0.95}])
    monkeypatch.setattr(
        wtune_tune, "resolve_case",
        lambda case, cores_per_node: (case_json, str(tmp_path / "namelist.input"), profile))

    sbatch_runner, sbatch_calls, job_to_configs_path = make_fake_sbatch(start=9000)
    sacct_runner = make_fake_sacct({})
    scancel_runner, scancel_calls = make_fake_scancel()
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path)

    def stop_immediately_proposer(state, batch_size):
        return dict(stopped=True, reason="test stub"), []

    args = wtune_tune.build_arg_parser().parse_args([
        "matthew", "--site", "scc", "--minutes", "30", "--nodes", "auto", "--max-nodes", "13",
        "--out", str(tmp_path / "out"), "--steps", "5",
        "--ledger-dir", str(tmp_path / "ledger"),
    ])

    report = wtune_tune.run(args, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
                             scancel_runner=scancel_runner, ledger_reader=ledger_reader,
                             proposer=stop_immediately_proposer, clock=FakeClock())

    assert report["case"] == "matthew"
    assert report["nodes"] == 1
    assert report["job_ids"]                                  # round0 (+confirm) actually ran
    assert report["ranking"]
    assert report["best"] is not None
    assert scancel_calls == []                                # budget was never exceeded
    assert os.path.exists(tmp_path / "out" / "tune_report.json")
    with open(tmp_path / "out" / "tune_report.json") as f:
        on_disk = json.load(f)
    assert on_disk["case"] == "matthew"


# ---------------------------------------------------------------------------
# regression: s_est must be refreshed from round0's own measurements (bug: it stayed pinned to
# the no-ledger fallback of 10 s/step forever, so every later job looked too slow and neither the
# adaptive round nor confirm ever ran even with 13.6 of 30 minutes still unused -- 2026-09-22
# conus12 run).
# ---------------------------------------------------------------------------
def test_update_s_est_prefers_this_runs_own_measurements_over_the_fallback():
    all_results = {"ANCHOR": _result("ANCHOR", 0.93), "G0001": _result("G0001", 0.30),
                   "G0002": _result("G0002", 0.45)}
    assert wtune_tune.update_s_est(all_results, fallback=10.0) == pytest.approx(0.45)  # median
    assert wtune_tune.update_s_est({}, fallback=10.0) == 10.0   # nothing measured yet: keep it


def test_round0_measuring_0_3s_per_step_lets_adaptive_and_confirm_both_run_within_budget(
        monkeypatch, tmp_path):
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    profile = _profile([{"nodes": 1, "legal": True, "efficiency_E": 0.95}])
    monkeypatch.setattr(
        wtune_tune, "resolve_case",
        lambda case, cores_per_node: (case_json, str(tmp_path / "namelist.input"), profile))

    sbatch_runner, sbatch_calls, job_to_configs_path = make_fake_sbatch(start=9700)
    sacct_runner = make_fake_sacct({})
    scancel_runner, scancel_calls = make_fake_scancel()
    # round0 measures 0.30/0.27/0.25 s/step here -- matching the real conus12 run's 0.30-0.93
    # s/step -- instead of the no-ledger fallback (CONSERVATIVE_S_PER_STEP=10s/step) every later
    # stage used to stay pinned to.
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=0.3)

    proposer_calls = {"n": 0}

    def one_round_proposer(state, batch_size):
        proposer_calls["n"] += 1
        if proposer_calls["n"] == 1:
            cfg = dict(id="R01D0001", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                       blocktime="0")
            return dict(stopped=False, round=1), [cfg]
        return dict(stopped=True, reason="test stub"), []

    args = wtune_tune.build_arg_parser().parse_args([
        "matthew", "--site", "scc", "--minutes", "30", "--nodes", "1", "--max-nodes", "1",
        "--out", str(tmp_path / "out"), "--steps", "20",
        "--ledger-dir", str(tmp_path / "ledger"),
    ])
    report = wtune_tune.run(args, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
                             scancel_runner=scancel_runner, ledger_reader=ledger_reader,
                             proposer=one_round_proposer, clock=FakeClock())

    assert proposer_calls["n"] == 2                              # adaptive round DID submit a batch
    assert any(row["confirmed"] for row in report["ranking"])    # confirm DID run
    assert scancel_calls == []                                   # everything fit inside the budget
    assert report["elapsed_minutes"] < report["budget_minutes"]  # budget was not exhausted


# ---------------------------------------------------------------------------
# regression: the full-run-prediction hook must feed wtune_predict this run's own measurements,
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
    assert wtune_tune.wtune_predict is not None, "wtune_predict.py must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)

    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}
    confirm_results = {"ANCHOR": dict(cid="ANCHOR", config=best_config, s_values=[0.90, 0.93],
                                      median=0.915, lo=0.90, hi=0.93, n=2)}

    probe_path = wtune_tune.write_probe_ledger(ctx, all_results, confirm_results)
    assert probe_path and os.path.exists(probe_path)
    with open(probe_path) as f:
        probe_rows = list(csv.DictReader(f))
    assert len(probe_rows) == 2   # confirm's own REPS=2 replicates, not round0's single value
    assert {r["s_per_step_median"] for r in probe_rows} == {"0.9", "0.93"}

    text = wtune_tune.build_full_run_prediction(ctx, best_config, nodes=1, all_results=all_results,
                                                 confirm_results=confirm_results)
    assert "source=measured" in text
    assert "no_probe_fallback" not in text

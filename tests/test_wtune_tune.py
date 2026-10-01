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

import wtune_config
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
    nodes, note = wtune_tune.choose_nodes_knee(_profile(table), max_nodes=2)
    assert nodes == 2
    # genuinely reached --max-nodes with speedup still clearing 1.5x at every step checked
    assert "--max-nodes cap=2" in note


def test_choose_nodes_knee_does_not_claim_cap_when_only_a_smaller_node_count_is_legal():
    # Regression: real sc26p2 run logged "using nodes=2" together with "every profiled doubling
    # step up to cap=13 still clears 1.5x" -- read as a contradiction (if it really cleared every
    # doubling up to the cap, it should have picked 13, or the largest legal <=13). The actual
    # cause: nodes 4/8/13 are simply not LEGAL for this case's grid, so the table only ever had
    # legal rows at 1 and 2 -- "using nodes=2" was already correct (the largest legal node count),
    # but the message wrongly implied the cap (13) had been checked and cleared. It must now say
    # so honestly and never mention "cap=13" as if doubling had been verified that far.
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.99},
        {"nodes": 2, "legal": True, "efficiency_E": 0.95},
        {"nodes": 4, "legal": False},
        {"nodes": 8, "legal": False},
        {"nodes": 13, "legal": False},
    ]
    nodes, note = wtune_tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 2                       # the largest LEGAL node count, correctly
    assert "cap=13" not in note             # never claim the --max-nodes cap was reached/checked
    assert "not reached" in note
    assert "legal" in note.lower()


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
    # No real namelist in this ctx (see _ctx()'s fake namelist_path), so every candidate's
    # wtune_predict.build_prediction() call raises -- predict_candidates flags pred_error for
    # all of them and rank_predicted_entries falls back to median s/step, reproducing the
    # pre-prediction ranking exactly (see rank_predicted_entries's docstring).
    ctx = _ctx()
    all_results = {
        "ANCHOR": _result("ANCHOR", 1.0),
        "G0001": _result("G0001", 0.5),
        "G0002": _result("G0002", 0.6),
        "G0003": _result("G0003", 0.7),
        "G0004": _result("G0004", 0.8),
    }
    ids = [c["id"] for c in
           wtune_tune.select_confirm_configs(ctx, 1, all_results, anchor_cid="ANCHOR")]
    assert ids == ["G0001", "G0002", "G0003", "ANCHOR"]


def test_select_confirm_configs_anchor_already_in_top3_not_duplicated():
    ctx = _ctx()
    all_results = {
        "ANCHOR": _result("ANCHOR", 0.4),
        "G0001": _result("G0001", 0.5),
        "G0002": _result("G0002", 0.6),
        "G0003": _result("G0003", 0.7),
    }
    ids = [c["id"] for c in
           wtune_tune.select_confirm_configs(ctx, 1, all_results, anchor_cid="ANCHOR")]
    assert ids == ["ANCHOR", "G0001", "G0002"]
    assert len(ids) == len(set(ids))


def test_select_confirm_configs_missing_anchor_is_fine():
    ctx = _ctx()
    all_results = {"G0001": _result("G0001", 0.5)}
    ids = [c["id"] for c in
           wtune_tune.select_confirm_configs(ctx, 1, all_results, anchor_cid=None)]
    assert ids == ["G0001"]


def test_run_confirm_drops_a_slow_outlier_instead_of_skipping_everything():
    # Regression for the 2026-09-22 sc26p2 run: "confirm: SKIP -- estimated end would exceed
    # budget" fired at elapsed 2.7/30 min on a case measured at 0.03-0.04 s/step, because top-3-
    # by-median was forced (round0 only measured 6 configs) to include one genuine outlier
    # alongside the anchor and a fast candidate -- and the OLD all-or-nothing check let that
    # single outlier veto confirming the two fast ones too, even though they trivially fit the
    # budget on their own. (500 s/step here, not sc26p2's real ~6.74: once the 2026-09-23
    # padded-vs-expected bug is fixed, a 6.74 s/step candidate's own REPS=2 job fits a 30-minute
    # budget fine on raw/expected arithmetic -- this needs a candidate that is genuinely
    # unaffordable even under CORRECT arithmetic, to keep exercising the drop path at all.)
    anchor_cfg = dict(id="ANCHOR", ranks=144, omp=1, nx=144, ny=1, numtiles=1, dist="block")
    g1_cfg = dict(id="G0001", ranks=144, omp=1, nx=72, ny=2, numtiles=1, dist="block")
    g2_cfg = dict(id="G0002", ranks=144, omp=1, nx=12, ny=12, numtiles=1, dist="cyclic")
    all_results = {
        "ANCHOR": _result("ANCHOR", 0.0283, anchor_cfg),
        "G0001": _result("G0001", 0.031, g1_cfg),
        "G0002": _result("G0002", 500.0, g2_cfg),   # the outlier -- genuinely unaffordable
    }

    sbatch_runner, sbatch_calls, job_to_configs_path = make_fake_sbatch(start=8000)
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=0.03)
    clock = FakeClock(start=0.0)
    ctx = _ctx(sbatch_runner=sbatch_runner, ledger_reader=ledger_reader, clock=clock,
               budget_s=1800.0, steps=20)
    clock.t = 162.0   # elapsed = 2.7 minutes, matching the real run

    job_id, confirm_results = wtune_tune.run_confirm(ctx, {}, all_results, anchor_cid="ANCHOR",
                                                       s_est=0.03)

    assert job_id is not None                            # confirm did NOT skip outright
    assert confirm_results is not None
    assert set(confirm_results) == {"ANCHOR", "G0001"}    # outlier dropped, fast ones confirmed
    with open(job_to_configs_path[job_id]) as f:
        submitted = json.load(f)
    assert {c["id"] for c in submitted["configs"]} == {"ANCHOR", "G0001"}


def test_run_confirm_skips_only_when_even_the_cheapest_candidate_alone_exceeds_budget():
    all_results = {"SLOW": _result("SLOW", 50.0, dict(id="SLOW", ranks=8, omp=1, nx=2, ny=4,
                                                        numtiles=1, dist="block"))}
    ctx = _ctx(budget_s=60.0, steps=20)   # 60s budget, nowhere near enough for one 50s/step config
    job_id, confirm_results = wtune_tune.run_confirm(ctx, {}, all_results, anchor_cid=None,
                                                       s_est=50.0)
    assert job_id is None
    assert confirm_results is None


def test_run_confirm_accepts_exact_conus25_numbers():
    # Regression for the 2026-09-23 conus25 run: "confirm candidate G0001 median=2.3145s/step
    # -> per_run~166.3s x REPS=2" then "confirm: SKIP ... estimated at 18:18" at elapsed 13:26 of
    # a 30-minute budget. 166.3s x REPS=2 = 5.5 minutes (raw/expected), not 18:18 (that was
    # job_hours, the x1.5 + 10min sbatch --time padding, used where a budget decision belongs).
    # elapsed 13:26 + ~5.5 min expected is comfortably under 30 min, so confirm must NOT skip.
    g1_cfg = dict(id="G0001", ranks=144, omp=1, nx=12, ny=12, numtiles=1, dist="block")
    all_results = {"G0001": _result("G0001", 2.3145, g1_cfg)}
    clock = FakeClock(start=0.0)
    clock.t = 13 * 60 + 26   # elapsed = 13:26
    assert wtune_tune.DEFAULT_STEPS == 20   # STEPS default, as reported
    sbatch_runner, _calls, job_to_configs_path = make_fake_sbatch(start=1000)
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=2.3145)
    ctx = _ctx(clock=clock, budget_s=30 * 60.0, steps=wtune_tune.DEFAULT_STEPS,
               sbatch_runner=sbatch_runner, ledger_reader=ledger_reader)

    job_id, confirm_results = wtune_tune.run_confirm(ctx, {}, all_results, anchor_cid=None,
                                                       s_est=2.3145)

    assert job_id is not None
    assert confirm_results is not None
    assert "G0001" in confirm_results
    # the exact expected duration this scenario must compute: 2*(120+20*2.3145) = 332.58s
    expected_total_run_s = 2 * (wtune_tune.DEFAULT_STARTUP_S + 20 * 2.3145)
    assert expected_total_run_s == pytest.approx(332.58, abs=0.01)
    assert ctx.elapsed() + expected_total_run_s < ctx.budget_s   # comfortably fits, unlike job_hours


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
# expected vs padded duration (2026-09-23 conus25 bug: budget decisions used the safety-padded
# sbatch --time estimate instead of the raw expected duration)
# ---------------------------------------------------------------------------
def test_estimate_run_time_separates_expected_from_padded():
    # 1 config, 1 rep, steps=20, s_est=2.3145, startup_s=120 (documented default) -- exactly the
    # reported conus25 numbers: per_run=166.29s, expected(REPS=2)=332.58s=5.5min, but padded
    # job_hours ~= 18 minutes (x1.5 safety factor + 10min fixed pad).
    est = wtune_tune.estimate_run_time(1, 2, 20, s_est=2.3145, startup_s=120.0)
    assert est["total_run_s"] == pytest.approx(332.58, abs=0.01)
    assert est["job_hours"] * 3600.0 == pytest.approx(1099.05, abs=1.0)   # ~18:19, the padded value
    assert est["total_run_s"] < est["job_hours"] * 3600.0 / 2   # expected is well under half the padded


def test_estimate_run_time_per_config_pairs_mode_ignores_s_est():
    cfgs = [({"id": "A"}, 1.0), ({"id": "B"}, 3.0)]
    est = wtune_tune.estimate_run_time(cfgs, 1, steps=10, startup_s=100.0)
    # per_run: (100+10*1)=110, (100+10*3)=130 -> total_run_s = 110+130 = 240
    assert est["total_run_s"] == pytest.approx(240.0)


def test_parse_slurm_elapsed():
    assert wtune_tune._parse_slurm_elapsed("00:01:30") == 90.0
    assert wtune_tune._parse_slurm_elapsed("01:23:45") == 5025.0
    assert wtune_tune._parse_slurm_elapsed("23:45") == 23 * 60 + 45
    assert wtune_tune._parse_slurm_elapsed("1-00:01:23") == 86400 + 83
    assert wtune_tune._parse_slurm_elapsed("") is None
    assert wtune_tune._parse_slurm_elapsed(None) is None
    assert wtune_tune._parse_slurm_elapsed("garbage") is None


def test_measure_startup_s_from_completed_jobs():
    cfg_a = dict(id="A", ranks=8, omp=1, nx=2, ny=4, numtiles=1, dist="block")
    cfg_b = dict(id="B", ranks=8, omp=1, nx=4, ny=2, numtiles=1, dist="block")
    job_configs = {"1001": [cfg_a, cfg_b]}
    states = {"1001": ("COMPLETED", "00:05:00")}          # 300s elapsed for 2 runs
    wave_results = {"A": dict(median=2.0), "B": dict(median=3.0)}   # steps=10: compute=20+30=50s
    startup_s, n_runs = wtune_tune.measure_startup_s(job_configs, states, wave_results, steps=10)
    assert startup_s == pytest.approx(125.0)               # (300-50)/2
    assert n_runs == 2


def test_measure_startup_s_ignores_non_completed_jobs():
    cfg_a = dict(id="A", ranks=8, omp=1, nx=2, ny=4, numtiles=1, dist="block")
    job_configs = {"1001": [cfg_a]}
    states = {"1001": ("CANCELLED", "00:30:00")}   # elapsed reflects the deadline, not real cost
    wave_results = {"A": dict(median=2.0)}
    startup_s, n_runs = wtune_tune.measure_startup_s(job_configs, states, wave_results, steps=10)
    assert startup_s is None
    assert n_runs == 0


def test_run_round0_measures_startup_s_from_slurm_elapsed_and_logs_the_source():
    sbatch_runner, _calls, _paths = make_fake_sbatch(start=1000)
    sacct_runner = make_fake_sacct({"1001": ("COMPLETED", "00:03:00")})   # 180s elapsed

    def ledger_reader(case, job_ids):
        return [dict(run_id="%s_ANCHOR_r1_%s" % (case, j), job_id=j, s_per_step_median="0.5")
                for j in job_ids]

    ctx = _ctx(sbatch_runner=sbatch_runner, sacct_runner=sacct_runner, ledger_reader=ledger_reader,
               nodes=1, max_nodes=1, budget_s=1800.0, steps=20)
    assert ctx.startup_s == wtune_tune.DEFAULT_STARTUP_S
    assert "documented constant" in ctx.startup_source

    wtune_tune.run_round0(ctx)

    # measured: (180s elapsed - 20 steps * 0.5 s/step compute) / 1 run = 170s
    assert ctx.startup_s == pytest.approx(170.0)
    assert "measured" in ctx.startup_source


def test_round0_topup_packs_many_configs_when_slots_is_one():
    # Regression for the throughput half of the 2026-09-23 conus25 report: nodes=2,
    # --max-nodes 3 -> slots=floor(3/2)=1, so round0 has only ONE parallel job available and must
    # pack MORE configs into it (not leave the budget idle) once wave 0 shows the real per-config
    # cost is cheap, rather than being stuck at wave 0's own single conservative-fallback-sized job.
    sbatch_runner, _calls, job_to_configs_path = make_fake_sbatch(start=7000)
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=0.05)

    ctx = _ctx(sbatch_runner=sbatch_runner, ledger_reader=ledger_reader, nodes=2, max_nodes=3,
               budget_s=1800.0, steps=20)
    assert wtune_tune.compute_slots(ctx.max_nodes, ctx.nodes) == 1

    job_ids, configs_by_cid, _rows = wtune_tune.run_round0(ctx)

    assert len(job_ids) >= 2               # wave0's own lone job + a top-up wave
    assert len(configs_by_cid) > 4         # far more than wave0's single conservatively-sized job


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
def test_run_adaptive_rounds_falls_back_to_diversity_sampling_after_kappa_escalation():
    # Regression for the 2026-09-22 sc26p2 run: predictor/decomp_search.py's own LCB-vs-incumbent
    # stop rule fired at round 1 with 90% of the budget unused. wtune_tune.py must not accept that
    # as final -- it retries with escalating kappa (more exploration pressure), and only once that
    # ALSO fails to unstick the proposer does it fall back to raw decomp_space diversity sampling.
    calls = []

    def fake_proposer(state, batch_size, kappa=None):
        calls.append(kappa)
        if len(calls) == 1:
            cfg = dict(id="R01D0001", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block",
                       blocktime="0")
            return dict(stopped=False, round=1), [cfg]
        return dict(stopped=True, reason="LCB does not beat incumbent"), []

    sbatch_runner, sbatch_calls, _paths = make_fake_sbatch(start=2000)

    def ledger_reader(case, job_ids):
        rows = []
        for j in job_ids:
            rows.append(dict(run_id="%s_R01D0001_r1_%s" % (case, j), job_id=j,
                              s_per_step_median="1.234"))
        return rows

    ctx = _ctx(sbatch_runner=sbatch_runner, ledger_reader=ledger_reader, proposer=fake_proposer,
               budget_s=3600.0, kappa=0.5)
    all_configs_by_cid, all_results = {}, {}
    wtune_tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)

    # round1 succeeds at kappa=0.5; round2's proposer stops at kappa=0.5, then gets retried with
    # KAPPA_MAX_ESCALATIONS=2 rounds of escalation (x2.0 each: 0.5 -> 1.0 -> 2.0) before giving up
    # on the Bayesian proposer and falling back to decomp_space diversity sampling -- and, with
    # ample budget (3600s) and plenty of legal space left, keeps doing that every round up to
    # ADAPTIVE_MAX_ROUNDS rather than accepting the very first "stopped" as final.
    assert calls[:4] == [0.5, 0.5, 1.0, 2.0]
    assert len(ctx.owned_jobs) == wtune_tune.ADAPTIVE_MAX_ROUNDS   # round1 + a fallback every round after
    assert sbatch_calls                      # the fallback really did submit real Slurm jobs
    assert all_results["R01D0001"]["median"] == pytest.approx(1.234)
    fallback_cids = [cid for cid in all_configs_by_cid if cid != "R01D0001"]
    assert fallback_cids and all(cid.startswith("D") for cid in fallback_cids)


def test_run_adaptive_rounds_stops_when_diversity_pool_also_exhausted():
    # The genuine terminal condition: the Bayesian proposer has nothing left AND every legal
    # decomp_space candidate is already accounted for -- must still stop cleanly, no phantom job.
    def always_stops_proposer(state, batch_size, kappa=None):
        return dict(stopped=True, reason="no candidates"), []

    sbatch_runner, sbatch_calls, _paths = make_fake_sbatch(start=3000)
    ctx = _ctx(sbatch_runner=sbatch_runner, proposer=always_stops_proposer, budget_s=3600.0)

    total_cores = ctx.nodes * ctx.cores_per_node
    candidates, _axis = wtune_tune.decomp_space.enumerate_space(ctx.case_json, total_cores,
                                                                  ctx.nodes, ctx.hardware_id)
    every_config = wtune_tune.decomp_space.to_campaign_configs(candidates, prefix="ALL")
    all_configs_by_cid = {c["id"]: c for c in every_config}
    all_results = {}

    wtune_tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)

    assert sbatch_calls == []
    assert ctx.owned_jobs == []


def test_run_adaptive_rounds_stops_at_75pct_projected_finish():
    def fake_proposer(state, batch_size, kappa=None):
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


def test_build_report_strict_guard_runs_against_the_real_namelist(tmp_path, monkeypatch):
    # decision 0056: build_report's final param_table (nproc_x/nproc_y/numtiles only, see
    # local_param_table) must pass wtune_compliance.assert_strict_only against the case's
    # own namelist -- confirm the guard actually runs (not skipped as "no real namelist")
    # and that the normal decomposition-only report passes it cleanly.
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text("&domains\n time_step = 60,\n nproc_x = -1,\n nproc_y = -1,\n"
                             " numtiles = 1,\n/\n")
    calls = []
    real_assert = wtune_tune.wc.assert_strict_only

    def spy(orig, final, **kw):
        calls.append((dict(orig), dict(final)))
        return real_assert(orig, final, **kw)

    monkeypatch.setattr(wtune_tune.wc, "assert_strict_only", spy)

    ctx = _ctx(namelist_path=str(namelist_path))
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                            confirm_results=None, anchor_cid=None)

    assert len(calls) == 1
    orig, final = calls[0]
    assert orig["nproc_x"] == [-1]
    assert final["nproc_x"] == ["2"] and final["nproc_y"] == ["4"] and "numtiles" not in {
        k for k in final if final[k] != orig.get(k)}


def test_strict_guard_reads_every_namelist_line_the_report_hands_out(tmp_path):
    # a solution-changing key slipped into namelist_lines must be caught, not just the
    # three decomposition keys the guard happens to expect today
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text("&domains\n time_step = 60,\n nproc_x = -1,\n/\n"
                             "&physics\n mp_physics = 8,\n/\n")
    ctx = _ctx(namelist_path=str(namelist_path))
    ok = {"namelist_lines": ["nproc_x=2,", "nproc_y=4,", "numtiles=1,", "mp_physics=8,"]}
    wtune_tune._assert_report_is_strict_only(ctx, ok)          # unchanged physics: passes
    bad = {"namelist_lines": ["nproc_x=2,", "mp_physics=6,"]}
    with pytest.raises(wtune_tune.wc.ComplianceError, match="mp_physics"):
        wtune_tune._assert_report_is_strict_only(ctx, bad)
    with pytest.raises(wtune_tune.wc.ComplianceError, match="time_step"):
        wtune_tune._assert_report_is_strict_only(ctx, {"namelist_lines": ["time_step=45,"]})


def test_build_report_propagates_a_strict_guard_violation(tmp_path, monkeypatch):
    # wiring test: a ComplianceError from the guard must not be swallowed by build_report.
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text("&domains\n nproc_x = -1,\n nproc_y = -1,\n numtiles = 1,\n/\n")

    def always_raise(orig, final, **kw):
        raise wtune_tune.wc.ComplianceError("simulated non-STRICT change")

    monkeypatch.setattr(wtune_tune.wc, "assert_strict_only", always_raise)

    ctx = _ctx(namelist_path=str(namelist_path))
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    with pytest.raises(wtune_tune.wc.ComplianceError):
        wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                confirm_results=None, anchor_cid=None)


def test_build_report_marks_confirmed_configs():
    ctx = _ctx()
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    confirm_results = {"G0001": dict(_result("G0001", 0.55, all_results["G0001"]["config"]),
                                     s_values=[0.55, 0.55], n=2)}     # both confirm reps ran
    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                      confirm_results=confirm_results, anchor_cid=None)
    assert report["ranking"][0]["confirmed"] is True
    assert report["ranking"][0]["median_s_per_step"] == pytest.approx(0.55)   # confirm wins


def test_build_report_partial_confirm_is_not_marked_confirmed():
    # confirm job CANCELLED at the budget deadline after 1 of confirm_reps=2 replicates (conus25
    # scc run: best.confirmed was True with n_measurements=1)
    ctx = _ctx()
    cfg = {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2, "ny": 4, "numtiles": 1, "dist": "block"}
    all_results = {"G0001": _result("G0001", 0.5, cfg)}
    confirm_results = {"G0001": _result("G0001", 0.55, cfg)}          # n=1 < confirm_reps=2
    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                      confirm_results=confirm_results, anchor_cid=None)
    assert report["ranking"][0]["confirmed"] is False
    assert report["best"]["confirmed"] is False
    assert report["best"]["n_measurements"] == 1


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

    def stop_immediately_proposer(state, batch_size, kappa=None):
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
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=0.3, reps=2)

    proposer_calls = {"n": 0}

    def one_round_proposer(state, batch_size, kappa=None):
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

    assert proposer_calls["n"] >= 2                              # adaptive round DID submit a batch
    assert any(row["confirmed"] for row in report["ranking"])    # confirm DID run
    assert len(report["ranking"]) > 1                             # round0's top-up widened coverage
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


# ---------------------------------------------------------------------------
# --config wiring (wtune_config.py): shared search filter, confirm_reps, prediction_cfg,
# CLI > file > defaults precedence.
# ---------------------------------------------------------------------------
def test_apply_search_filter_exempts_anchor_from_search_but_not_from_exclude():
    ctx = _ctx(search_cfg=dict(omp=[1]), exclude_rules=[dict(omp_gt=1, when="fire")], is_fire=True)
    anchor = dict(id="ANCHOR", config_source="anchor", omp=2, numtiles=1, nx=8, ny=9, dist="block")
    sampled_ok = dict(id="G0001", omp=1, numtiles=1, nx=8, ny=9, dist="block")
    sampled_bad_omp = dict(id="G0002", omp=2, numtiles=1, nx=8, ny=9, dist="block")

    out = wtune_tune._apply_search_filter(ctx, [anchor, sampled_ok, sampled_bad_omp])

    # anchor: omp=2 would fail search_cfg's omp=[1], but is exempt from that -- however this
    # anchor's omp=2 ALSO matches the `exclude` fire rule, which is never exempt, so it is dropped.
    assert [c["id"] for c in out] == ["G0001"]


def test_apply_search_filter_anchor_survives_when_exclude_does_not_match():
    ctx = _ctx(search_cfg=dict(omp=[1]), exclude_rules=[dict(omp_gt=1, when="fire")], is_fire=False)
    anchor = dict(id="ANCHOR", omp=2, numtiles=1, nx=8, ny=9, dist="block")
    out = wtune_tune._apply_search_filter(ctx, [anchor])
    assert [c["id"] for c in out] == ["ANCHOR"]   # exempt from search, exclude rule inert (not fire)


def test_apply_search_filter_is_a_no_op_with_default_ctx():
    ctx = _ctx()   # default search_cfg/exclude_rules/is_fire -- unrestricted, matches pre-config behaviour
    configs = [dict(id="G0001", omp=9, numtiles=37, nx=1, ny=91, dist="cyclic")]
    assert wtune_tune._apply_search_filter(ctx, configs) == configs


def test_sample_more_diverse_configs_respects_search_cfg_before_sampling():
    ctx = _ctx(case_json=dict(MATTHEW_LIKE, prep_dir="/fake/matthew"), nodes=1,
               search_cfg=dict(omp=[1, 2]))
    fresh = wtune_tune.sample_more_diverse_configs(ctx, [], k=50, seed=1)
    assert fresh
    assert all(c["omp"] in (1, 2) for c in fresh)


def test_run_confirm_uses_ctx_confirm_reps_not_a_hardcoded_default():
    sbatch_runner, sbatch_calls, _paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, confirm_reps=3)
    all_results = {"ANCHOR": _result("ANCHOR", 0.10, dict(id="ANCHOR", ranks=1, omp=1, nx=1, ny=1,
                                                            numtiles=1, dist="block"))}
    job_id, confirm_results = wtune_tune.run_confirm(ctx, {}, all_results, "ANCHOR", s_est=0.10)
    assert job_id is not None
    assert any("REPS=3" in c for c in sbatch_calls)
    assert not any("REPS=2" in c for c in sbatch_calls)


def test_confirm_reps_default_matches_the_previous_hardcoded_constant():
    ctx = _ctx()
    assert ctx.confirm_reps == wtune_tune.CONFIRM_REPS == 2


def test_build_full_run_prediction_applies_drift_band_override_and_flags_measure_r_rad(tmp_path):
    assert wtune_tune.wtune_predict is not None, "wtune_predict.py must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"),
               prediction_cfg=dict(io_target="nfs", drift_band_pct=9.5, measure_r_rad=True))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    text = wtune_tune.build_full_run_prediction(ctx, best_config, nodes=1, all_results=all_results,
                                                 confirm_results=None)

    assert "prediction.drift_band_pct=9.5 applied via calibration override" in text
    override_path = os.path.join(str(tmp_path / "out"), "calibration_override.json")
    assert os.path.exists(override_path)
    with open(override_path) as f:
        assert json.load(f)["drift_band_pct"] == 9.5
    assert "measure_r_rad requested" in text
    assert "NOT YET IMPLEMENTED" in text


def test_build_full_run_prediction_no_drift_override_by_default(tmp_path):
    assert wtune_tune.wtune_predict is not None, "wtune_predict.py must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    text = wtune_tune.build_full_run_prediction(ctx, best_config, nodes=1, all_results=all_results,
                                                 confirm_results=None)

    assert "calibration override" not in text
    assert "measure_r_rad" not in text
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "calibration_override.json"))


# ---------------------------------------------------------------------------
# ranking by PREDICTED whole-run wall time (2026-09-24 decision), not raw median s/step --
# predict_candidates/rank_predicted_entries. wtune_predict.build_prediction is monkeypatched in
# most of these: the point is to test THIS module's consumption of a prediction (ranking, tie
# rule, failure handling), not wtune_predict's own physics model (that belongs to
# test_wtune_predict.py). Tests that need a REAL prediction (the h:mm:ss column, the
# include_io=False guarantee end to end) reuse _PREDICT_TEST_NAMELIST/MATTHEW_LIKE, the same
# fixture the build_full_run_prediction tests above already rely on.
# ---------------------------------------------------------------------------
def test_predict_candidates_ranking_can_invert_relative_to_median_s_per_step(tmp_path, monkeypatch):
    # G_FAST has the lower measured s/step (0.5) but -- e.g. because its own probe measured an
    # expensive radiation ratio over the FULL run's many more radiation calls -- a much higher
    # PREDICTED total; G_SLOW is the reverse. The tuner must pick by predicted total.
    ctx = _ctx(out_dir=str(tmp_path / "out"))
    cfg_fast = dict(id="G_FAST", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    cfg_slow = dict(id="G_SLOW", ranks=48, omp=1, nx=6, ny=8, numtiles=1, dist="block")
    results = {
        "G_FAST": _result("G_FAST", 0.5, cfg_fast),
        "G_SLOW": _result("G_SLOW", 0.8, cfg_slow),
    }
    fake_totals = {72: 5000.0, 48: 3000.0}   # inverted vs s/step

    def fake_build_prediction(namelist_path, case_dir, nodes, config, ledger_paths,
                               calibration_path, io_form_override, io_target, case_name=None,
                               iofields_dir=None, rsl_paths=None, include_io=False):
        assert include_io is False
        total = fake_totals[config["ranks"]]
        return {"total_s": total, "range_low_s": total * 0.95, "range_high_s": total * 1.05,
                "compute": {"r_rad_source": "fake_test_source"}}

    monkeypatch.setattr(wtune_tune.wtune_predict, "build_prediction", fake_build_prediction)

    predicted = wtune_tune.predict_candidates(ctx, nodes=1, results=results)
    assert predicted["G_FAST"]["pred_total_s"] == 5000.0
    assert predicted["G_SLOW"]["pred_total_s"] == 3000.0
    assert predicted["G_FAST"]["pred_source"] == "fake_test_source"

    ranked = wtune_tune.rank_predicted_entries(predicted, anchor_cid=None)
    assert [r["cid"] for r in ranked] == ["G_SLOW", "G_FAST"]   # predicted total wins, not median

    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=results,
                                      confirm_results=None, anchor_cid=None)
    assert report["best"]["id"] == "G_SLOW"
    assert report["ranking"][0]["predicted_total_s"] == 3000.0


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

    monkeypatch.setattr(wtune_tune.wtune_predict, "build_prediction", fake_build_prediction)

    predicted = wtune_tune.predict_candidates(ctx, nodes=1, results=results)
    assert predicted["G_BAD"]["pred_total_s"] is None
    assert "synthetic prediction failure" in predicted["G_BAD"]["pred_error"]
    assert predicted["G_OK"]["pred_error"] is None

    ranked = wtune_tune.rank_predicted_entries(predicted, anchor_cid=None)
    # the failed prediction ranks LAST despite its far better raw median s/step
    assert [r["cid"] for r in ranked] == ["G_OK", "G_BAD"]

    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=results,
                                      confirm_results=None, anchor_cid=None)
    assert report["ranking"][-1]["id"] == "G_BAD"
    assert report["ranking"][-1]["pred_error"] and "synthetic prediction failure" \
        in report["ranking"][-1]["pred_error"]
    text = wtune_tune.render_report_text(report)
    assert "PREDICTION FAILED for G_BAD" in text


def test_rank_predicted_entries_tie_within_noise_floor_keeps_anchor_first():
    # All three differ by ~0.1-0.2% -- well inside predictor/targets.py NOISE (0.5% relative) for
    # every pairwise comparison, so none of these differences may reorder the outcome-independent
    # tie-break: anchor first, then the pre-ranking (dict-insertion) order.
    predicted = {
        "G0001": dict(cid="G0001", config={}, median=1.0, pred_total_s=1000.0,
                      pred_low_s=950.0, pred_high_s=1050.0, pred_error=None),
        "ANCHOR": dict(cid="ANCHOR", config={}, median=1.0, pred_total_s=1000.0 * 1.001,
                       pred_low_s=950.0, pred_high_s=1050.0, pred_error=None),
        "G0002": dict(cid="G0002", config={}, median=1.0, pred_total_s=1000.0 * 0.999,
                      pred_low_s=950.0, pred_high_s=1050.0, pred_error=None),
    }
    ranked = wtune_tune.rank_predicted_entries(predicted, anchor_cid="ANCHOR")
    assert [r["cid"] for r in ranked] == ["ANCHOR", "G0001", "G0002"]


def test_rank_predicted_entries_clustering_is_transitive_and_input_order_independent():
    # A=100.0, B=100.3 (0.3% above A, < NOISE=0.5% -> same cluster as A), C=100.6 (0.6% above A,
    # >= NOISE -> its own cluster) -- the non-transitive case a pairwise cmp_to_key comparator
    # gets wrong: A~B is a tie and B~C is a tie (each pair < 0.5% apart), but A~C is NOT (0.6%
    # apart) -- so the "within NOISE" relation is not transitive across the three. The clustering
    # algorithm must still produce ONE deterministic answer regardless of input (dict-insertion)
    # order: A and B are the anchor-relative-to-A's cluster HEAD, C starts a new cluster after it.
    def make(order):
        vals = {"A": 100.0, "B": 100.3, "C": 100.6}
        return {cid: dict(cid=cid, config={}, median=1.0, pred_total_s=vals[cid],
                           pred_low_s=vals[cid] * 0.95, pred_high_s=vals[cid] * 1.05,
                           pred_error=None)
                for cid in order}

    ranked_abc = wtune_tune.rank_predicted_entries(make(["A", "B", "C"]), anchor_cid=None)
    ranked_cba = wtune_tune.rank_predicted_entries(make(["C", "B", "A"]), anchor_cid=None)

    assert [r["cid"] for r in ranked_abc] == ["A", "B", "C"]
    assert [r["cid"] for r in ranked_cba] == ["A", "B", "C"]   # same answer regardless of input order


def test_rank_predicted_entries_anchor_sorts_first_within_its_cluster_even_when_not_fastest():
    # G_FAST (non-anchor) has the numerically lowest pred_total_s, but ANCHOR is within NOISE of
    # it (same cluster) -- the anchor must still sort first within that cluster.
    predicted = {
        "G_FAST": dict(cid="G_FAST", config={}, median=1.0, pred_total_s=100.0,
                       pred_low_s=95.0, pred_high_s=105.0, pred_error=None),
        "ANCHOR": dict(cid="ANCHOR", config={}, median=1.0, pred_total_s=100.2,
                       pred_low_s=95.0, pred_high_s=105.0, pred_error=None),
    }
    ranked = wtune_tune.rank_predicted_entries(predicted, anchor_cid="ANCHOR")
    assert [r["cid"] for r in ranked] == ["ANCHOR", "G_FAST"]


def test_render_report_text_shows_predicted_total_hms_column(tmp_path):
    assert wtune_tune.wtune_predict is not None, "wtune_predict.py must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    ctx.owned_jobs = ["1001"]
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    report = wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                      confirm_results=None, anchor_cid="ANCHOR")
    hms = report["ranking"][0]["predicted_total_hms"]
    assert hms and re.match(r"^\d+:\d{2}(:\d{2})?$", hms)
    assert report["ranking"][0]["predicted_total_s"] > 0

    text = wtune_tune.render_report_text(report)
    assert "pred_total" in text        # header column
    assert hms in text                 # the actual h:mm:ss value appears in the rendered table


def test_build_prediction_never_called_with_include_io_true(tmp_path, monkeypatch):
    assert wtune_tune.wtune_predict is not None, "wtune_predict.py must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    real_build_prediction = wtune_tune.wtune_predict.build_prediction
    seen_include_io = []

    def spy(*args, **kwargs):
        seen_include_io.append(kwargs.get("include_io", False))
        return real_build_prediction(*args, **kwargs)

    monkeypatch.setattr(wtune_tune.wtune_predict, "build_prediction", spy)

    wtune_tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                             confirm_results=None, anchor_cid="ANCHOR")

    assert seen_include_io, "build_prediction was never called (test fixture is stale)"
    assert all(v is False for v in seen_include_io)


def _patched_load_config(raw):
    """A stand-in for wtune_config.load_config that skips real YAML parsing entirely -- resolves
    `raw` (a plain dict, the same shape a parsed YAML file would produce) through
    wtune_config.resolve_config directly. Lets the --config wiring in wtune_tune.run() (CLI > file
    > defaults precedence, search filtering, probe/prediction plumbing) be exercised without
    requiring PyYAML in this test environment -- only wtune_config.py's OWN load_config/CLI tests
    (test_wtune_config.py) need real YAML and are skipped there when PyYAML is absent."""
    def fake(path):
        cfg, errs = wtune_config.resolve_config(raw)
        if errs:
            raise wtune_config.ConfigError(errs)
        return cfg
    return fake


def _run_e2e_with_config(tmp_path, raw_config=None, extra_argv=(), monkeypatch=None):
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    profile = _profile([{"nodes": 1, "legal": True, "efficiency_E": 0.95}])
    monkeypatch.setattr(wtune_tune, "resolve_case",
                         lambda case, cores_per_node: (case_json, str(tmp_path / "namelist.input"),
                                                        profile))
    if raw_config is not None:
        monkeypatch.setattr(wtune_config, "load_config", _patched_load_config(raw_config))

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

    args = wtune_tune.build_arg_parser().parse_args(argv)
    report = wtune_tune.run(args, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
                             scancel_runner=scancel_runner, ledger_reader=ledger_reader,
                             proposer=stop_immediately_proposer, clock=FakeClock())
    return report, sbatch_calls


def test_config_budget_minutes_is_used_when_no_cli_minutes_given(tmp_path, monkeypatch):
    report, _calls = _run_e2e_with_config(
        tmp_path, raw_config=dict(case="matthew", budget=dict(minutes=12)), monkeypatch=monkeypatch)
    assert report["budget_minutes"] == pytest.approx(12.0)


def test_cli_flag_overrides_config_file_value(tmp_path, monkeypatch):
    # budget.minutes: 12 in the file, but --minutes 41 on the command line must win (CLI > file).
    report, _calls = _run_e2e_with_config(
        tmp_path, raw_config=dict(case="matthew", budget=dict(minutes=12)),
        extra_argv=["--minutes", "41"], monkeypatch=monkeypatch)
    assert report["budget_minutes"] == pytest.approx(41.0)


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
    monkeypatch.setattr(wtune_config, "load_config",
                         _patched_load_config(dict(case="matthew", search=dict(omp=[0]), bogus=1)))
    args = wtune_tune.build_arg_parser().parse_args(
        ["matthew", "--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit) as exc_info:
        wtune_tune.run(args)
    msg = str(exc_info.value)
    assert "search.omp" in msg
    assert "bogus" in msg


def test_run_requires_case_from_cli_or_config():
    args = wtune_tune.build_arg_parser().parse_args(["--minutes", "30"])
    with pytest.raises(SystemExit, match="case is required"):
        wtune_tune.run(args)


def test_run_requires_minutes_from_cli_or_config():
    args = wtune_tune.build_arg_parser().parse_args(["matthew"])
    with pytest.raises(SystemExit, match="--minutes is required"):
        wtune_tune.run(args)


def test_run_rejects_case_dir_only_config_as_not_yet_runnable(tmp_path, monkeypatch):
    monkeypatch.setattr(wtune_config, "load_config",
                         _patched_load_config(dict(case_dir="/some/where", budget=dict(minutes=10))))
    args = wtune_tune.build_arg_parser().parse_args(
        ["--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit, match="case_dir"):
        wtune_tune.run(args)


def test_run_rejects_a_multi_value_budget_nodes_list(tmp_path, monkeypatch):
    monkeypatch.setattr(wtune_config, "load_config",
                         _patched_load_config(dict(case="matthew",
                                                    budget=dict(minutes=10, nodes=[1, 2, 4]))))
    args = wtune_tune.build_arg_parser().parse_args(
        ["--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit, match="not yet run"):
        wtune_tune.run(args)


# ---------------------------------------------------------------------------
# --nodes search: measured node sweep, objective choice, finalist search
# ---------------------------------------------------------------------------
NONMONO_CURVE = {1: 1.0, 2: 0.52, 4: 0.30, 8: 0.32, 13: 0.40}   # 8 is slower than 4 (non-monotone)
ALL_NODES_PROFILE = _profile([{"nodes": n, "legal": True, "efficiency_E": 0.9}
                              for n in (1, 2, 4, 8, 13)])


def make_scaling_fakes(curve, start=7000):
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
                for rep in (1, 2):
                    rows.append(dict(run_id="%s_%s_r%d_%s" % (case, c["id"], rep, j),
                                     s_per_step_median=str(curve[job_nodes[j]] / (1 + 0.05 * i))))
        return rows

    return sbatch, reader, job_nodes


def _run_cli(monkeypatch, tmp_path, argv, curve=NONMONO_CURVE, profile=ALL_NODES_PROFILE,
             total_steps=1000, grid=MATTHEW_LIKE):
    case_json = dict(grid, prep_dir=str(tmp_path / "matthew"))
    monkeypatch.setattr(wtune_tune, "resolve_case",
                        lambda case, cpn: (case_json, str(tmp_path / "namelist.input"), profile))
    monkeypatch.setattr(wtune_tune, "_total_steps", lambda nl: total_steps)
    sbatch, reader, job_nodes = make_scaling_fakes(curve)
    scancel, scancel_calls = make_fake_scancel()

    def stop_proposer(state, batch_size, kappa=None):
        return dict(stopped=True, reason="test stub"), []

    args = wtune_tune.build_arg_parser().parse_args(
        ["matthew", "--site", "scc", "--out", str(tmp_path / "out"), "--steps", "5",
         "--ledger-dir", str(tmp_path / "ledger")] + argv)
    report = wtune_tune.run(args, sbatch_runner=sbatch, sacct_runner=make_fake_sacct({}),
                             scancel_runner=scancel, ledger_reader=reader,
                             proposer=stop_proposer, clock=FakeClock())
    return report, job_nodes, scancel_calls


def _sweep_nodes(job_nodes, report):
    """node counts of the sweep jobs only (their out dir is .../nodes-n<N>)."""
    return sorted(n for j, n in job_nodes.items() if j in report["node_selection"]["sweep_jobs"])


def test_nodes_search_is_default_and_picks_n_on_nonmonotone_curve(monkeypatch, tmp_path, capsys):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    ns = report["node_selection"]
    assert ns["mode"] == "search" and ns["objective"] == "time"
    assert ns["chosen"] == 4                       # min median at 4; 8 is slower
    assert report["nodes"] == 4
    assert "UNCALIBRATED" not in capsys.readouterr().out
    assert [r["nodes"] for r in ns["table"]] == [1, 2, 4, 8]


def test_nodes_search_early_stop_does_not_submit_larger_n(monkeypatch, tmp_path):
    # big grid: every N (13 included) has a legal pure-MPI layout, so 13 is skipped by early stop
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"],
                                      grid=dict(e_we=400, e_sn=400, e_vert=45))
    ns = report["node_selection"]
    assert 13 not in job_nodes.values()
    assert any(s["nodes"] == 13 and "early stop" in s["reason"] for s in ns["skipped"])


def test_nodes_search_finalists_are_searched_at_two_node_counts(monkeypatch, tmp_path):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    ns = report["node_selection"]
    assert ns["finalists"] == [4, 8]               # 8 is within 10% of 4's median (0.32 vs 0.30)
    # a round0 job (not the sweep's own) was submitted at each finalist N
    sweep_jobs = set(ns["sweep_jobs"])
    search_nodes = {n for j, n in job_nodes.items() if j not in sweep_jobs}
    assert search_nodes == {4, 8}
    assert set(ns["finalist_best"]) == {"4", "8"}
    assert report["nodes"] in (4, 8)


def test_nodes_search_report_lists_all_three_objectives(monkeypatch, tmp_path, capsys):
    report, _jn, _sc = _run_cli(monkeypatch, tmp_path,
                                ["--minutes", "600", "--node-objective", "cost"])
    ns = report["node_selection"]
    assert set(ns["choices"]) == {"time", "cost", "knee"}
    assert ns["choices"]["time"]["nodes"] == 4
    assert ns["choices"]["cost"]["nodes"] == 1     # fewest node-hours without a deadline
    assert ns["choices"]["knee"]["nodes"] == 4     # eff(4)=0.83 >= 0.7, eff(8)=0.39 is not
    row = ns["table"][0]
    for k in ("nodes", "median", "n", "spread", "speedup", "efficiency", "wall_s", "node_hours"):
        assert k in row
    assert row["n"] == 2 and row["wall_s"] is not None
    out = capsys.readouterr().out
    assert "node selection" in out and "choice cost *" in out
    with open(tmp_path / "out" / "tune_report.json") as f:
        assert json.load(f)["node_selection"]["objective"] == "cost"


def test_nodes_search_budget_skips_n_that_would_overrun(monkeypatch, tmp_path, capsys):
    # 3-minute budget: the 10 s/step no-ledger fallback makes even N=1 (120 s startup + 5 steps
    # x 10 s, x REPS=2 = 340 s) overrun -> every N skipped, nothing submitted, proxy fallback.
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "3"])
    ns = report["node_selection"]
    assert ns["sweep_jobs"] == [] and ns["table"] == []     # no sweep job was submitted
    assert ns["skipped"] and all("exceed the budget" in s["reason"] or "no legal" in s["reason"]
                                 for s in ns["skipped"])
    assert any(s["nodes"] == 1 and "exceed the budget" in s["reason"] for s in ns["skipped"])
    assert "falling back to the proxy" in capsys.readouterr().out


def test_node_sweep_budget_skip_is_per_n(tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes({1: 0.01, 2: 0.01, 4: 0.01})
    # per-job estimate ~ 2 x (120 + 20*10) = 640 s at the fallback -> budget 700 s fits one job
    # per wave only while elapsed is 0; elapsed does not advance (fake clock), so all fit...
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=700.0, max_nodes=13,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    ctx.clock.t = 100.0                # ...so advance the clock: 100 + 640 > 700 -> all skipped
    ctx.start_time = 0.0
    sw = wtune_tune.run_node_sweep(ctx, [1, 2, 4])
    assert job_nodes == {} and len(sw["skipped"]) == 3
    assert "exceed the budget" in sw["skipped"][0]["reason"]


def test_node_sweep_candidates_on_matthew_like_grid_use_omp(tmp_path):
    # 91x100 grid (r_max 90), 72 cores/node: pure MPI at N>=2 is illegal, but ranks = N*72/omp is
    # legal for omp=2 (N=2), 4 (N=4), 8 (N=8) -> all are sweep candidates with a hybrid ANCHOR.
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0, 2: 0.5, 4: 0.3, 8: 0.2, 13: 0.1})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune_tune.run_node_sweep(ctx, [1, 2, 4, 8, 13], budget_frac=1.0)
    for n, omp in ((1, 1), (2, 2), (4, 4), (8, 8)):
        a = sw["anchors"][n]
        assert a["omp"] == omp and a["ranks"] * a["omp"] == n * 72     # all cores used
        assert a["numtiles"] == omp and a["nx"] * a["ny"] == a["ranks"]
        assert a["ranks"] <= 90
    assert {s["nodes"] for s in sw["skipped"]} == {13}
    assert "no legal decomposition at any omp in {" in sw["skipped"][0]["reason"]
    assert {1, 2, 4} <= set(sw["measurements"])


def test_node_sweep_skips_n_without_legal_decomposition(tmp_path):
    # 20x20 grid: r_max 4. N=1 (omp 18 -> 4 ranks), N=2 (omp 36), N=4 (omp 72) work; N=8 would
    # need 8 ranks at omp<=72 -> none legal.
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0, 2: 0.6, 4: 0.5, 8: 0.4})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0,
               case_json=dict(e_we=20, e_sn=20, e_vert=45, prep_dir="/fake"),
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune_tune.run_node_sweep(ctx, [1, 2, 4, 8], budget_frac=1.0)
    assert sorted(sw["measurements"]) == [1, 2, 4]
    assert [s["nodes"] for s in sw["skipped"]] == [8]
    assert sw["skipped"][0]["reason"].startswith("no legal decomposition at any omp in {1, 2, 3")
    assert sw["anchors"][1]["omp"] == 18


def test_node_sweep_omp_cap_limits_candidates(tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0, 2: 0.6})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, omp_cap=1,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune_tune.run_node_sweep(ctx, [1, 2], budget_frac=1.0)
    assert sorted(sw["measurements"]) == [1]            # N=2 needs omp=2 > cap
    assert sw["skipped"][0]["reason"] == "no legal decomposition at any omp in {1}"


def test_node_sweep_applies_chem_j_patch_guard(monkeypatch, tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0})
    monkeypatch.setattr(wtune_tune.wtune_extensions, "chem_j_patch_reason",
                        lambda e_sn, ny, nx=None: "j-patch too narrow (test)")
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=36000.0, chem_e_sn=[100],
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune_tune.run_node_sweep(ctx, [1])
    assert job_nodes == {} and "j-patch too narrow" in sw["skipped"][0]["reason"]


def test_nodes_search_stops_after_sweep_when_budget_exhausted(monkeypatch, tmp_path):
    # budget just covers the sweep jobs (elapsed jumps past it once they finish): finalists get
    # no budget, so the report is the chosen N with the sweep's ANCHOR config.
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0, 2: 0.5})
    clock = FakeClock()

    def reader_then_exhaust(case, job_ids):
        rows = reader(case, job_ids)
        clock.t = 10 ** 6
        return rows

    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader_then_exhaust, clock=clock,
               case_json=case_json, budget_s=36000.0, out_dir=str(tmp_path / "o"),
               ledger_dir=str(tmp_path / "l"))
    report = wtune_tune.run_node_search(
        ctx, _profile([{"nodes": 1, "legal": True}, {"nodes": 2, "legal": True}]),
        objective="time", deadline_s=None, min_efficiency=0.7, k_finalists=2, total_steps=100)
    assert report["nodes"] == 2 and "budget exhausted after the sweep" in report["nodes_note"]
    assert report["best"]["id"] == "ANCHOR"
    assert set(job_nodes.values()) == {1, 2}      # nothing but sweep jobs


def test_nodes_fixed_is_unchanged(monkeypatch, tmp_path, capsys):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "30", "--nodes", "4"])
    assert report["nodes"] == 4 and "node_selection" not in report
    assert set(job_nodes.values()) == {4}
    assert "node sweep" not in capsys.readouterr().out


def test_nodes_auto_is_unchanged_and_keeps_the_warning(monkeypatch, tmp_path, capsys):
    profile = _profile([{"nodes": 1, "legal": True, "efficiency_E": 0.95}])
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path,
                                      ["--minutes", "30", "--nodes", "auto"], profile=profile)
    assert report["nodes"] == 1 and "node_selection" not in report
    assert "UNCALIBRATED" in capsys.readouterr().out
    assert set(job_nodes.values()) == {1}


def test_nodes_search_with_rank_by_predicted_is_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="not supported with --rank-by predicted"):
        _run_cli(monkeypatch, tmp_path,
                 ["--minutes", "30", "--nodes", "search", "--rank-by", "predicted"])


def test_node_sweep_budget_cap_skips_larger_n(tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes(NONMONO_CURVE)
    clock = FakeClock()

    def slow_reader(case, job_ids):
        rows = reader(case, job_ids)
        clock.t += 15000.0                   # first wave "takes" 15000 s of a 36000 s budget
        return rows

    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=slow_reader, clock=clock, budget_s=36000.0,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune_tune.run_node_sweep(ctx, [1, 2, 4, 8], budget_frac=0.4)
    assert sorted(set(job_nodes.values())) == [1, 2, 4]      # wave [1,2,4] ran, 8 did not
    capped = {s["nodes"] for s in sw["skipped"] if "sweep budget cap" in s["reason"]}
    assert capped == {8}
    # a roomier cap lets the next wave run
    ctx2 = _ctx(sbatch_runner=sbatch, ledger_reader=slow_reader, clock=FakeClock(),
                budget_s=36000.0, out_dir=str(tmp_path / "o2"), ledger_dir=str(tmp_path / "l"))
    clock.t = 0.0
    ctx2.clock = clock
    ctx2.start_time = 0.0
    sw2 = wtune_tune.run_node_sweep(ctx2, [1, 2, 4, 8], budget_frac=0.9)
    assert 8 in sw2["measurements"]


def test_sweep_budget_frac_cli_and_config_default(monkeypatch, tmp_path):
    args = wtune_tune.build_arg_parser().parse_args(["matthew", "--sweep-budget-frac", "0.25"])
    assert args.sweep_budget_frac == 0.25
    assert wtune_tune.DEFAULT_SWEEP_BUDGET_FRAC == 0.4
    assert wtune_tune.build_arg_parser().parse_args(["matthew"]).sweep_budget_frac is None


def test_finalist_round0_uses_the_sweep_hybrid_anchor(monkeypatch, tmp_path):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    assert report["node_selection"]["finalists"] == [4, 8]
    # round0 job configs at each finalist N: ANCHOR is the hybrid layout using all cores
    for n, omp in ((4, 4), (8, 8)):
        cfg_path = tmp_path / "out" / ("n%d" % n) / "round0-s0" / "configs.json"
        with open(cfg_path) as f:
            cfgs = {c["id"]: c for c in json.load(f)["configs"]}
        a = cfgs["ANCHOR"]
        assert a["omp"] == omp and a["ranks"] * a["omp"] == n * 72
        assert a["numtiles"] == omp


def test_round0_anchor_override_replaces_only_the_anchor(tmp_path):
    sbatch, calls, paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch, budget_s=36000.0, max_nodes=13, nodes=2,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    ov = dict(id="ANCHOR", ranks=72, omp=2, nx=8, ny=9, numtiles=2, dist="block", blocktime="0",
              config_source="anchor")
    ctx.anchor_override = ov
    _jobs, by_cid, _rows = wtune_tune.run_round0(ctx)
    assert by_cid["ANCHOR"] == ov
    others = [c for cid, c in by_cid.items() if cid != "ANCHOR"]
    assert others and all(wtune_tune.sweep._identity(c) != wtune_tune.sweep._identity(ov)
                          for c in others)


def test_node_search_builds_full_run_prediction_once_for_the_winner(monkeypatch, tmp_path):
    calls = []

    def fake_pred(ctx, best_config, nodes, all_results, confirm_results, probe_measurements=None):
        calls.append((nodes, best_config and best_config.get("id")))
        return "PREDICTION-FOR-%d" % nodes

    monkeypatch.setattr(wtune_tune, "build_full_run_prediction", fake_pred)
    report, _jn, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    assert len(report["node_selection"]["finalists"]) == 2     # two finalists were searched
    assert len(calls) == 1                                      # ...but one prediction build
    assert calls[0][0] == report["nodes"]
    assert report["full_run_prediction"] == "PREDICTION-FOR-%d" % report["nodes"]

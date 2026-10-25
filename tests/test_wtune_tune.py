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
import math
import getpass
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


# ---------------------------------------------------------------------------
# slots computation
# ---------------------------------------------------------------------------
def test_compute_slots():
    assert wtune.tune.compute_slots(13, 4) == 3
    assert wtune.tune.compute_slots(13, 13) == 1
    assert wtune.tune.compute_slots(13, 20) == 1    # N alone exceeds max_nodes -> still 1 job
    assert wtune.tune.compute_slots(8, 2) == 4


def test_chunk_round_robin_keeps_anchor_in_first_chunk():
    chunks = wtune.tune.chunk_round_robin([{"id": "ANCHOR"}, {"id": "G1"}, {"id": "G2"}], 2)
    assert chunks[0][0]["id"] == "ANCHOR"
    assert sum(len(c) for c in chunks) == 3


def test_chunk_round_robin_drops_empty_chunks():
    chunks = wtune.tune.chunk_round_robin([{"id": "A"}], 5)
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
    nodes, note = wtune.tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 4
    assert "1.5" in note


def test_choose_nodes_knee_caps_by_max_nodes():
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.99},
        {"nodes": 2, "legal": True, "efficiency_E": 0.95},
        {"nodes": 4, "legal": True, "efficiency_E": 0.90},
    ]
    nodes, note = wtune.tune.choose_nodes_knee(_profile(table), max_nodes=2)
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
    nodes, note = wtune.tune.choose_nodes_knee(_profile(table), max_nodes=13)
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
    nodes, note = wtune.tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 1
    assert "1.5" in note


def test_choose_nodes_knee_skips_illegal_rows():
    table = [
        {"nodes": 1, "legal": False},
        {"nodes": 2, "legal": True, "efficiency_E": 0.85},
    ]
    nodes, _note = wtune.tune.choose_nodes_knee(_profile(table), max_nodes=13)
    assert nodes == 2


def test_choose_nodes_knee_empty_table_defaults_to_one():
    nodes, note = wtune.tune.choose_nodes_knee(_profile([]), max_nodes=13)
    assert nodes == 1
    assert "no legal" in note


def test_choose_nodes_min_picks_smallest_legal():
    table = [
        {"nodes": 1, "legal": False},
        {"nodes": 2, "legal": True, "efficiency_E": 0.9},
        {"nodes": 4, "legal": True, "efficiency_E": 0.8},
    ]
    nodes, note = wtune.tune.choose_nodes_min(_profile(table), max_nodes=13)
    assert nodes == 2
    assert "policy=min" in note


def test_choose_nodes_cap_picks_largest_legal_within_cap():
    table = [
        {"nodes": 1, "legal": True, "efficiency_E": 0.9},
        {"nodes": 2, "legal": True, "efficiency_E": 0.7},
        {"nodes": 4, "legal": True, "efficiency_E": 0.5},
        {"nodes": 8, "legal": True, "efficiency_E": 0.1},
    ]
    nodes, note = wtune.tune.choose_nodes_cap(_profile(table), max_nodes=4)
    assert nodes == 4
    assert "policy=cap" in note


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


def test_aggregate_by_cid_medians_across_replicates():
    configs = {"G0001": {"id": "G0001"}}
    rows = [
        dict(run_id="matthew_G0001_r1_2000", s_per_step_median="1.0"),
        dict(run_id="matthew_G0001_r2_2000", s_per_step_median="3.0"),
    ]
    out = wtune.tune.aggregate_by_cid(rows, configs)
    assert out["G0001"]["median"] == 2.0
    assert out["G0001"]["lo"] == 1.0 and out["G0001"]["hi"] == 3.0
    assert out["G0001"]["n"] == 2


# ---------------------------------------------------------------------------
# confirm stage picks top-3 + anchor
# ---------------------------------------------------------------------------
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

    job_id, confirm_results = wtune.tune.run_confirm(ctx, {}, all_results, anchor_cid="ANCHOR",
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
    job_id, confirm_results = wtune.tune.run_confirm(ctx, {}, all_results, anchor_cid=None,
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
    assert wtune.tune.DEFAULT_STEPS == 20   # STEPS default, as reported
    sbatch_runner, _calls, job_to_configs_path = make_fake_sbatch(start=1000)
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path, base_s=2.3145)
    ctx = _ctx(clock=clock, budget_s=30 * 60.0, steps=wtune.tune.DEFAULT_STEPS,
               sbatch_runner=sbatch_runner, ledger_reader=ledger_reader)

    job_id, confirm_results = wtune.tune.run_confirm(ctx, {}, all_results, anchor_cid=None,
                                                       s_est=2.3145)

    assert job_id is not None
    assert confirm_results is not None
    assert "G0001" in confirm_results
    # the exact expected duration this scenario must compute: 2*(120+20*2.3145) = 332.58s
    expected_total_run_s = 2 * (wtune.tune.DEFAULT_STARTUP_S + 20 * 2.3145)
    assert expected_total_run_s == pytest.approx(332.58, abs=0.01)
    assert ctx.elapsed() + expected_total_run_s < ctx.budget_s   # comfortably fits, unlike job_hours


# ---------------------------------------------------------------------------
# budget stop
# ---------------------------------------------------------------------------
def test_would_exceed_budget():
    assert wtune.tune.would_exceed_budget(elapsed_s=100, estimated_job_s=50, budget_s=200) is False
    assert wtune.tune.would_exceed_budget(elapsed_s=100, estimated_job_s=150, budget_s=200) is True


def test_run_round0_submits_nothing_when_budget_too_small():
    # budget_s=60 << even a single 1-config job's padded estimate at the 10s/step fallback
    # (no ledger rows -> CONSERVATIVE_S_PER_STEP): round0 must refuse to submit any slot.
    sbatch_runner, calls, _paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, budget_s=60.0)
    job_ids, configs_by_cid, rows = wtune.tune.run_round0(ctx)
    assert job_ids == []
    assert calls == []
    assert ctx.owned_jobs == []


def test_run_round0_submits_when_budget_allows():
    sbatch_runner, calls, paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, budget_s=3600.0, max_nodes=2, nodes=1)
    job_ids, configs_by_cid, rows = wtune.tune.run_round0(ctx)
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
    est = wtune.tune.estimate_run_time(1, 2, 20, s_est=2.3145, startup_s=120.0)
    assert est["total_run_s"] == pytest.approx(332.58, abs=0.01)
    assert est["job_hours"] * 3600.0 == pytest.approx(1099.05, abs=1.0)   # ~18:19, the padded value
    assert est["total_run_s"] < est["job_hours"] * 3600.0 / 2   # expected is well under half the padded


def test_estimate_run_time_per_config_pairs_mode_ignores_s_est():
    cfgs = [({"id": "A"}, 1.0), ({"id": "B"}, 3.0)]
    est = wtune.tune.estimate_run_time(cfgs, 1, steps=10, startup_s=100.0)
    # per_run: (100+10*1)=110, (100+10*3)=130 -> total_run_s = 110+130 = 240
    assert est["total_run_s"] == pytest.approx(240.0)


def test_parse_slurm_elapsed():
    assert wtune.tune._parse_slurm_elapsed("00:01:30") == 90.0
    assert wtune.tune._parse_slurm_elapsed("01:23:45") == 5025.0
    assert wtune.tune._parse_slurm_elapsed("23:45") == 23 * 60 + 45
    assert wtune.tune._parse_slurm_elapsed("1-00:01:23") == 86400 + 83
    assert wtune.tune._parse_slurm_elapsed("") is None
    assert wtune.tune._parse_slurm_elapsed(None) is None
    assert wtune.tune._parse_slurm_elapsed("garbage") is None


def test_measure_startup_s_from_completed_jobs():
    cfg_a = dict(id="A", ranks=8, omp=1, nx=2, ny=4, numtiles=1, dist="block")
    cfg_b = dict(id="B", ranks=8, omp=1, nx=4, ny=2, numtiles=1, dist="block")
    job_configs = {"1001": [cfg_a, cfg_b]}
    states = {"1001": ("COMPLETED", "00:05:00")}          # 300s elapsed for 2 runs
    wave_results = {"A": dict(median=2.0), "B": dict(median=3.0)}   # steps=10: compute=20+30=50s
    startup_s, n_runs = wtune.tune.measure_startup_s(job_configs, states, wave_results, steps=10)
    assert startup_s == pytest.approx(125.0)               # (300-50)/2
    assert n_runs == 2


def test_measure_startup_s_ignores_non_completed_jobs():
    cfg_a = dict(id="A", ranks=8, omp=1, nx=2, ny=4, numtiles=1, dist="block")
    job_configs = {"1001": [cfg_a]}
    states = {"1001": ("CANCELLED", "00:30:00")}   # elapsed reflects the deadline, not real cost
    wave_results = {"A": dict(median=2.0)}
    startup_s, n_runs = wtune.tune.measure_startup_s(job_configs, states, wave_results, steps=10)
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
    assert ctx.startup_s == wtune.tune.DEFAULT_STARTUP_S
    assert "documented constant" in ctx.startup_source

    wtune.tune.run_round0(ctx)

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
    assert wtune.tune.compute_slots(ctx.max_nodes, ctx.nodes) == 1

    job_ids, configs_by_cid, _rows = wtune.tune.run_round0(ctx)

    assert len(job_ids) >= 2               # wave0's own lone job + a top-up wave
    assert len(configs_by_cid) > 4         # far more than wave0's single conservatively-sized job


def test_round0_under_a_time_cap_keeps_adding_waves_until_the_cap(monkeypatch):
    # -O2 --minutes (fill): top-up waves sized off the measured startup and s/step go on while the
    # cap, less one race job, has room (conus25 2026-10-16 stopped after one wave at 16.7 of 45 min)
    def run(fill):
        clock = FakeClock()
        sbatch, _calls, paths = make_fake_sbatch(start=7000)

        def slow_sbatch(cmd):                     # every job then runs 200 s
            clock.t += 200.0
            return sbatch(cmd)

        ctx = _ctx(clock=clock, sbatch_runner=slow_sbatch, nodes=2, max_nodes=3, budget_s=1800.0,
                   steps=20, ledger_reader=make_ledger_reader_from_configs(paths, base_s=0.05))
        job_ids, by_cid, _rows = wtune.tune.run_round0(ctx, fill=fill)
        return ctx, job_ids, by_cid

    _c, ids1, by1 = run(False)
    ctx, ids2, by2 = run(True)
    assert len(ids1) == 2                         # wave0 + the one top-up
    assert len(ids2) > len(ids1) and len(by2) > len(by1)
    assert ctx.elapsed() <= ctx.budget_s          # never past the cap


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
    wtune.tune.wait_for_jobs(ctx, ["1001", "1002"], "test")
    assert scancel_calls == [["1001", "1002"]]


def test_wait_for_jobs_does_not_cancel_when_jobs_finish_in_time():
    scancel_runner, scancel_calls = make_fake_scancel()
    ctx = _ctx(sacct_runner=make_fake_sacct({}), scancel_runner=scancel_runner, budget_s=1800.0)
    wtune.tune.wait_for_jobs(ctx, ["1001", "1002"], "test")   # default fake state: COMPLETED
    assert scancel_calls == []


# ---------------------------------------------------------------------------
# adaptive rounds (proposer injected -- never imports predictor/decomp_search.py)
# ---------------------------------------------------------------------------
def test_run_adaptive_rounds_falls_back_to_diversity_sampling_after_kappa_escalation():
    # Regression for the 2026-09-22 sc26p2 run: predictor/decomp_search.py's own LCB-vs-incumbent
    # stop rule fired at round 1 with 90% of the budget unused. wtune/tune/ must not accept that
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
    wtune.tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)

    # round1 succeeds at kappa=0.5; round2's proposer stops at kappa=0.5, then gets retried with
    # KAPPA_MAX_ESCALATIONS=2 rounds of escalation (x2.0 each: 0.5 -> 1.0 -> 2.0) before giving up
    # on the Bayesian proposer and falling back to decomp_space diversity sampling -- and, with
    # ample budget (3600s) and plenty of legal space left, keeps doing that every round up to
    # ADAPTIVE_MAX_ROUNDS rather than accepting the very first "stopped" as final.
    assert calls[:4] == [0.5, 0.5, 1.0, 2.0]
    # round1's one config is one job; every fallback round after is dealt over the free slots,
    # through the last round
    assert wtune.tune.common.compute_slots(ctx.max_nodes, ctx.nodes) > 1
    last = "round%d" % wtune.tune.ADAPTIVE_MAX_ROUNDS
    assert sum(1 for c in sbatch_calls if last in c) > 1
    assert len(ctx.owned_jobs) > wtune.tune.ADAPTIVE_MAX_ROUNDS
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
    candidates, _axis = wtune.tune.decomp_space.enumerate_space(ctx.case_json, total_cores,
                                                                  ctx.nodes, ctx.hardware_id)
    every_config = wtune.tune.decomp_space.to_campaign_configs(candidates, prefix="ALL")
    all_configs_by_cid = {c["id"]: c for c in every_config}
    all_results = {}

    wtune.tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)

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
    wtune.tune.run_adaptive_rounds(ctx, all_configs_by_cid, all_results, s_est=2.0)
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
    report = wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
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
    # local_param_table) must pass wtune.compliance.assert_strict_only against the case's
    # own namelist -- confirm the guard actually runs (not skipped as "no real namelist")
    # and that the normal decomposition-only report passes it cleanly.
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text("&domains\n time_step = 60,\n nproc_x = -1,\n nproc_y = -1,\n"
                             " numtiles = 1,\n/\n")
    calls = []
    real_assert = wtune.tune.wc.assert_strict_only

    def spy(orig, final, **kw):
        calls.append((dict(orig), dict(final)))
        return real_assert(orig, final, **kw)

    monkeypatch.setattr(wtune.tune.wc, "assert_strict_only", spy)

    ctx = _ctx(namelist_path=str(namelist_path))
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
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


def test_build_report_marks_confirmed_configs():
    ctx = _ctx()
    all_results = {"G0001": _result("G0001", 0.5, {"id": "G0001", "ranks": 8, "omp": 1, "nx": 2,
                                                     "ny": 4, "numtiles": 1, "dist": "block"})}
    confirm_results = {"G0001": dict(_result("G0001", 0.55, all_results["G0001"]["config"]),
                                     s_values=[0.55, 0.55], n=2)}     # both confirm reps ran
    report = wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
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
    report = wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
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
        wtune.tune.context, "resolve_case",
        lambda case, cores_per_node: (case_json, str(tmp_path / "namelist.input"), profile))

    sbatch_runner, sbatch_calls, job_to_configs_path = make_fake_sbatch(start=9000)
    sacct_runner = make_fake_sacct({})
    scancel_runner, scancel_calls = make_fake_scancel()
    ledger_reader = make_ledger_reader_from_configs(job_to_configs_path)

    def stop_immediately_proposer(state, batch_size, kappa=None):
        return dict(stopped=True, reason="test stub"), []

    args = wtune.tune.build_arg_parser().parse_args([
        "matthew", "--site", "scc", "--minutes", "30", "--nodes", "auto", "--max-nodes", "13",
        "--out", str(tmp_path / "out"), "--steps", "5",
        "--ledger-dir", str(tmp_path / "ledger"),
    ])

    report = wtune.tune.run(args, sbatch_runner=sbatch_runner, sacct_runner=sacct_runner,
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
    assert wtune.tune.update_s_est(all_results, fallback=10.0) == pytest.approx(0.45)  # median
    assert wtune.tune.update_s_est({}, fallback=10.0) == 10.0   # nothing measured yet: keep it


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


# ---------------------------------------------------------------------------
# --config wiring (wtune/config.py): shared search filter, confirm_reps, prediction_cfg,
# CLI > file > defaults precedence.
# ---------------------------------------------------------------------------
def test_apply_search_filter_exempts_anchor_from_search_but_not_from_exclude():
    ctx = _ctx(search_cfg=dict(omp=[1]), exclude_rules=[dict(omp_gt=1, when="fire")], is_fire=True)
    anchor = dict(id="ANCHOR", config_source="anchor", omp=2, numtiles=1, nx=8, ny=9, dist="block")
    sampled_ok = dict(id="G0001", omp=1, numtiles=1, nx=8, ny=9, dist="block")
    sampled_bad_omp = dict(id="G0002", omp=2, numtiles=1, nx=8, ny=9, dist="block")

    out = wtune.tune._apply_search_filter(ctx, [anchor, sampled_ok, sampled_bad_omp])

    # anchor: omp=2 would fail search_cfg's omp=[1], but is exempt from that -- however this
    # anchor's omp=2 ALSO matches the `exclude` fire rule, which is never exempt, so it is dropped.
    assert [c["id"] for c in out] == ["G0001"]


def test_apply_search_filter_anchor_survives_when_exclude_does_not_match():
    ctx = _ctx(search_cfg=dict(omp=[1]), exclude_rules=[dict(omp_gt=1, when="fire")], is_fire=False)
    anchor = dict(id="ANCHOR", omp=2, numtiles=1, nx=8, ny=9, dist="block")
    out = wtune.tune._apply_search_filter(ctx, [anchor])
    assert [c["id"] for c in out] == ["ANCHOR"]   # exempt from search, exclude rule inert (not fire)


def test_apply_search_filter_is_a_no_op_with_default_ctx():
    ctx = _ctx()   # default search_cfg/exclude_rules/is_fire -- unrestricted, matches pre-config behaviour
    configs = [dict(id="G0001", omp=9, numtiles=37, nx=1, ny=91, dist="cyclic")]
    assert wtune.tune._apply_search_filter(ctx, configs) == configs


def test_sample_more_diverse_configs_respects_search_cfg_before_sampling():
    ctx = _ctx(case_json=dict(MATTHEW_LIKE, prep_dir="/fake/matthew"), nodes=1,
               search_cfg=dict(omp=[1, 2]))
    fresh = wtune.tune.sample_more_diverse_configs(ctx, [], k=50, seed=1)
    assert fresh
    assert all(c["omp"] in (1, 2) for c in fresh)


def test_run_confirm_uses_ctx_confirm_reps_not_a_hardcoded_default():
    sbatch_runner, sbatch_calls, _paths = make_fake_sbatch()
    ctx = _ctx(sbatch_runner=sbatch_runner, confirm_reps=3)
    all_results = {"ANCHOR": _result("ANCHOR", 0.10, dict(id="ANCHOR", ranks=1, omp=1, nx=1, ny=1,
                                                            numtiles=1, dist="block"))}
    job_id, confirm_results = wtune.tune.run_confirm(ctx, {}, all_results, "ANCHOR", s_est=0.10)
    assert job_id is not None
    assert any("REPS=3" in c for c in sbatch_calls)
    assert not any("REPS=2" in c for c in sbatch_calls)


def test_confirm_reps_default_matches_the_previous_hardcoded_constant():
    ctx = _ctx()
    assert ctx.confirm_reps == wtune.tune.CONFIRM_REPS == 2


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


def test_build_full_run_prediction_no_drift_override_by_default(tmp_path):
    assert wtune.tune.predict is not None, "wtune predict must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    text = wtune.tune.build_full_run_prediction(ctx, best_config, nodes=1, all_results=all_results,
                                                 confirm_results=None)

    assert "calibration override" not in text
    assert "measure_r_rad" not in text
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "calibration_override.json"))


# ---------------------------------------------------------------------------
# ranking by PREDICTED whole-run wall time (2026-09-24 decision), not raw median s/step --
# predict_candidates/rank_predicted_entries. wtune.predict.build_prediction is monkeypatched in
# most of these: the point is to test THIS module's consumption of a prediction (ranking, tie
# rule, failure handling), not wtune.predict's own physics model (that belongs to
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

    monkeypatch.setattr(wtune.tune.predict, "build_prediction", fake_build_prediction)

    predicted = wtune.tune.predict_candidates(ctx, nodes=1, results=results)
    assert predicted["G_FAST"]["pred_total_s"] == 5000.0
    assert predicted["G_SLOW"]["pred_total_s"] == 3000.0
    assert predicted["G_FAST"]["pred_source"] == "fake_test_source"

    ranked = wtune.tune.rank_predicted_entries(predicted, anchor_cid=None)
    assert [r["cid"] for r in ranked] == ["G_SLOW", "G_FAST"]   # predicted total wins, not median

    report = wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=results,
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
    ranked = wtune.tune.rank_predicted_entries(predicted, anchor_cid="ANCHOR")
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

    ranked_abc = wtune.tune.rank_predicted_entries(make(["A", "B", "C"]), anchor_cid=None)
    ranked_cba = wtune.tune.rank_predicted_entries(make(["C", "B", "A"]), anchor_cid=None)

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
    ranked = wtune.tune.rank_predicted_entries(predicted, anchor_cid="ANCHOR")
    assert [r["cid"] for r in ranked] == ["ANCHOR", "G_FAST"]


def test_render_report_text_shows_predicted_total_hms_column(tmp_path):
    assert wtune.tune.predict is not None, "wtune predict must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    ctx.owned_jobs = ["1001"]
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    report = wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                                      confirm_results=None, anchor_cid="ANCHOR")
    hms = report["ranking"][0]["predicted_total_hms"]
    assert hms and re.match(r"^\d+:\d{2}(:\d{2})?$", hms)
    assert report["ranking"][0]["predicted_total_s"] > 0

    text = wtune.tune.render_report_text(report)
    assert "pred_total" in text        # header column
    assert hms in text                 # the actual h:mm:ss value appears in the rendered table


def test_build_prediction_never_called_with_include_io_true(tmp_path, monkeypatch):
    assert wtune.tune.predict is not None, "wtune predict must be importable for this test"
    namelist_path = tmp_path / "namelist.input"
    namelist_path.write_text(_PREDICT_TEST_NAMELIST)
    best_config = dict(id="ANCHOR", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    ctx = _ctx(namelist_path=str(namelist_path),
               case_json=dict(MATTHEW_LIKE, prep_dir=str(tmp_path)), out_dir=str(tmp_path / "out"))
    all_results = {"ANCHOR": _result("ANCHOR", 0.93, best_config)}

    real_build_prediction = wtune.tune.predict.build_prediction
    seen_include_io = []

    def spy(*args, **kwargs):
        seen_include_io.append(kwargs.get("include_io", False))
        return real_build_prediction(*args, **kwargs)

    monkeypatch.setattr(wtune.tune.predict, "build_prediction", spy)

    wtune.tune.build_report(ctx, nodes=1, nodes_note="test", all_results=all_results,
                             confirm_results=None, anchor_cid="ANCHOR")

    assert seen_include_io, "build_prediction was never called (test fixture is stale)"
    assert all(v is False for v in seen_include_io)


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
    monkeypatch.setattr(wtune.config, "load_config",
                         _patched_load_config(dict(case="matthew", search=dict(omp=[0]), bogus=1)))
    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--minutes", "30", "--config", str(tmp_path / "wtune.yaml")])
    with pytest.raises(SystemExit) as exc_info:
        wtune.tune.run(args)
    msg = str(exc_info.value)
    assert "search.omp" in msg
    assert "bogus" in msg


def test_run_requires_case_from_cli_or_config():
    args = wtune.tune.build_arg_parser().parse_args(["--minutes", "30"])
    with pytest.raises(SystemExit, match="case is required"):
        wtune.tune.run(args)


def test_run_requires_minutes_from_cli_or_config():
    args = wtune.tune.build_arg_parser().parse_args(["matthew"])
    with pytest.raises(SystemExit, match="--minutes is required"):
        wtune.tune.run(args)


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


def _sweep_nodes(job_nodes, report):
    """node counts of the sweep jobs only (their out dir is .../nodes-n<N>)."""
    return sorted(n for j, n in job_nodes.items() if j in report["node_selection"]["sweep_jobs"])


def test_nodes_search_is_default_and_picks_n_on_nonmonotone_curve(monkeypatch, tmp_path, capsys):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path,
                                      ["--minutes", "600", "--hotspot-fill", "off"])
    ns = report["node_selection"]
    assert ns["mode"] == "search" and ns["objective"] == "time"
    assert ns["chosen"] == 4                       # min median at 4; 8 is slower
    assert report["nodes"] == 4
    out = capsys.readouterr().out
    assert "UNCALIBRATED" not in out
    # seeds + refinement of 4 first; under the cap the rest are measured after the sweep
    assert "measuring the N never sent: [6, 7, 9, 10, 12]" in out      # 11, 13: no legal layout
    assert [r["nodes"] for r in ns["table"]] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]


def test_nodes_search_early_stop_does_not_submit_larger_n(monkeypatch, tmp_path, capsys):
    # big grid: every N (13 included) has a legal pure-MPI layout; early stop keeps 13 out of the
    # sweep, and only the cap's measurement of the N never sent reaches it
    _run_cli(monkeypatch, tmp_path, ["--minutes", "600"], grid=dict(e_we=400, e_sn=400, e_vert=45))
    sweep, _sep, rest = capsys.readouterr().out.partition("measuring the N never sent: ")
    assert "early stop" in sweep and "nodes-n13 " not in sweep
    assert "nodes-n13 submitted" in rest


ISOLATED_CURVE = {**NONMONO_CURVE, 7: 0.20}      # 7 alone is fast: outside the 3..5 bracket


def test_cheap_untried_n_are_measured_before_the_finalists(monkeypatch, tmp_path, capsys):
    report, _jn, _sc = _run_cli(monkeypatch, tmp_path,
                                ["-O2", "--minutes", "600", "--hotspot-fill", "off"],
                                curve=ISOLATED_CURVE)
    out = capsys.readouterr().out
    assert "measuring them before the finalists" in out
    assert out.index("measuring the N never sent") < out.index("finalist N=7 ")
    assert report["node_selection"]["finalists"] == [7] and report["nodes"] == 7


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


def test_queue_cap_while_measuring_untried_n_keeps_the_sweep(monkeypatch, tmp_path, capsys):
    report, job_nodes, cancelled = _run_cli(
        monkeypatch, tmp_path, ["-O2", "--minutes", "600", "--hotspot-fill", "off",
                                "--max-queue-hours", "1"], curve=ISOLATED_CURVE, pending_nodes={7})
    out = capsys.readouterr().out
    assert "measuring them before the finalists" in out and "finalist N=" not in out
    assert report["nodes"] == 4 and not report.get("pipeline_aborted")
    assert report["nodes_note"].endswith("queue cap reached -- ANCHOR config")
    assert [job_nodes[j] for c in cancelled for j in c] == [7]
    assert job_nodes[max(job_nodes, key=int)] == 7


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


def test_a_stop_in_the_first_wave_keeps_its_finished_jobs(monkeypatch, tmp_path):
    stop = wtune.tune.context.STOP
    monkeypatch.setitem(stop, "reason", None)

    def on_sbatch(cmd, nodes):                     # N=4 sits in the queue when the stop comes
        if nodes == 4:
            stop["reason"] = "stopped by Ctrl-C"
    report, job_nodes, cancelled = _run_cli(
        monkeypatch, tmp_path, ["-O2", "--minutes", "600", "--hotspot-fill", "off"],
        pending_nodes={4}, on_sbatch=on_sbatch)
    assert report["nodes"] in (1, 2) and not report.get("pipeline_aborted")
    assert [job_nodes[j] for c in cancelled for j in c] == [4]


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


def test_a_stop_request_cancels_the_jobs_in_flight_at_the_next_poll(monkeypatch):
    monkeypatch.setitem(wtune.tune.context.STOP, "reason", "stopped by Ctrl-C")
    ctx = _ctx(clock=FakeClock(), sacct_runner=make_pending_sacct(None), budget_s=100.0,
               poll_interval=60.0)
    ctx.budget_mode, ctx.shared_queue = "running", {"s": 0.0, "cap_hit": False}
    with pytest.raises(wtune.tune.context.StopRequested, match="stopped by Ctrl-C"):
        wtune.tune.wait_for_jobs(ctx, ["3"], "t")
    assert ctx.shared_queue["cap_hit"] and ctx.cancelled_jobs == ["3"]


def test_without_a_time_cap_the_untried_n_stay_untried(monkeypatch, tmp_path, capsys):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["-O2", "--hotspot-fill", "off"],
                                      curve=ISOLATED_CURVE)
    assert "measuring the N never sent" not in capsys.readouterr().out
    assert 7 not in job_nodes.values() and report["nodes"] == 4


def test_nodes_search_finalists_are_searched_at_two_node_counts(monkeypatch, tmp_path):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    ns = report["node_selection"]
    assert ns["finalists"] == [4, 5]               # 5 (0.305) is the closest to 4 (0.30)
    # a round0 job (not the sweep's own) was submitted at each finalist N
    sweep_jobs = set(ns["sweep_jobs"])
    search_nodes = {n for j, n in job_nodes.items() if j not in sweep_jobs}
    assert search_nodes == {4, 5}
    assert set(ns["finalist_best"]) == {"4", "5"}
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
    # 2-minute budget: the 10 s/step no-ledger fallback makes even N=1 (120 s startup + 5 steps
    # x 10 s = 170 s, one run) overrun -> every N skipped, nothing submitted, proxy fallback.
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "2"])
    ns = report["node_selection"]
    assert ns["sweep_jobs"] == [] and ns["table"] == []     # no sweep job was submitted
    assert ns["skipped"] and all(("exceed the budget" in s["reason"] or "no legal" in s["reason"]
                                  or "not measured" in s["reason"]) for s in ns["skipped"])
    assert any(s["nodes"] == 1 and "exceed the budget" in s["reason"] for s in ns["skipped"])
    assert "falling back to the proxy" in capsys.readouterr().out


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


def test_node_seeds_all_measures_the_n_the_geometric_walk_misses(tmp_path):
    # fastest at N=6, but N=7 is >3% slower than N=8: the walk from seeds 1, 2, 4, 8 stops at 8
    curve = {1: 1.0, 2: 0.55, 3: 0.6, 4: 0.4, 5: 0.5, 6: 0.25, 7: 0.45, 8: 0.3}
    seen = {}
    for seeds in ("geometric", "all"):
        sbatch, reader, job_nodes = make_scaling_fakes(curve)
        ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, max_nodes=8,
                   node_seeds=seeds, out_dir=str(tmp_path / seeds / "o"),
                   ledger_dir=str(tmp_path / seeds / "l"))
        seen[seeds] = set(wtune.tune.run_node_sweep(ctx, list(curve), budget_frac=1.0)
                          ["measurements"])
    assert 6 not in seen["geometric"]
    assert seen["all"] == set(curve)


def test_hotspot_fill_measures_more_n_in_the_idle_nodes_of_a_wave(tmp_path, capsys):
    curve = {n: 1.0 / n for n in range(1, 14)}
    sbatch, reader, job_nodes = make_scaling_fakes(curve)
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, max_nodes=13,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, list(curve), budget_frac=1.0)
    first_wave = [job_nodes[j] for j in sorted(job_nodes)][:4]
    assert first_wave == [1, 2, 4, 6]            # seeds 1, 2, 4 leave 6 of 13 nodes idle
    assert "hotspot fill: N=[6] in the idle nodes of wave [1, 2, 4]" in capsys.readouterr().out
    assert {1, 2, 4, 6, 8} <= set(sw["measurements"])


def test_sweep_steps_follow_the_cost_of_one_step():
    ctx = _ctx(steps=30, startup_s=120.0)
    assert [wtune.tune.node_search._sweep_steps(ctx, s) for s in (0.01, 6.0, 30.0)] == [40, 20, 10]
    ctx = _ctx(steps=30, probe_cfg=dict(wtune.config.DEFAULTS["probe"], sweep_steps=None))
    assert wtune.tune.node_search._sweep_steps(ctx, 6.0) == 30


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


def test_single_run_sweeps_run_the_close_calls_once_more(tmp_path, capsys):
    sbatch, reader, job_nodes = make_scaling_fakes(OPT6_CURVE)
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0, max_nodes=13,
               case_json=dict(BIG_GRID, prep_dir="/fake/big"), hotspot_fill=False,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, list(range(1, 14)), budget_frac=1.0)
    # 6 (0.27) is the fastest; 7 (0.28) is within 2 x 3% of it, 5 (0.29) is not
    assert "node sweep: measuring [6, 7] once more" in capsys.readouterr().out
    assert [len(sw["measurements"][n]) for n in (5, 6, 7)] == [2, 4, 4]
    assert len(sw["race_jobs"]) == 2


def test_o2_runs_the_layouts_that_tie_with_the_fastest_once_more(tmp_path, capsys):
    from wtune.tune import candidates, rounds
    sbatch, reader, _jn = make_scaling_fakes({1: 0.30})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    a = candidates._sweep_anchor(ctx, 1)[0]
    res = {cid: dict(cid=cid, config=dict(a, id=cid), s_values=[m], median=m, lo=m, hi=m, n=1,
                     run_ids=["r"]) for cid, m in (("A", 0.30), ("B", 0.31), ("C", 0.40))}
    rounds._race_leaders(ctx, res, 0.3)
    assert "race: submitted job 7001 (A, B once more" in capsys.readouterr().out
    assert [res[c]["n"] for c in "ABC"] == [3, 3, 1]        # the fake reports 2 runs per job


def test_node_sweep_candidates_on_matthew_like_grid_use_omp(tmp_path):
    # 91x100 grid (r_max 90), 72 cores/node: pure MPI at N>=2 is illegal, but ranks = N*72/omp is
    # legal for omp=2 (N=2), 4 (N=4), 8 (N=8) -> all are sweep candidates with a hybrid ANCHOR.
    sbatch, reader, job_nodes = make_scaling_fakes({1: 1.0, 2: 0.5, 4: 0.3, 8: 0.2, 13: 0.1})
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=360000.0,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, [1, 2, 4, 8, 13], budget_frac=1.0)
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
    sw = wtune.tune.run_node_sweep(ctx, [1, 2, 4, 8], budget_frac=1.0)
    assert sorted(sw["measurements"]) == [1, 2, 4]
    assert [s["nodes"] for s in sw["skipped"]] == [8]
    assert sw["skipped"][0]["reason"].startswith("no legal decomposition at any omp in {1, 2, 3")
    assert sw["anchors"][1]["omp"] == 18


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


def test_nodes_search_stops_after_sweep_when_budget_exhausted(monkeypatch, tmp_path):
    # budget just covers the sweep jobs (elapsed jumps past it once they finish): finalists get
    # no budget, so the report is the chosen N with the sweep's ANCHOR config.
    case_json = dict(MATTHEW_LIKE, prep_dir=str(tmp_path / "matthew"))
    sbatch, reader, job_nodes = make_scaling_fakes(_interp({1: 1.0, 2: 0.5}))
    clock = FakeClock()

    def reader_then_exhaust(case, job_ids):
        rows = reader(case, job_ids)
        clock.t = 10 ** 6
        return rows

    ctx = _ctx(hotspot_fill=False, sbatch_runner=sbatch, ledger_reader=reader_then_exhaust, clock=clock,
               case_json=case_json, budget_s=36000.0, out_dir=str(tmp_path / "o"),
               ledger_dir=str(tmp_path / "l"))
    report = wtune.tune.run_node_search(
        ctx, _profile([{"nodes": 1, "legal": True}, {"nodes": 2, "legal": True}]),
        objective="time", deadline_s=None, min_efficiency=0.7, k_finalists=2, total_steps=100)
    assert report["nodes"] == 2 and "budget exhausted after the sweep" in report["nodes_note"]
    assert report["best"]["id"] == "ANCHOR"
    assert set(job_nodes.values()) == {1, 2, 4}   # nothing but the seed sweep (4 stopped early)


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


def test_node_sweep_budget_cap_skips_larger_n(tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes(NONMONO_CURVE)
    clock = FakeClock()

    def slow_reader(case, job_ids):
        rows = reader(case, job_ids)
        clock.t += 15000.0                   # first wave "takes" 15000 s of a 36000 s budget
        return rows

    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=slow_reader, clock=clock, budget_s=36000.0,
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, [1, 2, 4, 8], budget_frac=0.4)
    assert sorted(set(job_nodes.values())) == [1, 2, 4]      # wave [1,2,4] ran, 8 did not
    capped = {s["nodes"] for s in sw["skipped"] if "sweep budget cap" in s["reason"]}
    assert capped == {8}
    # a roomier cap lets the next wave run
    ctx2 = _ctx(sbatch_runner=sbatch, ledger_reader=slow_reader, clock=FakeClock(),
                budget_s=36000.0, out_dir=str(tmp_path / "o2"), ledger_dir=str(tmp_path / "l"))
    clock.t = 0.0
    ctx2.clock = clock
    ctx2.start_time = 0.0
    sw2 = wtune.tune.run_node_sweep(ctx2, [1, 2, 4, 8], budget_frac=0.9)
    assert 8 in sw2["measurements"]


def test_sweep_budget_frac_cli_and_config_default(monkeypatch, tmp_path):
    args = wtune.tune.build_arg_parser().parse_args(["matthew", "--sweep-budget-frac", "0.25"])
    assert args.sweep_budget_frac == 0.25
    assert wtune.tune.DEFAULT_SWEEP_BUDGET_FRAC == 0.4
    assert wtune.tune.build_arg_parser().parse_args(["matthew"]).sweep_budget_frac is None


def test_finalist_round0_uses_the_sweep_hybrid_anchor(monkeypatch, tmp_path):
    report, job_nodes, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    assert report["node_selection"]["finalists"] == [4, 5]
    # round0 job configs at each finalist N: ANCHOR is the hybrid layout using all cores
    for n, omp in ((4, 4), (5, 4)):
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
    _jobs, by_cid, _rows = wtune.tune.run_round0(ctx)
    assert by_cid["ANCHOR"] == ov
    others = [c for cid, c in by_cid.items() if cid != "ANCHOR"]
    assert others and all(wtune.tune.sweep._identity(c) != wtune.tune.sweep._identity(ov)
                          for c in others)


def test_node_search_builds_full_run_prediction_once_for_the_winner(monkeypatch, tmp_path):
    calls = []

    def fake_pred(ctx, best_config, nodes, all_results, confirm_results, probe_measurements=None):
        calls.append((nodes, best_config and best_config.get("id")))
        return "PREDICTION-FOR-%d" % nodes

    monkeypatch.setattr(wtune.tune.prediction, "build_full_run_prediction", fake_pred)
    report, _jn, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "600"])
    assert len(report["node_selection"]["finalists"]) == 2     # two finalists were searched
    assert len(calls) == 1                                      # ...but one prediction build
    assert calls[0][0] == report["nodes"]
    assert report["full_run_prediction"] == "PREDICTION-FOR-%d" % report["nodes"]


# ---------------------------------------------------------------------------
# coarse-to-fine node search, cores-per-node dimension, MaxRSS
# ---------------------------------------------------------------------------
BIG_GRID = dict(e_we=400, e_sn=400, e_vert=45)       # every N in 1..13 has a legal layout
OPT6_CURVE = _interp({1: 1.0, 2: 0.55, 4: 0.33, 5: 0.29, 6: 0.27, 7: 0.28, 8: 0.30, 13: 0.50})


def test_node_sweep_coarse_to_fine_finds_optimum_at_6(tmp_path):
    sbatch, reader, job_nodes = make_scaling_fakes(OPT6_CURVE)
    ctx = _ctx(sbatch_runner=sbatch, ledger_reader=reader, budget_s=3600000.0,
               case_json=dict(BIG_GRID, prep_dir="/fake"),
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, list(range(1, 14)), budget_frac=1.0)
    measured = set(sw["measurements"])
    assert {1, 2, 4, 8, 13} <= measured                 # seeds
    assert {5, 6, 7} <= measured                        # refinement around 8 -> 7 -> 6 -> 5
    assert wtune_nodes_choose(sw) == 6
    assert 3 not in measured and 10 not in measured and 11 not in measured
    why = {s["nodes"]: s["reason"] for s in sw["skipped"]}
    assert "outside refinement bracket" in why[3] and "outside refinement bracket" in why[11]
    assert set(why) == set(range(1, 14)) - measured     # every untested N is reported


def wtune_nodes_choose(sw):
    import wtune.nodes
    return wtune.nodes.choose(wtune.nodes.summarize(sw["measurements"]), "time")[0]


def test_node_sweep_refinement_stops_once_the_best_neighbours_are_measured(tmp_path):
    # optimum at seed 4: neighbours 3 and 5 are measured and worse -> 6, 7 (inside the seed
    # bracket 2..8 but beyond 5) are NOT measured and reported against the final bracket 3..5
    sbatch, reader, job_nodes = make_scaling_fakes(
        _interp({1: 1.0, 2: 0.55, 4: 0.30, 5: 0.34, 8: 0.45, 13: 0.60}))
    ctx = _ctx(hotspot_fill=False, sbatch_runner=sbatch, ledger_reader=reader, budget_s=3600000.0,
               case_json=dict(BIG_GRID, prep_dir="/fake"),
               out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"))
    sw = wtune.tune.run_node_sweep(ctx, list(range(1, 14)), budget_frac=1.0)
    assert {3, 5} <= set(sw["measurements"]) and 6 not in sw["measurements"]
    why = {s["nodes"]: s["reason"] for s in sw["skipped"]}
    assert "outside refinement bracket (3..5" in why[6] and "outside refinement bracket" in why[10]


def _cores_cli(monkeypatch, tmp_path, mult):
    calls = []
    report, job_nodes, _sc = _run_cli(
        monkeypatch, tmp_path, ["--minutes", "6000", "--node-finalists", "1"], grid=BIG_GRID,
        core_mult=mult, sbatch_calls=calls)
    return report, job_nodes, calls


def test_cores_per_node_dimension_picks_64_when_clearly_faster(monkeypatch, tmp_path):
    mult = lambda c: {72: 1.0, 70: 1.0, 64: 0.90, 48: 1.2, 36: 1.5}.get(c, 9.0)
    report, job_nodes, calls = _cores_cli(monkeypatch, tmp_path, mult)
    ns = report["node_selection"]
    n = ns["final"]
    assert n == 4 and ns["core_choice"][str(n)]["cores"] == 64
    assert report["cores_used_per_node"] == 64 and ns["final_cores_per_node"] == 64
    table = ns["core_tables"][str(n)]
    assert [r["cores"] for r in table] == [72, 70, 64, 48, 36]
    row64 = next(r for r in table if r["cores"] == 64)
    assert row64["chosen"] and row64["ranks_per_node"] * row64["omp"] == 64
    # the finalist's round0 ANCHOR uses 64 cores per node: ranks * omp == N * 64
    with open(tmp_path / "out" / ("n%d" % n) / "round0-s0" / "configs.json") as f:
        a = {c["id"]: c for c in json.load(f)["configs"]}["ANCHOR"]
    assert a["ranks"] * a["omp"] == n * 64
    # report text carries the core-use table and the chosen marker
    assert "cores per node at N=%d" % n in wtune.tune.render_report_text(report)


def test_cores_per_node_tie_within_noise_prefers_the_fuller_c(monkeypatch, tmp_path):
    mult = lambda c: {72: 1.0, 70: 0.995, 64: 0.99, 48: 1.2, 36: 1.5}.get(c, 9.0)
    report, _jn, _calls = _cores_cli(monkeypatch, tmp_path, mult)
    ns = report["node_selection"]
    assert ns["core_choice"][str(ns["final"])]["cores"] == 72
    assert report["cores_used_per_node"] == 72


def test_cores_option_disables_dimension_and_validates(monkeypatch, tmp_path):
    report, job_nodes, _sc = _run_cli(
        monkeypatch, tmp_path, ["--minutes", "6000", "--cores-per-node-options", "72"],
        grid=BIG_GRID)
    assert report["node_selection"]["core_tables"] == {}
    assert not (tmp_path / "out" / "cores-n4-c64").exists()     # no cores job directories
    with pytest.raises(SystemExit, match="cores-per-node-options"):
        _run_cli(monkeypatch, tmp_path, ["--minutes", "30", "--cores-per-node-options", "99"])


def test_undersubscription_sbatch_keeps_the_full_node_and_config_carries_the_ranks(
        monkeypatch, tmp_path):
    mult = lambda c: {64: 0.90}.get(c, 1.0)
    _report, _jn, calls = _cores_cli(monkeypatch, tmp_path, mult)
    c64 = [c for c in calls if "cores-n4-c64" in c]
    assert len(c64) == 1
    line = c64[0]
    # whole nodes are still allocated exclusively with the hardware task count; WTUNE_TOTAL_CORES
    # stays nodes*cores_per_node; the 64-core layout lives in the config (ranks*omp = 64*N)
    assert "-N 4 --ntasks-per-node=72 --exclusive" in line
    assert "WTUNE_TOTAL_CORES=288" in line
    with open(re.search(r"CONFIG_FILE=(\S+)", line).group(1)) as f:
        a = json.load(f)["configs"][0]
    assert a["id"] == "ANCHOR" and a["ranks"] * a["omp"] == 4 * 64


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
                 n_points=3, planner=None):
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


def test_pipeline_full_phase_order_and_rebuild_overlaps_layout_search(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path)
    ev = env.events
    sb = [i for i, e in enumerate(ev) if e == "sbatch"]
    assert ev.index("rebuild_submit") < sb[0]             # rebuild submitted BEFORE phase 1 runs
    assert ev.index("rebuild_finish") > sb[-1]            # ...and awaited only after phase 1
    measures = [e for e in ev if isinstance(e, tuple)]
    assert measures and ev.index("rebuild_finish") < ev.index(measures[0])
    assert [p["name"] for p in report["pipeline"]["phases"]] == ["plan", "layout", "predict",
                                                                 "physics"]
    assert set(_status(report).values()) == {"ok"}, _status(report)
    # phase 2: one measurement per distinct (N, c) group (finalists 4 and 5), all 3 points each;
    # phase 3 last: PH candidates at exactly two points
    assert sorted(m[1] for m in measures[:2]) == [4, 5] and measures[0][3] == [0, 1, 2]
    assert measures[-1][2] == ["PH1", "PH2", "PH3"] and len(measures[-1][3]) == 2   # no EO1/EO2


def test_pipeline_full_physics_uses_three_candidates_at_t0_and_the_costliest_point(
        monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path)
    plan = [c for c in env.physics_calls if c[0] == "plan"][0]
    assert plan[1] == 3 and plan[2] == "matthew"          # max_candidates=3, case_name
    last = [e for e in env.events if isinstance(e, tuple)][-1]
    assert last[3] == [0, 1]                               # earliest (0) + costliest (1: 0.40)
    pipe = report["pipeline"]
    assert pipe["physics_points"] == [env.pts[0].at.isoformat(), env.pts[1].at.isoformat()]
    rows = {r["id"]: r for r in pipe["physics_advice"]}
    assert list(rows) == ["PH1", "PH2", "PH3", "EO1", "EO2"]
    assert all(rows[i]["measured"] for i in ("PH1", "PH2", "PH3"))
    assert all(not rows[i]["measured"] and rows[i]["recommendation"].startswith("estimate only")
               and rows[i]["fidelity"] == "not run" and rows[i]["prior_saving_pct"] == 12.0
               for i in ("EO1", "EO2"))
    assert pipe["physics_advice"][0]["predicted_saving_pct"] == pytest.approx(10.0)
    assert [c for c in env.physics_calls if c[0] == "fidelity"]       # the gate ran


def test_pipeline_full_report_fields_budget_and_queue_columns(monkeypatch, tmp_path, capsys):
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000)
    pipe = report["pipeline"]
    B = 6000 * 60.0
    assert pipe["B_s"] == B and pipe["budget_source"].startswith("--minutes")
    assert "alpha" not in pipe and report["budget_minutes"] == 6000
    assert pipe["T_updated_s"] == pytest.approx(10 ** 6 * 0.30 + 120.0, rel=0.05)   # info only
    for ph in pipe["phases"]:
        for k in ("start_s", "end_s", "running_s", "wall_s", "queue_wait_s", "limit_s", "status",
                  "controller_s", "queue_s_max", "job_run_s_max"):
            assert k in ph, k
    assert [p["limit_s"] for p in pipe["phases"]] == pytest.approx(
        [0.05 * B, 0.55 * B, 0.80 * B, B])                 # 5 / 50 / 25 / 20 %
    pr = pipe["prediction"]
    # 0.8.2: combine()'s 1030 = compute 1020 + predict_init 10; round0's measured startup
    # replaces the init (no namelist here -> no run-length I/O)
    assert pr["init_source"].startswith("measured round0 startup")
    assert pr["predicted_total_s"] == pytest.approx(1020.0 + pr["init_s"]) and pr["time_suggestion"]
    assert pr["nodes"] == report["nodes"]
    assert "p1" not in pipe                           # no tune-time/run-time verdict
    assert pipe["hard_deadline_s"] == B and pipe["queue_cap_hit"] is False
    out = capsys.readouterr().out
    assert "wtune tune -O3" in out and "queue wait" in out and "physics advice" in out
    with open(tmp_path / "out" / "tune_report.json") as f:
        assert json.load(f)["pipeline"]["B_s"] == B


def test_no_time_cap_by_default_and_max_nodes_is_the_cluster(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=None)
    pipe = report["pipeline"]
    assert pipe["B_s"] is None and pipe["hard_deadline_s"] is None
    assert pipe["budget_source"] == "no time cap" and report["budget_minutes"] is None
    assert set(_status(report).values()) == {"ok"}                 # nothing skipped for budget
    assert report["node_selection"]["table"][-1]["nodes"] >= 8     # waves use up to 13 nodes
    json.dumps(report["pipeline"])                                  # no inf/nan leaks into the json


def test_pipeline_full_no_physics_advice_rolls_its_share_back(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000, physics_advice=False)
    pipe = report["pipeline"]
    assert _status(report)["physics"].startswith("not run (physics advice is opt-in")
    assert pipe["physics_advice"] is None
    sh = pipe["shares"]
    assert sh["physics"] == 0.0 and sh["layout"] == pytest.approx(0.5 + 0.2 * 2 / 3)
    assert sh["predict"] == pytest.approx(0.25 + 0.2 / 3) and sum(sh.values()) == pytest.approx(1.0)
    lims = [p["limit_s"] for p in pipe["phases"]]
    assert lims[2] == lims[3] == pytest.approx(6000 * 60.0)       # prediction runs to B
    assert not any(isinstance(e, tuple) and any(c.startswith("PH") for c in e[2])
                   for e in env.events)


def test_pipeline_full_budget_truncates_phases_2_and_3(monkeypatch, tmp_path):
    # the first sweep wave 'runs' for 85% of B of job time: every later
    # phase-1 job is skipped for budget, and phases 2 and 3 are past their limits
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000, bump_after_first_read=0.85)
    ph = _status(report)
    assert ph["predict"].startswith("not run (budget)")
    assert report["pipeline"]["prediction"] is None
    assert report["nodes"]                                          # layout result still reported
    # phase 3 is NOT skipped because phase 2 was: unused time rolls forward. It runs on phase 1's
    # best layout, without a phase-2 twin, so the gate is the provisional config-gate
    assert ph["physics"] == "ok"
    pipe = report["pipeline"]
    assert pipe["physics_fidelity_gate"].startswith("config-gate, PROVISIONAL")
    assert pipe["physics_baseline"].startswith("phase-1 best layout")
    assert [r["id"] for r in pipe["physics_advice"] if r["measured"]] == ["PH1", "PH2", "PH3"]


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


def test_pipeline_full_skips_physics_for_extension_cases(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, extension=True)
    ph = _status(report)
    assert ph["physics"].startswith("not run (extension case")
    assert report["pipeline"]["extension"] is True and ph["predict"] == "ok"
    assert not any(isinstance(e, tuple) and any(c.startswith("PH") for c in e[2])
                   for e in env.events)


def test_pipeline_full_physics_not_run_without_runner_or_module(monkeypatch, tmp_path):
    report, _e = _pipe_run(monkeypatch, tmp_path, runner_ok=False)
    assert _status(report)["physics"].startswith(
        "not run (campaign runner has no per-config physics namelists")
    report, _e = _pipe_run(monkeypatch, tmp_path, physics=False)
    assert _status(report)["physics"] == "not run (wtune.physics_advice not importable)"


def test_pipeline_full_queue_wait_is_not_charged_to_the_budget(monkeypatch, tmp_path):
    # every job is PENDING for 6 polls (20 s each, the clock advances by them) before it
    # completes: far more queue wait than the 30-minute budget over the whole run, yet nothing is
    # cancelled or skipped for budget -- only RUNNING time counts
    scancel, scancel_calls = make_fake_scancel()
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=30, sacct=make_pending_sacct(12),
                            scancel=scancel)
    pipe = report["pipeline"]
    assert pipe["total_queue_wait_s"] > 30 * 60                     # more queue than budget
    assert pipe["total_running_s"] == pytest.approx(0.0)           # nothing was RUNNING
    assert pipe["total_wall_s"] >= pipe["total_queue_wait_s"]
    assert set(_status(report).values()) == {"ok"}, _status(report)
    assert scancel_calls == [] and not pipe["queue_cap_hit"]
    layout = next(p for p in pipe["phases"] if p["name"] == "layout")
    assert layout["queue_wait_s"] > 0 and layout["running_s"] < layout["wall_s"]


def test_pipeline_full_controller_work_is_not_charged_and_is_reported(monkeypatch, tmp_path):
    # report building is slow (900 s of controller time per finalist) but the
    # budget clock only counts running jobs: running time stays 0, controller time shows them
    real_build = wtune.tune.build_report

    def slow_build(ctx, *a, **k):
        ctx.clock.t += 900.0
        return real_build(ctx, *a, **k)

    monkeypatch.setattr(wtune.tune.report, "build_report", slow_build)
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=30)
    pipe = report["pipeline"]
    assert pipe["total_running_s"] == pytest.approx(0.0)
    assert pipe["total_controller_s"] >= 900.0 * 2                  # >= 2 finalist report builds
    assert pipe["total_wall_s"] == pytest.approx(
        pipe["total_running_s"] + pipe["total_queue_wait_s"] + pipe["total_controller_s"])
    assert set(_status(report).values()) == {"ok"}                  # nothing skipped for budget
    assert sum(p["controller_s"] for p in pipe["phases"]) == pytest.approx(
        pipe["total_controller_s"])
    assert pipe["over_deadline_s"] == 0.0


def test_pipeline_full_max_queue_hours_stops_the_run(monkeypatch, tmp_path, capsys):
    scancel, scancel_calls = make_fake_scancel()
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=30, sacct=make_pending_sacct(None),
                            extra_args=["--max-queue-hours", "1"], scancel=scancel)
    pipe = report["pipeline"]
    assert report["pipeline_aborted"] and pipe["queue_cap_hit"]
    ph = _status(report)
    assert ph["layout"].startswith("stopped: queue-only wait")
    assert ph["predict"] == "not run (max queue wait exceeded)"
    assert ph["physics"] == "not run (max queue wait exceeded)"
    assert pipe["total_queue_wait_s"] >= 3600 and scancel_calls          # in-flight jobs cancelled
    assert "STOPPED: max queue wait exceeded" in capsys.readouterr().out


def test_wait_for_jobs_running_budget_ignores_pending_and_raises_at_the_queue_cap():
    clock = FakeClock()
    ctx = _ctx(clock=clock, sacct_runner=make_pending_sacct(5), budget_s=100.0, poll_interval=60.0)
    ctx.budget_mode = "running"
    ctx.shared_queue = {"s": 0.0, "cap_hit": False}
    states = wtune.tune.wait_for_jobs(ctx, ["1", "2"], "t")        # 5 PENDING polls = 300 s of queue
    assert states["1"][0] == "COMPLETED"
    assert ctx.queue_wait_s == pytest.approx(300.0) and ctx.elapsed() == pytest.approx(0.0)
    ctx2 = _ctx(clock=FakeClock(), sacct_runner=make_pending_sacct(None), budget_s=100.0,
                poll_interval=60.0)
    ctx2.budget_mode, ctx2.shared_queue, ctx2.max_queue_s = "running", {"s": 0.0, "cap_hit": False}, 600.0
    with pytest.raises(wtune.tune.QueueCapExceeded, match="max-queue-hours"):
        wtune.tune.wait_for_jobs(ctx2, ["3"], "t")
    assert ctx2.shared_queue["cap_hit"] and ctx2.cancelled_jobs == ["3"]


def test_wait_for_jobs_charges_only_the_intervals_a_job_is_running():
    seen = {"n": 0}

    def sacct(cmd):
        if "--format=JobID,State,Elapsed" not in cmd:
            return ""
        seen["n"] += 1
        state = "PENDING" if seen["n"] <= 3 else ("RUNNING" if seen["n"] <= 6 else "COMPLETED")
        return "1|%s|00:00:00" % state

    ctx = _ctx(clock=FakeClock(), sacct_runner=sacct, budget_s=10000.0, poll_interval=60.0)
    ctx.budget_mode = "running"
    ctx.shared_queue = {"s": 0.0, "cap_hit": False, "run_s": 0.0}
    ctx.clock.t = 0.0
    wtune.tune.wait_for_jobs(ctx, ["1"], "t")
    assert ctx.queue_wait_s == pytest.approx(180.0)                 # 3 polls PENDING
    assert ctx.elapsed() == pytest.approx(180.0)                    # 3 polls RUNNING: the budget
    ctx.clock.t += 5000.0                                           # controller work: not charged
    assert ctx.elapsed() == pytest.approx(180.0)
    assert ctx.wall_elapsed() == pytest.approx(180.0 + 180.0 + 5000.0)


def test_pipeline_full_rejects_fixed_nodes(monkeypatch, tmp_path):
    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--pipeline", "full", "--minutes", "30", "--nodes", "4"])
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                        lambda case, cpn: (dict(BIG_GRID, prep_dir="/x"), "/x/n", ALL_NODES_PROFILE))
    with pytest.raises(SystemExit, match="runs its own node search"):
        wtune.tune.run(args)


def test_physics_runner_supported_reads_the_campaign_files(tmp_path):
    (tmp_path / "strict_campaign.sbatch").write_text("echo nothing\n")
    (tmp_path / "configs.py").write_text("# no marker\n")
    script = str(tmp_path / "strict_campaign.sbatch")
    assert wtune.tune.physics_runner_supported(script) is False
    (tmp_path / "configs.py").write_text("cfg.get('physics_namelists')\n")
    assert wtune.tune.physics_runner_supported(script) is True


def test_pipeline_full_window_frames_only_where_phase3_needs_them(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path)
    p2 = [(a, c) for a, c in env.measure_cfgs if not any(x["id"].startswith("PH") for x in c)]
    p3 = [(a, c) for a, c in env.measure_cfgs if any(x["id"].startswith("PH") for x in c)]
    assert p2 and p3
    for anchor, cfgs in p2:                    # phase 2: twin runs write mid + end frames
        assert anchor["window_frame"] == "mid_end"
        assert all(c["window_frame"] == "mid_end" for c in cfgs)
    anchor, cfgs = p3[0]                       # phase 3: baseline and every candidate
    assert anchor["window_frame"] == "mid_end" and all(c["window_frame"] == "mid_end" for c in cfgs)
    # elsewhere frames cost I/O: not requested without phase 3
    report, env = _pipe_run(monkeypatch, tmp_path, physics_advice=False)
    assert all("window_frame" not in a and all("window_frame" not in c for c in cs)
               for a, cs in env.measure_cfgs)
    report, env = _pipe_run(monkeypatch, tmp_path, extension=True)
    assert all("window_frame" not in a for a, cs in env.measure_cfgs)


def test_pipeline_full_gate_gets_frame_lists_prognostic_fields_and_frames_are_deleted(
        monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path)
    fid = [c for c in env.physics_calls if c[0] == "fidelity"]
    assert fid and all(c[1]["judged_fields"][0] == "T" for c in fid)
    assert all(c[1]["out_prefix"].endswith(tuple("_pt%d" % i for i in range(3))) for c in fid)
    assert report["pipeline"]["frames_deleted"] > 0 and report["pipeline"]["frames_kept"] is False
    left = [f for _r, _d, fs in os.walk(tmp_path / "ledger") for f in fs if f.endswith(".nc")]
    assert left == []                                       # window frames removed after the gate


def test_pipeline_full_keep_frames_keeps_them(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, extra_args=["--keep-frames"])
    assert report["pipeline"]["frames_kept"] is True and report["pipeline"]["frames_deleted"] == 0
    left = [f for _r, _d, fs in os.walk(tmp_path / "ledger") for f in fs if f.endswith(".nc")]
    assert left


def test_pipeline_full_physics_needs_a_window_of_two_model_minutes(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, steps=1)          # 1 step x 60 s window
    assert _status(report)["physics"].startswith("not run (window < 2 min")


def test_cli_help_renders():
    # argparse %-formats every help string: an unescaped % crashed `--help` on scc
    text = wtune.tune.build_arg_parser().format_help()
    assert "--pipeline" in text and "--keep-frames" in text and "--max-queue-hours" in text


def test_points_by_cli_flag_parses_and_defaults_to_none():
    p = wtune.tune.build_arg_parser()
    assert p.parse_args(["matthew", "--site", "scc"]).points_by is None
    assert p.parse_args(["matthew", "--site", "scc", "--points-by", "cost-indicators"]).points_by \
        == "cost-indicators"
    assert "--points-by" in p.format_help()


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


def test_preflight_down_node_is_not_probed_and_excluded():
    states = {"work1": "down*", "work2": "idle", "work3": "drained", "work4": "idle"}
    runner, log = make_health_runner(states)
    h = _preflight(runner, requested=4)
    probed = [a[a.index("-w") + 1] for a in log if a[0] == "srun"]
    assert sorted(probed) == ["work2", "work4"]                 # down/drained: no probe
    assert h["failed"] == ["work1", "work3"]
    assert h["nodes"]["work1"]["reason"] == "sinfo state down*" and not h["nodes"]["work1"]["ok"]
    assert h["healthy"] == 2 and h["max_nodes_effective"] == 2 and h["max_nodes_requested"] == 4
    assert os.environ["WTUNE_PREFLIGHT_EXCLUDE"] == "work1,work3"
    assert "head" not in h["nodes"]
    assert h["exclusions"] == ["head", "work1", "work3"]


def test_preflight_no_home_node_and_probe_command_shape():
    states = {"work2": "idle", "work5": "idle"}
    runner, log = make_health_runner(states, bad={"work2": "no $HOME"})
    h = _preflight(runner, requested=13)
    assert h["nodes"]["work2"] == dict(state="idle", ok=False, probed=True,
                                       reason="probe failed rc=3: no $HOME")
    assert h["nodes"]["work5"]["ok"] and h["failed"] == ["work2"]
    srun = next(a for a in log if a[0] == "srun" and "work5" in a)
    assert srun[:13] == ["srun", "-p", "compute", "-w", "work5", "-N1", "-n1", "-t", "1",
                         "--immediate=30", "--job-name=wtune-preflight-%d-work5" % os.getpid(),
                         "--exclude=head", "bash"]
    script = srun[-1]
    for needle in ("$HOME", "/case", "/src", "/wrf/main/wrf.exe"):
        assert needle in script
    assert h["max_nodes_effective"] == 1


def test_preflight_timeout_fails_an_idle_node_but_not_a_busy_one():
    states = {"work2": "idle", "work3": "allocated", "work4": "mixed"}
    runner, _log = make_health_runner(states, bad={n: "timeout" for n in states})
    h = _preflight(runner, requested=13, timeout_s=7)
    assert h["nodes"]["work2"]["reason"] == "probe timed out after 7s" and not h["nodes"]["work2"]["ok"]
    # busy: unprobed. A mixed node need not answer either: on f1-arm (shared nodes, busy queue)
    # every mixed node timed out and got excluded, leaving only the nodes that were taken.
    for n in ("work3", "work4"):
        assert h["nodes"][n]["ok"] and h["nodes"][n]["probed"] is False
    assert h["failed"] == ["work2"] and h["healthy"] == 2


def test_preflight_keeps_nodes_the_scheduler_did_not_allocate_and_cancels_timed_out_probes():
    """scc 2026-10-13: a reserved idle node and a probe over the per-user submit limit were
    excluded as unhealthy, and a killed srun left its job PENDING (job 4921)."""
    states = {"work2": "idle", "work3": "idle", "work4": "idle"}
    runner, log = make_health_runner(states, bad={
        "work2": "srun: error: Unable to allocate resources: Requested nodes are busy",
        "work3": "srun: error: QOSMaxSubmitJobPerUserLimit", "work4": "timeout"})
    h = _preflight(runner, requested=13)
    for n in ("work2", "work3"):
        assert h["nodes"][n]["ok"] and h["nodes"][n]["probed"] is False
        assert h["nodes"][n]["reason"].startswith("not allocated, not probed: srun: error:")
    assert h["failed"] == ["work4"]
    assert ["scancel", "--user=" + getpass.getuser(), "--name=wtune-preflight-%d-work4" % os.getpid()] in log
    assert not any(a[0] == "scancel" and any("work2" in x for x in a) for a in log)


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


def test_the_input_space_check_comes_before_the_node_preflight(monkeypatch, tmp_path):
    """dbbf82d review: a tune that cannot convert its inputs probed every node first."""
    calls = []

    def runner(argv, timeout):
        calls.append(argv)
        raise AssertionError("preflight ran")
    monkeypatch.setattr(wtune.tune.cli._jobs, "check_input_space",
                        lambda case, **k: (_ for _ in ()).throw(SystemExit("no space")))
    with pytest.raises(SystemExit, match="no space"):
        _run_cli(monkeypatch, tmp_path, ["-O1"], preflight_runner=runner)
    assert calls == []


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


def test_preflight_sinfo_unavailable_is_skipped_and_says_so():
    def runner(argv, timeout):
        raise FileNotFoundError("sinfo")

    h = _preflight(runner, requested=13)
    assert h["skipped"].startswith("sinfo unavailable") and h["max_nodes_effective"] == 13
    assert h["failed"] == [] and "WTUNE_PREFLIGHT_EXCLUDE" not in os.environ


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


def test_no_node_preflight_flag_skips_it(monkeypatch, tmp_path):
    runner, log = make_health_runner({"work1": "down*"})
    report, _jn, _sc = _run_cli(monkeypatch, tmp_path, ["--minutes", "30", "--nodes", "2",
                                                       "--no-node-preflight"],
                                preflight_runner=runner)
    assert log == [] and "node_health" not in report


def test_pipeline_runs_the_preflight_first_and_again_before_phase_2(monkeypatch, tmp_path):
    runner, log = make_health_runner({"work%d" % i: "idle" for i in range(1, 14)})
    events = []

    def logging_runner(argv, timeout):
        if argv[0] == "sinfo":
            events.append("preflight")
        return runner(argv, timeout)

    report, env = _pipe_run(monkeypatch, tmp_path, preflight_runner=logging_runner)
    ev = env.events
    assert events == ["preflight", "preflight"]
    nh = report["node_health"]
    assert [c["when"] for c in nh["checks"]] == ["start", "before phase 2"]
    assert set(_status(report).values()) == {"ok"}


# ---------------------------------------------------------------------------
# phase-2 fit (matthew-pipe3: 9:20 left, phase 2 "not run (budget)"), short-run guard, ordering
# ---------------------------------------------------------------------------
def _matthew_rebuilt(n=5):
    t0 = _dt.datetime(2016, 10, 6)
    return [dict(index=i, point=_types.SimpleNamespace(at=t0 + _dt.timedelta(hours=3 * i),
                                                        reason="r"),
                 case_id="mp%d" % i, T=t0, steps=32, spinup_min=20, stepra=6, dt=150.0)
            for i in range(n)]


def test_phase2_estimate_formula_reproduces_the_9m20_skip():
    # matthew-pipe3: 5 points, N=4 -> concurrency floor(10/4)=2, 0.0164 s/step, 32 window steps
    # (rebuilt_steps(stepra=6)), spin-up 20 min = 8 steps at dt=150, restart read 30 s, startup 120 s
    # (the DEFAULT: the measured value lived on the finalist ctx copies and was never adopted)
    # est = ceil(points/concurrency) waves x (spin_s + sum(per_run_s)),
    #   per_run_s = startup + restart_read + steps*s*rad_factor ; spin_s = startup + spin_steps*s*rf
    ctx = _ctx(nodes=4, max_nodes=10, startup_s=120.0)
    ctx.stage1_max_running = 2
    anchor = dict(id="ANCHOR", ranks=288, omp=1, nx=16, ny=18, numtiles=1, dist="block")
    other = dict(id="L1", ranks=216, omp=1, nx=12, ny=18, numtiles=1, dist="block")
    s_by = {"ANCHOR": 0.0164, "L1": 0.0164}
    rebuilt = _matthew_rebuilt(5)
    top, est, skip = wtune.tune.fit_stage2(ctx, [other], anchor, rebuilt, s_by, None,
                                           left_s=9 * 60 + 20)
    m = wtune.tune.ranking_cost_model(120.0, 30.0, 32, 8, 6, 0.0164)
    assert est == pytest.approx(3 * (m["spin_s"] + 3 * m["per_run_s"]))   # 3 waves x (spin + 3 runs)
    assert est == pytest.approx(1718, abs=3) and top == [] and skip     # 28.6 min vs 9:20 left
    # even the measured ~50 s startup does not make "2 layouts x 5 points" fit 9:20 ...
    ctx.startup_s = 50.0
    assert wtune.tune.fit_stage2(ctx, [other], anchor, rebuilt, s_by, None, left_s=560.0)[2]
    # ... so phase 2 must degrade (fewer points / winner only) instead of being skipped
    lays = [dict(nodes=4, cores=72, cfg=anchor, median=0.0164),
            dict(nodes=3, cores=72, cfg=other, median=0.0166)]
    plans = wtune.tune.phase2_plans(lays, rebuilt, None)
    assert [p[0] for p in plans] == [
        "2 layout(s) x all 5 points", "2 layout(s) x 2 points (t0 + costliest)",
        "winner only x all 5 points", "winner only x 2 points (t0 + costliest)",
        "winner only x 1 point (t0)"]
    assert [len(p[2]) for p in plans] == [5, 2, 5, 2, 1]


def test_pick_points_keeps_t0_and_the_costliest():
    rb = _matthew_rebuilt(5)
    got = wtune.tune.pick_points(rb, 2, s_normal=[0.3, 0.31, 0.5, 0.32, 0.33])
    assert [rp["index"] for rp in got] == [0, 2]                   # earliest + costliest measured
    assert [rp["index"] for rp in wtune.tune.pick_points(rb, 1)] == [0]
    assert len(wtune.tune.pick_points(rb, 9)) == 5


def test_pipeline_phase2_degrades_instead_of_skipping_when_it_does_not_fit(monkeypatch, tmp_path):
    # B = 700 s -> the phase-2 limit is 560 s (9:20) and phase 1 used no running time
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=700 / 60.0, n_points=5,
                            extra_args=("--hotspot-fill", "off"))
    pipe = report["pipeline"]
    plan = pipe["predict_plan"]
    assert plan["tried"][0]["plan"].startswith("2 layout(s) x all 5 points")
    assert plan["tried"][0]["estimate_s"] > plan["left_s"]            # would have been skipped
    assert plan["degraded"] and plan["chosen"] != plan["tried"][0]["plan"]
    ph = _status(report)
    assert ph["predict"].startswith("partial: degraded to"), ph
    assert pipe["prediction"] is not None and pipe["prediction"]["degraded"] is True
    assert pipe["prediction"]["points_used"] < 5
    assert ph["physics"].startswith(("ok", "partial"))


def test_pipeline_phase2_not_run_only_when_even_the_smallest_plan_does_not_fit(
        monkeypatch, tmp_path):
    # the first sweep wave "runs" for 79% of B: 7 s of the phase-2 share are left
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=700 / 60.0, n_points=5,
                            bump_after_first_read=0.79, extra_args=("--hotspot-fill", "off"))
    ph = _status(report)
    assert ph["predict"].startswith("not run (budget): the smallest plan")
    assert report["pipeline"]["prediction"] is None
    assert not [e for e in env.events if isinstance(e, tuple) and "PH1" not in e[2]]


# ---------------------------------------------------------------------------
# effort levels, pre-run estimate, step-down, prediction points from the case type
# ---------------------------------------------------------------------------
def test_level_estimates_sc26p1_like_numbers_and_printout():
    # SC26-P1 live: N=1 0.1261 s/step; startup 120 s default; 30 probe steps; 10 healthy nodes
    lay = wtune.tune.layout_estimates(s1=0.1261, steps_probe=30, startup_s=120.0, max_nodes=10)
    job = lambda n: 2 * (120.0 + 30 * 0.1261 / n)                       # noqa: E731
    # t0 wave N in {1,2,4,8,10} packs into 3 waves ([1,2,4], [8], [10])
    assert lay[1] == pytest.approx(job(1) + job(8) + job(10))
    assert lay[0] == 0.0 and lay[1] < lay[2] < lay[3]
    est = wtune.tune.level_estimates(s1=0.1261, steps_probe=30, startup_s=120.0, max_nodes=10,
                                     n_points=2)
    assert est[1] < est[2] < est[3] and all(est[k] >= lay[k] for k in (1, 2, 3))
    text = wtune.tune.format_level_estimates(est, 2)
    assert text.startswith("O1 ≈ ") and "O2 ≈ " in text and "O3 ≈ " in text and "<-" in text
    assert text.endswith("(running time; queue wait extra)")
    # more prediction points (a case type that needs them) cost more, at every level
    est5 = wtune.tune.level_estimates(s1=0.1261, steps_probe=30, startup_s=120.0, max_nodes=10,
                                      n_points=5)
    assert all(est5[k] > est[k] for k in (1, 2, 3))
    assert wtune.tune.level_estimates(s1=0.1261, steps_probe=30, startup_s=120.0, max_nodes=10,
                                      n_points=0)[3] < est[3]


def test_highest_level_within_never_below_o1():
    est = {0: 0.0, 1: 300.0, 2: 700.0, 3: 2000.0}
    assert wtune.tune.highest_level_within(est, 5000.0, 3) == 3
    assert wtune.tune.highest_level_within(est, 800.0, 3) == 2
    assert wtune.tune.highest_level_within(est, 400.0, 3) == 1
    assert wtune.tune.highest_level_within(est, 10.0, 3) == 1          # the floor: one wave
    assert wtune.tune.highest_level_within(est, 5000.0, 2) == 2        # never above the request


def test_time_suggestion_rounds_up_to_15_min_plus_one_band():
    t = wtune.tune.time_suggestion_s
    assert t(25 * 60 + 27, 0.15) / 60 == pytest.approx(30 + 0.15 * 25.45, abs=0.01)   # 33.8 min
    assert wtune.tune._fmt_sbatch_time(t(25 * 60 + 27, 0.15)) == "00:34:00"
    assert t(10 * 3600, 0.07) == pytest.approx(1.2 * 10 * 3600)         # 10 h: the 1.2 x T floor
    assert t(60.0, 0.25) >= 1.2 * 60.0 and t(60.0, 0.25) >= 15 * 60    # tiny run: one 15 min step


def test_fallback_point_plan_bands_and_sc26p1_vs_10h_runs():
    f = wtune.tune._fallback_point_plan
    # SC26-P1-like: T ~ 25 min, 15 min precision: even the cold start (+-25% = 6.3 min) suffices
    assert f(25 * 60.0, 15 * 60.0, False, 5) == (0, "cold-start predict", 0.25)
    assert f(25 * 60.0, 15 * 60.0, True, 5) == (0, "t0 anchor (phase-1 data)", 0.15)
    # 60 min run at 15 min: t0 +-15% = 9 min ok after a measurement, cold start 15 min exactly ok
    assert f(60 * 60.0, 15 * 60.0, True, 5)[0] == 0
    # 2 h: t0 18 min > 15 -> t0 + costliest +-10% = 12 min -> ONE point
    assert f(120 * 60.0, 15 * 60.0, True, 5) == (1, "t0 + costliest point", 0.10)
    # 10 h run needs points: +-10% = 60 min, +-7% = 42 min > 15 -> not meetable: all available
    pts, method, band = f(600 * 60.0, 15 * 60.0, True, 5)
    assert (pts, band) == (5, 0.07) and method == "3+ points"
    # a 4 h run: +-7% = 16.8 min > 15 too; at 3 h: 12.6 <= 15 -> 3 points
    assert f(180 * 60.0, 15 * 60.0, True, 5) == (3, "3+ points", 0.07)
    assert f(180 * 60.0, 15 * 60.0, True, 2)[0] == 2                    # capped by availability


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


def test_plan_prediction_points_convection_case_gets_t0_plus_the_peak(monkeypatch):
    from wtune import cost_drivers as real
    monkeypatch.setattr(wtune.tune.prediction, "cost_drivers", real)
    monkeypatch.setattr(wtune.tune.prediction, "_case_types_and_track", lambda ctx: (["convection"], None))
    t0 = _dt.datetime(2026, 1, 1)
    hours = list(range(25))
    # fake indicator series: precipitation area peaks at hour 14
    precip = [1.0 + 10.0 * math.exp(-((h - 14) ** 2) / 4.0) for h in hours]
    cs = real.CostSeries(t0, [t0 + _dt.timedelta(hours=h) for h in hours],
                         dict(precip_area=precip), {"precip_area": 1.0}, [], {})
    assert cs.usable
    monkeypatch.setattr(wtune.tune.prediction, "_cost_series_for_ctx", lambda *a, **k: cs)
    wrapped = _hourly_points(30, t0)
    # a 90 min run at 15 min precision: t0-only convection (+-25% = 22 min) is too loose, the
    # t0 + peak base set (+-10% = 9 min) is enough
    r = wtune.tune.plan_prediction_points(_planner_ctx(), wrapped, 15.0, 90 * 60.0, False,
                                          run_start=t0, run_end=t0 + _dt.timedelta(hours=24),
                                          window_s=3600.0)
    assert r["source"] == "wtune.cost_drivers" and r["band"] == pytest.approx(0.10)
    assert [p[1] for p in r["points"]] == ["t0", "convective_peak"]
    assert [w["index"] for w in r["selected"]] == [14]          # needs_rebuild == [the peak hour]
    assert r["meets_precision"] is True and r["types"] == ["convection"]
    # the real planner's mapping uses TIME, not an index: a planned point set shifted by 20 min
    shifted = [dict(w, point=_types.SimpleNamespace(at=w["point"].at + _dt.timedelta(minutes=20),
                                                    reason="r")) for w in wrapped]
    r2 = wtune.tune.plan_prediction_points(_planner_ctx(), shifted, 15.0, 90 * 60.0, False,
                                           run_start=t0, run_end=t0 + _dt.timedelta(hours=24),
                                           window_s=3600.0)
    assert [w["index"] for w in r2["selected"]] == [14]          # nearest planned point


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


def test_pre_run_estimate_is_printed_before_any_job_and_actual_time_is_reported(
        monkeypatch, tmp_path, capsys):
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=None, level=2, cold=(40000.0, 0.05))
    out = capsys.readouterr().out
    first_job = min(out.index(m) for m in ("node sweep: nodes-n", "rebuilt point")
                    if m in out)
    i = out.index("預估調校時間 O1 ≈ ")
    assert i < first_job and "O2 ≈ " in out and "O3 ≈ " in out and "queue wait extra" in out
    lv = report["pipeline"]["level"]
    assert lv["requested"] == lv["effective"] == 2 and set(lv["estimate_s"]) == {"O0", "O1", "O2", "O3"}
    assert lv["actual_running_s"] == pytest.approx(report["pipeline"]["total_running_s"])
    assert "ACTUAL running time" in out and "estimated at start" in out


def test_pre_run_step_down_when_the_whole_run_is_shorter_than_the_tuning(monkeypatch, tmp_path,
                                                                        capsys):
    # cold-start whole run 300 s: every level's tuning estimate is longer -> floor O1
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=None, level=3, short_run_guard=True,
                            cold=(300.0, 0.05))
    lv = report["pipeline"]["level"]
    assert lv["requested"] == 3 and lv["effective"] == 1
    assert lv["stepped_down"][0]["when"] == "pre-run"
    assert "自動降為 O1" in capsys.readouterr().out
    assert [r["nodes"] for r in report["node_selection"]["table"]][:3] == [1, 2, 4]
    assert report["node_selection"]["finalists"] == [] and report["pipeline"]["prediction"]
    # --force-full keeps the requested level
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=None, level=3, cold=(300.0, 0.05))
    assert report["pipeline"]["level"]["effective"] == 3 and not report["pipeline"]["level"]["stepped_down"]


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


def test_levels_only_change_the_layout_search_and_physics(monkeypatch, tmp_path):
    r1, e1 = _pipe_run(monkeypatch, tmp_path / "a", level=1)
    r2, e2 = _pipe_run(monkeypatch, tmp_path / "b", level=2)
    r3, e3 = _pipe_run(monkeypatch, tmp_path / "c", level=3)
    t = lambda r: [x["nodes"] for x in r["node_selection"]["table"]]       # noqa: E731
    assert 3 not in t(r1) and 3 in t(r2) and 3 in t(r3)                     # refinement from O2
    assert r1["node_selection"]["finalists"] == [] and r2["node_selection"]["finalists"] == [4]
    assert len(r3["node_selection"]["finalists"]) == 2                      # top-2 layouts at O3
    assert not (tmp_path / "b" / "out" / "n4" / "confirm").exists()          # no confirm at O2
    assert (tmp_path / "c" / "out" / "n4" / "confirm").exists()
    # prediction points: the SAME for every level (the planner decides), here all 3
    for e in (e1, e2, e3):
        assert [m[3] for m in e.events if isinstance(m, tuple)][0] == [0, 1, 2]
    # physics advice only at O3
    assert _status(r1)["physics"].startswith("not run (physics advice needs -O3)")
    assert _status(r2)["physics"].startswith("not run (physics advice needs -O3)")
    assert _status(r3)["physics"] == "ok"
    # O1/O2 (and O3 without --physics-advice would too) predict the winner layout only: ONE fused
    # point job per point
    assert all(m[2] == ["W0"] for m in e1.events if isinstance(m, tuple))
    assert set(e1.measure_modes) == {"fused"} and set(e2.measure_modes) == {"fused"}


def test_case_type_planner_decides_points_and_surplus_rebuilds_are_cancelled(monkeypatch,
                                                                            tmp_path, capsys):
    # cold start asks for all 3 points (nothing measured), phase 1's measurement needs only 1
    def planner(planned, T, call):
        return (planned[:1], 0.10) if call >= 1 else (planned, 0.07)

    report, env = _pipe_run(monkeypatch, tmp_path, planner=planner)
    assert [c["call"] for c in env.planner_calls] == [0, 1]      # cold-start T, then measured T
    assert env.planner_calls[0]["T"] != env.planner_calls[1]["T"]
    assert env.planner_calls[0]["precision_min"] == 15.0
    pp = report["pipeline"]["prediction_plan"]
    assert pp["points"] == [0] and pp["measured"] is True and pp["rebuild_cancelled"]
    assert env.planner_calls[0]["window_s"] > 0 and env.planner_calls[0]["types"] == ["common"]
    cancelled = [j for c in env.scancel_calls for j in c]
    assert sorted(cancelled) == sorted(pp["rebuild_cancelled"]) and len(cancelled) == 2
    assert "cancelling 2 surplus rebuild job(s)" in capsys.readouterr().out
    meas = [m for m in env.events if isinstance(m, tuple)][0]
    assert meas[3] == [0]                                               # phase 2 at ONE point
    pred = report["pipeline"]["prediction"]
    assert pred["band"] == 0.10 and report["pipeline"]["prediction_band"]["band"] == 0.10
    assert pred["time_suggestion"] == wtune.tune._fmt_sbatch_time(
        wtune.tune.time_suggestion_s(pred["predicted_total_s"], 0.10))


def test_case_type_planner_needing_no_point_means_no_rebuild_and_no_phase_2(monkeypatch,
                                                                           tmp_path):
    # SC26-P1-like: the t0 anchor meets the precision -> no rebuild job is ever submitted
    report, env = _pipe_run(monkeypatch, tmp_path, planner=lambda planned, T, call: ([], 0.15))
    assert "rebuild_submit" not in env.events and not env.scancel_calls
    ph = _status(report)
    assert ph["predict"].startswith("not run (the t0 anchor suffices for this case at ±15 min")
    assert ph["physics"].startswith("not run (no rebuilt points)")
    pred = report["pipeline"]["prediction"]
    assert pred["points_used"] == 0 and pred["band"] == 0.15
    # 0.8.2: T_updated_s's startup term is the one in effect at the sweep (the default); the
    # prediction uses round0's measured startup instead
    assert pred["predicted_total_s"] == pytest.approx(
        report["pipeline"]["T_updated_compute_s"] + pred["init_s"])
    assert pred["init_source"].startswith("measured round0 startup")
    lo, hi = pred["predicted_range_s"]
    assert lo == pytest.approx(0.85 * pred["predicted_total_s"]) and hi > pred["predicted_total_s"]
    assert report["pipeline"]["time_suggestion"] == wtune.tune._fmt_sbatch_time(
        wtune.tune.time_suggestion_s(pred["predicted_total_s"], 0.15))


def test_precision_flag_reaches_the_planner(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, extra_args=["--precision", "30"])
    assert {c["precision_min"] for c in env.planner_calls} == {30.0}


def test_o0_runs_no_job_and_reports_the_cold_start_band(monkeypatch, tmp_path, capsys):
    case_json = dict(BIG_GRID, prep_dir=str(tmp_path / "matthew"))
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                        lambda case, cpn: (case_json, str(tmp_path / "namelist.input"),
                                           ALL_NODES_PROFILE))
    monkeypatch.setattr(wtune.tune.context, "_total_steps", lambda nl: 1000)
    monkeypatch.setattr(wtune.tune.prediction, "cold_start_prediction",
                        lambda ctx, nodes, cfg, steps: (1687.0, 0.05, "fake cold start"))

    def no_sbatch(cmd):
        raise AssertionError("O0 must not submit anything")

    def no_probe(argv, timeout):
        raise AssertionError("O0 must not probe nodes (srun)")

    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--site", "scc", "--out", str(tmp_path / "out"), "-O0",
         "--ledger-dir", str(tmp_path / "ledger")])
    report = wtune.tune.run(args, sbatch_runner=no_sbatch, preflight_runner=no_probe,
                            clock=FakeClock())
    pred = report["pipeline"]["prediction"]
    assert report["pipeline_only"] and report["pipeline"]["level"]["effective"] == 0
    assert pred["predicted_total_s"] == 1687.0 and pred["band"] == 0.25
    assert pred["predicted_range_s"] == pytest.approx([1687 * 0.75, 1687 * 1.25])
    assert pred["time_suggestion"] == wtune.tune._fmt_sbatch_time(
        wtune.tune.time_suggestion_s(1687.0, 0.25))
    out = capsys.readouterr().out
    assert "O0: no jobs" in out and "±25%" in out and "whole-run time ±25%" in out
    with open(tmp_path / "out" / "tune_report.json") as f:
        assert json.load(f)["pipeline"]["prediction_band"]["band"] == 0.25


# --- 0.8.2: run-length I/O in the pipeline's whole-run prediction ---------------------------
FAKE_IO = dict(total_s=600.0, history_s=400.0, boundary_s=100.0, aux_s=60.0, restart_s=40.0)


def _with_fake_io(monkeypatch, io=FAKE_IO):
    calls = []

    def fake(path):
        calls.append(path)
        return dict(io)
    monkeypatch.setattr(wtune.tune.prediction, "pipeline_run_length_io", fake)
    return calls


def test_pipeline_winner_prediction_adds_run_length_io(monkeypatch, tmp_path, capsys):
    calls = _with_fake_io(monkeypatch)
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000)
    pipe = report["pipeline"]
    pr = pipe["prediction"]
    assert calls == [str(tmp_path / "namelist.input")]
    # combine(): 1030 = compute 1020 + predict_init 10 -> init := round0's measured startup
    init = pr["init_s"]
    assert pr["init_source"].startswith("measured round0 startup (") and init != 10.0
    T = 1020.0 + init + 600.0
    assert pr["compute_s"] == pytest.approx(1020.0) and pr["io_s"] == 600.0
    assert pr["predicted_total_s"] == pytest.approx(T)
    assert pr["predicted_total_hms"] == wtune.tune._fmt_hms(T)
    assert pr["time_suggestion"] == pipe["time_suggestion"] == wtune.tune._fmt_sbatch_time(
        wtune.tune.time_suggestion_s(T, pr["band"]))
    assert pr["error_min"] == pytest.approx(pr["band"] * T / 60.0)
    assert pipe["predicted_real_s"] == pytest.approx(T)
    assert pipe["predicted_real_source"].endswith("+ run-length I/O")
    lo, hi = pr["predicted_range_s"]
    shift = T - 1030.0
    assert (lo, hi) == pytest.approx((1030.0 * 0.95 + shift, 1030.0 * 1.05 + shift))
    assert all(c["predicted_total_s"] > 600.0 for c in pr["candidates"])
    assert pr["breakdown"].startswith("compute 1020s + history 400s + boundary 100s + aux 60s "
                                      "+ restart 40s + init ")
    assert pr["breakdown"].endswith("= %.0fs" % T)
    assert "each extra run segment adds ~%.0f s" % init in pr["segment_note"]
    out = capsys.readouterr().out
    assert "  " + pr["breakdown"] + "  [init: measured round0 startup (" in out
    assert "NOTE: each extra run segment adds" in out
    assert "whole run: " + pr["breakdown"] in report["full_run_prediction"]
    with open(tmp_path / "out" / "tune_report.json") as f:
        saved = json.load(f)["pipeline"]["prediction"]
    assert saved["predicted_total_s"] == pytest.approx(T) and saved["io"]["total_s"] == 600
    assert saved["init_s"] == pytest.approx(init) and saved["init_source"] == pr["init_source"]


def test_pipeline_t0_anchor_prediction_adds_run_length_io(monkeypatch, tmp_path):
    _with_fake_io(monkeypatch)
    report, env = _pipe_run(monkeypatch, tmp_path, planner=lambda planned, T, call: ([], 0.15))
    pred = report["pipeline"]["prediction"]
    T = report["pipeline"]["T_updated_compute_s"] + pred["init_s"] + 600.0
    assert pred["points_used"] == 0 and pred["init_source"].startswith("measured round0")
    assert pred["predicted_total_s"] == pytest.approx(T)
    assert pred["predicted_range_s"] == pytest.approx([0.85 * T, 1.15 * T])
    assert report["pipeline"]["time_suggestion"] == wtune.tune._fmt_sbatch_time(
        wtune.tune.time_suggestion_s(T, 0.15))


def test_o0_prediction_adds_run_length_io(monkeypatch, tmp_path, capsys):
    _with_fake_io(monkeypatch)
    case_json = dict(BIG_GRID, prep_dir=str(tmp_path / "matthew"))
    monkeypatch.setattr(wtune.tune.context, "resolve_case",
                        lambda case, cpn: (case_json, str(tmp_path / "namelist.input"),
                                           ALL_NODES_PROFILE))
    monkeypatch.setattr(wtune.tune.context, "_total_steps", lambda nl: 1000)
    monkeypatch.setattr(wtune.tune.prediction, "cold_start_prediction",
                        lambda ctx, nodes, cfg, steps: (1687.0, 0.05, "fake cold start"))
    args = wtune.tune.build_arg_parser().parse_args(
        ["matthew", "--site", "scc", "--out", str(tmp_path / "out"), "-O0",
         "--ledger-dir", str(tmp_path / "ledger")])
    report = wtune.tune.run(args, sbatch_runner=lambda cmd: None, clock=FakeClock(),
                            preflight_runner=lambda argv, timeout: None)
    pred = report["pipeline"]["prediction"]
    assert pred["predicted_total_s"] == pytest.approx(2287.0)
    assert pred["predicted_range_s"] == pytest.approx([2287 * 0.75, 2287 * 1.25])
    assert report["pipeline"]["time_suggestion"] == pred["time_suggestion"] == \
        wtune.tune._fmt_sbatch_time(wtune.tune.time_suggestion_s(2287.0, 0.25))
    assert pred["breakdown"].endswith("= 2287s") and "fake cold start + run-length I/O" == \
        pred["source"]


def test_add_run_length_io_without_a_model_keeps_the_total():
    pred = dict(predicted_total_s=1000.0, predicted_range_s=[900.0, 1100.0],
                candidates=[dict(predicted_total_s=1000.0)])
    wtune.tune.add_run_length_io(pred, None, 880.0, 120.0, 0.10)
    assert pred["predicted_total_s"] == 1000.0 and pred["predicted_range_s"] == [900.0, 1100.0]
    assert pred["candidates"][0]["predicted_total_s"] == 1000.0
    assert "NOT included" in pred["breakdown"] and pred["segment_note"] is None


def test_prediction_init_prefers_the_measured_round0_startup():
    ctx = _types.SimpleNamespace(startup_s=120.0, startup_runs=None,
                                 startup_source="documented constant (scc_sweep.STARTUP_S)")
    assert wtune.tune.prediction_init(ctx, 125.6, "wtune.predict.predict_init") == (
        125.6, "wtune.predict.predict_init")
    ctx.startup_s, ctx.startup_runs = 49.9, 5
    ctx.startup_source = "measured in phase 1 (median of 2 finalist(s))"
    assert wtune.tune.prediction_init(ctx, 125.6, "wtune.predict.predict_init") == (
        49.9, "measured round0 startup (5 runs)")


def test_pipeline_run_length_io_reads_the_case_namelist(tmp_path, capsys):
    p = tmp_path / "namelist.input"
    p.write_text(" &time_control\n run_hours = 48,\n start_year = 2005, start_month = 08,"
                 " start_day = 27,\n interval_seconds = 21600\n history_interval = 60,\n"
                 " restart_interval = 100000,\n /\n &domains\n time_step = 72, max_dom = 1,"
                 " e_we = 170, e_sn = 130, e_vert = 40,\n /\n &physics\n radt = 12,\n /\n")
    io = wtune.tune.pipeline_run_length_io(str(p), io="netcdf")
    assert io["total_s"] == pytest.approx(41.6, abs=0.1)        # SC26-P2: 49 frames + 8 bdy
    ad = wtune.tune.pipeline_run_length_io(str(p))              # default: ADIOS2
    assert ad["io_format"] == "adios2" and ad["total_s"] < io["total_s"]
    assert wtune.tune.pipeline_run_length_io(str(tmp_path / "missing")) is None
    assert "run-length I/O model unavailable" in capsys.readouterr().out


def test_report_prints_node_health_before_the_pipeline_summary(monkeypatch, tmp_path, capsys):
    runner, _log = make_health_runner({"work%d" % i: "idle" for i in range(1, 14)})
    report, env = _pipe_run(monkeypatch, tmp_path, preflight_runner=runner)
    out = capsys.readouterr().out
    assert out.index("--- node health pre-flight") < out.index("=== wtune tune -O3")
    assert "check before phase 2" in out and "check start" in out


def test_phase1_measured_startup_is_adopted_by_phase_2(monkeypatch, tmp_path):
    # round0 measures the startup on the finalist ctx copies; the default 120 s made phase 2 look
    # 2-4x too expensive (matthew-pipe3)
    monkeypatch.setattr(wtune.tune.budget, "measure_startup_s", lambda *a, **k: (55.0, 3))
    report, env = _pipe_run(monkeypatch, tmp_path, n_points=5)
    assert report["pipeline"]["predict_plan"]["startup_s"] == 55.0


# ---------------------------------------------------------------------------
# default level = full layout search; opt-in guard / physics; point-job modes
# ---------------------------------------------------------------------------
SC26_LIKE = dict(e_we=150, e_sn=150, e_vert=45, max_dom=2, e_we_list=[150, 100],
                 e_sn_list=[150, 100])


def test_full_space_enumeration_covers_every_axis_and_sc26p1_counts():
    # SC26-P1: d01 150x150 + d02 100x100 (r_max 100), 72 cores/node
    counts = {}
    for n in (1, 4, 8, 10):
        ctx = _ctx(case_json=dict(SC26_LIKE, prep_dir="/x"), nodes=n, cores_per_node=72,
                   hardware_id="scc-gh200")
        configs, info = wtune.tune.enumerate_full_space(ctx)
        counts[n] = len(configs)
        assert info["raw"] > info["unique"] == info["legal"] == len(configs) and not info["excluded"]
        idents = {wtune.tune.sweep._identity(c) for c in configs}
        assert len(idents) == len(configs)                          # no duplicate layout
        assert all(c["dist"] == "block" and c["blocktime"] == "0" for c in configs)
        assert all(c["ranks"] * c["omp"] <= n * 72 and c["nx"] * c["ny"] == c["ranks"]
                   for c in configs)
        # every domain legal: floor(extent / nproc) >= 10 on d01 AND d02
        assert all(150 // c["nx"] >= 10 and 100 // c["nx"] >= 10 and 150 // c["ny"] >= 10
                   and 100 // c["ny"] >= 10 for c in configs)
        axes = dict(omp={c["omp"] for c in configs}, tiles={c["numtiles"] for c in configs},
                    shapes={(c["nx"], c["ny"]) for c in configs}, ranks={c["ranks"] for c in configs})
        assert len(axes["omp"]) >= 5 and len(axes["tiles"]) >= 3     # threads and tiles vary
        assert len(axes["shapes"]) > len(axes["ranks"]) and len(axes["ranks"]) > 1   # nx x ny shapes
    # SC26-P1; includes the omp=72 layouts with numtiles == omp (one tile per thread)
    assert counts == {1: 340, 4: 247, 8: 161, 10: 200}


def test_default_level_enumerates_the_full_layout_space_at_the_finalists(monkeypatch, tmp_path):
    # no cap, no guard, no physics: the finalists' round0 measures EVERY enumerated layout
    report, env = _pipe_run(monkeypatch, tmp_path, full_cap=None, grid=MATTHEW_LIKE,
                            physics_advice=False)
    pipe, ns = report["pipeline"], report["node_selection"]
    assert pipe["B_s"] is None and not pipe["level"]["stepped_down"]
    finalists = ns["finalists"]
    assert len(finalists) == 2
    enumerated = ns["enumerated"]
    assert set(enumerated) == {"N=%d c=72" % n for n in finalists}
    measured = [r for r in ns["layouts"] if r["source"] == "finalist search"]
    per_n = {n: {(r["ranks"], r["omp"], r["nx"], r["ny"], r["numtiles"])
                 for r in measured if r["nodes"] == n} for n in finalists}
    for n in finalists:
        # every enumerated layout of that N (the ANCHOR override replaces its twin) is in the table
        assert len(per_n[n]) >= enumerated["N=%d c=72" % n]["legal"] - 1
    # no adaptive proposer rounds on an exhausted space; confirm reps exist at the finalists
    assert not [e for e in env.events if e == "adaptive"]
    assert any(r["reps"] >= 2 for r in measured)


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


def test_guard_is_opt_in_and_a_time_cap_does_not_enable_it(monkeypatch, tmp_path):
    # default: no guard even though the whole run (1000 x 0.3 s) is far shorter than the search
    report, env = _pipe_run(monkeypatch, tmp_path)
    assert report["pipeline"]["level"]["effective"] == 3 and not report["pipeline"]["level"]["stepped_down"]
    assert "short_run" not in report["pipeline"]
    # --stop-if-short turns it on
    report, env = _pipe_run(monkeypatch, tmp_path, short_run_guard=True)
    assert report["pipeline"]["level"]["stepped_down"]
    # a --minutes cap alone does not: the cap bounds the search, the level stays as asked
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000, total_steps=1000)
    assert report["pipeline"]["level"]["effective"] == 3 and not report["pipeline"]["level"]["stepped_down"]


def test_a_case_with_no_prediction_point_gives_its_share_to_the_layout_search(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000, n_points=0)
    lims = {p["name"]: p["limit_s"] for p in report["pipeline"]["phases"]}
    assert lims["layout"] == pytest.approx(lims["predict"]) == pytest.approx(0.80 * 6000 * 60.0)
    report, env = _pipe_run(monkeypatch, tmp_path, minutes=6000)          # 3 points: 50 % only
    lims = {p["name"]: p["limit_s"] for p in report["pipeline"]["phases"]}
    assert lims["layout"] == pytest.approx(0.55 * 6000 * 60.0)
    # the removed flags are gone
    for flag in ("--force-full", "--no-physics-advice"):
        with pytest.raises(SystemExit):
            wtune.tune.build_arg_parser().parse_args(["matthew", flag])


def test_physics_advice_is_opt_in(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path, physics_advice=False)
    assert _status(report)["physics"].startswith("not run (physics advice is opt-in")
    assert report["pipeline"]["physics_advice"] is None
    report, env = _pipe_run(monkeypatch, tmp_path)               # --physics-advice given
    assert _status(report)["physics"] in ("ok",) or _status(report)["physics"].startswith("partial")
    assert report["pipeline"]["physics_advice"]


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


def test_level_estimates_grow_with_the_full_space_size():
    kw = dict(s1=0.1261, steps_probe=30, startup_s=120.0, max_nodes=10, n_points=2)
    small = wtune.tune.level_estimates(n_full=10, **kw)
    big = wtune.tune.level_estimates(n_full=350, **kw)
    assert big[3] > 3 * small[3] and big[2] == small[2] and big[1] == small[1]


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


def test_fused_point_job_sets_the_runner_flags_and_skips_the_drift_gate(monkeypatch, tmp_path):
    w0 = dict(id="W0", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block", blocktime="0")
    cmd, written, meas, gates = _point_job(monkeypatch, tmp_path, "fused", [w0])
    for flag in ("POINT_FUSED=1", "ANCHOR_END=0", "WINDOW_EXACT=1", "RANKING_START=spinup",
                 "REPS=1", "MODE=timing"):
        assert flag in cmd, flag
    assert written == ["ANCHOR", "W0"]                  # placeholder + the one candidate, no END
    assert list(meas) == ["W0"] and gates[0]["point_mode"] == "fused" and gates[0]["ok"]
    assert "SPINUP_MINUTES=20" in cmd and cmd.count("SPINUP_MINUTES") == 1     # a single target


def test_two_config_point_job_drops_anchorend_and_the_drift_gate(monkeypatch, tmp_path):
    l1 = dict(id="L1", ranks=64, omp=1, nx=8, ny=8, numtiles=1, dist="block", blocktime="0")
    cmd, written, meas, gates = _point_job(monkeypatch, tmp_path, "no_end", [l1])
    assert "ANCHOR_END=0" in cmd and "POINT_FUSED" not in cmd and "WINDOW_EXACT" not in cmd
    assert written == ["ANCHOR", "L1"] and set(meas) == {"ANCHOR", "L1"}
    assert gates[0]["ok"] and gates[0]["anchor_drift"] is None


def test_multi_config_point_job_keeps_anchorend_and_the_drift_gate(monkeypatch, tmp_path):
    cfgs = [dict(id=i, ranks=64, omp=1, nx=8, ny=8, numtiles=1, dist="block", blocktime="0")
            for i in ("L1", "L2")]
    cmd, written, meas, gates = _point_job(monkeypatch, tmp_path, "default", cfgs)
    assert "ANCHOR_END" not in cmd and "POINT_FUSED" not in cmd
    assert written == ["ANCHOR", "L1", "L2", "ANCHOREND"] and "ANCHOREND" not in meas
    assert gates[0]["point_mode"] == "default" and gates[0]["anchor_drift"] == pytest.approx(0.0)


def test_phase2_uses_fused_for_winner_only_and_default_for_physics(monkeypatch, tmp_path):
    report, env = _pipe_run(monkeypatch, tmp_path)               # O3 + --physics-advice
    modes = [(m[2], mode) for m, mode in zip([e for e in env.events if isinstance(e, tuple)],
                                             env.measure_modes)]
    # with --physics-advice phase 2 asks for window frames (the twin), so no fused path: ANCHOR
    # alone, drift gate off; the physics job has 3+ configs and keeps ANCHOREND
    assert modes[0] == ([], "no_end")
    assert modes[-1][1] in (None, "default") and modes[-1][0][0].startswith("PH")
    # without physics the same plan is ONE fused wrf.exe per point
    report, env = _pipe_run(monkeypatch, tmp_path / "np", physics_advice=False)
    assert set(env.measure_modes) == {"fused"}


def test_layout_row_keeps_the_rank_order_it_was_measured_with():
    from wtune.tune import node_search
    cfg = dict(id="c1", ranks=144, omp=1, nx=12, ny=12, numtiles=1, dist="cyclic")
    assert node_search._layout_row("finalist search", 2, 72, cfg, [0.5, 0.6])["dist"] == "cyclic"


def test_a_fire_case_is_tuned_single_threaded_without_any_config():
    """scc c34 2026-10-14: the fire rule lived only in the wtune.yaml template, so a tune without
    --config swept the node counts at omp 2, 4, 8 -- one probe crashed, one hung for an hour."""
    cfg, errors = wtune.config.resolve_config({})
    assert cfg["exclude"] == [dict(omp_gt=1, when="fire")] and not errors
    assert wtune.config.resolve_config({"exclude": []})[0]["exclude"] == []     # lifted on request
    fire = _ctx(exclude_rules=cfg["exclude"], is_fire=True)
    for n in (1, 2, 4):
        anchor, cores, why = wtune.tune.candidates._sweep_anchor_fitting(fire, n)
        assert anchor["omp"] == 1 and anchor["ranks"] <= 90, why
    assert wtune.tune.candidates._sweep_anchor_fitting(fire, 2)[1] == 45       # 90 ranks on 2 nodes
    other = _ctx(exclude_rules=cfg["exclude"], is_fire=False)
    assert wtune.tune.candidates._sweep_anchor_fitting(other, 2)[0]["omp"] == 2


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


def test_clear_leader_stops_only_when_the_round_found_nothing_better():
    old = {"A": _r("A", 1.00), "B": _r("B", 1.10)}
    # the leader came from an earlier round and leads by more than the noise: stop
    assert "A (1 s/step) leads B" in wtune.tune.rounds._clear_leader(old, {"B": old["B"]}, 0.03)
    # this round produced the leader: keep going
    assert wtune.tune.rounds._clear_leader(old, {"A": old["A"]}, 0.03) is None
    # within the noise: keep going
    close = {"A": _r("A", 1.00), "B": _r("B", 1.02)}
    assert wtune.tune.rounds._clear_leader(close, {"B": close["B"]}, 0.03) is None


def test_adaptive_rounds_stop_early_when_the_leader_is_clear():
    def proposer(state, batch_size, kappa=None):
        return dict(stopped=False, round=1), [dict(id="SLOW%d" % len(state["rounds"]), ranks=72,
                                                   omp=1, nx=8, ny=9, numtiles=1, dist="block",
                                                   blocktime="0")]

    sbatch_runner, _calls, _paths = make_fake_sbatch(start=4000)

    def ledger_reader(case, job_ids):
        return [dict(run_id="%s_SLOW%d_r1_%s" % (case, i, j), job_id=j, s_per_step_median="2.0")
                for i, j in enumerate(job_ids)]

    ctx = _ctx(sbatch_runner=sbatch_runner, ledger_reader=ledger_reader, proposer=proposer,
               budget_s=3600.0)
    all_results = {"BEST": _r("BEST", 1.0)}
    wtune.tune.run_adaptive_rounds(ctx, {}, all_results, s_est=2.0)
    assert len(ctx.owned_jobs) == 1                    # round 1 found nothing faster: stop


def test_summary_says_the_pick_its_run_its_margin_what_was_not_run_and_the_cost():
    from wtune.tune import report as rep
    row = dict(nodes=4, cores_per_node=128, ranks=512, omp=1, nx=16, ny=32, numtiles=1,
               median=4.1282, reps=2)
    r = dict(winner=row, pipeline=dict(predicted_real_s=912.0, predicted_real_source="t0",
                                       total_running_s=1680.0, total_queue_wait_s=600.0),
             node_selection=dict(runner_up=dict(row, cores_per_node=144, ranks=576, nx=24, ny=24,
                                                median=4.2), winner_vs_runner_up=dict(
                                     gap=0.0174, beyond_noise=False, noise=0.03),
                                 skipped=[dict(nodes=1, reason="memory: predicted 300 GiB"),
                                          dict(nodes=2, reason="memory: N=3 ... killed"),
                                          dict(nodes=3, reason="memory: predicted 229 GiB"),
                                          dict(nodes=13, reason="early stop: N=13 ...")]))
    assert rep.summary_lines(r) == [
        "best:     N=4 c/n=128 512 ranks x 1 threads 16x32 tiles=1: 4.1282 s/step (2 runs)",
        "run:      predicted 15:12 for the whole simulation (t0)",
        "margin:   1.7% ahead of N=4 c/n=144 576 ranks x 1 threads 24x24 tiles=1 -> within "
        "noise: either is fine (noise 3%)",
        "not run:  N=13 early stop; N=1-3 memory",
        "tuning:   running 28:00, queue 10:00 (not counted)"]


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


def test_a_tune_inside_a_project_keeps_its_bp_cache_in_the_project(monkeypatch, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "project.yaml").write_text("{}\n")
    monkeypatch.delenv("WTUNE_BP_CACHE", raising=False)
    monkeypatch.delenv("WTUNE_PROJECT", raising=False)
    wtune.tune.cli._use_project_bp_cache(_types.SimpleNamespace(project=str(proj)))
    assert os.environ["WTUNE_BP_CACHE"] == str(proj / "input" / "bp-cache")
    monkeypatch.setenv("WTUNE_BP_CACHE", "/elsewhere")              # an explicit choice wins
    wtune.tune.cli._use_project_bp_cache(_types.SimpleNamespace(project=str(proj)))
    assert os.environ["WTUNE_BP_CACHE"] == "/elsewhere"


def test_confirm_includes_the_fastest_measured_config(monkeypatch):
    """f1-arm conus12 2026-10-14: the prediction ranked block layouts first and the measured
    fastest (cyclic, 0.132 vs 0.187 s/step) was never confirmed."""
    from wtune.tune import rounds

    def entry(cid, median, pred):
        return {"cid": cid, "median": median, "pred_total_s": pred, "config": {"id": cid}}

    predicted = {e["cid"]: e for e in (entry("ANCHOR", 0.17, 100.0), entry("B1", 0.187, 90.0),
                                       entry("B2", 0.19, 91.0), entry("B3", 0.20, 92.0),
                                       entry("C1", 0.132, 95.0))}
    monkeypatch.setattr(rounds, "predict_candidates", lambda ctx, n, res: predicted)
    cids = [e["cid"] for e in rounds.select_confirm_entries(None, 1, {}, "ANCHOR")]
    assert cids == ["B1", "B2", "B3", "C1", "ANCHOR"]
    predicted["C1"]["pred_total_s"] = 80.0                    # already in the top 3: no extra
    cids = [e["cid"] for e in rounds.select_confirm_entries(None, 1, {}, "ANCHOR")]
    assert cids == ["C1", "B1", "B2", "ANCHOR"]
    predicted["C1"]["pred_total_s"] = 95.0                    # the fastest has no config
    predicted["D1"] = dict(entry("D1", 0.10, 99.0), config=None)
    cids = [e["cid"] for e in rounds.select_confirm_entries(None, 1, {}, "ANCHOR")]
    assert cids == ["B1", "B2", "B3", "C1", "ANCHOR"]           # dbbf82d review: C1 was dropped

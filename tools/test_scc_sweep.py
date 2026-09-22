#!/usr/bin/env python3
"""pytest suite for tools/scc_sweep.py.

Anchor legality is checked against predictor/experiment.py's own documented ground truth for
matthew's grid (e_we=91, e_sn=100 -- see that module's _max_legal_ranks/_generated_baseline
docstrings: "matthew 91x100 at total_cores=112: ... true max legal rank count is 90, shape
9x10"), not re-derived here. Run with:
  /home/nk/miniforge3/envs/hqvit-pixi/bin/python3 -m pytest tools/test_scc_sweep.py
"""
import argparse
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scc_sweep as sw  # noqa: E402

MATTHEW_LIKE = dict(e_we=91, e_sn=100, e_vert=45)
TINY_ILLEGAL = dict(e_we=5, e_sn=5, e_vert=10)   # floor(5/1) < MIN_PATCH(10): no legal shape ever


def _args(**over):
    base = dict(
        cases=["synth"], nodes=[1], k=3, seed=1, reps=1, steps=30,
        root=None, dry_run=False, ledger_dir=None, hardware_id="scc-gh200",
        cores_per_node=72, max_nodes_in_flight=13, force=False,
        env_source="$HOME/wtune-scc/env.sh",
        sbatch_script="$HOME/wtune/campaign/strict_campaign.sbatch",
        sacct_cmd="sacct", sbatch_runner=None, sacct_runner=None,
    )
    base.update(over)
    return argparse.Namespace(**base)


def _fake_runners(start_job_id=1000):
    calls = dict(sbatch=[], sacct=[])
    counter = {"n": start_job_id}

    def sbatch_runner(cmd_str):
        calls["sbatch"].append(cmd_str)
        counter["n"] += 1
        return "Submitted batch job %d\n" % counter["n"]

    def sacct_runner(cmd):
        calls["sacct"].append(cmd)
        return ""   # no rows -> every job_id comes back UNKNOWN / not in-flight

    return sbatch_runner, sacct_runner, calls


# ---------------------------------------------------------------------------
# anchor legality
# ---------------------------------------------------------------------------
def test_anchor_matches_documented_ground_truth_112_cores():
    # predictor/experiment.py's own documented example (_max_legal_ranks/_generated_baseline
    # docstrings): matthew's 91x100 grid at total_cores=112 -> max legal ranks is 90, shape 9x10.
    a = sw.build_anchor(MATTHEW_LIKE, total_cores=112, nodes=1, hardware_id="scc-gh200")
    assert a is not None
    assert (a["ranks"], a["nx"], a["ny"]) == (90, 9, 10)
    assert a["omp"] == 1
    assert a["numtiles"] == 1
    assert a["dist"] == "block"          # overridden from _generated_baseline's own 'cyclic'
    assert a["blocktime"] == "0"


def test_anchor_realistic_scc_single_node_72_cores():
    # N=1 node on scc (72 cores/node): 91x100 divides exactly (8x9=72), no downward scan needed.
    a = sw.build_anchor(MATTHEW_LIKE, total_cores=72, nodes=1, hardware_id="scc-gh200")
    assert a is not None
    assert a["ranks"] == 72
    assert (a["nx"], a["ny"]) in ((8, 9), (9, 8))
    assert a["dist"] == "block"
    assert a["numtiles"] == 1


def test_anchor_none_when_totally_illegal():
    assert sw.build_anchor(TINY_ILLEGAL, total_cores=72, nodes=1, hardware_id="scc-gh200") is None


# ---------------------------------------------------------------------------
# skip when legal space empty
# ---------------------------------------------------------------------------
def test_build_configs_skips_when_space_empty():
    built = sw.build_configs("tiny", TINY_ILLEGAL, nodes=1, k=5, seed=1,
                              hardware_id="scc-gh200", cores_per_node=72)
    assert built["configs"] == []
    assert built["space_size"] == 0
    assert built["skip_reason"] is not None
    assert "tiny" in built["skip_reason"]


def test_build_configs_nonempty_case_has_anchor_plus_sampled():
    built = sw.build_configs("matthew", MATTHEW_LIKE, nodes=1, k=5, seed=1,
                              hardware_id="scc-gh200", cores_per_node=72)
    assert built["skip_reason"] is None
    assert built["configs"][0]["id"] == "ANCHOR"
    assert built["n_anchor"] == 1
    assert 1 <= len(built["configs"]) <= 6   # anchor + up to 5 sampled (fewer if deduped)


# ---------------------------------------------------------------------------
# time-limit formatting and cap
# ---------------------------------------------------------------------------
def test_estimate_time_formula():
    # per_run = 120 + 30*2.0 = 180s; total = 5*1*180 = 900s = 0.25h
    # padded = 0.25*1.5 + 10/60 = 0.375 + 0.16667 = 0.541667h -> 32.5min -> ceil 33min
    t = sw.estimate_time(n_configs=5, reps=1, steps=30, s_est=2.0)
    assert t["total_run_s"] == 900.0
    assert abs(t["job_hours"] - 0.5416666666666666) < 1e-9
    assert t["time_limit"] == "00:33:00"
    # timeout = max(600, 3*180) = 600
    assert t["timeout_s"] == 600


def test_estimate_time_timeout_above_floor():
    t = sw.estimate_time(n_configs=1, reps=1, steps=30, s_est=100.0)
    # per_run = 120 + 3000 = 3120; timeout = 3*3120 = 9360 > 600
    assert t["timeout_s"] == 9360


def test_estimate_time_reps_multiplies():
    t1 = sw.estimate_time(n_configs=4, reps=1, steps=30, s_est=2.0)
    t2 = sw.estimate_time(n_configs=4, reps=3, steps=30, s_est=2.0)
    assert abs(t2["total_run_s"] - 3 * t1["total_run_s"]) < 1e-9


def test_estimate_time_cap_at_12h():
    t = sw.estimate_time(n_configs=10000, reps=5, steps=200, s_est=50.0)
    assert t["job_hours"] == 12.0
    assert t["time_limit"] == "12:00:00"
    assert t["uncapped_hours"] > 12.0   # the cap actually did something


def test_estimate_s_per_step_scales_and_falls_back():
    with tempfile.TemporaryDirectory() as tmp:
        # measured 4.0 s/step at n_nodes=2 -> target nodes=4 should scale to 2.0
        path = os.path.join(tmp, "ledger_matthew_555.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_base", "hardware_id", "n_nodes", "s_per_step_median"])
            w.writerow(["matthew", "scc-gh200", "2", "4.0"])
            w.writerow(["matthew", "scc-gh200", "2", ""])          # failed row, ignored
            w.writerow(["other_case", "scc-gh200", "2", "9.0"])    # different case, ignored
            w.writerow(["matthew", "f1-spr-icpnp", "2", "1.0"])    # different hardware, ignored
        s_est, source, n = sw.estimate_s_per_step("matthew", nodes=4, ledger_dir=tmp,
                                                    hardware_id="scc-gh200")
        assert source == "ledger"
        assert n == 1
        assert abs(s_est - 2.0) < 1e-9

        s_est2, source2, n2 = sw.estimate_s_per_step("nosuchcase", nodes=4, ledger_dir=tmp,
                                                       hardware_id="scc-gh200")
        assert source2 == "fallback_10s"
        assert n2 == 0
        assert s_est2 == sw.CONSERVATIVE_S_PER_STEP


# ---------------------------------------------------------------------------
# manifest idempotency (fake sbatch/sacct via injectable runner functions)
# ---------------------------------------------------------------------------
def test_submit_is_idempotent_and_force_resubmits():
    sw.campaign_configs.CASES["synth_manifest_test"] = dict(MATTHEW_LIKE, steps=30, enabled=True)
    sbatch_runner, sacct_runner, calls = _fake_runners()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sweep")
        args = _args(cases=["synth_manifest_test"], nodes=[1, 2], root=root,
                     ledger_dir=os.path.join(tmp, "ledger"),
                     sbatch_runner=sbatch_runner, sacct_runner=sacct_runner)

        assert sw.cmd_submit(args) == 0
        assert len(calls["sbatch"]) == 2
        rows = sw.read_manifest(os.path.join(root, "manifest.csv"))
        assert {(r["case"], r["nodes"]) for r in rows} == {
            ("synth_manifest_test", "1"), ("synth_manifest_test", "2")}
        first_job_ids = {r["nodes"]: r["job_id"] for r in rows}

        # re-run with identical args: no new sbatch calls, manifest unchanged
        assert sw.cmd_submit(args) == 0
        assert len(calls["sbatch"]) == 2
        rows2 = sw.read_manifest(os.path.join(root, "manifest.csv"))
        assert {r["nodes"]: r["job_id"] for r in rows2} == first_job_ids

        # --force resubmits and overwrites the manifest row with a new job id
        args.force = True
        assert sw.cmd_submit(args) == 0
        assert len(calls["sbatch"]) == 4
        rows3 = sw.read_manifest(os.path.join(root, "manifest.csv"))
        assert len(rows3) == 2   # still one row per (case,nodes), not appended
        for r in rows3:
            assert r["job_id"] != first_job_ids[r["nodes"]]


def test_submit_configs_json_written_and_config_file_path_absolute():
    sw.campaign_configs.CASES["synth_cfgpath_test"] = dict(MATTHEW_LIKE, steps=30, enabled=True)
    sbatch_runner, sacct_runner, calls = _fake_runners()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sweep")
        args = _args(cases=["synth_cfgpath_test"], nodes=[1], root=root,
                     ledger_dir=os.path.join(tmp, "ledger"),
                     sbatch_runner=sbatch_runner, sacct_runner=sacct_runner)
        sw.cmd_submit(args)
        configs_path = os.path.join(root, "synth_cfgpath_test", "n1", "configs.json")
        assert os.path.isfile(configs_path)
        assert "CONFIG_FILE=%s" % configs_path in calls["sbatch"][0]
        assert "--export=ALL,SITE=scc,WTUNE_TOTAL_CORES=72,CASE=synth_cfgpath_test,MODE=timing" \
            in calls["sbatch"][0]


# ---------------------------------------------------------------------------
# max-nodes-in-flight gate
# ---------------------------------------------------------------------------
def test_max_nodes_in_flight_stops_and_reports_remaining():
    sw.campaign_configs.CASES["synth_budget_test"] = dict(MATTHEW_LIKE, steps=30, enabled=True)
    sbatch_runner, sacct_runner, calls = _fake_runners()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sweep")
        # 8 + 4 + 2 = 14 > 13: the third pair must be refused, the first two submitted.
        args = _args(cases=["synth_budget_test"], nodes=[8, 4, 2], root=root,
                     ledger_dir=os.path.join(tmp, "ledger"), max_nodes_in_flight=13,
                     sbatch_runner=sbatch_runner, sacct_runner=sacct_runner)
        assert sw.cmd_submit(args) == 0
        assert len(calls["sbatch"]) == 2
        rows = sw.read_manifest(os.path.join(root, "manifest.csv"))
        assert {r["nodes"] for r in rows} == {"8", "4"}


def test_max_nodes_in_flight_allows_exact_budget():
    sw.campaign_configs.CASES["synth_budget_exact_test"] = dict(MATTHEW_LIKE, steps=30, enabled=True)
    sbatch_runner, sacct_runner, calls = _fake_runners()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sweep")
        args = _args(cases=["synth_budget_exact_test"], nodes=[8, 4], root=root,
                     ledger_dir=os.path.join(tmp, "ledger"), max_nodes_in_flight=12,
                     sbatch_runner=sbatch_runner, sacct_runner=sacct_runner)
        assert sw.cmd_submit(args) == 0
        assert len(calls["sbatch"]) == 2   # 8 + 4 == 12, exactly at the limit, both go through


if __name__ == "__main__":
    raise SystemExit(__import__("pytest").main([__file__, "-v"]))

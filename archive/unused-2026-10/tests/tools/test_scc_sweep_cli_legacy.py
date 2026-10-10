# Tests of the scc_sweep plan/submit/status CLI, cut verbatim from tests/tools/test_scc_sweep.py
# at 33a1ef7. Not runnable as is (imports and MATTHEW_LIKE stayed in the original file).


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


def test_oversized_pair_does_not_block_smaller_pairs_behind_it():
    # regression: a pair that does not fit used to STOP the loop, so a 13-node pair stalled every
    # 1..8-node pair queued after it for hours on scc. It must be deferred, the rest submitted.
    sw.campaign_configs.CASES["synth_hol_test"] = dict(MATTHEW_LIKE, steps=30, enabled=True)
    sbatch_runner, sacct_runner, calls = _fake_runners()
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "sweep")
        args = _args(cases=["synth_hol_test"], nodes=[8, 8, 1], root=root,
                     ledger_dir=os.path.join(tmp, "ledger"), max_nodes_in_flight=13,
                     sbatch_runner=sbatch_runner, sacct_runner=sacct_runner)
        args.nodes = [8, 13, 1]   # 8 fits, 13 does not (8+13 > 13), 1 must still go through
        assert sw.cmd_submit(args) == 0
        rows = sw.read_manifest(os.path.join(root, "manifest.csv"))
        assert {r["nodes"] for r in rows} == {"8", "1"}


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

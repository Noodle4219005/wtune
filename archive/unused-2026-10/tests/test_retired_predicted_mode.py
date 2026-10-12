"""Tests of the retired `wtune tune --rank-by predicted` mode and of the dead code removed with
it (2026-10-10), cut verbatim from the files named below as at 1fa9fe8. Each section needs its
source file's imports and helpers; not part of the main suite, not runnable as is."""


# --- cut from tests/test_wtune_rank_predicted.py ----------------------------------------------

def test_predicted_mode_every_chunk_has_anchor_and_anchor_end(tmp_path):
    sb, calls, paths = make_fake_sbatch()
    ctx = _ctx(out_dir=str(tmp_path), sbatch_runner=sb)
    ctx.ranking = wt.ranking_context("matthewp1", "/r/rebuilt_cases.json", 20, 36, ANCHOR)
    ctx.ranking["dt"] = 36.0
    configs = [dict(ANCHOR)] + [_cfg("G%d" % i) for i in range(1, 5)]
    wt._submit_round0_wave(ctx, configs, 2, 0.01, "round0")
    ids = _chunk_configs(paths)
    # predicted mode deals CANDIDATES only; every slot then gets its own ANCHOR/ANCHOREND
    assert ids == [["ANCHOR", "G1", "G3", "ANCHOREND"], ["ANCHOR", "G2", "G4", "ANCHOREND"]]
    for c in calls:
        assert "CASE=matthewp1," in c and "STEPS=36," in c
        for kv in ("CASES_JSON=/r/rebuilt_cases.json", "RANKING_START=spinup",
                   "SPINUP_MINUTES=20", "RANKING_STRICT_RESTART=1", "RANKING_RESTART_IO_FORM=14"):
            assert kv in c.split("--export=")[1].split()[0].split(",")
        assert re.search(r"SPINUP_TIMEOUT_S=\d+", c)
    end = json.load(open(list(paths.values())[0].split(",")[0]))["configs"][-1]
    assert end["config_source"] == "anchor_end" and end["nx"] == ANCHOR["nx"]


def _stage1_rows(job, anchor, anchor_end, cands, bad=None):
    bad = bad or {}
    rows = []
    for cid, s in [("ANCHOR", anchor)] + cands + [("ANCHOREND", anchor_end)]:
        notes = bad.get(cid, "config=%s node_dirty=none restart_target=21 restart_used=1 "
                             "foreign_load=0" % cid)
        rows.append(dict(run_id="matthewp1_%s_r1_%s" % (cid, job), s_per_step_median=str(s),
                         notes=notes))
    return rows


def test_stage1_normalizes_by_own_job_anchor(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    # job 11 on slow nodes (anchor 2.0), job 12 on fast nodes (anchor 1.0): raw medians say G1
    # (1.1) beats G2 (1.6), normalized says G2 (0.8x) beats G1 (1.1x).
    rows = (_stage1_rows("11", 2.0, 2.0, [("G2", 1.6)])
            + _stage1_rows("12", 1.0, 1.0, [("G1", 1.1)]))
    wt._record_stage1_gates(ctx, ["11", "12"], rows, "round0")
    res = wt.aggregate_by_cid(rows, {"G1": _cfg("G1"), "G2": _cfg("G2"), "ANCHOR": ANCHOR})
    ranked, excluded = wt.stage1_normalized_ranking(res, ctx.stage1_jobs)
    assert [e["cid"] for e in ranked] == ["G2", "G1"] and not excluded
    assert ranked[0]["norm_key"] == pytest.approx(0.8) and ranked[0]["median"] == 1.6


def test_stage1_anchor_end_missing_uses_anchor_alone(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    rows = [r for r in _stage1_rows("11", 2.0, 9.9, [("G1", 1.0)]) if "ANCHOREND" not in r["run_id"]]
    wt._record_stage1_gates(ctx, ["11"], rows, "round0")
    assert ctx.stage1_jobs["11"]["s_anchor_job"] == 2.0


def test_stage1_mean_of_anchor_and_anchor_end_and_drift_only_warns(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    rows = _stage1_rows("11", 1.0, 1.1, [("G1", 1.05)])        # 10% drift
    wt._record_stage1_gates(ctx, ["11"], rows, "round0")
    info = ctx.stage1_jobs["11"]
    assert info["s_anchor_job"] == pytest.approx(1.05) and not info["excluded"]
    res = wt.aggregate_by_cid(rows, {"G1": _cfg("G1")})
    ranked, _ex = wt.stage1_normalized_ranking(res, ctx.stage1_jobs)
    assert ranked[0]["norm_key"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad_notes,why", [
    ("config=G1 node_dirty=none restart_target=21 restart_used=1 foreign_load=1 "
     "foreign=work5:77:root:node:91.0", "foreign_load=1"),
    ("status=SKIPPED reason=physics_mismatch config=G1 restart_target=21 restart_used=0 "
     "node_dirty=none foreign_load=0", "restart_used=0"),
])
def test_stage1_job_excluded(tmp_path, bad_notes, why):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.ranking = wt.ranking_context("matthewp1", "/r.json", 21, 36, ANCHOR)
    rows = (_stage1_rows("11", 1.0, 1.0, [("G1", 0.5), ("G3", 0.6)], bad={"G1": bad_notes})
            + _stage1_rows("12", 1.0, 1.0, [("G2", 0.9)]))
    wt._record_stage1_gates(ctx, ["11", "12"], rows, "round0")
    assert ctx.stage1_jobs["11"]["excluded"] and why in ctx.stage1_jobs["11"]["reason"]
    res = wt.aggregate_by_cid(rows, {c: _cfg(c) for c in ("G1", "G2", "G3")})
    ranked, excluded = wt.stage1_normalized_ranking(res, ctx.stage1_jobs)
    assert [e["cid"] for e in ranked] == ["G2"]
    assert {x["cid"] for x in excluded} == {"G1", "G3"}
    assert any(g.get("excluded") for g in ctx.ranking_gates)


def test_cli_rank_by_flags():
    a = wt.build_arg_parser().parse_args(["conus25", "--rank-by", "predicted",
                                          "--restart-io-form", "2"])
    assert a.rank_by == "predicted" and a.restart_io_form == 2
    a = wt.build_arg_parser().parse_args(["conus25"])
    assert a.rank_by is None and a.restart_io_form is None
    with pytest.raises(SystemExit):
        wt.build_arg_parser().parse_args(["conus25", "--rank-by", "bogus"])
    with pytest.raises(SystemExit):
        wt.build_arg_parser().parse_args(["conus25", "--restart-io-form", "11"])


def test_stage1_cutoff_cancels_unstarted_slot_as_not_measured(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {"1": [(500, "RUNNING")], "2": [(10 ** 9, "PENDING")]})
    ctx, scalls = _running_ctx(tmp_path, clock, sacct, budget_s=1000, max_wall_s=10 ** 6)
    ctx.job_cids = {"1": ["ANCHOR", "G1", "ANCHOREND"], "2": ["ANCHOR", "G2", "G3", "ANCHOREND"]}
    states = wt.wait_for_jobs(ctx, ["1", "2"], "round0",
                              pending_cutoff=lambda: ctx.elapsed() >= 200)
    assert scalls == [["2"]]                       # only the never-started slot
    assert ctx.not_measured == [dict(job_id="2", cids=["G2", "G3"], stage="round0",
                                     reason="job still PENDING at the stage-1 cutoff")]
    assert states["1"][0] == "COMPLETED" and clock.t == pytest.approx(500)
    assert "2" in ctx.cancelled_jobs


def test_estimate_stage2_waves_and_model():
    # 3 points, 2 slots -> 2 waves; no stepra -> rad factor 1: spin = 120 + 35; run = 120+30+36
    est = wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 2, 120.0)
    assert est == pytest.approx(2 * ((120 + 35) + 5 * (150 + 36)))
    assert wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 3, 120.0) == pytest.approx(est / 2)
    # stepra 17: 36-step run has 3 radiation steps at r_rad 3.93
    f = 1 + 3 * (wt.RANKING_R_RAD - 1) / 36.0
    assert wt.rad_factor(36, 17) == pytest.approx(f)
    est = wt.estimate_stage2_s(1, 1, 36, 21, 36.0, 1.0, 1, 120.0, 0.0, stepra=17)
    assert est == pytest.approx((120 + 35 * wt.rad_factor(35, 17)) + (120 + 36 * f))


def _reserved_ctx(tmp_path, budget_s, **kw):
    ctx = _ctx(out_dir=str(tmp_path), budget_s=budget_s, **kw)
    ctx.ranking = wt.ranking_context("matthewp0", "/r.json", 21, 36, ANCHOR)
    ctx.ranking["dt"] = 36.0
    ctx.s_prior = 1.0
    ctx.startup_s = 120.0
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    # as if a first stage-1 wave had run 3 jobs at once at 1.0 s/step: concurrency 3, full top-3
    ctx.stage1_max_running = 3
    ctx.stage1_jobs = {"0": dict(s_anchor_job=1.0, excluded=False)}
    return ctx


def test_reserve_blocks_stage1_jobs_that_would_eat_stage2(tmp_path):
    reserve = wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 3, 120.0)      # 1642.5 s
    sb, calls, _p = make_fake_sbatch()
    ctx = _reserved_ctx(tmp_path, budget_s=reserve + 500, sbatch_runner=sb)
    assert wt.stage2_reserve_s(ctx) == pytest.approx(reserve)
    ok, why = wt._stage1_may_start(ctx, 400)
    assert ok
    ok, why = wt._stage1_may_start(ctx, 600)
    assert not ok and "stage-2 reserve" in why
    wt._submit_round0_wave(ctx, [dict(ANCHOR)] + [_cfg("G%d" % i) for i in range(1, 6)], 1,
                           1.0, "round0")
    assert calls == []                              # 7 runs + spin-up do not fit before stage 2
    # measured anchor normalizers replace the prior
    ctx.stage1_jobs = {"9": dict(s_anchor_job=0.5, excluded=False)}
    assert wt.stage2_reserve_s(ctx) == pytest.approx(
        wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 0.5, 3, 120.0))


def test_reserve_is_zero_and_checks_inert_in_t0_mode(tmp_path):
    ctx = _ctx(out_dir=str(tmp_path))
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    ctx.s_prior = 1.0
    assert wt.stage2_reserve_s(ctx) == 0.0
    assert wt._stage1_may_start(ctx, 10 ** 9) == (True, None)
    assert wt._stage1_cutoff(ctx) is None


def test_stage1_cutoff_fires_when_reserve_reached(tmp_path):
    clock = FakeClock()
    ctx = _reserved_ctx(tmp_path, budget_s=2000, clock=clock)
    cut = wt._stage1_cutoff(ctx)
    assert not cut()
    clock.t = 2000 - wt.stage2_reserve_s(ctx)
    assert cut()


def test_run_predicted_branch_sets_running_budget_and_max_points(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(wt.context, "resolve_case", lambda case, cpn: (
        dict(e_we=91, e_sn=100, e_vert=45, prep_dir=str(tmp_path)), str(tmp_path / "nl"),
        {"core_count_priors": {"efficiency_table": []}}))

    def fake_run_predicted(ctx, nodes, nodes_note, s_est):
        seen.update(mode=ctx.budget_mode, wall=ctx.max_wall_s,
                    max_points=ctx.probe_cfg["max_points"])
        return {}

    monkeypatch.setattr(wt.predicted, "run_predicted", fake_run_predicted)
    monkeypatch.setattr(wt.predicted, "render_predicted_report_text", lambda r: "")
    base = ["matthew", "--minutes", "60", "--nodes", "1", "--steps", "5",
            "--out", str(tmp_path / "o"), "--ledger-dir", str(tmp_path / "l"),
            "--rank-by", "predicted"]
    wt.run(wt.build_arg_parser().parse_args(base), sacct_runner=make_fake_sacct({}),
           clock=FakeClock())
    assert seen == dict(mode="running", wall=3 * 60 * 60.0, max_points=3)
    wt.run(wt.build_arg_parser().parse_args(base + ["--max-points", "5",
                                                    "--max-wall-minutes", "90"]),
           sacct_runner=make_fake_sacct({}), clock=FakeClock())
    assert seen == dict(mode="running", wall=90 * 60.0, max_points=5)


def test_probe_key_explicit(tmp_path):
    pytest.importorskip("yaml")
    y = tmp_path / "c.yaml"
    y.write_text("probe:\n  max_points: 5\n")
    assert wt._probe_key_explicit(str(y), "max_points")
    y.write_text("probe:\n  steps: 30\n")
    assert not wt._probe_key_explicit(str(y), "max_points")
    assert not wt._probe_key_explicit(None, "max_points")


def _shrink_ctx(tmp_path, budget_s, sb):
    ctx = _reserved_ctx(tmp_path, budget_s=budget_s, sbatch_runner=sb)
    ctx.reserve = dict(n_points=3, n_configs=5, steps=36, spinup_min=21, dt=36.0, slots=3)
    return ctx


def test_stage1_slot_is_shrunk_not_skipped(tmp_path):
    # no stepra -> rad factor 1: per_run = 120+30+36 = 186, spin-up = 120+35 = 155,
    # reserve = 155 + 5*186 = 1085 (3 points, 3 concurrent)
    sb, calls, paths = make_fake_sbatch()
    budget = 1085 + (155 + 5 * 186) + 10             # fits ANCHOR + 3 + ANCHOREND, not more
    ctx = _shrink_ctx(tmp_path, budget, sb)
    assert wt.stage2_reserve_s(ctx) == pytest.approx(1085)
    cands = [_cfg("G%d" % i) for i in range(1, 7)]
    wt._submit_round0_wave(ctx, [dict(ANCHOR)] + cands, 1, 1.0, "round0")
    assert _chunk_configs(paths) == [["ANCHOR", "G1", "G2", "G3", "ANCHOREND"]]
    assert [t["cid"] for t in ctx.stage1_trimmed] == ["G4", "G5", "G6"]


def test_stage1_minimum_is_anchor_one_candidate_anchor_end(tmp_path):
    sb, calls, paths = make_fake_sbatch()
    ctx = _shrink_ctx(tmp_path, 1085 + (155 + 3 * 186), sb)          # exactly the minimum
    wt._submit_round0_wave(ctx, [_cfg("G1"), _cfg("G2")], 3, 1.0, "round0")
    # 2 candidates over min(3 slots, 2) = 2 slots; each slot is exactly the minimum
    assert _chunk_configs(paths) == [["ANCHOR", "G1", "ANCHOREND"], ["ANCHOR", "G2", "ANCHOREND"]]
    sb2, calls2, _p = make_fake_sbatch()
    ctx = _shrink_ctx(tmp_path / "b", 1085 + (155 + 3 * 186) - 1, sb2)   # 1 s short
    wt._submit_round0_wave(ctx, [_cfg("G1"), _cfg("G2")], 3, 1.0, "round0")
    assert calls2 == []                              # only now is a slot skipped


def _predicted_env(monkeypatch, n_points=3, dt=15.0, stepra=40, s_led=1.3):
    pts = [types.SimpleNamespace(at=T0 + dt_.timedelta(hours=6 * i, minutes=20), reason="auto",
                                 source="rebuild") for i in range(n_points)]
    monkeypatch.setattr(wt.wtune_predict, "parse_namelist", lambda p: None)
    monkeypatch.setattr(wt.wtune_predict, "get_d01_steps", lambda nl: (1440, 21600, dt, None))
    monkeypatch.setattr(wt.points, "plan_predicted_points", lambda ctx, spin: (
        None, T0, T0 + dt_.timedelta(hours=24), dt, stepra, pts, []))
    monkeypatch.setattr(wt.wtune_predict, "read_wrfinput_sample_points", lambda d: ([], None))
    monkeypatch.setattr(wt.sweep, "estimate_s_per_step", lambda *a, **k: (s_led, "ledger", 5))
    return pts


def test_feasibility_precheck_refuses_before_submitting(monkeypatch, tmp_path):
    _predicted_env(monkeypatch)
    called = []
    monkeypatch.setattr(wt.points, "prepare_rebuilt_points", lambda *a, **k: called.append(1))
    sb, calls, _p = make_fake_sbatch()
    ctx = _ctx(out_dir=str(tmp_path), budget_s=20 * 60.0, nodes=4, max_nodes=12,
               sbatch_runner=sb, case_json=dict(e_we=1501, e_sn=1201, e_vert=50,
                                                prep_dir=str(tmp_path)))
    ctx.budget_mode = "running"
    ctx.stage1_mode = "pstar"                        # pstar: --minutes must hold both stages
    with pytest.raises(SystemExit, match=r"predicted mode needs at least \d+ min of running time "
                                         r"for matthew at 4 node\(s\) \(have 20\)"):
        wt.run_predicted(ctx, 4, "user", 1.3)
    assert called == [] and calls == []


@pytest.mark.parametrize("stage2_extra_s", [0.0, 3000.0])
def test_conus25_like_runs_stage1_and_stage2(monkeypatch, tmp_path, stage2_extra_s):
    """Observed (attempt 1, job 1388): ~200 s per candidate run (restart read + ~87 s startup +
    82 steps x ~1.3 s); rebuild prep 7 min; 4 nodes, max-nodes 12. 80 min of running budget:
    with nothing observed yet, stage-2 concurrency is 1 (conservative), which puts the minimum
    at ~77 min for 3 points (60 min is now refused up front -- attempt 3 really needed ~81).
    stage2_extra_s > 0: stage 2 runs 50 min past its estimate -> completes, overrun reported."""
    _predicted_env(monkeypatch)
    clock = FakeClock()
    ledger_dir = tmp_path / "ledger"
    submitted = {}                                   # job -> (t_submit, duration, config ids)
    counter = {"n": 2000}

    def sbatch(cmd):
        counter["n"] += 1
        jid = str(counter["n"])
        path = re.search(r"CONFIG_FILE=(\S+)", cmd).group(1)
        ids = [c["id"] for c in json.load(open(path))["configs"]]
        submitted[jid] = (clock.t, (87 + 80 * 1.3) + 200.0 * len(ids), ids)
        return "Submitted batch job %s\n" % jid

    def sacct(cmd):
        out = []
        for j in cmd[cmd.index("-j") + 1].split(","):
            t_sub, dur, _ids = submitted.get(j, (0, 0, []))
            if clock.t < t_sub + 30:
                st = "PENDING"
            elif clock.t < t_sub + 30 + dur:
                st = "RUNNING"
            else:
                st = "COMPLETED"
            out.append("%s|%s|%s" % (j, st, wt.sweep.campaign_experiment._format_hms(dur / 60.0)
                                     if st == "COMPLETED" else "00:00:00"))
        return "\n".join(out)

    def reader(case, job_ids):
        rows = []
        for j in job_ids:
            _t, _d, ids = submitted[j]
            for i, cid in enumerate(ids):
                s = 1.3 if cid in ("ANCHOR", "ANCHOREND") else 1.3 * (0.9 + 0.01 * (int(j) % 7) + 0.003 * i)
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cid, j), s_per_step_median=str(s),
                                 notes="config=%s node_dirty=none restart_target=20 "
                                       "restart_used=1 foreign_load=0" % cid))
        return rows

    def prepare(ctx, points, anchor, *a, **k):
        clock.t += 420.0                             # 7 min of controller + prep running time
        return [dict(index=i, point=p, case_id="matthewp%d" % i, T=p.at - dt_.timedelta(minutes=20),
                     cases_json=str(tmp_path / "rc.json"), prep_dir="", template="", steps=82,
                     spinup_min=20, stepra=40, dt=15.0, plan_dir="", job_id="1")
                for i, p in enumerate(points)]

    stage2 = {}

    def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
        stage2["cids"] = [c["id"] for c in configs]
        stage2["left_s"] = ctx.budget_s - ctx.elapsed()
        clock.t += stage2_extra_s
        meas = {}
        for c in [anchor] + list(configs):
            meas[c["id"]] = [PM(at=rp["point"].at, reason="auto", source="rebuild",
                                s_normal=s_by_cid.get(c["id"], 1.3), n_rad=2, r=3.5)
                             for rp in rebuilt]
        return meas, [], ["9001"]

    monkeypatch.setattr(wt.points, "prepare_rebuilt_points", prepare)
    monkeypatch.setattr(wt.points, "measure_candidates_at_points", measure)
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(ledger_dir), budget_s=4800.0,
               nodes=4, max_nodes=12, clock=clock, poll_interval=20.0, sbatch_runner=sbatch,
               sacct_runner=sacct, scancel_runner=scancel, ledger_reader=reader,
               proposer=lambda state, bs, kappa=None: (dict(stopped=True, reason="stub"), []),
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path)))
    ctx.budget_mode = "running"
    ctx.max_wall_s = 3 * 3600.0
    ctx.stage1_mode = "pstar"                        # this scenario pins the P*-restart stage 1
    report = wt.run_predicted(ctx, 4, "user", 1.3)
    assert report["stage1_ranking"], "stage 1 measured nothing"
    assert len(report["stage1_ranking"]) >= 3
    assert stage2["cids"] and len(stage2["cids"]) >= 1
    assert report["ranking"] and report["stage2_skipped_reason"] is None
    assert scalls == []                              # nothing hit the budget or the wall cap
    assert "concurrent" in report["stage2_reserve_note"]
    assert report["stage1_max_running"] == 3 and report["stage2_concurrency"] == 3
    if stage2_extra_s:
        assert report["stage2_overrun_s"] > 0 and "STAGE 2 OVERRUN" in \
            wt.render_predicted_report_text(report)
    else:
        assert ctx.elapsed() <= ctx.budget_s and not report["stage2_overrun_s"]


def test_conus25_60min_is_refused_with_conservative_concurrency(monkeypatch, tmp_path):
    _predicted_env(monkeypatch)
    monkeypatch.setattr(wt.points, "prepare_rebuilt_points", lambda *a, **k: pytest.fail("submitted"))
    ctx = _ctx(out_dir=str(tmp_path), budget_s=3600.0, nodes=4, max_nodes=12,
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path)))
    ctx.budget_mode = "running"
    ctx.stage1_mode = "pstar"
    with pytest.raises(SystemExit, match=r"needs at least 7\d min .*1 concurrent"):
        wt.run_predicted(ctx, 4, "user", 1.3)


def test_serial_stage1_jobs_give_concurrency_1_reserve(tmp_path):
    clock = FakeClock()
    # both submitted at t=0; the busy cluster runs them one at a time (1417/1418)
    sacct, _c = timeline_sacct(clock, {"1417": [(300, "RUNNING")],
                                       "1418": [(300, "PENDING"), (600, "RUNNING")]})
    ctx, _s = _conc_ctx(tmp_path, clock, sacct)
    assert wt.stage2_concurrency(ctx) == 1           # nothing observed yet: conservative
    wt.wait_for_jobs(ctx, ["1417", "1418"], "round0")
    assert ctx.stage1_max_running == 1 and wt.stage2_concurrency(ctx) == 1
    ctx.stage1_jobs = {"1417": dict(s_anchor_job=1.0, excluded=False)}
    assert wt.stage2_reserve_s(ctx) == pytest.approx(
        wt.estimate_stage2_s(3, 5, 36, 21, 36.0, 1.0, 1, 120.0, 30.0))   # 3 waves
    assert "1 concurrent: observed max 1" in wt.describe_stage2_reserve(ctx)


def test_parallel_stage1_jobs_raise_concurrency_capped(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {j: [(300, "RUNNING")] for j in ("1", "2", "3", "4")})
    ctx, _s = _conc_ctx(tmp_path, clock, sacct)
    wt.wait_for_jobs(ctx, ["1", "2", "3", "4"], "round0")
    assert ctx.stage1_max_running == 4 and wt.stage2_concurrency(ctx) == 3   # floor(12/4)


def test_concurrency_not_observed_outside_stage1(tmp_path):
    clock = FakeClock()
    sacct, _c = timeline_sacct(clock, {j: [(300, "RUNNING")] for j in ("1", "2", "3")})
    ctx, _s = _conc_ctx(tmp_path, clock, sacct)
    ctx.ranking = None                               # rebuild prep / stage 2
    wt.wait_for_jobs(ctx, ["1", "2", "3"], "rebuild-prep")
    assert ctx.stage1_max_running == 0


def test_stage1_option_cli_yaml_default():
    assert wt.build_arg_parser().parse_args(["x"]).stage1 is None
    assert wt.build_arg_parser().parse_args(["x", "--stage1", "pstar"]).stage1 == "pstar"
    with pytest.raises(SystemExit):
        wt.build_arg_parser().parse_args(["x", "--stage1", "bogus"])
    cfg, errs = wtune_config.resolve_config({})
    assert cfg["probe"]["stage1"] == "t0" and not errs
    _c, errs = wtune_config.resolve_config({"probe": {"stage1": "bogus"}})
    assert any("probe.stage1" in e for e in errs)
    assert _ctx().stage1_mode == "t0"


def test_stage1_t0_feasibility_checks_stage1_only(monkeypatch, tmp_path, capsys):
    _predicted_env(monkeypatch)
    monkeypatch.setattr(wt.points, "submit_rebuilt_points", lambda *a, **k: pytest.fail("submitted"))
    ctx = _ctx(out_dir=str(tmp_path / "o"), budget_s=5 * 60.0, nodes=4, max_nodes=12, steps=20,
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path / "c")))
    ctx.budget_mode = "running"
    # one minimal t0 job = 3 x (120 + 20 x 1.3) = 438 s = 7:18 > 5 min -> refused
    with pytest.raises(SystemExit, match=r"needs at least 8 min .*stage-1 minimum 7:18 = one t0 "
                                         r"job"):
        wt.run_predicted(ctx, 4, "user", 1.3)
    # 10 min holds it: accepted even though stage 2 alone is estimated at ~65 min
    ctx.budget_s = 10 * 60.0

    class Submitted(Exception):
        pass

    monkeypatch.setattr(wt.points, "submit_rebuilt_points",
                        lambda *a, **k: (_ for _ in ()).throw(Submitted()))
    with pytest.raises(Submitted):
        wt.run_predicted(ctx, 4, "user", 1.3)
    out = capsys.readouterr().out
    assert re.search(r"stage 2 will need about \d+ min MORE, on top of --minutes \(3 point\(s\)"
                     r" x \(spin-up \+ 5 runs\), concurrency 1", out)


def test_stage1_t0_end_to_end(monkeypatch, tmp_path):
    _predicted_env(monkeypatch)
    clock = FakeClock()
    events = []
    submitted = {}
    counter = {"n": 3000}
    cmds = []

    def sbatch(cmd):
        counter["n"] += 1
        jid = str(counter["n"])
        cmds.append(cmd)
        path = re.search(r"CONFIG_FILE=(\S+)", cmd).group(1)
        ids = [c["id"] for c in json.load(open(path))["configs"]]
        submitted[jid] = (clock.t, 150.0 * len(ids), ids)
        events.append(("stage1-submit", jid))
        return "Submitted batch job %s\n" % jid

    def sacct(cmd):
        out = []
        for j in cmd[cmd.index("-j") + 1].split(","):
            t_sub, dur, _ids = submitted.get(j, (0, 0, []))
            st = ("PENDING" if clock.t < t_sub + 30 else
                  "RUNNING" if clock.t < t_sub + 30 + dur else "COMPLETED")
            out.append("%s|%s|%s" % (j, st, wt.sweep.campaign_experiment._format_hms(dur / 60.0)))
        return "\n".join(out)

    def num(cid):
        return int(re.sub(r"\D", "", cid) or 0)

    def reader(case, job_ids):
        assert case == "matthew"                     # t0 stage 1: the case's own ledgers
        rows = []
        for j in job_ids:
            _t, _d, ids = submitted[j]
            scale = 1.5 if int(j) % 2 else 1.0       # alternate slow/fast allocations
            for cid in ids:
                base = 1.3 if cid in ("ANCHOR", "ANCHOREND") else 1.3 * (0.8 + 0.01 * num(cid))
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cid, j),
                                 s_per_step_median=str(scale * base),
                                 notes="config=%s node_dirty=none" % cid))
        return rows

    def submit_prep(ctx, points, anchor, *a, **k):
        events.append(("prep-submit",))
        ctx.owned_jobs.extend(["p1", "p2", "p3"])
        return dict(job_ids=["p1", "p2", "p3"], points=points)

    def finish_prep(ctx, handle):
        events.append(("prep-wait",))
        return [dict(index=i, point=p, case_id="matthewp%d" % i, T=p.at - dt_.timedelta(minutes=20),
                     cases_json="/r.json", steps=82, spinup_min=20, stepra=40, dt=15.0,
                     job_id="p%d" % (i + 1))
                for i, p in enumerate(handle["points"])]

    stage2 = {}

    def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
        events.append(("stage2",))
        stage2["cids"] = [c["id"] for c in configs]
        stage2["ranking_ctx"] = ctx.ranking
        meas = {c["id"]: [PM(at=rp["point"].at, reason="auto", source="rebuild",
                             s_normal=s_by_cid.get(c["id"], 1.3), n_rad=2, r=3.5)
                          for rp in rebuilt] for c in [anchor] + list(configs)}
        return meas, [], ["9001"]

    monkeypatch.setattr(wt.points, "submit_rebuilt_points", submit_prep)
    monkeypatch.setattr(wt.points, "finish_rebuilt_points", finish_prep)
    monkeypatch.setattr(wt.points, "prepare_rebuilt_points", lambda *a, **k: pytest.fail("pstar path"))
    monkeypatch.setattr(wt.points, "measure_candidates_at_points", measure)
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(tmp_path / "l"), budget_s=5400.0,
               nodes=4, max_nodes=12, steps=20, clock=clock, poll_interval=20.0,
               sbatch_runner=sbatch, sacct_runner=sacct, scancel_runner=scancel,
               ledger_reader=reader,
               proposer=lambda state, bs, kappa=None: (dict(stopped=True, reason="stub"), []),
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path / "c")))
    ctx.budget_mode = "running"
    ctx.max_wall_s = 3 * 5400.0
    report = wt.run_predicted(ctx, 4, "user", 1.3)

    # order: prep submitted first, stage 1 ran beside it, prep awaited only before stage 2
    kinds = [e[0] for e in events]
    assert kinds[0] == "prep-submit"
    assert kinds.index("prep-wait") > max(i for i, k in enumerate(kinds) if k == "stage1-submit")
    assert kinds[-1] == "stage2" and kinds[-2] == "prep-wait"
    # stage 1 = plain t0 jobs: no ranking context, t0 STEPS, anchors in every job
    for c in cmds:
        exp = c.split("--export=")[1].split()[0]
        assert "RANKING_START" not in exp and "CASES_JSON" not in exp
        assert ",CASE=matthew," in exp and ",STEPS=20," in exp
    for jid, (_t, _d, ids) in submitted.items():
        assert ids[0] == "ANCHOR" and ids[-1] == "ANCHOREND" and len(ids) >= 3
    assert ctx.stage1_max_running >= 1 and report["stage1_mode"] == "t0"
    assert not report["stage1_excluded"]             # no restart_used required at t0
    # shortlist = the 3 smallest ANCHOR-NORMALIZED keys, not the raw medians
    ranked = report["stage1_ranking"]
    assert [e["norm_key"] for e in ranked] == sorted(e["norm_key"] for e in ranked)
    by_num = sorted((e["id"] for e in ranked), key=num)
    assert stage2["cids"] == by_num[:3]
    raw_top = [e["id"] for e in sorted(ranked, key=lambda e: e["median_s_per_step"])[:3]]
    assert raw_top != stage2["cids"]                 # the slow allocations would have skewed it
    assert stage2["ranking_ctx"] is None and report["ranking"] and scalls == []


def _t0_fakes(clock, run_s=150.0, queue_s=30.0):
    submitted, cmds, counter = {}, [], {"n": 4000}

    def sbatch(cmd):
        counter["n"] += 1
        jid = str(counter["n"])
        cmds.append(cmd)
        path = re.search(r"CONFIG_FILE=(\S+)", cmd).group(1)
        ids = [c["id"] for c in json.load(open(path))["configs"]]
        submitted[jid] = (clock.t, run_s * len(ids), ids)
        return "Submitted batch job %s\n" % jid

    def sacct(cmd):
        out = []
        for j in cmd[cmd.index("-j") + 1].split(","):
            t_sub, dur, _ids = submitted.get(j, (0, 0, []))
            st = ("PENDING" if clock.t < t_sub + queue_s else
                  "RUNNING" if clock.t < t_sub + queue_s + dur else "COMPLETED")
            out.append("%s|%s|%s" % (j, st, wt.sweep.campaign_experiment._format_hms(dur / 60.0)))
        return "\n".join(out)

    def reader(case, job_ids):
        rows = []
        for j in job_ids:
            _t, _d, ids = submitted.get(j, (0, 0, []))
            for cid in ids:
                n = int(re.sub(r"\D", "", cid) or 0)
                s_ = 1.3 if cid in ("ANCHOR", "ANCHOREND") else 1.3 * (0.8 + 0.01 * n)
                rows.append(dict(run_id="%s_%s_r1_%s" % (case, cid, j), s_per_step_median=str(s_),
                                 notes="config=%s node_dirty=none" % cid))
        return rows

    return sbatch, sacct, reader, submitted, cmds


def _stub_proposer(state, bs, kappa=None):
    return dict(stopped=True, reason="stub"), []


def _t0_ctx(tmp_path, clock, fakes, budget_s, **kw):
    sbatch, sacct, reader, _sub, _cmds = fakes
    scancel, scalls = make_fake_scancel()
    ctx = _ctx(out_dir=str(tmp_path / "out"), ledger_dir=str(tmp_path / "l"), budget_s=budget_s,
               nodes=4, max_nodes=12, steps=20, clock=clock, poll_interval=20.0,
               sbatch_runner=sbatch, sacct_runner=sacct, scancel_runner=scancel,
               ledger_reader=reader, proposer=_stub_proposer,
               case_json=dict(e_we=1501, e_sn=1201, e_vert=50, prep_dir=str(tmp_path / "c")),
               **kw)
    return ctx, scalls


def _t0_mode_candidate_count(monkeypatch, tmp_path, budget_s):
    """What the plain t0 tuner's search (round0 + adaptive rounds; confirm adds no candidate)
    measures on the same fake timeline and budget."""
    monkeypatch.setattr(wt.sweep, "estimate_s_per_step", lambda *a, **k: (1.3, "ledger", 5))
    clock = FakeClock()
    fakes = _t0_fakes(clock)
    ctx, _s = _t0_ctx(tmp_path / "t0", clock, fakes, budget_s)
    job_ids0, cbc, rows0 = wt.run_round0(ctx)
    res = wt.aggregate_by_cid(rows0, cbc)
    wt.run_adaptive_rounds(ctx, cbc, res, 1.3)
    return len([c for c in res if c != "ANCHOR"])


def _run_pred_t0(monkeypatch, tmp_path, budget_s, stage2_s=1200.0, stage2_cap_s=None):
    _predicted_env(monkeypatch)
    clock = FakeClock()
    fakes = _t0_fakes(clock)
    events = []

    def submit_prep(ctx, points, anchor, *a, **k):
        clock.t += 200.0                             # ERA5 fetch + stage: not stage-1 work
        events.append("prep-submit")
        return dict(job_ids=["p1", "p2", "p3"], points=points)

    def finish_prep(ctx, handle):
        events.append("prep-wait")
        return [dict(index=i, point=p, case_id="matthewp%d" % i, T=p.at - dt_.timedelta(minutes=20),
                     cases_json="/r.json", steps=82, spinup_min=20, stepra=40, dt=15.0,
                     job_id="p%d" % (i + 1)) for i, p in enumerate(handle["points"])]

    stage2 = {}

    def measure(ctx, configs, rebuilt, anchor, s_by_cid=None, **k):
        events.append("stage2")
        stage2["cids"] = [c["id"] for c in configs]
        clock.t += stage2_s                          # stage 2's own running time
        meas = {c["id"]: [PM(at=rp["point"].at, reason="auto", source="rebuild",
                             s_normal=s_by_cid.get(c["id"], 1.3), n_rad=2, r=3.5)
                          for rp in rebuilt] for c in [anchor] + list(configs)}
        return meas, [], ["9001"]

    monkeypatch.setattr(wt.points, "submit_rebuilt_points", submit_prep)
    monkeypatch.setattr(wt.points, "finish_rebuilt_points", finish_prep)
    monkeypatch.setattr(wt.points, "measure_candidates_at_points", measure)
    ctx, scalls = _t0_ctx(tmp_path / "pred", clock, fakes, budget_s)
    ctx.budget_mode = "running"
    ctx.max_wall_s = 3 * budget_s
    ctx.stage2_budget_s = stage2_cap_s
    report = wt.run_predicted(ctx, 4, "user", 1.3)
    return report, ctx, stage2, events, scalls


def test_stage1_t0_30min_measures_at_least_the_t0_search(monkeypatch, tmp_path):
    n_t0 = _t0_mode_candidate_count(monkeypatch, tmp_path, 1800.0)
    report, ctx, stage2, events, scalls = _run_pred_t0(monkeypatch, tmp_path, 1800.0)
    n_pred = len(report["stage1_ranking"]) + len(report["stage1_excluded"])
    assert n_pred >= n_t0 >= 1, (n_pred, n_t0)
    # stage 1 used (about) --minutes, stage 2 ran AFTER it and is reported separately
    assert report["stage1_running_min"] <= 30.0 + 1.0
    assert report["stage2_running_min"] >= 20.0
    assert report["total_running_min"] == pytest.approx(
        report["stage1_running_min"] + report["stage2_running_min"] + report["prep_submit_min"])
    assert report["prep_submit_min"] == pytest.approx(200.0 / 60.0)
    assert report["stage2_estimate_min"] > 0 and stage2["cids"] and len(stage2["cids"]) == 3
    assert report["stage2_overrun_s"] is None and scalls == []
    assert events[0] == "prep-submit" and events[-2:] == ["prep-wait", "stage2"]
    assert "stage 2 separate" in report["budget_semantics"]
    assert "running time: stage 1" in wt.render_predicted_report_text(report)
    # the wall cap was derived from --minutes + the stage-2 estimate, not 3x --minutes alone
    assert ctx.max_wall_s > 3 * 1800.0


def test_stage2_minutes_cap_trims_shortlist(monkeypatch, tmp_path):
    full, _c, stage2, _e, _s = _run_pred_t0(monkeypatch, tmp_path / "a", 1800.0)
    assert len(stage2["cids"]) == 3 and full["stage2_minutes_cap"] is None
    cap = full["stage2_estimate_s"] - 1.0            # one second short of the full top-3
    report, _c, stage2, _e, _s = _run_pred_t0(monkeypatch, tmp_path / "b", 1800.0,
                                               stage2_cap_s=cap)
    assert len(stage2["cids"]) == 2 and report["stage2_estimate_s"] <= cap
    assert report["stage2_minutes_cap"] == pytest.approx(cap / 60.0)


def test_stage1_t0_holds_no_stage2_reserve_pstar_does(tmp_path):
    ctx = _reserved_ctx(tmp_path, budget_s=3600.0)   # pstar-style: ranking context set
    assert wt.stage1_reserve_s(ctx) > 0
    ctx2 = _ctx(out_dir=str(tmp_path), budget_s=3600.0)
    ctx2.stage1 = dict(mode="t0", anchor=ANCHOR)
    ctx2.reserve = dict(ctx.reserve)
    ctx2.s_prior = 1.0
    ctx2.reserve_in_stage1 = False
    assert wt.stage1_reserve_s(ctx2) == 0.0
    assert wt._stage1_may_start(ctx2, 3500.0)[0] and not wt._stage1_may_start(ctx2, 3700.0)[0]


def test_stage2_minutes_cli_and_yaml():
    assert wt.build_arg_parser().parse_args(["x", "--stage2-minutes", "25"]).stage2_minutes == 25
    cfg, errs = wtune_config.resolve_config({})
    assert cfg["probe"]["stage2_minutes"] is None and not errs
    _c, errs = wtune_config.resolve_config({"probe": {"stage2_minutes": -1}})
    assert any("stage2_minutes" in e for e in errs)



# --- cut from tests/test_wtune_ranking_submit.py ----------------------------------------------

def _cmd(spinup_minutes):
    return wtune_tune.build_ranking_sbatch_command(
        "conus25", 4, 72, "01:00:00", 1800, 2, 50, "/x/configs.json", "/x/out", spinup_minutes)


def test_spinup_minutes_never_contains_a_comma_inside_export():
    cmd = _cmd("60:122")
    export = re.search(r"--export=(\S+)", cmd).group(1)
    pairs = export.split(",")
    assert "SPINUP_MINUTES=60:122" in pairs
    assert all("=" in p or p == "ALL" for p in pairs)


def test_spinup_minutes_with_comma_is_refused():
    with pytest.raises(ValueError, match="sbatch --export"):
        _cmd("60,122")



# --- cut from tests/test_wtune_tune_fresh_inputs.py -------------------------------------------

def test_trajectory_command_carries_extra_export():
    cmd = prediction.build_ranking_sbatch_command(
        "conus25", 4, 72, "01:00:00", 900, 2, 50, "/x/c.json", "/x", "60:120",
        extra_export=[("WTUNE_BP_FRESH_DIR", "/x/bp_fresh")])
    assert re.search(r"SPINUP_MINUTES=60:120(,CAMPAIGN=\S+)?,WTUNE_BP_FRESH_DIR=/x/bp_fresh ", cmd)
    plain = prediction.build_ranking_sbatch_command(
        "conus25", 4, 72, "01:00:00", 900, 2, 50, "/x/c.json", "/x", "60:120")
    assert "WTUNE_BP_FRESH_DIR" not in plain



# --- cut from tests/test_wtune_tune.py --------------------------------------------------------

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


def test_nodes_search_with_rank_by_predicted_is_refused(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="not supported with --rank-by predicted"):
        _run_cli(monkeypatch, tmp_path,
                 ["--minutes", "30", "--nodes", "search", "--rank-by", "predicted"])



# --- cut from tests/test_refactor_golden_wtune_tune_pkg.py ------------------------------------

def test_golden_run_c3_prediction(monkeypatch, tmp_path):
    import datetime as _dt
    calls = []
    _fake_predict_stack(monkeypatch, calls)
    t = _dt.datetime(2026, 1, 1, 9, 0)
    pts = [types.SimpleNamespace(at=t, reason="t0", source="trajectory", steps=30),
           types.SimpleNamespace(at=t, reason="sunlit", source="rebuild", steps=30),
           types.SimpleNamespace(at=t, reason="peak", source="restart", steps=30)]
    monkeypatch.setattr(wtune_tune.wtune_probe_plan, "plan_points",
                        lambda *a, **k: calls.append(("plan", a[1:6], sorted(k))) or
                        (list(pts), ["fake note"]))
    _patch(monkeypatch, "run_trajectory_points",
           lambda ctx, points, best, nodes, run_start, stepra, out, **k: (
               calls.append(("traj", [p.reason for p in points], nodes, str(run_start), stepra,
                             os.path.basename(out), sorted(k.items()))) or (["m1", "m2"], "501")))
    _patch(monkeypatch, "run_rebuild_points",
           lambda ctx, points, best, nodes, case_dir, out, run_start, **k: (
               calls.append(("rebuild", [p.reason for p in points], case_dir,
                             os.path.basename(out), sorted(k.items()))) or (["m3"], "601,602")))
    ctx = T._ctx(out_dir=str(tmp_path / "o"), ledger_dir=str(tmp_path / "l"),
                 case_json=dict(T.MATTHEW_LIKE, prep_dir=str(tmp_path / "case")))
    ctx.prediction_cfg["drift_band_pct"] = 6.0
    ctx.probe_cfg["spinup_min"] = 30
    best = dict(id="D1", ranks=72, omp=1, nx=8, ny=9, numtiles=1, dist="block")
    result, points, notes, job_ids = wtune_tune.run_c3_prediction(
        ctx, best, 2, str(tmp_path / "c3"), sbatch_script="/x/strict_campaign.sbatch",
        campaign_dir="/x", grib_dir="/g")
    G._check("pkg_run_c3_prediction.txt", G._norm(json.dumps(
        dict(result=result, n_points=len(points), notes=notes, job_ids=job_ids, calls=calls),
        indent=1, sort_keys=True, default=repr), tmp_path) + "\n")



# --- cut from tests/test_refactor_golden_wtune_tune.py ----------------------------------------

# --- run_predicted (--rank-by predicted): the end-to-end scenarios of test_wtune_rank_predicted,
# re-run with run_predicted's return value and printed text snapshotted --------------------------
import test_wtune_rank_predicted as R  # noqa: E402

PREDICTED = {
    "pstar_conus25": (R.test_conus25_like_runs_stage1_and_stage2, dict(stage2_extra_s=0.0)),
    "pstar_conus25_overrun": (R.test_conus25_like_runs_stage1_and_stage2,
                              dict(stage2_extra_s=3000.0)),
    "t0_end_to_end": (R.test_stage1_t0_end_to_end, {}),
    "pstar_refused": (R.test_feasibility_precheck_refuses_before_submitting, {}),
    "pstar_refused_60min": (R.test_conus25_60min_is_refused_with_conservative_concurrency, {}),
}

# DISPATCH_ARGV / DISPATCH_ERRORS entries of test_golden_run_dispatch (their snapshots were deleted
# from fixtures/golden/wtune_tune/dispatch.json):
#   "predicted": ["--minutes", "30", "--nodes", "4", "--rank-by", "predicted", "--stage2-minutes",
#                 "7", "--max-wall-minutes", "50", "--stage1", "pstar", "--restart-io-form", "2"]
#   "predicted_points": ["--minutes", "30", "--nodes", "auto", "--rank-by", "predicted",
#                        "--max-points", "3", "--points-by", "cost-indicators"]
#   errors: "pipeline_predicted": ["-O2", "--rank-by", "predicted"],
#           "search_predicted": ["--minutes", "30", "--nodes", "search", "--rank-by", "predicted"],
#           "bad_stage1": ["--minutes", "30", "--nodes", "2"] with args.stage1 = "bogus"


@pytest.mark.parametrize("name", sorted(PREDICTED))
def test_golden_run_predicted(name, monkeypatch, tmp_path, capsys):
    fn, kw = PREDICTED[name]
    real = wtune_tune.run_predicted
    reports = []

    def recording(*a, **k):
        reports.append(real(*a, **k))
        return reports[-1]
    monkeypatch.setattr(wtune_tune, "run_predicted", recording)
    fn(monkeypatch, tmp_path, **kw)
    out = capsys.readouterr().out
    rep = json.dumps(reports, indent=1, sort_keys=True, default=str)
    _check("predicted_%s.txt" % name, _norm(_norm(out, tmp_path), T._SCRATCH))
    _check("predicted_%s.json" % name, _norm(_norm(rep, tmp_path), T._SCRATCH) + "\n")
